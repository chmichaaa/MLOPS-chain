"""
Small internal MLOps console, separate from the served app (main.py) on
purpose: it needs a GitHub push token and write access to the actual git
checkout, which the production `app` container (rebuilt/redeployed by CI on
every push) should never carry. Runs as its own docker-compose service,
bind-mounting the real repo checkout at /repo.

Two ways to get a model into Production:
- Upload a dataset -> pushed through git/DVC -> the REAL CI/CD chain
  (unit_tests -> validate -> integration_tests -> build -> deploy) trains and
  gates it. This route never touches MLflow directly.
- Upload an already-trained model file -> evaluated immediately against the
  current eval split, using the exact same gate math training_pipeline.py
  uses (training_pipeline.evaluate_pipeline) -> promoted to Production
  immediately if it clears config.F1_THRESHOLD. No CI run, no redeploy --
  predict.py picks it up on its own within its cache TTL. This path
  deserializes an uploaded file (cloudpickle) -- a real code-execution risk,
  accepted knowingly and mitigated by requiring login.
"""
import concurrent.futures
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from html import escape

import cloudpickle
import mlflow
import pandas as pd
import requests as http
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from mlflow.tracking import MlflowClient
from passlib.context import CryptContext
from sqlalchemy import create_engine
from starlette.middleware.sessions import SessionMiddleware

from prediction_model.config import config
from prediction_model.processing.data_handling import load_full_dataset, day_block_split
from prediction_model.training_pipeline import evaluate_pipeline, register_and_promote, infer_model_type
from dataset_uploader import users, live_view, charts
from dataset_uploader.logic import (
    now_gmt1,
    format_timestamp,
    format_epoch_millis,
    format_hyperparams,
    source_label,
    stage_css_class,
    missing_dataset_columns,
    role_label,
    validate_username,
    validate_password,
    deletion_error,
    demotion_error,
    summarize_incidents,
    format_age,
    gate_position,
    GMT_PLUS_1 as GMT_PLUS_1_TZ,
)
from live_feed import store as live_store
from live_feed.generator import SCENARIO_LABELS

REPO_DIR = "/repo"
DATASET_PATH = os.path.join(REPO_DIR, "prediction_model", "datasets", "dataset.csv")
DATASET_META_PATH = os.path.join(REPO_DIR, "prediction_model", "datasets", "dataset.meta.json")
GITHUB_ACTIONS_URL = "https://github.com/chmichaaa/MLOPS-chain/actions"

# config.TRACKING_URI (http://mlflow-server:5000) is the INTERNAL docker
# hostname -- correct for server-to-server calls, but meaningless to a
# browser outside the docker network. Links/iframes rendered in the user's
# browser need the box's actual public address instead.
# Browser-facing links are derived from the host the browser actually used to
# reach this console, not from configuration. MLflow and Grafana sit on the
# same box behind the same address, so the incoming request already carries
# the correct answer -- and a value that cannot be edited cannot go stale. A
# hand-set PUBLIC_HOST silently broke every link on this page when the box's
# address changed: the console kept embedding a dead IP with nothing to say
# it was wrong, because from the server's side nothing was.
# Deliberately NOT the old PUBLIC_HOST variable: a stale PUBLIC_HOST left in
# .env pointed every embed at a dead address, twice. Reading a new, explicitly
# named variable means an old value is simply ignored and links follow the
# request -- no manual clean-up needed for the fix to take effect.
PUBLIC_HOST_OVERRIDE = os.environ.get("CONSOLE_PUBLIC_HOST", "").strip()


def public_host(request: Request):
    """The address to build browser-facing URLs from. CONSOLE_PUBLIC_HOST wins
    when set, for deployments that reach the console through a proxy or a
    different name than the services themselves; otherwise the request's own
    hostname is used, which is correct by construction.
    """
    return PUBLIC_HOST_OVERRIDE or (request.url.hostname or "localhost")


def mlflow_url(request: Request):
    return f"http://{public_host(request)}:5000"


def grafana_url(request: Request):
    return f"http://{public_host(request)}:3000"


# --------------------------------------------------------------------------
# presentation shell -- no template engine needed for pages this simple.
# Design: a technical "instrument panel" identity fitting an anomaly-
# detection/MLOps console -- IBM Plex Mono for data (versions, scores,
# timestamps), IBM Plex Sans for prose, a deep teal signal accent kept
# separate from the semantic pass/fail colors, dark-first with a real light
# palette alongside it (prefers-color-scheme, not a toggle -- this is an
# internal tool, not something that needs a switch).
# --------------------------------------------------------------------------

PAGE_CSS = """
:root {
  color-scheme: light dark;

  /* 4px rhythm -- every gap, pad and margin resolves here. */
  --s1: 4px;  --s2: 8px;  --s3: 12px; --s4: 16px;
  --s5: 20px; --s6: 28px; --s7: 36px; --s8: 48px;
  --r-sm: 5px; --r-md: 8px;

  /* Neutrals carry a slight green-cool bias so they sit with the accent
     rather than reading as default grey. */
  --bg: #f4f7f6;
  --surface: #ffffff;
  --surface-2: #edf2f0;
  --surface-3: #e3eae8;
  --border: #dbe4e1;
  --border-2: #c5d2ce;
  --ink: #0b1614;
  --ink-2: #44554f;
  --ink-3: #75867f;

  --accent: #0b7a6e;
  --accent-2: #085f56;
  --accent-soft: #dff1ee;
  --on-accent: #ffffff;

  --ok: #15794e;   --ok-soft: #e0f3e8;
  --warn: #8a5709; --warn-soft: #faefd6;
  --crit: #a83a26; --crit-soft: #fae6e2;

  --rail: #0a1614;
  --rail-ink: #8da49e;
  --rail-ink-2: #ffffff;
  --rail-hover: rgba(255,255,255,.05);
  --rail-sel: rgba(255,255,255,.09);

  --shadow: 0 1px 2px rgba(11,22,20,.04);
  --shadow-2: 0 1px 3px rgba(11,22,20,.05), 0 8px 24px rgba(11,22,20,.05);

  --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #070e0d;
    --surface: #0e1817;
    --surface-2: #14201e;
    --surface-3: #1b2926;
    --border: #1f2f2c;
    --border-2: #2c403b;
    --ink: #e9f2ef;
    --ink-2: #a2b6b1;
    --ink-3: #748983;

    --accent: #34d8c2;
    --accent-2: #74ecda;
    --accent-soft: #0d2f2b;
    --on-accent: #04211d;

    --ok: #3fd086;   --ok-soft: #0d2c1d;
    --warn: #e4b264; --warn-soft: #2d2210;
    --crit: #ff7d66; --crit-soft: #331511;

    --rail: #050b0a;
    --rail-ink: #7a908a;
    --rail-ink-2: #eef7f4;

    --shadow: 0 1px 2px rgba(0,0,0,.3);
    --shadow-2: 0 1px 3px rgba(0,0,0,.4);
  }
}

* { box-sizing: border-box; }
/* The hidden attribute must win over any display a class sets: author CSS
   otherwise overrides the browser's own [hidden] rule, so hiding an element
   whose class sets display:flex silently did nothing -- which left the live
   view's "waiting" placeholder on screen above a fully rendered chart, and
   the embed placeholders squeezing the frames they were meant to give way to. */
[hidden] { display: none !important; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 14px; line-height: 1.5;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}
/* Mono is the structural face: anything that is a label, a heading, a
   number or a control reads as instrumentation. Sans carries prose only. */
.mono, .readout dd, td.mono, th, .eyebrow, .pill,
h1, h2, .kpi-value, .tile-value, .brand, .sidebar nav a, button, .link-btn {
  font-family: var(--mono); font-variant-numeric: tabular-nums;
}
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-2); text-decoration: underline; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: var(--r-sm); }

/* ---------- shell ---------- */
.layout { display: grid; grid-template-columns: 244px minmax(0,1fr); min-height: 100vh; }
.sidebar {
  background: var(--rail); color: var(--rail-ink);
  display: flex; flex-direction: column; padding: var(--s5) 0 var(--s4);
  position: sticky; top: 0; height: 100vh;
}
.sidebar .brand {
  font-size: 13px; font-weight: 600; color: var(--rail-ink-2); letter-spacing: -.01em;
  padding: 0 var(--s5) var(--s6); display: flex; align-items: center; gap: var(--s2);
}
.brand-mark {
  width: 7px; height: 7px; border-radius: 1px; flex-shrink: 0;
  background: var(--accent); box-shadow: 0 0 10px var(--accent);
}
.sidebar nav { display: flex; flex-direction: column; gap: 1px; padding: 0 var(--s3); }
.sidebar nav a {
  color: var(--rail-ink); font-size: 12.5px; letter-spacing: -.01em;
  padding: 9px var(--s3); border-radius: var(--r-sm); display: flex; align-items: center;
  transition: background .12s ease, color .12s ease;
}
.sidebar nav a:hover { color: var(--rail-ink-2); background: var(--rail-hover); text-decoration: none; }
.sidebar nav a.active { color: var(--rail-ink-2); background: var(--rail-sel); }
.sidebar-foot {
  margin-top: auto; padding: var(--s4) var(--s5) 0;
  border-top: 1px solid rgba(255,255,255,.06);
  display: flex; flex-direction: column; gap: var(--s2); font-size: 12px;
}
.sidebar-user { color: var(--rail-ink-2); font-family: var(--mono); overflow-wrap: anywhere; }
.sidebar-foot a, .sidebar-foot .link-btn { color: var(--rail-ink); font-family: var(--sans); font-size: 12px; }
.sidebar-foot a:hover, .sidebar-foot .link-btn:hover { color: var(--rail-ink-2); }
.main { min-width: 0; padding: var(--s6) var(--s6) var(--s8); max-width: 1280px; }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar { position: static; height: auto; padding: var(--s4) 0 var(--s2); }
  .sidebar .brand { padding: 0 var(--s4) var(--s3); }
  .sidebar nav { flex-direction: row; flex-wrap: wrap; gap: var(--s1); padding: 0 var(--s3); }
  .sidebar-foot { flex-direction: row; align-items: center; gap: var(--s4); margin-top: var(--s3); padding: var(--s3) var(--s4) 0; }
  .main { padding: var(--s5) var(--s4) var(--s7); }
}

/* ---------- type ---------- */
h1 { font-size: 19px; font-weight: 600; letter-spacing: -.02em; margin: 0 0 var(--s1); line-height: 1.25; }
h2 { font-size: 12.5px; font-weight: 600; margin: 0; letter-spacing: -.005em; overflow-wrap: break-word; }
.page-sub { color: var(--ink-2); font-size: 13px; margin: 0 0 var(--s5); max-width: 68ch; }
p { color: var(--ink-2); margin: 0 0 var(--s3); }
p:last-child { margin-bottom: 0; }
.eyebrow { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-3); font-weight: 600; }
.text-muted { color: var(--ink-3); }

/* ---------- surfaces ---------- */
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s5); box-shadow: var(--shadow); margin-bottom: var(--s4);
}
.panel-header {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: var(--s2) var(--s4); margin-bottom: var(--s4);
  padding-bottom: var(--s3); border-bottom: 1px solid var(--border);
}
.panel-header:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
.panel-note { font-size: 11px; color: var(--ink-3); font-family: var(--sans); }

.pill {
  display: inline-flex; align-items: center; gap: var(--s1);
  padding: 3px 8px; border-radius: 3px;
  font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em;
  white-space: nowrap; flex-shrink: 0;
}
.pill-good { color: var(--ok); background: var(--ok-soft); }
.pill-warn { color: var(--warn); background: var(--warn-soft); }
.pill-bad { color: var(--crit); background: var(--crit-soft); }
.pill-neutral { color: var(--ink-3); background: var(--surface-2); }

.readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px,1fr)); gap: var(--s5) var(--s6); margin: 0; }
.readout > div { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.readout dt { font-size: 10px; text-transform: uppercase; letter-spacing: .09em; color: var(--ink-3); font-weight: 600; font-family: var(--mono); }
.readout dd { margin: 0; font-size: 13.5px; color: var(--ink); overflow-wrap: anywhere; }

/* ---------- tables ---------- */
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--r-md); background: var(--surface); box-shadow: var(--shadow); }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 10px var(--s4); font-size: 12.5px; border-bottom: 1px solid var(--border); white-space: nowrap; }
/* Figures align on their decimal point; text and timestamps stay left. */
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
th { font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-3); font-weight: 600; background: var(--surface-2); }
tbody tr:last-child td, tr:last-child td { border-bottom: none; }
tbody tr { transition: background .1s ease; }
tbody tr:hover td { background: var(--surface-2); }

/* ---------- forms ---------- */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: var(--s4); align-items: start; }
@media (max-width: 780px) { .grid-2 { grid-template-columns: 1fr; } }
form.stack { display: flex; flex-direction: column; gap: var(--s3); max-width: 340px; }
label { display: flex; flex-direction: column; gap: 5px; font-size: 11px; color: var(--ink-3); font-family: var(--mono); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
input[type=text], input[type=password], input[type=file] {
  font-family: var(--sans); font-size: 13.5px; padding: 9px var(--s3);
  border: 1px solid var(--border-2); border-radius: var(--r-sm);
  background: var(--surface); color: var(--ink); width: 100%; text-transform: none; letter-spacing: 0;
  transition: border-color .12s ease, box-shadow .12s ease;
}
input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
button {
  font-size: 12px; font-weight: 600; letter-spacing: .02em;
  padding: 9px var(--s5); border-radius: var(--r-sm);
  border: 1px solid var(--accent); background: var(--accent); color: var(--on-accent);
  cursor: pointer; align-self: flex-start; transition: background .12s ease, border-color .12s ease;
}
button:hover { background: var(--accent-2); border-color: var(--accent-2); }
.link-btn { background: none; border: none; padding: 0; font-size: 12px; font-weight: 500; color: var(--accent); cursor: pointer; }
.link-btn:hover { color: var(--accent-2); text-decoration: underline; }
td .link-btn { margin-right: var(--s3); }

/* ---------- auth ---------- */
.auth-shell { max-width: 340px; margin: 0 auto; padding: 15vh var(--s5) var(--s7); }
.auth-brand { font-family: var(--mono); font-weight: 600; font-size: 14px; color: var(--ink); display: flex; align-items: center; gap: var(--s2); margin-bottom: var(--s6); }
.auth-shell h1 { font-size: 16px; margin-bottom: var(--s5); }
.auth-shell form.stack { max-width: none; }
.auth-shell button { align-self: stretch; text-align: center; margin-top: var(--s1); }
.auth-shell p { font-size: 12px; margin-top: var(--s4); }

/* ---------- feedback ---------- */
.notice { padding: var(--s3) var(--s4); border-radius: var(--r-md); border: 1px solid var(--border); margin-bottom: var(--s4); font-size: 12.5px; overflow-wrap: anywhere; }
.notice-good { border-color: var(--ok); background: var(--ok-soft); color: var(--ok); }
.notice-bad { border-color: var(--crit); background: var(--crit-soft); color: var(--crit); }
.notice pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: var(--s2) 0 0; font-family: var(--mono); font-size: 11px; }
/* ---------- page head ---------- */
.page-head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: var(--s3) var(--s5); margin-bottom: var(--s5); }
.page-head .page-sub { margin-bottom: 0; }
.page-meta { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s4); font-size: 11.5px; color: var(--ink-3); font-family: var(--mono); }

/* ---------- KPI cards ---------- */
.kpi-strip { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: var(--s3); margin-bottom: var(--s4); }
.kpi-strip.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
@media (max-width: 1180px) { .kpi-strip, .kpi-strip.five { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 620px) { .kpi-strip, .kpi-strip.five { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md); padding: var(--s3) var(--s4) 13px; box-shadow: var(--shadow); min-width: 0; }
.kpi-label { font-size: 9.5px; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-3); font-weight: 600; font-family: var(--mono); }
.kpi-value { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 20px; font-weight: 600; margin-top: 5px; letter-spacing: -.02em; line-height: 1.15; overflow-wrap: anywhere; }
.kpi-value.good { color: var(--ok); }
.kpi-value.crit { color: var(--crit); }
.kpi-sub { font-size: 11px; color: var(--ink-3); margin-top: 3px; min-height: 1em; overflow-wrap: anywhere; }

/* ---------- layout grids ---------- */
.ov-grid, .metric-grid { display: grid; gap: var(--s4); align-items: stretch; }
.ov-grid { grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); }
.metric-grid { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
.metric-grid .span-2 { grid-column: 1 / -1; }
@media (max-width: 1080px) { .ov-grid, .metric-grid { grid-template-columns: 1fr; } .metric-grid .span-2 { grid-column: auto; } }
.ov-grid > .panel, .metric-grid > .panel { margin-bottom: 0; display: flex; flex-direction: column; min-width: 0; }
.ov-grid .card-link { margin-top: auto; padding-top: var(--s4); }

/* ---------- health list ---------- */
.health-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; }
.hl-row { display: flex; align-items: center; justify-content: space-between; gap: var(--s3); padding: 11px 0; border-bottom: 1px solid var(--border); min-width: 0; }
.hl-row:last-child { border-bottom: none; padding-bottom: 0; }
.hl-row:first-child { padding-top: 0; }
.hl-name { display: flex; align-items: center; gap: var(--s2); font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink); white-space: nowrap; }
.hl-detail { font-size: 12px; color: var(--ink-3); text-align: right; overflow-wrap: anywhere; }
.hl-row.down .hl-detail { color: var(--crit); }
.health-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--ink-3); display: inline-block; }
.hl-row.up .health-dot { background: var(--ok); box-shadow: 0 0 0 3px var(--ok-soft); }
.hl-row.down .health-dot { background: var(--crit); box-shadow: 0 0 0 3px var(--crit-soft); }

/* ---------- server-rendered charts (dataset_uploader/charts.py) ---------- */
.chart-svg { width: 100%; height: auto; display: block; }
.chart-svg .c-plot { fill: var(--surface-2); opacity: .45; }
.chart-svg .c-grid { stroke: var(--border); stroke-width: 1; }
.chart-svg .c-label { fill: var(--ink-3); font-family: var(--mono); font-size: 10.5px; }
.chart-svg .c-area.s0 { fill: var(--accent); opacity: .09; }
.chart-svg .c-line { fill: none; stroke-width: 1.7; stroke-linejoin: round; stroke-linecap: round; }
.chart-svg .c-line.s0 { stroke: var(--accent); }
.chart-svg .c-line.s1 { stroke: #6f8cff; }
.chart-svg .c-line.s2 { stroke: var(--warn); }
.chart-svg .c-band { fill: var(--crit); opacity: .09; }
.chart-svg .c-mark { fill: var(--crit); }
.chart-svg .c-threshold { stroke: var(--ink-2); stroke-width: 1; stroke-dasharray: 4 4; opacity: .7; }
.chart-svg .c-threshold-label { fill: var(--ink-2); font-family: var(--mono); font-size: 10.5px;
  paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; }
.chart-empty { display: flex; align-items: center; justify-content: center; min-height: 170px; border: 1px dashed var(--border-2); border-radius: var(--r-sm); background: var(--surface-2); color: var(--ink-3); font-size: 12.5px; text-align: center; padding: var(--s4); }
.chart-empty.ok { color: var(--ok); border-color: var(--ok); background: var(--ok-soft); min-height: 90px; font-family: var(--mono); font-size: 12px; }
.chart-legend { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s5); margin-top: var(--s3); font-size: 11px; color: var(--ink-3); }
.chart-legend span { display: inline-flex; align-items: center; gap: var(--s2); }
.c-key { width: 10px; height: 3px; border-radius: 2px; display: inline-block; background: var(--accent); }
.c-key.s1 { background: #6f8cff; }
.c-key.s2 { background: var(--warn); }
.c-key.band { height: 10px; width: 10px; background: var(--crit); opacity: .35; }
.c-key.mark { height: 7px; width: 7px; border-radius: 50%; background: var(--crit); }
.legend-link { margin-left: auto; font-size: 12px; }
.table-wrap.flat { box-shadow: none; }
.card-link { margin: var(--s4) 0 0; font-size: 12.5px; }
.state-line { font-family: var(--mono); font-size: 18px; font-weight: 600; letter-spacing: -.02em; margin: 0 0 var(--s3); }
.state-line.idle { color: var(--ink-3); }
.model-name { font-family: var(--mono); font-size: 17px; font-weight: 600; letter-spacing: -.02em; color: var(--ink); margin: 0; }
.model-type { font-family: var(--mono); font-size: 12px; color: var(--ink-3); margin: 2px 0 var(--s4); }
.gate { padding: var(--s4); background: var(--surface-2); border-radius: var(--r-sm); margin-bottom: var(--s4); }
.gate-top { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s3); margin-bottom: var(--s3); }
.gate-value { font-family: var(--mono); font-size: 24px; font-weight: 600; letter-spacing: -.03em; color: var(--ink); }
.gate-bar { position: relative; height: 6px; border-radius: 3px; background: var(--surface-3); }
.gate-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--ok); }
.gate-fill.below { background: var(--crit); }
.gate-mark { position: absolute; top: -4px; bottom: -4px; width: 2px; margin-left: -1px; border-radius: 1px; background: var(--ink); }
.gate-caption { position: relative; display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10.5px; color: var(--ink-3); margin-top: var(--s2); }
.gate-label { position: absolute; top: 0; transform: translateX(-50%); color: var(--ink-2); white-space: nowrap; }
.facts { display: grid; grid-template-columns: auto 1fr; gap: 9px var(--s4); margin: 0; }
.facts dt { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .09em; color: var(--ink-3); font-weight: 600; align-self: center; }
.facts dd { margin: 0; font-family: var(--mono); font-size: 12.5px; text-align: right; color: var(--ink); overflow-wrap: anywhere; }
.embed-holder { min-height: 72vh; display: flex; }
.embed-frame { width: 100%; height: 72vh; border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--surface); display: block; }
.embed-fallback { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: var(--s2);
  border: 1px dashed var(--border-2); border-radius: var(--r-sm); background: var(--surface-2);
  padding: var(--s6); text-align: center; color: var(--ink-2); font-size: 12.5px; }
.embed-fallback-title { font-family: var(--mono); font-weight: 600; color: var(--ink); font-size: 13px; margin: 0; }
.embed-fallback p { margin: 0; }
.embed-fallback code { font-family: var(--mono); font-size: 11.5px; overflow-wrap: anywhere; }
.field-block { margin: var(--s4) 0 0; }
.field-block-tight { margin: var(--s2) 0 0; }
.checkbox-row { flex-direction: row; align-items: center; gap: var(--s2); text-transform: none; letter-spacing: 0; }
.checkbox-row input { width: auto; }
.inline-form { display: inline; }
.empty-cell { color: var(--ink-3); text-align: center; }
"""

# Inlined so the tab mark needs no static file route, and matches the accent
# square in the sidebar brand.
FAVICON = (
    '<link rel="icon" href="data:image/svg+xml,'
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%230a1614'/%3E"
    "%3Crect x='9' y='9' width='14' height='14' rx='3' fill='%2334d8c2'/%3E"
    "%3C/svg%3E\">"
)

FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
)


BRAND = '<span class="brand-mark"></span>Sentinel<span class="text-muted">/</span>Ops'

# nav key -> (href, label). The key is what callers pass as `active` so the
# current section highlights itself; without it every page looks identical in
# the sidebar and you lose your place.
NAV_ITEMS = (
    ("overview", "/", "Overview"),
    ("live", "/live", "Live telemetry"),
    ("metrics", "/monitoring", "Service metrics"),
    ("experiments", "/experiments", "Experiments"),
    ("deploy", "/upload", "Deploy model"),
    ("history", "/history", "Model history"),
)


def page(title, body, account=None, auth=False, extra_css="", active="", refresh_seconds=None):
    head = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} · Sentinel Ops</title>
  {f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">' if refresh_seconds else ''}
  {FAVICON}
  {FONT_LINK}
  <style>{PAGE_CSS}{extra_css}</style>
</head>
<body>"""

    if auth or not account:
        return f"""{head}
  <div class="auth-shell">
    <div class="auth-brand">{BRAND}</div>
    {body}
  </div>
</body>
</html>
"""

    links = "".join(
        f'<a href="{href}" class="{"active" if key == active else ""}">{label}</a>'
        for key, href, label in NAV_ITEMS
    )
    if account["is_admin"]:
        links += f'<a href="/admin/users" class="{"active" if active == "users" else ""}">Users</a>'

    return f"""{head}
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">{BRAND}</div>
      <nav>{links}</nav>
      <div class="sidebar-foot">
        <span class="sidebar-user">{escape(account["username"])}</span>
        <a href="/account">Account settings</a>
        <form action="/logout" method="post">
          <button type="submit" class="link-btn">Sign out</button>
        </form>
      </div>
    </aside>
    <main class="main">
      {body}
    </main>
  </div>
</body>
</html>
"""


def current_account(request: Request):
    """Resolves the session's username to the live account row, so a role
    change (or a deleted account) takes effect on the next request rather
    than persisting until that user happens to log out.
    """
    username = request.session.get("username")
    if not username:
        return None
    return users.get_by_username(db_engine, username)


def require_login(request: Request):
    account = current_account(request)
    if not account:
        return None, RedirectResponse("/login", status_code=303)
    return account, None


def require_admin(request: Request):
    """Same contract as require_login, but non-admins get a 403 page rather
    than a redirect -- they ARE logged in, so bouncing them to /login would
    just loop them back here.
    """
    account, redirect = require_login(request)
    if redirect:
        return None, redirect
    if not account["is_admin"]:
        return None, HTMLResponse(
            page(
                "Not allowed",
                "<h1>Not allowed</h1><div class='notice notice-bad'>Only an administrator can manage "
                "user accounts.</div><p><a href='/'>Back to the dashboard</a></p>",
                account,
            ),
            status_code=403,
        )
    return account, None


def stage_pill(stage):
    return f'<span class="pill {stage_css_class(stage)}">{escape(stage)}</span>'


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

# Self-service signup exists ONLY to bootstrap the very first account, which
# becomes the administrator. Once any account exists, accounts are created by
# an administrator at /admin/users -- so a stranger who reaches this box's
# public IP can't mint themselves a login even if SIGNUP_CODE leaks.
SIGNUP_CLOSED_HTML = (
    "<h1>Create account</h1>"
    "<div class='notice notice-bad'>Self-service signup is closed. Accounts on this console are "
    "created by an administrator.</div>"
    "<p><a href='/login'>Back to log in</a></p>"
)


def _auth_error(title, message, status_code):
    return HTMLResponse(
        page(title, f"<h1>{escape(title)}</h1><div class='notice notice-bad'>{escape(message)}</div>"
                    f"<p><a href='/login'>Back to log in</a></p>", auth=True),
        status_code=status_code,
    )


@app.get("/signup", response_class=HTMLResponse)
def signup_form():
    if users.count(db_engine) > 0:
        return HTMLResponse(page("Sign up", SIGNUP_CLOSED_HTML, auth=True), status_code=403)
    return page("Sign up", """
        <h1>Create the first account</h1>
        <p>This account becomes the console administrator and is the only one that can create
        further accounts.</p>
        <form class="stack" action="/signup" method="post">
          <label>Username <input name="username" type="text" required></label>
          <label>Password <input name="password" type="password" required></label>
          <label>Signup code <input name="signup_code" type="text" required></label>
          <button type="submit">Create administrator</button>
        </form>
    """, auth=True)


@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...), signup_code: str = Form(...)):
    if users.count(db_engine) > 0:
        return HTMLResponse(page("Sign up", SIGNUP_CLOSED_HTML, auth=True), status_code=403)
    if signup_code != SIGNUP_CODE:
        return _auth_error("Sign up", "Wrong signup code.", 403)

    for error in (validate_username(username), validate_password(password)):
        if error:
            return _auth_error("Sign up", error, 400)

    try:
        users.create(db_engine, username, pwd_context.hash(password), is_admin=True)
    except Exception:
        return _auth_error("Sign up", "That username is already taken.", 400)

    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form():
    # The bootstrap link only appears while no account exists at all; after
    # that there is nothing for a visitor to sign up for.
    bootstrap = (
        '<p><a href="/signup">First time here? Create the administrator account</a></p>'
        if users.count(db_engine) == 0 else
        '<p class="text-muted">Accounts are created by an administrator.</p>'
    )
    return page("Log in", f"""
        <h1>Log in</h1>
        <form class="stack" action="/login" method="post">
          <label>Username <input name="username" type="text" required></label>
          <label>Password <input name="password" type="password" required></label>
          <button type="submit">Log in</button>
        </form>
        {bootstrap}
    """, auth=True)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    account = users.get_by_username(db_engine, username)
    if account is None or not pwd_context.verify(password, account["password_hash"]):
        return _auth_error("Log in", "Wrong username or password.", 401)

    request.session["username"] = username
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# --------------------------------------------------------------------------
# dashboard + history (reads MLflow's Model Registry -- the deployment record)
# --------------------------------------------------------------------------

def _registry_versions():
    client = MlflowClient()
    try:
        return client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
    except Exception:
        return []


def _sorted_versions():
    return sorted(_registry_versions(), key=lambda v: int(v.version), reverse=True)


def _production_version():
    """The model version currently serving predictions, or None if nothing
    has been promoted yet (a fresh deployment). Returns None rather than
    raising so /live and the dashboard both degrade to "no model yet"
    instead of erroring when MLflow is empty or unreachable.
    """
    return next((v for v in _sorted_versions() if v.current_stage == "Production"), None)


def _embed_panel(title, src, direct_url, frame_id, what):
    """An embedded tool with an honest failure state.

    A cross-origin frame that fails to load renders as nothing at all -- no
    error, no clue -- which is what made these pages look broken for a long
    time. So the frame starts hidden behind a placeholder and is only revealed
    once it actually loads; if the load event never arrives, the placeholder
    becomes a diagnosis naming the address it tried.
    """
    return f"""
    <div class="panel">
      <div class="panel-header">
        <h2>{escape(title)}</h2>
        <span class="panel-note"><a href="{escape(direct_url)}" target="_blank" rel="noopener">Open directly &rarr;</a></span>
      </div>
      <div class="embed-holder">
        <div class="embed-fallback" id="{frame_id}-fallback">
          <p class="embed-fallback-title">Loading{escape(what)}&hellip;</p>
        </div>
        <iframe class="embed-frame" id="{frame_id}" src="{escape(src)}"
                title="{escape(title)}" hidden></iframe>
      </div>
    </div>
    <script>
    (function () {{
      var frame = document.getElementById('{frame_id}');
      var fallback = document.getElementById('{frame_id}-fallback');
      var loaded = false;
      frame.addEventListener('load', function () {{
        loaded = true; frame.hidden = false; fallback.hidden = true;
      }});
      setTimeout(function () {{
        if (loaded) {{ return; }}
        fallback.innerHTML =
          '<p class="embed-fallback-title">Could not load{escape(what)}</p>' +
          '<p>Embedded from <code>{escape(src)}</code>. If that address is not reachable from this ' +
          'browser, the frame stays empty.</p>' +
          '<p>Open it directly to see the underlying error. If the address above is not where this ' +
          'deployment actually serves, unset <code>CONSOLE_PUBLIC_HOST</code> so links follow the address ' +
          'you reached this console on.</p>' +
          '<p><a href="{escape(direct_url)}" target="_blank" rel="noopener">Open directly &rarr;</a></p>';
      }}, 8000);
    }})();
    </script>
    """


# --------------------------------------------------------------------------
# system health -- checked server-side, on the internal network
# --------------------------------------------------------------------------

# Long-lived, so a check that overruns its timeout can be abandoned without
# the request waiting on it: a `with` block would join every thread on exit.
_health_pool = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="health")


def _check_prediction_api():
    # A real scoring request, not just a ping: the welcome route answers even
    # when no model can be loaded, which is precisely the failure that matters.
    window = [
        {metric: params["mean"] for metric, params in config.EASY_DATA_METRIC_PARAMS.items()}
    ] * config.WINDOW_SIZE
    response = http.post(
        f"{APP_INTERNAL_URL}/prediction_api", json={"readings": window}, timeout=HEALTH_TIMEOUT_SECONDS
    )
    if response.status_code == 200:
        return True, "Scoring requests"
    return False, f"HTTP {response.status_code} on a test prediction"


def _check_collector():
    rows = live_store.recent_readings(1)
    if not rows:
        return False, "No readings recorded yet"
    age = (datetime.now(timezone.utc) - rows[-1]["t"]).total_seconds()
    return age <= STALE_AFTER_SECONDS, f"Last reading {format_age(age)}"


def _check_grafana():
    response = http.get(f"{GRAFANA_INTERNAL_URL}/api/health", timeout=HEALTH_TIMEOUT_SECONDS)
    return response.ok, ("Serving dashboards" if response.ok else f"HTTP {response.status_code}")


def _check_mlflow():
    response = http.get(f"{config.TRACKING_URI}/health", timeout=HEALTH_TIMEOUT_SECONDS)
    return response.ok, ("Tracking server up" if response.ok else f"HTTP {response.status_code}")


def _check_prometheus():
    response = http.get(f"{PROMETHEUS_INTERNAL_URL}/-/ready", timeout=HEALTH_TIMEOUT_SECONDS)
    return response.ok, ("Collecting metrics" if response.ok else f"HTTP {response.status_code}")


def _prom_range(query):
    """One range query over the metrics window, as a list of
    {"metric", "xs", "ys"} series. NaN and infinite samples (what
    histogram_quantile returns for an idle interval) become gaps.
    """
    end = time.time()
    response = http.get(
        f"{PROMETHEUS_INTERNAL_URL}/api/v1/query_range",
        params={"query": query, "start": end - METRICS_WINDOW_SECONDS, "end": end, "step": METRICS_STEP_SECONDS},
        timeout=HEALTH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "success":
        raise RuntimeError(body.get("error", "query failed"))
    series = []
    for result in body["data"]["result"]:
        xs, ys = [], []
        for stamp, raw in result["values"]:
            xs.append(float(stamp))
            try:
                value = float(raw)
            except ValueError:
                value = None
            ys.append(value if value is not None and value == value and abs(value) != float("inf") else None)
        series.append({"metric": result["metric"], "xs": xs, "ys": ys})
    return series


def _gather_health():
    """Runs every check concurrently and returns (tiles, production_version).

    The registry lookup doubles as a health check and as the data for the
    serving-model card, so it runs once. Anything that raises or overruns is a
    failed check with the reason attached, never a failed page.
    """
    jobs = {
        "api": _health_pool.submit(_check_prediction_api),
        "registry": _health_pool.submit(_production_version),
        "collector": _health_pool.submit(_check_collector),
        "grafana": _health_pool.submit(_check_grafana),
        "mlflow": _health_pool.submit(_check_mlflow),
        "prometheus": _health_pool.submit(_check_prometheus),
    }
    results, production = {}, None
    for key, future in jobs.items():
        try:
            value = future.result(timeout=HEALTH_TIMEOUT_SECONDS + 4)
        except concurrent.futures.TimeoutError:
            results[key] = (False, "No response (timed out)")
            continue
        except Exception as exc:
            results[key] = (False, _short_error(exc))
            continue
        if key == "registry":
            production = value
            results[key] = (
                (True, f"v{value.version} in Production") if value else (False, "No model in Production")
            )
        else:
            results[key] = value

    tiles = [
        ("Prediction API", *results["api"]),
        ("Model registry", *results["registry"]),
        ("Telemetry", *results["collector"]),
        ("Grafana", *results["grafana"]),
        ("MLflow", *results["mlflow"]),
        ("Prometheus", *results["prometheus"]),
    ]
    return tiles, production


# The live view polls every few seconds; hitting MLflow's registry on every
# poll made the stream hang whenever MLflow was slow -- right after a restart,
# typically -- and a hung request never errors, so the page just waited
# forever. The serving version changes rarely, so it is cached briefly and the
# lookup is bounded; on failure the last known answer is kept.
PRODUCTION_CACHE_SECONDS = 30
_production_cache = {"value": None, "at": float("-inf")}


def _production_version_cached():
    now = time.monotonic()
    if now - _production_cache["at"] < PRODUCTION_CACHE_SECONDS:
        return _production_cache["value"]
    try:
        value = _health_pool.submit(_production_version).result(timeout=HEALTH_TIMEOUT_SECONDS)
    except Exception:
        value = _production_cache["value"]
    _production_cache.update(value=value, at=now)
    return value


def _short_error(exc):
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if "Connection refused" in text or "Failed to establish" in text or "NameResolution" in text:
        return "Unreachable"
    return text[:90]


def _updated_stamp():
    return datetime.now(GMT_PLUS_1_TZ).strftime("%H:%M:%S")


def _health_list_html(tiles):
    rows = "".join(
        f"""<li class="hl-row {'up' if ok else 'down'}">
              <span class="hl-name"><i class="health-dot"></i>{escape(name)}</span>
              <span class="hl-detail">{escape(detail)}</span>
            </li>"""
        for name, ok, detail in tiles
    )
    down = sum(1 for _, ok, _ in tiles if not ok)
    pill = ('<span class="pill pill-good">All up</span>' if not down
            else f'<span class="pill pill-bad">{down} down</span>')
    return f"""
    <section class="panel">
      <div class="panel-header"><h2>System health</h2>{pill}</div>
      <ul class="health-list">{rows}</ul>
    </section>
    """


def _kpi(label, value, sub="", tone=""):
    return f"""<div class="kpi">
      <div class="kpi-label">{escape(label)}</div>
      <div class="kpi-value {tone}">{escape(value)}</div>
      <div class="kpi-sub">{escape(sub)}</div>
    </div>"""


def _serving_panel(request, production):
    if production is None:
        return """
        <section class="panel">
          <div class="panel-header"><h2>Serving model</h2><span class="pill pill-neutral">None</span></div>
          <p class="state-line idle">Nothing deployed</p>
          <p>No model has cleared the quality gate yet. Retrain from a dataset or promote a trained model
          from <a href="/upload">Deploy model</a>.</p>
        </section>
        """, None

    run = mlflow.get_run(production.run_id)
    metrics = run.data.metrics
    tags = production.tags or {}
    f1 = metrics.get("f1_score")
    fill, marker, clears = gate_position(f1, config.F1_THRESHOLD)
    run_url = f"{mlflow_url(request)}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"

    def figure(key):
        value = metrics.get(key)
        return f"{value:.3f}" if value is not None else "--"

    html = f"""
    <section class="panel">
      <div class="panel-header"><h2>Serving model</h2>{stage_pill('Production')}</div>
      <p class="model-name">{escape(config.REGISTERED_MODEL_NAME)} <span class="text-muted">v{escape(str(production.version))}</span></p>
      <p class="model-type">{escape(tags.get('model_type', 'unknown'))}</p>
      <div class="gate">
        <div class="gate-top">
          <span class="eyebrow">F1 against quality gate</span>
          <span class="gate-value">{figure('f1_score')}</span>
        </div>
        <div class="gate-bar" role="img" aria-label="F1 {figure('f1_score')} against a gate of {config.F1_THRESHOLD}">
          <span class="gate-fill {'' if clears else 'below'}" style="width:{fill:.1f}%"></span>
          <span class="gate-mark" style="left:{marker:.1f}%"></span>
        </div>
        <div class="gate-caption">
          <span>0</span>
          <span class="gate-label" style="left:{marker:.1f}%">gate {config.F1_THRESHOLD}</span>
          <span>1.0</span>
        </div>
      </div>
      <dl class="facts">
        <dt>Precision</dt><dd>{figure('precision')}</dd>
        <dt>Recall</dt><dd>{figure('recall')}</dd>
        <dt>Source</dt><dd>{escape(source_label(tags.get('source', 'unknown')))}</dd>
        <dt>Registered</dt><dd>{escape(format_epoch_millis(production.creation_timestamp))}</dd>
      </dl>
      <p class="card-link"><a href="{run_url}" target="_blank" rel="noopener">View run in MLflow &rarr;</a></p>
    </section>
    """
    return html, f1


def _incident_rows(incidents, limit=6):
    if not incidents:
        return '<tr><td colspan="5" class="empty-cell">No incidents in the last hour.</td></tr>'
    rows = ""
    for event in list(reversed(incidents))[:limit]:
        label = SCENARIO_LABELS.get(event["name"], event["name"])
        latency = event["detection_latency_seconds"]
        pill = ('<span class="pill pill-good">Detected</span>' if event["detected"]
                else '<span class="pill pill-warn">Undetected</span>')
        rows += f"""<tr>
          <td>{escape(label)}</td>
          <td class="mono">{escape(event['started'].astimezone(GMT_PLUS_1_TZ).strftime('%H:%M:%S'))}</td>
          <td class="mono num">{event['duration_seconds']:.0f}s</td>
          <td>{pill}</td>
          <td class="mono num">{'--' if latency is None else f'{latency:.1f}s'}</td>
        </tr>"""
    return rows


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    tiles, production = _gather_health()
    try:
        readings = live_store.recent_readings(OVERVIEW_WINDOW)
    except Exception:
        readings = []
    incidents = summarize_incidents(readings)

    try:
        serving_html, f1 = _serving_panel(request, production)
    except Exception as exc:
        serving_html, f1 = f"""
        <section class="panel">
          <div class="panel-header"><h2>Serving model</h2><span class="pill pill-bad">Unavailable</span></div>
          <p>Could not read the model registry: {escape(_short_error(exc))}</p>
        </section>
        """, None

    scored = [r for r in readings if r["is_anomaly"] is not None]
    flagged = [r for r in scored if r["is_anomaly"]]
    detected = [e for e in incidents if e["detected"]]
    latencies = [e["detection_latency_seconds"] for e in detected if e["detection_latency_seconds"] is not None]
    latest = readings[-1] if readings else None
    age = (datetime.now(timezone.utc) - latest["t"]).total_seconds() if latest else None

    if latest is None:
        state_pill = '<span class="pill pill-neutral">No data</span>'
    elif age > STALE_AFTER_SECONDS:
        state_pill = '<span class="pill pill-neutral">Stale</span>'
    elif latest["is_anomaly"]:
        state_pill = '<span class="pill pill-bad">Anomaly</span>'
    else:
        state_pill = '<span class="pill pill-good">Normal</span>'

    kpis = "".join([
        _kpi("Serving model", f"v{production.version}" if production else "--",
             (production.tags or {}).get("model_type", "") if production else "nothing deployed"),
        _kpi("F1 score", f"{f1:.3f}" if f1 is not None else "--", f"gate {config.F1_THRESHOLD}",
             "" if f1 is None else ("good" if f1 >= config.F1_THRESHOLD else "crit")),
        _kpi("Flag rate", f"{100.0 * len(flagged) / len(scored):.1f}%" if scored else "--", "last hour"),
        _kpi("Incidents", str(len(incidents)), "last hour"),
        _kpi("Detected", f"{len(detected)} / {len(incidents)}" if incidents else "--",
             f"{100.0 * len(detected) / len(incidents):.0f}% caught" if incidents else "no incidents"),
        _kpi("Time to detect", f"{sum(latencies) / len(latencies):.1f}s" if latencies else "--", "mean"),
    ])

    # An hour of raw readings is ~1200 points -- too dense to read as a line.
    # Bucketed to 30s keeping each bucket's minimum, so every dip survives.
    bx, by, bf = charts.bucket_min(
        [r["t"].timestamp() for r in readings],
        [r["anomaly_score"] for r in readings],
        [bool(r["is_anomaly"]) for r in readings],
        OVERVIEW_BUCKET_SECONDS,
    )
    chart = charts.line_chart(
        [{"xs": bx, "ys": by, "label": "anomaly score"}],
        width=720,
        height=300,
        y_fmt=lambda v: f"{v:.2f}",
        threshold=0.0,
        threshold_label="anomaly boundary",
        bands=[(e["started"].timestamp(), e["ended"].timestamp()) for e in incidents],
        marks=[(x, y) for x, y, f in zip(bx, by, bf) if f and y is not None],
        empty_text="No telemetry recorded in the last hour.",
    )

    down = sum(1 for _, ok, _ in tiles if not ok)
    head_pill = ('<span class="pill pill-good">All systems operational</span>' if not down
                 else f'<span class="pill pill-bad">{down} service{"s" if down > 1 else ""} down</span>')

    body = f"""
    <div class="page-head">
      <div>
        <h1>Overview</h1>
        <p class="page-sub">Anomaly detection across EC2, ELB and RDS signals.</p>
      </div>
      <div class="page-meta">{head_pill}<span>Updated {_updated_stamp()} &middot; refreshes every 30s</span></div>
    </div>

    <div class="kpi-strip">{kpis}</div>

    <div class="ov-grid">
      <section class="panel">
        <div class="panel-header">
          <h2>Anomaly score &mdash; last hour</h2>
          {state_pill}
        </div>
        {chart}
        <div class="chart-legend">
          <span><i class="c-key s0"></i>score</span>
          <span><i class="c-key band"></i>incident window</span>
          <span><i class="c-key mark"></i>flagged reading</span>
          <a class="legend-link" href="/live">Live telemetry &rarr;</a>
        </div>
      </section>
      {serving_html}

      <section class="panel">
        <div class="panel-header">
          <h2>Recent incidents</h2>
          <span class="panel-note">detection measured against recorded incident windows</span>
        </div>
        <div class="table-wrap flat">
          <table>
            <tr><th>Type</th><th>Started</th><th class="num">Duration</th><th>Status</th><th class="num">Time to detect</th></tr>
            {_incident_rows(incidents)}
          </table>
        </div>
      </section>
      {_health_list_html(tiles)}
    </div>
    """
    return page("Overview", body, account, active="overview", refresh_seconds=30)


# --------------------------------------------------------------------------
# live monitoring feed (produced by live_feed/generator.py)
# --------------------------------------------------------------------------

@app.get("/live", response_class=HTMLResponse)
def live(request: Request):
    account, redirect = require_login(request)
    if redirect:
        return redirect
    return page("Live telemetry", live_view.LIVE_BODY, account, extra_css=live_view.LIVE_CSS, active="live")


@app.get("/live/data")
def live_data(request: Request):
    """JSON behind the /live page. Login-gated like every other console
    route -- these are the box's own operational readings, not public data.
    """
    account, redirect = require_login(request)
    if redirect:
        return redirect

    readings = live_store.recent_readings(LIVE_WINDOW)
    incidents = summarize_incidents(readings)
    latest = readings[-1] if readings else None

    now = datetime.now(timezone.utc)
    last_seen = latest["t"] if latest else None
    stale = last_seen is None or (now - last_seen).total_seconds() > STALE_AFTER_SECONDS

    scored = [r for r in readings if r["is_anomaly"] is not None]
    flagged = [r for r in scored if r["is_anomaly"]]

    production = _production_version_cached()
    return {
        "model": {
            "version": production.version if production else None,
            "type": (production.tags or {}).get("model_type", "unknown") if production else None,
        },
        "last_seen": last_seen.isoformat() if last_seen else None,
        "stale": stale,
        "current_anomaly": bool(latest and latest["is_anomaly"]),
        "current_incident": SCENARIO_LABELS.get(latest["incident"]) if latest and latest["incident"] else None,
        "anomaly_rate_pct": (100.0 * len(flagged) / len(scored)) if scored else 0.0,
        "readings": [
            {
                "t": r["t"].isoformat(),
                "cpu_usage_pct": r["cpu_usage_pct"],
                "network_in_bytes": r["network_in_bytes"],
                "elb_request_count": r["elb_request_count"],
                "rds_cpu_usage_pct": r["rds_cpu_usage_pct"],
                "anomaly_score": r["anomaly_score"],
                "is_anomaly": r["is_anomaly"],
                "incident": r["incident"],
            }
            for r in readings
        ],
        "incidents": [
            {
                "name": event["name"],
                "label": SCENARIO_LABELS.get(event["name"], event["name"]),
                "started": event["started"].isoformat(),
                "readings": event["readings"],
                "duration_seconds": event["duration_seconds"],
                "detected": event["detected"],
                "detection_latency_seconds": event["detection_latency_seconds"],
            }
            for event in incidents
        ],
    }


def _series_label(metric):
    handler = metric.get("handler", "all")
    status = metric.get("status")
    return f"{handler} ({status})" if status else handler


def _last(series):
    values = [y for s in series for y in s["ys"][-1:] if y is not None]
    return values


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring(request: Request):
    """Request throughput, p95 latency and 5xx rate for the prediction service,
    drawn natively from Prometheus.

    This page used to embed Grafana's own app in a frame. Embedding a
    third-party single-page app makes the page depend on the viewer's browser
    -- storage rules, extensions, cached bundles -- and it failed there
    repeatedly while Grafana itself was healthy. Querying Prometheus directly
    with the dashboard's own PromQL gives the same numbers, renders identically
    everywhere, and matches the rest of the console. The full Grafana
    dashboard stays one click away.
    """
    account, redirect = require_login(request)
    if redirect:
        return redirect

    jobs = {key: _health_pool.submit(_prom_range, query) for key, query in SERVICE_QUERIES.items()}
    data, failure = {}, None
    for key, future in jobs.items():
        try:
            data[key] = future.result(timeout=HEALTH_TIMEOUT_SECONDS + 4)
        except Exception as exc:
            data[key], failure = [], _short_error(exc)

    grafana_link = f"{grafana_url(request)}/d/mlops-app/mlops-app?orgId=1"
    head = f"""
    <div class="page-head">
      <div>
        <h1>Service metrics</h1>
        <p class="page-sub">Throughput, latency and errors for the prediction service over the last hour,
        from Prometheus.</p>
      </div>
      <div class="page-meta"><a href="{escape(grafana_link)}" target="_blank" rel="noopener">Open in Grafana &rarr;</a>
        <span>Updated {_updated_stamp()} &middot; refreshes every 30s</span></div>
    </div>
    """

    if failure and not any(data.values()):
        body = head + f"""
        <div class="panel">
          <div class="panel-header"><h2>Metrics unavailable</h2><span class="pill pill-bad">Prometheus</span></div>
          <p>Could not query Prometheus: {escape(failure)}. The full dashboard may still be reachable in
          <a href="{escape(grafana_link)}" target="_blank" rel="noopener">Grafana</a>.</p>
        </div>
        """
        return page("Service metrics", body, account, active="metrics", refresh_seconds=30)

    throughput, latency, errors = data["throughput"], data["latency"], data["errors"]
    latency_ms = [{**s, "ys": [None if y is None else y * 1000 for y in s["ys"]]} for s in latency]

    now_rps = sum(_last(throughput)) if _last(throughput) else None
    now_p95 = max(_last(latency_ms)) if _last(latency_ms) else None
    now_err = sum(_last(errors)) if _last(errors) else 0.0
    served = sum(y * METRICS_STEP_SECONDS for s in throughput for y in s["ys"] if y is not None)
    p95_points = [y for s in latency_ms for y in s["ys"] if y is not None]
    slo_ms = LATENCY_SLO_SECONDS * 1000
    within = (100.0 * sum(1 for y in p95_points if y <= slo_ms) / len(p95_points)) if p95_points else None

    kpis = "".join([
        _kpi("Throughput", f"{now_rps:.2f} req/s" if now_rps is not None else "--", "current, 5m rate"),
        _kpi("p95 latency", f"{now_p95:.0f} ms" if now_p95 is not None else "--", f"objective {slo_ms:.0f} ms",
             "" if now_p95 is None else ("good" if now_p95 <= slo_ms else "crit")),
        _kpi("Within objective", f"{within:.1f}%" if within is not None else "--", "of the last hour",
             "" if within is None else ("good" if within >= 99 else "crit")),
        _kpi("Error rate", f"{now_err:.3f} req/s", "5xx responses", "good" if now_err == 0 else "crit"),
        _kpi("Requests served", f"{served:,.0f}", "last hour"),
    ])

    for group in (throughput, latency_ms, errors):
        for s in group:
            s["label"] = _series_label(s["metric"])

    throughput_chart = charts.line_chart(throughput, width=520, height=240, y_fmt=lambda v: f"{v:.2f}",
                                         y_floor=0.0, empty_text="No requests recorded in the last hour.")
    latency_chart = charts.line_chart(latency_ms, width=520, height=240, y_fmt=lambda v: f"{v:.0f} ms",
                                      y_floor=0.0, threshold=slo_ms, threshold_label=f"objective {slo_ms:.0f} ms",
                                      empty_text="No latency samples in the last hour.")
    has_errors = any(y for s in errors for y in s["ys"] if y)
    errors_html = (
        charts.line_chart(errors, width=1080, height=220, y_fmt=lambda v: f"{v:.3f}", y_floor=0.0)
        + charts.legend(errors)
        if has_errors else
        '<div class="chart-empty ok">No 5xx responses in the last hour.</div>'
    )

    body = head + f"""
    <div class="kpi-strip five">{kpis}</div>
    <div class="metric-grid">
      <section class="panel">
        <div class="panel-header"><h2>Request throughput</h2><span class="panel-note">requests per second</span></div>
        {throughput_chart}{charts.legend(throughput)}
      </section>
      <section class="panel">
        <div class="panel-header"><h2>p95 latency</h2><span class="panel-note">95th percentile response time</span></div>
        {latency_chart}{charts.legend(latency_ms)}
      </section>
      <section class="panel span-2">
        <div class="panel-header"><h2>Server errors</h2><span class="panel-note">5xx responses per second</span></div>
        {errors_html}
      </section>
    </div>
    """
    return page("Service metrics", body, account, active="metrics", refresh_seconds=30)


@app.get("/experiments", response_class=HTMLResponse)
def experiments(request: Request):
    """MLflow, framed with the registry state the console already knows, so
    the page says what is deployed before handing over to MLflow's own UI for
    the run-level detail.
    """
    account, redirect = require_login(request)
    if redirect:
        return redirect

    versions = _sorted_versions()
    production = next((v for v in versions if v.current_stage == "Production"), None)
    summary = f"""
    <div class="panel">
      <div class="panel-header">
        <h2>Registry</h2>
        {stage_pill('Production') if production else '<span class="pill pill-neutral">None deployed</span>'}
      </div>
      <dl class="readout">
        <div><dt>Registered model</dt><dd class="mono">{escape(config.REGISTERED_MODEL_NAME)}</dd></div>
        <div><dt>Experiment</dt><dd class="mono">{escape(config.EXPERIMENT_NAME)}</dd></div>
        <div><dt>Versions</dt><dd class="mono">{len(versions)}</dd></div>
        <div><dt>Serving</dt><dd class="mono">{('v' + production.version) if production else '--'}</dd></div>
        <div><dt>Quality gate</dt><dd class="mono">F1 &ge; {config.F1_THRESHOLD}</dd></div>
      </dl>
    </div>
    """

    body = f"""
    <h1>Experiments</h1>
    <p class="page-sub">Every training run, its metrics and the registered model versions promoted from them,
    tracked in MLflow.</p>
    {summary}
    {_embed_panel(
        "MLflow tracking",
        mlflow_url(request),
        mlflow_url(request),
        "mlflow-frame",
        " MLflow",
    )}
    """
    return page("Experiments", body, account, active="experiments")


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    rows = ""
    for v in _sorted_versions():
        run = mlflow.get_run(v.run_id)
        metrics = run.data.metrics
        tags = v.tags or {}
        run_url = f"{mlflow_url(request)}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        rows += f"""
        <tr>
          <td class="mono">v{v.version}</td>
          <td>{stage_pill(v.current_stage)}</td>
          <td>{escape(tags.get('model_type', 'unknown'))}</td>
          <td class="mono num">{metrics.get('f1_score', float('nan')):.4f}</td>
          <td class="mono num">{metrics.get('precision', float('nan')):.4f}</td>
          <td>{escape(source_label(tags.get('source', 'unknown')))}</td>
          <td>{escape(tags.get('dataset_uploaded_by', 'unknown'))}</td>
          <td class="mono">{escape(format_timestamp(tags.get('dataset_uploaded_at', 'unknown')))}</td>
          <td class="mono">{escape(format_epoch_millis(v.creation_timestamp))}</td>
          <td><a href="{run_url}" target="_blank">View</a></td>
        </tr>
        """

    body = f"""
    <h1>Model history</h1>
    <p class="page-sub">Every version registered to the model registry, newest first.</p>
    <div class="table-wrap">
      <table>
        <tr>
          <th>Version</th><th>Stage</th><th>Model type</th><th class="num">F1</th><th class="num">Precision</th>
          <th>Source</th><th>Uploaded by</th><th>Dataset at</th><th>Registered at</th><th>Run</th>
        </tr>
        {rows or '<tr><td colspan="10" class="empty-cell">No models registered yet.</td></tr>'}
      </table>
    </div>
    """
    return page("Model history", body, account, active="history")


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------

@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request, message: str = "", ok: str = ""):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    notice = ""
    if message:
        notice_class = "notice-good" if ok else "notice-bad"
        notice = f'<div class="notice {notice_class}">{escape(message)}</div>'

    body = f"""
    <h1>Deploy a model</h1>
    <p class="page-sub">Two routes to Production: retrain from a new dataset through the delivery pipeline, or promote an already-trained model directly.</p>
    {notice}
    <div class="grid-2">
      <div class="panel">
        <div class="panel-header"><h2>New dataset</h2></div>
        <p>Versioned through DVC and trained by the delivery pipeline. Takes several minutes and only reaches Production if it clears the quality gate.</p>
        <form class="stack" action="/upload/dataset" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".csv" required>
          <button type="submit">Upload dataset</button>
        </form>
      </div>
      <div class="panel">
        <div class="panel-header"><h2>Pre-trained model</h2></div>
        <p>Evaluated immediately against the held-out split and promoted straight to Production
        if it clears the gate. Live within a few minutes, no pipeline run required.</p>
        <form class="stack" action="/upload/model" method="post" enctype="multipart/form-data">
          <input type="file" name="file" required>
          <button type="submit">Upload model</button>
        </form>
      </div>
    </div>
    """
    return page("Deploy model", body, account, active="deploy")


def _run(cmd, cwd=REPO_DIR):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _push_to_github(commit_message, username):
    remote_url = _run(["git", "remote", "get-url", "origin"]).strip()
    repo_path = remote_url.split("github.com/")[-1].removesuffix(".git")
    token_url = f"https://x-access-token:{GIT_TOKEN}@github.com/{repo_path}.git"

    _run(["git", "config", "user.name", username])
    _run(["git", "config", "user.email", f"{username}@dataset-uploader"])
    _run(["git", "commit", "-m", commit_message])
    # Pushed to an explicit token URL (never written to .git/config, which is
    # the real bind-mounted host repo) rather than `git push origin main`.
    _run(["git", "push", token_url, "HEAD:main"])


@app.post("/upload/dataset", response_class=HTMLResponse)
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        return RedirectResponse(f"/upload?message=Could not read CSV: {exc}", status_code=303)

    missing = missing_dataset_columns(df.columns)
    if missing:
        return RedirectResponse(f"/upload?message=Missing required columns: {sorted(missing)}", status_code=303)

    try:
        _run(["git", "pull"])

        with open(DATASET_PATH, "wb") as f:
            f.write(contents)

        uploaded_at = now_gmt1().isoformat()
        with open(DATASET_META_PATH, "w") as f:
            json.dump({"uploaded_by": account["username"], "uploaded_at": uploaded_at}, f)

        # The committed .dvc/config points at localhost:9000 (a local-dev
        # default -- see README.md's DVC section); from inside this
        # container MinIO is reachable at config.MINIO_ENDPOINT_URL
        # (http://minio:9000) instead, so override the remote the same way
        # CI does before every dvc pull/push.
        _run(["dvc", "remote", "modify", "--local", "myremote", "endpointurl", config.MINIO_ENDPOINT_URL])
        _run(["dvc", "add", DATASET_PATH])
        _run(["dvc", "push"])
        _run(["git", "add", "prediction_model/datasets/dataset.csv.dvc", "prediction_model/datasets/dataset.meta.json"])
        _push_to_github(f"Update dataset (uploaded by {account['username']})", account["username"])
    except Exception as exc:
        return HTMLResponse(page(
            "Upload failed",
            f"<h1>Upload failed</h1><div class='notice notice-bad'><pre>{escape(str(exc))}</pre></div><p><a href='/upload'>Back</a></p>",
            account,
        ))

    return HTMLResponse(page(
        "Upload succeeded",
        f"""
        <h1>Upload succeeded</h1>
        <div class="notice notice-good">Dataset published. The delivery pipeline is now running.</div>
        <p><a href="{GITHUB_ACTIONS_URL}" target="_blank">Follow the pipeline run</a></p>
        """,
        account,
    ))


@app.post("/upload/model", response_class=HTMLResponse)
async def upload_model(request: Request, file: UploadFile = File(...)):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    contents = await file.read()
    try:
        pipeline = cloudpickle.loads(contents)
        if not hasattr(pipeline, "predict"):
            raise ValueError("Uploaded object has no .predict() method -- not a usable pipeline.")
    except Exception as exc:
        return HTMLResponse(page(
            "Upload failed",
            f"<h1>Upload failed</h1><div class='notice notice-bad'>Could not load model: {escape(str(exc))}</div><p><a href='/upload'>Back</a></p>",
            account,
        ))

    try:
        dataset = load_full_dataset()
        _, eval_df = day_block_split(dataset)
        metrics = evaluate_pipeline(pipeline, eval_df)
    except Exception as exc:
        return HTMLResponse(page(
            "Upload failed",
            f"<h1>Upload failed</h1><div class='notice notice-bad'>Could not evaluate model: {escape(str(exc))}</div><p><a href='/upload'>Back</a></p>",
            account,
        ))

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.set_tags({"source": "model_upload", "uploaded_by": account["username"]})
        # infer_model_type introspects the uploaded pipeline's own class
        # (rather than assuming it's an LSTMAutoencoder) since this route
        # accepts any object with .predict() -- see the module docstring's
        # note on that accepted tradeoff. Logged as a param (not just a tag)
        # so register_and_promote can read it back via mlflow.get_run and
        # copy it onto the registered model VERSION if this gets promoted.
        mlflow.log_param('model_type', infer_model_type(pipeline))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, config.MODEL_NAME.lstrip("/"), serialization_format="cloudpickle")
        run_id = run.info.run_id

    f1 = metrics["f1_score"]
    if f1 < config.F1_THRESHOLD:
        return HTMLResponse(page(
            "Model rejected",
            f"<h1>Model rejected</h1><div class='notice notice-bad'>F1 = {f1:.4f}, below the required {config.F1_THRESHOLD}. Not promoted.</div><p><a href='/upload'>Back</a></p>",
            account,
        ))

    register_and_promote(
        run_id, source="model_upload",
        dataset_meta={"uploaded_by": account["username"], "uploaded_at": now_gmt1().isoformat()},
    )
    return HTMLResponse(page(
        "Model promoted",
        f"<h1>Model promoted</h1><div class='notice notice-good'>F1 = {f1:.4f}. Promoted to Production &mdash; serving within a few minutes.</div><p><a href='/'>Dashboard</a></p>",
        account,
    ))


# --------------------------------------------------------------------------
# account management (admin-only, see dataset_uploader/users.py)
# --------------------------------------------------------------------------

def _users_page(account, message="", ok=False):
    notice = ""
    if message:
        notice = f'<div class="notice {"notice-good" if ok else "notice-bad"}">{escape(message)}</div>'

    rows = ""
    for row in users.list_all(db_engine):
        is_self = row["id"] == account["id"]
        pill = "pill-good" if row["is_admin"] else "pill-neutral"
        # Each control is its own POST form: these are state changes, so they
        # must not be reachable by a link someone can prefetch or share.
        role_action = "demote" if row["is_admin"] else "promote"
        role_button = f"""
            <form action="/admin/users/{row['id']}/role" method="post" class="inline-form">
              <input type="hidden" name="action" value="{role_action}">
              <button type="submit" class="link-btn">{role_action.capitalize()}</button>
            </form>
        """
        delete_button = f"""
            <form action="/admin/users/{row['id']}/delete" method="post" class="inline-form"
                  onsubmit="return confirm('Delete {escape(row['username'])}?');">
              <button type="submit" class="link-btn">Delete</button>
            </form>
        """
        rows += f"""
        <tr>
          <td>{escape(row['username'])}{' <span class="text-muted">(you)</span>' if is_self else ''}</td>
          <td><span class="pill {pill}">{escape(role_label(row['is_admin']))}</span></td>
          <td class="mono">{escape(format_epoch_millis(int(row['created_at'].timestamp() * 1000)) if row['created_at'] else 'unknown')}</td>
          <td>{role_button}{delete_button}</td>
        </tr>
        """

    body = f"""
    <h1>Users</h1>
    <p class="page-sub">Console accounts and their access level.</p>
    {notice}
    <div class="panel">
      <div class="panel-header"><h2>New account</h2></div>
      <p>New accounts are members by default &mdash; promote them here if they should also manage users.</p>
      <form class="stack" action="/admin/users/create" method="post">
        <label>Username <input name="username" type="text" required></label>
        <label>Password <input name="password" type="password" required></label>
        <label class="checkbox-row">
          <input type="checkbox" name="is_admin" value="1"> Administrator
        </label>
        <button type="submit">Create account</button>
      </form>
    </div>
    <div class="table-wrap">
      <table>
        <tr><th>Username</th><th>Role</th><th>Created</th><th>Actions</th></tr>
        {rows}
      </table>
    </div>
    """
    return page("Users", body, account, active="users")


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request, message: str = "", ok: str = ""):
    account, redirect = require_admin(request)
    if redirect:
        return redirect
    return _users_page(account, message, ok == "1")


@app.post("/admin/users/create")
def admin_create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    is_admin: str = Form(None),
):
    account, redirect = require_admin(request)
    if redirect:
        return redirect

    for error in (validate_username(username), validate_password(password)):
        if error:
            return RedirectResponse(f"/admin/users?message={error}", status_code=303)

    try:
        users.create(db_engine, username, pwd_context.hash(password), is_admin=bool(is_admin))
    except Exception:
        return RedirectResponse(
            f"/admin/users?message=That username is already taken.", status_code=303
        )

    return RedirectResponse(f"/admin/users?message=Created {username}.&ok=1", status_code=303)


@app.post("/admin/users/{user_id}/delete")
def admin_delete_user(request: Request, user_id: int):
    account, redirect = require_admin(request)
    if redirect:
        return redirect

    target = users.get_by_id(db_engine, user_id)
    if target is None:
        return RedirectResponse("/admin/users?message=No such account.", status_code=303)

    error = deletion_error(account["id"], target, users.admin_count(db_engine))
    if error:
        return RedirectResponse(f"/admin/users?message={error}", status_code=303)

    users.delete(db_engine, user_id)
    return RedirectResponse(
        f"/admin/users?message=Deleted {target['username']}.&ok=1", status_code=303
    )


@app.post("/admin/users/{user_id}/role")
def admin_set_role(request: Request, user_id: int, action: str = Form(...)):
    account, redirect = require_admin(request)
    if redirect:
        return redirect

    target = users.get_by_id(db_engine, user_id)
    if target is None:
        return RedirectResponse("/admin/users?message=No such account.", status_code=303)

    promote = action == "promote"
    if not promote:
        error = demotion_error(account["id"], target, users.admin_count(db_engine))
        if error:
            return RedirectResponse(f"/admin/users?message={error}", status_code=303)

    users.set_admin(db_engine, user_id, promote)
    verb = "Promoted" if promote else "Demoted"
    return RedirectResponse(
        f"/admin/users?message={verb} {target['username']}.&ok=1", status_code=303
    )


@app.get("/account", response_class=HTMLResponse)
def account_page(request: Request, message: str = "", ok: str = ""):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    notice = ""
    if message:
        notice = f'<div class="notice {"notice-good" if ok == "1" else "notice-bad"}">{escape(message)}</div>'

    body = f"""
    <h1>Account settings</h1>
    {notice}
    <div class="panel">
      <div class="panel-header">
        <h2>{escape(account['username'])}</h2>
        <span class="pill {'pill-good' if account['is_admin'] else 'pill-neutral'}">{escape(role_label(account['is_admin']))}</span>
      </div>
      <p>Change your password. An administrator set your initial one, so they know it until you do.</p>
      <form class="stack" action="/account/password" method="post">
        <label>Current password <input name="current_password" type="password" required></label>
        <label>New password <input name="new_password" type="password" required></label>
        <button type="submit">Change password</button>
      </form>
    </div>
    """
    return page("Account settings", body, account, active="")


@app.post("/account/password")
def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    if not pwd_context.verify(current_password, account["password_hash"]):
        return RedirectResponse("/account?message=Current password is wrong.", status_code=303)

    error = validate_password(new_password)
    if error:
        return RedirectResponse(f"/account?message={error}", status_code=303)

    users.set_password(db_engine, account["id"], pwd_context.hash(new_password))
    return RedirectResponse("/account?message=Password changed.&ok=1", status_code=303)
