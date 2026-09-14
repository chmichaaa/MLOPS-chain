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
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from html import escape

import cloudpickle
import mlflow
import pandas as pd
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from mlflow.tracking import MlflowClient
from passlib.context import CryptContext
from sqlalchemy import create_engine
from starlette.middleware.sessions import SessionMiddleware

from prediction_model.config import config
from prediction_model.processing.data_handling import load_full_dataset, day_block_split
from prediction_model.training_pipeline import evaluate_pipeline, register_and_promote, infer_model_type
from dataset_uploader import users, live_view
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
_PUBLIC_HOST_DEFAULT = "localhost"
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", _PUBLIC_HOST_DEFAULT)
# Flags the common misconfiguration where PUBLIC_HOST was never set on a real
# deployment: the /monitoring iframe would otherwise just silently render
# blank (the browser tries to load the VIEWER's OWN localhost:3000, not the
# server's), with nothing telling you why. See monitoring() below.
PUBLIC_HOST_IS_DEFAULT = PUBLIC_HOST == _PUBLIC_HOST_DEFAULT
MLFLOW_PUBLIC_URL = f"http://{PUBLIC_HOST}:5000"
# /d/<uid>/<slug> (with the slug) rather than the bare /d/<uid> -- the
# sluggless form 302-redirects to the slugged one, and that redirect can drop
# Grafana's embedding-allowed response headers in some browser/proxy
# combinations, leaving the iframe blank even with GF_SECURITY_ALLOW_EMBEDDING
# set correctly. Requesting the final URL directly avoids the redirect
# entirely. Must match grafana/provisioning/dashboards/mlops-app.json's
# uid/title exactly ("mlops-app" / "MLOps App" -> slug "mlops-app").
GRAFANA_DASHBOARD_URL = f"http://{PUBLIC_HOST}:3000/d/mlops-app/mlops-app?orgId=1&kiosk&refresh=30s"

SIGNUP_CODE = os.environ["SIGNUP_CODE"]
GIT_TOKEN = os.environ["GIT_TOKEN"]

db_engine = create_engine(os.environ["DATABASE_URL"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(title="MLOps Console")
app.add_middleware(SessionMiddleware, secret_key=os.environ["SESSION_SECRET_KEY"])

users.init_schema(db_engine)
# The live feed's writer (live_feed/generator.py) creates this too, but the
# console must not 500 on /live just because the feed container hasn't
# started yet -- both call the same idempotent DDL.
live_store.init_schema()


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
  --bg: #f5f8f7;
  --surface: #ffffff;
  --surface-2: #eaf0ee;
  --border: #d8e2df;
  --text: #11201c;
  --text-muted: #5a6d67;
  --accent: #0e7d72;
  --accent-strong: #0a5f56;
  --accent-contrast: #ffffff;
  --good: #1c8a5a;
  --good-bg: #e2f4ea;
  --bad: #b3402b;
  --bad-bg: #fbe8e4;
  --shadow: 0 1px 2px rgba(17, 32, 28, 0.06);
  --radius: 7px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b1412;
    --surface: #101c19;
    --surface-2: #16241f;
    --border: #223330;
    --text: #e6f1ee;
    --text-muted: #8fa69f;
    --accent: #35d6c1;
    --accent-strong: #7be9db;
    --accent-contrast: #06231f;
    --good: #3ecf83;
    --good-bg: #0f2e1e;
    --bad: #ff7a63;
    --bad-bg: #341712;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 15px;
  line-height: 1.6;
}
p, dl, dd, ul, ol { margin: 0 0 0.85rem; }
p:last-child, dl:last-child { margin-bottom: 0; }
.mono, .readout dd, td.mono { font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
a { color: var(--accent-strong); text-decoration: none; }
a:hover { text-decoration: underline; }
a:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.shell { max-width: 980px; margin: 0 auto; padding: 0 1.5rem 3.5rem; }
.topbar { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.75rem; padding: 1.15rem 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 2.5rem; }
.brand { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.92rem; letter-spacing: 0.01em; white-space: nowrap; }
.brand .dot { color: var(--accent); }
.topbar nav { display: flex; flex-wrap: wrap; gap: 0.5rem 1.4rem; flex: 1 1 auto; min-width: 0; }
.topbar nav a { color: var(--text-muted); font-size: 0.88rem; white-space: nowrap; }
.topbar nav a:hover { color: var(--text); text-decoration: none; }
.user-chip { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 0.9rem; font-size: 0.85rem; color: var(--text-muted); margin-left: auto; }
.link-btn { background: none; border: none; padding: 0; font: inherit; color: var(--text-muted); cursor: pointer; text-decoration: underline; }
.link-btn:hover { color: var(--text); }
h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 1.5rem; }
h2 { font-size: 1rem; font-weight: 600; margin: 0; min-width: 0; overflow-wrap: break-word; }
p { color: var(--text-muted); }
.eyebrow { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 600; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.3rem 1.5rem; box-shadow: var(--shadow); margin-bottom: 1.5rem; }
.panel-header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.5rem 1rem; margin-bottom: 1.1rem; }
.pill { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.18rem 0.65rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; flex-shrink: 0; }
.pill-good { color: var(--good); background: var(--good-bg); }
.pill-bad { color: var(--bad); background: var(--bad-bg); }
.pill-neutral { color: var(--text-muted); background: var(--surface-2); }
.readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1.1rem 1.5rem; margin: 0; }
.readout > div { display: flex; flex-direction: column; gap: 0.25rem; min-width: 0; }
.readout dt { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); }
.readout dd { margin: 0; font-size: 0.95rem; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 0.6rem 0.85rem; font-size: 0.85rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:last-child td { border-bottom: none; }
th { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; }
@media (max-width: 720px) { .grid-2 { grid-template-columns: 1fr; } }
form.stack { display: flex; flex-direction: column; gap: 0.9rem; max-width: 360px; }
label { display: flex; flex-direction: column; gap: 0.32rem; font-size: 0.85rem; color: var(--text-muted); }
input[type=text], input[type=password], input[type=file] {
  font: inherit; padding: 0.55rem 0.7rem; border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--surface); color: var(--text); width: 100%;
}
input:focus { border-color: var(--accent); }
button { font: inherit; font-weight: 600; padding: 0.58rem 1.15rem; border-radius: var(--radius); border: 1px solid var(--accent); background: var(--accent); color: var(--accent-contrast); cursor: pointer; align-self: flex-start; }
button:hover { background: var(--accent-strong); border-color: var(--accent-strong); }
.auth-shell { max-width: 380px; margin: 4.5rem auto; padding: 0 1.5rem; }
.text-muted { color: var(--text-muted); }
.notice { padding: 0.9rem 1.1rem; border-radius: var(--radius); border: 1px solid var(--border); margin-bottom: 1.5rem; font-size: 0.9rem; overflow-wrap: anywhere; }
.notice-good { border-color: var(--good); background: var(--good-bg); color: var(--good); }
.notice-bad { border-color: var(--bad); background: var(--bad-bg); color: var(--bad); }
.notice pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5rem 0 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.embed-frame { width: 100%; height: 85vh; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); }
"""

FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
)


def page(title, body, account=None, auth=False, extra_css=""):
    nav = ""
    if account:
        admin_link = '<a href="/admin/users">Users</a>' if account["is_admin"] else ""
        nav = f"""
        <header class="topbar">
          <div class="brand">MLOps<span class="dot">::</span>Console</div>
          <nav>
            <a href="/">Dashboard</a>
            <a href="/live">Live</a>
            <a href="/monitoring">Monitoring</a>
            <a href="/upload">Upload</a>
            <a href="/history">History</a>
            {admin_link}
          </nav>
          <div class="user-chip">
            <span>{escape(account["username"])}</span>
            <a href="/account">Account</a>
            <form action="/logout" method="post">
              <button type="submit" class="link-btn">Log out</button>
            </form>
          </div>
        </header>
        """
    wrapper_class = "auth-shell" if auth else "shell"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} · MLOps Console</title>
  {FONT_LINK}
  <style>{PAGE_CSS}{extra_css}</style>
</head>
<body>
  {nav}
  <div class="{wrapper_class}">
    {body}
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


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    production = _production_version()

    if production is None:
        prod_html = """
        <div class="panel">
          <div class="panel-header"><h2>Production model</h2></div>
          <p style="margin:0;">No model has been promoted to Production yet. Upload a dataset or a trained model to get started.</p>
        </div>
        """
    else:
        run = mlflow.get_run(production.run_id)
        metrics = run.data.metrics
        tags = production.tags or {}
        run_url = f"{MLFLOW_PUBLIC_URL}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        prod_html = f"""
        <div class="panel">
          <div class="panel-header">
            <h2>Production model &middot; v{production.version}</h2>
            {stage_pill('Production')}
          </div>
          <dl class="readout">
            <div><dt>Model name</dt><dd class="mono">{escape(config.REGISTERED_MODEL_NAME)}</dd></div>
            <div><dt>Model type</dt><dd class="mono">{escape(tags.get('model_type', 'unknown'))}</dd></div>
            <div><dt>F1 score</dt><dd class="mono">{metrics.get('f1_score', float('nan')):.4f} <span class="text-muted">/ {config.F1_THRESHOLD}</span></dd></div>
            <div><dt>Precision</dt><dd class="mono">{metrics.get('precision', float('nan')):.4f}</dd></div>
            <div><dt>Recall</dt><dd class="mono">{metrics.get('recall', float('nan')):.4f}</dd></div>
            <div><dt>Accuracy</dt><dd class="mono">{metrics.get('accuracy', float('nan')):.4f}</dd></div>
            <div><dt>Source</dt><dd>{escape(source_label(tags.get('source', 'unknown')))}</dd></div>
            <div><dt>Dataset by</dt><dd>{escape(tags.get('dataset_uploaded_by', 'unknown'))}</dd></div>
            <div><dt>Dataset at</dt><dd class="mono">{escape(format_timestamp(tags.get('dataset_uploaded_at', 'unknown')))}</dd></div>
            <div><dt>Registered at</dt><dd class="mono">{escape(format_epoch_millis(production.creation_timestamp))}</dd></div>
          </dl>
          <p style="margin:1.1rem 0 0;"><span class="eyebrow">Hyperparameters</span><br>
            <span class="mono">{escape(format_hyperparams(run.data.params))}</span>
          </p>
          <p style="margin:0.6rem 0 0;"><a href="{run_url}" target="_blank">View this run in MLflow</a></p>
        </div>
        """

    body = f"""
    <h1>Dashboard</h1>
    {prod_html}
    <p><a href="{MLFLOW_PUBLIC_URL}" target="_blank">Open MLflow</a> &middot; <a href="{GITHUB_ACTIONS_URL}" target="_blank">Open GitHub Actions</a></p>
    """
    return page("Dashboard", body, account)


# --------------------------------------------------------------------------
# live monitoring feed (produced by live_feed/generator.py)
# --------------------------------------------------------------------------

LIVE_WINDOW = 180
# A reading is expected every live_feed TICK_SECONDS; treat the feed as stale
# well before an operator would start wondering, but with enough slack that a
# single slow scoring round-trip doesn't flip the indicator to red.
STALE_AFTER_SECONDS = 30


@app.get("/live", response_class=HTMLResponse)
def live(request: Request):
    account, redirect = require_login(request)
    if redirect:
        return redirect
    return page("Live", live_view.LIVE_BODY, account, extra_css=live_view.LIVE_CSS)


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

    production = _production_version()
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
                "detected": event["detected"],
                "detection_latency_seconds": event["detection_latency_seconds"],
            }
            for event in incidents
        ],
    }


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring(request: Request):
    account, redirect = require_login(request)
    if redirect:
        return redirect

    # Embeds the real Grafana dashboard (grafana/provisioning/dashboards/
    # mlops-app.json) rather than reimplementing charts here -- Grafana's own
    # anonymous-viewer access is enabled (docker-compose.yml's grafana
    # service) specifically so this iframe doesn't prompt for a second login;
    # editing/admin still requires the real Grafana login, anonymous access
    # is view-only. &kiosk hides Grafana's own nav chrome for a cleaner embed.
    warning = ""
    if PUBLIC_HOST_IS_DEFAULT:
        # The #1 cause of "the embedded dashboard is just blank": PUBLIC_HOST
        # was never set in .env on this deployment, so the iframe's src still
        # points at "localhost" -- meaning the VIEWER's OWN machine, not this
        # server. Surface that plainly instead of leaving a mysterious empty
        # box (see PUBLIC_HOST_IS_DEFAULT above for why the iframe alone can't
        # tell you this).
        warning = f"""
        <div class="notice notice-bad">
          <strong>PUBLIC_HOST is not set</strong> -- the dashboard below is trying to load from
          <code>{escape(GRAFANA_DASHBOARD_URL)}</code>, which is your own machine, not this server.
          Set <code>PUBLIC_HOST</code> in <code>.env</code> on the server to its public IP or domain,
          then restart the <code>dataset-uploader</code> service (<code>docker compose up -d dataset-uploader</code>).
        </div>
        """

    body = f"""
    <h1>Monitoring</h1>
    {warning}
    <iframe class="embed-frame" src="{GRAFANA_DASHBOARD_URL}" title="Grafana dashboard"></iframe>
    <p><a href="http://{PUBLIC_HOST}:3000" target="_blank">Open Grafana directly</a></p>
    """
    return page("Monitoring", body, account)


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
        run_url = f"{MLFLOW_PUBLIC_URL}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        rows += f"""
        <tr>
          <td class="mono">v{v.version}</td>
          <td>{stage_pill(v.current_stage)}</td>
          <td>{escape(tags.get('model_type', 'unknown'))}</td>
          <td class="mono">{metrics.get('f1_score', float('nan')):.4f}</td>
          <td class="mono">{metrics.get('precision', float('nan')):.4f}</td>
          <td>{escape(source_label(tags.get('source', 'unknown')))}</td>
          <td>{escape(tags.get('dataset_uploaded_by', 'unknown'))}</td>
          <td class="mono">{escape(format_timestamp(tags.get('dataset_uploaded_at', 'unknown')))}</td>
          <td class="mono">{escape(format_epoch_millis(v.creation_timestamp))}</td>
          <td><a href="{run_url}" target="_blank">View</a></td>
        </tr>
        """

    body = f"""
    <h1>Model history</h1>
    <div class="table-wrap">
      <table>
        <tr>
          <th>Version</th><th>Stage</th><th>Model type</th><th>F1</th><th>Precision</th>
          <th>Source</th><th>Uploaded by</th><th>Dataset at</th><th>Registered at</th><th>Run</th>
        </tr>
        {rows or '<tr><td colspan="10" style="color:var(--text-muted);text-align:center;">No models registered yet.</td></tr>'}
      </table>
    </div>
    """
    return page("History", body, account)


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
    <h1>Upload</h1>
    {notice}
    <div class="grid-2">
      <div class="panel">
        <div class="panel-header"><h2>Upload a dataset</h2></div>
        <p>Pushed through git/DVC and trained by the real CI/CD pipeline &mdash; takes real time.</p>
        <form class="stack" action="/upload/dataset" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".csv" required>
          <button type="submit">Upload dataset</button>
        </form>
      </div>
      <div class="panel">
        <div class="panel-header"><h2>Upload a trained model</h2></div>
        <p>Evaluated immediately against the current eval split and promoted straight to
        Production if it clears the gate &mdash; no CI run, live within a few minutes via
        predict.py's cache refresh.</p>
        <form class="stack" action="/upload/model" method="post" enctype="multipart/form-data">
          <input type="file" name="file" required>
          <button type="submit">Upload model</button>
        </form>
      </div>
    </div>
    """
    return page("Upload", body, account)


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
        <div class="notice notice-good">Dataset pushed. This triggers the real CI/CD pipeline.</div>
        <p><a href="{GITHUB_ACTIONS_URL}" target="_blank">Watch it on GitHub Actions</a></p>
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
        f"<h1>Model promoted</h1><div class='notice notice-good'>F1 = {f1:.4f}. Promoted to Production &mdash; live within a few minutes as predict.py's cache refreshes.</div><p><a href='/'>Dashboard</a></p>",
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
            <form action="/admin/users/{row['id']}/role" method="post" style="display:inline;">
              <input type="hidden" name="action" value="{role_action}">
              <button type="submit" class="link-btn">{role_action.capitalize()}</button>
            </form>
        """
        delete_button = f"""
            <form action="/admin/users/{row['id']}/delete" method="post" style="display:inline;"
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
    {notice}
    <div class="panel">
      <div class="panel-header"><h2>Create an account</h2></div>
      <p>New accounts are members by default &mdash; promote them here if they should also manage users.</p>
      <form class="stack" action="/admin/users/create" method="post">
        <label>Username <input name="username" type="text" required></label>
        <label>Password <input name="password" type="password" required></label>
        <label style="flex-direction:row; align-items:center; gap:0.5rem;">
          <input type="checkbox" name="is_admin" value="1" style="width:auto;"> Administrator
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
    return page("Users", body, account)


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
    <h1>Your account</h1>
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
    return page("Account", body, account)


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
