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
from html import escape

import cloudpickle
import mlflow
import pandas as pd
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from mlflow.tracking import MlflowClient
from passlib.context import CryptContext
from sqlalchemy import create_engine, text
from starlette.middleware.sessions import SessionMiddleware

from prediction_model.config import config
from prediction_model.processing.data_handling import load_full_dataset, day_block_split
from prediction_model.training_pipeline import evaluate_pipeline, register_and_promote
from dataset_uploader.logic import (
    now_gmt1,
    format_timestamp,
    source_label,
    stage_css_class,
    missing_dataset_columns,
)

REPO_DIR = "/repo"
DATASET_PATH = os.path.join(REPO_DIR, "prediction_model", "datasets", "dataset.csv")
DATASET_META_PATH = os.path.join(REPO_DIR, "prediction_model", "datasets", "dataset.meta.json")
GITHUB_ACTIONS_URL = "https://github.com/chmichaaa/MLOPS-chain/actions"

# config.TRACKING_URI (http://mlflow-server:5000) is the INTERNAL docker
# hostname -- correct for server-to-server calls, but meaningless to a
# browser outside the docker network. Links/iframes rendered in the user's
# browser need the box's actual public address instead.
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "localhost")
MLFLOW_PUBLIC_URL = f"http://{PUBLIC_HOST}:5000"
GRAFANA_DASHBOARD_URL = f"http://{PUBLIC_HOST}:3000/d/mlops-app?orgId=1&kiosk&refresh=30s"

SIGNUP_CODE = os.environ["SIGNUP_CODE"]
GIT_TOKEN = os.environ["GIT_TOKEN"]

db_engine = create_engine(os.environ["DATABASE_URL"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(title="MLOps Console")
app.add_middleware(SessionMiddleware, secret_key=os.environ["SESSION_SECRET_KEY"])

with db_engine.begin() as conn:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
    )


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


def page(title, body, user=None, auth=False):
    nav = ""
    if user:
        nav = f"""
        <header class="topbar">
          <div class="brand">MLOps<span class="dot">::</span>Console</div>
          <nav>
            <a href="/">Dashboard</a>
            <a href="/monitoring">Monitoring</a>
            <a href="/upload">Upload</a>
            <a href="/history">History</a>
          </nav>
          <div class="user-chip">
            <span>{escape(user)}</span>
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
  <style>{PAGE_CSS}</style>
</head>
<body>
  {nav}
  <div class="{wrapper_class}">
    {body}
  </div>
</body>
</html>
"""


def current_user(request: Request):
    return request.session.get("username")


def require_login(request: Request):
    user = current_user(request)
    if not user:
        return None, RedirectResponse("/login", status_code=303)
    return user, None


def stage_pill(stage):
    return f'<span class="pill {stage_css_class(stage)}">{escape(stage)}</span>'


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

@app.get("/signup", response_class=HTMLResponse)
def signup_form():
    return page("Sign up", """
        <h1>Create account</h1>
        <form class="stack" action="/signup" method="post">
          <label>Username <input name="username" type="text" required></label>
          <label>Password <input name="password" type="password" required></label>
          <label>Signup code <input name="signup_code" type="text" required></label>
          <button type="submit">Create account</button>
        </form>
        <p><a href="/login">Already have an account? Log in</a></p>
    """, auth=True)


@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...), signup_code: str = Form(...)):
    if signup_code != SIGNUP_CODE:
        return HTMLResponse(
            page("Sign up", "<h1>Create account</h1><div class='notice notice-bad'>Wrong signup code.</div><p><a href='/signup'>Try again</a></p>", auth=True),
            status_code=403,
        )

    password_hash = pwd_context.hash(password)
    try:
        with db_engine.begin() as conn:
            conn.execute(
                text("INSERT INTO users (username, password_hash) VALUES (:u, :p)"),
                {"u": username, "p": password_hash},
            )
    except Exception:
        return HTMLResponse(
            page("Sign up", "<h1>Create account</h1><div class='notice notice-bad'>That username is already taken.</div><p><a href='/signup'>Try again</a></p>", auth=True),
            status_code=400,
        )

    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form():
    return page("Log in", """
        <h1>Log in</h1>
        <form class="stack" action="/login" method="post">
          <label>Username <input name="username" type="text" required></label>
          <label>Password <input name="password" type="password" required></label>
          <button type="submit">Log in</button>
        </form>
        <p><a href="/signup">Need an account? Sign up</a></p>
    """, auth=True)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with db_engine.begin() as conn:
        row = conn.execute(
            text("SELECT password_hash FROM users WHERE username = :u"), {"u": username}
        ).fetchone()

    if row is None or not pwd_context.verify(password, row[0]):
        return HTMLResponse(
            page("Log in", "<h1>Log in</h1><div class='notice notice-bad'>Wrong username or password.</div><p><a href='/login'>Try again</a></p>", auth=True),
            status_code=401,
        )

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


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    user, redirect = require_login(request)
    if redirect:
        return redirect

    versions = sorted(_registry_versions(), key=lambda v: int(v.version), reverse=True)
    production = next((v for v in versions if v.current_stage == "Production"), None)

    if production is None:
        prod_html = """
        <div class="panel">
          <div class="panel-header"><h2>Production model</h2></div>
          <p style="margin:0;">No model has been promoted to Production yet. Upload a dataset or a trained model to get started.</p>
        </div>
        """
    else:
        run = mlflow.get_run(production.run_id)
        f1 = run.data.metrics.get("f1_score", float("nan"))
        tags = production.tags or {}
        prod_html = f"""
        <div class="panel">
          <div class="panel-header">
            <h2>Production model &middot; v{production.version}</h2>
            {stage_pill('Production')}
          </div>
          <dl class="readout">
            <div><dt>F1 score</dt><dd class="mono">{f1:.4f} <span class="text-muted">/ {config.F1_THRESHOLD}</span></dd></div>
            <div><dt>Source</dt><dd>{escape(source_label(tags.get('source', 'unknown')))}</dd></div>
            <div><dt>Dataset by</dt><dd>{escape(tags.get('dataset_uploaded_by', 'unknown'))}</dd></div>
            <div><dt>Dataset at</dt><dd class="mono">{escape(format_timestamp(tags.get('dataset_uploaded_at', 'unknown')))}</dd></div>
          </dl>
        </div>
        """

    body = f"""
    <h1>Dashboard</h1>
    {prod_html}
    <p><a href="{MLFLOW_PUBLIC_URL}" target="_blank">Open MLflow</a> &middot; <a href="{GITHUB_ACTIONS_URL}" target="_blank">Open GitHub Actions</a></p>
    """
    return page("Dashboard", body, user)


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring(request: Request):
    user, redirect = require_login(request)
    if redirect:
        return redirect

    # Embeds the real Grafana dashboard (grafana/provisioning/dashboards/
    # mlops-app.json) rather than reimplementing charts here -- Grafana's own
    # anonymous-viewer access is enabled (docker-compose.yml's grafana
    # service) specifically so this iframe doesn't prompt for a second login;
    # editing/admin still requires the real Grafana login, anonymous access
    # is view-only. &kiosk hides Grafana's own nav chrome for a cleaner embed.
    body = f"""
    <h1>Monitoring</h1>
    <iframe class="embed-frame" src="{GRAFANA_DASHBOARD_URL}" title="Grafana dashboard"></iframe>
    <p><a href="http://{PUBLIC_HOST}:3000" target="_blank">Open Grafana directly</a></p>
    """
    return page("Monitoring", body, user)


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    user, redirect = require_login(request)
    if redirect:
        return redirect

    versions = sorted(_registry_versions(), key=lambda v: int(v.version), reverse=True)
    rows = ""
    for v in versions:
        run = mlflow.get_run(v.run_id)
        f1 = run.data.metrics.get("f1_score", float("nan"))
        tags = v.tags or {}
        rows += f"""
        <tr>
          <td class="mono">v{v.version}</td>
          <td>{stage_pill(v.current_stage)}</td>
          <td class="mono">{f1:.4f}</td>
          <td>{escape(source_label(tags.get('source', 'unknown')))}</td>
          <td>{escape(tags.get('dataset_uploaded_by', 'unknown'))}</td>
          <td class="mono">{escape(format_timestamp(tags.get('dataset_uploaded_at', 'unknown')))}</td>
        </tr>
        """

    body = f"""
    <h1>Model history</h1>
    <div class="table-wrap">
      <table>
        <tr><th>Version</th><th>Stage</th><th>F1</th><th>Source</th><th>Uploaded by</th><th>Uploaded at</th></tr>
        {rows or '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;">No models registered yet.</td></tr>'}
      </table>
    </div>
    """
    return page("History", body, user)


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------

@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request, message: str = "", ok: str = ""):
    user, redirect = require_login(request)
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
    return page("Upload", body, user)


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
    user, redirect = require_login(request)
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
            json.dump({"uploaded_by": user, "uploaded_at": uploaded_at}, f)

        # The committed .dvc/config points at localhost:9000 (a local-dev
        # default -- see README.md's DVC section); from inside this
        # container MinIO is reachable at config.MINIO_ENDPOINT_URL
        # (http://minio:9000) instead, so override the remote the same way
        # CI does before every dvc pull/push.
        _run(["dvc", "remote", "modify", "--local", "myremote", "endpointurl", config.MINIO_ENDPOINT_URL])
        _run(["dvc", "add", DATASET_PATH])
        _run(["dvc", "push"])
        _run(["git", "add", "prediction_model/datasets/dataset.csv.dvc", "prediction_model/datasets/dataset.meta.json"])
        _push_to_github(f"Update dataset (uploaded by {user})", user)
    except Exception as exc:
        return HTMLResponse(page(
            "Upload failed",
            f"<h1>Upload failed</h1><div class='notice notice-bad'><pre>{escape(str(exc))}</pre></div><p><a href='/upload'>Back</a></p>",
            user,
        ))

    return HTMLResponse(page(
        "Upload succeeded",
        f"""
        <h1>Upload succeeded</h1>
        <div class="notice notice-good">Dataset pushed. This triggers the real CI/CD pipeline.</div>
        <p><a href="{GITHUB_ACTIONS_URL}" target="_blank">Watch it on GitHub Actions</a></p>
        """,
        user,
    ))


@app.post("/upload/model", response_class=HTMLResponse)
async def upload_model(request: Request, file: UploadFile = File(...)):
    user, redirect = require_login(request)
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
            user,
        ))

    try:
        dataset = load_full_dataset()
        _, eval_df = day_block_split(dataset)
        metrics = evaluate_pipeline(pipeline, eval_df)
    except Exception as exc:
        return HTMLResponse(page(
            "Upload failed",
            f"<h1>Upload failed</h1><div class='notice notice-bad'>Could not evaluate model: {escape(str(exc))}</div><p><a href='/upload'>Back</a></p>",
            user,
        ))

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.set_tags({"source": "model_upload", "uploaded_by": user})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, config.MODEL_NAME.lstrip("/"), serialization_format="cloudpickle")
        run_id = run.info.run_id

    f1 = metrics["f1_score"]
    if f1 < config.F1_THRESHOLD:
        return HTMLResponse(page(
            "Model rejected",
            f"<h1>Model rejected</h1><div class='notice notice-bad'>F1 = {f1:.4f}, below the required {config.F1_THRESHOLD}. Not promoted.</div><p><a href='/upload'>Back</a></p>",
            user,
        ))

    register_and_promote(
        run_id, source="model_upload",
        dataset_meta={"uploaded_by": user, "uploaded_at": now_gmt1().isoformat()},
    )
    return HTMLResponse(page(
        "Model promoted",
        f"<h1>Model promoted</h1><div class='notice notice-good'>F1 = {f1:.4f}. Promoted to Production &mdash; live within a few minutes as predict.py's cache refreshes.</div><p><a href='/'>Dashboard</a></p>",
        user,
    ))
