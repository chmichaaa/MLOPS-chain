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

REPO_DIR = "/repo"
DATASET_PATH = os.path.join(REPO_DIR, "prediction_model", "datasets", "dataset.csv")
DATASET_META_PATH = os.path.join(REPO_DIR, "prediction_model", "datasets", "dataset.meta.json")
REQUIRED_COLUMNS = {"timestamp", "label", "is_synthetic_anomaly", *config.METRIC_COLUMNS}

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
# tiny HTML helpers -- no template engine needed for pages this simple
# --------------------------------------------------------------------------

def page(title, body, user=None):
    nav = ""
    if user:
        nav = f"""
        <nav>
          <a href="/">Dashboard</a>
          <a href="/upload">Upload</a>
          <a href="/history">History</a>
          <span class="spacer"></span>
          <span>{user}</span>
          <form action="/logout" method="post" style="display:inline">
            <button type="submit">Log out</button>
          </form>
        </nav>
        """
    return f"""
    <!doctype html>
    <html>
    <head>
      <title>{title}</title>
      <style>
        body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
        nav {{ display: flex; gap: 1rem; align-items: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #ddd; }}
        nav a {{ text-decoration: none; color: #2563eb; }}
        .spacer {{ flex: 1; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0; }}
        .ok {{ color: #16a34a; }}
        .fail {{ color: #dc2626; }}
        input, button {{ padding: 0.4rem; margin: 0.25rem 0; }}
        form.block {{ display: flex; flex-direction: column; max-width: 400px; }}
      </style>
    </head>
    <body>
      {nav}
      {body}
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


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

@app.get("/signup", response_class=HTMLResponse)
def signup_form():
    return page("Sign up", """
        <h1>Create account</h1>
        <form class="block" action="/signup" method="post">
          <label>Username <input name="username" required></label>
          <label>Password <input name="password" type="password" required></label>
          <label>Signup code <input name="signup_code" required></label>
          <button type="submit">Sign up</button>
        </form>
        <p><a href="/login">Already have an account? Log in</a></p>
    """)


@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...), signup_code: str = Form(...)):
    if signup_code != SIGNUP_CODE:
        return HTMLResponse(page("Sign up", "<p>Wrong signup code.</p><p><a href='/signup'>Try again</a></p>"), status_code=403)

    password_hash = pwd_context.hash(password)
    try:
        with db_engine.begin() as conn:
            conn.execute(
                text("INSERT INTO users (username, password_hash) VALUES (:u, :p)"),
                {"u": username, "p": password_hash},
            )
    except Exception:
        return HTMLResponse(page("Sign up", "<p>That username is already taken.</p><p><a href='/signup'>Try again</a></p>"), status_code=400)

    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form():
    return page("Log in", """
        <h1>Log in</h1>
        <form class="block" action="/login" method="post">
          <label>Username <input name="username" required></label>
          <label>Password <input name="password" type="password" required></label>
          <button type="submit">Log in</button>
        </form>
        <p><a href="/signup">Need an account? Sign up</a></p>
    """)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with db_engine.begin() as conn:
        row = conn.execute(
            text("SELECT password_hash FROM users WHERE username = :u"), {"u": username}
        ).fetchone()

    if row is None or not pwd_context.verify(password, row[0]):
        return HTMLResponse(page("Log in", "<p>Wrong username or password.</p><p><a href='/login'>Try again</a></p>"), status_code=401)

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
        prod_html = "<p>No model has been promoted to Production yet.</p>"
    else:
        run = mlflow.get_run(production.run_id)
        f1 = run.data.metrics.get("f1_score", float("nan"))
        tags = production.tags or {}
        prod_html = f"""
        <div class="card">
          <h2>Production model: v{production.version}</h2>
          <p>F1 score: <b>{f1:.4f}</b> (threshold: {config.F1_THRESHOLD})</p>
          <p>Source: {tags.get('source', 'unknown')}</p>
          <p>Dataset uploaded by: {tags.get('dataset_uploaded_by', 'unknown')} at {tags.get('dataset_uploaded_at', 'unknown')}</p>
          <p>Promoted: {production.last_updated_timestamp}</p>
        </div>
        """

    body = f"""
    <h1>MLOps Console</h1>
    {prod_html}
    <p>
      <a href="{config.TRACKING_URI}" target="_blank">Open MLflow</a> &middot;
      <a href="https://github.com/chmichaaa/MLOPS-chain/actions" target="_blank">Open GitHub Actions</a>
    </p>
    """
    return page("Dashboard", body, user)


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
          <td>v{v.version}</td>
          <td>{v.current_stage}</td>
          <td>{f1:.4f}</td>
          <td>{tags.get('source', 'unknown')}</td>
          <td>{tags.get('dataset_uploaded_by', 'unknown')}</td>
          <td>{tags.get('dataset_uploaded_at', 'unknown')}</td>
        </tr>
        """

    body = f"""
    <h1>Model history</h1>
    <table>
      <tr><th>Version</th><th>Stage</th><th>F1</th><th>Source</th><th>Uploaded by</th><th>Uploaded at</th></tr>
      {rows or '<tr><td colspan="6">No models registered yet.</td></tr>'}
    </table>
    """
    return page("History", body, user)


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------

@app.get("/upload", response_class=HTMLResponse)
def upload_form(request: Request, message: str = ""):
    user, redirect = require_login(request)
    if redirect:
        return redirect

    body = f"""
    <h1>Upload</h1>
    {f'<div class="card">{message}</div>' if message else ''}

    <div class="card">
      <h2>Upload a dataset</h2>
      <p>Pushed through git/DVC and trained by the real CI/CD pipeline -- takes real time.</p>
      <form class="block" action="/upload/dataset" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <button type="submit">Upload dataset</button>
      </form>
    </div>

    <div class="card">
      <h2>Upload a trained model</h2>
      <p>Evaluated immediately against the current eval split and promoted straight to
      Production if it clears the gate -- no CI run, live within a few minutes via
      predict.py's cache refresh.</p>
      <form class="block" action="/upload/model" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Upload model</button>
      </form>
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

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return RedirectResponse(f"/upload?message=Missing required columns: {sorted(missing)}", status_code=303)

    try:
        _run(["git", "pull"])

        with open(DATASET_PATH, "wb") as f:
            f.write(contents)

        uploaded_at = datetime.now(timezone.utc).isoformat()
        with open(DATASET_META_PATH, "w") as f:
            json.dump({"uploaded_by": user, "uploaded_at": uploaded_at}, f)

        _run(["dvc", "add", DATASET_PATH])
        _run(["dvc", "push"])
        _run(["git", "add", "prediction_model/datasets/dataset.csv.dvc", "prediction_model/datasets/dataset.meta.json"])
        _push_to_github(f"Update dataset (uploaded by {user})", user)
    except Exception as exc:
        return HTMLResponse(page("Upload failed", f"<div class='card fail'><pre>{exc}</pre></div><p><a href='/upload'>Back</a></p>", user))

    return HTMLResponse(page(
        "Upload succeeded",
        """
        <div class="card ok">
          <p>Dataset pushed. This triggers the real CI/CD pipeline.</p>
          <p><a href="https://github.com/chmichaaa/MLOPS-chain/actions" target="_blank">Watch it on GitHub Actions</a></p>
        </div>
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
        return HTMLResponse(page("Upload failed", f"<div class='card fail'><p>Could not load model: {exc}</p></div><p><a href='/upload'>Back</a></p>", user))

    try:
        dataset = load_full_dataset()
        _, eval_df = day_block_split(dataset)
        metrics = evaluate_pipeline(pipeline, eval_df)
    except Exception as exc:
        return HTMLResponse(page("Upload failed", f"<div class='card fail'><p>Could not evaluate model: {exc}</p></div><p><a href='/upload'>Back</a></p>", user))

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
            f"<div class='card fail'><p>F1 = {f1:.4f}, below the required {config.F1_THRESHOLD}. Not promoted.</p></div><p><a href='/upload'>Back</a></p>",
            user,
        ))

    register_and_promote(
        run_id, source="model_upload",
        dataset_meta={"uploaded_by": user, "uploaded_at": datetime.now(timezone.utc).isoformat()},
    )
    return HTMLResponse(page(
        "Model promoted",
        f"<div class='card ok'><p>F1 = {f1:.4f}. Promoted to Production -- live within a few minutes as predict.py's cache refreshes.</p></div><p><a href='/'>Dashboard</a></p>",
        user,
    ))
