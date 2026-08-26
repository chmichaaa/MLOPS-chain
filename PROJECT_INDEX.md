# Project Index — Infrastructure Anomaly Detection MLOps Pipeline

This document indexes and explains every part of this repository: what it does, how the
pieces connect, and why key design decisions were made. It's meant as a map you can use
instead of re-reading every file from scratch.

## 1. What this project is

An end-to-end MLOps pipeline that detects **infrastructure anomalies** (unusual CPU,
network, load-balancer, and RDS behavior on AWS) using unsupervised anomaly-detection
models (Isolation Forest / One-Class SVM). It is a pivot of an earlier "loan prediction"
demo project — a couple of comments in the code still reference that history (e.g.
`config.py`'s note on the `S3_BUCKET` rename), but no loan-prediction code remains.

It demonstrates full MLOps maturity: data versioning, experiment tracking, CI, automated
quality gates, containerized deployment to a single EC2 instance, and continuous
monitoring/retraining — not just a trained model. The actual model quality is secondary to
the point of the project: proving the promotion gate itself works (train → evaluate → assert
F1 ≥ threshold → deploy only if it clears the bar, otherwise nothing ships) — see §3. Which
data you feed the pipeline (harder real incidents vs. easy synthetic data) is what decides
whether a given run clears the bar, not any special-cased pipeline logic.

**The 4 monitored metrics** (`config.METRIC_COLUMNS`): `cpu_usage_pct`,
`network_in_bytes`, `elb_request_count`, `rds_cpu_usage_pct`.

## 2. Directory map

```
MLOps-Project/
├── main.py                      FastAPI app (serving layer)
├── locustfile.py                Load test (p95 latency SLA)
├── Dockerfile                   App image (runtime)
├── Dockerfile.mlflow            MLflow tracking-server image
├── docker-compose.yml           Full stack (local dev AND single-EC2 prod): Postgres, MinIO, MLflow, Airflow, app
├── requirements.txt             Python dependencies (pinned)
├── .env.example                 Template for docker-compose secrets
│
├── prediction_model/            Core ML package
│   ├── config/config.py         ALL tunables — read this first
│   ├── training_pipeline.py     Hyperopt search + MLflow logging + the promotion gate
│   ├── predict.py               Loads best MLflow model, serves predictions (used by main.py)
│   ├── VERSION                  Package version string
│   ├── processing/
│   │   ├── build_dataset.py           Writes dataset.csv from real NAB data + synthetic injection (harder)
│   │   ├── build_synthetic_dataset.py Writes dataset.csv from fully synthetic data (easier)
│   │   ├── data_handling.py           Dataset loader + day_block_split (leak-free train/eval split)
│   │   ├── preprocessing.py           RollingWindowFeatures sklearn transformer
│   │   └── evaluation.py              point_adjust / windowed_f1 (transparency metrics, unit-tested)
│   └── datasets/
│       ├── dataset.csv                 Whichever generator wrote it most recently (DVC-tracked)
│       └── dataset.csv.dvc             DVC pointer file (pinned MD5 hash)
│
├── drift_monitoring/
│   └── check_drift.py           Automated Evidently drift check, called by Airflow (see §6)
│
├── airflow/
│   ├── Dockerfile               Airflow image + this project's requirements.txt
│   └── dags/
│       ├── dag_ingestion.py         Daily: rebuild dataset.csv, then trigger training
│       ├── dag_training.py          Triggered only: run training_pipeline.py as a subprocess
│       └── dag_drift_retrain.py     Every 5 min: check drift, trigger retraining if detected
│
├── docker/
│   └── postgres-init.sh          Creates `mlflow` + `airflow` + `dataset_uploader` databases on first Postgres start
│
├── dataset_uploader/             Internal admin console (login, dashboard, dataset/model upload) -- see §7.1
│   ├── app.py                    FastAPI app
│   └── Dockerfile                Separate image from the served app (needs git + a GitHub push token)
│
├── tests/
│   ├── test_evaluation.py        Unit: point_adjust, windowed_f1
│   ├── test_preprocessing.py     Unit: RollingWindowFeatures
│   ├── test_prediction.py        Integration: predict.py against a live MLflow model
│   └── test_model_quality.py     Integration: best run clears F1_THRESHOLD
│
├── .github/workflows/main.yml    CI/CD: test → validate → build → deploy
├── .dvc/config                   DVC remote: MinIO bucket `s3://infra-monitoring`
└── README.md                     Original project write-up (architecture diagrams, DVC/dataset explainer)
```

## 3. The core idea: one pipeline, one threshold, you choose the data

This is the single most important design decision in the repo, and it's worth
understanding before anything else makes sense.

`training_pipeline.py` always trains against whatever
`prediction_model/datasets/dataset.csv` currently contains, evaluates it on a held-out
split, and asserts the best model's pointwise F1 clears `config.F1_THRESHOLD` (0.75)
before anything is eligible for deployment (§8's `validate` job). There is exactly one
dataset file, one threshold, one MLflow experiment (`config.EXPERIMENT_NAME`) — no
branching logic based on which "kind" of data it is.

What controls whether a given run passes or fails is entirely **which data you feed
it**, via two interchangeable generators that both write to the same `dataset.csv`:

| | `processing/build_dataset.py` (harder) | `processing/build_synthetic_dataset.py` (easier) |
|---|---|---|
| Data | 4 real AWS CloudWatch series (NAB `realAWSCloudwatch`) from a real April-2014 incident, plus disclosed synthetic anomaly + background layers | Fully synthetic Gaussian noise + a handful of sustained multi-sample shift events |
| Anomaly shape | Gradual multivariate regime shifts (hours-long) | Sustained shifts, clearly separable by construction |
| Achieved F1 (pointwise) | ~0.59 | ~0.90+ |
| Gate outcome at 0.75 | Blocked | Promoted |

**Why real data doesn't clear the bar**: Isolation Forest / One-Class SVM are built to
isolate point/segment outliers that stand out from the baseline. The real NAB incidents
are gradual, multi-hour regime shifts — subtler than that — so this model class
structurally tops out around F1 ≈ 0.55–0.60 on them (verified via ROC-AUC ≈ 0.57 on a
leak-free split). That's a genuine model/data-fit limitation, not a pipeline bug — and
it's exactly the "blocked" case worth being able to demonstrate on demand, by running
`build_dataset.py`.

**Why the easy generator exists**: to demonstrate the other direction — that the same
gate code (`training_pipeline.train_and_select`) genuinely promotes a model when the
data supports it, not just always blocks. Run `build_synthetic_dataset.py` any time you
want a guaranteed-comfortable pass to demo. Both generators are just utilities; neither
is "the real pipeline" — `training_pipeline.py` doesn't know or care which one produced
its input.

## 4. Data pipeline

### 4.1 Building `dataset.csv` (`processing/build_dataset.py`)
1. Downloads 4 real CloudWatch series from NAB's GitHub (`realAWSCloudwatch`), caches them
   locally under `prediction_model/.nab_cache/`.
2. Merges them on timestamp (`cpu`/`network` share a 5-min grid exactly; `elb`/`rds` are
   joined with `merge_asof` tolerance of 30/15 min respectively).
3. Labels each row `1` if it falls in any of the 4 files' NAB-provided anomaly windows
   (union across all 4 = 4 real incidents, 1195/4032 rows). 9 of 15 real calendar days
   already contain an incident — only 6 are clean.
4. **Injects synthetic anomalies** (`inject_synthetic_anomalies`): sustained ±N-std-dev
   shifts across all 4 metrics simultaneously, placed only inside days that *already*
   have a real incident (so no clean day gets pulled from training). Flagged via
   `is_synthetic_anomaly`.
5. **Extends with synthetic background** (`extend_with_synthetic_background`):
   block-bootstraps the real clean days (resampling contiguous ~4h chunks with
   replacement, not fabricating new dynamics) to add more normal days. Necessary because
   15 real days alone contain more incident time than clean time — without this, eval
   would be ~41% anomalous, unrealistic for a monitoring dataset. Flagged via
   `is_synthetic_background`.
6. Every row's provenance survives as boolean flag columns, so `training_pipeline.py` can
   report real-vs-synthetic detection performance separately instead of hiding it in one
   aggregate number.

Run with: `python -m prediction_model.processing.build_dataset`

### 4.2 Building the easy alternative (`processing/build_synthetic_dataset.py`)
Purely synthetic, 5000 rows (~17 days), i.i.d. stationary Gaussian noise per metric (no
diurnal pattern — deliberately trivial to learn), with `EASY_DATA_N_EVENTS` sustained
shift events (one per day, `EASY_DATA_EVENT_LEN` samples each) injected on top — same
injection style as `build_dataset.py`'s `inject_synthetic_anomalies`, so this data works
with the same `day_block_split` and `WINDOW_SIZE` rather than needing special-cased
handling. **Writes to the same `config.DATA_FILE`** as `build_dataset.py` — running this
overwrites whatever `dataset.csv` currently holds.

Run with: `python -m prediction_model.processing.build_synthetic_dataset`

### 4.3 DVC — freezing a specific dataset snapshot
`build_dataset.py` is the source-of-truth *generator* (deterministic, fixed seeds).
`dataset.csv.dvc` is a **frozen, traceable pointer** (MD5 hash) to one specific output of
that generator, so:
- `training_pipeline.py` always trains against a known, reproducible file without
  re-hitting NAB's GitHub URLs or silently drifting if NAB's upstream files change.
- Anyone can restore that exact version via `dvc pull` / `dvc checkout`.

Remote: `s3://infra-monitoring` on MinIO (`.dvc/config`), the same bucket used for MLflow
artifacts and batch-prediction uploads. The committed `endpointurl` (`localhost:9000`) is
a local-dev default — override per-machine with `dvc remote modify --local myremote
endpointurl http://<host>:9000` (never commit that override; env vars aren't expanded
inside the tracked `.dvc/config`).

To publish a new frozen snapshot after changing the generator or its config:
```
python -m prediction_model.processing.build_dataset
dvc add prediction_model/datasets/dataset.csv
dvc push
git add prediction_model/datasets/dataset.csv.dvc
git commit -m "Update dataset snapshot"
```

**Resolved by the single-EC2 deployment**: CI builds the Docker image on a GitHub Actions
runner, not on the EC2 target itself, so `dvc pull` during that build needs MinIO reachable
from GitHub's infrastructure. Since MinIO and MLflow now run on the same EC2 instance that
serves the app (see §9), that instance's public IP is what `MINIO_ENDPOINT_URL`/
`MLFLOW_TRACKING_URI` point at — no VPC-local runner needed. Tradeoff: ports 5000/9000 are
internet-facing (MinIO still needs its access key/secret; MLflow's tracking server has no
built-in auth) — acceptable for this demo, not something to point at real production data.

### 4.4 Train/eval split — `data_handling.day_block_split`
Splits by **whole calendar day**, never by row: any day containing an anomaly (real or
synthetic) goes entirely to eval (the model is never fit on a labelled anomaly).
Remaining clean days split `EVAL_CLEAN_FRACTION` (0.55) to eval / rest to train. Keeps
each day temporally contiguous, which the rolling-window feature and windowed evaluation
both depend on.

## 5. Model training and the promotion gate

### 5.1 Feature engineering (`processing/preprocessing.py`)
`RollingWindowFeatures`: expands each of the 4 raw metrics into `[raw, rolling_mean]`
(8 columns total), rolling mean over `WINDOW_SIZE` samples (12 = ~1 hour, since data is
~5-min sampled). Kept **signed** rather than z-scored/absolute deliberately — several
metrics carry directional anomaly signal (e.g. a *drop* in `rds_cpu_usage_pct` correlates
with real incidents; verified via per-metric ROC-AUC up to 0.78).

Then `MinMaxScaler`, then the anomaly model (`IsolationForest` or `OneClassSVM`), built
inline by `training_pipeline.build_pipeline()`.

### 5.2 Hyperparameter search + gate (`training_pipeline.py`, shared `train_and_select`)
- Runs Hyperopt TPE search: 25 evals over Isolation Forest, then (if
  `COMPARE_OCSVM=True`) 15 evals over One-Class SVM, using **one seeded RNG**
  (`HYPEROPT_SEED`) threaded through both searches in sequence — needed so the winner is
  reproducible run-to-run (otherwise search-order randomness alone could flip which
  model type wins).
- Every trial is logged as a nested MLflow run: params, `f1_score` (pointwise, the
  selection metric), `f1_score_unpenalized`, `f1_score_point_adjusted`, `f1_score_windowed`,
  `precision`, `recall`, `accuracy`, `recall_real_incidents`, `recall_synthetic_incidents`,
  `false_positive_rate_normal`, `meets_precision_floor`, plus the fitted pipeline itself
  (`mlflow.sklearn.log_model`).
- **Precision floor** (`config.PRECISION_FLOOR = 0.4`): trials with precision below this
  are zeroed out for selection purposes (their real F1 is still logged) — otherwise an
  aggressive "flag almost everything" model can post a deceptively high F1 on an
  anomaly-heavy eval set while being useless in practice (was hitting an 80%
  false-positive rate on normal data before this was added).
- After both searches, the function pulls the single best run across the whole
  experiment by `metrics.f1_score` and **asserts** it clears `f1_threshold`. This assert
  is the actual promotion gate — it's what makes `python -m prediction_model.training_pipeline`
  exit non-zero (failing the CI job / Airflow task) if the bar isn't met.
- `train_and_select(X_train, eval_df, experiment_name, f1_threshold)` is intentionally
  small — it doesn't know or care which generator produced `dataset.csv` (§3); it just
  trains, evaluates, and gates on whatever it's handed.
- On success, it calls `register_and_promote(run_id, source='training', ...)` — this
  registers the winning model under `config.REGISTERED_MODEL_NAME` in MLflow's Model
  Registry and transitions it to the `Production` stage, archiving whatever was there
  before. This is the actual deployment record (see §7.1) and what `predict.py` serves.
- The per-trial metric computation is factored into `evaluate_pipeline(pipeline,
  eval_df)`, callable on any already-fitted pipeline — this is what lets
  `dataset_uploader`'s direct model-upload path (§7.1) use the *exact* same gate math
  instead of a second implementation of it.

### 5.3 Transparency metrics (`processing/evaluation.py`)
Two additional metrics are computed and logged but **never used for selection or
thresholding**, specifically because they can be gamed/inflated independently of real
detection quality:
- `point_adjust` / `f1_score_point_adjusted`: standard time-series adjustment (Xu et al.
  2018) — if any point in a ground-truth anomaly segment is flagged, the whole segment
  counts as detected. Known to inflate scores on long segments (Kim et al. 2021).
- `windowed_f1`: F1 over fixed-size, per-day, non-overlapping time buckets rather than
  per-point.

### 5.4 `config.py` — read this before changing anything
This is the single source of truth for every tunable: dataset paths, split fractions,
synthetic-injection parameters, search spaces' eval counts, `F1_THRESHOLD`, S3/MinIO
settings, MLflow tracking URI/experiment name, drift threshold, model cache TTL. Nearly
every constant has an inline comment explaining *why* its specific value was chosen —
worth reading directly rather than summarizing further here.

## 6. Serving, monitoring, and drift-triggered retraining

### 6.1 FastAPI app (`main.py`)
Endpoints:
- `GET /` — health/welcome message.
- `POST /prediction_api` — real-time prediction. Body: `{"readings": [...]}`, a
  chronological (oldest→newest) window of `MetricReading` objects; scores the most
  recent point using `predict.generate_predictions`.
- `POST /prediction_ui` — manual-testing form helper (paste comma-separated lines).
- `POST /batch_prediction` — upload a CSV, scores every row
  (`generate_predictions_batch`), appends `anomaly_score`/`is_anomaly` columns, uploads
  the result to S3/MinIO under `datadrift/<date>/<file>_<timestamp>.csv` (this is also
  what `check_drift.py` later reads as "the latest batch"), and streams the CSV back.
- `/metrics` — exposed automatically by `prometheus_fastapi_instrumentator`, scraped by
  Prometheus and visualized in Grafana.

### 6.2 Prediction serving (`predict.py`)
Loads whatever is currently staged `Production` for `config.REGISTERED_MODEL_NAME` in
MLflow's Model Registry (not "best F1 in the experiment" — see §5.2/§7.1 for what
actually promotes a model there), and caches it in a module-level dict for
`MODEL_CACHE_TTL_SECONDS` (600s) to avoid hitting MLflow (and re-downloading the model
artifact) on every request. Re-checks for a newer best run after the TTL expires.

### 6.3 Drift monitoring
`drift_monitoring/check_drift.py` reads the most recently uploaded batch-prediction CSV
from S3/MinIO, compares it against the training reference (via Evidently's
`DataDriftPreset`) computed from `day_block_split`'s train set, and flags drift if
`share_of_drifted_columns >= config.DRIFT_THRESHOLD` (0.5) or Evidently's own
`dataset_drift` flag fires. Logs `detection_latency_seconds` (time from upload to check)
to a `drift_monitoring` MLflow experiment — the concrete evidence for the project's
"<10 min detection delay" requirement (see §6.4).

### 6.4 Airflow orchestration (`airflow/dags/`)
Three DAGs, chained by triggers rather than independent schedules:
1. **`dag_ingestion.py`** (`ingestion_pipeline`, daily) — rebuilds `dataset.csv` from NAB
   sources, then triggers `training_pipeline` (no wait).
2. **`dag_training.py`** (`training_pipeline`, `schedule_interval=None`) — triggered only
   (by ingestion, by drift detection, or manually), runs
   `python -m prediction_model.training_pipeline` as a subprocess via `BashOperator`
   (not imported directly, because that module runs its Hyperopt search — and asserts
   the F1 gate — at import time; a subprocess's real exit code is what lets a failed
   assert fail the Airflow task, i.e., correctly refuse to promote a bad model).
3. **`dag_drift_retrain.py`** (`drift_monitoring_retrain`, every
   `DRIFT_CHECK_SCHEDULE_MINUTES`=5 min, `max_active_runs=1`) — runs `check_drift()`,
   branches to `trigger_retraining` (fires `training_pipeline`) or a no-op `EmptyOperator`
   depending on the result. A 5-minute schedule bounds worst-case detection latency to 10
   minutes by construction (an upload landing right after a check waits at most one more
   full interval); `check_drift.py`'s logged `detection_latency_seconds` is the actual
   measured number per run.

Airflow runs in **standalone mode** (single container — scheduler + webserver +
triggerer in one process), appropriate for this demo/student-scale deployment; a
production Airflow would split these into separate services. Project code is not baked
into the Airflow image — `docker-compose.yml` mounts the repo root at
`/opt/airflow/project` (on `PYTHONPATH`) so DAG/code changes don't need a rebuild.

## 7. Local development stack (`docker-compose.yml`)

Five services, meant to fully substitute for AWS-hosted equivalents during local/demo
use:

| Service | Role | URL |
|---|---|---|
| `postgres` | Backend store for MLflow, Airflow, and dataset-uploader (three logical DBs, created by `docker/postgres-init.sh`) | `localhost:5432` |
| `minio` | S3-compatible object store — substitutes for real AWS S3 (MLflow artifacts, DVC remote, batch-prediction/drift uploads) | API `localhost:9000`, console `localhost:9001` |
| `minio-init` | One-shot: creates the `infra-monitoring` bucket, then exits | — |
| `mlflow-server` | Tracking server (`Dockerfile.mlflow`), backed by Postgres + MinIO | `localhost:5000` |
| `airflow` | Standalone Airflow (`airflow/Dockerfile`) | `localhost:8080` (creds printed in container logs on first start) |
| `app` | The FastAPI serving app itself (§6.1) — `image:` resolves to `$ECR_IMAGE` if set, else a local build | `localhost:8005` |
| `dataset-uploader` | Internal admin console (§7.1) | `localhost:8090` |

Setup: `cp .env.example .env` (edit passwords), then `docker compose up -d --build`. This
same compose file — same file, unmodified — is what runs on the production EC2 instance
too (see §9): local dev and "prod" are the identical stack, just with `ECR_IMAGE` set (or
not) in `.env`.

### 7.1 The MLOps console (`dataset_uploader/`)

A small, separate FastAPI app — deliberately not part of `main.py` — that gives a
logged-in user two ways to get a new model into `Production`, plus visibility into
everything that's happened so far. Kept separate specifically because it needs a
GitHub push token and write access to the real git checkout, neither of which belong
in the `app` image that CI rebuilds and redeploys on every push.

**Auth**: full multi-user accounts (a `users` table in a new `dataset_uploader`
Postgres database) rather than a single shared token. `/signup` requires a
`SIGNUP_CODE` (env var) in addition to username/password, so a random visitor to the
box's public IP can't just create their own login. `/login` sets a signed session
cookie (`SESSION_SECRET_KEY`).

**Deployment record = MLflow's Model Registry, not a new database.** Both routes below
call `training_pipeline.register_and_promote()`, which registers the model under
`config.REGISTERED_MODEL_NAME` and transitions it to `Production` (archiving whatever
was there before), tagged with `source` (`training` or `model_upload`) and the dataset's
`uploaded_by`/`uploaded_at` (from `dataset.meta.json`, written by `build_dataset.py`,
`build_synthetic_dataset.py`, or the dataset-upload route below). The console's
dashboard (`/`) and history page (`/history`) just read this registry — there's no
separate deployment-log table to keep in sync.

- **`POST /upload/dataset`** — the full-pipeline path. Validates the CSV's columns,
  `git pull`s the bind-mounted repo, overwrites `dataset.csv` + `dataset.meta.json`
  (tagged with the logged-in username), `dvc add`/`dvc push`, commits (authored as that
  username) and pushes to `main` using a token passed inline in the push URL — never
  written to `.git/config`, since that file lives in the real, bind-mounted host repo.
  This is what actually triggers the real CI/CD chain (§8); nothing gets registered here
  directly, `training_pipeline.py`'s own gate does that once `validate` runs.
- **`POST /upload/model`** — the fast path. Deserializes an uploaded, already-trained
  pipeline (`cloudpickle`) and scores it against the current eval split using
  `training_pipeline.evaluate_pipeline()` — the same gate math, not a reimplementation.
  If it clears `config.F1_THRESHOLD`, it's logged as a fresh MLflow run and immediately
  promoted to `Production`; no CI run, no rebuild/redeploy — `predict.py` picks it up on
  its own the next time its cache TTL expires. **This path knowingly accepts a real
  security tradeoff**: loading an uploaded file means deserializing it, which can execute
  arbitrary code if the file is malicious. It's login-gated, not sandboxed — treat it as
  trusted-user-only, not something to expose to anyone you wouldn't hand shell access to.

## 8. CI/CD (`.github/workflows/main.yml`)

Five sequential/dependent jobs on push/PR to `main`:

1. **`unit_tests`** — fast, no external dependencies: `test_evaluation.py` +
   `test_preprocessing.py`.
2. **`validate`** (needs `unit_tests`) — `dvc pull`s whatever `dataset.csv` is currently
   frozen, runs `training_pipeline.py`. **Fails the job** if the best model's F1 <
   `config.F1_THRESHOLD` (0.75). This is dataset-agnostic — see §3 for how to control
   which outcome you get.
3. **`integration_tests`** (needs `validate`) — `test_prediction.py` +
   `test_model_quality.py` against the now-populated MLflow experiment.
4. **`build`** (needs `integration_tests`) — builds the Docker image, pushes to AWS ECR.
   Unreachable on any run where `validate` failed, since `integration_tests` (its
   dependency) never runs — that's the "not deployed" half of the gate demonstration,
   enforced purely by the job dependency graph, no extra logic needed.
5. **`deploy`** (needs `build`) — SSHes into a single, already-provisioned EC2 instance
   (`appleboy/ssh-action`) and runs `docker compose pull app && docker compose up -d app`
   against the same `docker-compose.yml` described in §7, refreshing just the `app`
   service to the image `build` just pushed.

Secrets used: `MLFLOW_TRACKING_URI`, `MINIO_ENDPOINT_URL`,
`MINIO_ACCESS_KEY_ID`/`MINIO_SECRET_ACCESS_KEY` (MinIO-specific, deliberately separate
from AWS IAM creds), `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`REPO_NAME` for ECR, and
`EC2_HOST`/`EC2_USER`/`EC2_SSH_KEY` for the `deploy` job's SSH connection.

## 9. Deployment (single EC2 instance via `docker compose`)

No Kubernetes involved — the whole stack (Postgres, MinIO, MLflow, Airflow, and the app;
see §7) runs as one `docker-compose.yml` on one already-provisioned Linux EC2 instance.
Deployment is just "make the `app` container run the latest gate-approved image":

- The `build` job (§8) pushes the app image to ECR, tagged `latest`, only once
  `validate` and `integration_tests` have both passed.
- The `deploy` job SSHes into the EC2 host and runs, in order: `git pull` (picks up any
  `docker-compose.yml`/config changes), `aws ecr get-login-password | docker login`
  (requires the EC2 instance to have its own ECR-pull-capable AWS credentials — an IAM
  instance profile is the recommended way, not credentials baked into the box), then
  `docker compose pull app && docker compose up -d app`.
- `app`'s `image:` in `docker-compose.yml` is `${ECR_IMAGE:-mlops-infra-anomaly:local}` —
  locally that variable is unset, so `docker compose up --build app` builds from the
  Dockerfile as before; on the EC2 host, the CI deploy step writes it into a
  `.env.deploy` file each run so `docker compose pull` fetches the exact tag `build` just
  pushed, rather than rebuilding on the box.
- Ports that must be open on the instance's security group: `8005` (the app, and for
  `locustfile.py` load tests), `5000`/`9000` (MLflow/MinIO, so CI's `dvc pull` and MLflow
  API calls — running on GitHub-hosted runners, not inside any VPC — can reach them; see
  §4.3's "Resolved by the single-EC2 deployment" note), and `22` (SSH, ideally restricted).
- **`locustfile.py`** — independent load test verifying the p95 < 200ms budget against
  whatever host you point it at: `locust -f locustfile.py --host http://<ec2-host>:8005
  --headless -u 20 -r 5 --run-time 1m`.

There is deliberately no canary/blue-green step here — a single container is replaced
in place. If a staged rollout becomes worth demonstrating later, that's a bigger, separate
change (either two containers behind a local reverse proxy, or revisiting Kubernetes).

## 10. Tests (`tests/`)

| File | Type | What it checks |
|---|---|---|
| `test_evaluation.py` | Unit | `point_adjust`, `windowed_f1` correctness — no external deps |
| `test_preprocessing.py` | Unit | `RollingWindowFeatures` transformer — no external deps |
| `test_prediction.py` | Integration | `predict.py` against a live MLflow model |
| `test_model_quality.py` | Integration | Best run clears `F1_THRESHOLD`, precision floor, and logs all transparency metrics (non-NaN) |

Unit tests need nothing but the repo; integration tests need a reachable MLflow instance
with the relevant experiments already populated (they `pytest.skip` gracefully if not).

## 11. Things to know if you're about to change something

- **To control whether the next run passes or fails the gate, run a different
  generator** (`build_dataset.py` for harder/real data, `build_synthetic_dataset.py` for
  easy data) before training — don't add per-dataset branches to `training_pipeline.py`
  or `config.py`. Keeping the pipeline dataset-agnostic is the whole point of §3's design.
- **`config.py` is the first place to look** for any behavior change — nearly every
  constant is documented in-place with the reasoning behind its exact value.
- **DVC's committed `.dvc/config` endpoint is a local-dev default** (`localhost:9000`) —
  any other environment needs a `--local` override, never committed.
- **CI, MLflow, and MinIO all need to reach the same EC2 host's public IP** — see §4.3 and
  §9 for the ports that must stay open on that instance's security group.
- **Port 8090 (`dataset-uploader`, §7.1) should NOT be opened to `0.0.0.0/0`** the way the
  demo ports are — it can push to your GitHub repo and deserialize uploaded files. Scope
  its security-group rule to your own IP.
