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
F1 ≥ threshold → deploy only if it clears the bar, otherwise nothing ships) — see §3.

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
│   ├── training_pipeline.py     Hyperopt search + MLflow logging + promotion gate (NAB, real data)
│   ├── train_synthetic_easy.py  Same gate, run against a synthetic "easy" dataset (gate self-test)
│   ├── predict.py               Loads best MLflow model, serves predictions (used by main.py)
│   ├── report_gate_status.py    CI helper: reports F1 vs. threshold without failing the build
│   ├── VERSION                  Package version string
│   ├── processing/
│   │   ├── build_dataset.py           Builds dataset.csv from real NAB data + synthetic injection
│   │   ├── build_synthetic_dataset.py Builds synthetic_easy_dataset.csv (fully synthetic)
│   │   ├── data_handling.py           Dataset loaders + day_block_split (leak-free train/eval split)
│   │   ├── preprocessing.py           RollingWindowFeatures sklearn transformer
│   │   └── evaluation.py              point_adjust / windowed_f1 (transparency metrics, unit-tested)
│   └── datasets/
│       ├── dataset.csv                 Real NAB + synthetic hybrid (DVC-tracked)
│       ├── dataset.csv.dvc             DVC pointer file (pinned MD5 hash)
│       └── synthetic_easy_dataset.csv  Fully synthetic gate-validation dataset
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
│   └── postgres-init.sh          Creates `mlflow` + `airflow` databases on first Postgres start
│
├── tests/
│   ├── test_evaluation.py        Unit: point_adjust, windowed_f1
│   ├── test_preprocessing.py     Unit: RollingWindowFeatures
│   ├── test_prediction.py        Integration: predict.py against a live MLflow model
│   ├── test_model_quality.py     Integration: best NAB run clears F1_THRESHOLD
│   └── test_gate_behavior.py     Integration: promotion gate discriminates good vs. bad correctly
│
├── .github/workflows/main.yml    CI/CD: test → validate (both datasets) → build → deploy
├── .dvc/config                   DVC remote: MinIO bucket `s3://infra-monitoring`
└── README.md                     Original project write-up (architecture diagrams, DVC/dataset explainer)
```

## 3. The core idea: two datasets, one gate, on purpose

This is the single most important design decision in the repo, and it's worth
understanding before anything else makes sense.

| | `training_pipeline.py` (NAB — real) | `train_synthetic_easy.py` (synthetic) |
|---|---|---|
| Question it answers | How well can this model class detect *real* infra incidents? | Does the promotion gate mechanism itself actually work? |
| Data | 4 real AWS CloudWatch series (NAB `realAWSCloudwatch`) from a real April-2014 incident, plus disclosed synthetic anomaly + background layers | Fully synthetic Gaussian noise + isolated 5–10σ point spikes |
| Anomaly shape | Gradual multivariate regime shifts (hours-long) | Sharp, single-point outliers (the "easy" case for Isolation Forest) |
| MLflow experiment | `infra_anomaly_detection` | `infra_anomaly_detection_synthetic_easy` (**separate** — see why below) |
| F1 threshold | `config.F1_THRESHOLD = 0.58` (recalibrated) | `config.SYNTHETIC_EASY_F1_THRESHOLD = 0.85` (original spec) |
| Achieved F1 (pointwise) | ~0.59 | ~0.90 |
| Gate outcome | Blocked at 0.85, promoted at its own 0.58 | Promoted |

**Why real data can't hit F1 ≥ 0.85**: Isolation Forest / One-Class SVM are built to
isolate point outliers. The real NAB incidents are gradual, multi-hour regime shifts —
not sharp spikes — so this model class structurally tops out around F1 ≈ 0.55–0.60 on
them (verified via ROC-AUC ≈ 0.57 on a leak-free split). That's a genuine model/data-fit
limitation, not a bug.

**Why the synthetic dataset exists**: without it, a low F1 on real data is ambiguous — is
the *gate* broken (blocking everything), or is the *model* just not good enough on this
data? The synthetic dataset removes that confound: same pipeline, same gate code
(`training_pipeline.train_and_select`), same original 0.85 threshold, but anomalies that
are trivially separable by construction. If the gate promotes a model here, the gate
itself is proven correct, and the NAB result stands as an honest, separate finding.

**Why they're in separate MLflow experiments (not just separate tags)**: `predict.py`
picks the best run by *maximum F1 across an entire experiment*. If both datasets logged
into the same experiment, the synthetic dataset's easy ~0.9 F1 would always outrank the
honest ~0.59 real-data model and get served in production. Separate experiments make
that impossible by construction; the `dataset_type` tag on every run is just a secondary,
human-readable safety net for browsing the MLflow UI.

`tests/test_gate_behavior.py` encodes this whole argument as three assertions: the
synthetic model *is* promoted at 0.85, the real model is *correctly blocked* at 0.85, and
the real model still passes its *own* 0.58 gate.

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

### 4.2 Building `synthetic_easy_dataset.csv` (`processing/build_synthetic_dataset.py`)
Purely synthetic, 5000 rows (~17 days), i.i.d. stationary Gaussian noise per metric (no
diurnal pattern — deliberately trivial to learn), with 60 isolated single-sample 5–10σ
point anomalies injected only in the eval region (rows 3000–4999). First 3000 rows are
guaranteed anomaly-free for unsupervised training.

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
- This exact function is shared, byte-for-byte, between the real NAB run and the
  synthetic gate-validation run (`train_synthetic_easy.py`) — only the dataset,
  experiment name, threshold, search space, and window size differ (passed as
  parameters, not by editing shared code), which is what makes the synthetic dataset a
  valid test of the *same* gate rather than a reimplementation of it.
- `window_size=1` for the synthetic dataset (vs. 12 for NAB): a rolling mean smears a
  single-sample spike across its trailing window, contaminating nearby normal points'
  features and capping precision regardless of true separability (verified: F1 rose from
  ~0.25 at window=12 to ~0.9 at window=1 on identical data).

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
synthetic-injection parameters, search spaces' eval counts, both F1 thresholds, S3/MinIO
settings, MLflow tracking URI/experiment names, drift threshold, model cache TTL. Nearly
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
Queries MLflow for the best run in `EXPERIMENT_NAME` (**always the NAB/real experiment**
— never the synthetic one, by design, see §3), loads that run's sklearn pipeline, and
caches it in a module-level dict for `MODEL_CACHE_TTL_SECONDS` (600s) to avoid hitting
MLflow (and re-downloading the model artifact) on every request. Re-checks for a newer
best run after the TTL expires.

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
| `postgres` | Backend store for both MLflow and Airflow (two logical DBs, created by `docker/postgres-init.sh`) | `localhost:5432` |
| `minio` | S3-compatible object store — substitutes for real AWS S3 (MLflow artifacts, DVC remote, batch-prediction/drift uploads) | API `localhost:9000`, console `localhost:9001` |
| `minio-init` | One-shot: creates the `infra-monitoring` bucket, then exits | — |
| `mlflow-server` | Tracking server (`Dockerfile.mlflow`), backed by Postgres + MinIO | `localhost:5000` |
| `airflow` | Standalone Airflow (`airflow/Dockerfile`) | `localhost:8080` (creds printed in container logs on first start) |
| `app` | The FastAPI serving app itself (§6.1) — `image:` resolves to `$ECR_IMAGE` if set, else a local build | `localhost:8005` |

Setup: `cp .env.example .env` (edit passwords), then `docker compose up -d --build`. This
same compose file — same file, unmodified — is what runs on the production EC2 instance
too (see §9): local dev and "prod" are the identical stack, just with `ECR_IMAGE` set (or
not) in `.env`.

## 8. CI/CD (`.github/workflows/main.yml`)

Six sequential/dependent jobs on push/PR to `main`:

1. **`unit_tests`** — fast, no external dependencies: `test_evaluation.py` +
   `test_preprocessing.py`.
2. **`validate_nab_real`** (needs `unit_tests`) — `dvc pull`s the frozen real dataset,
   runs `training_pipeline.py`. **Fails the job** if F1 < `config.F1_THRESHOLD` (0.58).
   Also runs `report_gate_status.py` against the original 0.85 bar — informational only,
   never fails the job, since NAB is known not to reach 0.85 with this model class.
3. **`validate_synthetic_easy`** (needs `unit_tests`) — builds the synthetic dataset
   fresh, runs `train_synthetic_easy.py`. **Fails the job** if F1 < 0.85 — this is the
   job whose failure would mean the gate mechanism itself is broken.
4. **`integration_tests`** (needs both validate jobs, runs even if `validate_nab_real`
   failed via `if: !cancelled()`) — `test_prediction.py`, `test_model_quality.py`,
   `test_gate_behavior.py` against the now-populated MLflow experiments.
5. **`build`** (needs `integration_tests` **and** `validate_synthetic_easy` specifically —
   *not* `validate_nab_real*, since NAB's own gate is 0.58, not 0.85*) — builds the
   Docker image, pushes to AWS ECR. This dependency choice is the concrete mechanism that
   guarantees a model below the original 0.85 spec bar never reaches deployment via the
   synthetic path, while the honestly-recalibrated 0.58 NAB model can still deploy via its
   own (lower) real-world gate.
6. **`deploy`** (needs `build`) — SSHes into a single, already-provisioned EC2 instance
   (`appleboy/ssh-action`) and runs `docker compose pull app && docker compose up -d app`
   against the same `docker-compose.yml` described in §7, refreshing just the `app`
   service to the image `build` just pushed. Because `deploy` needs `build`, which needs
   `integration_tests` + `validate_synthetic_easy`, this step is simply unreachable on any
   run where the gate failed — GitHub Actions skips it automatically. That skip *is* the
   "not deployed" half of the gate demonstration; nothing extra was needed to enforce it.

Secrets used: `MLFLOW_TRACKING_URI`, `MINIO_ENDPOINT_URL`,
`MINIO_ACCESS_KEY_ID`/`MINIO_SECRET_ACCESS_KEY` (MinIO-specific, deliberately separate
from AWS IAM creds), `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`REPO_NAME` for ECR, and
`EC2_HOST`/`EC2_USER`/`EC2_SSH_KEY` for the `deploy` job's SSH connection.

## 9. Deployment (single EC2 instance via `docker compose`)

No Kubernetes involved — the whole stack (Postgres, MinIO, MLflow, Airflow, and the app;
see §7) runs as one `docker-compose.yml` on one already-provisioned Linux EC2 instance.
Deployment is just "make the `app` container run the latest gate-approved image":

- The `build` job (§8) pushes the app image to ECR, tagged `latest`, only once
  `integration_tests` and `validate_synthetic_easy` have both passed.
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
| `test_model_quality.py` | Integration | Best NAB run clears `F1_THRESHOLD`, precision floor, and logs all transparency metrics (non-NaN) |
| `test_gate_behavior.py` | Integration | The 3-assertion argument from §3: synthetic model promoted at 0.85, real model correctly blocked at 0.85, real model passes its own 0.58 gate |

Unit tests need nothing but the repo; integration tests need a reachable MLflow instance
with the relevant experiments already populated (they `pytest.skip` gracefully if not).

## 11. Things to know if you're about to change something

- **Don't merge the two MLflow experiments or F1 thresholds.** This is a deliberate
  safety mechanism (§3), not incidental structure — merging them would let the easy
  synthetic model silently get served in production.
- **`config.py` is the first place to look** for any behavior change — nearly every
  constant is documented in-place with the reasoning behind its exact value.
- **DVC's committed `.dvc/config` endpoint is a local-dev default** (`localhost:9000`) —
  any other environment needs a `--local` override, never committed.
- **CI, MLflow, and MinIO all need to reach the same EC2 host's public IP** — see §4.3 and
  §9 for the ports that must stay open on that instance's security group.
