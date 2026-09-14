# Machine Learning Opearations (MLOps)

## MLOps maturity level 4

## Overview :
This project implements a robust MLOps pipeline, facilitating the continuous integration, continuous deployment, and monitoring of machine learning models. The infrastructure leverages AWS and various open-source tools (MLflow, Airflow, DVC, Evidently, Prometheus/Grafana) to ensure reproducibility and maintainability, deployed to a single EC2 instance via `docker compose` (see [PROJECT_INDEX.md](PROJECT_INDEX.md) section 9).

## Key Features :

**Data Versioning** : DVC (see [Data Pipeline & Versioning](#data-pipeline--versioning-dvc) below)

**Continuous Integration(CI)** : Triggered through ‘main.yml’ , building the code (docker), tests the code(Pytest),pushes the docker image to AWS ECR. 

**Experiment Tracking / Model Versioning** : MLflow 

**Continuous Deployment(CD)** : Deploys FastAPI to a single AWS EC2 instance for real-time and batch predictions. GitHub Actions builds the image, pushes it to ECR, then SSHes into the EC2 host and runs `docker compose pull app && docker compose up -d app` — but only once the promotion gate (F1 >= `config.F1_THRESHOLD`) has passed; see [The Promotion Gate](#the-promotion-gate) below for what actually gates this. MLflow, Postgres, and MinIO run on that same EC2 instance via `docker-compose.yml`, so the whole chain — train, gate, deploy, serve — lives on one box.

**Continuous Monitoring(CM)** : Integrating the ‘/metrics’ method of  FastAPI in Prometheus and visualizes endpoints in Grafana.  

**Continuous Training(CT)** : Triggers code execution through GitHub Actions when new data is pushed to the remote DVC location and committed to Git. 

**Drift Monitoring** : `drift_monitoring/check_drift.py` compares each new batch-prediction upload against the training reference using Evidently's `DataDriftPreset`, run on a schedule by Airflow (`dag_drift_retrain.py`) and logged to MLflow — see [PROJECT_INDEX.md](PROJECT_INDEX.md) section 6.3.


## Data Pipeline & Versioning (DVC)

Two separate mechanisms cover the dataset, each with one job:

**`prediction_model/processing/build_dataset.py` — source of truth for *regenerating* data.**
Downloads 4 real AWS CloudWatch series from the Numenta Anomaly Benchmark
(NAB, `realAWSCloudwatch`), merges them into one aligned, labelled time
series, and adds two disclosed synthetic layers on top (injected anomaly
events + a background-normal extension — see the module's own docstring for
the full rationale). Run it whenever you want a *fresh* dataset built from
source:

    python -m prediction_model.processing.build_dataset

This is deterministic (fixed seeds) and writes `prediction_model/datasets/dataset.csv`.

**DVC — a frozen, traceable *snapshot* of one specific run of the above.**
`prediction_model/datasets/dataset.csv.dvc` pins the exact MD5 hash of the
dataset version the training pipeline actually trains against, so:
- `training_pipeline.py` always runs against a known, reproducible dataset,
  without re-hitting NAB's GitHub URLs on every run, and without silently
  drifting if NAB's source files ever change upstream.
- Anyone can restore that exact version with `dvc pull`/`dvc checkout`,
  without regenerating it.

To publish a new frozen version after changing `build_dataset.py` or its
config (e.g. `config.SYNTHETIC_N_EVENTS`):

    python -m prediction_model.processing.build_dataset   # regenerate from source
    dvc add prediction_model/datasets/dataset.csv          # compute the new hash
    dvc push                                                # upload to the MinIO remote
    git add prediction_model/datasets/dataset.csv.dvc
    git commit -m "Update dataset snapshot"

**Remote**: `s3://infra-monitoring` on MinIO (`.dvc/config`), the same bucket
used for MLflow artifacts and batch-prediction uploads (see MinIO in
docker-compose.yml). The committed `endpointurl` (`http://localhost:9000`) is
a local-dev default — it only resolves correctly when `dvc push`/`dvc
pull`/`docker build` run on the same host as MinIO. For any other
environment (a CI runner, an EC2 instance where MinIO lives elsewhere),
override it with a machine-local config that is never committed (DVC does
not expand environment variables inside the tracked `.dvc/config`):

    dvc remote modify --local myremote endpointurl http://<your-minio-host>:9000

The `Dockerfile` build accepts this as a build-arg (`MINIO_ENDPOINT_URL`) and
applies the same override automatically before `dvc pull` — see
`.github/workflows/main.yml` for how CI supplies it via a `MINIO_ENDPOINT_URL`
secret (along with `MINIO_ACCESS_KEY_ID`/`MINIO_SECRET_ACCESS_KEY`, kept
deliberately separate from the real AWS IAM credentials used for ECR/EC2 deployment).

**Resolved by the single-EC2-instance deployment**: earlier versions of this
project targeted EKS, which meant CI (running on a GitHub-hosted runner, not
inside your VPC) couldn't reach a VPC-local MinIO for `dvc pull`/MLflow calls
without extra networking work. Now that MinIO and MLflow run on the same
publicly-reachable EC2 instance that serves the app, CI just talks to that
instance's public IP like any other remote service (`MINIO_ENDPOINT_URL`/
`MLFLOW_TRACKING_URI` secrets point at it). This does mean ports 5000 and 9000
are internet-facing — MinIO still requires its access key/secret, but MLflow's
tracking server has no built-in auth, so treat this as a demo-appropriate
tradeoff, not a production-hardened setup.


## The Promotion Gate

One pipeline, one dataset file, one threshold: `training_pipeline.py` always
trains against whatever `prediction_model/datasets/dataset.csv` currently
contains, evaluates it, and asserts the best model's pointwise F1 clears
`config.F1_THRESHOLD` (0.75) before anything logs to MLflow as promotable.
CI only reaches the `build`/`deploy` jobs if that assert passes — see
`.github/workflows/main.yml`'s single `validate` job.

What controls whether a given run passes or fails is entirely **which data
you feed it**, via two interchangeable generators that both write to the same
`dataset.csv`:

    python -m prediction_model.processing.build_dataset            # harder: real AWS incidents, sits right around the bar
    python -m prediction_model.processing.build_synthetic_dataset  # easier: synthetic, clears the bar comfortably

Run `build_dataset.py` to demo the gate blocking a model that doesn't
reach 0.75 (real NAB incidents are gradual multivariate regime shifts, not
sharp point outliers). Run `build_synthetic_dataset.py` to demo the gate
promoting one (clearly-separable synthetic shifts reach F1 ~0.9+). Either way
it's the same gate code (`training_pipeline.train_and_select`) making the
call, not two different code paths.

The model trained against that gate is an `LSTMAutoencoder`
(`prediction_model/processing/lstm_autoencoder.py`) — it reconstructs a
trailing window of rows and scores by reconstruction error, so error rises as
the real trajectory drifts from learned-normal dynamics across the window,
not just at one point. This replaced an earlier Isolation Forest/One-Class
SVM comparison: both scored each row's engineered features independently,
with no notion of trajectory — a structural mismatch for incidents that are
gradual, multi-hour regime shifts rather than point outliers.



## Live Telemetry

A `live-feed` service (`live_feed/generator.py`) emits a continuous stream of
host/load-balancer/database metrics and scores every reading through the **deployed
model** over HTTP, exactly as any client would. Signals are driven by AR(1) processes
around the baselines the model was fitted on, so they drift smoothly and move together
the way real host metrics do, with four correlated degradation modes (traffic surge, CPU
saturation, database contention, network saturation) ramping in and out on a schedule.

The console's `/live` view shows the current verdict, per-metric tiles with sparklines
and trend, an anomaly-score chart with incident windows and flagged readings, and an
incident log reporting detection coverage and time-to-detect. The stream is generated
rather than collected from a real fleet; everything downstream of the reading — the
serving app, the registered model, the scoring path, the verdicts — is real. See
[PROJECT_INDEX.md](PROJECT_INDEX.md) section 7.2.

## Data Monitoring :

Drift and data-quality checks run programmatically via `drift_monitoring/check_drift.py`
(Evidently `DataDriftPreset`), not a manual dashboard — see
[PROJECT_INDEX.md](PROJECT_INDEX.md) section 6.3 for how it's wired into Airflow.

## Continuous Monitoring(CM)

FastAPI exposes `/metrics` via `prometheus_fastapi_instrumentator` (see `main.py`), ready
to be scraped by a Prometheus instance and visualized in Grafana.

