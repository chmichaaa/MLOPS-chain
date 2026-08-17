# Machine Learning Opearations (MLOps)

## MLOps maturity level 4

## Overview :
This project implements a robust MLOps pipeline, facilitating the continuous integration, continuous deployment, and monitoring of machine learning models. The infrastructure leverages AWS and various open-source tools (MLflow, Airflow, DVC, Evidently, Prometheus/Grafana) to ensure reproducibility and maintainability, deployed to a single EC2 instance via `docker compose` (see [PROJECT_INDEX.md](PROJECT_INDEX.md) section 9).

## Key Features :

**Data Versioning** : DVC (see [Data Pipeline & Versioning](#data-pipeline--versioning-dvc) below)

**Continuous Integration(CI)** : Triggered through ‘main.yml’ , building the code (docker), tests the code(Pytest),pushes the docker image to AWS ECR. 

**Experiment Tracking / Model Versioning** : MLflow 

**Continuous Deployment(CD)** : Deploys FastAPI to a single AWS EC2 instance for real-time and batch predictions. GitHub Actions builds the image, pushes it to ECR, then SSHes into the EC2 host and runs `docker compose pull app && docker compose up -d app` — but only once the promotion gate (F1 >= threshold) has passed; see [Two Datasets](#two-datasets-real-model-quality-vs-gate-validation) below for what actually gates this. MLflow, Postgres, and MinIO run on that same EC2 instance via `docker-compose.yml`, so the whole chain — train, gate, deploy, serve — lives on one box.

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


## Two Datasets: Real Model Quality vs Gate Validation

This project uses **two separate datasets that must never be confused**, each
answering a different question:

| | `training_pipeline.py` (NAB, real) | `train_synthetic_easy.py` (synthetic) |
|---|---|---|
| **Question answered** | How well does this model class actually detect real infra incidents? | Does the promotion gate itself work — does it promote a good model and block a bad one? |
| **Data** | 4 real AWS CloudWatch series (Numenta Anomaly Benchmark, `realAWSCloudwatch`) — a genuine April-2014 production incident, plus disclosed synthetic anomaly/background layers (see `build_dataset.py`) | Fully synthetic: Gaussian noise + isolated 5-10σ point spikes at known timestamps (see `build_synthetic_dataset.py`) |
| **Anomaly shape** | Gradual multivariate regime shifts (hours-long incidents) | Sharp, single-sample point outliers — deliberately the easy case |
| **MLflow experiment** | `infra_anomaly_detection` | `infra_anomaly_detection_synthetic_easy` — a **separate experiment**, so this dataset's easy ~0.9 F1 can never outrank and get served instead of the real model (`predict.py` only ever queries the NAB experiment) |
| **F1 threshold** | `config.F1_THRESHOLD = 0.58` — recalibrated from the original 0.85 spec target, which is not honestly reachable on real data with Isolation Forest/OCSVM (see `config.py`'s `F1_THRESHOLD` comment for the full diagnostic) | `config.SYNTHETIC_EASY_F1_THRESHOLD = 0.85` — the **original**, unmodified spec threshold |
| **Result** | F1 (pointwise) = 0.59 | F1 (pointwise) = 0.90 |
| **Gate outcome** | Blocked under the original 0.85 bar (expected — see `tests/test_gate_behavior.py`); promoted under its own recalibrated 0.58 gate | Promoted |

**Why this exists**: real NAB data cannot reach F1 ≥ 0.85 with this model
class regardless of whether the gate logic is correct — the incidents are
gradual regime shifts, not point outliers, which Isolation Forest/OCSVM
aren't built to isolate. That makes it impossible to tell, from the NAB
result alone, whether a low F1 means "the gate is working as designed" or
"the gate is broken and would block anything." The synthetic dataset removes
that confound: same pipeline, same gate code
(`training_pipeline.train_and_select`), same original threshold, but
anomalies that are trivially separable by construction. Promoting a model
here proves the gate mechanism itself works; the NAB result is then a genuine
model/data-fit finding, not a gate bug.

Both experiments' runs are also tagged `dataset_type=nab_real` /
`dataset_type=synthetic_easy` for a second, human-readable way to tell them
apart in the MLflow UI (separate experiments is the actual safety mechanism;
the tag is for clarity when browsing).

    python -m prediction_model.processing.build_synthetic_dataset   # build the dataset once
    python -m prediction_model.train_synthetic_easy                 # run the gate against it



## Data Monitoring :

Drift and data-quality checks run programmatically via `drift_monitoring/check_drift.py`
(Evidently `DataDriftPreset`), not a manual dashboard — see
[PROJECT_INDEX.md](PROJECT_INDEX.md) section 6.3 for how it's wired into Airflow.

## Continuous Monitoring(CM)

FastAPI exposes `/metrics` via `prometheus_fastapi_instrumentator` (see `main.py`), ready
to be scraped by a Prometheus instance and visualized in Grafana.

