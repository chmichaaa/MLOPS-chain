import os


current_directory = os.path.dirname(os.path.realpath(__file__)) #current directory of the script

PACKAGE_ROOT = os.path.dirname(current_directory) #parent directory of current directory


DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

# Single labelled dataset (real NAB series + injected synthetic anomalies, see
# processing/build_dataset.py). training_pipeline.py splits it into train/eval
# by whole calendar day (data_handling.day_block_split) -- there is no
# separate train/test file, unlike the original loan-prediction dataset.
DATA_FILE = 'dataset.csv'

TARGET = 'label'

# Real AWS CloudWatch metrics (NAB realAWSCloudwatch, see processing/build_dataset.py)
METRIC_COLUMNS = ['cpu_usage_pct', 'network_in_bytes', 'elb_request_count', 'rds_cpu_usage_pct']

# Rolling window size (in samples) used to compute the moving-mean feature.
# Data is sampled roughly every 5 minutes, so 12 samples ~= 1 hour lookback.
WINDOW_SIZE = 12

# day_block_split: fraction of anomaly-free calendar days held out for eval
# (never trained on), so eval also measures generalisation to unseen normal
# behaviour, not just incident detection. Any day containing an anomaly (real
# or synthetic) always goes entirely to eval. Set high (0.55) because with the
# synthetic background extension below there are 61 clean days total -- most
# can go to eval (to dilute the anomaly ratio to something realistic) while
# still leaving ~27 days (~7800 rows) for training.
EVAL_CLEAN_FRACTION = 0.55
SPLIT_SEED = 42

ROWS_PER_DAY = 288  # ~5-minute sampling

# Synthetic anomaly injection (see processing/build_dataset.py for the full
# rationale): sustained shifts, clearly separable by construction, placed only
# inside days that already contain a real incident.
SYNTHETIC_N_EVENTS = 70          # target; actual count is capped by available room
SYNTHETIC_EVENT_LEN = 18         # samples (~90 min at 5-min sampling)
SYNTHETIC_MAGNITUDE_STD = 6      # shift size, in std deviations of each metric
SYNTHETIC_SEED = 7

# Synthetic background (normal) extension (see processing/build_dataset.py):
# the 15 real days alone contain more incident time than clean time (9 of 15
# days already have a real incident), so there isn't enough real normal data
# to both train robustly AND reach a realistic eval anomaly ratio. These extra
# days are block-bootstrapped from the real clean days, not fabricated from
# scratch, and are flagged via is_synthetic_background for transparency.
SYNTHETIC_NORMAL_DAYS = 55
SYNTHETIC_NORMAL_BLOCK_LEN = 48  # samples (~4h blocks, preserves diurnal shape)
SYNTHETIC_NORMAL_SEED = 99

# Isolation Forest is the primary model; One-Class SVM is trained alongside
# for comparison when COMPARE_OCSVM is True (both logged to the same MLflow
# experiment, best one wins by validation F1).
COMPARE_OCSVM = True
MAX_EVALS_IF = 25
MAX_EVALS_OCSVM = 15

# Seeds Hyperopt's TPE search (via rstate=numpy.random.default_rng(seed) in
# training_pipeline.py) so the sequence of hyperparameter trials -- and
# therefore which model/config wins -- is identical across runs on the same
# dataset. Without this, two runs could pick different winners (e.g.
# Isolation Forest vs One-Class SVM) purely from search-order randomness, even
# though every other source of randomness in this pipeline was already fixed
# (day_block_split's SPLIT_SEED, the synthetic injection seeds, IsolationForest's
# own random_state=42). One seed for both searches (IF, then OCSVM) -- they
# run sequentially against the same rstate object, so the overall sequence is
# still fully deterministic run-to-run.
HYPEROPT_SEED = 42

# Hyperopt trials with precision below this floor are rejected outright
# (heavily penalised), regardless of their F1 -- otherwise an aggressive
# "flag almost everything" model can still post a passable F1 on an
# anomaly-heavy eval set without being a usable detector (it was hitting an
# 80% false-positive rate on normal data before this was added).
PRECISION_FLOOR = 0.4

# The promotion gate: train_and_select (training_pipeline.py) asserts the best
# run's pointwise F1 clears this bar, and only a passing run ever reaches the
# CI `build`/`deploy` jobs (.github/workflows/main.yml). This is the ONE
# threshold the pipeline knows about -- whether a given run passes or fails is
# controlled entirely by which dataset.csv you feed it (see the two generators
# in processing/: build_dataset.py produces genuinely hard, real-incident-based
# data that sits right around this bar; build_synthetic_dataset.py produces
# easy, clearly-separable data that clears it comfortably), not by branching
# pipeline logic. f1_score_point_adjusted and f1_score_windowed are also logged
# in MLflow for transparency but are NOT used for model selection/threshold
# (point-adjustment in particular is known to inflate scores independently of
# real detection quality on long anomaly segments -- Kim et al. 2021).
F1_THRESHOLD = 0.75

# Bucket used both for MLflow artifacts (docker-compose.yml points MLflow's
# --default-artifact-root at it) and for batch-prediction/drift-monitoring
# uploads. Renamed from the original "loanprediction" (a leftover from the
# pre-pivot loan-approval project) now that everything is infra-metrics.
S3_BUCKET = os.environ.get("S3_BUCKET", "infra-monitoring")

FOLDER = "datadrift"

# S3-compatible endpoint for boto3 clients (main.py's upload_to_s3,
# drift_monitoring/check_drift.py). Defaults to the local docker-compose MinIO
# service; leave unset to talk to real AWS S3 instead.
MINIO_ENDPOINT_URL = os.environ.get("MINIO_ENDPOINT_URL", "http://localhost:9000")

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

EXPERIMENT_NAME = "infra_anomaly_detection"

MODEL_NAME = "/AnomalyDetection-model"

# MLflow Model Registry name: the promotion gate (train_and_select) registers every
# passing model under this name and transitions it to the "Production" stage
# (archiving whatever was there before). predict.py always serves whatever is
# currently staged Production here -- this IS the deployment record (who/when/which
# run), so there's no separate deployment-log database. Both the CI training path and
# dataset_uploader's direct model-upload path promote through this same mechanism.
REGISTERED_MODEL_NAME = "AnomalyDetection"

# drift_monitoring/check_drift.py: Evidently DataDriftPreset flags a dataset as
# drifted if the SHARE of drifted columns exceeds this threshold. Checked every
# DRIFT_CHECK_SCHEDULE_MINUTES by the Airflow DAG (dag_drift_retrain.py), which
# bounds worst-case detection latency to 2x that interval -- see the DAG for
# the < 10 min requirement.
DRIFT_THRESHOLD = 0.5
DRIFT_CHECK_SCHEDULE_MINUTES = 5

# predict.py caches the loaded pipeline for this long before re-checking MLflow
# for a newer best run -- avoids querying MLflow (and re-downloading the model
# artifact) on every single prediction request, which was dominating request
# latency. 10 minutes keeps served predictions reasonably fresh after a
# retrain without adding MLflow round-trips to the request path.
MODEL_CACHE_TTL_SECONDS = 600


# =============================================================================
# Easy demo dataset (prediction_model/processing/build_synthetic_dataset.py)
# =============================================================================
# An alternative, fully-synthetic generator for DATA_FILE -- run it instead of
# build_dataset.py when you want data that clearly clears F1_THRESHOLD (e.g. to
# demo the gate promoting a model), the same way build_dataset.py's real data
# sits right around the bar (to demo the gate blocking one). Both write to the
# same DATA_FILE/DATAPATH, so whichever you ran most recently is what
# training_pipeline.py trains against -- that choice, not any branching
# pipeline logic, is what controls pass/fail.
#
# Baseline is stationary Gaussian noise (no diurnal pattern), with a handful
# of sustained multi-sample shifts injected on top -- same injection style as
# build_dataset.py's inject_synthetic_anomalies (clustered events, not
# scattered single-point spikes), so this data works with the same
# day_block_split and the same WINDOW_SIZE as build_dataset.py's output,
# rather than needing its own special-cased split/window handling.
EASY_DATA_SEED = 123
EASY_DATA_N_ROWS = 5000          # ~17 days at 5-min sampling
EASY_DATA_N_EVENTS = 6           # sustained-shift events, one per day
EASY_DATA_EVENT_LEN = 18         # samples per event (~90 min at 5-min sampling)
EASY_DATA_MAGNITUDE_STD = 8      # shift size, in std deviations of each metric

# Baseline (mean, std) per metric -- stationary Gaussian noise, no diurnal
# pattern or drift (deliberately: this dataset's normal region should be
# trivially learnable, so it clears F1_THRESHOLD by a comfortable margin).
EASY_DATA_METRIC_PARAMS = {
    'cpu_usage_pct': {'mean': 30.0, 'std': 5.0},
    'network_in_bytes': {'mean': 500_000.0, 'std': 50_000.0},
    'elb_request_count': {'mean': 100.0, 'std': 15.0},
    'rds_cpu_usage_pct': {'mean': 20.0, 'std': 4.0},
}
