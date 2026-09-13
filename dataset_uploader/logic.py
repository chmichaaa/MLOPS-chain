"""Pure, side-effect-free helpers factored out of dataset_uploader/app.py so
they're unit-testable without a real Postgres/MLflow connection -- app.py
itself does real I/O at import time (create_engine, a CREATE TABLE, reading
required env vars), which is exactly what makes it unsuitable for a fast
unit test. Same reasoning as why processing/evaluation.py is split out from
training_pipeline.py elsewhere in this codebase.

Tested in tests/test_dataset_uploader_logic.py, run in CI's fast unit_tests
job alongside test_evaluation.py/test_preprocessing.py.
"""
from datetime import datetime, timedelta, timezone

from prediction_model.config import config

GMT_PLUS_1 = timezone(timedelta(hours=1))
SOURCE_LABELS = {"training": "CI/CD pipeline", "model_upload": "Direct upload"}
REQUIRED_DATASET_COLUMNS = {"timestamp", "label", "is_synthetic_anomaly", *config.METRIC_COLUMNS}


def now_gmt1():
    return datetime.now(GMT_PLUS_1)


def format_timestamp(value):
    """Renders a stored ISO timestamp (already in GMT+1 -- see now_gmt1) as a
    short, human display string. Passes through unrecognized values (e.g.
    the "unknown" default from load_dataset_metadata) unchanged.
    """
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M GMT+1")
    except (TypeError, ValueError):
        return value


def format_epoch_millis(ms):
    """Renders an MLflow ModelVersion timestamp (creation_timestamp /
    last_updated_timestamp -- epoch milliseconds, UTC) in the same short
    GMT+1 display format format_timestamp uses, so "Registered at" reads
    consistently with "Dataset at" elsewhere in the console instead of
    mixing a raw epoch int in with ISO-string fields.
    """
    if ms is None:
        return "unknown"
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(GMT_PLUS_1)
    return dt.strftime("%Y-%m-%d %H:%M GMT+1")


def format_hyperparams(params):
    """params: an MLflow run's .data.params dict. Renders every param except
    'model_type' (shown separately) as "key=value" pairs -- used to show a
    model's actual hyperparameters (e.g. LSTMAutoencoder's hidden_size/epochs)
    on the console. Uploaded models often log nothing beyond model_type (see
    training_pipeline.infer_model_type), so this returns a clear placeholder
    rather than an empty string in that case.
    """
    pairs = [f"{key}={value}" for key, value in sorted(params.items()) if key != "model_type"]
    return ", ".join(pairs) if pairs else "no hyperparameters logged"


def source_label(value):
    return SOURCE_LABELS.get(value, value)


def stage_css_class(stage):
    return {"Production": "pill-good", "Archived": "pill-neutral"}.get(stage, "pill-neutral")


def missing_dataset_columns(columns):
    """columns: an iterable of column names (e.g. a DataFrame's .columns).
    Returns the set of required columns not present -- empty set means the
    schema is valid.
    """
    return REQUIRED_DATASET_COLUMNS - set(columns)
