import time

import pandas as pd
from prediction_model.config import config
import mlflow
from mlflow.tracking import MlflowClient


# Module-level cache: avoids querying MLflow (and re-downloading the model
# artifact) on every prediction request. See config.MODEL_CACHE_TTL_SECONDS.
_cache = {"pipeline": None, "version": None, "loaded_at": 0.0}


def _load_best_pipeline():
    """Loads whatever is currently staged 'Production' for
    config.REGISTERED_MODEL_NAME -- training_pipeline.py's gate and
    dataset_uploader's direct model-upload path are the only two things that
    ever promote a model there (see training_pipeline.register_and_promote),
    so this is always "the model that passed the gate most recently."
    """
    now = time.time()
    if _cache["pipeline"] is not None and (now - _cache["loaded_at"]) < config.MODEL_CACHE_TTL_SECONDS:
        return _cache["pipeline"]

    client = MlflowClient()
    versions = client.get_latest_versions(config.REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No model is currently in the 'Production' stage for '{config.REGISTERED_MODEL_NAME}' -- "
            "run training_pipeline.py, or promote one via dataset_uploader, first."
        )
    current_version = versions[0].version

    if current_version != _cache["version"]:
        model_uri = f"models:/{config.REGISTERED_MODEL_NAME}/Production"
        _cache["pipeline"] = mlflow.sklearn.load_model(model_uri)
        _cache["version"] = current_version

    _cache["loaded_at"] = now
    return _cache["pipeline"]


def generate_predictions(window_data):
    """window_data: list of raw metric readings (dicts with config.METRIC_COLUMNS keys),
    in chronological order, oldest first / most recent last. Scores the most recent point.
    """
    data = pd.DataFrame(window_data)[config.METRIC_COLUMNS]
    pipeline = _load_best_pipeline()
    anomaly_score = pipeline.decision_function(data)[-1]
    is_anomaly = pipeline.predict(data)[-1] == -1
    return {"anomaly_score": float(anomaly_score), "is_anomaly": bool(is_anomaly)}


def generate_predictions_batch(data_input):
    """data_input: DataFrame of raw metric readings (config.METRIC_COLUMNS), in
    chronological order. Scores every row.
    """
    data = data_input[config.METRIC_COLUMNS]
    pipeline = _load_best_pipeline()
    anomaly_scores = pipeline.decision_function(data)
    is_anomaly = pipeline.predict(data) == -1
    return {"anomaly_score": anomaly_scores, "is_anomaly": is_anomaly}
