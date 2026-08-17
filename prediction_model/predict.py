import time

import pandas as pd
from prediction_model.config import config
import mlflow


# Module-level cache: avoids querying MLflow (and re-downloading the model
# artifact) on every prediction request. See config.MODEL_CACHE_TTL_SECONDS.
_cache = {"pipeline": None, "run_id": None, "loaded_at": 0.0}


def _load_best_pipeline():
    now = time.time()
    if _cache["pipeline"] is not None and (now - _cache["loaded_at"]) < config.MODEL_CACHE_TTL_SECONDS:
        return _cache["pipeline"]

    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(experiment_ids=experiment_id, order_by=['metrics.f1_score DESC'])
    best_run = runs_df.iloc[0]
    best_run_id = best_run['run_id']

    if best_run_id != _cache["run_id"]:
        best_model_uri = 'runs:/' + best_run_id + config.MODEL_NAME
        _cache["pipeline"] = mlflow.sklearn.load_model(best_model_uri)
        _cache["run_id"] = best_run_id

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
