import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from prediction_model.config import config
from prediction_model.processing.data_handling import load_full_dataset, day_block_split, load_dataset_metadata
from prediction_model.processing.evaluation import point_adjust, windowed_f1
from prediction_model.processing.lstm_autoencoder import LSTMAutoencoder
import prediction_model.processing.preprocessing as pp

MODEL_TYPE = 'lstm_autoencoder'


def build_pipeline(params, window_size=config.WINDOW_SIZE):
    model = LSTMAutoencoder(
        seq_len=config.LSTM_SEQ_LEN,
        hidden_size=params['hidden_size'],
        latent_size=params['latent_size'],
        epochs=params['epochs'],
        lr=params['lr'],
        threshold_percentile=params['threshold_percentile'],
        random_state=42,
    )

    return Pipeline(
        [
            ('RollingWindowFeatures', pp.RollingWindowFeatures(window=window_size)),
            ('MinMaxScale', MinMaxScaler()),
            ('AnomalyModel', model),
        ]
    )


# threshold_percentile means roughly "what fraction of training-window
# reconstruction errors count as the normal ceiling." hidden_size/latent_size
# kept small -- this dataset's train split is a few thousand rows
# (day_block_split), not enough to justify a larger network, and every trial
# here has a real wall-clock training cost (a full neural net fit, not a
# single IsolationForest.fit() call).
lstm_autoencoder_space = {
    'hidden_size': hp.choice('lstm_hidden_size', [16, 32, 64]),
    'latent_size': hp.choice('lstm_latent_size', [8, 16, 32]),
    'epochs': hp.choice('lstm_epochs', [10, 20, 30]),
    'lr': hp.loguniform('lstm_lr', np.log(1e-4), np.log(1e-2)),
    'threshold_percentile': hp.uniform('lstm_threshold_percentile', 80, 99),
}


def infer_model_type(pipeline):
    """Best-effort label for "what model is this," used to tag the registered
    model version (see register_and_promote) so it's visible directly in
    MLflow's Model Registry / the MLOps console without digging into the run.
    Trained-here pipelines always report MODEL_TYPE via the 'model_type'
    MLflow param logged in train_and_select below; this function only exists
    for the dataset_uploader /upload/model path, where the uploaded object can
    be anything with a .predict() method (see PROJECT_INDEX.md's note on that
    route's accepted security tradeoff) -- so introspect its actual class
    rather than assume it's this project's own Pipeline shape.
    """
    if isinstance(pipeline, Pipeline) and 'AnomalyModel' in pipeline.named_steps:
        return type(pipeline.named_steps['AnomalyModel']).__name__
    return type(pipeline).__name__


def evaluate_pipeline(pipeline, eval_df):
    """Scores an already-fitted pipeline against eval_df and returns the same
    metrics dict train_and_select logs per trial. Extracted so this is the
    ONE gate implementation -- dataset_uploader's direct model-upload path
    (evaluating an uploaded, already-trained pipeline instead of one this
    module just fitted) calls this too, rather than re-implementing the
    metric/precision-floor logic a second time.
    """
    X_eval = eval_df[config.METRIC_COLUMNS]
    y_eval = eval_df['label']
    is_synthetic_anomaly_eval = eval_df['is_synthetic_anomaly'].values

    raw_pred = pipeline.predict(X_eval)
    y_pred = (raw_pred == -1).astype(int)

    f1_pointwise = f1_score(y_eval, y_pred, zero_division=0)
    f1_adjusted = f1_score(y_eval, point_adjust(y_eval.values, y_pred), zero_division=0)
    f1_windowed = windowed_f1(eval_df, y_pred, config.WINDOW_SIZE)
    precision = precision_score(y_eval, y_pred, zero_division=0)
    recall = recall_score(y_eval, y_pred, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)

    # -1.0 (out of a recall/rate metric's valid [0,1] range) marks
    # "not applicable" -- e.g. no real (non-synthetic) anomalies present in
    # this particular eval split. Deliberately not NaN: some MLflow backends
    # mishandle repeatedly logging the exact same NaN value across many fast
    # successive runs.
    NOT_APPLICABLE = -1.0
    real_mask = (y_eval.values == 1) & (is_synthetic_anomaly_eval == 0)
    synth_mask = (y_eval.values == 1) & (is_synthetic_anomaly_eval == 1)
    normal_mask = y_eval.values == 0
    recall_real = float(y_pred[real_mask].mean()) if real_mask.any() else NOT_APPLICABLE
    recall_synthetic = float(y_pred[synth_mask].mean()) if synth_mask.any() else NOT_APPLICABLE
    false_positive_rate_normal = float(y_pred[normal_mask].mean()) if normal_mask.any() else NOT_APPLICABLE

    # Reject (don't select) trials that don't clear the precision floor,
    # regardless of their F1 -- see config.PRECISION_FLOOR. The true,
    # unpenalised F1 is still returned separately so it stays visible.
    meets_precision_floor = precision >= config.PRECISION_FLOOR
    selection_f1 = f1_pointwise if meets_precision_floor else 0.0

    return {
        # f1_score is the primary selection/threshold metric: pointwise F1,
        # zeroed out if the precision floor isn't met.
        'f1_score': selection_f1,
        'f1_score_unpenalized': f1_pointwise,
        'f1_score_point_adjusted': f1_adjusted,
        'f1_score_windowed': f1_windowed,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'recall_real_incidents': recall_real,
        'recall_synthetic_incidents': recall_synthetic,
        'false_positive_rate_normal': false_positive_rate_normal,
        'meets_precision_floor': float(meets_precision_floor),
    }


def register_and_promote(run_id, source, dataset_meta=None):
    """Registers the given run's model under config.REGISTERED_MODEL_NAME and
    transitions it straight to the Production stage, archiving whatever was
    Production before. This IS the deployment record -- both the training
    gate (below) and dataset_uploader's direct model-upload path call this
    same function to mark "this is now the model predict.py serves," so
    there's exactly one mechanism to check, not a separate deployment-log
    database plus this.

    Also tags the registered model VERSION (not just the underlying run) with
    model_type/f1_score, so "what model is this" is visible directly on
    MLflow's Model Registry page and the MLOps console's dashboard/history --
    without those, both only showed dataset provenance, not the model itself.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{config.MODEL_NAME.lstrip('/')}"
    registered = mlflow.register_model(model_uri=model_uri, name=config.REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    dataset_meta = dataset_meta or {}
    run = mlflow.get_run(run_id)
    model_type = run.data.params.get('model_type', 'unknown')
    f1_score_value = run.data.metrics.get('f1_score')

    tags = {
        "source": source,
        "model_type": model_type,
        "dataset_uploaded_by": dataset_meta.get("uploaded_by", "unknown"),
        "dataset_uploaded_at": dataset_meta.get("uploaded_at", "unknown"),
    }
    if f1_score_value is not None:
        tags["f1_score"] = f"{f1_score_value:.4f}"

    for key, value in tags.items():
        client.set_model_version_tag(config.REGISTERED_MODEL_NAME, registered.version, key, value)

    return registered


def train_and_select(X_train, eval_df, experiment_name, f1_threshold):
    """Runs the Hyperopt search over LSTMAutoencoder against X_train/eval_df,
    logs every trial to MLflow under `experiment_name`, and asserts the best
    trial's pointwise F1 clears `f1_threshold`. On success, registers and
    promotes that model to Production (register_and_promote) before returning
    the best run (a pandas Series, as returned by mlflow.search_runs).

    This is THE promotion gate: whatever dataset.csv contains when this runs
    is what gets trained/evaluated against config.F1_THRESHOLD -- see
    config.py and processing/build_dataset.py / build_synthetic_dataset.py
    for how to control whether that data clears the bar or not.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    X_eval = eval_df[config.METRIC_COLUMNS]
    y_eval = eval_df['label']
    is_synthetic_anomaly_eval = eval_df['is_synthetic_anomaly'].values
    dataset_meta = load_dataset_metadata()

    print(f"train: {len(X_train)} rows | eval: {len(X_eval)} rows, "
          f"{y_eval.mean():.1%} anomalous "
          f"({((y_eval == 1) & (is_synthetic_anomaly_eval == 0)).sum()} real anomaly, "
          f"{is_synthetic_anomaly_eval.sum()} synthetic anomaly, "
          f"{(y_eval == 0).sum()} normal)")

    def objective(params):
        pipeline = build_pipeline(params)

        with mlflow.start_run(nested=True):
            pipeline.fit(X_train)
            metrics = evaluate_pipeline(pipeline, eval_df)

            mlflow.log_param('model_type', MODEL_TYPE)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    'dataset_uploaded_by': dataset_meta.get('uploaded_by', 'unknown'),
                    'dataset_uploaded_at': dataset_meta.get('uploaded_at', 'unknown'),
                }
            )
            mlflow.sklearn.log_model(
                pipeline, config.MODEL_NAME.lstrip('/'), serialization_format='cloudpickle'
            )

        return {'loss': 1 - metrics['f1_score'], 'status': STATUS_OK}

    print(f"Tuning LSTM Autoencoder ({config.MAX_EVALS_LSTM} evals)...")
    fmin(
        fn=objective,
        space=lstm_autoencoder_space,
        algo=tpe.suggest,
        max_evals=config.MAX_EVALS_LSTM,
        trials=Trials(),
        rstate=np.random.default_rng(config.HYPEROPT_SEED),
    )

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=['metrics.f1_score DESC'])
    best_run = runs_df.iloc[0]
    best_f1 = best_run['metrics.f1_score']

    print(
        f"Best run: {MODEL_TYPE} | "
        f"f1_score (pointwise, precision-floor enforced) = {best_f1:.4f} | "
        f"f1_score_point_adjusted = {best_run['metrics.f1_score_point_adjusted']:.4f} | "
        f"f1_score_windowed = {best_run['metrics.f1_score_windowed']:.4f} | "
        f"precision = {best_run['metrics.precision']:.4f} | "
        f"recall_real_incidents = {best_run['metrics.recall_real_incidents']:.4f} | "
        f"recall_synthetic_incidents = {best_run['metrics.recall_synthetic_incidents']:.4f} | "
        f"false_positive_rate_normal = {best_run['metrics.false_positive_rate_normal']:.4f}"
    )

    assert best_f1 >= f1_threshold, (
        f"Best model F1 (pointwise) = {best_f1:.4f} on the eval set, "
        f"below the required threshold of {f1_threshold}."
    )

    register_and_promote(best_run['run_id'], source='training', dataset_meta=dataset_meta)

    return best_run


if __name__ == "__main__":
    dataset = load_full_dataset()
    train_df, eval_df = day_block_split(dataset)
    X_train = train_df[config.METRIC_COLUMNS]

    train_and_select(
        X_train=X_train,
        eval_df=eval_df,
        experiment_name=config.EXPERIMENT_NAME,
        f1_threshold=config.F1_THRESHOLD,
    )
