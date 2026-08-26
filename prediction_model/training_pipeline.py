import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
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
import prediction_model.processing.preprocessing as pp


def build_pipeline(model_type, params, window_size=config.WINDOW_SIZE):
    if model_type == 'isolation_forest':
        model = IsolationForest(
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            contamination=params['contamination'],
            bootstrap=params['bootstrap'],
            random_state=42,
        )
    else:
        model = OneClassSVM(kernel='rbf', nu=params['nu'], gamma=params['gamma'])

    return Pipeline(
        [
            ('RollingWindowFeatures', pp.RollingWindowFeatures(window=window_size)),
            ('MinMaxScale', MinMaxScaler()),
            ('AnomalyModel', model),
        ]
    )


isolation_forest_space = {
    'n_estimators': hp.choice('if_n_estimators', [50, 100, 150, 200, 300]),
    'max_features': hp.uniform('if_max_features', 0.5, 1.0),
    'contamination': hp.uniform('if_contamination', 0.05, 0.5),
    'bootstrap': hp.choice('if_bootstrap', [True, False]),
}

one_class_svm_space = {
    'nu': hp.uniform('ocsvm_nu', 0.05, 0.5),
    'gamma': hp.choice('ocsvm_gamma', ['scale', 'auto']),
}


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
    client.set_model_version_tag(config.REGISTERED_MODEL_NAME, registered.version, "source", source)
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, registered.version,
        "dataset_uploaded_by", dataset_meta.get("uploaded_by", "unknown"),
    )
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, registered.version,
        "dataset_uploaded_at", dataset_meta.get("uploaded_at", "unknown"),
    )
    return registered


def train_and_select(X_train, eval_df, experiment_name, f1_threshold):
    """Runs the Hyperopt search (Isolation Forest + optional One-Class SVM
    comparison) against X_train/eval_df, logs every trial to MLflow under
    `experiment_name`, and asserts the best trial's pointwise F1 clears
    `f1_threshold`. On success, registers and promotes that model to
    Production (register_and_promote) before returning the best run (a
    pandas Series, as returned by mlflow.search_runs).

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

    def make_objective(model_type):
        def objective(params):
            pipeline = build_pipeline(model_type, params)

            with mlflow.start_run(nested=True):
                pipeline.fit(X_train)
                metrics = evaluate_pipeline(pipeline, eval_df)

                mlflow.log_param('model_type', model_type)
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

        return objective

    # Single seeded generator, threaded through both searches in the same
    # order every run -- see config.HYPEROPT_SEED for why this is needed for
    # a reproducible winner.
    rstate = np.random.default_rng(config.HYPEROPT_SEED)

    print(f"Tuning Isolation Forest ({config.MAX_EVALS_IF} evals)...")
    fmin(
        fn=make_objective('isolation_forest'),
        space=isolation_forest_space,
        algo=tpe.suggest,
        max_evals=config.MAX_EVALS_IF,
        trials=Trials(),
        rstate=rstate,
    )

    if config.COMPARE_OCSVM:
        print(f"Tuning One-Class SVM ({config.MAX_EVALS_OCSVM} evals)...")
        fmin(
            fn=make_objective('one_class_svm'),
            space=one_class_svm_space,
            algo=tpe.suggest,
            max_evals=config.MAX_EVALS_OCSVM,
            trials=Trials(),
            rstate=rstate,
        )

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=['metrics.f1_score DESC'])
    best_run = runs_df.iloc[0]
    best_f1 = best_run['metrics.f1_score']
    best_model_type = best_run['params.model_type']

    print(
        f"Best run: {best_model_type} | "
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
