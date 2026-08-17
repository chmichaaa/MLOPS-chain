"""Quality gate: the best MLflow run for this experiment must clear
config.F1_THRESHOLD (pointwise F1, precision-floor enforced -- see
training_pipeline.py / config.py for why this isn't the spec's literal 0.85).
Independent of training_pipeline.py's own assert so it can run standalone in
CI against whatever model is currently registered, without re-running the
multi-minute Hyperopt search.
"""
import pytest
import mlflow

from prediction_model.config import config

mlflow.set_tracking_uri(config.TRACKING_URI)


@pytest.fixture(scope='module')
def best_run():
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        pytest.skip(f"no MLflow experiment '{config.EXPERIMENT_NAME}' yet -- run training_pipeline.py first")
    runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=['metrics.f1_score DESC'])
    if runs_df.empty:
        pytest.skip("no runs logged yet -- run training_pipeline.py first")
    return runs_df.iloc[0]


def test_best_run_meets_f1_threshold(best_run):
    assert best_run['metrics.f1_score'] >= config.F1_THRESHOLD


def test_best_run_meets_precision_floor(best_run):
    assert best_run['metrics.precision'] >= config.PRECISION_FLOOR


def test_best_run_logs_transparency_metrics(best_run):
    # These exist specifically so a passing f1_score can't hide a degenerate
    # model (e.g. high point-adjusted F1 from a near-random detector, or a
    # high false-positive rate on normal data) -- see config.py's F1_THRESHOLD
    # comment for the full history of why this matters here.
    for metric in (
        'metrics.f1_score_point_adjusted',
        'metrics.f1_score_windowed',
        'metrics.false_positive_rate_normal',
        'metrics.recall_real_incidents',
        'metrics.recall_synthetic_incidents',
    ):
        assert metric in best_run.index
        assert best_run[metric] == best_run[metric]  # not NaN
