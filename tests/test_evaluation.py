import numpy as np
import pandas as pd

from prediction_model.processing.evaluation import point_adjust, windowed_f1


def test_point_adjust_credits_whole_segment_from_one_hit():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0])  # only the middle point flagged
    adjusted = point_adjust(y_true, y_pred)
    assert adjusted.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_point_adjust_leaves_undetected_segment_untouched():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])  # no hit at all
    adjusted = point_adjust(y_true, y_pred)
    assert adjusted.tolist() == [0, 0, 0, 0]


def test_point_adjust_handles_multiple_segments_independently():
    y_true = np.array([1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 0, 0, 0])  # only the first segment gets a hit
    adjusted = point_adjust(y_true, y_pred)
    assert adjusted.tolist() == [1, 1, 0, 0, 0, 0]


def test_point_adjust_noop_when_no_true_anomalies():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 1, 0, 1])
    adjusted = point_adjust(y_true, y_pred)
    assert adjusted.tolist() == y_pred.tolist()


def _eval_df(labels, dates):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(dates),
        "label": labels,
    })


def test_windowed_f1_perfect_prediction_scores_one():
    dates = ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10", "2024-01-01 00:15",
             "2024-01-02 00:00", "2024-01-02 00:05"]
    labels = np.array([0, 0, 1, 1, 1, 0])
    eval_df = _eval_df(labels, dates)
    assert windowed_f1(eval_df, labels, window=2) == 1.0


def test_windowed_f1_zero_when_nothing_flagged():
    dates = ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:10", "2024-01-01 00:15"]
    labels = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    eval_df = _eval_df(labels, dates)
    assert windowed_f1(eval_df, y_pred, window=2) == 0.0


def test_windowed_f1_does_not_bucket_across_day_boundary():
    # Two consecutive rows straddle midnight; if windowing ignored the day
    # boundary it would bucket them together into one all-anomalous window.
    dates = ["2024-01-01 23:55", "2024-01-02 00:00"]
    labels = np.array([1, 0])
    y_pred = np.array([1, 0])
    eval_df = _eval_df(labels, dates)
    # bucketed separately per day (window=2, but only 1 row per day) -> 2 windows, perfect match
    assert windowed_f1(eval_df, y_pred, window=2) == 1.0
