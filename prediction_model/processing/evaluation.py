"""Evaluation helpers for anomaly detection, shared by training_pipeline.py and
tests. Kept in their own module (rather than inline in training_pipeline.py)
specifically so they're unit-testable without triggering that module's
multi-minute Hyperopt search, which runs at import time.
"""
import numpy as np
from sklearn.metrics import f1_score


def point_adjust(y_true, y_pred):
    """Standard point-adjustment for time-series anomaly evaluation (Xu et al. 2018):
    if any point inside a ground-truth anomaly segment is flagged, the whole
    segment counts as detected. Used for transparency only -- NOT for model
    selection or the F1 threshold, since it can inflate scores on long
    segments independently of real detection quality (Kim et al. 2021).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred).copy()
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if y_pred[i:j].any():
                y_pred[i:j] = 1
            i = j
        else:
            i += 1
    return y_pred


def windowed_f1(eval_df, y_pred, window):
    """F1 over fixed-size, non-overlapping time buckets (a bucket counts as
    detected if ANY point in it is flagged) rather than per-point. Bucketed
    separately within each calendar day so we never window across a day that
    was excluded from eval. Used for transparency only -- NOT for model
    selection or the F1 threshold.

    eval_df must have a 'timestamp' column (used to group by calendar day) and
    a 'label' column (ground truth); y_pred is aligned positionally with eval_df.
    """
    dates = eval_df['timestamp'].dt.date.values
    y_true = eval_df['label'].values
    yt_win, yp_win = [], []
    for day in np.unique(dates):
        day_mask = dates == day
        day_true = y_true[day_mask]
        day_pred = y_pred[day_mask]
        for s in range(0, len(day_true), window):
            yt_win.append(1 if day_true[s:s + window].any() else 0)
            yp_win.append(1 if day_pred[s:s + window].any() else 0)
    return f1_score(yt_win, yp_win, zero_division=0)
