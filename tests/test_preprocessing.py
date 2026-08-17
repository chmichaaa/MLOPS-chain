import pandas as pd

from prediction_model.processing.preprocessing import RollingWindowFeatures


def test_output_columns_are_raw_and_rolling_mean_per_metric():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = RollingWindowFeatures(window=2).transform(X)
    assert set(out.columns) == {"a_raw", "b_raw", "a_rolling_mean", "b_rolling_mean"}
    assert len(out) == len(X)


def test_raw_columns_are_unchanged():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = RollingWindowFeatures(window=2).transform(X)
    assert out["a_raw"].tolist() == X["a"].tolist()


def test_rolling_mean_values_and_min_periods_one():
    # window=2, min_periods=1: first row has no history so the mean is just
    # itself; later rows average the trailing 2.
    X = pd.DataFrame({"a": [2.0, 4.0, 6.0, 8.0]})
    out = RollingWindowFeatures(window=2).transform(X)
    assert out["a_rolling_mean"].tolist() == [2.0, 3.0, 5.0, 7.0]


def test_no_rows_dropped_regardless_of_window_size():
    X = pd.DataFrame({"a": range(5)}, dtype=float)
    out = RollingWindowFeatures(window=100).transform(X)
    assert len(out) == 5
    assert not out.isna().any().any()
