import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingWindowFeatures(BaseEstimator, TransformerMixin):
    """Expands each metric column into [raw, rolling_mean].

    Kept signed (not z-scored/absolute) deliberately: several of these AWS
    metrics carry a *directional* anomaly signal -- e.g. a drop in
    rds_cpu_usage_pct or a spike in network_in_bytes correlates with the real
    incidents (verified via per-metric ROC-AUC, up to ~0.78 for single signed
    metrics). Symmetric z-scoring destroys that directionality and empirically
    performed worse for Isolation Forest/OCSVM on this dataset.

    Rolling mean is computed over `window` samples using min_periods=1 so no
    row is dropped (the first `window - 1` rows use whatever history is
    available). X must already be in chronological order.
    """

    def __init__(self, window=12):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        means = X.rolling(window=self.window, min_periods=1).mean()
        return pd.concat([X.add_suffix('_raw'), means.add_suffix('_rolling_mean')], axis=1)
