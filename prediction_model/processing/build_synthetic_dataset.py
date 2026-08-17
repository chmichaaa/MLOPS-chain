"""
Builds synthetic_easy_dataset.csv -- a fully synthetic, deliberately EASY
anomaly detection dataset.

## What this is NOT
This is not a measurement of real-world model quality. That measurement
already exists: prediction_model/processing/build_dataset.py builds the real
NAB (`realAWSCloudwatch`) dataset, and the honest result on it is documented
in config.py (F1_THRESHOLD = 0.58, pointwise). Nothing about that pipeline or
its results changes here.

## What this IS
A second, transparent dataset built to answer one specific question: does the
promotion gate (train -> evaluate -> assert F1 >= threshold -> log to MLflow)
actually work in both directions -- promoting a good model, blocking a bad
one? Real NAB data can't answer that on its own, because Isolation Forest /
One-Class SVM structurally cannot reach F1 >= 0.85 on it regardless of the
gate's correctness (the real incidents are gradual multivariate regime
shifts, not point outliers -- see config.py's F1_THRESHOLD comment for the
full diagnostic). So this dataset removes that confound: same 4 metrics, same
model classes, same pipeline code, but the anomalies are isolated single-point
spikes (5-10 sigma, one sample each) -- exactly what Isolation Forest's
splitting mechanism is built to isolate. If the gate still doesn't promote a
model here, the gate itself is broken; if it does, the gate works and the
NAB result really is a genuine model/data-fit limitation, not a bug.

## Generation
- 4 metrics, same names/units as the real dataset: cpu_usage_pct,
  network_in_bytes, elb_request_count, rds_cpu_usage_pct.
- Normal behaviour: i.i.d. Gaussian noise around a fixed mean/std per metric
  (config.SYNTHETIC_EASY_METRIC_PARAMS) -- deliberately stationary, no
  diurnal pattern or drift, so the "normal" region is trivially learnable and
  any promotion failure can only be about anomaly separability, not
  non-stationarity (that confound is what the NAB dataset already covers).
- First SYNTHETIC_EASY_TRAIN_ROWS rows are guaranteed anomaly-free (the
  unsupervised training region); anomalies are injected only in the
  remaining eval region.
- SYNTHETIC_EASY_N_ANOMALIES isolated point anomalies: single samples (not
  sustained events -- no gradual drift), each metric shifted independently by
  a random sign and a magnitude drawn from SYNTHETIC_EASY_ANOMALY_SIGMA_RANGE
  standard deviations.
- Labels are exact, strict point labels (1 = this exact sample is anomalous)
  -- no windowing, no point-adjustment, unlike the NAB pipeline's labelled
  incident windows.
- Deterministic under config.SYNTHETIC_DATA_SEED.
"""
import os

import numpy as np
import pandas as pd

from prediction_model.config import config


def build():
    rng = np.random.default_rng(config.SYNTHETIC_DATA_SEED)

    n = config.SYNTHETIC_EASY_N_ROWS
    timestamps = pd.date_range(start='2024-01-01', periods=n, freq='5min')

    data = {}
    for metric, params in config.SYNTHETIC_EASY_METRIC_PARAMS.items():
        data[metric] = rng.normal(loc=params['mean'], scale=params['std'], size=n)
    df = pd.DataFrame(data)
    df.insert(0, 'timestamp', timestamps)
    df['label'] = 0
    df['is_synthetic_anomaly'] = 0  # every anomaly here is synthetic by construction

    eval_start = config.SYNTHETIC_EASY_TRAIN_ROWS
    eval_positions = rng.choice(
        np.arange(eval_start, n), size=config.SYNTHETIC_EASY_N_ANOMALIES, replace=False
    )

    sigma_min, sigma_max = config.SYNTHETIC_EASY_ANOMALY_SIGMA_RANGE
    for pos in eval_positions:
        for metric, params in config.SYNTHETIC_EASY_METRIC_PARAMS.items():
            magnitude = rng.uniform(sigma_min, sigma_max) * params['std']
            sign = rng.choice([-1.0, 1.0])
            df.loc[pos, metric] += sign * magnitude
        df.loc[pos, 'label'] = 1
        df.loc[pos, 'is_synthetic_anomaly'] = 1

    return df


def run():
    dataset = build()
    os.makedirs(config.DATAPATH, exist_ok=True)
    out_path = os.path.join(config.DATAPATH, config.SYNTHETIC_EASY_DATA_FILE)
    dataset.to_csv(out_path, index=False)

    n_anomalies = int(dataset['label'].sum())
    print(f"synthetic_easy dataset: {len(dataset)} rows -> {out_path}")
    print(f"  train region: rows 0-{config.SYNTHETIC_EASY_TRAIN_ROWS - 1} (anomaly-free)")
    print(f"  eval region: rows {config.SYNTHETIC_EASY_TRAIN_ROWS}-{len(dataset) - 1} "
          f"({n_anomalies} isolated point anomalies, "
          f"{n_anomalies / (len(dataset) - config.SYNTHETIC_EASY_TRAIN_ROWS):.1%} of eval region)")
    return out_path


if __name__ == "__main__":
    run()
