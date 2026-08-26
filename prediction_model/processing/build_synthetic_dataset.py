"""
Builds an easy, fully-synthetic alternative for DATA_FILE -- see
config.py's "Easy demo dataset" section for the full rationale. Run this
instead of build_dataset.py when you want data that clearly clears
F1_THRESHOLD; run build_dataset.py instead for the harder, real-incident-based
case. Both write to the same DATAPATH/DATA_FILE, so whichever ran most
recently is what training_pipeline.py trains against.

Generation, deterministic under config.EASY_DATA_SEED:
- 4 metrics, same names/units as build_dataset.py's output: cpu_usage_pct,
  network_in_bytes, elb_request_count, rds_cpu_usage_pct.
- Normal behaviour: i.i.d. Gaussian noise around a fixed mean/std per metric
  (config.EASY_DATA_METRIC_PARAMS) -- stationary, no diurnal pattern or drift,
  so the "normal" region is trivially learnable.
- config.EASY_DATA_N_EVENTS sustained shift events (config.EASY_DATA_EVENT_LEN
  samples each, magnitude config.EASY_DATA_MAGNITUDE_STD standard deviations,
  random sign per metric), each placed in a distinct day -- same injection
  style as build_dataset.py's inject_synthetic_anomalies, so this data is
  compatible with the same day_block_split and WINDOW_SIZE.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prediction_model.config import config


def build():
    rng = np.random.RandomState(config.EASY_DATA_SEED)

    n = config.EASY_DATA_N_ROWS
    timestamps = pd.date_range(start='2024-01-01', periods=n, freq='5min')

    data = {}
    for metric, params in config.EASY_DATA_METRIC_PARAMS.items():
        data[metric] = rng.normal(loc=params['mean'], scale=params['std'], size=n)
    df = pd.DataFrame(data)
    df.insert(0, 'timestamp', timestamps)
    df['label'] = 0
    df['is_synthetic_anomaly'] = 0

    dates = df['timestamp'].dt.date
    unique_days = dates.unique()
    event_days = rng.choice(unique_days, size=min(config.EASY_DATA_N_EVENTS, len(unique_days)), replace=False)

    stds = df[config.METRIC_COLUMNS].std()
    event_len = config.EASY_DATA_EVENT_LEN
    for day in event_days:
        day_positions = np.where(dates.values == day)[0]
        start = rng.choice(day_positions[:-event_len]) if len(day_positions) > event_len else day_positions[0]
        span = slice(start, min(start + event_len, n))
        direction = rng.choice([1, -1], size=len(config.METRIC_COLUMNS))
        shift = config.EASY_DATA_MAGNITUDE_STD * stds.values * direction
        idx = df.index[span]
        df.loc[idx, config.METRIC_COLUMNS] = df.loc[idx, config.METRIC_COLUMNS].values + shift
        df.loc[idx, 'label'] = 1
        df.loc[idx, 'is_synthetic_anomaly'] = 1

    return df


def run():
    """Builds the easy dataset and writes it to config.DATAPATH/config.DATA_FILE
    -- the same file build_dataset.py writes to. Run whichever generator
    matches the outcome you want to demo next.
    """
    dataset = build()
    os.makedirs(config.DATAPATH, exist_ok=True)
    out_path = os.path.join(config.DATAPATH, config.DATA_FILE)
    dataset.to_csv(out_path, index=False)

    meta_path = os.path.join(config.DATAPATH, "dataset.meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {"uploaded_by": "build_synthetic_dataset.py (local)", "uploaded_at": datetime.now(timezone.utc).isoformat()},
            f,
        )

    n_anomalies = int(dataset['label'].sum())
    print(f"easy demo dataset: {len(dataset)} rows -> {out_path}")
    print(f"  anomaly rows: {n_anomalies} ({dataset['label'].mean():.1%} of total)")
    return out_path


if __name__ == "__main__":
    run()
