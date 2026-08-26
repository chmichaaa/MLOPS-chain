import json
import os
import numpy as np
import pandas as pd
from prediction_model.config import config

DATASET_META_FILE = 'dataset.meta.json'


def load_full_dataset():
    """Load the labelled dataset.csv, built by either processing/build_dataset.py
    (real NAB series + injected synthetic anomalies) or
    processing/build_synthetic_dataset.py (fully synthetic, easy demo data) --
    whichever ran most recently, chronologically ordered.
    """
    filepath = os.path.join(config.DATAPATH, config.DATA_FILE)
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def load_dataset_metadata():
    """Load dataset.meta.json (who/when produced the current dataset.csv) --
    written by build_dataset.py, build_synthetic_dataset.py, and
    dataset_uploader's dataset-upload path. Defaults if the file is missing
    (e.g. dataset.csv predates this) so callers never have to special-case it.
    """
    filepath = os.path.join(config.DATAPATH, DATASET_META_FILE)
    if not os.path.exists(filepath):
        return {"uploaded_by": "unknown", "uploaded_at": "unknown"}
    with open(filepath) as f:
        return json.load(f)


def day_block_split(data, seed=None):
    """Split into (train_df, eval_df) by whole calendar day, never by row.

    Any day containing an anomaly (real or synthetic) goes entirely to eval --
    the model must never be fit on a labelled anomaly. Remaining clean days are
    split EVAL_CLEAN_FRACTION to eval (so eval also covers genuinely unseen
    normal behaviour, not just incidents) and the rest to train. Splitting by
    whole days keeps both resulting sets temporally contiguous within each day,
    which rolling-window feature computation and windowed evaluation both rely on.
    """
    seed = config.SPLIT_SEED if seed is None else seed
    dates = data['timestamp'].dt.date
    anomaly_days = set(dates[data['label'] == 1].unique())
    clean_days = sorted(set(dates.unique()) - anomaly_days)

    rng = np.random.RandomState(seed)
    shuffled = list(clean_days)
    rng.shuffle(shuffled)
    n_eval_clean = max(1, int(len(shuffled) * config.EVAL_CLEAN_FRACTION)) if shuffled else 0
    eval_clean_days = set(shuffled[:n_eval_clean])
    eval_days = anomaly_days | eval_clean_days

    eval_mask = dates.isin(eval_days)
    train_df = data[~eval_mask].reset_index(drop=True)
    eval_df = data[eval_mask].reset_index(drop=True)
    return train_df, eval_df
