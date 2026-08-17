import os
import numpy as np
import pandas as pd
from prediction_model.config import config


def load_full_dataset():
    """Load the full labelled dataset.csv built by processing/build_dataset.py
    (real NAB series + injected synthetic anomalies, chronologically ordered).
    """
    filepath = os.path.join(config.DATAPATH, config.DATA_FILE)
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def load_synthetic_easy_dataset():
    """Load synthetic_easy_dataset.csv, built by
    processing/build_synthetic_dataset.py -- a fully synthetic dataset used
    only to validate the promotion gate itself, not real model quality (see
    that module's docstring). Split is trivial (no day-block logic needed):
    the first config.SYNTHETIC_EASY_TRAIN_ROWS rows are anomaly-free by
    construction.
    """
    filepath = os.path.join(config.DATAPATH, config.SYNTHETIC_EASY_DATA_FILE)
    dataset = pd.read_csv(filepath, parse_dates=["timestamp"])
    train_df = dataset.iloc[:config.SYNTHETIC_EASY_TRAIN_ROWS].reset_index(drop=True)
    eval_df = dataset.iloc[config.SYNTHETIC_EASY_TRAIN_ROWS:].reset_index(drop=True)
    return train_df, eval_df


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
