"""Live metric simulator: continuously generates realistic server metrics,
scores each one through the REAL deployed model, and records both the reading
and the verdict so the console's /live page can show the system actually
supervising something.

Why it POSTs to the app's /prediction_api instead of importing
predict.generate_predictions directly: the point of this feed is to exercise
the deployed pipeline end to end -- the app container, whichever model it
loaded from the MLflow registry, and the same serving path a real client
uses. An in-process call would demonstrate none of that, and would also
bypass the Prometheus metrics the Grafana dashboard is built on.

## What "realistic" means here, and why it's anchored to config
Baselines come from config.EASY_DATA_METRIC_PARAMS rather than invented
numbers, because the model's MinMaxScaler was fitted on data with those
ranges: a feed with its own arbitrary ranges would sit entirely outside
learned-normal and get flagged as anomalous every single tick, which
demonstrates nothing. For the same reason the diurnal swing is kept small
(DIURNAL_AMPLITUDE_STD below 1 standard deviation) -- build_synthetic_dataset.py
deliberately trains on stationary noise with no daily pattern, so a large
swing here would read as an anomaly rather than as normal daily traffic.

Injected incidents are sustained multi-standard-deviation shifts, the same
shape (and magnitude scale) as build_synthetic_dataset.py's
EASY_DATA_MAGNITUDE_STD injections, so the deployed model has a genuine
chance of catching them -- and the scenario name is recorded alongside every
reading, which is what lets /live grade detections against ground truth
instead of taking the model's word for it.
"""
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import requests

from prediction_model.config import config
from live_feed import store

PREDICTION_API_URL = os.environ.get("PREDICTION_API_URL", "http://app:8005/prediction_api")
TICK_SECONDS = float(os.environ.get("LIVE_FEED_TICK_SECONDS", "3"))

# One full "day" of the diurnal traffic curve, compressed into this many
# seconds of wall clock. A real 24h period would be invisible on a dashboard
# someone watches for two minutes; 10 minutes makes the baseline visibly
# breathe without ever leaving the normal band.
DIURNAL_PERIOD_SECONDS = float(os.environ.get("LIVE_FEED_DIURNAL_PERIOD_SECONDS", "600"))
DIURNAL_AMPLITUDE_STD = 0.8

# Deterministic incident schedule rather than random timing: a demo should
# always show an incident within a known window instead of leaving you
# waiting on a coin flip. Each cycle runs normal first, then the incident.
INCIDENT_EVERY_TICKS = int(os.environ.get("LIVE_FEED_INCIDENT_EVERY_TICKS", "40"))
INCIDENT_LENGTH_TICKS = int(os.environ.get("LIVE_FEED_INCIDENT_LENGTH_TICKS", "14"))

RETENTION_HOURS = float(os.environ.get("LIVE_FEED_RETENTION_HOURS", "24"))
PRUNE_EVERY_TICKS = 200

REQUEST_TIMEOUT_SECONDS = 10

# Percentages are clamped to a physically sensible range -- an 8-sigma shift
# on a metric that is a percentage should still never render as "CPU 118%" on
# a dashboard meant to look like a real monitoring tool.
BOUNDS = {
    "cpu_usage_pct": (0.0, 100.0),
    "rds_cpu_usage_pct": (0.0, 100.0),
    "network_in_bytes": (0.0, None),
    "elb_request_count": (0.0, None),
}


@dataclass(frozen=True)
class Scenario:
    """shifts: metric -> offset in standard deviations of that metric."""
    name: str
    label: str
    shifts: dict = field(default_factory=dict)


# Correlated, plausible failure modes rather than "one metric goes up":
# real incidents move several signals together, which is exactly the
# multivariate shape this project's model is meant to pick up.
SCENARIOS = (
    Scenario(
        "traffic_surge", "Traffic surge",
        {"elb_request_count": 7.0, "cpu_usage_pct": 6.0, "network_in_bytes": 7.0, "rds_cpu_usage_pct": 3.5},
    ),
    Scenario(
        "cpu_runaway", "Runaway process",
        {"cpu_usage_pct": 8.0, "rds_cpu_usage_pct": 1.0},
    ),
    Scenario(
        # Throughput drops while the database saturates -- the one scenario
        # where metrics move in opposite directions, which a detector keyed
        # only on "values went up" would miss.
        "db_contention", "Database contention",
        {"rds_cpu_usage_pct": 8.0, "cpu_usage_pct": 4.0, "elb_request_count": -3.0, "network_in_bytes": -3.0},
    ),
    Scenario(
        "network_flood", "Network flood",
        {"network_in_bytes": 8.0, "cpu_usage_pct": 3.0},
    ),
)

SCENARIO_LABELS = {scenario.name: scenario.label for scenario in SCENARIOS}


def current_scenario(tick):
    """The incident schedule: within each INCIDENT_EVERY_TICKS cycle, the
    last INCIDENT_LENGTH_TICKS ticks are an incident, cycling through
    SCENARIOS in order. Returns None during normal traffic.
    """
    if INCIDENT_LENGTH_TICKS <= 0 or INCIDENT_EVERY_TICKS <= 0:
        return None
    phase = tick % INCIDENT_EVERY_TICKS
    if phase < INCIDENT_EVERY_TICKS - INCIDENT_LENGTH_TICKS:
        return None
    return SCENARIOS[(tick // INCIDENT_EVERY_TICKS) % len(SCENARIOS)]


def _clamp(value, metric):
    low, high = BOUNDS.get(metric, (None, None))
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def generate_reading(rng, scenario, elapsed_seconds):
    """One tick of simulated telemetry: per-metric baseline + diurnal
    modulation + gaussian noise, plus the scenario's sustained shift when an
    incident is in progress.
    """
    diurnal = math.sin(2 * math.pi * elapsed_seconds / DIURNAL_PERIOD_SECONDS)
    reading = {}
    for metric, params in config.EASY_DATA_METRIC_PARAMS.items():
        mean, std = params["mean"], params["std"]
        value = mean + DIURNAL_AMPLITUDE_STD * std * diurnal + rng.normal(0.0, std)
        if scenario is not None:
            value += scenario.shifts.get(metric, 0.0) * std
        reading[metric] = round(float(_clamp(value, metric)), 2)
    return reading


def score(window):
    """Scores the most recent reading in `window` through the deployed app.
    Returns None (rather than raising) if the app is unreachable or has no
    Production model yet -- the feed must keep running and keep recording, so
    the console can show the gap instead of the whole service dying.
    """
    try:
        response = requests.post(
            PREDICTION_API_URL, json={"readings": window}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"live_feed: scoring failed ({exc})", flush=True)
        return None


def run():
    store.init_schema()
    rng = np.random.default_rng()
    window = deque(maxlen=config.WINDOW_SIZE)
    started = time.time()
    tick = 0

    print(
        f"live_feed: posting to {PREDICTION_API_URL} every {TICK_SECONDS}s "
        f"(incident every {INCIDENT_EVERY_TICKS} ticks for {INCIDENT_LENGTH_TICKS} ticks)",
        flush=True,
    )

    while True:
        scenario = current_scenario(tick)
        reading = generate_reading(rng, scenario, time.time() - started)
        window.append(reading)

        verdict = score(list(window))
        try:
            store.insert_reading(reading, verdict, scenario.name if scenario else None)
        except Exception as exc:
            print(f"live_feed: could not store reading ({exc})", flush=True)

        if tick % PRUNE_EVERY_TICKS == 0:
            try:
                store.prune_older_than(RETENTION_HOURS)
            except Exception as exc:
                print(f"live_feed: prune failed ({exc})", flush=True)

        tick += 1
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    run()
