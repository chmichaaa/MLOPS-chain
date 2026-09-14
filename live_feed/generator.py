"""Telemetry collector for the live monitoring view.

Emits a continuous stream of host/load-balancer/database metrics, scores each
reading through the deployed anomaly-detection model, and records the reading
alongside the verdict for the console's /live page.

Scoring goes over HTTP to the serving app's /prediction_api rather than
importing predict.py in-process: that exercises the deployed path end to end
-- the app container, whichever model version it currently has loaded from the
MLflow registry, and the same interface any other client uses -- and it keeps
feeding the Prometheus request metrics the Grafana dashboards are built on.

## Signal model
Metrics are driven by a shared load factor plus a per-metric idiosyncratic
component, both AR(1) processes, so values drift smoothly and move together
the way real host metrics do rather than jumping independently each sample.
Both processes are unit-variance and stationary, so each metric's marginal
distribution stays N(mean, std) from config.EASY_DATA_METRIC_PARAMS -- the
same ranges the serving model's scaler was fitted on. That matters: a stream
with its own arbitrary ranges would sit outside the model's learned-normal
region and be flagged on every reading, which would tell you nothing about
whether detection works.

SMOOTHING (the AR(1) coefficient) is deliberately moderate and configurable.
The model is fitted on data with little temporal correlation, so a very
smooth stream is temporally *unfamiliar* to it even when the value range is
right, and can raise reconstruction error on healthy traffic. If normal
traffic starts drawing false positives, lower LIVE_FEED_SMOOTHING.
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

# AR(1) coefficient for both the shared and per-metric components: 0 is
# independent samples, approaching 1 is a slow wander. See the module
# docstring for why this is capped rather than pushed higher.
SMOOTHING = float(os.environ.get("LIVE_FEED_SMOOTHING", "0.6"))

# How strongly each metric follows the shared load factor versus its own
# noise. Request volume and network throughput track load most closely; the
# database is the most loosely coupled, since it absorbs load through its own
# caching and connection pooling before CPU moves.
LOAD_COUPLING = {
    "elb_request_count": 0.85,
    "network_in_bytes": 0.80,
    "cpu_usage_pct": 0.75,
    "rds_cpu_usage_pct": 0.55,
}

# Daily traffic curve, compressed so the baseline visibly breathes on a
# dashboard someone watches for a couple of minutes rather than a full day.
# Amplitude is held well under one standard deviation so ordinary daily
# variation never reads as an anomaly.
CYCLE_PERIOD_SECONDS = float(os.environ.get("LIVE_FEED_CYCLE_PERIOD_SECONDS", "600"))
CYCLE_AMPLITUDE_STD = 0.45
NOISE_AMPLITUDE_STD = 0.85

# Incident cadence. Deterministic rather than random so the page always shows
# activity within a known window instead of depending on a coin flip.
INCIDENT_EVERY_TICKS = int(os.environ.get("LIVE_FEED_INCIDENT_EVERY_TICKS", "40"))
INCIDENT_LENGTH_TICKS = int(os.environ.get("LIVE_FEED_INCIDENT_LENGTH_TICKS", "14"))
# Onset and recovery are gradual, not a step change: real degradations ramp as
# load builds and drain as it clears, which is also the multi-sample regime
# shift this project's model is built to pick up.
INCIDENT_RAMP_TICKS = 3

RETENTION_HOURS = float(os.environ.get("LIVE_FEED_RETENTION_HOURS", "24"))
PRUNE_EVERY_TICKS = 200

REQUEST_TIMEOUT_SECONDS = 10

# A percentage metric must never render as 118% on an operations dashboard,
# however far the underlying shift pushes it.
BOUNDS = {
    "cpu_usage_pct": (0.0, 100.0),
    "rds_cpu_usage_pct": (0.0, 100.0),
    "network_in_bytes": (0.0, None),
    "elb_request_count": (0.0, None),
}


@dataclass(frozen=True)
class Scenario:
    """shifts: metric -> offset in standard deviations at full intensity."""
    name: str
    label: str
    shifts: dict = field(default_factory=dict)


# Correlated, plausible degradation modes rather than one metric moving alone:
# real incidents move several signals together, which is the multivariate
# shape the detector is meant to pick up.
SCENARIOS = (
    Scenario(
        "traffic_surge", "Traffic surge",
        {"elb_request_count": 7.0, "cpu_usage_pct": 6.0, "network_in_bytes": 7.0, "rds_cpu_usage_pct": 3.5},
    ),
    Scenario(
        "cpu_saturation", "CPU saturation",
        {"cpu_usage_pct": 8.0, "rds_cpu_usage_pct": 1.0},
    ),
    Scenario(
        # Throughput falls while the database saturates -- the one mode where
        # signals move in opposite directions, which a detector keyed only on
        # "values went up" would miss entirely.
        "db_contention", "Database contention",
        {"rds_cpu_usage_pct": 8.0, "cpu_usage_pct": 4.0, "elb_request_count": -3.0, "network_in_bytes": -3.0},
    ),
    Scenario(
        "network_saturation", "Network saturation",
        {"network_in_bytes": 8.0, "cpu_usage_pct": 3.0},
    ),
)

SCENARIO_LABELS = {scenario.name: scenario.label for scenario in SCENARIOS}


class _Ar1:
    """Unit-variance, zero-mean AR(1) process.

    The innovation is scaled by sqrt(1 - rho^2) so the stationary variance
    stays exactly 1 whatever rho is -- that's what lets SMOOTHING change how
    the stream *looks* without changing the distribution the model is scored
    against.
    """

    def __init__(self, rho, rng):
        self.rho = rho
        self.rng = rng
        self.value = float(rng.normal())

    def step(self):
        innovation = math.sqrt(max(0.0, 1.0 - self.rho ** 2)) * self.rng.normal()
        self.value = self.rho * self.value + innovation
        return self.value


def incident_at(tick):
    """The incident state for a tick: (scenario, intensity), where intensity
    ramps 0 -> 1 -> 0 across the incident. (None, 0.0) during healthy traffic.
    """
    if INCIDENT_LENGTH_TICKS <= 0 or INCIDENT_EVERY_TICKS <= 0:
        return None, 0.0
    phase = tick % INCIDENT_EVERY_TICKS
    start = INCIDENT_EVERY_TICKS - INCIDENT_LENGTH_TICKS
    if phase < start:
        return None, 0.0

    scenario = SCENARIOS[(tick // INCIDENT_EVERY_TICKS) % len(SCENARIOS)]
    position = phase - start
    ramp = min(INCIDENT_RAMP_TICKS, max(1, INCIDENT_LENGTH_TICKS // 2))
    if position < ramp:
        intensity = (position + 1) / ramp
    elif position >= INCIDENT_LENGTH_TICKS - ramp:
        intensity = (INCIDENT_LENGTH_TICKS - position) / ramp
    else:
        intensity = 1.0
    return scenario, min(1.0, max(0.0, intensity))


def _clamp(value, metric):
    low, high = BOUNDS.get(metric, (None, None))
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


class MetricStream:
    """Produces successive readings. Holds the AR(1) state, so readings are a
    continuous trajectory rather than independent draws.
    """

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.load = _Ar1(SMOOTHING, self.rng)
        self.idiosyncratic = {metric: _Ar1(SMOOTHING, self.rng) for metric in config.METRIC_COLUMNS}

    def next_reading(self, scenario, intensity, elapsed_seconds):
        load = self.load.step()
        cycle = math.sin(2 * math.pi * elapsed_seconds / CYCLE_PERIOD_SECONDS)

        reading = {}
        for metric in config.METRIC_COLUMNS:
            params = config.EASY_DATA_METRIC_PARAMS[metric]
            coupling = LOAD_COUPLING[metric]
            # Weighted so the combination stays unit variance: the metric
            # follows the shared load factor without inflating its own spread.
            combined = coupling * load + math.sqrt(max(0.0, 1.0 - coupling ** 2)) * self.idiosyncratic[metric].step()

            offset = CYCLE_AMPLITUDE_STD * cycle + NOISE_AMPLITUDE_STD * combined
            if scenario is not None:
                offset += scenario.shifts.get(metric, 0.0) * intensity

            value = params["mean"] + params["std"] * offset
            reading[metric] = round(float(_clamp(value, metric)), 2)
        return reading


def score(window):
    """Scores the most recent reading in `window` through the serving app.
    Returns None rather than raising when the app is unreachable or has no
    model in Production yet: the collector must keep running and keep
    recording, so the page shows a gap instead of the stream dying.
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
    stream = MetricStream()
    window = deque(maxlen=config.WINDOW_SIZE)
    started = time.time()
    tick = 0

    print(
        f"live_feed: posting to {PREDICTION_API_URL} every {TICK_SECONDS}s",
        flush=True,
    )

    while True:
        scenario, intensity = incident_at(tick)
        reading = stream.next_reading(scenario, intensity, time.time() - started)
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
