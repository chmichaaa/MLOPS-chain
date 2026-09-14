"""Unit tests for the live feed's pure simulation logic -- no database, no
HTTP, no model. Runs in CI's fast unit_tests job alongside the other
dependency-free tests.

live_feed/store.py and the scoring call are deliberately not covered here:
both are thin I/O wrappers whose only real behaviour is the database and the
serving app themselves.
"""
import numpy as np

from prediction_model.config import config
from live_feed import generator


def test_current_scenario_is_none_during_normal_traffic():
    # Each cycle runs normal first, then the incident, so tick 0 is normal.
    assert generator.current_scenario(0) is None


def test_current_scenario_returns_an_incident_at_the_end_of_each_cycle():
    last_tick_of_cycle = generator.INCIDENT_EVERY_TICKS - 1
    assert generator.current_scenario(last_tick_of_cycle) is not None


def test_incident_lasts_the_configured_number_of_ticks():
    every = generator.INCIDENT_EVERY_TICKS
    length = generator.INCIDENT_LENGTH_TICKS
    incident_ticks = [t for t in range(every) if generator.current_scenario(t) is not None]
    assert len(incident_ticks) == length


def test_consecutive_cycles_use_different_scenarios():
    every = generator.INCIDENT_EVERY_TICKS
    first = generator.current_scenario(every - 1)
    second = generator.current_scenario(2 * every - 1)
    assert first.name != second.name


def test_generate_reading_covers_every_monitored_metric():
    rng = np.random.default_rng(0)
    reading = generator.generate_reading(rng, None, elapsed_seconds=0.0)
    assert set(reading) == set(config.METRIC_COLUMNS)


def test_normal_readings_stay_near_the_configured_baseline():
    # Normal traffic must land inside what the model learned as normal --
    # a feed that drifts outside it would be flagged every tick and would
    # demonstrate nothing. Diurnal swing plus noise should stay well within
    # a few standard deviations of the mean.
    rng = np.random.default_rng(0)
    for elapsed in range(0, 600, 25):
        reading = generator.generate_reading(rng, None, elapsed_seconds=float(elapsed))
        for metric, params in config.EASY_DATA_METRIC_PARAMS.items():
            deviation = abs(reading[metric] - params["mean"]) / params["std"]
            assert deviation < 5.0


def test_incident_readings_shift_the_metrics_the_scenario_names():
    rng = np.random.default_rng(0)
    scenario = generator.SCENARIOS[0]  # traffic_surge: every metric shifts up
    reading = generator.generate_reading(rng, scenario, elapsed_seconds=0.0)
    for metric, shift in scenario.shifts.items():
        params = config.EASY_DATA_METRIC_PARAMS[metric]
        moved = (reading[metric] - params["mean"]) / params["std"]
        assert moved > shift / 2


def test_percentage_metrics_are_never_physically_impossible():
    rng = np.random.default_rng(0)
    for scenario in generator.SCENARIOS:
        for _ in range(50):
            reading = generator.generate_reading(rng, scenario, elapsed_seconds=0.0)
            assert 0.0 <= reading["cpu_usage_pct"] <= 100.0
            assert 0.0 <= reading["rds_cpu_usage_pct"] <= 100.0
            assert reading["network_in_bytes"] >= 0.0
            assert reading["elb_request_count"] >= 0.0


def test_every_scenario_has_a_display_label():
    for scenario in generator.SCENARIOS:
        assert generator.SCENARIO_LABELS[scenario.name] == scenario.label
