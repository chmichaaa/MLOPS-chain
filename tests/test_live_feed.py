"""Unit tests for the telemetry collector's signal generation -- no database,
no HTTP, no model. Runs in CI's fast unit_tests job alongside the other
dependency-free tests.

live_feed/store.py and the scoring call are deliberately not covered here:
both are thin I/O wrappers whose only real behaviour is the database and the
serving app themselves.
"""
import numpy as np

from prediction_model.config import config
from live_feed import generator


def _stream(seed=0):
    return generator.MetricStream(np.random.default_rng(seed))


def test_traffic_is_healthy_at_the_start_of_each_cycle():
    scenario, intensity = generator.incident_at(0)
    assert scenario is None
    assert intensity == 0.0


def test_an_incident_runs_at_the_end_of_each_cycle():
    scenario, intensity = generator.incident_at(generator.INCIDENT_EVERY_TICKS - 1)
    assert scenario is not None
    assert intensity > 0.0


def test_incident_lasts_the_configured_number_of_ticks():
    ticks = [t for t in range(generator.INCIDENT_EVERY_TICKS)
             if generator.incident_at(t)[0] is not None]
    assert len(ticks) == generator.INCIDENT_LENGTH_TICKS


def test_incident_ramps_up_and_back_down():
    every, length = generator.INCIDENT_EVERY_TICKS, generator.INCIDENT_LENGTH_TICKS
    start = every - length
    intensities = [generator.incident_at(t)[1] for t in range(start, every)]
    assert intensities[0] < 1.0            # ramps in rather than stepping
    assert max(intensities) == 1.0         # reaches full magnitude
    assert intensities[-1] < 1.0           # and recovers rather than cutting out


def test_consecutive_cycles_use_different_incident_types():
    every = generator.INCIDENT_EVERY_TICKS
    first = generator.incident_at(every - 1)[0]
    second = generator.incident_at(2 * every - 1)[0]
    assert first.name != second.name


def test_reading_covers_every_monitored_metric():
    reading = _stream().next_reading(None, 0.0, 0.0)
    assert set(reading) == set(config.METRIC_COLUMNS)


def test_healthy_readings_stay_inside_the_learned_normal_range():
    # Healthy traffic must land where the serving model learned "normal" --
    # a stream that drifts outside it would be flagged on every reading and
    # would say nothing about whether detection works.
    stream = _stream()
    for tick in range(400):
        reading = stream.next_reading(None, 0.0, tick * 3.0)
        for metric, params in config.EASY_DATA_METRIC_PARAMS.items():
            deviation = abs(reading[metric] - params["mean"]) / params["std"]
            assert deviation < 5.0, f"{metric} drifted {deviation:.1f} sigma from baseline"


def test_successive_readings_are_correlated_not_independent():
    # The whole point of the AR(1) driver: real telemetry drifts rather than
    # redrawing from scratch each sample. Compare the average step size against
    # what independent draws from the same distribution would give.
    stream = _stream()
    values = [stream.next_reading(None, 0.0, 0.0)["cpu_usage_pct"] for _ in range(400)]
    std = config.EASY_DATA_METRIC_PARAMS["cpu_usage_pct"]["std"]
    mean_step = np.mean(np.abs(np.diff(values)))
    independent_step = np.mean(np.abs(np.diff(np.random.default_rng(1).normal(0, std, 400))))
    assert mean_step < independent_step


def test_metrics_move_together_under_shared_load():
    stream = _stream()
    readings = [stream.next_reading(None, 0.0, 0.0) for _ in range(300)]
    cpu = np.array([r["cpu_usage_pct"] for r in readings])
    requests = np.array([r["elb_request_count"] for r in readings])
    assert np.corrcoef(cpu, requests)[0, 1] > 0.3


def test_incident_shifts_the_metrics_it_names():
    stream = _stream()
    scenario = generator.SCENARIOS[0]  # traffic surge: every signal rises
    reading = stream.next_reading(scenario, 1.0, 0.0)
    for metric, shift in scenario.shifts.items():
        params = config.EASY_DATA_METRIC_PARAMS[metric]
        moved = (reading[metric] - params["mean"]) / params["std"]
        assert moved > shift / 2


def test_partial_intensity_produces_a_smaller_shift_than_full():
    full = _stream().next_reading(generator.SCENARIOS[0], 1.0, 0.0)["cpu_usage_pct"]
    partial = _stream().next_reading(generator.SCENARIOS[0], 0.25, 0.0)["cpu_usage_pct"]
    assert partial < full


def test_percentage_metrics_are_never_physically_impossible():
    stream = _stream()
    for scenario in generator.SCENARIOS:
        for _ in range(50):
            reading = stream.next_reading(scenario, 1.0, 0.0)
            assert 0.0 <= reading["cpu_usage_pct"] <= 100.0
            assert 0.0 <= reading["rds_cpu_usage_pct"] <= 100.0
            assert reading["network_in_bytes"] >= 0.0
            assert reading["elb_request_count"] >= 0.0


def test_every_incident_type_has_a_display_label():
    for scenario in generator.SCENARIOS:
        assert generator.SCENARIO_LABELS[scenario.name] == scenario.label
