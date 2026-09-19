"""Unit tests for dataset_uploader's pure logic (dataset_uploader/logic.py) --
no database or MLflow connection needed, unlike dataset_uploader/app.py itself
which does real I/O at import time. Runs in CI's fast unit_tests job alongside
test_evaluation.py/test_preprocessing.py.
"""
from datetime import datetime, timedelta, timezone

from prediction_model.config import config
from dataset_uploader.logic import (
    GMT_PLUS_1,
    format_timestamp,
    format_epoch_millis,
    format_hyperparams,
    source_label,
    stage_css_class,
    missing_dataset_columns,
    now_gmt1,
    role_label,
    validate_username,
    validate_password,
    deletion_error,
    demotion_error,
    summarize_incidents,
)


def test_format_timestamp_renders_short_gmt1_string():
    assert format_timestamp("2026-08-26T19:14:24.394857+01:00") == "2026-08-26 19:14 GMT+1"


def test_format_timestamp_passes_through_unrecognized_values():
    assert format_timestamp("unknown") == "unknown"


def test_source_label_maps_known_values():
    assert source_label("training") == "CI/CD pipeline"
    assert source_label("model_upload") == "Direct upload"


def test_source_label_passes_through_unknown_values():
    assert source_label("something_else") == "something_else"


def test_stage_css_class_mapping():
    assert stage_css_class("Production") == "pill-good"
    assert stage_css_class("Archived") == "pill-neutral"
    assert stage_css_class("None") == "pill-neutral"


def test_missing_dataset_columns_detects_a_real_gap():
    missing = missing_dataset_columns(["timestamp", "label"])
    assert "is_synthetic_anomaly" in missing
    assert "cpu_usage_pct" in missing


def test_missing_dataset_columns_empty_when_schema_is_valid():
    columns = ["timestamp", "label", "is_synthetic_anomaly", *config.METRIC_COLUMNS]
    assert missing_dataset_columns(columns) == set()


def test_now_gmt1_uses_a_fixed_plus_one_offset():
    assert now_gmt1().utcoffset() == GMT_PLUS_1.utcoffset(None)


def test_format_epoch_millis_renders_gmt1_string():
    # 2024-01-01T00:00:00Z in epoch millis -> 01:00 in GMT+1
    assert format_epoch_millis(1704067200000) == "2024-01-01 01:00 GMT+1"


def test_format_epoch_millis_handles_none():
    assert format_epoch_millis(None) == "unknown"


def test_format_hyperparams_excludes_model_type_and_sorts_keys():
    params = {"model_type": "LSTMAutoencoder", "lr": "0.001", "epochs": "20"}
    assert format_hyperparams(params) == "epochs=20, lr=0.001"


def test_format_hyperparams_placeholder_when_nothing_but_model_type():
    assert format_hyperparams({"model_type": "LSTMAutoencoder"}) == "no hyperparameters logged"


# --------------------------------------------------------------------------
# account management guards
# --------------------------------------------------------------------------

def _account(user_id, is_admin=False, username="someone"):
    return {"id": user_id, "username": username, "is_admin": is_admin}


def test_role_label_distinguishes_admins():
    assert role_label(True) == "Administrator"
    assert role_label(False) == "Member"


def test_validate_username_accepts_reasonable_names():
    assert validate_username("chmichaaa") is None
    assert validate_username("ops.team_2") is None


def test_validate_username_rejects_too_short_and_bad_characters():
    assert validate_username("ab") is not None
    assert validate_username("has space") is not None
    assert validate_username("semi;colon") is not None
    assert validate_username("") is not None


def test_validate_password_enforces_minimum_length():
    assert validate_password("longenough") is None
    assert validate_password("short") is not None


def test_cannot_delete_your_own_account():
    assert deletion_error(1, _account(1, is_admin=True), admin_total=2) is not None


def test_cannot_delete_the_last_administrator():
    assert deletion_error(1, _account(2, is_admin=True), admin_total=1) is not None


def test_can_delete_another_admin_when_more_than_one_remains():
    assert deletion_error(1, _account(2, is_admin=True), admin_total=2) is None


def test_can_delete_a_member():
    assert deletion_error(1, _account(2, is_admin=False), admin_total=1) is None


def test_cannot_demote_the_last_administrator():
    assert demotion_error(1, _account(1, is_admin=True), admin_total=1) is not None


def test_demoting_a_member_is_a_noop_not_an_error():
    assert demotion_error(1, _account(2, is_admin=False), admin_total=1) is None


# --------------------------------------------------------------------------
# live feed incident summarization
# --------------------------------------------------------------------------

def _reading(second, incident=None, is_anomaly=False):
    return {
        "t": datetime(2026, 1, 1, 12, 0, second, tzinfo=timezone.utc),
        "incident": incident,
        "is_anomaly": is_anomaly,
    }


def test_summarize_incidents_ignores_normal_readings():
    assert summarize_incidents([_reading(0), _reading(1)]) == []


def test_summarize_incidents_groups_one_contiguous_event():
    readings = [
        _reading(0),
        _reading(1, "traffic_surge"),
        _reading(2, "traffic_surge"),
        _reading(3, "traffic_surge"),
        _reading(4),
    ]
    events = summarize_incidents(readings)
    assert len(events) == 1
    assert events[0]["name"] == "traffic_surge"
    assert events[0]["readings"] == 3
    assert events[0]["detected"] is False
    assert events[0]["detection_latency_seconds"] is None


def test_summarize_incidents_records_detection_and_latency():
    readings = [
        _reading(10, "cpu_runaway"),
        _reading(12, "cpu_runaway", is_anomaly=True),
        _reading(14, "cpu_runaway", is_anomaly=True),
    ]
    event = summarize_incidents(readings)[0]
    assert event["detected"] is True
    # Latency measured from the first injected reading to the FIRST flag, not
    # the last -- 12s minus 10s.
    assert event["detection_latency_seconds"] == 2.0


def test_summarize_incidents_separates_events_split_by_normal_readings():
    readings = [
        _reading(0, "traffic_surge"),
        _reading(1),
        _reading(2, "traffic_surge"),
    ]
    events = summarize_incidents(readings)
    assert len(events) == 2


def test_summarize_incidents_separates_back_to_back_different_scenarios():
    readings = [
        _reading(0, "traffic_surge"),
        _reading(1, "db_contention"),
    ]
    events = summarize_incidents(readings)
    assert [event["name"] for event in events] == ["traffic_surge", "db_contention"]


# --------------------------------------------------------------------------
# overview presentation
# --------------------------------------------------------------------------

from dataset_uploader.logic import format_age, gate_position  # noqa: E402


def test_format_age_picks_the_coarsest_sensible_unit():
    assert format_age(4) == "4s ago"
    assert format_age(60) == "1m ago"
    assert format_age(7200) == "2h ago"
    assert format_age(90000) == "1d ago"


def test_format_age_treats_zero_and_clock_skew_as_just_now():
    # A database clock slightly ahead of the app would otherwise read "-2s ago".
    assert format_age(0) == "just now"
    assert format_age(-3) == "just now"
    assert format_age(None) == "--"


def test_gate_position_marks_a_model_that_clears_the_gate():
    fill, marker, clears = gate_position(0.91, 0.75)
    assert (round(fill), round(marker), clears) == (91, 75, True)


def test_gate_position_marks_a_model_below_the_gate():
    assert gate_position(0.59, 0.75)[2] is False


def test_gate_position_clamps_and_tolerates_bad_input():
    assert gate_position(1.4, 0.75)[0] == 100.0
    assert gate_position(None, 0.75) == (0.0, 75.0, False)
    assert gate_position("bad", 0.75)[0] == 0.0
