"""Unit tests for dataset_uploader's pure logic (dataset_uploader/logic.py) --
no database or MLflow connection needed, unlike dataset_uploader/app.py itself
which does real I/O at import time. Runs in CI's fast unit_tests job alongside
test_evaluation.py/test_preprocessing.py.
"""
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
