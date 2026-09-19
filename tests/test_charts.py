"""Unit tests for the console's server-rendered charts
(dataset_uploader/charts.py) -- pure functions, no I/O. Runs in CI's fast
unit_tests job.
"""
from xml.dom import minidom

from dataset_uploader.charts import bucket_min, line_chart, nice_step


def test_nice_step_lands_on_round_numbers():
    # 0..224 ms must grid at 50s, not the 56 / 112 / 168 an even split gives.
    assert nice_step(224) == 50
    assert nice_step(1.0) == 0.25
    assert nice_step(0.4) == 0.1


def test_nice_step_survives_a_zero_span():
    assert nice_step(0) == 1.0


def test_bucket_min_keeps_the_dip_a_mean_would_erase():
    xs = [0, 5, 10, 15, 20, 25]
    ys = [0.10, -0.40, 0.10, 0.10, 0.10, 0.10]
    flags = [False, True, False, False, False, False]
    bx, by, bf = bucket_min(xs, ys, flags, 30)
    assert by == [-0.40]
    assert bf == [True]


def test_bucket_min_turns_missing_stretches_into_gaps():
    # Two readings a whole empty bucket apart must not be joined by a line.
    bx, by, bf = bucket_min([0, 90], [0.1, 0.2], [False, False], 30)
    assert by == [0.1, None, 0.2]


def test_line_chart_emits_well_formed_svg():
    svg = line_chart([{"xs": [0, 60, 120], "ys": [0.3, 0.33, 0.31], "label": "a"}],
                     width=520, threshold=0.2, threshold_label="objective")
    minidom.parseString(svg)
    assert 'viewBox="0 0 520' in svg


def test_line_chart_reports_emptiness_instead_of_drawing_nothing():
    assert "chart-empty" in line_chart([{"xs": [0], "ys": [0.1], "label": "a"}])
    assert "chart-empty" in line_chart([])


def test_line_chart_breaks_the_line_across_a_gap():
    svg = line_chart([{"xs": [0, 30, 60, 90, 120], "ys": [1, 2, None, 2, 1], "label": "a"}])
    assert svg.count("<polyline") == 2
