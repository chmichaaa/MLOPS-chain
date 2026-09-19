"""Server-rendered SVG time-series charts for the console's per-request pages
(Overview, Service metrics).

The live view draws client-side because it redraws every few seconds; these
pages render once per request, so drawing on the server keeps them free of
client JS and of any third-party charting code, and they render identically
in every browser. Charts use a fixed viewBox with the browser's default
uniform scaling, so labels and markers never stretch. Callers pass a `width`
close to the panel the chart sits in: text is sized in viewBox units, so a
chart drawn at 800 and shown at 500 would shrink its labels to illegibility.
"""
import math
from datetime import datetime, timezone
from html import escape

from dataset_uploader.logic import GMT_PLUS_1

PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 58, 16, 14, 28
TARGET_TICKS = 4


def _clock(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone(GMT_PLUS_1).strftime("%H:%M")


def nice_step(span, target=TARGET_TICKS):
    """A gridline spacing from the 1-2-2.5-5 series, so axes read 0 / 50 /
    100 / 150 rather than 0 / 56 / 112 / 168."""
    if span <= 0:
        return 1.0
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if span / step <= target + 0.5:
            return step
    return 10 * magnitude


def bucket_min(xs, ys, flags, bucket_seconds):
    """Collapses points into fixed time buckets, keeping each bucket's minimum.

    For an anomaly score the dip is the signal, so a mean would erase exactly
    what the chart exists to show; the minimum preserves every excursion
    while cutting a dense hour of samples down to something legible. A bucket
    that holds any flagged point stays flagged. Empty stretches become None,
    so they draw as gaps rather than as a line bridging missing time.
    """
    out_x, out_y, out_f = [], [], []
    current, last_key = None, None
    for x, y, flag in zip(xs, ys, flags):
        key = int(x // bucket_seconds)
        if current is not None and key != current[0]:
            out_x.append(current[0] * bucket_seconds + bucket_seconds / 2)
            out_y.append(current[1])
            out_f.append(current[2])
            if key - current[0] > 1:
                out_x.append((current[0] + 1) * bucket_seconds + bucket_seconds / 2)
                out_y.append(None)
                out_f.append(False)
            current = None
        if current is None:
            current = [key, y, bool(flag)]
        else:
            if y is not None and (current[1] is None or y < current[1]):
                current[1] = y
            current[2] = current[2] or bool(flag)
        last_key = key
    if current is not None:
        out_x.append(current[0] * bucket_seconds + bucket_seconds / 2)
        out_y.append(current[1])
        out_f.append(current[2])
    return out_x, out_y, out_f


def _segments(xs, ys):
    """Contiguous runs of present values. A gap stays a gap: bridging it with
    a line would imply measurements that were never taken.
    """
    run, runs = [], []
    for x, y in zip(xs, ys):
        if y is None:
            if run:
                runs.append(run)
            run = []
        else:
            run.append((x, y))
    if run:
        runs.append(run)
    return runs


def line_chart(series, *, width=800, height=200, y_fmt=lambda v: f"{v:.2f}", y_floor=None,
               threshold=None, threshold_label="", bands=(), marks=(), empty_text="No data in range"):
    """series: list of {"xs": [epoch seconds], "ys": [float | None], "label": str}.
    y_floor: pin the bottom of the scale (0 for rates and latencies).
    threshold: a reference value drawn as a dashed line (a gate, an objective).
    bands: (start, end) epoch ranges shaded behind the traces.
    marks: (x, y) points drawn as emphasised dots.
    """
    points = [(x, y) for s in series for x, y in zip(s["xs"], s["ys"]) if y is not None]
    if len(points) < 2:
        return f'<div class="chart-empty">{escape(empty_text)}</div>'

    x_lo = min(p[0] for p in points)
    x_hi = max(p[0] for p in points)
    y_lo = min(p[1] for p in points)
    y_hi = max(p[1] for p in points)
    if threshold is not None:
        y_lo, y_hi = min(y_lo, threshold), max(y_hi, threshold)
    if y_floor is not None:
        y_lo = min(y_lo, y_floor)
    if y_hi == y_lo:
        y_hi = y_lo + (abs(y_lo) or 1.0)
    headroom = (y_hi - y_lo) * 0.08
    y_hi += headroom
    if y_floor is None:
        y_lo -= headroom

    # Snap the domain to whole gridline steps so every gridline is a round
    # number and the top and bottom lines coincide with the plot edges.
    step = nice_step(y_hi - y_lo)
    y_lo = math.floor(y_lo / step + 1e-9) * step
    y_hi = math.ceil(y_hi / step - 1e-9) * step
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or step

    plot_w = width - PAD_LEFT - PAD_RIGHT
    plot_h = height - PAD_TOP - PAD_BOTTOM
    bottom = PAD_TOP + plot_h

    def sx(x):
        return PAD_LEFT + (x - x_lo) / x_span * plot_w

    def sy(y):
        return PAD_TOP + (1 - (y - y_lo) / y_span) * plot_h

    out = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">']
    out.append(f'<rect class="c-plot" x="{PAD_LEFT}" y="{PAD_TOP}" width="{plot_w}" height="{plot_h}"/>')

    tick_count = int(round(y_span / step))
    for i in range(tick_count + 1):
        value = y_lo + step * i
        y = sy(value)
        out.append(f'<line class="c-grid" x1="{PAD_LEFT}" y1="{y:.1f}" x2="{width - PAD_RIGHT}" y2="{y:.1f}"/>')
        out.append(f'<text class="c-label" x="{PAD_LEFT - 9}" y="{y + 3.5:.1f}" text-anchor="end">{escape(y_fmt(value))}</text>')

    for start, end in bands:
        a, b = sx(max(start, x_lo)), sx(min(end, x_hi))
        if b > a - 1:
            out.append(f'<rect class="c-band" x="{a:.1f}" y="{PAD_TOP}" width="{max(b - a, 3):.1f}" height="{plot_h}"/>')

    if threshold is not None:
        y = sy(threshold)
        out.append(f'<line class="c-threshold" x1="{PAD_LEFT}" y1="{y:.1f}" x2="{width - PAD_RIGHT}" y2="{y:.1f}"/>')

    for index, s in enumerate(series):
        tone = f"s{index % 3}"
        for run in _segments(s["xs"], s["ys"]):
            if len(run) < 2:
                continue
            line = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in run)
            if index == 0:
                area = f"{sx(run[0][0]):.1f},{bottom} {line} {sx(run[-1][0]):.1f},{bottom}"
                out.append(f'<polygon class="c-area {tone}" points="{area}"/>')
            out.append(f'<polyline class="c-line {tone}" points="{line}"/>')

    for x, y in marks:
        if x_lo <= x <= x_hi and y is not None:
            out.append(f'<circle class="c-mark" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.8"/>')

    # Drawn last, so the label sits above the traces it annotates.
    if threshold is not None and threshold_label:
        y = sy(threshold)
        out.append(f'<text class="c-threshold-label" x="{PAD_LEFT + 8}" y="{y - 6:.1f}">{escape(threshold_label)}</text>')

    ticks = 4
    for i in range(ticks):
        x = x_lo + x_span * i / (ticks - 1)
        anchor = "start" if i == 0 else ("end" if i == ticks - 1 else "middle")
        out.append(f'<text class="c-label" x="{sx(x):.1f}" y="{height - 8}" text-anchor="{anchor}">{_clock(x)}</text>')

    out.append("</svg>")
    return "".join(out)


def legend(series):
    """A legend is only worth drawing when there is more than one trace."""
    if len(series) < 2:
        return ""
    items = "".join(
        f'<span><i class="c-key s{i % 3}"></i>{escape(s["label"])}</span>' for i, s in enumerate(series)
    )
    return f'<div class="chart-legend">{items}</div>'
