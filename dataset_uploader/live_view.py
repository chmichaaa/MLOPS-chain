"""The console's /live page: markup, page-specific CSS, and the client-side
rendering for the live monitoring feed.

Kept out of app.py because it's a sizeable blob of CSS/JS with no routing or
permission logic in it, and because none of it needs server-side
interpolation -- the page ships static and pulls everything from /live/data,
which is also what lets it refresh in place instead of reloading.

Charts are hand-rolled SVG rather than a charting library: the console
already renders its own HTML without a template engine or JS framework, the
shapes needed here are polylines and rects, and this keeps the page free of
CDN dependencies and matching the existing design tokens exactly.
"""

LIVE_CSS = """
.live-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: var(--text-muted); }
.live-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--good); display: inline-block; box-shadow: 0 0 0 0 var(--good); animation: live-pulse 2s infinite; flex-shrink: 0; }
.live-dot.stale { background: var(--bad); animation: none; }
@keyframes live-pulse {
  0% { box-shadow: 0 0 0 0 rgba(28, 138, 90, 0.5); }
  70% { box-shadow: 0 0 0 7px rgba(28, 138, 90, 0); }
  100% { box-shadow: 0 0 0 0 rgba(28, 138, 90, 0); }
}
@media (prefers-reduced-motion: reduce) { .live-dot { animation: none; } }
.banner { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1rem; padding: 1rem 1.25rem; border-radius: var(--radius); border: 1px solid var(--border); margin-bottom: 1.5rem; }
.banner-title { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; }
.banner-sub { font-size: 0.85rem; opacity: 0.85; }
.banner.ok { border-color: var(--good); background: var(--good-bg); color: var(--good); }
.banner.alert { border-color: var(--bad); background: var(--bad-bg); color: var(--bad); }
.banner.idle { color: var(--text-muted); background: var(--surface-2); }
.tile-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.95rem 1.1rem; box-shadow: var(--shadow); min-width: 0; }
.tile-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); font-weight: 600; }
.tile-value { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.35rem; font-weight: 600; margin-top: 0.15rem; overflow-wrap: anywhere; }
.spark { width: 100%; height: 42px; margin-top: 0.5rem; display: block; }
.spark polyline { fill: none; stroke: var(--accent); stroke-width: 1.6; vector-effect: non-scaling-stroke; }
.chart { width: 100%; height: 240px; display: block; }
.chart .series { fill: none; stroke: var(--accent); stroke-width: 1.8; vector-effect: non-scaling-stroke; }
.chart .zero { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 4 4; vector-effect: non-scaling-stroke; opacity: 0.7; }
.chart .band { fill: var(--bad); opacity: 0.12; }
.chart .hit { fill: var(--bad); }
.legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.25rem; margin-top: 0.9rem; font-size: 0.78rem; color: var(--text-muted); }
.legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; flex-shrink: 0; }
.swatch.series { background: var(--accent); }
.swatch.band { background: var(--bad); opacity: 0.35; }
.swatch.hit { background: var(--bad); border-radius: 50%; }
.empty-state { color: var(--text-muted); font-size: 0.9rem; }
"""

LIVE_BODY = """
<h1>Live monitoring</h1>

<div class="live-strip">
  <span><span class="live-dot" id="live-dot"></span> <span id="live-state">connecting…</span></span>
  <span id="live-model"></span>
  <span id="live-rate"></span>
</div>

<div class="banner idle" id="banner">
  <span class="banner-title" id="banner-title">Waiting for data…</span>
  <span class="banner-sub" id="banner-sub"></span>
</div>

<div class="tile-grid" id="tiles"></div>

<div class="panel">
  <div class="panel-header">
    <h2>Anomaly score</h2>
    <span class="text-muted" style="font-size:0.8rem;" id="chart-range"></span>
  </div>
  <svg class="chart" id="chart" viewBox="0 0 1000 240" preserveAspectRatio="none" role="img"
       aria-label="Anomaly score over time"></svg>
  <div class="legend">
    <span><i class="swatch series"></i> anomaly score</span>
    <span><i class="swatch band"></i> injected incident (ground truth)</span>
    <span><i class="swatch hit"></i> model flagged an anomaly</span>
    <span>below the dashed line = anomalous</span>
  </div>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>Injected incidents vs. detection</h2>
    <span class="text-muted" style="font-size:0.8rem;">ground truth from the feed, verdicts from the deployed model</span>
  </div>
  <div class="table-wrap">
    <table>
      <tr><th>Incident</th><th>Started</th><th>Readings</th><th>Detected</th><th>Detection latency</th></tr>
      <tbody id="incident-rows">
        <tr><td colspan="5" class="empty-state" style="text-align:center;">No incidents recorded yet.</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
(function () {
  var REFRESH_MS = 3000;
  var LABELS = {
    cpu_usage_pct: 'EC2 CPU',
    network_in_bytes: 'Network in',
    elb_request_count: 'ELB requests',
    rds_cpu_usage_pct: 'RDS CPU'
  };
  var METRICS = ['cpu_usage_pct', 'network_in_bytes', 'elb_request_count', 'rds_cpu_usage_pct'];

  function formatValue(metric, value) {
    if (value === null || value === undefined) { return '—'; }
    if (metric === 'network_in_bytes') { return (value / 1e6).toFixed(2) + ' MB'; }
    if (metric === 'elb_request_count') { return Math.round(value).toLocaleString(); }
    return value.toFixed(1) + '%';
  }

  function formatTime(iso) {
    if (!iso) { return '—'; }
    var d = new Date(iso);
    return d.toLocaleTimeString();
  }

  function el(tag, attrs, text) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var key in attrs) { node.setAttribute(key, attrs[key]); }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function sparkline(values) {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'spark');
    svg.setAttribute('viewBox', '0 0 200 42');
    svg.setAttribute('preserveAspectRatio', 'none');
    if (!values.length) { return svg; }
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = (max - min) || 1;
    var points = values.map(function (v, i) {
      var x = (i / Math.max(values.length - 1, 1)) * 200;
      var y = 39 - ((v - min) / span) * 36;
      return x.toFixed(1) + ',' + y.toFixed(1);
    }).join(' ');
    svg.appendChild(el('polyline', { points: points }));
    return svg;
  }

  function renderTiles(readings) {
    var host = document.getElementById('tiles');
    host.textContent = '';
    var latest = readings.length ? readings[readings.length - 1] : null;
    METRICS.forEach(function (metric) {
      var tile = document.createElement('div');
      tile.className = 'tile';

      var label = document.createElement('div');
      label.className = 'tile-label';
      label.textContent = LABELS[metric];

      var value = document.createElement('div');
      value.className = 'tile-value';
      value.textContent = latest ? formatValue(metric, latest[metric]) : '—';

      tile.appendChild(label);
      tile.appendChild(value);
      tile.appendChild(sparkline(readings.map(function (r) { return r[metric]; })));
      host.appendChild(tile);
    });
  }

  function renderChart(readings) {
    var svg = document.getElementById('chart');
    svg.textContent = '';
    var W = 1000, H = 240, TOP = 14, BOTTOM = 14;
    var n = readings.length;
    if (!n) { return; }

    var xOf = function (i) { return (i / Math.max(n - 1, 1)) * W; };
    var scored = readings.filter(function (r) { return r.anomaly_score !== null; });
    if (!scored.length) { return; }

    var values = scored.map(function (r) { return r.anomaly_score; });
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    // Always keep 0 in frame: it is the inlier/outlier boundary, so a chart
    // that scrolled it off would hide the only reference point that matters.
    min = Math.min(min, 0); max = Math.max(max, 0);
    var span = (max - min) || 1;
    var yOf = function (v) { return TOP + (1 - (v - min) / span) * (H - TOP - BOTTOM); };

    // Ground-truth incident bands first, so the series draws over them.
    var start = null;
    for (var i = 0; i <= n; i++) {
      var name = i < n ? readings[i].incident : null;
      if (name !== null && start === null) { start = i; }
      var ends = (name === null) || (start !== null && i < n && readings[i].incident !== readings[start].incident);
      if (start !== null && ends) {
        var x0 = xOf(start), x1 = xOf(Math.max(i - 1, start));
        svg.appendChild(el('rect', {
          class: 'band', x: x0, y: 0, width: Math.max(x1 - x0, 2), height: H
        }));
        start = (name !== null) ? i : null;
      }
    }

    svg.appendChild(el('line', { class: 'zero', x1: 0, y1: yOf(0), x2: W, y2: yOf(0) }));

    // Draw contiguous scored runs as separate polylines so a gap (app down,
    // or no Production model) reads as a gap instead of a straight line
    // bridging across missing time.
    var segment = [];
    var flush = function () {
      if (segment.length > 1) {
        svg.appendChild(el('polyline', { class: 'series', points: segment.join(' ') }));
      }
      segment = [];
    };
    readings.forEach(function (r, i) {
      if (r.anomaly_score === null) { flush(); return; }
      segment.push(xOf(i).toFixed(1) + ',' + yOf(r.anomaly_score).toFixed(1));
    });
    flush();

    readings.forEach(function (r, i) {
      if (r.is_anomaly && r.anomaly_score !== null) {
        svg.appendChild(el('circle', { class: 'hit', cx: xOf(i), cy: yOf(r.anomaly_score), r: 3 }));
      }
    });
  }

  function renderIncidents(incidents) {
    var body = document.getElementById('incident-rows');
    body.textContent = '';
    if (!incidents.length) {
      var row = document.createElement('tr');
      var cell = document.createElement('td');
      cell.colSpan = 5;
      cell.className = 'empty-state';
      cell.style.textAlign = 'center';
      cell.textContent = 'No incidents recorded yet.';
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    incidents.slice().reverse().forEach(function (incident) {
      var row = document.createElement('tr');
      var cells = [
        incident.label,
        formatTime(incident.started),
        String(incident.readings),
        incident.detected ? 'Detected' : 'Missed',
        incident.detection_latency_seconds === null ? '—' : incident.detection_latency_seconds.toFixed(1) + 's'
      ];
      cells.forEach(function (text, index) {
        var cell = document.createElement('td');
        if (index === 3) {
          var pill = document.createElement('span');
          pill.className = 'pill ' + (incident.detected ? 'pill-good' : 'pill-bad');
          pill.textContent = text;
          cell.appendChild(pill);
        } else {
          if (index !== 0) { cell.className = 'mono'; }
          cell.textContent = text;
        }
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  function renderStatus(payload) {
    var dot = document.getElementById('live-dot');
    var state = document.getElementById('live-state');
    var banner = document.getElementById('banner');
    var title = document.getElementById('banner-title');
    var sub = document.getElementById('banner-sub');

    document.getElementById('live-model').textContent = payload.model.version
      ? 'Model v' + payload.model.version + ' · ' + payload.model.type
      : 'no model in Production';
    document.getElementById('live-rate').textContent = payload.readings.length
      ? payload.anomaly_rate_pct.toFixed(1) + '% of recent readings flagged'
      : '';
    document.getElementById('chart-range').textContent = payload.readings.length
      ? payload.readings.length + ' readings'
      : '';

    dot.className = 'live-dot' + (payload.stale ? ' stale' : '');
    state.textContent = payload.stale
      ? 'feed stale — last reading ' + formatTime(payload.last_seen)
      : 'live · updated ' + formatTime(payload.last_seen);

    if (!payload.readings.length) {
      banner.className = 'banner idle';
      title.textContent = 'Waiting for the live feed…';
      sub.textContent = 'No readings recorded yet. Is the live-feed service running?';
      return;
    }
    if (payload.current_anomaly) {
      banner.className = 'banner alert';
      title.textContent = 'ANOMALY DETECTED';
      sub.textContent = payload.current_incident
        ? 'Model flagged the current reading · injected scenario: ' + payload.current_incident
        : 'Model flagged the current reading';
    } else {
      banner.className = 'banner ok';
      title.textContent = 'SYSTEM NORMAL';
      sub.textContent = payload.current_incident
        ? 'Incident in progress (' + payload.current_incident + ') — not yet flagged'
        : 'All monitored metrics within learned-normal behaviour';
    }
  }

  function refresh() {
    fetch('/live/data', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (payload) {
        if (!payload) { return; }
        renderStatus(payload);
        renderTiles(payload.readings);
        renderChart(payload.readings);
        renderIncidents(payload.incidents);
      })
      .catch(function () {
        document.getElementById('live-dot').className = 'live-dot stale';
        document.getElementById('live-state').textContent = 'lost connection to the console';
      });
  }

  refresh();
  setInterval(refresh, REFRESH_MS);
})();
</script>
"""
