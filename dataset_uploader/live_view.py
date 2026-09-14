"""The console's live telemetry view: markup, page-specific CSS, and the
client-side rendering.

Kept out of app.py because it's a sizeable block of CSS/JS with no routing or
permission logic in it, and because none of it needs server-side
interpolation -- the page ships static and pulls everything from /live/data,
which is what lets it update in place rather than reloading.

Charts are hand-rolled SVG rather than a charting library: the console renders
its own HTML without a template engine or JS framework, the shapes needed here
are polylines, rects and text, and this keeps the page free of CDN
dependencies while matching the existing design tokens exactly.
"""

LIVE_CSS = """
.status-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.75rem 1.5rem; margin-bottom: 1.3rem; }
.status-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.4rem; font-size: 0.82rem; color: var(--text-muted); }
.live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); display: inline-block; margin-right: 0.4rem; animation: live-pulse 2s infinite; flex-shrink: 0; }
.live-dot.stale { background: var(--bad); animation: none; }
@keyframes live-pulse {
  0% { box-shadow: 0 0 0 0 rgba(28, 138, 90, 0.5); }
  70% { box-shadow: 0 0 0 6px rgba(28, 138, 90, 0); }
  100% { box-shadow: 0 0 0 0 rgba(28, 138, 90, 0); }
}
@media (prefers-reduced-motion: reduce) { .live-dot { animation: none; } }
.banner { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1rem; padding: 0.95rem 1.2rem; border-radius: var(--radius); border: 1px solid var(--border); margin-bottom: 1.4rem; }
.banner-title { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1rem; letter-spacing: -0.01em; }
.banner-sub { font-size: 0.85rem; opacity: 0.9; }
.banner.ok { border-color: var(--good); background: var(--good-bg); color: var(--good); }
.banner.alert { border-color: var(--bad); background: var(--bad-bg); color: var(--bad); }
.banner.idle { color: var(--text-muted); background: var(--surface-2); }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.4rem; }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.85rem 1rem; box-shadow: var(--shadow); min-width: 0; }
.kpi-label { font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 600; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.25rem; font-weight: 600; margin-top: 0.2rem; overflow-wrap: anywhere; }
.tile-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 1rem; margin-bottom: 1.4rem; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.9rem 1.05rem; box-shadow: var(--shadow); min-width: 0; }
.tile-head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.5rem; }
.tile-label { font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 600; }
.tile-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--text-muted); white-space: nowrap; }
.tile-delta.up { color: var(--bad); }
.tile-delta.down { color: var(--good); }
.tile-value { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.3rem; font-weight: 600; margin-top: 0.1rem; overflow-wrap: anywhere; }
.spark { width: 100%; height: 40px; margin-top: 0.5rem; display: block; }
.spark polyline { fill: none; stroke: var(--accent); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.spark .spark-fill { fill: var(--accent); opacity: 0.08; stroke: none; }
.chart { width: 100%; height: 250px; display: block; }
.chart .series { fill: none; stroke: var(--accent); stroke-width: 1.8; vector-effect: non-scaling-stroke; }
.chart .grid { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; }
.chart .zero { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 4 4; vector-effect: non-scaling-stroke; opacity: 0.75; }
.chart .band { fill: var(--bad); opacity: 0.10; }
.chart .hit { fill: var(--bad); }
.chart text { fill: var(--text-muted); font-size: 11px; font-family: 'IBM Plex Mono', monospace; }
.legend { display: flex; flex-wrap: wrap; gap: 0.45rem 1.2rem; margin-top: 0.85rem; font-size: 0.76rem; color: var(--text-muted); }
.legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
.swatch { width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex-shrink: 0; }
.swatch.series { background: var(--accent); }
.swatch.band { background: var(--bad); opacity: 0.35; }
.swatch.hit { background: var(--bad); border-radius: 50%; }
.empty-state { color: var(--text-muted); font-size: 0.88rem; }
"""

LIVE_BODY = """
<div class="status-head">
  <div>
    <h1>Live telemetry</h1>
    <div class="status-meta">
      <span><span class="live-dot" id="live-dot"></span><span id="live-state">connecting…</span></span>
      <span id="live-model"></span>
    </div>
  </div>
</div>

<div class="banner idle" id="banner">
  <span class="banner-title" id="banner-title">Awaiting telemetry…</span>
  <span class="banner-sub" id="banner-sub"></span>
</div>

<div class="kpi-row" id="kpis"></div>

<div class="tile-grid" id="tiles"></div>

<div class="panel">
  <div class="panel-header">
    <h2>Anomaly score</h2>
    <span class="text-muted" style="font-size:0.78rem;" id="chart-range"></span>
  </div>
  <svg class="chart" id="chart" viewBox="0 0 1000 250" preserveAspectRatio="none" role="img"
       aria-label="Anomaly score over time"></svg>
  <div class="legend">
    <span><i class="swatch series"></i> anomaly score</span>
    <span><i class="swatch band"></i> incident window</span>
    <span><i class="swatch hit"></i> flagged reading</span>
    <span>below the dashed line = anomalous</span>
  </div>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>Incident log</h2>
    <span class="text-muted" style="font-size:0.78rem;">detection measured against recorded incident windows</span>
  </div>
  <div class="table-wrap">
    <table>
      <tr><th>Type</th><th>Started</th><th>Duration</th><th>Status</th><th>Time to detect</th></tr>
      <tbody id="incident-rows">
        <tr><td colspan="5" class="empty-state" style="text-align:center;">No incidents recorded.</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
(function () {
  var REFRESH_MS = 3000;
  var LABELS = {
    cpu_usage_pct: 'EC2 CPU',
    network_in_bytes: 'Network In',
    elb_request_count: 'ELB Requests',
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
    return new Date(iso).toLocaleTimeString();
  }

  function svgEl(tag, attrs, text) {
    var node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var key in attrs) { node.setAttribute(key, attrs[key]); }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function sparkline(values) {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'spark');
    svg.setAttribute('viewBox', '0 0 200 40');
    svg.setAttribute('preserveAspectRatio', 'none');
    if (!values.length) { return svg; }
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = (max - min) || 1;
    var xy = values.map(function (v, i) {
      var x = (i / Math.max(values.length - 1, 1)) * 200;
      var y = 37 - ((v - min) / span) * 34;
      return [x, y];
    });
    var points = xy.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
    svg.appendChild(svgEl('polygon', {
      'class': 'spark-fill',
      points: '0,40 ' + points + ' 200,40'
    }));
    svg.appendChild(svgEl('polyline', { points: points }));
    return svg;
  }

  function renderKpis(payload) {
    var host = document.getElementById('kpis');
    host.textContent = '';
    var detected = payload.incidents.filter(function (i) { return i.detected; }).length;
    var latencies = payload.incidents
      .filter(function (i) { return i.detection_latency_seconds !== null; })
      .map(function (i) { return i.detection_latency_seconds; });
    var meanLatency = latencies.length
      ? (latencies.reduce(function (a, b) { return a + b; }, 0) / latencies.length).toFixed(1) + 's'
      : '—';

    var items = [
      ['Readings in window', String(payload.readings.length)],
      ['Flagged', payload.anomaly_rate_pct.toFixed(1) + '%'],
      ['Incidents', String(payload.incidents.length)],
      ['Detected', payload.incidents.length ? detected + ' / ' + payload.incidents.length : '—'],
      ['Mean time to detect', meanLatency]
    ];
    items.forEach(function (item) {
      var card = document.createElement('div');
      card.className = 'kpi';
      var label = document.createElement('div');
      label.className = 'kpi-label';
      label.textContent = item[0];
      var value = document.createElement('div');
      value.className = 'kpi-value';
      value.textContent = item[1];
      card.appendChild(label);
      card.appendChild(value);
      host.appendChild(card);
    });
  }

  function renderTiles(readings) {
    var host = document.getElementById('tiles');
    host.textContent = '';
    var latest = readings.length ? readings[readings.length - 1] : null;

    METRICS.forEach(function (metric) {
      var values = readings.map(function (r) { return r[metric]; });
      var tile = document.createElement('div');
      tile.className = 'tile';

      var head = document.createElement('div');
      head.className = 'tile-head';
      var label = document.createElement('div');
      label.className = 'tile-label';
      label.textContent = LABELS[metric];
      head.appendChild(label);

      // Trend against the mean of the preceding window, so the number says
      // "where are we relative to recent normal" rather than reacting to the
      // jitter between two adjacent samples.
      if (values.length > 10) {
        var recent = values.slice(-5);
        var before = values.slice(0, -5);
        var avg = function (a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; };
        var change = avg(before) === 0 ? 0 : ((avg(recent) - avg(before)) / Math.abs(avg(before))) * 100;
        var delta = document.createElement('span');
        delta.className = 'tile-delta ' + (change > 1 ? 'up' : (change < -1 ? 'down' : ''));
        delta.textContent = (change >= 0 ? '▲ ' : '▼ ') + Math.abs(change).toFixed(1) + '%';
        head.appendChild(delta);
      }

      var value = document.createElement('div');
      value.className = 'tile-value';
      value.textContent = latest ? formatValue(metric, latest[metric]) : '—';

      tile.appendChild(head);
      tile.appendChild(value);
      tile.appendChild(sparkline(values));
      host.appendChild(tile);
    });
  }

  function renderChart(readings) {
    var svg = document.getElementById('chart');
    svg.textContent = '';
    var W = 1000, H = 250, TOP = 16, BOTTOM = 26, LEFT = 46;
    var n = readings.length;
    if (!n) { return; }

    var plotW = W - LEFT;
    var xOf = function (i) { return LEFT + (i / Math.max(n - 1, 1)) * plotW; };
    var scored = readings.filter(function (r) { return r.anomaly_score !== null; });
    if (!scored.length) { return; }

    var values = scored.map(function (r) { return r.anomaly_score; });
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    // Keep 0 in frame: it is the inlier/outlier boundary, so a chart that
    // scrolled it off would hide the only reference point that matters.
    min = Math.min(min, 0); max = Math.max(max, 0);
    var pad = (max - min) * 0.12 || 0.01;
    min -= pad; max += pad;
    var span = (max - min) || 1;
    var yOf = function (v) { return TOP + (1 - (v - min) / span) * (H - TOP - BOTTOM); };

    // horizontal gridlines + y axis labels
    for (var g = 0; g <= 3; g++) {
      var v = min + (span * g) / 3;
      var y = yOf(v);
      svg.appendChild(svgEl('line', { 'class': 'grid', x1: LEFT, y1: y, x2: W, y2: y }));
      svg.appendChild(svgEl('text', { x: LEFT - 8, y: y + 3.5, 'text-anchor': 'end' }, v.toFixed(3)));
    }

    // incident windows behind the series
    var start = null;
    for (var i = 0; i <= n; i++) {
      var name = i < n ? readings[i].incident : null;
      var ends = (name === null) || (start !== null && readings[i] && readings[i].incident !== readings[start].incident);
      if (name !== null && start === null) { start = i; ends = false; }
      if (start !== null && ends) {
        var x0 = xOf(start), x1 = xOf(Math.max(i - 1, start));
        svg.appendChild(svgEl('rect', {
          'class': 'band', x: x0, y: TOP, width: Math.max(x1 - x0, 2), height: H - TOP - BOTTOM
        }));
        start = (name !== null) ? i : null;
      }
    }

    svg.appendChild(svgEl('line', { 'class': 'zero', x1: LEFT, y1: yOf(0), x2: W, y2: yOf(0) }));

    // Contiguous scored runs as separate polylines, so a gap (app down, or no
    // model in Production) reads as a gap rather than a line bridging across
    // time that was never measured.
    var segment = [];
    var flush = function () {
      if (segment.length > 1) {
        svg.appendChild(svgEl('polyline', { 'class': 'series', points: segment.join(' ') }));
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
        svg.appendChild(svgEl('circle', { 'class': 'hit', cx: xOf(i), cy: yOf(r.anomaly_score), r: 2.8 }));
      }
    });

    // time axis: first, middle, last
    [0, Math.floor((n - 1) / 2), n - 1].forEach(function (i, pos) {
      svg.appendChild(svgEl('text', {
        x: xOf(i), y: H - 8,
        'text-anchor': pos === 0 ? 'start' : (pos === 2 ? 'end' : 'middle')
      }, formatTime(readings[i].t)));
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
      cell.textContent = 'No incidents recorded.';
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    incidents.slice().reverse().forEach(function (incident) {
      var row = document.createElement('tr');
      var cells = [
        incident.label,
        formatTime(incident.started),
        incident.duration_seconds !== null ? Math.round(incident.duration_seconds) + 's' : '—',
        incident.detected ? 'Detected' : 'Undetected',
        incident.detection_latency_seconds === null ? '—' : incident.detection_latency_seconds.toFixed(1) + 's'
      ];
      cells.forEach(function (text, index) {
        var cell = document.createElement('td');
        if (index === 3) {
          var pill = document.createElement('span');
          pill.className = 'pill ' + (incident.detected ? 'pill-good' : 'pill-warn');
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
      ? 'Scoring model: v' + payload.model.version + ' · ' + payload.model.type
      : 'No model in Production';
    document.getElementById('chart-range').textContent = payload.readings.length
      ? payload.readings.length + ' readings'
      : '';

    dot.className = 'live-dot' + (payload.stale ? ' stale' : '');
    state.textContent = payload.stale
      ? 'No data since ' + formatTime(payload.last_seen)
      : 'Streaming · ' + formatTime(payload.last_seen);

    if (!payload.readings.length) {
      banner.className = 'banner idle';
      title.textContent = 'Awaiting telemetry…';
      sub.textContent = 'No readings received from the collector yet.';
      return;
    }
    if (payload.current_anomaly) {
      banner.className = 'banner alert';
      title.textContent = 'ANOMALY DETECTED';
      sub.textContent = payload.current_incident
        ? 'Active incident: ' + payload.current_incident
        : 'Current reading flagged as anomalous';
    } else {
      banner.className = 'banner ok';
      title.textContent = 'ALL SYSTEMS NORMAL';
      sub.textContent = payload.current_incident
        ? 'Degradation in progress (' + payload.current_incident + ') — not yet flagged'
        : 'All monitored signals within expected behaviour';
    }
  }

  function refresh() {
    fetch('/live/data', { credentials: 'same-origin' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (payload) {
        if (!payload) { return; }
        renderStatus(payload);
        renderKpis(payload);
        renderTiles(payload.readings);
        renderChart(payload.readings);
        renderIncidents(payload.incidents);
      })
      .catch(function () {
        document.getElementById('live-dot').className = 'live-dot stale';
        document.getElementById('live-state').textContent = 'Connection lost';
      });
  }

  refresh();
  setInterval(refresh, REFRESH_MS);
})();
</script>
"""
