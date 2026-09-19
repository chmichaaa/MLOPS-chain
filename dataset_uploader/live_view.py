"""The console's live telemetry view: markup, page-specific CSS, and the
client-side rendering.

Kept out of app.py because it's a sizeable block of CSS/JS with no routing or
permission logic in it, and because none of it needs server-side
interpolation -- the page ships static and pulls everything from /live/data,
which is what lets it update in place rather than reloading.

The signals are drawn as one stacked panel on a shared time axis -- anomaly
score leading, the four monitored metrics beneath it -- rather than as
separate per-metric cards. Incident windows are drawn as bands spanning the
full height of the stack, so a movement in a metric and the detector's
response to it line up vertically and read as one event. Separate cards put
the same information side by side on unrelated scales, which is precisely
what makes that cause-and-effect reading impossible.

Charts are hand-rolled SVG rather than a charting library: the console
renders its own HTML without a template engine or JS framework, the shapes
needed are polylines, rects and text, and this keeps the page free of CDN
dependencies while matching the design tokens exactly.
"""

LIVE_CSS = """
.status-bar {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: var(--s3) var(--s5); margin-bottom: var(--s4);
  padding: var(--s4) var(--s5); border-radius: var(--r-md);
  border: 1px solid var(--border); background: var(--surface);
  border-left: 3px solid var(--ink-3); box-shadow: var(--shadow);
}
.status-bar.ok { border-left-color: var(--ok); }
.status-bar.alert { border-left-color: var(--crit); }
.status-left { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.status-title { font-family: var(--mono); font-weight: 600; font-size: 14px; letter-spacing: -.01em; }
.status-bar.ok .status-title { color: var(--ok); }
.status-bar.alert .status-title { color: var(--crit); }
.status-sub { font-size: 12.5px; color: var(--ink-2); }
.status-right { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2) var(--s5); font-size: 11.5px; color: var(--ink-3); font-family: var(--mono); }
.live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); display: inline-block; margin-right: var(--s2); animation: live-pulse 2s infinite; flex-shrink: 0; }
.live-dot.stale { background: var(--crit); animation: none; }
@keyframes live-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(21,121,78,.45); }
  70%  { box-shadow: 0 0 0 5px rgba(21,121,78,0); }
  100% { box-shadow: 0 0 0 0 rgba(21,121,78,0); }
}
@media (prefers-reduced-motion: reduce) { .live-dot { animation: none; } }

.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px,1fr)); gap: var(--s3); margin-bottom: var(--s4); }
.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md); padding: var(--s3) var(--s4); box-shadow: var(--shadow); min-width: 0; }
.kpi-label { font-size: 9.5px; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-3); font-weight: 600; font-family: var(--mono); }
.kpi-value { font-size: 18px; font-weight: 600; margin-top: 3px; letter-spacing: -.02em; overflow-wrap: anywhere; }
.kpi-value.crit { color: var(--crit); }

.signal { width: 100%; display: block; min-height: 260px; border-radius: var(--r-sm); }
.signal:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.signal-hint { font-size: 11px; color: var(--ink-3); font-family: var(--sans); margin-top: var(--s2); }
.signal-empty { display: flex; align-items: center; justify-content: center; min-height: 260px; color: var(--ink-3); font-size: 12.5px; }
/* Announced to assistive tech as the cursor moves; the stack itself is a
   graphic, so the readout has to exist as text somewhere. */
.sr-only {
  position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}
.skeleton { color: var(--ink-3); }
.signal .plot-bg { fill: var(--surface-2); opacity: .45; }
.signal .grid { stroke: var(--border); stroke-width: 1; vector-effect: non-scaling-stroke; }
.signal .axis { stroke: var(--border-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
.signal .zero { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 3 3; vector-effect: non-scaling-stroke; opacity: .6; }
.signal .trace { fill: none; stroke: var(--accent); stroke-width: 1.5; vector-effect: non-scaling-stroke; stroke-linejoin: round; stroke-linecap: round; }
.signal .trace-score { stroke-width: 2; }
.signal .fill { fill: var(--accent); opacity: .07; stroke: none; }
.signal .band { fill: var(--crit); opacity: .08; }
.signal .band-edge { stroke: var(--crit); stroke-width: 1; opacity: .3; vector-effect: non-scaling-stroke; }
.signal .hit { fill: var(--crit); }
.signal .row-label { fill: var(--ink-3); font-family: var(--mono); font-size: 9.5px; letter-spacing: .09em; }
.signal .row-value { fill: var(--ink); font-family: var(--mono); font-size: 12px; font-weight: 600; }
.signal .tick { fill: var(--ink-3); font-family: var(--mono); font-size: 9.5px; }
.signal .cursor { stroke: var(--ink-2); stroke-width: 1; vector-effect: non-scaling-stroke; opacity: .45; }
.signal .cursor-dot { fill: var(--accent); stroke: var(--surface); stroke-width: 1.5; }

.legend { display: flex; flex-wrap: wrap; gap: var(--s2) var(--s5); margin-top: var(--s3); padding-top: var(--s3); border-top: 1px solid var(--border); font-size: 11px; color: var(--ink-3); font-family: var(--sans); }
.legend span { display: inline-flex; align-items: center; gap: var(--s2); }
.swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; flex-shrink: 0; }
.swatch.series { background: var(--accent); }
.swatch.band { background: var(--crit); opacity: .35; }
.swatch.hit { background: var(--crit); border-radius: 50%; }
.empty-state { color: var(--ink-3); font-size: 12.5px; }
.empty-state.center { text-align: center; }
"""

LIVE_BODY = """
<h1>Live telemetry</h1>
<p class="page-sub">Continuous signal from the monitored EC2, ELB and RDS surface, scored on arrival by the
model currently in production.</p>

<div class="status-bar" id="status-bar">
  <div class="status-left">
    <span class="status-title" id="status-title">Awaiting telemetry</span>
    <span class="status-sub" id="status-sub">No readings received yet.</span>
  </div>
  <div class="status-right">
    <span><span class="live-dot" id="live-dot"></span><span id="live-state">connecting</span></span>
    <span id="live-model"></span>
  </div>
</div>

<!-- Rendered with placeholder values so the page has its finished shape on
     first paint, before the first poll returns, rather than reflowing in. -->
<div class="kpi-row" id="kpis">
  <div class="kpi"><div class="kpi-label">Readings</div><div class="kpi-value skeleton">--</div></div>
  <div class="kpi"><div class="kpi-label">Flagged</div><div class="kpi-value skeleton">--</div></div>
  <div class="kpi"><div class="kpi-label">Incidents</div><div class="kpi-value skeleton">--</div></div>
  <div class="kpi"><div class="kpi-label">Detected</div><div class="kpi-value skeleton">--</div></div>
  <div class="kpi"><div class="kpi-label">Time to detect</div><div class="kpi-value skeleton">--</div></div>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>Signals</h2>
    <span class="panel-note" id="chart-range"></span>
  </div>
  <div class="signal-empty" id="signal-empty">Waiting for the first readings…</div>
  <svg class="signal" id="signal" tabindex="0" role="img" hidden
       aria-label="Anomaly score and monitored metrics over time. Use arrow keys to read values."></svg>
  <p class="sr-only" id="signal-readout" aria-live="polite"></p>
  <div class="legend">
    <span><i class="swatch series"></i> signal</span>
    <span><i class="swatch band"></i> incident window</span>
    <span><i class="swatch hit"></i> flagged reading</span>
    <span>anomaly score below the dashed line is anomalous</span>
  </div>
  <p class="signal-hint">Hover the stack to inspect a moment, or focus it and use &larr; &rarr; (Home / End to jump, Esc to release).</p>
</div>

<div class="panel">
  <div class="panel-header">
    <h2>Incident log</h2>
    <span class="panel-note">detection measured against recorded incident windows</span>
  </div>
  <div class="table-wrap">
    <table>
      <tr><th>Type</th><th>Started</th><th class="num">Duration</th><th>Status</th><th class="num">Time to detect</th></tr>
      <tbody id="incident-rows">
        <tr><td colspan="5" class="empty-state center">No incidents recorded.</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
(function () {
  var REFRESH_MS = 3000;
  var NS = 'http://www.w3.org/2000/svg';
  var METRICS = ['cpu_usage_pct', 'network_in_bytes', 'elb_request_count', 'rds_cpu_usage_pct'];
  var LABELS = {
    cpu_usage_pct: 'EC2 CPU',
    network_in_bytes: 'NETWORK IN',
    elb_request_count: 'ELB REQ',
    rds_cpu_usage_pct: 'RDS CPU'
  };

  var cursorIndex = null;
  // Kept so a resize can redraw the stack at the new width without waiting
  // for the next poll -- the viewBox is measured, so width changes matter.
  var lastReadings = [];

  function fmt(key, v) {
    if (v === null || v === undefined || isNaN(v)) { return '--'; }
    if (key === 'network_in_bytes') { return (v / 1e6).toFixed(2) + ' MB'; }
    if (key === 'elb_request_count') { return Math.round(v).toLocaleString(); }
    if (key === 'anomaly_score') { return v.toFixed(3); }
    return v.toFixed(1) + '%';
  }

  function clock(iso) { return iso ? new Date(iso).toLocaleTimeString() : '--'; }

  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) { n.setAttribute(k, attrs[k]); }
    if (text !== undefined) { n.textContent = text; }
    return n;
  }

  function renderKpis(p) {
    var host = document.getElementById('kpis');
    host.textContent = '';
    var detected = p.incidents.filter(function (i) { return i.detected; }).length;
    var lat = p.incidents
      .filter(function (i) { return i.detection_latency_seconds !== null; })
      .map(function (i) { return i.detection_latency_seconds; });
    var mttd = lat.length ? (lat.reduce(function (a, b) { return a + b; }, 0) / lat.length).toFixed(1) + 's' : '--';

    [['Readings', String(p.readings.length), false],
     ['Flagged', p.anomaly_rate_pct.toFixed(1) + '%', p.anomaly_rate_pct > 0],
     ['Incidents', String(p.incidents.length), false],
     ['Detected', p.incidents.length ? detected + '/' + p.incidents.length : '--', false],
     ['Time to detect', mttd, false]
    ].forEach(function (item) {
      var card = document.createElement('div');
      card.className = 'kpi';
      var l = document.createElement('div');
      l.className = 'kpi-label';
      l.textContent = item[0];
      var v = document.createElement('div');
      v.className = 'kpi-value' + (item[2] ? ' crit' : '');
      v.textContent = item[1];
      card.appendChild(l); card.appendChild(v);
      host.appendChild(card);
    });
  }

  // One stacked chart on a shared x axis: the anomaly score leads, the raw
  // metrics follow, and incident bands run the full height so a movement in
  // a metric and the detector's response to it read as the same event.
  function renderSignal(readings) {
    var svg = document.getElementById('signal');
    var empty = document.getElementById('signal-empty');
    lastReadings = readings;
    svg.textContent = '';
    var n = readings.length;
    // The attribute, not the .hidden property: .hidden exists only on HTML
    // elements, so on an <svg> assigning it silently does nothing and the
    // markup's hidden attribute would keep the chart hidden for good.
    if (n) { svg.removeAttribute('hidden'); } else { svg.setAttribute('hidden', ''); }
    empty.hidden = !!n;
    if (!n) { return; }

    // The viewBox is measured from the element rather than fixed, so one user
    // unit is one CSS pixel at any width. A fixed viewBox stretched to fit
    // (preserveAspectRatio="none") scales x and y differently and visibly
    // distorts every label and marker in the stack.
    var W = Math.max(520, Math.round(svg.clientWidth || svg.parentNode.clientWidth || 1000));
    var L = 92, R = 78, TOP = 12, AXIS = 22;
    var SCORE_H = 100, ROW_H = 54, GAP = 12;
    var rows = [{ key: 'anomaly_score', label: 'ANOMALY SCORE', h: SCORE_H, score: true }]
      .concat(METRICS.map(function (m) { return { key: m, label: LABELS[m], h: ROW_H }; }));

    var H = TOP + rows.reduce(function (a, r) { return a + r.h + GAP; }, 0) + AXIS;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    svg.setAttribute('height', H);

    var plotW = W - L - R;
    var xOf = function (i) { return L + (i / Math.max(n - 1, 1)) * plotW; };

    // bands first, so every trace draws over them
    var bandTop = TOP, bandBottom = H - AXIS;
    var start = null;
    for (var i = 0; i <= n; i++) {
      var name = i < n ? readings[i].incident : null;
      var changed = start !== null && (name === null || name !== readings[start].incident);
      if (changed) {
        var x0 = xOf(start), x1 = xOf(Math.max(i - 1, start));
        svg.appendChild(el('rect', { 'class': 'band', x: x0, y: bandTop, width: Math.max(x1 - x0, 2), height: bandBottom - bandTop }));
        svg.appendChild(el('line', { 'class': 'band-edge', x1: x0, y1: bandTop, x2: x0, y2: bandBottom }));
        start = null;
      }
      if (name !== null && start === null) { start = i; }
    }

    var readIdx = cursorIndex !== null ? cursorIndex : n - 1;
    var y = TOP;

    rows.forEach(function (row) {
      var vals = readings.map(function (r) { return r[row.key]; });
      var present = vals.filter(function (v) { return v !== null && v !== undefined; });
      var top = y, bot = y + row.h;

      svg.appendChild(el('rect', { 'class': 'plot-bg', x: L, y: top, width: plotW, height: row.h }));
      svg.appendChild(el('text', { 'class': 'row-label', x: 0, y: top + 11 }, row.label));

      if (present.length) {
        var min = Math.min.apply(null, present), max = Math.max.apply(null, present);
        if (row.score) { min = Math.min(min, 0); max = Math.max(max, 0); }
        var pad = (max - min) * 0.18 || Math.abs(max || 1) * 0.1 || 1;
        min -= pad; max += pad;
        var span = (max - min) || 1;
        var yOf = function (v) { return bot - ((v - min) / span) * row.h; };

        svg.appendChild(el('line', { 'class': 'grid', x1: L, y1: top, x2: W - R, y2: top }));
        // only the extremes are labelled: the scale stays readable and the
        // stack does not fill with ticks nobody reads
        svg.appendChild(el('text', { 'class': 'tick', x: W - R + 8, y: top + 8 }, fmt(row.key, max)));
        svg.appendChild(el('text', { 'class': 'tick', x: W - R + 8, y: bot }, fmt(row.key, min)));

        if (row.score) {
          svg.appendChild(el('line', { 'class': 'zero', x1: L, y1: yOf(0), x2: W - R, y2: yOf(0) }));
        }

        // contiguous runs only -- a gap means nothing was scored then, and
        // must not be bridged by a line implying a measurement that never happened
        var seg = [], segs = [];
        readings.forEach(function (r, idx) {
          var v = r[row.key];
          if (v === null || v === undefined) { if (seg.length) { segs.push(seg); seg = []; } return; }
          seg.push([xOf(idx), yOf(v)]);
        });
        if (seg.length) { segs.push(seg); }

        segs.forEach(function (s) {
          if (s.length < 2) { return; }
          var pts = s.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
          svg.appendChild(el('polygon', {
            'class': 'fill',
            points: s[0][0].toFixed(1) + ',' + bot + ' ' + pts + ' ' + s[s.length - 1][0].toFixed(1) + ',' + bot
          }));
          svg.appendChild(el('polyline', { 'class': 'trace' + (row.score ? ' trace-score' : ''), points: pts }));
        });

        if (row.score) {
          readings.forEach(function (r, idx) {
            if (r.is_anomaly && r.anomaly_score !== null && r.anomaly_score !== undefined) {
              svg.appendChild(el('circle', { 'class': 'hit', cx: xOf(idx), cy: yOf(r.anomaly_score), r: 2.6 }));
            }
          });
        }

        // the value at the cursor (or the newest reading) doubles the stack
        // as a numeric readout, so no separate tile row is needed
        var cur = readings[readIdx] ? readings[readIdx][row.key] : null;
        svg.appendChild(el('text', { 'class': 'row-value', x: 0, y: top + 28 }, fmt(row.key, cur)));
        if (cur !== null && cur !== undefined) {
          svg.appendChild(el('circle', { 'class': 'cursor-dot', cx: xOf(readIdx), cy: yOf(cur), r: 3 }));
        }
      }
      y = bot + GAP;
    });

    svg.appendChild(el('line', { 'class': 'axis', x1: L, y1: H - AXIS, x2: W - R, y2: H - AXIS }));
    [0, Math.floor((n - 1) / 2), n - 1].forEach(function (i, pos) {
      svg.appendChild(el('text', {
        'class': 'tick', x: xOf(i), y: H - AXIS + 14,
        'text-anchor': pos === 0 ? 'start' : (pos === 2 ? 'end' : 'middle')
      }, clock(readings[i].t)));
    });

    if (cursorIndex !== null) {
      svg.appendChild(el('line', { 'class': 'cursor', x1: xOf(cursorIndex), y1: TOP, x2: xOf(cursorIndex), y2: H - AXIS }));
    }

    svg.onmousemove = function (evt) {
      var box = svg.getBoundingClientRect();
      var i = Math.round((((evt.clientX - box.left) / box.width * W) - L) / plotW * (n - 1));
      i = Math.max(0, Math.min(n - 1, i));
      if (i !== cursorIndex) { cursorIndex = i; renderSignal(readings); }
    };
    svg.onmouseleave = function () {
      if (cursorIndex !== null) { cursorIndex = null; renderSignal(readings); }
    };
  }

  // The stack is a graphic, so the value at the cursor has to exist as text
  // for anyone not reading it with a pointer.
  function announce(index) {
    var out = document.getElementById('signal-readout');
    if (index === null || !lastReadings[index]) { out.textContent = ''; return; }
    var r = lastReadings[index];
    var parts = [clock(r.t), 'anomaly score ' + fmt('anomaly_score', r.anomaly_score)];
    if (r.is_anomaly) { parts.push('flagged'); }
    if (r.incident) { parts.push('incident in progress'); }
    METRICS.forEach(function (m) { parts.push(LABELS[m] + ' ' + fmt(m, r[m])); });
    out.textContent = parts.join(', ');
  }

  function moveCursor(next) {
    var n = lastReadings.length;
    if (!n) { return; }
    cursorIndex = Math.max(0, Math.min(n - 1, next));
    renderSignal(lastReadings);
    announce(cursorIndex);
  }

  (function bindKeys() {
    var svg = document.getElementById('signal');
    svg.addEventListener('keydown', function (e) {
      var n = lastReadings.length;
      if (!n) { return; }
      var at = cursorIndex === null ? n - 1 : cursorIndex;
      if (e.key === 'ArrowLeft') { moveCursor(at - 1); }
      else if (e.key === 'ArrowRight') { moveCursor(at + 1); }
      else if (e.key === 'Home') { moveCursor(0); }
      else if (e.key === 'End') { moveCursor(n - 1); }
      else if (e.key === 'Escape') { cursorIndex = null; renderSignal(lastReadings); announce(null); }
      else { return; }
      e.preventDefault();
    });
    // Focusing lands on the newest reading, so the first arrow press moves
    // from somewhere meaningful rather than from nothing.
    svg.addEventListener('focus', function () {
      if (cursorIndex === null && lastReadings.length) { moveCursor(lastReadings.length - 1); }
    });
    svg.addEventListener('blur', function () {
      if (cursorIndex !== null) { cursorIndex = null; renderSignal(lastReadings); announce(null); }
    });
  })();

  function renderIncidents(incidents) {
    var body = document.getElementById('incident-rows');
    body.textContent = '';
    if (!incidents.length) {
      var tr = document.createElement('tr');
      var td = document.createElement('td');
      td.colSpan = 5; td.className = 'empty-state center';
      td.textContent = 'No incidents recorded.';
      tr.appendChild(td); body.appendChild(tr);
      return;
    }
    incidents.slice().reverse().forEach(function (inc) {
      var tr = document.createElement('tr');
      [inc.label,
       clock(inc.started),
       inc.duration_seconds !== null ? Math.round(inc.duration_seconds) + 's' : '--',
       inc.detected ? 'Detected' : 'Undetected',
       inc.detection_latency_seconds === null ? '--' : inc.detection_latency_seconds.toFixed(1) + 's'
      ].forEach(function (text, i) {
        var td = document.createElement('td');
        if (i === 3) {
          var pill = document.createElement('span');
          pill.className = 'pill ' + (inc.detected ? 'pill-good' : 'pill-warn');
          pill.textContent = text;
          td.appendChild(pill);
        } else {
          if (i !== 0) { td.className = (i === 2 || i === 4) ? 'mono num' : 'mono'; }
          td.textContent = text;
        }
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  function renderStatus(p) {
    var bar = document.getElementById('status-bar');
    var title = document.getElementById('status-title');
    var sub = document.getElementById('status-sub');
    var dot = document.getElementById('live-dot');

    document.getElementById('live-model').textContent = p.model.version
      ? 'model v' + p.model.version + ' / ' + p.model.type
      : 'no model in production';
    document.getElementById('chart-range').textContent = p.readings.length
      ? p.readings.length + ' readings'
      : '';

    dot.className = 'live-dot' + (p.stale ? ' stale' : '');
    document.getElementById('live-state').textContent = p.stale
      ? 'no data since ' + clock(p.last_seen)
      : 'streaming / ' + clock(p.last_seen);

    if (!p.readings.length) {
      bar.className = 'status-bar';
      title.textContent = 'Awaiting telemetry';
      sub.textContent = 'No readings received from the collector yet.';
    } else if (p.current_anomaly) {
      bar.className = 'status-bar alert';
      title.textContent = 'Anomaly detected';
      sub.textContent = p.current_incident
        ? 'Active incident: ' + p.current_incident
        : 'Current reading flagged as anomalous';
    } else {
      bar.className = 'status-bar ok';
      title.textContent = 'All systems normal';
      sub.textContent = p.current_incident
        ? 'Degradation in progress (' + p.current_incident + '), not yet flagged'
        : 'All monitored signals within expected behaviour';
    }
  }

  // Every failure is shown, never swallowed. The page used to drop any
  // non-200 response on the floor and keep saying "waiting" indefinitely,
  // which made a broken endpoint indistinguishable from a quiet one.
  function showFault(title, detail) {
    var bar = document.getElementById('status-bar');
    bar.className = 'status-bar alert';
    document.getElementById('status-title').textContent = title;
    document.getElementById('status-sub').innerHTML = detail;
    document.getElementById('live-dot').className = 'live-dot stale';
    document.getElementById('live-state').textContent = 'no data';
    var empty = document.getElementById('signal-empty');
    if (!lastReadings.length) { empty.innerHTML = detail; }
  }

  // One poll in flight at a time, each bounded: a request the server never
  // answers must surface as a fault, not leave the page waiting forever while
  // the interval stacks new requests on top of the hung ones.
  var inFlight = false;
  var FETCH_TIMEOUT_MS = 10000;

  function refresh() {
    if (inFlight) { return; }
    inFlight = true;
    var controller = window.AbortController ? new AbortController() : null;
    var timer = setTimeout(function () { if (controller) { controller.abort(); } }, FETCH_TIMEOUT_MS);

    fetch('/live/data', {
      credentials: 'same-origin',
      headers: { 'Accept': 'application/json' },
      signal: controller ? controller.signal : undefined
    })
      .then(function (r) {
        var type = r.headers.get('content-type') || '';
        // An expired session is redirected to the login page, which arrives
        // here as HTML with a 200 -- not an error fetch() would report.
        if (r.redirected || type.indexOf('json') === -1) { throw { kind: 'auth' }; }
        if (!r.ok) { throw { kind: 'http', status: r.status }; }
        return r.json();
      })
      .then(function (p) {
        renderStatus(p);
        renderKpis(p);
        renderSignal(p.readings);
        renderIncidents(p.incidents);
        if (!p.readings.length) {
          document.getElementById('signal-empty').innerHTML =
            'No readings recorded yet. The telemetry collector writes one every few seconds once it is ' +
            'running &mdash; its state is on the <a href="/">Overview</a> under System health.';
        }
      })
      .catch(function (err) {
        if (err && err.kind === 'auth') {
          showFault('Signed out', 'Your session has ended. <a href="/login">Sign in again</a> to resume the stream.');
        } else if (err && err.kind === 'http') {
          showFault('Telemetry unavailable',
            'The telemetry endpoint returned HTTP ' + err.status + '. Check System health on the <a href="/">Overview</a>.');
        } else if (err && err.name === 'AbortError') {
          showFault('Telemetry not responding',
            'The console did not answer within ' + (FETCH_TIMEOUT_MS / 1000) + 's. Check System health on the <a href="/">Overview</a>.');
        } else {
          showFault('Connection lost', 'Could not reach the console. Retrying every few seconds.');
        }
      })
      .then(function () {
        clearTimeout(timer);
        inFlight = false;
      });
  }

  refresh();
  // Hold the refresh while the pointer is reading the stack, so the chart
  // does not redraw out from under the cursor mid-inspection.
  setInterval(function () { if (cursorIndex === null) { refresh(); } }, REFRESH_MS);

  var resizeTimer = null;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (lastReadings.length) { renderSignal(lastReadings); }
    }, 120);
  });
})();
</script>
"""
