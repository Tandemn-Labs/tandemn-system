"""
Orca Web Dashboard — real-time job monitoring via SSE.

Provides two endpoints:
  GET /dashboard        → serves the single-page dashboard HTML
  GET /dashboard/stream → SSE stream of fleet-wide job/metrics/chunk data
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state for SSE enrichment
# ---------------------------------------------------------------------------

# SkyPilot price cache: (instance_type, market, region) -> (price_usd, cached_at)
_price_cache: dict[tuple, tuple[float, float]] = {}
_PRICE_CACHE_TTL = 300  # 5 min

# Synthetic event log
_event_log: deque[dict] = deque(maxlen=200)
_prev_job_status: dict[str, str] = {}
_prev_chunk_progress: dict[str, dict] = {}
_prev_replica_phases: dict[str, dict[str, str]] = {}

ACTIVE_PHASES = {"launching", "loading_model", "model_ready", "generating", "running"}


def _get_cached_price(instance_type: str, region: str, market: str) -> float | None:
    """Cache-wrapped SkyPilot price lookup (same pattern as metrics_db._get_price_per_hour)."""
    key = (instance_type, market, region)
    now = time.time()
    cached = _price_cache.get(key)
    if cached and now - cached[1] < _PRICE_CACHE_TTL:
        return cached[0]
    try:
        from sky import catalog
        price = catalog.get_hourly_cost(
            instance_type=instance_type,
            use_spot=(market == "spot"),
            region=region,
            zone=None,
            clouds="aws",
        )
        _price_cache[key] = (price, now)
        return price
    except Exception:
        logger.debug("dashboard: price lookup failed for %s", instance_type, exc_info=True)
        return None


def _emit_event(level: str, message: str, job_id: str = ""):
    """Append a synthetic event to the module-level log."""
    _event_log.append({
        "ts": time.time(),
        "level": level,
        "job_id": job_id,
        "message": message,
    })

dashboard_router = APIRouter()

# ---------------------------------------------------------------------------
# Dashboard HTML (single-page app with inline CSS + JS)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Orca Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f172a;--card:#1e293b;--card-hover:#263348;
  --text:#e2e8f0;--text-dim:#94a3b8;--text-muted:#64748b;
  --cyan:#22d3ee;--green:#4ade80;--red:#f87171;--yellow:#facc15;
  --blue:#60a5fa;--magenta:#c084fc;--orange:#fb923c;
  --border:#334155;--bar-bg:#0f172a;
}
html{font-size:14px}
body{
  font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--bg);color:var(--text);min-height:100vh;
}
a{color:var(--cyan);text-decoration:none}
a:hover{text-decoration:underline}

/* Header */
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:1rem 1.5rem;border-bottom:1px solid var(--border);
  background:#0b1120;position:sticky;top:0;z-index:100;
}
.header-left{display:flex;align-items:center;gap:1rem}
.logo{font-size:1.4rem;font-weight:700;color:var(--cyan);letter-spacing:0.05em}
.logo-sub{font-size:0.85rem;color:var(--text-dim);font-weight:400}
.header-right{display:flex;align-items:center;gap:1.25rem}
.conn-status{display:flex;align-items:center;gap:0.4rem;font-size:0.8rem;color:var(--text-dim)}
.conn-dot{width:8px;height:8px;border-radius:50%;transition:background 0.3s}
.conn-dot.ok{background:var(--green)}
.conn-dot.err{background:var(--red)}
.conn-dot.connecting{background:var(--yellow);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.server-url{font-size:0.75rem;color:var(--text-muted)}

/* Main container */
.main{padding:1.5rem;max-width:1600px;margin:0 auto}
.empty-state{
  text-align:center;padding:4rem 2rem;color:var(--text-muted);
  font-size:1rem;
}
.empty-state .whale{font-size:3rem;margin-bottom:1rem;display:block}

/* Job cards grid */
.grid{
  display:grid;grid-template-columns:1fr;gap:1rem;
}
@media(min-width:1024px){.grid{grid-template-columns:repeat(2,1fr)}}

/* Job card */
.card{
  background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:1.25rem;transition:background 0.2s,border-color 0.2s;
}
.card:hover{background:var(--card-hover);border-color:var(--cyan)}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem}
.model-name{font-size:1.1rem;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:70%}
.badge{
  display:inline-block;padding:0.15rem 0.55rem;border-radius:4px;
  font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;
}
.badge-succeeded{background:rgba(74,222,128,0.15);color:var(--green)}
.badge-failed{background:rgba(248,113,113,0.15);color:var(--red)}
.badge-generating{background:rgba(34,211,238,0.15);color:var(--cyan)}
.badge-launching{background:rgba(250,204,21,0.15);color:var(--yellow)}
.badge-loading_model,.badge-model_ready{background:rgba(96,165,250,0.15);color:var(--blue)}
.badge-running{background:rgba(34,211,238,0.15);color:var(--cyan)}
.badge-queued{background:rgba(148,163,184,0.15);color:var(--text-dim)}
.badge-cancelled{background:rgba(148,163,184,0.15);color:var(--text-muted)}

.job-id{font-size:0.75rem;color:var(--text-muted);margin-bottom:0.75rem;word-break:break-all}

/* Progress bar */
.progress-wrap{margin-bottom:0.75rem}
.progress-label{display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-dim);margin-bottom:0.3rem}
.progress-bar{
  height:6px;background:var(--bar-bg);border-radius:3px;overflow:hidden;
}
.progress-fill{
  height:100%;border-radius:3px;transition:width 0.5s ease;
  background:linear-gradient(90deg,var(--cyan),var(--blue));
}
.progress-fill.done{background:linear-gradient(90deg,var(--green),#22c55e)}
.progress-fill.failed{background:linear-gradient(90deg,var(--red),#dc2626)}

/* Stats row */
.stats{display:flex;flex-wrap:wrap;gap:0.5rem 1.25rem;margin-bottom:0.75rem}
.stat{font-size:0.75rem}
.stat-label{color:var(--text-muted)}
.stat-value{color:var(--text);font-weight:500}

/* Chunk progress */
.chunk-section{margin-top:0.75rem;border-top:1px solid var(--border);padding-top:0.75rem}
.chunk-header{font-size:0.75rem;font-weight:600;color:var(--text-dim);margin-bottom:0.4rem}

/* Replica table */
.replica-section{margin-top:0.75rem;border-top:1px solid var(--border);padding-top:0.75rem}
.replica-header{font-size:0.75rem;font-weight:600;color:var(--text-dim);margin-bottom:0.4rem}
.replica-table{width:100%;font-size:0.7rem;border-collapse:collapse}
.replica-table th{
  text-align:left;padding:0.3rem 0.5rem;color:var(--text-muted);
  border-bottom:1px solid var(--border);font-weight:500;
}
.replica-table td{padding:0.3rem 0.5rem;color:var(--text-dim)}
.replica-table tr:hover td{color:var(--text)}
.phase-badge{
  display:inline-block;padding:0.1rem 0.4rem;border-radius:3px;
  font-size:0.65rem;font-weight:600;text-transform:uppercase;
}
.phase-running{background:rgba(34,211,238,0.15);color:var(--cyan)}
.phase-launching{background:rgba(250,204,21,0.15);color:var(--yellow)}
.phase-loading_model{background:rgba(96,165,250,0.15);color:var(--blue)}
.phase-failed{background:rgba(248,113,113,0.15);color:var(--red)}
.phase-stopped{background:rgba(148,163,184,0.15);color:var(--text-muted)}

/* Charts */
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin:0.75rem 0}
.chart-wrap{background:var(--bg);border-radius:6px;padding:0.5rem;height:200px}
.chart-wrap canvas{width:100%!important;height:100%!important}
</style>
</head>
<body>
<div class="header">
  <div class="header-left">
    <span class="logo">ORCA</span>
    <span class="logo-sub">Dashboard</span>
  </div>
  <div class="header-right">
    <div class="conn-status">
      <span class="conn-dot connecting" id="connDot"></span>
      <span id="connLabel">Connecting...</span>
    </div>
    <span class="server-url" id="serverUrl"></span>
  </div>
</div>
<div class="main">
  <div id="content">
    <div class="empty-state">
      <span class="whale">&#128051;</span>
      Connecting to Orca server...
    </div>
  </div>
</div>
<script>
(function(){
  const serverUrl = window.location.origin;
  document.getElementById('serverUrl').textContent = serverUrl;

  let es = null;
  let reconnectTimer = null;

  /* ---- Timeseries storage ---- */
  const jobTimeseries = {};   // {job_id: [{timestamp, ...metrics}, ...]}
  const MAX_TS_POINTS = 300;
  const ACTIVE_STATUSES = new Set(['launching','loading_model','model_ready','generating','running']);

  function sanitizeId(s) { return s.replace(/[^a-zA-Z0-9]/g, '-'); }

  /* ---- ChartManager ---- */
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        grid: {color: '#334155'},
        ticks: {color: '#94a3b8', font: {size: 9}},
      },
      y: {
        grid: {color: '#334155'},
        ticks: {color: '#94a3b8', font: {size: 9}},
        beginAtZero: true,
      }
    },
    plugins: {
      legend: {
        labels: {color: '#e2e8f0', font: {size: 9}},
        position: 'top',
      }
    }
  };

  function mergeOpts(extra) {
    return JSON.parse(JSON.stringify(Object.assign({}, chartDefaults, extra || {})));
  }

  function pctScale() {
    return {y:{grid:{color:'#334155'},ticks:{color:'#94a3b8',font:{size:9}},beginAtZero:true,max:100}};
  }

  const CHART_DEFS = [
    {
      key: 'tp', label: 'Throughput', type: 'line',
      datasets: [
        {label:'Gen tok/s', borderColor:'#22d3ee', backgroundColor:'rgba(34,211,238,0.1)', field:'avg_generation_throughput_toks_per_s'},
        {label:'Prompt tok/s', borderColor:'#4ade80', backgroundColor:'rgba(74,222,128,0.1)', field:'avg_prompt_throughput_toks_per_s'},
      ],
    },
    {
      key: 'kv', label: 'KV Cache', type: 'line',
      datasets: [
        {label:'KV Cache %', borderColor:'#fb923c', backgroundColor:'rgba(251,146,60,0.25)', fill:true, field:'gpu_cache_usage_perc', scale:100},
      ],
      opts: {scales: pctScale()},
    },
    {
      key: 'sched', label: 'Scheduler', type: 'line',
      datasets: [
        {label:'Running', borderColor:'#22d3ee', backgroundColor:'rgba(34,211,238,0.25)', fill:true, field:'num_requests_running'},
        {label:'Waiting', borderColor:'#fb923c', backgroundColor:'rgba(251,146,60,0.25)', fill:true, field:'num_requests_waiting'},
        {label:'Swapped', borderColor:'#94a3b8', backgroundColor:'rgba(148,163,184,0.25)', fill:true, field:'num_requests_swapped'},
      ],
    },
    {
      key: 'gpu', label: 'GPU Utilization', type: 'line',
      datasets: [
        {label:'SM %', borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.1)', field:'gpu_sm_util_pct'},
        {label:'MemBW %', borderColor:'#4ade80', backgroundColor:'rgba(74,222,128,0.1)', field:'gpu_mem_bw_util_pct'},
      ],
      opts: {scales: pctScale()},
    },
    {
      key: 'lat', label: 'Latency', type: 'line',
      datasets: [
        {label:'TTFT p50', borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.1)', field:'ttft_ms_p50'},
        {label:'TTFT p95', borderColor:'#3b82f6', backgroundColor:'rgba(59,130,246,0.1)', field:'ttft_ms_p95', borderDash:[4,2]},
        {label:'TPOT p50', borderColor:'#fb923c', backgroundColor:'rgba(251,146,60,0.1)', field:'tpot_ms_p50'},
        {label:'TPOT p95', borderColor:'#f97316', backgroundColor:'rgba(249,115,22,0.1)', field:'tpot_ms_p95', borderDash:[4,2]},
      ],
    },
    {
      key: 'comp', label: 'Completions', type: 'line',
      datasets: [
        {label:'Success', borderColor:'#4ade80', backgroundColor:'rgba(74,222,128,0.1)', field:'request_success_total'},
        {label:'Preemptions', borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.1)', field:'num_preemptions_total'},
      ],
    },
  ];

  class ChartManager {
    constructor() {
      this.charts = {};  // {job_id: {key: Chart}}
    }

    getOrCreate(jobId, def, canvasId) {
      if (!this.charts[jobId]) this.charts[jobId] = {};
      if (this.charts[jobId][def.key]) return this.charts[jobId][def.key];

      const canvas = document.getElementById(canvasId);
      if (!canvas) return null;
      const ctx = canvas.getContext('2d');
      const opts = mergeOpts(def.opts);
      opts.plugins = opts.plugins || {};
      opts.plugins.legend = opts.plugins.legend || {};
      opts.plugins.legend.labels = {color: '#e2e8f0', font: {size: 9}};
      opts.plugins.legend.position = 'top';
      opts.plugins.title = {display: true, text: def.label, color: '#94a3b8', font: {size: 10}};

      const datasets = def.datasets.map(function(ds) {
        return {
          label: ds.label,
          borderColor: ds.borderColor,
          backgroundColor: ds.backgroundColor || 'transparent',
          borderWidth: 1.5,
          borderDash: ds.borderDash || [],
          tension: 0.3,
          pointRadius: 0,
          fill: ds.fill || false,
          data: [],
        };
      });

      const chart = new Chart(ctx, {
        type: def.type || 'line',
        data: {labels: [], datasets: datasets},
        options: opts,
      });
      this.charts[jobId][def.key] = chart;
      return chart;
    }

    update(jobId, ts) {
      if (!ts || !ts.length) return;
      const firstTs = ts[0].timestamp;
      const labels = ts.map(function(p) { return Math.round(p.timestamp - firstTs); });

      for (const def of CHART_DEFS) {
        const sid = sanitizeId(jobId);
        const canvasId = 'chart-' + def.key + '-' + sid;
        const chart = this.getOrCreate(jobId, def, canvasId);
        if (!chart) continue;

        chart.data.labels = labels;
        for (let i = 0; i < def.datasets.length; i++) {
          const dsDef = def.datasets[i];
          const scale = dsDef.scale || 1;
          chart.data.datasets[i].data = ts.map(function(p) {
            const v = p[dsDef.field];
            return v != null ? v * scale : null;
          });
        }
        chart.update('none');
      }
    }

    cleanup(activeJobIds) {
      const toRemove = [];
      for (const jid in this.charts) {
        if (!activeJobIds.has(jid)) toRemove.push(jid);
      }
      for (const jid of toRemove) {
        for (const key in this.charts[jid]) {
          this.charts[jid][key].destroy();
        }
        delete this.charts[jid];
      }
    }
  }

  const chartMgr = new ChartManager();
  let prevJobIdSet = new Set();

  function setConnState(state) {
    const dot = document.getElementById('connDot');
    const label = document.getElementById('connLabel');
    dot.className = 'conn-dot ' + ({connected:'ok',disconnected:'err',connecting:'connecting'}[state]||'connecting');
    label.textContent = {connected:'Connected',disconnected:'Disconnected',connecting:'Connecting...'}[state]||state;
  }

  function fmt(n) {
    if (n == null) return '\u2014';
    if (typeof n === 'number') {
      if (Number.isInteger(n)) return n.toLocaleString();
      return n.toLocaleString(undefined, {minimumFractionDigits:1, maximumFractionDigits:1});
    }
    return String(n);
  }
  function pct(n) {
    if (n == null) return '\u2014';
    return (n * 100).toFixed(1) + '%';
  }
  function relTime(ts) {
    if (!ts) return '\u2014';
    const d = (Date.now()/1000) - ts;
    if (d < 60) return Math.floor(d) + 's ago';
    if (d < 3600) return Math.floor(d/60) + 'm ago';
    if (d < 86400) return Math.floor(d/3600) + 'h ago';
    return Math.floor(d/86400) + 'd ago';
  }
  function absTime(ts) {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleString();
  }
  function badgeClass(status) {
    return 'badge badge-' + (status||'queued');
  }
  function phaseBadge(phase) {
    return 'phase-badge phase-' + (phase||'unknown');
  }
  function progressClass(status) {
    if (status === 'succeeded') return 'progress-fill done';
    if (status === 'failed') return 'progress-fill failed';
    return 'progress-fill';
  }

  function chartsHtml(jobId) {
    const sid = sanitizeId(jobId);
    let h = '<div class="charts-grid" id="charts-' + sid + '">';
    for (const def of CHART_DEFS) {
      h += '<div class="chart-wrap"><canvas id="chart-' + def.key + '-' + sid + '"></canvas></div>';
    }
    h += '</div>';
    return h;
  }

  function renderJobs(data) {
    const container = document.getElementById('content');
    const jobs = (data.jobs || []).slice().sort(function(a,b) { return (b.created_at||0) - (a.created_at||0); });
    if (!jobs.length) {
      container.innerHTML = '<div class="empty-state"><span class="whale">&#128051;</span>No jobs yet. Deploy a model to get started.</div>';
      chartMgr.cleanup(new Set());
      prevJobIdSet = new Set();
      return;
    }

    const curJobIds = new Set(jobs.map(function(j){ return j.job_id; }));
    // Only rebuild DOM when the set of job IDs changes
    const needsRebuild = curJobIds.size !== prevJobIdSet.size || [...curJobIds].some(function(id){ return !prevJobIdSet.has(id); });

    if (needsRebuild) {
      let html = '<div class="grid">';
      for (const job of jobs) {
        const jid = job.job_id;
        const sid = sanitizeId(jid);
        const isActive = ACTIVE_STATUSES.has(job.status);
        html += '<div class="card" id="card-' + sid + '">';
        html += '<div class="card-header">';
        html += '<span class="model-name">' + esc(job.model_name||'unknown') + '</span>';
        html += '<span class="' + badgeClass(job.status) + '" id="badge-' + sid + '">' + esc(job.status||'unknown') + '</span>';
        html += '</div>';
        html += '<div class="job-id">' + esc(jid) + '</div>';
        html += '<div class="progress-wrap" id="progwrap-' + sid + '"></div>';
        html += '<div class="stats" id="stats-' + sid + '"></div>';
        if (isActive) html += chartsHtml(jid);
        html += '<div id="chunks-' + sid + '"></div>';
        html += '<div id="replicas-' + sid + '"></div>';
        html += '</div>';
      }
      html += '</div>';
      container.innerHTML = html;
      prevJobIdSet = curJobIds;
      // Charts need canvases in DOM before creation, so cleanup old charts
      chartMgr.cleanup(curJobIds);
    }

    // Update each card's dynamic content
    for (const job of jobs) {
      const jid = job.job_id;
      const sid = sanitizeId(jid);
      const m = (data.metrics||{})[jid] || {};
      const ch = (data.chunks||{})[jid] || null;
      const reps = (data.replicas||{})[jid] || [];
      const prog = job.progress || 0;
      const isActive = ACTIVE_STATUSES.has(job.status);

      // Badge
      const badgeEl = document.getElementById('badge-' + sid);
      if (badgeEl) {
        badgeEl.className = badgeClass(job.status);
        badgeEl.textContent = job.status || 'unknown';
      }

      // Progress bar
      const progEl = document.getElementById('progwrap-' + sid);
      if (progEl) {
        progEl.innerHTML = '<div class="progress-label"><span>Progress</span><span>' + pct(prog) + '</span></div>'
          + '<div class="progress-bar"><div class="' + progressClass(job.status) + '" style="width:' + (prog*100).toFixed(1) + '%"></div></div>';
      }

      // Stats
      const statsEl = document.getElementById('stats-' + sid);
      if (statsEl) {
        let sh = '';
        sh += stat('Lines', fmt(job.num_lines));
        sh += stat('Created', relTime(job.created_at), absTime(job.created_at));
        sh += stat('Updated', relTime(job.last_updated_at));
        if (job.instance_type) sh += stat('Instance', job.instance_type);
        if (job.tp || job.pp) sh += stat('TP/PP', (job.tp||'?') + '/' + (job.pp||'?'));
        if (m.avg_generation_throughput_toks_per_s) sh += stat('Throughput', fmt(m.avg_generation_throughput_toks_per_s) + ' tok/s');
        if (m.gpu_cache_usage_perc != null && m.gpu_cache_usage_perc > 0) sh += stat('KV Cache', pct(m.gpu_cache_usage_perc));
        if (m.num_requests_running != null && m.num_requests_running > 0) sh += stat('Running', fmt(m.num_requests_running));
        if (m.num_requests_waiting != null && m.num_requests_waiting > 0) sh += stat('Waiting', fmt(m.num_requests_waiting));
        if (m.request_success_total) sh += stat('Completed', fmt(m.request_success_total));
        if (m.gpu_sm_util_pct) sh += stat('SM Util', fmt(m.gpu_sm_util_pct) + '%');
        statsEl.innerHTML = sh;
      }

      // Chunks
      const chunksEl = document.getElementById('chunks-' + sid);
      if (chunksEl) {
        if (ch && ch.total > 0) {
          const chProg = ch.total > 0 ? ch.completed / ch.total : 0;
          chunksEl.innerHTML = '<div class="chunk-section">'
            + '<div class="chunk-header">Chunks: ' + ch.completed + '/' + ch.total + ' completed</div>'
            + '<div class="progress-bar"><div class="progress-fill" style="width:' + (chProg*100).toFixed(1) + '%"></div></div>'
            + '<div class="stats" style="margin-top:0.4rem">'
            + stat('Pending', fmt(ch.pending))
            + stat('Inflight', fmt(ch.inflight))
            + stat('Failed', fmt(ch.failed))
            + '</div></div>';
        } else {
          chunksEl.innerHTML = '';
        }
      }

      // Replicas
      const repsEl = document.getElementById('replicas-' + sid);
      if (repsEl) {
        if (reps.length) {
          let rh = '<div class="replica-section">';
          rh += '<div class="replica-header">Replicas (' + reps.length + ')</div>';
          rh += '<table class="replica-table"><thead><tr>';
          rh += '<th>Replica</th><th>Phase</th><th>Region</th><th>Market</th><th>Instance</th>';
          rh += '</tr></thead><tbody>';
          for (const r of reps) {
            rh += '<tr>';
            rh += '<td style="font-size:0.65rem">' + esc(r.replica_id||'\u2014') + '</td>';
            rh += '<td><span class="' + phaseBadge(r.phase) + '">' + esc(r.phase||'\u2014') + '</span></td>';
            rh += '<td>' + esc(r.region||'\u2014') + '</td>';
            rh += '<td>' + esc(r.market||'\u2014') + '</td>';
            rh += '<td>' + esc(r.instance_type||'\u2014') + '</td>';
            rh += '</tr>';
          }
          rh += '</tbody></table></div>';
          repsEl.innerHTML = rh;
        } else {
          repsEl.innerHTML = '';
        }
      }

      // Accumulate timeseries and update charts for active jobs
      if (isActive && Object.keys(m).length > 0) {
        if (!jobTimeseries[jid]) jobTimeseries[jid] = [];
        const point = Object.assign({timestamp: Date.now() / 1000}, m);
        jobTimeseries[jid].push(point);
        if (jobTimeseries[jid].length > MAX_TS_POINTS) {
          jobTimeseries[jid] = jobTimeseries[jid].slice(-MAX_TS_POINTS);
        }
        chartMgr.update(jid, jobTimeseries[jid]);
      }
    }

    // Cleanup charts for removed jobs
    const curIds = new Set(jobs.map(function(j){ return j.job_id; }));
    chartMgr.cleanup(curIds);
    // Clean timeseries for removed jobs
    for (const jid in jobTimeseries) {
      if (!curIds.has(jid)) delete jobTimeseries[jid];
    }
  }

  function stat(label, value, title) {
    const t = title ? ' title="' + esc(title) + '"' : '';
    return '<span class="stat"' + t + '><span class="stat-label">' + esc(label) + ' </span><span class="stat-value">' + esc(String(value)) + '</span></span>';
  }

  function esc(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function connect() {
    if (es) { try { es.close(); } catch(e){} }
    setConnState('connecting');
    es = new EventSource(serverUrl + '/dashboard/stream');
    es.onopen = function() { setConnState('connected'); };
    es.onmessage = function(ev) {
      try {
        const data = JSON.parse(ev.data);
        renderJobs(data);
      } catch(e) { console.error('parse error', e); }
    };
    es.onerror = function() {
      setConnState('disconnected');
      es.close();
      es = null;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 3000);
    };
  }

  connect();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@dashboard_router.get("/dashboard")
async def serve_dashboard():
    """Serve the Orca web dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


@dashboard_router.get("/dashboard/stream")
async def dashboard_stream(request: Request):
    """SSE endpoint streaming fleet-wide job/metrics/chunk data every 2 s."""

    async def _generate():
        from orca_server.job_manager import get_job_tracker
        from orca_server.chunk_manager import get_chunk_manager

        while True:
            if await request.is_disconnected():
                break

            payload = {"jobs": [], "metrics": {}, "chunks": {}, "replicas": {}}
            mc = None
            cluster_mgr = None

            try:
                # ---- Jobs ----
                tracker = get_job_tracker()
                with tracker.lock:
                    job_items = list(tracker.jobs.items())

                for job_id, rec in job_items:
                    try:
                        job_data = {
                            "job_id": job_id,
                            "status": rec.status,
                            "progress": round(rec.state.progress_frac, 4),
                            "model_name": rec.state.spec.model_name,
                            "num_lines": rec.state.spec.num_lines,
                            "created_at": rec.created_at,
                            "last_updated_at": rec.last_updated_at,
                            "head_ip": rec.head_ip,
                            "endpoint_url": rec.endpoint_url,
                            "instance_type": rec.state.instance_types,
                            "tp": rec.state.tp,
                            "pp": rec.state.pp,
                            "slo_hours": rec.state.spec.slo_hours,
                            "avg_input_tokens": rec.state.spec.avg_input_tokens,
                            "avg_output_tokens": rec.state.spec.avg_output_tokens,
                            "market": rec.state.spec.market,
                            "submitted_at": rec.state.submitted_at,
                            "num_replicas": getattr(rec, "num_replicas", 1),
                            "is_chunked": getattr(rec, "is_chunked", False),
                        }
                        payload["jobs"].append(job_data)
                    except Exception:
                        logger.debug("dashboard: error serialising job %s", job_id, exc_info=True)

                # ---- Metrics ----
                try:
                    mc = getattr(request.app.state, "metrics_collector", None)
                    if mc is not None:
                        for job_id, _rec in job_items:
                            try:
                                snap = mc.get_aggregated(job_id)
                                if snap is not None:
                                    payload["metrics"][job_id] = {
                                        "avg_generation_throughput_toks_per_s": snap.avg_generation_throughput_toks_per_s,
                                        "avg_prompt_throughput_toks_per_s": snap.avg_prompt_throughput_toks_per_s,
                                        "gpu_cache_usage_perc": snap.gpu_cache_usage_perc,
                                        "num_requests_running": snap.num_requests_running,
                                        "num_requests_waiting": snap.num_requests_waiting,
                                        "num_requests_swapped": snap.num_requests_swapped,
                                        "request_success_total": snap.request_success_total,
                                        "num_preemptions_total": snap.num_preemptions_total,
                                        "gpu_sm_util_pct": snap.gpu_sm_util_pct,
                                        "gpu_mem_bw_util_pct": snap.gpu_mem_bw_util_pct,
                                        "ttft_ms_p50": snap.ttft_ms_p50,
                                        "ttft_ms_p95": snap.ttft_ms_p95,
                                        "tpot_ms_p50": snap.tpot_ms_p50,
                                        "tpot_ms_p95": snap.tpot_ms_p95,
                                    }
                            except Exception:
                                logger.debug("dashboard: metrics error for %s", job_id, exc_info=True)
                except Exception:
                    logger.debug("dashboard: metrics_collector error", exc_info=True)

                # ---- Chunks ----
                try:
                    if getattr(request.app.state, "redis_available", False):
                        cm = get_chunk_manager()
                        for job_id, rec in job_items:
                            try:
                                prog = cm.get_progress(job_id)
                                if prog and prog.get("total", 0) > 0:
                                    payload["chunks"][job_id] = {
                                        "total": prog["total"],
                                        "pending": prog["pending"],
                                        "inflight": prog["inflight"],
                                        "completed": prog["completed"],
                                        "failed": prog["failed"],
                                    }
                            except Exception:
                                logger.debug("dashboard: chunk error for %s", job_id, exc_info=True)
                except Exception:
                    logger.debug("dashboard: chunk_manager error", exc_info=True)

                # ---- Replicas ----
                try:
                    cluster_mgr = getattr(request.app.state, "cluster_manager", None)
                    if cluster_mgr is not None:
                        for job_id, _rec in job_items:
                            try:
                                states = cluster_mgr.get_replica_states(job_id)
                                if states:
                                    replicas = []
                                    for rid, rstate in states.items():
                                        replicas.append({
                                            "replica_id": rid,
                                            "phase": rstate.get("phase", "unknown"),
                                            "region": rstate.get("region", ""),
                                            "market": rstate.get("market", ""),
                                            "instance_type": rstate.get("instance_type", ""),
                                            "has_metrics": rstate.get("has_metrics", False),
                                            "running_since": rstate.get("running_since"),
                                        })
                                    if replicas:
                                        payload["replicas"][job_id] = replicas
                            except Exception:
                                logger.debug("dashboard: replica error for %s", job_id, exc_info=True)
                except Exception:
                    logger.debug("dashboard: cluster_manager error", exc_info=True)

                # ---- Cost (via SkyPilot pricing) ----
                payload["cost"] = {}
                try:
                    if cluster_mgr is None:
                        cluster_mgr = getattr(request.app.state, "cluster_manager", None)
                    now = time.time()
                    for job_id, rec in job_items:
                        try:
                            instance_type = rec.state.instance_types
                            if not instance_type:
                                continue
                            region = rec.state.spec.region or "us-east-1"
                            market = rec.state.spec.market or "spot"
                            price = _get_cached_price(instance_type, region, market)
                            if price is None:
                                continue

                            # Sum running hours across replicas
                            total_running_hours = 0.0
                            num_running = 0
                            rep_states = {}
                            try:
                                if cluster_mgr:
                                    rep_states = cluster_mgr.get_replica_states(job_id)
                            except Exception:
                                pass

                            for _rid, rs in rep_states.items():
                                rs_since = rs.get("running_since")
                                if rs_since and rs.get("phase") in ACTIVE_PHASES:
                                    total_running_hours += (now - rs_since) / 3600
                                    num_running += 1

                            # Fallback: use submitted_at if no per-replica data
                            if total_running_hours == 0 and rec.status in ACTIVE_PHASES:
                                total_running_hours = (now - rec.state.submitted_at) / 3600
                                num_running = getattr(rec, "num_replicas", 1) or 1

                            accrued = price * total_running_hours
                            progress = rec.state.progress_frac
                            projected = None
                            eta_sec = None
                            if progress > 0.01:
                                projected = accrued / progress
                                elapsed = total_running_hours * 3600
                                eta_sec = ((1.0 - progress) / progress) * elapsed

                            payload["cost"][job_id] = {
                                "price_per_hour": round(price, 4),
                                "accrued_usd": round(accrued, 4),
                                "projected_total_usd": round(projected, 4) if projected else None,
                                "eta_sec": round(eta_sec) if eta_sec else None,
                                "num_running_replicas": num_running,
                            }
                        except Exception:
                            logger.debug("dashboard: cost error for %s", job_id, exc_info=True)
                except Exception:
                    logger.debug("dashboard: cost section error", exc_info=True)

                # ---- Synthetic events ----
                try:
                    for job_id, rec in job_items:
                        jid_short = job_id[:12]
                        # Status transitions
                        prev_st = _prev_job_status.get(job_id)
                        if prev_st is not None and prev_st != rec.status:
                            lvl = "error" if rec.status == "failed" else "ok" if rec.status == "succeeded" else "info"
                            _emit_event(lvl, f"{jid_short} {prev_st} -> {rec.status}", job_id)
                        _prev_job_status[job_id] = rec.status

                        # Chunk milestones (25/50/75/100%)
                        ch = payload.get("chunks", {}).get(job_id)
                        if ch and ch.get("total", 0) > 0:
                            prev_ch = _prev_chunk_progress.get(job_id, {})
                            prev_pct = prev_ch.get("completed", 0) / ch["total"] * 100 if prev_ch.get("completed") is not None else 0
                            cur_pct = ch["completed"] / ch["total"] * 100
                            for milestone in (25, 50, 75, 100):
                                if prev_pct < milestone <= cur_pct:
                                    _emit_event("ok", f"{jid_short} chunks {milestone}% ({ch['completed']}/{ch['total']})", job_id)
                            _prev_chunk_progress[job_id] = dict(ch)

                        # Replica phase changes
                        reps = payload.get("replicas", {}).get(job_id, [])
                        prev_phases = _prev_replica_phases.get(job_id, {})
                        cur_phases = {}
                        for r in reps:
                            rid = r["replica_id"]
                            phase = r["phase"]
                            cur_phases[rid] = phase
                            if rid in prev_phases and prev_phases[rid] != phase:
                                lvl = "error" if phase in ("failed", "dead") else "info"
                                _emit_event(lvl, f"replica {rid[-8:]} -> {phase}", job_id)
                        _prev_replica_phases[job_id] = cur_phases

                    payload["events"] = list(_event_log)[-50]
                except Exception:
                    logger.debug("dashboard: events error", exc_info=True)
                    payload["events"] = []

                # ---- Timeseries (SSE resilience) ----
                payload["timeseries"] = {}
                try:
                    if mc is None:
                        mc = getattr(request.app.state, "metrics_collector", None)
                    if mc is not None:
                        for job_id, _rec in job_items:
                            if _rec.status in ACTIVE_PHASES:
                                try:
                                    recent = mc.get_recent(job_id, n=60)
                                    if recent:
                                        payload["timeseries"][job_id] = recent
                                except Exception:
                                    logger.debug("dashboard: timeseries error for %s", job_id, exc_info=True)
                except Exception:
                    logger.debug("dashboard: timeseries section error", exc_info=True)

            except Exception:
                logger.debug("dashboard: top-level SSE error", exc_info=True)

            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
