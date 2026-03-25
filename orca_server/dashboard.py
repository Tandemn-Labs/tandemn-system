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
  --bg:#0d1117;--card:#161b22;--card-hover:#1c2128;
  --text:#e6edf3;--text-dim:#8b949e;--text-muted:#6e7681;
  --cyan:#22d3ee;--green:#57ab5a;--red:#e5534b;--yellow:#c69026;
  --blue:#6cb6ff;--magenta:#9a7fd4;--orange:#c98c5a;
  --border:#30363d;--bar-bg:#21262d;
}
html{font-size:14px}
body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column}

/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;padding:6px 14px;border-bottom:1px solid var(--border);background:var(--card);flex-shrink:0}
.hdr-left{display:flex;align-items:center;gap:10px}
.logo{font-size:1.2rem;font-weight:700;color:var(--cyan);letter-spacing:.06em}
.logo-sub{font-size:.75rem;color:var(--text-dim)}
.hdr-right{display:flex;align-items:center;gap:12px}
.conn{display:flex;align-items:center;gap:5px;font-size:.7rem;color:var(--text-dim)}
.cdot{width:7px;height:7px;border-radius:50%;transition:background .3s}
.cdot.ok{background:var(--green)}.cdot.err{background:var(--red)}.cdot.wait{background:var(--yellow);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.surl{font-size:.65rem;color:var(--text-muted)}
.sec-hdr{background:var(--card);border-bottom:1px solid var(--border);padding:4px 10px;color:var(--text-muted);font-size:9px;letter-spacing:.08em;text-transform:uppercase;flex-shrink:0}

/* Layout */
.wrap{flex:1;display:flex;flex-direction:column;min-height:0}
.top-sec{display:flex;flex-shrink:0;border-bottom:1px solid var(--border)}
.wl-col{width:220px;flex-shrink:0;border-right:1px solid var(--border);display:flex;flex-direction:column}
.wl-block{padding:8px 10px;font-size:10px;line-height:1.9;flex:1}
.wl-row{display:flex}.wl-k{color:var(--text-muted);min-width:80px;flex-shrink:0}.wl-v{color:var(--text)}
.wl-v.hi{font-weight:600}

.chain-col{flex:1;display:flex;flex-direction:column;min-width:0}
.chain-inner{padding:10px 14px 6px;flex:1;display:flex;flex-direction:column;justify-content:center}
.chain-meta{font-size:10px;margin-bottom:5px;display:flex;align-items:center;gap:8px}
.chain-pct{font-weight:bold;min-width:32px}
.chain-eta{color:var(--text-muted);font-size:9px;margin-left:auto}
.prog-outer{height:3px;background:var(--bar-bg);border-radius:2px;margin-bottom:7px;overflow:hidden}
.prog-fill{height:3px;border-radius:2px;transition:width .5s,background .4s}

/* Cost bar */
.cost-bar{display:flex;border-top:1px solid var(--bar-bg);flex-shrink:0}
.cost-cell{flex:1;padding:7px 12px;display:flex;flex-direction:column;gap:2px}
.cost-cell+.cost-cell{border-left:1px solid var(--bar-bg)}
.cost-lbl{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em}
.cost-val{font-size:13px;font-weight:bold;font-variant-numeric:tabular-nums;transition:color .3s}
.cost-sub{font-size:9px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.cost-meter{height:2px;border-radius:1px;margin-top:4px;background:var(--bar-bg);overflow:hidden}
.cost-meter-fill{height:2px;border-radius:1px;transition:width .5s,background .4s}

/* Bottom */
.bot-sec{display:flex;flex:1;min-height:0}
.res-col{width:220px;flex-shrink:0;border-right:1px solid var(--border);display:flex;flex-direction:column;min-height:0}
.res-list{flex:1;overflow-y:auto;padding:6px 10px;display:flex;flex-direction:column;gap:4px}
.res-row{display:flex;align-items:center;font-size:10px;padding:3px 6px;border-radius:3px;border:1px solid var(--bar-bg);background:var(--bg);transition:border-color .3s,background .3s}
.res-row.on{background:#0d1f2e;border-color:var(--cyan)}
.res-cnt{color:var(--text-muted);min-width:22px;text-align:right;margin-right:6px;font-size:9px}
.res-gpu{flex:1;color:var(--text)}.res-rgn{color:var(--text-muted);font-size:9px;margin-left:4px}
.res-ind{width:6px;height:6px;border-radius:50%;background:var(--bar-bg);margin-left:6px;flex-shrink:0;transition:background .3s}
.res-ind.on{background:var(--cyan)}

.job-sel{flex-shrink:0;border-top:1px solid var(--border)}
.jbtns{display:flex;flex-direction:column}
.jbtn{display:flex;align-items:center;gap:8px;padding:5px 10px;cursor:pointer;border:none;background:transparent;width:100%;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text);border-bottom:1px solid var(--bar-bg);text-align:left}
.jbtn:last-child{border-bottom:none}
.jbtn:hover{background:var(--card)}.jbtn.act{background:var(--card-hover)}
.jbtn-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.jbtn-name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.jbtn-pct{font-size:9px;min-width:28px;text-align:right}

.log-col{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}
.log-area{flex:1;overflow-y:auto;padding:6px 14px}
.ll{font-size:9px;line-height:1.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;animation:fi .3s ease-in}
.ll.ok{color:var(--green)}.ll.info{color:var(--blue)}.ll.warn{color:var(--yellow)}.ll.error{color:var(--red)}.ll.dim{color:#444c56}
@keyframes fi{from{opacity:0;transform:translateY(2px)}to{opacity:1;transform:none}}

/* Charts toggle */
.charts-toggle{padding:4px 14px;border-top:1px solid var(--border);flex-shrink:0}
.charts-toggle button{background:none;border:1px solid var(--border);border-radius:3px;color:var(--text-dim);font-family:'JetBrains Mono',monospace;font-size:9px;padding:2px 8px;cursor:pointer}
.charts-toggle button:hover{border-color:var(--cyan);color:var(--text)}
.charts-wrap{border-top:1px solid var(--border);overflow-y:auto;max-height:450px}
.charts-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;padding:0.5rem}
.chart-wrap{background:var(--bg);border-radius:6px;padding:0.5rem;height:180px}
.chart-wrap canvas{width:100%!important;height:100%!important}

/* Chain SVG packets */
@keyframes pkt{0%{offset-distance:0%}100%{offset-distance:100%}}

/* Badge */
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:.6rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.badge-succeeded{background:rgba(87,171,90,.15);color:var(--green)}
.badge-failed{background:rgba(229,83,75,.15);color:var(--red)}
.badge-generating,.badge-running{background:rgba(34,211,238,.15);color:var(--cyan)}
.badge-launching{background:rgba(198,144,38,.15);color:var(--yellow)}
.badge-loading_model,.badge-model_ready{background:rgba(108,182,255,.15);color:var(--blue)}
.badge-queued,.badge-cancelled{background:rgba(139,148,158,.15);color:var(--text-dim)}

.empty-state{text-align:center;padding:4rem 2rem;color:var(--text-muted);font-size:1rem}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-left"><span class="logo">ORCA</span><span class="logo-sub">Dashboard</span></div>
  <div class="hdr-right">
    <div class="conn"><span class="cdot wait" id="connDot"></span><span id="connLabel">Connecting...</span></div>
    <span class="surl" id="serverUrl"></span>
  </div>
</div>
<div class="wrap" id="wrap">
  <div class="empty-state" id="empty">No jobs yet. Deploy a model to get started.</div>
</div>
<script>
(function(){
const serverUrl = window.location.origin;
document.getElementById('serverUrl').textContent = serverUrl;

/* Instance type to GPU name mapping */
const INST_GPU = {
  "p5.48xlarge":"H100","p4d.24xlarge":"A100","p4de.24xlarge":"A100",
  "p3.2xlarge":"V100","p3.8xlarge":"V100","p3.16xlarge":"V100","p3dn.24xlarge":"V100",
  "g6e.xlarge":"L40S","g6e.2xlarge":"L40S","g6e.4xlarge":"L40S","g6e.8xlarge":"L40S",
  "g6e.12xlarge":"L40S","g6e.16xlarge":"L40S","g6e.24xlarge":"L40S","g6e.48xlarge":"L40S",
  "g6.xlarge":"L4","g6.2xlarge":"L4","g6.4xlarge":"L4","g6.8xlarge":"L4",
  "g6.12xlarge":"L4","g6.16xlarge":"L4","g6.24xlarge":"L4","g6.48xlarge":"L4",
  "g5.xlarge":"A10G","g5.2xlarge":"A10G","g5.4xlarge":"A10G","g5.8xlarge":"A10G",
  "g5.12xlarge":"A10G","g5.16xlarge":"A10G","g5.24xlarge":"A10G","g5.48xlarge":"A10G",
};
function gpuName(inst) { return INST_GPU[inst] || inst || '\u2014'; }

let es = null, reconnectTimer = null, activeJobId = null;
let prevJobIds = new Set(), chartsVisible = false, structureBuilt = false;
const jobTimeseries = {};
const MAX_TS = 300;
const ACTIVE = new Set(['launching','loading_model','model_ready','generating','running']);
const eventBuffer = [];
const MAX_EVENTS = 200;

function sid(s) { return s.replace(/[^a-zA-Z0-9]/g, '-'); }
function esc(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function fmt(n) { if (n == null) return '\u2014'; return typeof n === 'number' ? (Number.isInteger(n) ? n.toLocaleString() : n.toFixed(1)) : String(n); }
function fmtUsd(n) { return n != null ? '$' + n.toFixed(2) : '\u2014'; }
function fmtTime(s) { if (s == null || s <= 0) return '\u2014'; if (s < 60) return Math.round(s) + 's'; if (s < 3600) { const m = Math.floor(s/60), ss = Math.round(s%60); return m + 'm ' + ss + 's'; } const h = Math.floor(s/3600), m = Math.floor((s%3600)/60); return h + 'h ' + m + 'm'; }
function pct(n) { return n != null ? (n * 100).toFixed(1) + '%' : '\u2014'; }

function setConn(state) {
  const d = document.getElementById('connDot'), l = document.getElementById('connLabel');
  d.className = 'cdot ' + ({connected:'ok',disconnected:'err',connecting:'wait'}[state]||'wait');
  l.textContent = {connected:'Connected',disconnected:'Disconnected',connecting:'Connecting...'}[state]||state;
}

/* ---- Chart infrastructure (kept from original) ---- */
const chartDefaults = {responsive:true,maintainAspectRatio:false,animation:false,
  scales:{x:{grid:{color:'#30363d'},ticks:{color:'#8b949e',font:{size:9}}},y:{grid:{color:'#30363d'},ticks:{color:'#8b949e',font:{size:9}},beginAtZero:true}},
  plugins:{legend:{labels:{color:'#e6edf3',font:{size:9}},position:'top'}}};
function mOpts(e){return JSON.parse(JSON.stringify(Object.assign({},chartDefaults,e||{})));}
function pScale(){return{y:{grid:{color:'#30363d'},ticks:{color:'#8b949e',font:{size:9}},beginAtZero:true,max:100}};}
const CDEFS=[
  {key:'tp',label:'Throughput (10s avg)',type:'line',datasets:[{label:'Gen tok/s',borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,0.1)',field:'avg_generation_throughput_toks_per_s'},{label:'Prompt tok/s',borderColor:'#57ab5a',backgroundColor:'rgba(87,171,90,0.1)',field:'avg_prompt_throughput_toks_per_s'}]},
  {key:'kv',label:'KV Cache',type:'line',datasets:[{label:'KV %',borderColor:'#c98c5a',backgroundColor:'rgba(201,140,90,0.25)',fill:true,field:'gpu_cache_usage_perc',scale:100}],opts:{scales:pScale()}},
  {key:'sched',label:'Scheduler',type:'line',datasets:[{label:'Running',borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,0.25)',fill:true,field:'num_requests_running'},{label:'Waiting',borderColor:'#c98c5a',backgroundColor:'rgba(201,140,90,0.25)',fill:true,field:'num_requests_waiting'},{label:'Swapped',borderColor:'#8b949e',backgroundColor:'rgba(139,148,158,0.25)',fill:true,field:'num_requests_swapped'}]},
  {key:'gpu',label:'GPU Utilization',type:'line',datasets:[{label:'SM %',borderColor:'#6cb6ff',backgroundColor:'rgba(108,182,255,0.1)',field:'gpu_sm_util_pct'},{label:'MemBW %',borderColor:'#57ab5a',backgroundColor:'rgba(87,171,90,0.1)',field:'gpu_mem_bw_util_pct'}],opts:{scales:pScale()}},
  {key:'lat',label:'Latency (ms)',type:'line',datasets:[{label:'TTFT p50',borderColor:'#6cb6ff',backgroundColor:'rgba(108,182,255,0.1)',field:'ttft_ms_p50'},{label:'TTFT p95',borderColor:'#3b82f6',borderDash:[4,2],field:'ttft_ms_p95'},{label:'TPOT p50',borderColor:'#c98c5a',backgroundColor:'rgba(201,140,90,0.1)',field:'tpot_ms_p50'},{label:'TPOT p95',borderColor:'#f97316',borderDash:[4,2],field:'tpot_ms_p95'}]},
  {key:'comp',label:'Completions',type:'line',datasets:[{label:'Success',borderColor:'#57ab5a',backgroundColor:'rgba(87,171,90,0.1)',field:'request_success_total'},{label:'Preemptions',borderColor:'#e5534b',backgroundColor:'rgba(229,83,75,0.1)',field:'num_preemptions_total'}]},
];
class ChartMgr{constructor(){this.c={}}
  getOrCreate(jid,def,cid){if(!this.c[jid])this.c[jid]={};if(this.c[jid][def.key])return this.c[jid][def.key];const cv=document.getElementById(cid);if(!cv)return null;const o=mOpts(def.opts);o.plugins=o.plugins||{};o.plugins.legend=o.plugins.legend||{};o.plugins.legend.labels={color:'#e6edf3',font:{size:9}};o.plugins.legend.position='top';o.plugins.title={display:true,text:def.label,color:'#8b949e',font:{size:10}};const ds=def.datasets.map(d=>({label:d.label,borderColor:d.borderColor,backgroundColor:d.backgroundColor||'transparent',borderWidth:1.5,borderDash:d.borderDash||[],tension:0.3,pointRadius:0,fill:d.fill||false,data:[]}));const ch=new Chart(cv.getContext('2d'),{type:def.type||'line',data:{labels:[],datasets:ds},options:o});this.c[jid][def.key]=ch;return ch}
  update(jid,ts){if(!ts||!ts.length)return;const t0=ts[0].timestamp;const lbls=ts.map(p=>Math.round(p.timestamp-t0));for(const def of CDEFS){const ch=this.getOrCreate(jid,def,'chart-'+def.key+'-'+sid(jid));if(!ch)continue;ch.data.labels=lbls;for(let i=0;i<def.datasets.length;i++){const dd=def.datasets[i],sc=dd.scale||1;ch.data.datasets[i].data=ts.map(p=>{const v=p[dd.field];return v!=null?v*sc:null})}ch.update('none')}}
  cleanup(ids){for(const j in this.c){if(!ids.has(j)){for(const k in this.c[j])this.c[j][k].destroy();delete this.c[j]}}}}
const chartMgr = new ChartMgr();

/* ---- Build structure ---- */
function buildStructure() {
  if (structureBuilt) return;
  structureBuilt = true;
  const w = document.getElementById('wrap');
  const e = document.getElementById('empty');
  if (e) e.remove();
  w.innerHTML = '<div class="top-sec">'
    + '<div class="wl-col"><div class="sec-hdr">workload</div><div class="wl-block" id="wl-block"></div></div>'
    + '<div class="chain-col"><div class="sec-hdr">chain</div>'
    + '<div class="chain-inner"><div class="chain-meta"><span id="chain-label" style="color:var(--text-muted)">initializing...</span><span class="chain-pct" id="chain-pct">0%</span><span class="chain-eta" id="chain-eta"></span></div>'
    + '<div class="prog-outer"><div class="prog-fill" id="prog-fill"></div></div>'
    + '<svg id="chain-svg" width="100%" height="66" style="display:block"></svg></div>'
    + '<div class="cost-bar" id="cost-bar"><div class="cost-cell"><span class="cost-lbl">cost accrued</span><span class="cost-val" id="c-accrued">\u2014</span><span class="cost-sub" id="c-accrued-sub"></span><div class="cost-meter"><div class="cost-meter-fill" id="c-meter" style="width:0%"></div></div></div>'
    + '<div class="cost-cell"><span class="cost-lbl">projected total</span><span class="cost-val" id="c-proj">\u2014</span><span class="cost-sub" id="c-proj-sub">at current rate</span></div>'
    + '<div class="cost-cell"><span class="cost-lbl">time to complete</span><span class="cost-val" id="c-ttc">\u2014</span><span class="cost-sub" id="c-ttc-sub">projected</span></div>'
    + '<div class="cost-cell"><span class="cost-lbl">throughput</span><span class="cost-val" id="c-tps">\u2014</span><span class="cost-sub" id="c-tps-sub">10s avg</span></div></div></div></div>'
    + '<div class="bot-sec"><div class="res-col"><div class="sec-hdr">resource pool</div><div class="res-list" id="res-list"></div>'
    + '<div class="job-sel"><div class="sec-hdr">jobs</div><div class="jbtns" id="jbtns"></div></div></div>'
    + '<div class="log-col"><div class="sec-hdr">event log</div><div class="log-area" id="log-area"></div></div></div>'
    + '<div class="charts-toggle" id="charts-toggle"><button id="charts-btn">Show Charts</button></div>'
    + '<div class="charts-wrap" id="charts-wrap" style="display:none"></div>';
  document.getElementById('charts-btn').onclick = function() {
    chartsVisible = !chartsVisible;
    document.getElementById('charts-wrap').style.display = chartsVisible ? '' : 'none';
    this.textContent = chartsVisible ? 'Hide Charts' : 'Show Charts';
  };
}

/* ---- Render ---- */
function render(data) {
  const jobs = (data.jobs || []).slice().sort((a,b) => (b.created_at||0) - (a.created_at||0));
  if (!jobs.length) {
    document.getElementById('wrap').innerHTML = '<div class="empty-state">No jobs yet. Deploy a model to get started.</div>';
    structureBuilt = false;
    chartMgr.cleanup(new Set());
    return;
  }
  buildStructure();

  // Auto-select active job
  if (!activeJobId || !jobs.find(j => j.job_id === activeJobId)) activeJobId = jobs[0].job_id;
  const job = jobs.find(j => j.job_id === activeJobId);
  const m = (data.metrics||{})[activeJobId] || {};
  const ch = (data.chunks||{})[activeJobId] || null;
  const reps = (data.replicas||{})[activeJobId] || [];
  const cost = (data.cost||{})[activeJobId] || null;
  const prog = job.progress || 0;
  const isActive = ACTIVE.has(job.status);

  // Workload panel
  const wl = document.getElementById('wl-block');
  if (wl) {
    let h = '';
    h += '<div class="wl-row"><span class="wl-k">Model</span><span class="wl-v hi" style="color:var(--cyan)">' + esc(job.model_name) + '</span></div>';
    h += '<div class="wl-row"><span class="wl-k">Prompts</span><span class="wl-v hi" style="color:var(--cyan)">' + fmt(job.num_lines) + '</span></div>';
    if (job.avg_input_tokens) h += '<div class="wl-row"><span class="wl-k">Avg input</span><span class="wl-v">' + fmt(job.avg_input_tokens) + ' tok</span></div>';
    if (job.avg_output_tokens) h += '<div class="wl-row"><span class="wl-k">Max out</span><span class="wl-v">' + fmt(job.avg_output_tokens) + ' tok</span></div>';
    if (job.slo_hours) h += '<div class="wl-row"><span class="wl-k">SLO</span><span class="wl-v" style="color:var(--orange)">' + job.slo_hours + ' hours</span></div>';
    if (job.tp || job.pp) h += '<div class="wl-row"><span class="wl-k">Config</span><span class="wl-v" style="color:var(--magenta)">TP=' + (job.tp||'?') + ' PP=' + (job.pp||'?') + '</span></div>';
    if (job.instance_type) h += '<div class="wl-row"><span class="wl-k">Instance</span><span class="wl-v">' + esc(job.instance_type) + '</span></div>';
    if (job.market) h += '<div class="wl-row"><span class="wl-k">Market</span><span class="wl-v">' + esc(job.market) + '</span></div>';
    if (ch && ch.total > 1) h += '<div class="wl-row"><span class="wl-k">Chunks</span><span class="wl-v">' + ch.completed + '/' + ch.total + ' done</span></div>';
    h += '<div class="wl-row"><span class="wl-k">Status</span><span class="badge badge-' + (job.status||'queued') + '">' + esc(job.status) + '</span></div>';
    wl.innerHTML = h;
  }

  // Chain label + progress
  const lbl = document.getElementById('chain-label');
  if (lbl) {
    if (job.status === 'succeeded') { lbl.textContent = 'complete'; lbl.style.color = 'var(--green)'; }
    else if (job.status === 'failed') { lbl.textContent = 'failed'; lbl.style.color = 'var(--red)'; }
    else if (isActive) { lbl.textContent = 'running' + (reps.length ? ' \u00b7 ' + reps.length + ' replica' + (reps.length>1?'s':'') : ''); lbl.style.color = 'var(--text)'; }
    else { lbl.textContent = job.status || 'initializing...'; lbl.style.color = 'var(--text-muted)'; }
  }
  const cp = document.getElementById('chain-pct');
  if (cp) { cp.textContent = Math.round(prog * 100) + '%'; cp.style.color = 'var(--cyan)'; }
  const pf = document.getElementById('prog-fill');
  if (pf) { pf.style.width = (prog * 100).toFixed(1) + '%'; pf.style.background = job.status === 'failed' ? 'var(--red)' : job.status === 'succeeded' ? 'var(--green)' : 'var(--cyan)'; }
  const eta = document.getElementById('chain-eta');
  if (eta) { eta.textContent = cost && cost.eta_sec ? 'eta ~' + fmtTime(cost.eta_sec) : job.status === 'succeeded' ? 'done' : ''; }

  // Chain SVG
  renderChainSVG(reps, isActive);

  // Cost bar
  if (cost) {
    const ca = document.getElementById('c-accrued');
    if (ca) { ca.textContent = fmtUsd(cost.accrued_usd); ca.style.color = 'var(--text)'; }
    const cs = document.getElementById('c-accrued-sub');
    if (cs) cs.textContent = fmtUsd(cost.price_per_hour) + '/hr \u00b7 ' + (cost.num_running_replicas||0) + ' replica' + ((cost.num_running_replicas||0)!==1?'s':'');
    const cp2 = document.getElementById('c-proj');
    if (cp2) { cp2.textContent = fmtUsd(cost.projected_total_usd); cp2.style.color = cost.projected_total_usd != null ? 'var(--text)' : 'var(--text-muted)'; }
    const ct = document.getElementById('c-ttc');
    if (ct) { ct.textContent = cost.eta_sec ? fmtTime(cost.eta_sec) : job.status === 'succeeded' ? 'done' : '\u2014'; ct.style.color = job.status === 'succeeded' ? 'var(--green)' : 'var(--text)'; }
    const cts = document.getElementById('c-ttc-sub');
    if (cts) cts.textContent = job.slo_hours ? 'slo: ' + job.slo_hours + 'h' : 'projected';
  }
  // Throughput in cost bar
  const tpsEl = document.getElementById('c-tps');
  if (tpsEl) {
    const tps = m.avg_generation_throughput_toks_per_s;
    tpsEl.textContent = tps ? fmt(tps) + ' tok/s' : '\u2014';
    tpsEl.style.color = tps ? 'var(--green)' : 'var(--text-muted)';
  }

  // Resource pool
  const rl = document.getElementById('res-list');
  if (rl) {
    const groups = {};
    reps.forEach(r => {
      const key = (r.instance_type||'unknown') + '|' + (r.region||'');
      if (!groups[key]) groups[key] = {inst: r.instance_type, region: r.region, count: 0, active: false};
      groups[key].count++;
      if (ACTIVE.has(r.phase)) groups[key].active = true;
    });
    let rh = '';
    for (const k in groups) {
      const g = groups[k];
      const on = g.active ? ' on' : '';
      rh += '<div class="res-row' + on + '"><span class="res-cnt">' + g.count + '\u00d7</span><span class="res-gpu">' + esc(gpuName(g.inst)) + '</span><span class="res-rgn">' + esc(g.region||'') + '</span><span class="res-ind' + on + '"></span></div>';
    }
    if (!rh) rh = '<div style="padding:8px;font-size:10px;color:var(--text-muted)">No replicas</div>';
    rl.innerHTML = rh;
  }

  // Job buttons
  const jb = document.getElementById('jbtns');
  if (jb) {
    let jh = '';
    jobs.forEach(j => {
      const p = Math.round((j.progress||0) * 100);
      const label = j.status === 'succeeded' ? 'done' : j.status === 'failed' ? 'fail' : p + '%';
      const act = j.job_id === activeJobId ? ' act' : '';
      const col = j.job_id === activeJobId ? 'var(--cyan)' : 'var(--text-muted)';
      jh += '<button class="jbtn' + act + '" data-jid="' + esc(j.job_id) + '"><span class="jbtn-dot" style="background:' + col + '"></span><span class="jbtn-name">' + esc(j.model_name||j.job_id.slice(0,12)) + '</span><span class="jbtn-pct" style="color:' + col + '">' + label + '</span></button>';
    });
    jb.innerHTML = jh;
    jb.querySelectorAll('.jbtn').forEach(btn => {
      btn.onclick = function() { activeJobId = this.dataset.jid; render(lastData); };
    });
  }

  // Event log
  const la = document.getElementById('log-area');
  if (la && data.events && data.events.length) {
    // Merge new events
    const existing = new Set(eventBuffer.map(e => e.ts + e.message));
    data.events.forEach(ev => {
      const key = ev.ts + ev.message;
      if (!existing.has(key)) { eventBuffer.push(ev); existing.add(key); }
    });
    while (eventBuffer.length > MAX_EVENTS) eventBuffer.shift();
    // Render
    la.innerHTML = '';
    eventBuffer.forEach(ev => {
      const d = document.createElement('div');
      d.className = 'll ' + (ev.level || 'dim');
      const t = new Date(ev.ts * 1000).toLocaleTimeString('en-US', {hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit'});
      d.textContent = '[' + t + '] ' + ev.message;
      la.appendChild(d);
    });
    la.scrollTop = la.scrollHeight;
  }

  // Charts
  if (chartsVisible) {
    const cw = document.getElementById('charts-wrap');
    const cid = sid(activeJobId);
    // Rebuild chart canvases if job changed
    if (cw && cw.dataset.jid !== activeJobId) {
      cw.dataset.jid = activeJobId;
      let ch2 = '<div class="charts-grid">';
      CDEFS.forEach(d => { ch2 += '<div class="chart-wrap"><canvas id="chart-' + d.key + '-' + cid + '"></canvas></div>'; });
      ch2 += '</div>';
      cw.innerHTML = ch2;
      chartMgr.cleanup(new Set([activeJobId]));
    }
  }

  // Timeseries accumulation + SSE resilience
  if (data.timeseries && data.timeseries[activeJobId]) {
    const serverTs = data.timeseries[activeJobId];
    if (serverTs.length > 0 && (!jobTimeseries[activeJobId] || jobTimeseries[activeJobId].length < serverTs.length)) {
      jobTimeseries[activeJobId] = serverTs;
    }
  }
  if (isActive && Object.keys(m).length > 0) {
    if (!jobTimeseries[activeJobId]) jobTimeseries[activeJobId] = [];
    jobTimeseries[activeJobId].push(Object.assign({timestamp: Date.now()/1000}, m));
    if (jobTimeseries[activeJobId].length > MAX_TS) jobTimeseries[activeJobId] = jobTimeseries[activeJobId].slice(-MAX_TS);
  }
  if (chartsVisible) chartMgr.update(activeJobId, jobTimeseries[activeJobId] || []);

  // Cleanup old timeseries
  const curIds = new Set(jobs.map(j => j.job_id));
  for (const jid in jobTimeseries) { if (!curIds.has(jid)) delete jobTimeseries[jid]; }
  chartMgr.cleanup(curIds);
}

function renderChainSVG(reps, isActive) {
  const svg = document.getElementById('chain-svg');
  if (!svg) return;
  const W = svg.parentElement.clientWidth || 400, H = 66;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const n = reps.length;
  if (!n) { svg.innerHTML = '<text x="' + W/2 + '" y="33" text-anchor="middle" fill="#6e7681" font-size="10" font-family="JetBrains Mono">No replicas</text>'; return; }
  const bw = 80, bh = 34, gap = 40, gy = H/2 - bh/2;
  const totalW = n * bw + (n-1) * gap;
  const sx = Math.max(8, (W - totalW) / 2);
  let html = '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="var(--cyan)" stroke-width="1.4" stroke-linecap="round"/></marker></defs>';
  for (let i = 0; i < n; i++) {
    const r = reps[i];
    const x = sx + i * (bw + gap);
    const phaseColors = {running:'var(--cyan)',generating:'var(--cyan)',model_ready:'var(--blue)',loading_model:'var(--blue)',launching:'var(--yellow)',failed:'var(--red)',dead:'var(--red)'};
    const col = phaseColors[r.phase] || 'var(--text-muted)';
    const on = ACTIVE.has(r.phase);
    // Connector line
    if (i < n - 1) {
      const ly = gy + bh/2;
      html += '<line x1="' + (x+bw) + '" y1="' + ly + '" x2="' + (x+bw+gap) + '" y2="' + ly + '" stroke="' + (on?'var(--cyan)':'#30363d') + '" stroke-width="1" marker-end="url(#arr)"/>';
      // Animated packet
      if (isActive && on) {
        html += '<circle r="3" fill="var(--cyan)" opacity="0.7"><animateMotion dur="2s" repeatCount="indefinite" path="M' + (x+bw) + ',' + ly + ' L' + (x+bw+gap) + ',' + ly + '"/></circle>';
      }
    }
    // Node box
    html += '<rect x="' + x + '" y="' + gy + '" width="' + bw + '" height="' + bh + '" rx="3" fill="var(--card)" stroke="' + col + '" stroke-width="' + (on?'1.5':'0.5') + '"/>';
    html += '<text x="' + (x+bw/2) + '" y="' + (gy+12) + '" text-anchor="middle" fill="' + (on?'var(--text)':'#444c56') + '" font-size="9" font-family="JetBrains Mono">' + esc(gpuName(r.instance_type)) + '</text>';
    html += '<text x="' + (x+bw/2) + '" y="' + (gy+24) + '" text-anchor="middle" fill="' + col + '" font-size="8" font-family="JetBrains Mono">' + esc((r.region||'').slice(-6)) + '</text>';
  }
  svg.innerHTML = html;
}

let lastData = {}, pollTimer = null, usePolling = false, sseGotData = false;

function poll() {
  fetch(serverUrl + '/dashboard/poll').then(r => r.json()).then(data => {
    setConn('connected');
    lastData = data; render(lastData);
  }).catch(() => setConn('disconnected'));
}

function startPolling() {
  if (pollTimer) return;
  usePolling = true;
  console.log('SSE unavailable, falling back to polling');
  poll();
  pollTimer = setInterval(poll, 2000);
}

function connect() {
  if (es) { try { es.close(); } catch(e){} }
  setConn('connecting');
  sseGotData = false;
  es = new EventSource(serverUrl + '/dashboard/stream');
  // If SSE delivers nothing in 5s, switch to polling
  const sseTimeout = setTimeout(function() { if (!sseGotData) { es.close(); es = null; startPolling(); } }, 5000);
  es.onopen = function() { setConn('connected'); };
  es.onmessage = function(ev) {
    sseGotData = true; clearTimeout(sseTimeout);
    try { lastData = JSON.parse(ev.data); render(lastData); } catch(e) { console.error('parse error', e); }
  };
  es.onerror = function() {
    clearTimeout(sseTimeout);
    setConn('disconnected'); es.close(); es = null;
    if (!sseGotData) { startPolling(); return; }
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

def _build_dashboard_payload(app_state) -> dict:
    """Build the dashboard payload dict (shared by SSE stream and poll endpoint)."""
    from orca_server.job_manager import get_job_tracker
    from orca_server.chunk_manager import get_chunk_manager

    payload = {"jobs": [], "metrics": {}, "chunks": {}, "replicas": {}}
    mc = getattr(app_state, "metrics_collector", None)
    cluster_mgr = getattr(app_state, "cluster_manager", None)
    redis_ok = getattr(app_state, "redis_available", False)

    try:
        tracker = get_job_tracker()
        with tracker.lock:
            job_items = list(tracker.jobs.items())

        for job_id, rec in job_items:
            try:
                payload["jobs"].append({
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
                })
            except Exception:
                logger.debug("dashboard: error serialising job %s", job_id, exc_info=True)

        # Metrics
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

        # Chunks
        if redis_ok:
            try:
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

        # Replicas
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

        # Cost
        payload["cost"] = {}
        now = time.time()
        for job_id, rec in job_items:
            try:
                instance_type = rec.state.instance_types
                # Fall back to replica instance_type if job-level isn't set yet
                if not instance_type and cluster_mgr:
                    for _rid, rs in cluster_mgr.get_replica_states(job_id).items():
                        if rs.get("instance_type"):
                            instance_type = rs["instance_type"]
                            break
                if not instance_type:
                    continue
                region = rec.state.spec.region or "us-east-1"
                market = rec.state.spec.market or "spot"
                price = _get_cached_price(instance_type, region, market)
                if price is None:
                    continue
                total_hours = 0.0
                num_running = 0
                try:
                    if cluster_mgr:
                        for _rid, rs in cluster_mgr.get_replica_states(job_id).items():
                            rs_since = rs.get("running_since")
                            if rs_since and rs.get("phase") in ACTIVE_PHASES:
                                total_hours += (now - rs_since) / 3600
                                num_running += 1
                except Exception:
                    pass
                if total_hours == 0 and rec.status in ACTIVE_PHASES:
                    total_hours = (now - rec.state.submitted_at) / 3600
                    num_running = getattr(rec, "num_replicas", 1) or 1
                accrued = price * total_hours
                progress = rec.state.progress_frac
                projected = accrued / progress if progress > 0.01 else None
                eta_sec = ((1.0 - progress) / progress) * total_hours * 3600 if progress > 0.01 else None
                payload["cost"][job_id] = {
                    "price_per_hour": round(price, 4),
                    "accrued_usd": round(accrued, 4),
                    "projected_total_usd": round(projected, 4) if projected else None,
                    "eta_sec": round(eta_sec) if eta_sec else None,
                    "num_running_replicas": num_running,
                }
            except Exception:
                logger.debug("dashboard: cost error for %s", job_id, exc_info=True)

        # Synthetic events
        for job_id, rec in job_items:
            jid_short = job_id[:12]
            prev_st = _prev_job_status.get(job_id)
            if prev_st is not None and prev_st != rec.status:
                lvl = "error" if rec.status == "failed" else "ok" if rec.status == "succeeded" else "info"
                _emit_event(lvl, f"{jid_short} {prev_st} -> {rec.status}", job_id)
            _prev_job_status[job_id] = rec.status
            ch = payload["chunks"].get(job_id)
            if ch and ch.get("total", 0) > 0:
                prev_ch = _prev_chunk_progress.get(job_id, {})
                prev_pct = prev_ch.get("completed", 0) / ch["total"] * 100 if prev_ch.get("completed") is not None else 0
                cur_pct = ch["completed"] / ch["total"] * 100
                for ms in (25, 50, 75, 100):
                    if prev_pct < ms <= cur_pct:
                        _emit_event("ok", f"{jid_short} chunks {ms}% ({ch['completed']}/{ch['total']})", job_id)
                _prev_chunk_progress[job_id] = dict(ch)
            reps = payload["replicas"].get(job_id, [])
            prev_phases = _prev_replica_phases.get(job_id, {})
            cur_phases = {}
            for r in reps:
                rid, phase = r["replica_id"], r["phase"]
                cur_phases[rid] = phase
                if rid in prev_phases and prev_phases[rid] != phase:
                    _emit_event("error" if phase in ("failed", "dead") else "info", f"replica {rid[-8:]} -> {phase}", job_id)
            _prev_replica_phases[job_id] = cur_phases
        payload["events"] = list(_event_log)[-50]

        # Timeseries
        payload["timeseries"] = {}
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
        logger.debug("dashboard: top-level payload error", exc_info=True)

    return payload


@dashboard_router.get("/dashboard")
async def serve_dashboard():
    """Serve the Orca web dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


@dashboard_router.get("/dashboard/poll")
async def dashboard_poll(request: Request):
    """REST endpoint returning the same payload as the SSE stream (for proxy-hostile envs)."""
    return _build_dashboard_payload(request.app.state)


@dashboard_router.get("/dashboard/stream")
async def dashboard_stream(request: Request):
    """SSE endpoint streaming fleet-wide job/metrics/chunk data every 2 s."""

    async def _generate():
        yield ": connected\nretry: 3000\n\n"
        while True:
            if await request.is_disconnected():
                break
            payload = _build_dashboard_payload(request.app.state)
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
