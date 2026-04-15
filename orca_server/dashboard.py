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
_peak_cost: dict[str, dict] = {}  # job_id -> last known cost dict (persists after completion)

# Synthetic event log
_event_log: deque[dict] = deque(maxlen=200)
_prev_job_status: dict[str, str] = {}
_prev_chunk_progress: dict[str, dict] = {}
_prev_replica_phases: dict[str, dict[str, str]] = {}

ACTIVE_PHASES = {"launching", "loading_model", "model_ready", "generating", "running"}
TERMINAL_REPLICA_PHASES = {"completed", "failed", "dead", "killed", "swapped_out"}


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
# Dashboard HTML (loaded from external file to keep this module clean)
# ---------------------------------------------------------------------------

from pathlib import Path as _Path
DASHBOARD_HTML = (_Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _build_dashboard_payload(app_state) -> dict:
    """Build the dashboard payload dict (shared by SSE stream and poll endpoint)."""
    from orca_server.job_manager import get_job_tracker
    from orca_server.chunk_manager import get_chunk_manager

    payload = {"jobs": [], "metrics": {}, "chunks": {}, "replicas": {}, "quota": []}
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

        # Enrich progress with per-request granularity from metrics
        _enriched_progress = {}
        for job_id, rec in job_items:
            base = rec.state.progress_frac
            m = payload["metrics"].get(job_id, {})
            rst = m.get("request_success_total", 0)
            nl = rec.state.spec.num_lines or 0
            if rst > 0 and nl > 0 and base < 1.0:
                _enriched_progress[job_id] = max(base, min(rst / nl, 0.99))
            else:
                _enriched_progress[job_id] = base
        # Patch jobs array with enriched progress
        for j in payload["jobs"]:
            if j["job_id"] in _enriched_progress:
                j["progress"] = round(_enriched_progress[j["job_id"]], 4)

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
                            rep_info = {
                                "replica_id": rid,
                                "phase": rstate.get("phase", "unknown"),
                                "region": rstate.get("region", ""),
                                "market": rstate.get("market", ""),
                                "instance_type": rstate.get("instance_type", ""),
                                "has_metrics": rstate.get("has_metrics", False),
                                "running_since": rstate.get("running_since"),
                            }
                            # Per-replica metrics
                            if mc is not None:
                                rsnap = mc.get_replica_latest(job_id, rid)
                                if rsnap:
                                    rep_info["request_success_total"] = rsnap.request_success_total
                                    rep_info["throughput_toks"] = getattr(rsnap, "avg_generation_throughput_toks_per_s", None)
                            replicas.append(rep_info)
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
                    if cluster_mgr and rec.status in ACTIVE_PHASES:
                        for _rid, rs in cluster_mgr.get_replica_states(job_id).items():
                            if rs.get("phase") in TERMINAL_REPLICA_PHASES:
                                continue
                            start_ts = (
                                rs.get("launched_at")
                                or rs.get("running_since")
                                or rec.state.submitted_at
                            )
                            elapsed_hours = max(0.0, now - start_ts) / 3600
                            total_hours += elapsed_hours * max(1, rs.get("num_instances") or 1)
                            num_running += 1
                except Exception:
                    pass
                if total_hours == 0 and rec.status in ACTIVE_PHASES:
                    num_running = getattr(rec, "num_replicas", 1) or 1
                    total_hours = (now - rec.state.submitted_at) / 3600 * num_running
                accrued = price * total_hours
                progress = _enriched_progress.get(job_id, rec.state.progress_frac)
                projected = accrued / progress if progress > 0.01 else None
                eta_sec = ((1.0 - progress) / progress) * total_hours * 3600 if progress > 0.01 else None
                cost_data = {
                    "price_per_hour": round(price, 4),
                    "accrued_usd": round(accrued, 4),
                    "projected_total_usd": round(projected, 4) if projected else None,
                    "eta_sec": round(eta_sec) if eta_sec else None,
                    "num_running_replicas": num_running,
                }
                # Persist peak cost so it survives after replicas stop
                if accrued > 0:
                    _peak_cost[job_id] = cost_data
                elif job_id in _peak_cost:
                    cost_data = _peak_cost[job_id]
                    cost_data["num_running_replicas"] = 0
                    cost_data["eta_sec"] = None
                payload["cost"][job_id] = cost_data
            except Exception:
                logger.debug("dashboard: cost error for %s", job_id, exc_info=True)
        # Fall back to peak cost for jobs where instance_type wasn't found
        for job_id, _rec in job_items:
            if job_id not in payload["cost"] and job_id in _peak_cost:
                payload["cost"][job_id] = _peak_cost[job_id]

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
        payload["events"] = list(_event_log)[-50:]

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

        # Quota
        try:
            qt = getattr(app_state, "quota_tracker", None)
            if qt is not None and hasattr(qt, "full_quota_summary"):
                summary = qt.full_quota_summary()
                if not summary.empty:
                    payload["quota"] = summary.to_dict("records")
        except Exception:
            pass

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
