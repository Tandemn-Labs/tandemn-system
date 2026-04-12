from contextlib import asynccontextmanager
import asyncio
import logging
import math
import uuid
import time
import os
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sky
from fastapi import FastAPI, Form, Header, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

from orca_server.config import (
    AWS_INSTANCES,
    CHUNK_RECLAIM_INTERVAL,
    CHUNK_SIZE_BYTES,
    INSTANCE_TO_GPU,
    ORCA_API_KEY,
    PLACEMENT_PRIORITY,
    PLACEMENT_SOLVER,
    S3_UPLOAD_BUCKET,
    S3_UPLOAD_PREFIX,
)
from orca_server.monitoring import MetricsSnapshot
from orca_server.input_parser import parse_input_file_stats
from orca_server.job_manager import get_cluster_manager, get_job_tracker, jobtracker_snapshot
from orca_server.launcher import (
    sp_launch_vllm_batch_with_fallback,
    sp_launch_vllm_online,
    launch_chunked_replicas,
    _launch_chunked_replica,
)
from orca_server.job_manager import sky_down_with_retry
import threading
from orca_server.chunk_manager import get_chunk_manager
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from placement.roofline_magic import (
    RooflineAWSAllocation,
    resolve_gpu_type_to_instance,
    check_user_specified_feasibility,
)
from quota.region_selector import (
    get_ordered_regions,
    get_instance_family,
    get_cached_quotas,
)
from storage.storage_factory import get_storage_backend
from orca_server.job_templates import real_magic
from quota.tracker import VPCQuotaTracker


from orca_server.utils import make_job_id as _make_job_id


# ---------------------------------------------------------------------------
# Replica log forwarding
# ---------------------------------------------------------------------------
_replica_log_locks: dict[str, threading.Lock] = {}
_replica_log_locks_lock = threading.Lock()

# Per-replica previous snapshot for computing throughput deltas at ingest time
_ingest_prev_snaps: dict[str, "MetricsSnapshot"] = {}


def _write_replica_logs(job_id: str, replica_id: str, log_lines: list):
    """Append forwarded log lines to the per-replica log file."""
    tracker = get_job_tracker()
    rec = tracker.get(job_id)
    if not rec:
        return
    job_dirname = getattr(rec, "_job_dirname", None)
    if not job_dirname:
        return

    replica_dir = Path(f"outputs/{job_dirname}/replicas")
    replica_dir.mkdir(parents=True, exist_ok=True)
    log_path = replica_dir / f"{replica_id}.log"

    # Per-file lock to prevent interleaved writes from concurrent requests
    with _replica_log_locks_lock:
        if replica_id not in _replica_log_locks:
            _replica_log_locks[replica_id] = threading.Lock()
        lock = _replica_log_locks[replica_id]

    with lock:
        with open(log_path, "a") as f:
            for entry in log_lines:
                ts = entry.get("ts", 0)
                msg = entry.get("msg", "")
                if msg:
                    timestr = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                    f.write(f"{timestr}  {msg}\n")


async def _resolve_input_file(input_file: str) -> tuple[str, str | None]:
    """If input_file is an S3 URI, download via storage backend to a temp file.

    Returns (local_path, tmp_path_to_cleanup_or_None).
    """
    if input_file.startswith("s3://"):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        await storage_backend.download_file(input_file, tmp_path, user="system")
        return tmp_path, tmp_path
    return input_file, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Validate critical env vars ──
    from orca_server.config import ORCA_SERVER_URL, HF_TOKEN
    if not ORCA_SERVER_URL or ORCA_SERVER_URL == "placeholder":
        logger.error(
            "[Startup] ⚠ ORCA_SERVER_URL is not set! Replicas will not be able to "
            "call back to the control plane. Set it in .env or pass --url/--tunnel. "
            "Jobs WILL fail silently without this."
        )
    else:
        logger.info(f"[Startup] ORCA_SERVER_URL = {ORCA_SERVER_URL}")
    if not HF_TOKEN:
        logger.warning(
            "[Startup] HF_TOKEN not set — gated models (Llama, Gemma) will fail to download. "
            "Set HF_TOKEN in .env if you need gated models."
        )

    # Initialize cluster manager
    app.state.cluster_manager = get_cluster_manager()

    # Initialize quota tracker (replays persisted reservations from SQLite)
    app.state.quota_tracker = VPCQuotaTracker()

    # Refresh AWS quotas in background (non-blocking)
    import threading
    from quota.region_selector import refresh_quotas_from_aws
    threading.Thread(
        target=refresh_quotas_from_aws,
        kwargs={"quota_tracker": app.state.quota_tracker},
        daemon=True,
    ).start()

    from orca_server.monitoring import get_metrics_collector
    from orca_server.metrics_db import get_metrics_db
    app.state.metrics_collector = get_metrics_collector()
    app.state.metrics_db = get_metrics_db()

    # Redis health check — required for chunked multi-replica jobs
    try:
        import redis as _redis
        from orca_server.config import REDIS_URL
        _r = _redis.from_url(REDIS_URL, socket_connect_timeout=3, socket_timeout=3)
        _r.ping()
        logger.info(f"[Redis] Connected at {REDIS_URL}")
        app.state.redis_available = True
    except Exception as _e:
        logger.warning(
            f"[Redis] ⚠ Unavailable at {REDIS_URL}: {_e} — "
            "all batch jobs will fail (chunked mode is the default path). "
            "Start Redis: docker run -d -p 6379:6379 redis"
        )
        app.state.redis_available = False

    # Reconcile stale reservations against live SkyPilot clusters
    try:
        request_id = sky.status()
        clusters = sky.get(request_id)
        live = {c['name'] for c in clusters} if clusters else set()
        app.state.quota_tracker.reconcile(live)
    except Exception as e:
        logger.warning(f"[Quota] Could not reconcile on startup: {e}")

    # Active reclaim loop: scans all jobs every CHUNK_RECLAIM_INTERVAL seconds
    # and re-queues inflight chunks whose leases have expired.
    async def _reclaim_loop():
        while True:
            await asyncio.sleep(CHUNK_RECLAIM_INTERVAL)
            if not getattr(app.state, "redis_available", False):
                continue
            try:
                cm = get_chunk_manager()
                keys = list(cm._r.scan_iter("chunk:job:*:inflight"))
                for key in keys:
                    # key format: "chunk:job:{job_id}:inflight"
                    prefix = "chunk:job:"
                    suffix = ":inflight"
                    job_id = key[len(prefix):-len(suffix)]
                    result = cm.reclaim_expired_chunks(job_id)
                    if result["reclaimed"] or result["failed"]:
                        logger.info(
                            f"[Reclaim] {job_id}: reclaimed={result['reclaimed']} "
                            f"failed={result['failed']}"
                        )
                    # If reclaim pushed a job to all_done, trigger assembly
                    if result["failed"]:
                        progress = cm.get_progress(job_id)
                        if progress and progress["all_done"]:
                            asyncio.create_task(_assemble_output(job_id))
            except Exception as e:
                logger.warning(f"[Reclaim] Error: {e}")

    reclaim_task = asyncio.create_task(_reclaim_loop())

    # Replica watchdog: detect dead replicas via heartbeat, force-reclaim.
    # Recovery is Koi's responsibility — watchdog fires /job/replica-failed,
    # Koi's agent decides config, calls scale_chain_tool → Orca /job/{id}/scale.
    watchdog_task = None
    if getattr(app.state, "redis_available", False):
        from orca_server.watchdog import ReplicaWatchdog

        watchdog = ReplicaWatchdog(
            metrics_collector=app.state.metrics_collector,
            cluster_manager=app.state.cluster_manager,
            job_tracker=get_job_tracker(),
            chunk_manager_fn=get_chunk_manager,
            assembly_callback=lambda jid: asyncio.create_task(_assemble_output(jid)),
        )
        app.state.watchdog = watchdog
        watchdog_task = asyncio.create_task(watchdog.run())
        logger.info("[Watchdog] Started replica watchdog")

    yield

    if watchdog_task:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
    reclaim_task.cancel()
    try:
        await reclaim_task
    except asyncio.CancelledError:
        pass

    # Shutdown: join non-daemon monitor threads so they can finish teardown
    cm = app.state.cluster_manager
    threads = cm.get_active_threads()
    stale_clusters = []
    if threads:
        logger.info(f"[Shutdown] Waiting for {len(threads)} monitor thread(s) to finish teardown...")
        for name, t in threads.items():
            logger.info(f"[Shutdown] Joining thread for cluster {name}...")
            t.join(timeout=120)
            if t.is_alive():
                logger.warning(f"[Shutdown] Thread for {name} did not finish within 120s")
                stale_clusters.append(name)
        logger.info("[Shutdown] All monitor threads joined.")

    # Force-teardown clusters whose monitor threads didn't finish in time
    for name in stale_clusters:
        logger.info(f"[Shutdown] Force-tearing down orphaned cluster {name}...")
        try:
            sky_down_with_retry(name)
        except Exception as e:
            logger.error(f"[Shutdown] Failed to tear down {name}: {e}")


app = FastAPI(
    title="Tandemn Server",
    description="API for receiving job requests",
    lifespan=lifespan,
)

from orca_server.dashboard import dashboard_router
app.include_router(dashboard_router)

storage_backend = get_storage_backend()


def get_quota_tracker():
    return app.state.quota_tracker


@app.get("/quota/status")
async def quota_status():
    """Get current quota usage summary and active cluster reservations."""
    tracker = get_quota_tracker()
    summary = tracker.status_summary()
    reservations = tracker.get_reservations()
    return {
        "status": "success",
        "quota_usage": summary.to_dict(orient="records"),
        "active_reservations": reservations,
    }


# ---- Pricing via SkyPilot catalog (always up-to-date) ----
import sky.catalog as _sky_catalog

_pricing_cache: dict[str, float] = {}

def _get_instance_price(instance_type: str, region: str) -> float:
    """Get on-demand hourly price from SkyPilot's AWS catalog. Cached."""
    key = f"{instance_type}:{region}"
    if key in _pricing_cache:
        return _pricing_cache[key]
    try:
        cost = _sky_catalog.get_hourly_cost(
            instance_type=instance_type, use_spot=False,
            region=region, zone=None, clouds="aws",
        )
        _pricing_cache[key] = cost
        return cost
    except Exception as e:
        logger.warning(f"[Resources] Price lookup failed for {instance_type} in {region}: {e}")
        return None

# GPU types Koi supports (matches Koi's GPU_SPECS)
_KOI_GPU_TYPES = {"H100", "A100", "L40S", "L4", "A10G"}

# Multi-GPU instances useful for LLM inference (mirrors roofline_magic filter)
_KOI_INSTANCE_PREFIXES = (
    "p5.", "p4d.", "p4de.",
    "g6e.12xlarge", "g6e.24xlarge", "g6e.48xlarge",
    "g5.12xlarge", "g5.24xlarge", "g5.48xlarge",
    "g6.12xlarge", "g6.24xlarge", "g6.48xlarge",
)


@app.get("/resources")
async def resources():
    """Raw instance catalog + quota pools for Koi.

    Returns two lists:
      instances — one entry per relevant instance type with GPU specs and pricing
      quotas   — per (family, region, market) vCPU limits from AWS

    Koi's Oracle joins these: for a candidate (instance_type, TP, PP, DP),
    it looks up the instance's quota_family, finds the best region, and checks
    whether required_vcpus <= available_vcpus.
    """
    tracker = get_quota_tracker()

    def _build_catalog():
        quota_df = tracker.quota_df
        summary = tracker.full_quota_summary()

        instances = []
        for inst_type, (gpu_name, gpu_count, vcpus, vram) in AWS_INSTANCES.items():
            if gpu_name not in _KOI_GPU_TYPES:
                continue
            if not inst_type.startswith(_KOI_INSTANCE_PREFIXES):
                continue

            inst_row = quota_df[quota_df["Instance_Type"] == inst_type]
            if inst_row.empty:
                continue
            family_type = inst_row["Family_Type"].iloc[0]

            # us-east-1 used as reference region; Koi picks actual region from quotas
            price = _get_instance_price(inst_type, "us-east-1")
            if price is None:
                continue

            instances.append({
                "instance_type": inst_type,
                "gpu_type": gpu_name,
                "gpus_per_instance": gpu_count,
                "vcpus": vcpus,
                "quota_family": family_type,
                "gpu_memory_gb": float(vram),
                "interconnect": "NVLink" if inst_type.startswith("p") else "PCIe",
                "cost_per_instance_hour_usd": round(price, 4),
            })

        quotas = []
        for _, row in summary.iterrows():
            baseline = int(row["Baseline"])
            if baseline <= 0:
                continue
            quotas.append({
                "family": row["Family"],
                "region": row["Region"],
                "market": row["Market"],
                "baseline_vcpus": baseline,
                "used_vcpus": int(row["Used"]),
            })
        # Compute allocated GPUs from running clusters (ground truth)
        cm = app.state.cluster_manager
        allocated_gpus = {}
        with cm.lock:
            for _, info in cm.active_clusters.items():
                inst = info.get("instance_type")
                n = info.get("num_instances", 0)
                if inst in AWS_INSTANCES:
                    gpu_name, gpu_count, _, _ = AWS_INSTANCES[inst]
                    allocated_gpus[gpu_name] = allocated_gpus.get(gpu_name, 0) + gpu_count * n

        return instances, quotas, allocated_gpus

    instances, quotas, allocated_gpus = await asyncio.to_thread(_build_catalog)

    return {
        "vpc_id": "orca-cluster",
        "snapshot_time": datetime.now().isoformat(),
        "instances": instances,
        "quotas": quotas,
        "allocated_gpus": allocated_gpus,
    }


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Get status and progress for a specific job."""
    tracker = get_job_tracker()
    rec = tracker.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {
        "job_id": job_id,
        "status": rec.status,
        "progress": round(rec.state.progress_frac, 4),
        "num_lines": rec.state.spec.num_lines,
        "model_name": rec.state.spec.model_name,
        "head_ip": rec.head_ip,
        "endpoint_url": rec.endpoint_url,
        "created_at": rec.created_at,
        "last_updated_at": rec.last_updated_at,
    }


@app.get("/jobs")
async def list_jobs():
    """List all tracked jobs."""
    tracker = get_job_tracker()
    with tracker.lock:
        return {"jobs": [
            {
                "job_id": jid,
                "status": rec.status,
                "progress": round(rec.state.progress_frac, 4),
                "model_name": rec.state.spec.model_name,
                "num_lines": rec.state.spec.num_lines,
                "created_at": rec.created_at,
            }
            for jid, rec in tracker.jobs.items()
        ]}


@app.get("/job/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Get latest live metrics snapshot for a running job."""
    snap = app.state.metrics_collector.get_aggregated(job_id)
    if snap is None:
        raise HTTPException(404, f"No metrics for {job_id}")
    return snap.to_dict()


@app.get("/job/{job_id}/replicas")
async def get_job_replicas(job_id: str):
    """Get per-replica state and metrics availability for a chunked job."""
    cm = app.state.cluster_manager
    mc = app.state.metrics_collector
    states = cm.get_replica_states(job_id)
    metrics_ids = set(mc.list_replica_ids(job_id))
    replicas = []
    for rid, info in states.items():
        replicas.append({
            "replica_id": rid,
            "phase": info.get("phase", "unknown"),
            "region": info.get("region"),
            "market": info.get("market"),
            "instance_type": info.get("instance_type"),
            "has_metrics": rid in metrics_ids,
        })
    return {"replicas": replicas}


@app.get("/job/{job_id}/replicas/{replica_id}/metrics")
async def get_replica_metrics(job_id: str, replica_id: str):
    """Get latest live metrics snapshot for a specific replica."""
    snap = app.state.metrics_collector.get_replica_latest(job_id, replica_id)
    if snap is None:
        raise HTTPException(404, f"No metrics for replica {replica_id}")
    return snap.to_dict()


@app.get("/job/{job_id}/metrics/stream")
async def stream_job_metrics(job_id: str):
    """SSE stream of live metrics for a running job (1 event/sec)."""
    async def _gen():
        loop = asyncio.get_running_loop()
        gen = app.state.metrics_collector.sse_generator(job_id)
        while True:
            try:
                chunk = await loop.run_in_executor(None, next, gen)
                yield chunk
            except StopIteration:
                break
    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus exposition of live metrics for all active jobs."""
    text = app.state.metrics_collector.prometheus_exposition()
    return Response(content=text, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/debug/inject-replica")
async def debug_inject_replica(request: Request):
    """
    Debug/test only — inject a fake running replica so the watchdog can be
    tested without real GPUs.

    POST /debug/inject-replica
    {
        "job_id":     "test-job-1",
        "replica_id": "test-job-1-r0",
        "num_chunks": 10
    }

    After calling this, pump heartbeats via POST /job/{job_id}/metrics/ingest
    (replica_id + snapshots). Stop pumping and the watchdog will detect death
    after REPLICA_DEAD_THRESHOLD_SEC seconds.
    """
    body = await request.json()
    job_id     = body["job_id"]
    replica_id = body["replica_id"]
    num_chunks = body.get("num_chunks", 10)

    jt = get_job_tracker()
    cm = app.state.cluster_manager
    mc = app.state.metrics_collector

    # 1. Create job record with is_chunked=True
    from orca_server.job_manager import JobRecord
    from quota.tracker import JobSpec, JobState
    with jt.lock:
        if job_id not in jt.jobs:
            spec = JobSpec(job_id=job_id, model_name="debug-model", num_lines=1000,
                           avg_input_tokens=512, avg_output_tokens=256, slo_hours=2.0)
            state = JobState(spec=spec, submitted_at=time.time())
            jt.jobs[job_id] = JobRecord(state=state, status="generating")
    jt.set_chunked_info(job_id, num_chunks, 1)

    # 2. Register replica as running in cluster manager
    cm.set_replica_state(job_id, replica_id, phase="running", running_since=time.time())

    # 3. Start metrics collection so ring buffer exists
    mc.start_collecting(job_id)
    mc.start_replica_collecting(job_id, replica_id)

    return {"ok": True, "job_id": job_id, "replica_id": replica_id,
            "msg": "Now POST /job/{job_id}/metrics/ingest to pump heartbeats. Stop to trigger watchdog."}


@app.post("/job/{job_id}/metrics/ingest")
async def ingest_job_metrics(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Called by in-cluster sidecar every 5s with an accumulated batch of Prometheus snapshots."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    snapshots_raw = body.get("snapshots", [])
    replica_id = body.get("replica_id")

    # Update progress bar if runner included done/total counts
    done = body.get("done")
    total = body.get("total")
    if done is not None and total and total > 0:
        from orca_server.job_manager import get_job_tracker
        get_job_tracker().update_progress(job_id, done / total)

    if not snapshots_raw:
        return {"ok": True, "ingested": 0}

    mc = app.state.metrics_collector
    db = app.state.metrics_db

    # Ensure job-level and per-replica collectors exist (auto-create on first ingest)
    mc.start_collecting(job_id)
    if replica_id:
        mc.start_replica_collecting(job_id, replica_id)
        # Update phase to "running" on first ingest (handles recovered replicas)
        cm = app.state.cluster_manager
        states = cm.get_replica_states(job_id)
        if replica_id in states and states[replica_id].get("phase") in ("launching", "provisioned", "dead"):
            cm.set_replica_state(job_id, replica_id, phase="running", running_since=time.time())
        # Clear watchdog dead tracking if this replica recovered
        wd = getattr(app.state, "watchdog", None)
        if wd:
            wd.clear_dead(replica_id)
        # Un-exclude from metrics aggregation if previously excluded (replica ID reuse)
        rkey = f"{job_id}:{replica_id}"
        if rkey in mc._excluded_replicas:
            mc._excluded_replicas.discard(rkey)
            mc._excluded_cumulative.pop(rkey, None)
            logger.info("[MetricsCollector] Un-excluded recovered replica %s", rkey)

    ingested = 0
    batch_for_db = []
    prev_snap = None

    for item in snapshots_raw:
        ts   = item.get("timestamp", time.time())
        text = item.get("prometheus_text", "")
        if not text.strip():
            continue

        snap = MetricsSnapshot.from_prometheus_text(job_id, text, ts)
        snap.replica_id = replica_id

        # Merge live per-token counters from SSE stream (smooth, no completion-burst)
        live_gen = item.get("live_gen_tokens_total")
        live_prompt = item.get("live_prompt_tokens_total")
        if live_gen is not None:
            snap.live_gen_tokens_total = float(live_gen)
        if live_prompt is not None:
            snap.live_prompt_tokens_total = float(live_prompt)

        # Compute throughput from counter deltas and persist in the snapshot.
        # Ring buffer still computes its own windowed throughput for latest(),
        # but this ensures timeseries CSV has non-zero throughput values.
        prev_key = f"{job_id}:{replica_id or ''}"
        prev = _ingest_prev_snaps.get(prev_key)
        if prev is not None:
            dt = snap.timestamp - prev.timestamp
            if dt > 0.1:
                if snap.live_gen_tokens_total > 0:
                    snap.avg_generation_throughput_toks_per_s = max(0, (
                        snap.live_gen_tokens_total - prev.live_gen_tokens_total
                    ) / dt)
                    snap.avg_prompt_throughput_toks_per_s = max(0, (
                        snap.live_prompt_tokens_total - prev.live_prompt_tokens_total
                    ) / dt)
                else:
                    snap.avg_generation_throughput_toks_per_s = max(0, (
                        snap.generation_tokens_total - prev.generation_tokens_total
                    ) / dt)
                    snap.avg_prompt_throughput_toks_per_s = max(0, (
                        snap.prompt_tokens_total - prev.prompt_tokens_total
                    ) / dt)
        _ingest_prev_snaps[prev_key] = snap

        # Merge GPU hardware utilization from sidecar payload
        gpu_sm = item.get("gpu_sm_util_pct")
        gpu_bw = item.get("gpu_mem_bw_util_pct")
        if gpu_sm is not None:
            snap.gpu_sm_util_pct = float(gpu_sm)
        if gpu_bw is not None:
            snap.gpu_mem_bw_util_pct = float(gpu_bw)

        # Merge per-GPU metrics (forwarded raw from sidecar's pynvml)
        per_gpu = item.get("per_gpu")
        if per_gpu:
            snap.per_gpu = per_gpu

        # Write into aggregated job-level ring buffer
        with mc._lock:
            jc = mc._jobs.get(job_id)
        if jc:
            with jc.lock:
                jc.buffer.append(snap)

        # Write into per-replica ring buffer
        if replica_id:
            rkey = f"{job_id}:{replica_id}"
            with mc._lock:
                rc = mc._replicas.get(rkey)
            if rc:
                with rc.lock:
                    rc.buffer.append(snap)

        batch_for_db.append(snap.to_dict())
        ingested += 1

    # Persist timeseries directly (sidecar already batched; skip 60s flush timer)
    if batch_for_db:
        try:
            db.append_timeseries(job_id, batch_for_db)
        except Exception as e:
            logger.warning("[Ingest] timeseries write failed for %s: %s", job_id, e)

    # --- Replica log forwarding ---
    log_lines = body.get("log_lines", [])
    if log_lines and replica_id:
        _write_replica_logs(job_id, replica_id, log_lines)

    return {"ok": True, "ingested": ingested}


@app.post("/job/{job_id}/metrics/summary")
async def ingest_replica_summary(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Called by each replica after all chunks are done with its build_metrics dict."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    replica_id = body.get("replica_id")
    metrics = body.get("metrics")
    if not replica_id or not metrics:
        raise HTTPException(status_code=400, detail="replica_id and metrics required")

    app.state.metrics_db.push_replica_summary(job_id, replica_id, metrics)
    logger.info("[Summary] Stored replica summary for %s / %s", job_id, replica_id)
    return {"ok": True}


@app.post("/job/{job_id}/phase")
async def update_job_phase(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Called by in-cluster runner to report lifecycle phases (loading_model, model_ready, generating)."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    phase = body.get("phase")
    replica_id = body.get("replica_id")

    if phase == "replica_complete" and replica_id:
        # Replica finished its chunks; clean up metrics + schedule teardown
        cm = app.state.cluster_manager
        cm.set_replica_state(job_id, replica_id, phase="completed")
        app.state.metrics_collector.exclude_replica(job_id, replica_id)
        try:
            app.state.quota_tracker.release_cluster(replica_id)
        except Exception:
            pass
        import threading
        threading.Thread(target=sky_down_with_retry, args=(replica_id,), daemon=True).start()
        logger.info("[Phase] Replica %s completed, metrics stopped, teardown scheduled", replica_id)
        return {"ok": True}

    if phase:
        get_job_tracker().update_status(job_id, phase)
        if phase == "generating":
            app.state.metrics_collector.set_baseline(job_id)
        if phase == "model_ready" and replica_id:
            _notify_koi_replica_ready(job_id, replica_id)
    return {"ok": True}


def _notify_koi_replica_ready(job_id: str, replica_id: str):
    """Fire /job/started webhook to Koi when vLLM is ready to serve."""
    from orca_server.config import KOI_SERVICE_URL, INSTANCE_TO_GPU
    if not KOI_SERVICE_URL:
        return
    cm = app.state.cluster_manager
    state = cm.get_replica_states(job_id).get(replica_id, {})
    koi_info = state.get("koi_webhook_info")
    if not koi_info:
        return
    # Adjust SLO for provisioning time already elapsed
    import time as _time
    original_slo = koi_info.get("slo_deadline_hours", 8.0)
    deploy_ts = koi_info.get("deploy_timestamp")
    if deploy_ts:
        elapsed_hours = (_time.time() - deploy_ts) / 3600
        adjusted_slo = max(0.1, original_slo - elapsed_hours)
    else:
        adjusted_slo = original_slo

    try:
        import requests as _req
        _req.post(f"{KOI_SERVICE_URL}/job/started", json={
            "job_id": replica_id,
            "group_id": koi_info.get("group_id"),
            "decision_id": koi_info.get("decision_id"),
            "gpu_type": INSTANCE_TO_GPU.get(state.get("instance_type", ""), "unknown"),
            "instance_type": state.get("instance_type", "unknown"),
            "tp": state.get("tp", 1),
            "pp": state.get("pp", 1),
            "dp": 1,
            "slo_deadline_hours": adjusted_slo,
            "total_tokens": koi_info.get("total_tokens", 0),
            "predicted_tps": 0.0,
            "is_fallback": state.get("config_index", 0) > 0,
        }, timeout=5)
        logger.info("[Koi] Notified model_ready: %s (%s), SLO adjusted %.2fh→%.2fh (%.0fmin provisioning)",
                    replica_id, state.get("instance_type"), original_slo, adjusted_slo,
                    (original_slo - adjusted_slo) * 60)
    except Exception as e:
        logger.warning("[Koi] Failed to notify model_ready: %s", e)


@app.post("/job/{job_id}/chunks/pull")
async def pull_chunk(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Replica calls to get next chunk. Returns 204 when queue is empty."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    replica_id = body.get("replica_id", "unknown")

    cm = get_chunk_manager()
    chunk_info = cm.pull_chunk(job_id, replica_id)
    if chunk_info is None:
        return Response(status_code=204)
    return chunk_info


@app.post("/job/{job_id}/chunks/complete")
async def complete_chunk(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Replica reports chunk done. Triggers assembly when all chunks are complete."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    chunk_id = body.get("chunk_id")
    replica_id = body.get("replica_id", "unknown")

    cm = get_chunk_manager()
    progress = cm.complete_chunk(job_id, chunk_id, replica_id)

    # Update job tracker progress
    if progress["total"] > 0:
        frac = (progress["completed"] + progress["failed"]) / progress["total"]
        get_job_tracker().update_progress(job_id, frac)

    if progress["all_done"]:
        asyncio.create_task(_assemble_output(job_id))

    return progress


@app.post("/job/{job_id}/chunks/renew")
async def renew_chunk_lease(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Replica calls every CHUNK_RENEW_INTERVAL to extend its lease on an inflight chunk."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    chunk_id = body.get("chunk_id")
    replica_id = body.get("replica_id", "unknown")

    cm = get_chunk_manager()
    result = cm.renew_lease(job_id, chunk_id, replica_id)
    return result


@app.get("/job/{job_id}/chunks/progress")
async def chunk_progress(
    job_id: str,
    authorization: Optional[str] = Header(None),
):
    """Chunk-level progress for CLI."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    cm = get_chunk_manager()
    progress = cm.get_progress(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail=f"No chunk queue for job {job_id}")
    return progress


async def _assemble_output(job_id: str):
    """Combine per-chunk outputs into final ordered output.jsonl."""
    job_logger = logging.getLogger(f"orca.job.{job_id}")
    job_logger.info(f"[Assembly] Starting output assembly for {job_id}")

    cm = get_chunk_manager()

    # Assembly-once guard: prevent two concurrent triggers (reclaim loop + complete_chunk)
    # from both running assembly. SETNX returns True only for the winner; 5-min TTL as safety net.
    lock_key = f"chunk:job:{job_id}:assembling"
    if not cm._r.set(lock_key, "1", nx=True, ex=300):
        job_logger.info(f"[Assembly] Job {job_id} assembly already in progress, skipping")
        return

    meta = cm._r.hgetall(f"chunk:job:{job_id}:meta")
    if not meta:
        job_logger.info(f"[Assembly] Job {job_id} metadata gone (already assembled?), skipping")
        return
    s3_output_base = meta.get("s3_output_base", "")
    ordered_ids = cm.get_output_order(job_id)

    failed_ids = cm.get_failed_chunk_ids(job_id)
    if failed_ids:
        job_logger.warning(
            f"[Assembly] {len(failed_ids)} chunk(s) permanently failed and will be skipped: "
            f"{sorted(failed_ids)}"
        )

    combined_path = f"/tmp/assembly_{job_id}.jsonl"
    assembly_failures = []
    expected_chunks = len(ordered_ids) - len(failed_ids)
    try:
        with open(combined_path, "w") as combined:
            for cid in ordered_ids:
                if cid in failed_ids:
                    continue
                chunk_info = cm.get_chunk_info(job_id, cid)
                s3_out = chunk_info.get("s3_output_path", "")
                local_tmp = f"/tmp/assemble_{job_id}_{cid}.jsonl"
                try:
                    await storage_backend.download_file(s3_out, local_tmp, user="system")
                except Exception as dl_err:
                    job_logger.error(f"[Assembly] Failed to download {s3_out}: {dl_err}")
                    assembly_failures.append(cid)
                    continue
                with open(local_tmp) as cf:
                    for line in cf:
                        combined.write(line)
                os.unlink(local_tmp)

        # Upload combined output
        final_s3 = f"{s3_output_base}/output.jsonl"
        await storage_backend.upload_file(combined_path, final_s3, user="system")
        os.unlink(combined_path)
        job_logger.info(f"[Assembly] Uploaded combined output to {final_s3}")

        # Write all outputs into the original job directory (where job.log lives),
        # then rename with success-/partial- prefix — matching single-cluster behavior.
        from orca_server.job_manager import download_output_from_s3, prefix_job_dirname, close_job_logger
        tracker = get_job_tracker()
        rec = tracker.get(job_id)
        job_dirname = getattr(rec, '_job_dirname', job_id) if rec else job_id
        base_dir = Path(f"outputs/{job_dirname}")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download assembled output.jsonl into the base directory
        download_output_from_s3(final_s3, job_dirname, logger=job_logger)

        # Aggregate per-replica summaries into job-level metrics
        from orca_server.metrics_db import get_metrics_db
        db = get_metrics_db()
        local_metrics_path = None
        try:
            agg = db.aggregate_replica_summaries(job_id)
            if agg:
                # Write aggregated metrics.csv to temp, push into DB
                import csv as _csv
                metrics_csv_path = f"/tmp/assembly_metrics_{job_id}.csv"
                with open(metrics_csv_path, "w", newline="") as mf:
                    writer = _csv.writer(mf)
                    writer.writerow(["metric", "value"])
                    for k, v in agg.items():
                        if isinstance(v, float):
                            v = f"{v:.4f}"
                        writer.writerow([k, v])
                # Determine context for push_run
                actual_region = ""
                actual_market = "spot"
                solver = "chunked"
                if rec:
                    actual_region = getattr(rec.state, "actual_region", "") or ""
                    actual_market = getattr(rec.state, "actual_market", "spot") or "spot"
                db.push_run(
                    job_id, metrics_csv_path,
                    actual_region=actual_region,
                    actual_market=actual_market,
                    solver=solver,
                    job_dirname=job_dirname,
                )
                os.unlink(metrics_csv_path)
                job_logger.info(f"[Assembly] Wrote aggregated metrics to DB for {job_id}")

                # Save metrics.csv to the base experiment directory
                local_metrics_path = base_dir / "metrics.csv"
                with open(local_metrics_path, "w", newline="") as mf:
                    writer = _csv.writer(mf)
                    writer.writerow(["metric", "value"])
                    for k, v in agg.items():
                        if isinstance(v, float):
                            v = f"{v:.4f}"
                        writer.writerow([k, v])
                job_logger.info(f"[Assembly] Saved metrics.csv to {local_metrics_path}")

                # Upload aggregated metrics.csv to S3
                metrics_s3 = f"{s3_output_base}/metrics.csv"
                await storage_backend.upload_file(str(local_metrics_path), metrics_s3, user="system")
                job_logger.info(f"[Assembly] Uploaded aggregated metrics.csv to {metrics_s3}")
            else:
                job_logger.info(f"[Assembly] No replica summaries found for {job_id}, skipping metrics aggregation")
        except Exception as me:
            job_logger.warning(f"[Assembly] Metrics aggregation failed for {job_id}: {me}")

        # Export timeseries to the base experiment directory
        try:
            import csv as _csv2
            ts_data = db.get_timeseries(job_id)
            if ts_data:
                ts_path = base_dir / "timeseries.csv"
                # Use all keys from first sample as columns
                all_keys = list(ts_data[0].keys())
                with open(ts_path, "w", newline="") as tf:
                    writer = _csv2.DictWriter(tf, fieldnames=all_keys, extrasaction="ignore")
                    writer.writeheader()
                    for row in ts_data:
                        writer.writerow(row)
                job_logger.info(f"[Assembly] Saved timeseries.csv ({len(ts_data)} samples) to {ts_path}")

                # Generate timeseries PDF
                try:
                    from orca_server.plot_timeseries import plot_timeseries as _plot_ts
                    pdf_path = base_dir / "timeseries.pdf"
                    _metrics_arg = str(local_metrics_path) if local_metrics_path and local_metrics_path.exists() else None
                    _plot_ts(str(ts_path), str(pdf_path), metrics_csv_path=_metrics_arg)
                    job_logger.info(f"[Assembly] Generated timeseries.pdf at {pdf_path}")
                except Exception as pe:
                    job_logger.warning(f"[Assembly] Timeseries plot failed: {pe}")
        except Exception as te:
            job_logger.warning(f"[Assembly] Timeseries export failed for {job_id}: {te}")

        if assembly_failures:
            downloaded = expected_chunks - len(assembly_failures)
            job_logger.error(
                f"[Assembly] {len(assembly_failures)}/{expected_chunks} chunks failed to download: "
                f"{assembly_failures}. Partial output uploaded ({downloaded} chunks)."
            )
            get_job_tracker().update_status(job_id, "failed")
        else:
            get_job_tracker().update_status(job_id, "succeeded")
        get_job_tracker().update_progress(job_id, 1.0)
        cm.cleanup_job(job_id)

        # Clean up per-replica state to prevent memory leaks
        for key in list(_ingest_prev_snaps):
            if key.startswith(f"{job_id}:"):
                del _ingest_prev_snaps[key]
        with _replica_log_locks_lock:
            for key in list(_replica_log_locks):
                if key.startswith(f"{job_id}:") or key.startswith(job_id):
                    del _replica_log_locks[key]

        if assembly_failures:
            job_logger.info(f"[Assembly] Job {job_id} completed with {len(assembly_failures)} missing chunks (partial output available)")
        else:
            job_logger.info(f"[Assembly] Job {job_id} completed successfully")

        # Notify Koi of job completion (if KOI_SERVICE_URL is set)
        # Send parent job_id — Koi looks up all chains in this group and
        # records ONE aggregate outcome, then unregisters all chains.
        from orca_server.config import KOI_SERVICE_URL
        if KOI_SERVICE_URL:
            try:
                import requests as _req
                final_status = "failed" if assembly_failures else "succeeded"
                koi_payload = {
                    "job_id": job_id,
                    "status": final_status,
                    "metrics": agg if agg else {},
                }
                _req.post(f"{KOI_SERVICE_URL}/job/complete", json=koi_payload, timeout=10)
                job_logger.info(f"[Assembly] Notified Koi: job {job_id} completed ({final_status})")
            except Exception as ke:
                job_logger.warning(f"[Assembly] Failed to notify Koi: {ke}")

        # Rename directory with success-/partial-/failed- prefix (after all writes are done)
        status_prefix = "failed" if assembly_failures else ("partial" if failed_ids else "success")
        prefixed_dirname = prefix_job_dirname(job_dirname, status_prefix)
        target_dir = Path(f"outputs/{prefixed_dirname}")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        job_logger.info(f"[Assembly] {status_prefix.upper()}: outputs/{prefixed_dirname}")
        close_job_logger(job_logger)
        base_dir.rename(target_dir)

    except Exception as e:
        job_logger.error(f"[Assembly] Failed for {job_id}: {e}")
        get_job_tracker().update_status(job_id, "failed")
        cm._r.delete(lock_key)  # release assembly lock so retry can happen immediately
        if os.path.exists(combined_path):
            os.unlink(combined_path)
        # Rename to failed- prefix so the directory is clearly marked
        try:
            rec = get_job_tracker().get(job_id)
            _dirname = getattr(rec, '_job_dirname', job_id) if rec else job_id
            _base = Path(f"outputs/{_dirname}")
            if _base.exists():
                _failed = Path(f"outputs/{prefix_job_dirname(_dirname, 'failed')}")
                _failed.parent.mkdir(parents=True, exist_ok=True)
                job_logger.info(f"[Assembly] FAILED: outputs/{prefix_job_dirname(_dirname, 'failed')}")
                close_job_logger(job_logger)
                _base.rename(_failed)
        except Exception:
            pass


@app.get("/job/{job_id}/throughput")
async def get_job_throughput(job_id: str, window: float = 60.0):
    """Sustained throughput for the controller: rolling window + epoch (since baseline)."""
    result = app.state.metrics_collector.get_sustained_throughput(job_id, window)
    if result is None:
        raise HTTPException(404, "No throughput data (insufficient samples)")
    result["job_id"] = job_id
    return result


@app.get("/job/{job_id}/replicas/summaries")
async def get_replica_summaries_endpoint(job_id: str):
    """Return per-replica build_metrics dicts for a completed chunked job."""
    summaries = app.state.metrics_db.get_replica_summaries(job_id)
    if not summaries:
        raise HTTPException(404, f"No replica summaries for {job_id}")
    return {"job_id": job_id, "count": len(summaries), "summaries": summaries}


# ---------------------------------------------------------------------------
# Scale / Kill / Swap primitives
# ---------------------------------------------------------------------------

TERMINAL_PHASES = {"dead", "completed", "failed", "swapped_out", "killed"}


def _infer_job_config(job_id: str) -> dict:
    """Infer GPU/TP/PP/instance_type from an existing job's active replicas.

    Returns dict with keys: instance_type, gpu_type, tp_size, pp_size, model_name.
    Raises HTTPException if no config can be inferred.
    """
    from orca_server.config import INSTANCE_TO_GPU

    cm = app.state.cluster_manager
    jt = get_job_tracker()
    rec = jt.get(job_id)

    # Try active replicas first
    states = cm.get_replica_states(job_id)
    for rid, info in states.items():
        if info.get("phase") in TERMINAL_PHASES:
            continue
        inst = info.get("instance_type")
        if not inst:
            cluster_info = cm.active_clusters.get(rid, {})
            inst = cluster_info.get("instance_type")
        if inst:
            gpu = INSTANCE_TO_GPU.get(inst, inst)
            return {
                "instance_type": inst,
                "gpu_type": gpu,
                "tp_size": info.get("tp_size", 1),
                "pp_size": info.get("pp_size", 1),
                "model_name": "",
            }

    # Fallback to JobRecord spec
    if rec and hasattr(rec, "state") and hasattr(rec.state, "spec"):
        spec = rec.state.spec
        state = rec.state
        inst_types = getattr(state, "instance_types", None)
        inst = inst_types if isinstance(inst_types, str) else (inst_types[0] if inst_types else None)
        if inst:
            gpu = INSTANCE_TO_GPU.get(inst, getattr(state, "gpu_base", inst))
            return {
                "instance_type": inst,
                "gpu_type": gpu,
                "tp_size": getattr(state, "tp", 1) or 1,
                "pp_size": getattr(state, "pp", 1) or 1,
                "model_name": getattr(spec, "model_name", ""),
            }

    raise HTTPException(400, "Cannot infer GPU config from job — specify --gpu explicitly")


def _do_scale(job_id: str, count: int, gpu_type: str, tp_size: int, pp_size: int,
              on_demand: bool = False, instance_type: str = None) -> dict:
    """Launch N new replicas for an existing chunked job. Returns replica names + version.

    This is a fire-and-forget operation — replicas launch in background threads
    and join the existing Redis chunk queue.
    """
    cm = app.state.cluster_manager
    jt = get_job_tracker()
    rec = jt.get(job_id)
    chunk_mgr = get_chunk_manager()

    # Resolve instance type if not provided
    if not instance_type:
        instance_type, _ = resolve_gpu_type_to_instance(gpu_type, tp_size)

    # Generate versioned replica names
    version = cm.next_swap_version(job_id)
    new_replicas = [f"{job_id}-v{version}-r{i}" for i in range(count)]

    # Pre-register
    for rid in new_replicas:
        cm.set_replica_state(job_id, rid, phase="launching")
        cm.register_for_job(job_id, rid)

    # Build config
    new_config = MagicOutput(
        decision_id=job_id,
        engine="vllm",
        instance_type=instance_type,
        tp_size=tp_size,
        pp_size=pp_size,
        replicas=count,
        num_instances=pp_size,
    )

    # Build BatchedRequest from job metadata
    meta = chunk_mgr._r.hgetall(f"chunk:job:{job_id}:meta")
    s3_base = meta.get("s3_output_base", f"s3://{S3_UPLOAD_BUCKET}/scale")
    spec = rec.state.spec if rec and hasattr(rec, "state") and hasattr(rec.state, "spec") else None
    original_request = BatchedRequest(
        user_id="scale",
        model_name=(spec.model_name if spec else meta.get("model_name", "unknown")),
        input_file=f"{s3_base}/scale_input.jsonl",
        output_file="output.jsonl",
        description="scale",
        task_type="batch",
        task_priority="normal",
        engine="vllm",
        slo_mode="cost_first",
        placement="user_specified",
        num_lines=int(meta.get("total_chunks", 1)) * 100,
        avg_input_tokens=(spec.avg_input_tokens if spec else 2000),
        avg_output_tokens=(spec.avg_output_tokens if spec else 1024),
        prefer_spot=not on_demand,
    )

    job_dirname = getattr(rec, "_job_dirname", None) or f"scale-{job_id}"

    # Reconstruct koi_webhook_info from existing replicas so Koi gets notified
    existing_states = cm.get_replica_states(job_id)
    koi_info_source = next(
        (s.get("koi_webhook_info") for s in existing_states.values() if s.get("koi_webhook_info")),
        None,
    )
    scale_koi_info = {
        "decision_id": koi_info_source.get("decision_id") if koi_info_source else None,
        "group_id": job_id,
        "slo_deadline_hours": koi_info_source.get("slo_deadline_hours", 8.0) if koi_info_source else 8.0,
        "total_tokens": koi_info_source.get("total_tokens", 0) if koi_info_source else 0,
        "deploy_timestamp": time.time(),
    } if koi_info_source else None

    def _launch_thread(replica_id):
        import asyncio as _aio
        replica_config = new_config.model_copy(update={
            "decision_id": replica_id,
            "replicas": 1,
            "num_instances": pp_size,
        })
        try:
            loop = _aio.new_event_loop()
            loop.run_until_complete(_launch_chunked_replica(
                original_request, replica_config, replica_id,
                parent_job_id=job_id,
                job_dirname=job_dirname,
                persist=False,
                koi_webhook_info=scale_koi_info,
            ))
            loop.close()
        except Exception as e:
            logger.error("[Scale] Failed to launch replica %s: %s", replica_id, e)
            cm.set_replica_state(job_id, replica_id, phase="failed")

    for rid in new_replicas:
        t = threading.Thread(target=_launch_thread, args=(rid,), daemon=False,
                             name=f"orca-scale-{rid[:16]}")
        t.start()

    logger.info("[Scale] Launched %d new replicas for %s: %s", count, job_id, new_replicas)
    return {"new_replicas": new_replicas, "version": version}


def _do_kill(job_id: str, replica_ids: list[str], phase: str = "killed") -> dict:
    """Kill specific replicas: reclaim chunks, exclude metrics, tear down clusters.

    Returns {"killed": [...], "skipped": [...], "reclaimed": N}.
    """
    cm = app.state.cluster_manager
    mc = app.state.metrics_collector
    chunk_mgr = get_chunk_manager()

    # Filter out replicas already in terminal state
    states = cm.get_replica_states(job_id)
    to_kill = []
    skipped = []
    for rid in replica_ids:
        info = states.get(rid, {})
        if info.get("phase") in TERMINAL_PHASES:
            skipped.append(rid)
        else:
            to_kill.append(rid)

    if not to_kill:
        return {"killed": [], "skipped": skipped, "reclaimed": 0}

    # Force-reclaim inflight chunks back to pending queue
    result = chunk_mgr.force_reclaim(job_id, to_kill)
    logger.info("[Kill] Reclaimed chunks from %s: %s", to_kill, result)

    # Exclude from metrics, update phase, release quota, tear down
    for rid in to_kill:
        mc.exclude_replica(job_id, rid)
        cm.set_replica_state(job_id, rid, phase=phase)
        try:
            app.state.quota_tracker.release_cluster(rid)
        except Exception:
            pass
        threading.Thread(
            target=sky_down_with_retry, args=(rid,),
            daemon=True, name=f"kill-down-{rid[:12]}",
        ).start()

    logger.info("[Kill] Killed replicas for %s: %s (phase=%s)", job_id, to_kill, phase)
    return {"killed": to_kill, "skipped": skipped, "reclaimed": result.get("reclaimed", 0)}


# ---------------------------------------------------------------------------
# Scale endpoint
# ---------------------------------------------------------------------------

@app.post("/job/{job_id}/scale")
async def scale_replicas(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Add replicas to a running chunked job."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    jt = get_job_tracker()
    rec = jt.get(job_id)
    chunk_mgr = get_chunk_manager()
    progress = chunk_mgr.get_progress(job_id)
    if rec is None and progress is None:
        raise HTTPException(404, "Job not found")
    if progress is None:
        raise HTTPException(400, "Not a chunked job — scale requires chunked deployment")
    if rec and rec.status in ("succeeded", "failed", "cancelled"):
        raise HTTPException(409, f"Job status '{rec.status}' cannot be scaled")

    body = await request.json()
    count = body.get("count")
    if not count or count < 1:
        raise HTTPException(400, "count is required and must be >= 1")

    gpu_type = body.get("gpu_type")
    tp_size = body.get("tp_size")
    pp_size = body.get("pp_size")
    on_demand = body.get("on_demand", False)
    force = body.get("force", False)

    # Infer config from existing job if not specified
    if not gpu_type:
        inferred = _infer_job_config(job_id)
        gpu_type = inferred["gpu_type"]
        tp_size = tp_size or inferred["tp_size"]
        pp_size = pp_size or inferred["pp_size"]
    tp_size = tp_size or 1
    pp_size = pp_size or 1

    # Resolve + validate
    try:
        instance_type, gpu_count = resolve_gpu_type_to_instance(gpu_type, tp_size)
    except Exception as e:
        raise HTTPException(400, f"Cannot resolve GPU config: {e}")

    if not force:
        model_name = ""
        spec = None
        if rec and hasattr(rec, "state") and hasattr(rec.state, "spec"):
            spec = rec.state.spec
            model_name = getattr(spec, "model_name", "")
        try:
            feasibility = check_user_specified_feasibility(
                model_name=model_name,
                instance_type=instance_type,
                gpu_count=tp_size * pp_size,
                tp=tp_size,
                pp=pp_size,
                avg_input_tokens=getattr(spec, "avg_input_tokens", 2000) if spec else 2000,
                avg_output_tokens=getattr(spec, "avg_output_tokens", 1024) if spec else 1024,
            )
            if not feasibility.get("feasible", True):
                return {"status": "confirm",
                        "message": feasibility.get("reason", "Config may not be feasible"),
                        "detail": feasibility}
        except Exception as e:
            logger.warning("[Scale] Feasibility check failed, proceeding anyway: %s", e)

    result = _do_scale(job_id, count, gpu_type, tp_size, pp_size,
                       on_demand=on_demand, instance_type=instance_type)
    return {"status": "scaling", **result}


# ---------------------------------------------------------------------------
# Kill endpoint
# ---------------------------------------------------------------------------

@app.post("/job/{job_id}/kill")
async def kill_replicas(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Kill specific replicas of a job — reclaim chunks, exclude metrics, tear down."""
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    jt = get_job_tracker()
    rec = jt.get(job_id)
    cm = app.state.cluster_manager
    states = cm.get_replica_states(job_id)
    if rec is None and not states:
        raise HTTPException(404, "Job not found")

    body = await request.json()
    replica_ids = body.get("replica_ids", [])
    if not replica_ids:
        raise HTTPException(400, "replica_ids is required")

    result = _do_kill(job_id, replica_ids)
    return {"status": "killing", **result}


# ---------------------------------------------------------------------------
# Swap endpoint (refactored: composes scale + monitor → kill)
# ---------------------------------------------------------------------------

@app.post("/job/{job_id}/swap")
async def swap_replicas(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Hot-swap replicas to a new GPU/TP/PP config mid-job.

    Composes scale (launch new replicas) + monitor (wait for readiness) + kill
    (tear down old replicas). Old replicas keep running until new ones are ready.
    """
    if ORCA_API_KEY and authorization != f"Bearer {ORCA_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    jt = get_job_tracker()
    rec = jt.get(job_id)
    chunk_mgr = get_chunk_manager()
    progress = chunk_mgr.get_progress(job_id)
    if rec is None and progress is None:
        raise HTTPException(404, "Job not found")
    if progress is None:
        raise HTTPException(400, "Job is not a chunked job — swap requires a chunked deployment")
    if rec and rec.status in ("succeeded", "failed", "cancelled"):
        raise HTTPException(409, f"Job status '{rec.status}' is not swappable")

    cm = app.state.cluster_manager
    if cm._swap_in_progress.get(job_id):
        raise HTTPException(409, "Swap already in progress for this job")

    body = await request.json()
    gpu_type = body.get("gpu_type")
    tp_size = body.get("tp_size", 1)
    pp_size = body.get("pp_size", 1)
    num_replicas = body.get("num_replicas")
    ready_threshold = body.get("ready_threshold", 1)
    on_demand = body.get("on_demand", False)
    force = body.get("force", False)

    if not gpu_type:
        raise HTTPException(400, "gpu_type is required")

    # Validate feasibility
    try:
        instance_type, gpu_count = resolve_gpu_type_to_instance(gpu_type, tp_size)
    except Exception as e:
        raise HTTPException(400, f"Cannot resolve GPU config: {e}")

    if not force:
        feasibility = check_user_specified_feasibility(
            gpu_type=gpu_type, tp_size=tp_size, pp_size=pp_size,
            model_name=getattr(rec.state, "spec", None) and rec.state.spec.model_name or "",
        )
        if not feasibility.get("feasible", True):
            return {"status": "confirm", "message": feasibility.get("reason", "Config may not be feasible"),
                    "detail": feasibility}

    # Snapshot old replicas
    old_states = cm.get_replica_states(job_id)
    old_replicas = [rid for rid, info in old_states.items()
                    if info.get("phase") not in TERMINAL_PHASES]

    if num_replicas is None:
        num_replicas = len(old_replicas) or 1
    ready_threshold = min(ready_threshold, num_replicas)

    # Mark swap in progress
    with cm.lock:
        cm._swap_in_progress[job_id] = True

    # Scale up new replicas
    scale_result = _do_scale(job_id, num_replicas, gpu_type, tp_size, pp_size,
                             on_demand=on_demand, instance_type=instance_type)

    # Start monitor that kills old replicas when new ones are ready
    asyncio.create_task(_swap_monitor(job_id, old_replicas,
                                      scale_result["new_replicas"], ready_threshold))

    return {
        "status": "swapping",
        "old_replicas": old_replicas,
        "new_replicas": scale_result["new_replicas"],
        "ready_threshold": ready_threshold,
        "version": scale_result["version"],
    }


async def _swap_monitor(job_id: str, old_replicas: list[str],
                         new_replicas: list[str], ready_threshold: int):
    """Wait for K new replicas to send first ingest POST, then kill old ones."""
    mc = app.state.metrics_collector
    cm = app.state.cluster_manager
    deadline = time.time() + 1800  # 30 min timeout
    ready_set: set[str] = set()

    logger.info(
        "[Swap] Monitor started for %s: waiting for %d/%d new replicas",
        job_id, ready_threshold, len(new_replicas),
    )

    while len(ready_set) < ready_threshold and time.time() < deadline:
        await asyncio.sleep(5)

        jt = get_job_tracker()
        rec = jt.get(job_id)
        if rec and rec.status in ("succeeded", "failed", "cancelled"):
            logger.info("[Swap] Job %s ended during swap, aborting monitor", job_id)
            break

        for rid in new_replicas:
            if rid in ready_set:
                continue
            key = f"{job_id}:{rid}"
            with mc._lock:
                rc = mc._replicas.get(key)
            if rc:
                with rc.lock:
                    if rc.buffer:
                        ready_set.add(rid)
                        logger.info("[Swap] New replica %s is active (%d/%d)",
                                    rid, len(ready_set), ready_threshold)

    if len(ready_set) >= ready_threshold:
        logger.info("[Swap] Threshold met for %s, killing old replicas: %s", job_id, old_replicas)
        _do_kill(job_id, old_replicas, phase="swapped_out")
    else:
        logger.warning(
            "[Swap] Timeout for %s: only %d/%d new replicas ready. Old replicas kept alive.",
            job_id, len(ready_set), ready_threshold,
        )

    with cm.lock:
        cm._swap_in_progress.pop(job_id, None)


@app.get("/analytics/runs")
async def list_analytics_runs(
    model: Optional[str] = None,
    gpu: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List completed runs with optional model/gpu filters."""
    runs = app.state.metrics_db.list_runs(
        model=model, gpu=gpu, limit=min(limit, 200), offset=offset
    )
    return {"status": "success", "count": len(runs), "runs": runs}


@app.get("/analytics/runs/{run_id}")
async def get_analytics_run(run_id: int):
    """Get a single completed run by DB ID."""
    run = app.state.metrics_db.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return {"status": "success", "run": run}


@app.get("/analytics/runs/{run_id}/timeseries")
async def get_run_timeseries(
    run_id: int,
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    """Full timeseries for a completed run. Supports optional unix timestamp range."""
    run = app.state.metrics_db.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    ts = app.state.metrics_db.get_timeseries(run["job_id"], start=start, end=end)
    return {"status": "success", "job_id": run["job_id"], "count": len(ts), "timeseries": ts}


# A: Stats on input file computed instead of from request
# A: When to respond to user request (prob return early + observability)
@app.post("/submit/batch")
async def submit_batch(request: BatchedRequest):
    """
    Submit a batched inference job request.

    The placement solver can be:
    - "roofline": Deterministic roofline-based solver (default)
    - "user_specified": User provides GPU/TP/PP directly
    """
    # For chunked jobs the CLI already parsed stats and uploaded chunks — skip file download
    if request.chunks and request.num_lines is not None and request.avg_input_tokens is not None:
        num_lines = request.num_lines
        avg_input_tokens = request.avg_input_tokens
        max_input_tokens = request.max_input_tokens
    else:
        # Download S3 input if needed, then parse local file for stats
        local_input, tmp_cleanup = await _resolve_input_file(request.input_file)
        try:
            num_lines, avg_input_tokens, max_input_tokens = parse_input_file_stats(
                local_input, model_name=request.model_name
            )
        finally:
            if tmp_cleanup:
                os.unlink(tmp_cleanup)

        # Update request with parsed values (these override any user-provided values)
        request = request.model_copy(
            update={
                "num_lines": num_lines,
                "avg_input_tokens": avg_input_tokens,
                "max_input_tokens": max_input_tokens,
            }
        )

    # Collect log messages before the job logger is created
    early_messages = []

    msg = f"[InputStats] num_lines={num_lines}, avg_input={avg_input_tokens}, max_input={max_input_tokens}"
    logger.info(msg)
    early_messages.append(("INFO", msg))

    # Get multiple fallback solutions for retry logic
    # Request field takes priority, then fall back to env var
    use_solver = request.placement_solver or PLACEMENT_SOLVER
    quota_warning = None  # Set by user_specified path if no viable regions

    if use_solver == "user_specified":
        # ---------- User-specified path ----------
        gpu_type = request.gpu_type
        tp = request.tp_size or 1
        pp = request.pp_size or 1

        # Resolve GPU type to the smallest AWS instance with enough GPUs for TP
        try:
            instance_type, gpu_count = resolve_gpu_type_to_instance(gpu_type, tp)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "invalid_placement",
                "message": str(e),
            }

        # Run feasibility check via LLM_placement_solver
        result = check_user_specified_feasibility(
            model_name=request.model_name,
            instance_type=instance_type,
            gpu_count=gpu_count,
            tp=tp,
            pp=pp,
            avg_input_tokens=request.avg_input_tokens,
            avg_output_tokens=request.avg_output_tokens,
            max_input_tokens=request.max_input_tokens or 0,
            max_output_tokens=request.max_output_tokens or 0,
        )

        # Lightweight quota check (cached, non-blocking)
        quota_warning = None
        partitions_per_inst = gpu_count // tp
        num_instances = math.ceil(pp / partitions_per_inst)
        instance_family = get_instance_family(instance_type)
        quotas = get_cached_quotas(instance_family)
        viable_regions = get_ordered_regions(
            instance_type=instance_type,
            num_nodes=num_instances,
            quotas=quotas,
            prefer_spot=getattr(request, "prefer_spot", True),
        )
        if not viable_regions:
            quota_warning = f"No quota for {instance_type} in any region. Launch will likely fail."

        if not result["feasible"] and not request.force:
            return {
                "status": "confirm",
                "message": result["reason"],
                "feasibility": result,
                "config": {
                    "gpu_type": gpu_type,
                    "tp": tp,
                    "pp": pp,
                    "instance_type": instance_type,
                },
                "quota_warning": quota_warning,
                "hint": "Re-submit with force=true to launch anyway",
            }

        # Build MagicOutput directly (partitions_per_inst, num_instances already computed above)
        configs = [
            MagicOutput(
                decision_id=_make_job_id(request.model_name),
                engine=request.engine or "vllm",
                instance_type=instance_type,
                tp_size=tp,
                pp_size=pp,
                replicas=1,
                max_model_len=result["max_model_len"],
                num_instances=num_instances,
            )
        ]

        # Append Koi alternatives as fallback configs
        if getattr(request, "koi_alternatives", None):
            for alt in request.koi_alternatives:
                alt_gpu = alt.get("gpu_type")
                alt_tp = alt.get("tp", 1)
                alt_pp = alt.get("pp", 1)
                if not alt_gpu:
                    continue
                try:
                    alt_inst, alt_gpu_count = resolve_gpu_type_to_instance(alt_gpu, alt_tp)
                except ValueError:
                    continue
                alt_num_inst = max(1, (alt_tp * alt_pp) // alt_gpu_count)
                configs.append(
                    MagicOutput(
                        decision_id=_make_job_id(request.model_name),
                        engine=request.engine or "vllm",
                        instance_type=alt_inst,
                        tp_size=alt_tp,
                        pp_size=alt_pp,
                        replicas=1,
                        max_model_len=result["max_model_len"],
                        num_instances=alt_num_inst,
                    )
                )
            if len(configs) > 1:
                logger.info(f"[Placement] {len(configs)} configs total (primary + {len(configs)-1} Koi alternatives)")

        sol = result.get("solution") or {}
        msg = (
            f"[Placement] user_specified: {instance_type} TP={tp} PP={pp} "
            f"max_model_len={result['max_model_len']} "
            f"throughput={sol.get('throughput_tokens_per_sec', 'N/A')} tok/s "
            f"cost=${sol.get('cost_per_hour', 'N/A')}/hr"
        )
        logger.info(msg)
        early_messages.append(("INFO", msg))

    elif use_solver == "roofline":
        solver = RooflineAWSAllocation(
            perfdb_dir="./data/perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            priority=PLACEMENT_PRIORITY,
        )
        configs = solver.process_batch_multi(request, top_k=5)
        if not configs:
            configs = [solver._fallback_config(request)]
        # Capture solver log for the job log file (always, regardless of success)
        if getattr(solver, "last_solve_log", ""):
            for line in solver.last_solve_log.splitlines():
                early_messages.append(("INFO", f"[Solver] {line}"))
    else:
        return {
            "status": "error",
            "error_type": "invalid_solver",
            "message": f"Unknown solver: '{use_solver}'. Use 'roofline' or 'user_specified'.",
        }

    msg = f"[Placement] Using solver: {use_solver}"
    logger.info(msg)
    early_messages.append(("INFO", msg))
    msg = f"[Placement] Primary: {configs[0].instance_type} TP={configs[0].tp_size} PP={configs[0].pp_size}"
    logger.info(msg)
    early_messages.append(("INFO", msg))
    if len(configs) > 1:
        msg = f"[Placement] Fallbacks: {len(configs) - 1}"
        logger.info(msg)
        early_messages.append(("INFO", msg))
    if configs[0].estimated_runtime_hours is not None:
        eta = configs[0].estimated_runtime_hours
        slo_ok = configs[0].meets_slo
        slo_str = "meets SLO" if slo_ok else "MISSES SLO"
        msg = f"[SLO] ETA: {eta:.2f}h / deadline: {request.slo_deadline_hours:.1f}h — {slo_str}"
        logger.info(msg)
        early_messages.append(("WARNING" if not slo_ok else "INFO", msg))

    # Pre-launch check: ensure max_model_len can accommodate the longest prompt
    max_output = request.max_output_tokens or request.avg_output_tokens
    if configs[0].max_model_len is not None and max_input_tokens is not None:
        required_context = max_input_tokens + max_output
        if required_context > configs[0].max_model_len:
            return {
                "status": "error",
                "error_type": "context_length_exceeded",
                "message": (
                    f"Longest prompt ({max_input_tokens} tokens) + max_output ({max_output}) = "
                    f"{required_context} exceeds max_model_len ({configs[0].max_model_len}) "
                    f"for {configs[0].instance_type} TP={configs[0].tp_size} PP={configs[0].pp_size}. "
                    f"Some requests would be skipped at runtime."
                ),
                "detail": {
                    "max_input_tokens": max_input_tokens,
                    "max_output_tokens": max_output,
                    "required_context": required_context,
                    "max_model_len": configs[0].max_model_len,
                    "instance_type": configs[0].instance_type,
                    "tp_size": configs[0].tp_size,
                    "pp_size": configs[0].pp_size,
                },
            }

    # ── Chunked path: CLI already split + uploaded chunks to S3 ──
    if request.chunks and not getattr(app.state, "redis_available", False):
        raise HTTPException(503, "Redis unavailable — chunked jobs require Redis. "
                            "Start Redis (docker run -d -p 6379:6379 redis) and restart the server.")

    if request.chunks:
        effective_replicas = request.replicas or len(request.chunks)
        primary = configs[0]
        job_id = primary.decision_id

        # S3 output base (same bucket as first chunk)
        first_chunk_s3 = request.chunks[0]["s3_input_path"]
        s3_base = "/".join(first_chunk_s3.split("/")[:3])
        from orca_server.job_manager import generate_job_dirname
        job_dirname = generate_job_dirname(request, use_solver, primary.tp_size, primary.pp_size, primary.instance_type)
        s3_output_base = f"{s3_base}/{job_dirname}"

        cm = get_chunk_manager()
        cm.create_job_queue(job_id, request.chunks, request.model_name, s3_output_base)

        # Mark job as chunked so watchdog monitors replica heartbeats
        get_job_tracker().set_chunked_info(job_id, len(request.chunks), effective_replicas)

        msg = f"[Chunked] {len(request.chunks)} chunks, {effective_replicas} replicas"
        logger.info(msg)
        early_messages.append(("INFO", msg))

        success = await launch_chunked_replicas(
            request, configs, effective_replicas,
            solver=use_solver, early_messages=early_messages,
            quota_tracker=get_quota_tracker(),
            persist=getattr(request, "persist", False),
        )

        if success:
            return {
                "status": "launched",
                "job_id": job_id,
                "config": {
                    "instance_type": primary.instance_type,
                    "tp_size": primary.tp_size,
                    "pp_size": primary.pp_size,
                },
                "chunks": len(request.chunks),
                "replicas": effective_replicas,
                "input_stats": {
                    "num_lines": num_lines,
                    "avg_input_tokens": avg_input_tokens,
                    "max_input_tokens": max_input_tokens,
                },
                "quota_warning": quota_warning,
                "message": f"Chunked job launched with {effective_replicas} replicas. Track: GET /job/{job_id}",
            }
        else:
            return {
                "status": "error",
                "job_id": job_id,
                "message": "Failed to launch chunked replicas",
            }

    # ── Single-cluster path (existing, unchanged) ──
    success, used_config = await sp_launch_vllm_batch_with_fallback(
        request, configs, solver=use_solver, early_messages=early_messages,
        quota_tracker=get_quota_tracker(),
        persist=getattr(request, "persist", False),
    )

    if success:
        # Notify Koi that the job launched (starts monitoring)
        from orca_server.config import KOI_SERVICE_URL, INSTANCE_TO_GPU
        if KOI_SERVICE_URL:
            try:
                import requests as _req
                _req.post(f"{KOI_SERVICE_URL}/job/started", json={
                    "job_id": used_config.decision_id,
                    "decision_id": request.koi_decision_id,
                    "gpu_type": INSTANCE_TO_GPU.get(used_config.instance_type, "unknown"),
                    "instance_type": used_config.instance_type,
                    "tp": used_config.tp_size,
                    "pp": used_config.pp_size,
                    "dp": getattr(request, "replicas", 1) or 1,
                    "slo_deadline_hours": request.slo_deadline_hours or 8.0,
                    "total_tokens": (request.num_lines or 0) * ((request.avg_input_tokens or 0) + (request.avg_output_tokens or 0)),
                }, timeout=5)
                logger.info(f"[Launch] Notified Koi: job {used_config.decision_id} started")
            except Exception as ke:
                logger.warning(f"[Launch] Failed to notify Koi of job start: {ke}")

        resp = {
            "status": "launched",
            "job_id": used_config.decision_id,
            "config": {
                "instance_type": used_config.instance_type,
                "tp_size": used_config.tp_size,
                "pp_size": used_config.pp_size,
            },
            "input_stats": {
                "num_lines": num_lines,
                "avg_input_tokens": avg_input_tokens,
                "max_input_tokens": max_input_tokens,
            },
            "quota_warning": quota_warning,
            "message": f"Job submitted. Check progress at GET /job/{used_config.decision_id}",
        }
        if used_config.estimated_runtime_hours is not None:
            resp["estimated_runtime_hours"] = round(used_config.estimated_runtime_hours, 2)
            resp["meets_slo"] = used_config.meets_slo
            resp["slo_deadline_hours"] = request.slo_deadline_hours
        return resp
    else:
        return {
            "status": "error",
            "job_id": configs[0].decision_id,
            "message": "Failed to launch in all regions with all instance types",
        }


@app.post("/test/placement")
async def test_placement(request: BatchedRequest):
    """
    Run placement logic only (no cloud deployment).

    Accepts the same BatchedRequest as /submit/batch but only runs the solver
    and returns the placement decision(s) with performance/cost estimates.
    """
    # If the client already sent parsed stats, skip the S3 download entirely
    if request.num_lines is not None and request.avg_input_tokens is not None and request.max_input_tokens is not None:
        num_lines = request.num_lines
        avg_input_tokens = request.avg_input_tokens
        max_input_tokens = request.max_input_tokens
    else:
        local_input, tmp_cleanup = await _resolve_input_file(request.input_file)
        try:
            num_lines, avg_input_tokens, max_input_tokens = parse_input_file_stats(
                local_input, model_name=request.model_name
            )
        finally:
            if tmp_cleanup:
                os.unlink(tmp_cleanup)
        request = request.model_copy(
            update={
                "num_lines": num_lines,
                "avg_input_tokens": avg_input_tokens,
                "max_input_tokens": max_input_tokens,
            }
        )

    use_solver = request.placement_solver or PLACEMENT_SOLVER
    solver_log = ""

    if use_solver == "user_specified":
        gpu_type = request.gpu_type
        tp = request.tp_size or 1
        pp = request.pp_size or 1

        try:
            instance_type, gpu_count = resolve_gpu_type_to_instance(gpu_type, tp)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "invalid_placement",
                "message": str(e),
            }

        result = check_user_specified_feasibility(
            model_name=request.model_name,
            instance_type=instance_type,
            gpu_count=gpu_count,
            tp=tp,
            pp=pp,
            avg_input_tokens=request.avg_input_tokens,
            avg_output_tokens=request.avg_output_tokens,
            max_input_tokens=request.max_input_tokens or 0,
            max_output_tokens=request.max_output_tokens or 0,
        )

        # Lightweight quota check (cached, non-blocking)
        partitions_per_inst = gpu_count // tp
        num_instances = math.ceil(pp / partitions_per_inst)
        instance_family = get_instance_family(instance_type)
        quotas = get_cached_quotas(instance_family)
        viable_regions = get_ordered_regions(
            instance_type=instance_type,
            num_nodes=num_instances,
            quotas=quotas,
            prefer_spot=getattr(request, "prefer_spot", True),
        )
        quota_warning = None
        if not viable_regions:
            quota_warning = f"No quota for {instance_type} in any region. Launch will likely fail."

        sol = result.get("solution") or {}
        configs = [
            {
                "instance_type": instance_type,
                "gpu_type": gpu_type,
                "tp_size": tp,
                "pp_size": pp,
                "num_instances": num_instances,
                "max_model_len": result.get("max_model_len"),
                "throughput_tokens_per_sec": sol.get("throughput_tokens_per_sec"),
                "cost_per_hour": sol.get("cost_per_hour"),
                "cost_per_million_tokens": sol.get("cost_per_million_tokens"),
                "feasible": result["feasible"],
                "reason": result.get("reason"),
                "quota_warning": quota_warning,
            }
        ]

    elif use_solver == "roofline":
        solver = RooflineAWSAllocation(
            perfdb_dir="./data/perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            priority=PLACEMENT_PRIORITY,
        )
        magic_outputs = solver.process_batch_multi(request, top_k=5)
        solver_log = getattr(solver, "last_solve_log", "")
        if not magic_outputs:
            magic_outputs = [solver._fallback_config(request)]

        configs = []
        for mo in magic_outputs:
            cfg = {
                "instance_type": mo.instance_type,
                "gpu_type": INSTANCE_TO_GPU.get(mo.instance_type, "unknown"),
                "tp_size": mo.tp_size,
                "pp_size": mo.pp_size,
                "num_instances": mo.num_instances or mo.num_nodes,
                "max_model_len": mo.max_model_len,
                "replicas": mo.replicas,
            }
            if mo.throughput_tokens_per_sec is not None:
                cfg["throughput_tokens_per_sec"] = mo.throughput_tokens_per_sec
            if mo.cost_per_hour is not None:
                cfg["cost_per_hour"] = mo.cost_per_hour
            if mo.cost_per_million_tokens is not None:
                cfg["cost_per_million_tokens"] = mo.cost_per_million_tokens
            if mo.estimated_runtime_hours is not None:
                cfg["estimated_runtime_hours"] = round(mo.estimated_runtime_hours, 2)
            if mo.meets_slo is not None:
                cfg["meets_slo"] = mo.meets_slo
            if mo.is_fallback:
                cfg["is_fallback"] = True
            configs.append(cfg)
    elif use_solver == "llm":
        from placement.advisor.advisor import PlacementAdvisor
        from placement.roofline_magic import quota_to_gpu_pool
        from utils.utils import load_aws_quota_csv
        try:
            _quota_df = load_aws_quota_csv("./quota/aws_gpu_quota_by_region.csv")
            _gpu_pool = quota_to_gpu_pool(_quota_df) or None
        except Exception:
            _gpu_pool = None  # fall back to all instances if quota unavailable
        advisor = PlacementAdvisor(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        magic_outputs = await asyncio.to_thread(
            advisor.recommend,
            request.model_name,
            request.avg_input_tokens or avg_input_tokens,
            request.avg_output_tokens or 256,
            request.num_lines or num_lines,
            request.slo_deadline_hours or 4.0,
            _gpu_pool,
        )
        configs = []
        for mo in magic_outputs:
            cfg = {
                "instance_type": mo.instance_type,
                "gpu_type": INSTANCE_TO_GPU.get(mo.instance_type, "unknown"),
                "tp_size": mo.tp_size,
                "pp_size": mo.pp_size,
                "num_instances": mo.num_instances or mo.num_nodes,
                "max_model_len": mo.max_model_len,
                "replicas": mo.replicas,
            }
            if mo.throughput_tokens_per_sec is not None:
                cfg["throughput_tokens_per_sec"] = mo.throughput_tokens_per_sec
            if mo.cost_per_hour is not None:
                cfg["cost_per_hour"] = mo.cost_per_hour
            if mo.cost_per_million_tokens is not None:
                cfg["cost_per_million_tokens"] = mo.cost_per_million_tokens
            if mo.estimated_runtime_hours is not None:
                cfg["estimated_runtime_hours"] = round(mo.estimated_runtime_hours, 2)
            if mo.meets_slo is not None:
                cfg["meets_slo"] = mo.meets_slo
            configs.append(cfg)
        if not configs:
            return {
                "status": "error",
                "error_type": "no_candidates",
                "message": "Advisor found no feasible configurations. Check model name and quota.",
            }
    else:
        return {
            "status": "error",
            "error_type": "invalid_solver",
            "message": f"Unknown solver: '{use_solver}'. Use 'roofline', 'llm', or 'user_specified'.",
        }

    # Context length check
    max_output = request.max_output_tokens or request.avg_output_tokens
    context_warning = None
    if configs and configs[0].get("max_model_len") and max_input_tokens:
        required_context = max_input_tokens + max_output
        if required_context > configs[0]["max_model_len"]:
            context_warning = (
                f"Longest prompt ({max_input_tokens} tokens) + max_output ({max_output}) = "
                f"{required_context} exceeds max_model_len ({configs[0]['max_model_len']})"
            )

    # SLO warning
    slo_warning = None
    if configs and configs[0].get("meets_slo") is False:
        eta = configs[0].get("estimated_runtime_hours", "?")
        slo_warning = (
            f"No config meets the {request.slo_deadline_hours:.1f}h SLO. "
            f"Best estimate: {eta}h. Consider adding more replicas or relaxing the deadline."
        )

    response = {
        "status": "ok",
        "solver": use_solver,
        "input_stats": {
            "num_lines": num_lines,
            "avg_input_tokens": avg_input_tokens,
            "max_input_tokens": max_input_tokens,
        },
        "placements": configs,
    }
    if context_warning:
        response["context_warning"] = context_warning
    if slo_warning:
        response["slo_warning"] = slo_warning
    if solver_log:
        response["solver_log"] = solver_log
        os.makedirs("temp", exist_ok=True)
        with open("temp/solver.log", "w") as f:
            f.write(solver_log)

    return response


@app.post("/submit/online")
async def submit_online(request: OnlineServingRequest):
    """
    Submit an online inference job request.

    Receives a OnlineServingRequest and returns a confirmation of receipt.
    """
    launch_config = real_magic(request)
    logger.info(f"[Online] Launch config: {launch_config}")

    match launch_config.engine:
        case "vllm":
            endpoint_url = await sp_launch_vllm_online(request, launch_config)
            return {
                "status": "success",
                "job_id": launch_config.decision_id,
                "endpoint": endpoint_url,
                "model": request.model_name,
                "message": f"vLLM server launched at {endpoint_url}",
            }


##### Storage stuff #####
@app.post("/storage/presigned_upload")
async def presign_upload(
    remote_path: str = Form(...), user: str = Form(...), expires: int = Form(600)
):
    payload = await storage_backend.presigned_upload(remote_path, user, expires)
    return {"status": "success", **payload}


@app.get("/storage/presigned_download")
async def presign_download(user: str, remote_path: str, expires: int = 600):
    payload = await storage_backend.presigned_download(remote_path, user, expires)
    return {"status": "success", **payload}


@app.post("/storage/upload")
async def upload_file_to_storage(
    file: UploadFile = File(...), remote_path: str = Form(None), user: str = Form("default")
):
    """Upload a file to storage backend using streaming via temp file.
    If remote_path is omitted, the server auto-generates an S3 URI under S3_UPLOAD_BUCKET/S3_UPLOAD_PREFIX."""
    try:
        if not remote_path:
            filename = file.filename or f"upload_{int(time.time())}"
            remote_path = f"s3://{S3_UPLOAD_BUCKET}/{S3_UPLOAD_PREFIX}/{user}/{int(time.time())}_{filename}"
        elif not remote_path.startswith("s3://"):
            remote_path = f"s3://{S3_UPLOAD_BUCKET}/{S3_UPLOAD_PREFIX}/{user}/{remote_path}"
        logger.info(f"[Storage] Uploading file for user {user} to {remote_path}")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            chunk_size = CHUNK_SIZE_BYTES
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        storage_uri = await storage_backend.upload_file(tmp_path, remote_path, user)
        os.unlink(tmp_path)
        logger.info(f"[Storage] Successfully uploaded file to {storage_uri}")
        return {
            "status": "success",
            "storage_uri": storage_uri,
            "remote_path": remote_path,
            "user": user,
            "filename": file.filename,
        }
    except Exception as e:
        logger.error(f"[Storage] Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/storage/list/{user}")
async def list_user_files(user: str, prefix: str = ""):
    """List all files for a user in their storage space."""
    try:
        logger.info(f"[Storage] Listing files for user {user} with prefix '{prefix}'")
        files = await storage_backend.list_files(prefix, user)
        return {
            "status": "success",
            "user": user,
            "prefix": prefix,
            "files": files,
            "count": len(files),
        }
    except Exception as e:
        logger.error(f"[Storage] Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.get("/storage/download/{user}/{file_path:path}")
async def download_file_from_storage(user: str, file_path: str):
    """Download a file from storage backend using streaming."""
    try:
        logger.info(f"[Storage] Downloading file {file_path} for user {user}")
        filename = file_path.split("/")[-1] or "download"

        async def file_stream_iterator():
            async for chunk in storage_backend.stream_file(file_path, user):
                yield chunk

        return StreamingResponse(
            file_stream_iterator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"[Storage] Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.get("/storage/download_s3")
async def download_s3_file(path: str, user: str = "default"):
    """Download a file by full S3 URI (s3://bucket/key). Used by chunked runners."""
    if not path.startswith("s3://"):
        raise HTTPException(status_code=400, detail="path must be a full s3:// URI")
    try:
        logger.info(f"[Storage] download_s3 path={path} user={user}")
        filename = path.split("/")[-1] or "download"

        async def file_stream_iterator():
            async for chunk in storage_backend.stream_file(path, user):
                yield chunk

        return StreamingResponse(
            file_stream_iterator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"[Storage] Error downloading S3 file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.delete("/storage/delete/{user}/{file_path:path}")
async def delete_file_from_storage(user: str, file_path: str):
    """Delete a file from storage backend."""
    try:
        logger.info(f"[Storage] Deleting file {file_path} for user {user}")
        success = await storage_backend.delete_file(file_path, user)
        if success:
            return {
                "status": "success",
                "message": f"File {file_path} deleted successfully",
                "user": user,
                "file_path": file_path,
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete file {file_path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Storage] Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.delete("/storage/delete_s3")
async def delete_s3_file(path: str, user: str = "default"):
    """Delete a file by full S3 URI (s3://bucket/key)."""
    if not path.startswith("s3://"):
        raise HTTPException(status_code=400, detail="path must be a full s3:// URI")
    try:
        logger.info(f"[Storage] delete_s3 path={path} user={user}")
        success = await storage_backend.delete_file(path, user)
        if success:
            return {"status": "success", "path": path}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete {path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Storage] Error deleting S3 file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.get("/storage/exists/{user}/{file_path:path}")
async def check_file_exists(user: str, file_path: str):
    """Check if a file exists in storage backend."""
    try:
        exists = await storage_backend.file_exists(file_path, user)
        return {
            "status": "success",
            "user": user,
            "file_path": file_path,
            "exists": exists,
        }
    except Exception as e:
        logger.error(f"[Storage] Error checking file existence: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking file: {str(e)}")


@app.post("/storage/multipart/start")
async def multipart_start(remote_path: str = Form(...), user: str = Form(...)):
    return await storage_backend.multipart_upload_start(remote_path, user)


@app.post("/storage/multipart/sign-part")
async def multipart_sign_part(
    upload_id: str = Form(...),
    user: str = Form(...),
    remote_path: str = Form(...),
    part_number: int = Form(...),
    expires: int = Form(600),
):
    return await storage_backend.multipart_sign_part(
        upload_id, user, remote_path, part_number, expires
    )


@app.post("/storage/multipart/complete")
async def multipart_complete(
    user: str = Form(...),
    remote_path: str = Form(...),
    upload_id: str = Form(...),
    parts: str = Form(...),
):
    parts_list = json.loads(parts)
    if not isinstance(parts_list, list):
        raise ValueError("parts is not a list")
    return await storage_backend.multipart_complete(
        remote_path, user, upload_id, parts_list
    )


if __name__ == "__main__":
    import argparse as _ap
    import uvicorn

    _parser = _ap.ArgumentParser(description="Orca control plane server")
    _parser.add_argument("--url", help="Public URL for this server (e.g. Cloudflare tunnel URL). Overrides ORCA_SERVER_URL env var.")
    _parser.add_argument("--port", type=int, default=26336)
    _parser.add_argument("--tunnel", action="store_true", help="Auto-start a Cloudflare tunnel and use its URL")
    _args = _parser.parse_args()

    if _args.tunnel:
        import sys
        import subprocess as _sp
        import re as _re
        import signal as _sig

        import shutil as _sh
        _cf_bin = _sh.which("cloudflared") or os.path.join(os.path.dirname(__file__), "cloudflared")
        if not os.path.isfile(_cf_bin):
            print("[Server] ERROR: cloudflared not found on PATH or in repo root")
            sys.exit(1)
        print(f"[Server] Starting Cloudflare tunnel on port {_args.port}...")
        _tunnel_proc = _sp.Popen(
            [_cf_bin, "tunnel", "--url", f"http://localhost:{_args.port}"],
            stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
        )
        _tunnel_url = None
        for _line in iter(_tunnel_proc.stdout.readline, ""):
            _m = _re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", _line)
            if _m:
                _tunnel_url = _m.group(1)
                break
        if not _tunnel_url:
            print("[Server] ERROR: Could not detect Cloudflare tunnel URL")
            _tunnel_proc.kill()
            sys.exit(1)
        _args.url = _tunnel_url
        print(f"[Server] Tunnel ready: {_tunnel_url}")

        def _cleanup_tunnel(*_a):
            _tunnel_proc.terminate()
            _tunnel_proc.wait(timeout=5)
        import atexit
        atexit.register(_cleanup_tunnel)
        _sig.signal(_sig.SIGINT, lambda *_a: (atexit._run_exitfuncs(), sys.exit(0)))
        _sig.signal(_sig.SIGTERM, lambda *_a: (atexit._run_exitfuncs(), sys.exit(0)))

    if _args.url:
        from orca_server import config as _cfg
        _cfg.ORCA_SERVER_URL = _args.url
        os.environ["ORCA_SERVER_URL"] = _args.url
        print(f"[Server] ORCA_SERVER_URL set to {_args.url}")

    uvicorn.run(app, host="0.0.0.0", port=_args.port)
