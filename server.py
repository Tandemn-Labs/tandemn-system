from contextlib import asynccontextmanager
import asyncio
import logging
import math
import uuid
import time
import os
import tempfile
import json
from typing import Optional

import sky
from fastapi import FastAPI, Form, Header, UploadFile, File, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

from orca_server.config import (
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
)
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
    # Startup logic
    # Solvers are now created per-request in submit_batch()

    # Initialize cluster manager
    app.state.cluster_manager = get_cluster_manager()

    # Initialize quota tracker (replays persisted reservations from SQLite)
    app.state.quota_tracker = VPCQuotaTracker()

    from orca_server.monitoring import get_metrics_collector
    from orca_server.metrics_db import get_metrics_db
    app.state.metrics_collector = get_metrics_collector()
    app.state.metrics_db = get_metrics_db()

    # Reconcile stale reservations against live SkyPilot clusters
    try:
        request_id = sky.status()
        clusters = sky.get(request_id)
        live = {c['name'] for c in clusters} if clusters else set()
        app.state.quota_tracker.reconcile(live)
    except Exception as e:
        logger.warning(f"[Quota] Could not reconcile on startup: {e}")

    yield

    # Shutdown: join non-daemon monitor threads so they can finish teardown
    cm = app.state.cluster_manager
    threads = cm.get_active_threads()
    if threads:
        logger.info(f"[Shutdown] Waiting for {len(threads)} monitor thread(s) to finish teardown...")
        for name, t in threads.items():
            logger.info(f"[Shutdown] Joining thread for cluster {name}...")
            t.join(timeout=120)
            if t.is_alive():
                logger.warning(f"[Shutdown] Thread for {name} did not finish within 120s")
        logger.info("[Shutdown] All monitor threads joined.")


app = FastAPI(
    title="Tandemn Server",
    description="API for receiving job requests",
    lifespan=lifespan,
)

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
    snap = app.state.metrics_collector.get_latest(job_id)
    if snap is None:
        raise HTTPException(404, f"No metrics for {job_id}")
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

    ingested = 0
    batch_for_db = []
    prev_snap = None

    for item in snapshots_raw:
        ts   = item.get("timestamp", time.time())
        text = item.get("prometheus_text", "")
        if not text.strip():
            continue

        snap = MetricsSnapshot.from_prometheus_text(job_id, text, ts)

        # Compute throughput from counter deltas (vLLM >=0.10 removed the gauge)
        if snap.avg_generation_throughput_toks_per_s == 0 and prev_snap is not None:
            dt = snap.timestamp - prev_snap.timestamp
            if dt > 0:
                snap.avg_generation_throughput_toks_per_s = (
                    snap.generation_tokens_total - prev_snap.generation_tokens_total
                ) / dt
                snap.avg_prompt_throughput_toks_per_s = (
                    snap.prompt_tokens_total - prev_snap.prompt_tokens_total
                ) / dt
        prev_snap = snap

        # Write into ring buffer for live SSE / REST
        with mc._lock:
            jc = mc._jobs.get(job_id)
        if jc:
            with jc.lock:
                jc.buffer.append(snap)
                # Don't add to _unflushed — we persist directly below

        batch_for_db.append(snap.to_dict())
        ingested += 1

    # Persist timeseries directly (sidecar already batched; skip 60s flush timer)
    if batch_for_db:
        try:
            db.append_timeseries(job_id, batch_for_db)
        except Exception as e:
            logger.warning("[Ingest] timeseries write failed for %s: %s", job_id, e)

    return {"ok": True, "ingested": ingested}


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
    if phase:
        get_job_tracker().update_status(job_id, phase)
        if phase == "generating":
            app.state.metrics_collector.set_baseline(job_id)
    return {"ok": True}


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
        frac = progress["completed"] / progress["total"]
        get_job_tracker().update_progress(job_id, frac)

    if progress["all_done"]:
        asyncio.create_task(_assemble_output(job_id))

    return progress


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
    meta = cm._r.hgetall(f"chunk:job:{job_id}:meta")
    s3_output_base = meta.get("s3_output_base", "")
    ordered_ids = cm.get_output_order(job_id)

    combined_path = f"/tmp/assembly_{job_id}.jsonl"
    try:
        with open(combined_path, "w") as combined:
            for cid in ordered_ids:
                chunk_info = cm.get_chunk_info(job_id, cid)
                s3_out = chunk_info.get("s3_output_path", "")
                local_tmp = f"/tmp/assemble_{job_id}_{cid}.jsonl"
                try:
                    await storage_backend.download_file(s3_out, local_tmp, user="system")
                except Exception as dl_err:
                    job_logger.error(f"[Assembly] Failed to download {s3_out}: {dl_err}")
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

        # Download to local outputs/
        from orca_server.job_manager import download_output_from_s3
        tracker = get_job_tracker()
        rec = tracker.get(job_id)
        if rec:
            job_dirname = getattr(rec, '_job_dirname', job_id)
            download_output_from_s3(final_s3, job_dirname, logger=job_logger)

        get_job_tracker().update_status(job_id, "succeeded")
        get_job_tracker().update_progress(job_id, 1.0)
        cm.cleanup_job(job_id)
        job_logger.info(f"[Assembly] Job {job_id} completed successfully")

    except Exception as e:
        job_logger.error(f"[Assembly] Failed for {job_id}: {e}")
        get_job_tracker().update_status(job_id, "failed")
        if os.path.exists(combined_path):
            os.unlink(combined_path)


@app.get("/job/{job_id}/throughput")
async def get_job_throughput(job_id: str, window: float = 60.0):
    """Sustained throughput for the controller: rolling window + epoch (since baseline)."""
    result = app.state.metrics_collector.get_sustained_throughput(job_id, window)
    if result is None:
        raise HTTPException(404, "No throughput data (insufficient samples)")
    result["job_id"] = job_id
    return result


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
    if request.chunks and len(request.chunks) > 1:
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

        msg = f"[Chunked] {len(request.chunks)} chunks, {effective_replicas} replicas"
        logger.info(msg)
        early_messages.append(("INFO", msg))

        success = await launch_chunked_replicas(
            request, primary, effective_replicas,
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
        return {
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
            configs.append(cfg)
    else:
        return {
            "status": "error",
            "error_type": "invalid_solver",
            "message": f"Unknown solver: '{use_solver}'. Use 'roofline' or 'user_specified'.",
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
    _args = _parser.parse_args()

    if _args.url:
        from orca_server import config as _cfg
        _cfg.ORCA_SERVER_URL = _args.url
        os.environ["ORCA_SERVER_URL"] = _args.url
        print(f"[Server] ORCA_SERVER_URL set to {_args.url}")

    uvicorn.run(app, host="0.0.0.0", port=_args.port)
