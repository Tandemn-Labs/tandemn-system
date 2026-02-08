"""
Background async tasks that run alongside the FastAPI server.

- lease_reaper_loop: reclaims chunks from dead/preempted workers
- combiner_loop: concatenates output chunks when a job completes
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def lease_reaper_loop(app, interval: int = 30):
    """Periodically reap expired leases for all active jobs.

    Runs every `interval` seconds. For each job with status
    'launching' or 'running', calls reap_expired_leases to
    re-enqueue chunks whose workers died (lease TTL expired).
    """
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await _reap_all_jobs(app)
            except Exception:
                logger.exception("Reaper tick failed")
    except asyncio.CancelledError:
        logger.info("Lease reaper stopped")


async def _reap_all_jobs(app):
    """Scan for active jobs and reap their expired leases."""
    r = app.state.redis
    cq = app.state.chunk_queue

    active_statuses = {"queued", "launching", "running"}
    async for key in r.scan_iter("job:*:meta"):
        parts = key.split(":")
        if len(parts) != 3:
            continue
        job_id = parts[1]

        status = await r.hget(key, "status")
        if status not in active_statuses:
            continue

        reaped = await cq.reap_expired_leases(job_id)
        if reaped > 0:
            logger.info(f"Reaper: reclaimed {reaped} expired leases for job {job_id}")

        # Transition from 'launching' to 'running' once work starts
        if status == "launching":
            completed = await r.hget(f"job:{job_id}:meta", "completed_count")
            if completed and int(completed) > 0:
                await cq.update_job_status(job_id, "running")


async def combiner_loop(app, interval: int = 15):
    """Periodically check if any running jobs are done and update status.

    When all chunks for a job are completed (or failed), marks the job
    as 'succeeded' or 'failed'. Output combining is left as a TODO
    since it depends on the storage backend abstraction.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await _check_completed_jobs(app)
            except Exception:
                logger.exception("Combiner tick failed")
    except asyncio.CancelledError:
        logger.info("Combiner stopped")


async def _check_completed_jobs(app):
    """Scan for running jobs and finalize when all chunks are done."""
    r = app.state.redis
    cq = app.state.chunk_queue

    async for key in r.scan_iter("job:*:meta"):
        parts = key.split(":")
        if len(parts) != 3:
            continue
        job_id = parts[1]

        status = await r.hget(key, "status")
        if status not in {"launching", "running"}:
            continue

        progress = await cq.get_job_progress(job_id)
        if not progress.is_done:
            continue

        if progress.failed > 0:
            await cq.update_job_status(job_id, "failed")
            logger.warning(
                f"Job {job_id} finished with {progress.failed} failed chunks "
                f"({progress.completed} completed)"
            )
        else:
            # TODO: trigger actual output combining via storage_backend
            # Workers wrote output to {prefix}/output/{chunk_id}.jsonl
            # Combined output should go to {prefix}/output/final_output.jsonl
            await cq.update_job_status(job_id, "succeeded")
            logger.info(
                f"Job {job_id}: all {progress.completed} chunks completed, "
                f"status=succeeded"
            )
