"""ReplicaWatchdog — heartbeat-based dead replica detection + force-reclaim + recovery.

Uses the existing MetricsCollector per-replica ring buffers as the heartbeat source:
the timestamp of the most recent snapshot IS the last heartbeat. Zero-cost since the
buffer already exists from sidecar ingest endpoint writes.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

from orca_server.config import (
    RECOVERY_COOLDOWN_SEC,
    REPLICA_DEAD_THRESHOLD_SEC,
    WATCHDOG_POLL_INTERVAL_SEC,
)

if TYPE_CHECKING:
    from orca_server.chunk_manager import ChunkManager
    from orca_server.job_manager import ClusterManager, JobTracker
    from orca_server.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class ReplicaWatchdog:
    """Detect dead replicas via sidecar heartbeat, force-reclaim chunks, optionally recover."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        cluster_manager: ClusterManager,
        job_tracker: JobTracker,
        chunk_manager_fn: Callable[[], ChunkManager],
        dead_threshold_sec: float = REPLICA_DEAD_THRESHOLD_SEC,
        poll_interval_sec: float = WATCHDOG_POLL_INTERVAL_SEC,
        recovery_cooldown_sec: float = RECOVERY_COOLDOWN_SEC,
        assembly_callback: Callable[[str], None] | None = None,
        recover_callback: Callable[[str, str], None] | None = None,
    ):
        self._mc = metrics_collector
        self._cm = cluster_manager
        self._jt = job_tracker
        self._chunk_manager_fn = chunk_manager_fn
        self._dead_threshold = dead_threshold_sec
        self._poll_interval = poll_interval_sec
        self._recovery_cooldown = recovery_cooldown_sec
        self._assembly_callback = assembly_callback
        self._recover_callback = recover_callback

        # replica_id → timestamp when declared dead (avoid re-processing)
        self._dead_replicas: dict[str, float] = {}
        # replica_id → timestamp of last recovery attempt
        self._last_recovery: dict[str, float] = {}

    def clear_dead(self, replica_id: str) -> None:
        """Clear dead tracking for a replica (e.g. after it recovers and sends heartbeat)."""
        self._dead_replicas.pop(replica_id, None)

    async def run(self) -> None:
        """Main loop — call via asyncio.create_task() in server lifespan."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                self._check_all_jobs()
            except Exception as e:
                logger.warning("[Watchdog] Error in poll cycle: %s", e)

    def _check_all_jobs(self) -> None:
        """Scan all active chunked jobs for dead replicas."""
        from orca_server.job_manager import get_job_tracker

        jt = self._jt
        now = time.time()

        for job_id, rec in list(jt.jobs.items()):
            if not getattr(rec, "is_chunked", False):
                continue
            if rec.status in ("succeeded", "failed", "cancelled"):
                continue

            states = self._cm.get_replica_states(job_id)
            for replica_id, state in states.items():
                phase = state.get("phase", "")
                if phase in ("launching", "provisioned", "completed", "swapped_out", "killed"):
                    continue  # not expected to heartbeat yet (or already done)

                # Failed replicas: force-reclaim their chunks (monitor thread set phase
                # to "failed" before watchdog could catch it as stale "running")
                if phase == "failed" and replica_id not in self._dead_replicas:
                    self._handle_dead_replica(job_id, replica_id, None)
                    continue

                if phase == "dead" and replica_id in self._dead_replicas:
                    continue  # already handled

                last_hb = self._get_last_heartbeat(job_id, replica_id)

                if phase == "running" and last_hb is not None:
                    if now - last_hb > self._dead_threshold:
                        self._handle_dead_replica(job_id, replica_id, last_hb)
                elif phase == "running" and last_hb is None:
                    # Running but never sent a heartbeat — check how long in running state
                    running_since = state.get("running_since")
                    if running_since and now - running_since > self._dead_threshold:
                        self._handle_dead_replica(job_id, replica_id, None)

    def _get_last_heartbeat(self, job_id: str, replica_id: str) -> float | None:
        """Extract last heartbeat timestamp from the replica's ring buffer."""
        key = f"{job_id}:{replica_id}"
        with self._mc._lock:
            rc = self._mc._replicas.get(key)
        if rc is None:
            return None
        with rc.lock:
            if not rc.buffer:
                return None
            return rc.buffer[-1].timestamp

    def _is_graceful_completion(self, job_id: str, replica_id: str) -> bool:
        """Check if a replica finished its work gracefully (not a failure)."""
        state = self._cm.get_replica_states(job_id).get(replica_id, {})
        if state.get("phase") not in ("running",):
            return False
        # Check if it actually processed requests
        snap = self._mc.get_replica_latest(job_id, replica_id)
        if snap is None or snap.request_success_total <= 0:
            return False
        # Job must still be active (not already terminal)
        rec = self._jt.get(job_id)
        if rec and rec.status in ("succeeded", "failed", "cancelled"):
            return False
        return True

    def _handle_dead_replica(self, job_id: str, replica_id: str, last_hb: float | None) -> None:
        """Force-reclaim chunks and optionally schedule recovery."""
        if replica_id in self._dead_replicas:
            return  # already processed

        # Check for graceful completion before treating as dead
        if self._is_graceful_completion(job_id, replica_id):
            self._dead_replicas[replica_id] = time.time()
            logger.info(
                "[Watchdog] Replica %s completed gracefully (last heartbeat: %s)",
                replica_id, f"{last_hb:.1f}" if last_hb else "never",
            )
            self._cm.set_replica_state(job_id, replica_id, phase="completed")
            self._mc.exclude_replica(job_id, replica_id)
            # Trigger cluster teardown
            from orca_server.job_manager import sky_down_with_retry
            sky_down_with_retry(replica_id)
            return

        self._dead_replicas[replica_id] = time.time()
        logger.warning(
            "[Watchdog] Replica %s declared DEAD (last heartbeat: %s, threshold: %ss)",
            replica_id, f"{last_hb:.1f}" if last_hb else "never", self._dead_threshold,
        )

        # 1. Update replica phase
        self._cm.set_replica_state(job_id, replica_id, phase="dead")

        # 2. Force-reclaim inflight chunks
        cm = self._chunk_manager_fn()
        result = cm.force_reclaim(job_id, [replica_id])
        logger.info(
            "[Watchdog] Force-reclaimed for %s: reclaimed=%d, failed=%d",
            replica_id, result["reclaimed"], result["failed"],
        )

        # 3. Check if force-reclaim completed the job (failed chunks → all_done)
        progress = cm.get_progress(job_id)
        if progress and progress["all_done"]:
            logger.info("[Watchdog] Job %s is all_done after force-reclaim, triggering assembly", job_id)
            if self._assembly_callback:
                self._assembly_callback(job_id)
            return

        # 4. Decide whether to recover
        if not self._should_recover(job_id, replica_id):
            return

        # 5. Schedule recovery
        self._last_recovery[replica_id] = time.time()
        logger.info("[Watchdog] Scheduling recovery for %s", replica_id)
        if self._recover_callback:
            self._recover_callback(job_id, replica_id)

    def _should_recover(self, job_id: str, replica_id: str) -> bool:
        """Decide whether to relaunch a dead replica."""
        # Cooldown: don't recover if we recently tried
        last = self._last_recovery.get(replica_id, 0)
        if time.time() - last < self._recovery_cooldown:
            logger.info("[Watchdog] Recovery cooldown active for %s, skipping", replica_id)
            return False

        # Job nearly done: cold start waste
        cm = self._chunk_manager_fn()
        progress = cm.get_progress(job_id)
        if progress is None:
            return False
        total = progress["total"]
        if total == 0:
            return False
        done_frac = (progress["completed"] + progress["failed"]) / total
        if done_frac >= 0.95:
            logger.info(
                "[Watchdog] Job %s is %.0f%% done, skipping recovery for %s",
                job_id, done_frac * 100, replica_id,
            )
            return False

        # Job already terminal
        rec = self._jt.get(job_id)
        if rec and rec.status in ("succeeded", "failed", "cancelled"):
            return False

        return True
