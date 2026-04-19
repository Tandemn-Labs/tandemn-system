"""ReplicaWatchdog — heartbeat-based dead replica detection + force-reclaim.

Uses the existing MetricsCollector per-replica ring buffers as the heartbeat source:
the timestamp of the most recent snapshot IS the last heartbeat. Zero-cost since the
buffer already exists from sidecar ingest endpoint writes.

Recovery is handled by Koi: watchdog fires /job/replica-failed webhook → Koi's agent
decides config → calls scale_chain_tool → Orca's /job/{id}/scale launches replacement.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

from orca_server.config import (
    REPLICA_DEAD_THRESHOLD_SEC,
    WATCHDOG_POLL_INTERVAL_SEC,
)

if TYPE_CHECKING:
    from orca_server.chunk_manager import ChunkManager
    from orca_server.job_manager import ClusterManager, JobTracker
    from orca_server.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class ReplicaWatchdog:
    """Detect dead replicas via sidecar heartbeat, force-reclaim chunks."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        cluster_manager: ClusterManager,
        job_tracker: JobTracker,
        chunk_manager_fn: Callable[[], ChunkManager],
        dead_threshold_sec: float = REPLICA_DEAD_THRESHOLD_SEC,
        poll_interval_sec: float = WATCHDOG_POLL_INTERVAL_SEC,
        assembly_callback: Callable[[str], None] | None = None,
    ):
        self._mc = metrics_collector
        self._cm = cluster_manager
        self._jt = job_tracker
        self._chunk_manager_fn = chunk_manager_fn
        self._dead_threshold = dead_threshold_sec
        self._poll_interval = poll_interval_sec
        self._assembly_callback = assembly_callback

        # replica_id → timestamp when declared dead (avoid re-processing)
        self._dead_replicas: dict[str, float] = {}

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
            if rec.status in ("succeeded", "failed", "cancelled"):
                continue

            states = self._cm.get_replica_states(job_id)
            for replica_id, state in states.items():
                phase = state.get("phase", "")
                if phase in (
                    "launching",
                    "provisioned",
                    "completed",
                    "swapped_out",
                    "killed",
                ):
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
        """Check if a replica finished its work gracefully (not a failure).

        Uses per-replica inflight count, NOT job-level all_done. A replica is
        graceful only if it owns zero inflight chunks. This prevents the race
        where OTHER replicas finish the job while THIS replica dies with work.
        """
        state = self._cm.get_replica_states(job_id).get(replica_id, {})
        if state.get("phase") not in ("running",):
            return False
        rec = self._jt.get(job_id)
        if rec and rec.status in ("succeeded", "failed", "cancelled"):
            return False
        cm = self._chunk_manager_fn()
        try:
            inflight = cm.get_replica_inflight_count(job_id, replica_id)
            return inflight == 0
        except Exception:
            return False

    def _handle_dead_replica(
        self, job_id: str, replica_id: str, last_hb: float | None
    ) -> None:
        """Force-reclaim chunks and notify Koi. Recovery is Koi's responsibility."""
        if replica_id in self._dead_replicas:
            return  # already processed

        # Check for graceful completion before treating as dead
        if self._is_graceful_completion(job_id, replica_id):
            self._dead_replicas[replica_id] = time.time()
            logger.info(
                "[Watchdog] Replica %s completed gracefully (last heartbeat: %s)",
                replica_id,
                f"{last_hb:.1f}" if last_hb else "never",
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
            replica_id,
            f"{last_hb:.1f}" if last_hb else "never",
            self._dead_threshold,
        )

        # 1. Update replica phase
        self._cm.set_replica_state(job_id, replica_id, phase="dead")
        self._mc.exclude_replica(job_id, replica_id)

        # 1b. Notify Koi that this replica died
        try:
            from orca_server.config import KOI_SERVICE_URL

            if KOI_SERVICE_URL:
                import requests as _req

                state = self._cm.get_replica_states(job_id).get(replica_id, {})
                koi_info = state.get("koi_webhook_info") or {}
                _req.post(
                    f"{KOI_SERVICE_URL}/job/replica-failed",
                    json={
                        "job_id": replica_id,
                        "group_id": job_id,
                        "decision_id": koi_info.get("decision_id"),
                        "instance_type": state.get("instance_type", "unknown"),
                        "region": state.get("region", "unknown"),
                        "market": state.get("market", "unknown"),
                        "status": "failed",
                        "reason": f"Heartbeat timeout ({self._dead_threshold}s)",
                    },
                    timeout=5,
                )
        except Exception as exc:
            logger.warning("[Watchdog] Failed to notify Koi of replica death: %s", exc)

        # 2. Force-reclaim inflight chunks
        cm = self._chunk_manager_fn()
        result = cm.force_reclaim(job_id, [replica_id])
        logger.info(
            "[Watchdog] Force-reclaimed for %s: reclaimed=%d, failed=%d",
            replica_id,
            result["reclaimed"],
            result["failed"],
        )

        # 3. Check if force-reclaim completed the job (failed chunks → all_done)
        progress = cm.get_progress(job_id)
        if progress and progress["all_done"]:
            logger.info(
                "[Watchdog] Job %s is all_done after force-reclaim, triggering assembly",
                job_id,
            )
            if self._assembly_callback:
                self._assembly_callback(job_id)
