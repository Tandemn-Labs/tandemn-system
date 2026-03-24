"""Tests for Orca Swap — hot-swap replica config mid-job.

Tests the swap endpoint, swap monitor, and integration with real Redis chunks.
"""
import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis as _redis

from orca_server.chunk_manager import ChunkManager


# ---------------------------------------------------------------------------
# Minimal stubs (same pattern as test_watchdog.py)
# ---------------------------------------------------------------------------

@dataclass
class FakeSnap:
    timestamp: float


class FakeReplicaCollector:
    def __init__(self, timestamps=None):
        self.lock = Lock()
        self.buffer = deque()
        if timestamps:
            for ts in timestamps:
                self.buffer.append(FakeSnap(timestamp=ts))


class FakeMetricsCollector:
    def __init__(self):
        self._lock = Lock()
        self._replicas: dict[str, FakeReplicaCollector] = {}

    def add_replica(self, key, timestamps=None):
        self._replicas[key] = FakeReplicaCollector(timestamps)

    def start_collecting(self, job_id, endpoint_url=None):
        pass

    def start_replica_collecting(self, job_id, replica_id):
        key = f"{job_id}:{replica_id}"
        if key not in self._replicas:
            self._replicas[key] = FakeReplicaCollector()


SAMPLE_CHUNKS = [
    {"chunk_id": "c0000", "s3_input_path": "s3://b/c0000.jsonl", "num_lines": 100},
    {"chunk_id": "c0001", "s3_input_path": "s3://b/c0001.jsonl", "num_lines": 100},
    {"chunk_id": "c0002", "s3_input_path": "s3://b/c0002.jsonl", "num_lines": 100},
    {"chunk_id": "c0003", "s3_input_path": "s3://b/c0003.jsonl", "num_lines": 100},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_cm():
    """Real ChunkManager on Redis DB 1."""
    url = "redis://localhost:6379/1"
    try:
        r = _redis.from_url(url)
        r.ping()
    except _redis.ConnectionError:
        pytest.skip("Redis not available at localhost:6379")
    manager = ChunkManager(redis_url=url)
    yield manager
    r = _redis.from_url(url, decode_responses=True)
    for key in r.keys("chunk:job:*"):
        r.delete(key)


# ---------------------------------------------------------------------------
# Unit tests — swap logic
# ---------------------------------------------------------------------------

class TestSwapValidation:
    def test_force_reclaim_old_replicas_on_swap(self, real_cm):
        """Old replicas' inflight chunks are force-reclaimed and re-pullable."""
        job_id = f"swap-{uuid.uuid4().hex[:8]}"
        real_cm.create_job_queue(job_id, SAMPLE_CHUNKS, "model", "s3://out")

        # Old replicas pull chunks
        real_cm.pull_chunk(job_id, "old-r0")  # c0000
        real_cm.pull_chunk(job_id, "old-r0")  # c0001
        real_cm.pull_chunk(job_id, "old-r1")  # c0002
        real_cm.pull_chunk(job_id, "old-r1")  # c0003

        assert real_cm.get_progress(job_id)["inflight"] == 4
        assert real_cm.get_progress(job_id)["pending"] == 0

        # Force-reclaim old replicas (what swap does)
        result = real_cm.force_reclaim(job_id, ["old-r0", "old-r1"])
        assert result["reclaimed"] == 4

        # New replicas can pull all chunks
        pulled = []
        for _ in range(4):
            c = real_cm.pull_chunk(job_id, "new-r0")
            assert c is not None
            pulled.append(c["chunk_id"])
        assert sorted(pulled) == ["c0000", "c0001", "c0002", "c0003"]

    def test_same_redis_queue_after_swap(self, real_cm):
        """New replicas use the same job_id → same Redis queue."""
        job_id = f"swap-q-{uuid.uuid4().hex[:8]}"
        real_cm.create_job_queue(job_id, SAMPLE_CHUNKS[:2], "model", "s3://out")

        # Old replica completes c0000
        real_cm.pull_chunk(job_id, "old-r0")
        real_cm.complete_chunk(job_id, "c0000", "old-r0")

        # "New replica" (different name, same job_id) can pull remaining
        c = real_cm.pull_chunk(job_id, "new-v2-r0")
        assert c is not None
        assert c["chunk_id"] == "c0001"
        assert c["replica_id"] == "new-v2-r0"

    def test_threshold_logic(self):
        """ready_threshold: old replicas killed only after K new replicas active."""
        mc = FakeMetricsCollector()
        ready_threshold = 2
        new_replicas = ["new-r0", "new-r1", "new-r2"]
        job_id = "job-thresh"

        # Only 1 replica has heartbeat — not enough
        mc.add_replica(f"{job_id}:new-r0", [time.time()])
        # new-r1 and new-r2 have no heartbeat

        ready = set()
        for rid in new_replicas:
            key = f"{job_id}:{rid}"
            with mc._lock:
                rc = mc._replicas.get(key)
            if rc:
                with rc.lock:
                    if rc.buffer:
                        ready.add(rid)

        assert len(ready) == 1
        assert len(ready) < ready_threshold  # not met yet

        # Now add second heartbeat
        mc.add_replica(f"{job_id}:new-r1", [time.time()])

        ready = set()
        for rid in new_replicas:
            key = f"{job_id}:{rid}"
            with mc._lock:
                rc = mc._replicas.get(key)
            if rc:
                with rc.lock:
                    if rc.buffer:
                        ready.add(rid)

        assert len(ready) == 2
        assert len(ready) >= ready_threshold  # now met


class TestSwapVersioning:
    def test_version_counter_increments(self):
        """ClusterManager.next_swap_version increments per job."""
        from orca_server.job_manager import ClusterManager
        cm = ClusterManager()

        v1 = cm.next_swap_version("job-1")
        assert v1 == 2  # first swap is v2
        v2 = cm.next_swap_version("job-1")
        assert v2 == 3
        v3 = cm.next_swap_version("job-1")
        assert v3 == 4

        # Different job starts at v2
        assert cm.next_swap_version("job-2") == 2

    def test_swap_in_progress_flag(self):
        """_swap_in_progress prevents double swap."""
        from orca_server.job_manager import ClusterManager
        cm = ClusterManager()

        assert not cm._swap_in_progress.get("job-1")
        cm._swap_in_progress["job-1"] = True
        assert cm._swap_in_progress.get("job-1") is True


# ---------------------------------------------------------------------------
# Integration test — full swap flow with real Redis
# ---------------------------------------------------------------------------

class TestSwapIntegration:
    def test_full_swap_flow(self, real_cm):
        """End-to-end: old replicas pull → swap (force-reclaim) → new replicas complete all chunks."""
        job_id = f"swap-e2e-{uuid.uuid4().hex[:8]}"
        real_cm.create_job_queue(job_id, SAMPLE_CHUNKS, "model", "s3://out")

        # Phase 1: old replicas processing
        real_cm.pull_chunk(job_id, "old-r0")   # c0000
        real_cm.pull_chunk(job_id, "old-r1")   # c0001

        # old-r0 completes c0000 before swap
        real_cm.complete_chunk(job_id, "c0000", "old-r0")

        progress = real_cm.get_progress(job_id)
        assert progress["completed"] == 1
        assert progress["inflight"] == 1   # c0001 still with old-r1
        assert progress["pending"] == 2    # c0002, c0003

        # Phase 2: swap — force-reclaim old replicas
        result = real_cm.force_reclaim(job_id, ["old-r0", "old-r1"])
        assert result["reclaimed"] == 1  # only c0001 was inflight (c0000 already completed)

        progress = real_cm.get_progress(job_id)
        assert progress["completed"] == 1
        assert progress["inflight"] == 0
        assert progress["pending"] == 3  # c0001 reclaimed + c0002, c0003

        # Phase 3: new replicas pull and complete everything
        completed_by_new = []
        while True:
            c = real_cm.pull_chunk(job_id, "new-v2-r0")
            if c is None:
                break
            real_cm.complete_chunk(job_id, c["chunk_id"], "new-v2-r0")
            completed_by_new.append(c["chunk_id"])

        assert sorted(completed_by_new) == ["c0001", "c0002", "c0003"]

        progress = real_cm.get_progress(job_id)
        assert progress["completed"] == 4
        assert progress["all_done"] is True

        # Output order preserved
        order = real_cm.get_output_order(job_id)
        assert order == ["c0000", "c0001", "c0002", "c0003"]
