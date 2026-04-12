"""Tests for orca_server.watchdog — ReplicaWatchdog dead replica detection."""
import time
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from unittest.mock import MagicMock, patch

import pytest
import redis as _redis

from orca_server.watchdog import ReplicaWatchdog


# ---------------------------------------------------------------------------
# Minimal stubs that replicate the structures the watchdog reads
# ---------------------------------------------------------------------------

@dataclass
class FakeSnap:
    timestamp: float
    request_success_total: float = 0.0


class FakeReplicaCollector:
    """Mimics _JobCollector enough for the watchdog to read buffer[-1].timestamp."""
    def __init__(self, timestamps: list[float] | None = None):
        self.lock = Lock()
        self.buffer = deque()
        if timestamps:
            for ts in timestamps:
                self.buffer.append(FakeSnap(timestamp=ts))


class FakeMetricsCollector:
    def __init__(self):
        self._lock = Lock()
        self._replicas: dict[str, FakeReplicaCollector] = {}

    def add_replica(self, key: str, timestamps: list[float] | None = None):
        self._replicas[key] = FakeReplicaCollector(timestamps)

    def get_replica_latest(self, job_id: str, replica_id: str):
        key = f"{job_id}:{replica_id}"
        rc = self._replicas.get(key)
        if rc is None or not rc.buffer:
            return None
        return rc.buffer[-1]

    def stop_replica_collecting(self, job_id: str, replica_id: str):
        self._replicas.pop(f"{job_id}:{replica_id}", None)

    def exclude_replica(self, job_id: str, replica_id: str):
        """Mimics MetricsCollector.exclude_replica — keeps data but marks excluded."""
        pass  # no-op for tests; watchdog just needs this to exist


class FakeJobRecord:
    def __init__(self, status="running", is_chunked=True):
        self.status = status
        self.is_chunked = is_chunked


class FakeJobTracker:
    def __init__(self):
        self.jobs: dict[str, FakeJobRecord] = {}

    def get(self, job_id):
        return self.jobs.get(job_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mc():
    return FakeMetricsCollector()


@pytest.fixture
def cm_cluster():
    """Mock ClusterManager."""
    mock = MagicMock()
    mock.get_replica_states.return_value = {}
    return mock


@pytest.fixture
def jt():
    return FakeJobTracker()


@pytest.fixture
def chunk_mgr():
    mock = MagicMock()
    mock.force_reclaim.return_value = {"reclaimed": 0, "failed": 0}
    mock.get_progress.return_value = {"total": 20, "completed": 10, "failed": 0,
                                       "pending": 5, "inflight": 5, "all_done": False}
    return mock


@pytest.fixture
def watchdog(mc, cm_cluster, jt, chunk_mgr):
    return ReplicaWatchdog(
        metrics_collector=mc,
        cluster_manager=cm_cluster,
        job_tracker=jt,
        chunk_manager_fn=lambda: chunk_mgr,
        dead_threshold_sec=45,
        poll_interval_sec=1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeadDetection:
    def test_dead_replica_detection(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Replica with stale heartbeat (60s ago) → marked dead, force_reclaim called."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 60])

        watchdog._check_all_jobs()

        cm_cluster.set_replica_state.assert_called_with(job_id, "job-1-r0", phase="dead")
        chunk_mgr.force_reclaim.assert_called_once_with(job_id, ["job-1-r0"])

    def test_healthy_replica_not_flagged(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Replica with fresh heartbeat (5s ago) → not touched."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 5])

        watchdog._check_all_jobs()

        cm_cluster.set_replica_state.assert_not_called()
        chunk_mgr.force_reclaim.assert_not_called()


class TestNoSelfRecovery:
    def test_dead_replica_no_recovery_attempted(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Dead replica: force-reclaim and webhook fire, but NO recovery launched.
        Recovery is Koi's responsibility via scale_chain_tool."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 60])

        watchdog._check_all_jobs()

        # Force-reclaim happened
        chunk_mgr.force_reclaim.assert_called_once()
        # Phase set to dead
        cm_cluster.set_replica_state.assert_called_with(job_id, "job-1-r0", phase="dead")
        # No recovery attributes exist on watchdog
        assert not hasattr(watchdog, '_recover_callback')
        assert not hasattr(watchdog, '_last_recovery')


class TestAllReplicasDie:
    def test_all_replicas_die_simultaneously(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """3 replicas all stale → all 3 detected dead, force_reclaim called for each."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        stale_ts = time.time() - 60
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
            "job-1-r1": {"phase": "running"},
            "job-1-r2": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [stale_ts])
        mc.add_replica(f"{job_id}:job-1-r1", [stale_ts])
        mc.add_replica(f"{job_id}:job-1-r2", [stale_ts])

        watchdog._check_all_jobs()

        assert chunk_mgr.force_reclaim.call_count == 3
        reclaimed_replicas = {call.args[1][0] for call in chunk_mgr.force_reclaim.call_args_list}
        assert reclaimed_replicas == {"job-1-r0", "job-1-r1", "job-1-r2"}


class TestIdempotency:
    def test_dead_replica_not_reprocessed(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """_handle_dead_replica called twice for same ID → force_reclaim only once."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 60])

        watchdog._check_all_jobs()
        assert chunk_mgr.force_reclaim.call_count == 1

        # Second poll — replica is now phase="dead" but also in _dead_replicas
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "dead"},
        }
        watchdog._check_all_jobs()
        assert chunk_mgr.force_reclaim.call_count == 1  # still 1, not 2


class TestIsChunkedGuard:
    def test_non_chunked_job_still_monitored(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """is_chunked guard was removed — watchdog monitors ALL jobs (all jobs are chunked now)."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running", is_chunked=False)
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 120])  # stale, should be dead

        watchdog._check_all_jobs()

        # Watchdog detects the dead replica regardless of is_chunked flag
        cm_cluster.set_replica_state.assert_called()
        chunk_mgr.force_reclaim.assert_called_once()

    def test_chunked_job_detected(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Same job with is_chunked=True → stale replica IS detected dead."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running", is_chunked=True)
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 120])

        watchdog._check_all_jobs()

        cm_cluster.set_replica_state.assert_called_with(job_id, "job-1-r0", phase="dead")
        chunk_mgr.force_reclaim.assert_called_once()


class TestLaunchingGuard:
    def test_launching_replica_not_flagged(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Replica with phase='launching' and no heartbeat → NOT marked dead."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "launching"},
        }
        # No heartbeat entry at all

        watchdog._check_all_jobs()

        cm_cluster.set_replica_state.assert_not_called()
        chunk_mgr.force_reclaim.assert_not_called()

    def test_running_no_heartbeat_flagged(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Replica phase='running' with running_since set but no heartbeat → marked dead."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running", "running_since": time.time() - 120},
        }
        # No heartbeat entry → _get_last_heartbeat returns None

        watchdog._check_all_jobs()

        cm_cluster.set_replica_state.assert_called_with(job_id, "job-1-r0", phase="dead")
        chunk_mgr.force_reclaim.assert_called_once()


class TestAssemblyTrigger:
    def test_assembly_on_all_done(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """force_reclaim causes all_done=True → assembly callback triggered.

        _is_graceful_completion runs first and calls get_progress — must see
        inflight work (all_done=False) so the replica is treated as dead, not
        gracefully completed.  After force_reclaim, get_progress returns
        all_done=True → assembly fires.
        """
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "running"},
        }
        mc.add_replica(f"{job_id}:job-1-r0", [time.time() - 60])

        # 1st call: _is_graceful_completion — inflight work remains → not graceful
        # 2nd call: after force_reclaim — last chunk failed → all_done
        chunk_mgr.get_progress.side_effect = [
            {"total": 5, "completed": 4, "failed": 0,
             "pending": 0, "inflight": 1, "all_done": False},
            {"total": 5, "completed": 4, "failed": 1,
             "pending": 0, "inflight": 0, "all_done": True},
        ]

        assembly_called = []
        watchdog._assembly_callback = lambda jid: assembly_called.append(jid)

        watchdog._check_all_jobs()

        assert assembly_called == [job_id]


class TestProvisionedPhase:
    def test_provisioned_replica_not_flagged_dead(self, watchdog, mc, cm_cluster, jt, chunk_mgr):
        """Replica in 'provisioned' phase (vLLM loading) → watchdog skips it.
        Previously monitor_replica set phase='running' immediately, causing
        false dead detection during model load."""
        job_id = "job-1"
        jt.jobs[job_id] = FakeJobRecord(status="running")
        cm_cluster.get_replica_states.return_value = {
            "job-1-r0": {"phase": "provisioned"},  # vLLM still loading
        }
        # No heartbeat — hasn't started serving yet

        watchdog._check_all_jobs()

        # Watchdog should skip provisioned replicas entirely
        cm_cluster.set_replica_state.assert_not_called()
        chunk_mgr.force_reclaim.assert_not_called()


class TestPPValidation:
    def test_pp_exceeds_num_nodes_raises(self):
        """PP=2 with num_nodes=1 should be caught before sky.launch, not crash vLLM."""
        import asyncio
        from models.resources import MagicOutput
        from models.requests import BatchedRequest
        from orca_server.launcher import _launch_chunked_replica

        config = MagicOutput(
            decision_id="test-r0", engine="vllm",
            instance_type="g6e.12xlarge",
            tp_size=4, pp_size=2,
            replicas=1, num_instances=1,  # 1 node but PP=2 needs 2
        )
        request = BatchedRequest(
            user_id="test", model_name="test-model",
            input_file="s3://b/in.jsonl", output_file="out.jsonl",
            description="test", task_type="batch", task_priority="normal",
            engine="vllm", slo_mode="cost_first", placement="user_specified",
            avg_output_tokens=1024,
        )
        with pytest.raises(ValueError, match="PP=2 requires 2 nodes but config has 1"):
            asyncio.get_event_loop().run_until_complete(
                _launch_chunked_replica(
                    request, config, "test-r0",
                    parent_job_id="test-job",
                    job_dirname="test",
                )
            )

    def test_pp1_single_node_ok(self):
        """PP=1 with num_nodes=1 should NOT raise (validation passes)."""
        from models.resources import MagicOutput
        config = MagicOutput(
            decision_id="test-r0", engine="vllm",
            instance_type="g6e.12xlarge",
            tp_size=4, pp_size=1,
            replicas=1, num_instances=1,
        )
        # No error expected for PP=1, just verify the condition
        assert not (config.pp_size > 1 and config.num_nodes < config.pp_size)


# ---------------------------------------------------------------------------
# Integration test — watchdog + real Redis chunks
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"chunk_id": "c0000", "s3_input_path": "s3://b/c0000.jsonl", "num_lines": 100},
    {"chunk_id": "c0001", "s3_input_path": "s3://b/c0001.jsonl", "num_lines": 100},
    {"chunk_id": "c0002", "s3_input_path": "s3://b/c0002.jsonl", "num_lines": 100},
]


@pytest.fixture
def real_cm():
    """Real ChunkManager connected to Redis DB 1."""
    from orca_server.chunk_manager import ChunkManager
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


class TestWatchdogRedisIntegration:
    def test_dead_replica_chunks_reclaimed_and_repullable(self, real_cm):
        """End-to-end: dead replica's inflight chunks are force-reclaimed and a new replica pulls them."""
        job_id = f"test-wd-{uuid.uuid4().hex[:8]}"
        real_cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/out")

        # r0 pulls c0000 and c0001, r1 pulls c0002
        real_cm.pull_chunk(job_id, "r0")  # c0000
        real_cm.pull_chunk(job_id, "r0")  # c0001
        real_cm.pull_chunk(job_id, "r1")  # c0002

        progress = real_cm.get_progress(job_id)
        assert progress["inflight"] == 3
        assert progress["pending"] == 0

        # Build a watchdog with real ChunkManager, mocked everything else
        mc = FakeMetricsCollector()
        cm_cluster = MagicMock()
        cm_cluster.get_replica_states.return_value = {
            "r0": {"phase": "running"},
            "r1": {"phase": "running"},
        }
        jt = FakeJobTracker()
        jt.jobs[job_id] = FakeJobRecord(status="running")

        # r0 is dead (stale heartbeat), r1 is alive
        mc.add_replica(f"{job_id}:r0", [time.time() - 120])
        mc.add_replica(f"{job_id}:r1", [time.time() - 2])

        wd = ReplicaWatchdog(
            metrics_collector=mc,
            cluster_manager=cm_cluster,
            job_tracker=jt,
            chunk_manager_fn=lambda: real_cm,
            dead_threshold_sec=45,
            poll_interval_sec=1,
        )

        wd._check_all_jobs()

        # r0 declared dead, its chunks reclaimed
        cm_cluster.set_replica_state.assert_called_once_with(job_id, "r0", phase="dead")

        progress = real_cm.get_progress(job_id)
        assert progress["inflight"] == 1   # only r1's c0002
        assert progress["pending"] == 2    # c0000 + c0001 back in pending

        # A new replica (r2) can now pull the reclaimed chunks
        c = real_cm.pull_chunk(job_id, "r2")
        assert c is not None
        assert c["chunk_id"] in ("c0000", "c0001")
        assert c["replica_id"] == "r2"

        c2 = real_cm.pull_chunk(job_id, "r2")
        assert c2 is not None
        assert c2["chunk_id"] in ("c0000", "c0001")

        # No more pending
        c3 = real_cm.pull_chunk(job_id, "r2")
        assert c3 is None

    def test_all_replicas_dead_chunks_fully_recoverable(self, real_cm):
        """All replicas die → all inflight chunks reclaimed → new replicas can pull all of them."""
        job_id = f"test-wd-all-{uuid.uuid4().hex[:8]}"
        real_cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/out")

        real_cm.pull_chunk(job_id, "r0")  # c0000
        real_cm.pull_chunk(job_id, "r1")  # c0001
        real_cm.pull_chunk(job_id, "r0")  # c0002

        mc = FakeMetricsCollector()
        cm_cluster = MagicMock()
        cm_cluster.get_replica_states.return_value = {
            "r0": {"phase": "running"},
            "r1": {"phase": "running"},
        }
        jt = FakeJobTracker()
        jt.jobs[job_id] = FakeJobRecord(status="running")

        # Both replicas stale
        stale = time.time() - 120
        mc.add_replica(f"{job_id}:r0", [stale])
        mc.add_replica(f"{job_id}:r1", [stale])

        wd = ReplicaWatchdog(
            metrics_collector=mc,
            cluster_manager=cm_cluster,
            job_tracker=jt,
            chunk_manager_fn=lambda: real_cm,
            dead_threshold_sec=45,
            poll_interval_sec=1,
        )

        wd._check_all_jobs()

        progress = real_cm.get_progress(job_id)
        assert progress["inflight"] == 0
        assert progress["pending"] == 3  # all chunks back

        # New replica can pull all 3
        pulled = []
        for _ in range(3):
            c = real_cm.pull_chunk(job_id, "r-new")
            assert c is not None
            pulled.append(c["chunk_id"])
        assert sorted(pulled) == ["c0000", "c0001", "c0002"]
