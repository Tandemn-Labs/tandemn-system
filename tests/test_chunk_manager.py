"""Tests for orca_server.chunk_manager — Redis-backed chunk queue.

Requires a running Redis at REDIS_URL (default localhost:6379).
Use: docker-compose up -d redis && .venv/bin/python -m pytest tests/test_chunk_manager.py -v
"""
import time
import uuid

import pytest
import redis

from orca_server.chunk_manager import ChunkManager

SAMPLE_CHUNKS = [
    {"chunk_id": "c0000", "s3_input_path": "s3://test-bucket/chunks/c0000.jsonl", "num_lines": 1000},
    {"chunk_id": "c0001", "s3_input_path": "s3://test-bucket/chunks/c0001.jsonl", "num_lines": 1000},
    {"chunk_id": "c0002", "s3_input_path": "s3://test-bucket/chunks/c0002.jsonl", "num_lines": 500},
]


@pytest.fixture
def cm():
    """ChunkManager connected to test Redis."""
    url = "redis://localhost:6379/1"  # Use DB 1 for tests
    try:
        r = redis.from_url(url)
        r.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available at localhost:6379")
    manager = ChunkManager(redis_url=url)
    yield manager
    # Cleanup after each test
    r = redis.from_url(url, decode_responses=True)
    for key in r.keys("chunk:job:*"):
        r.delete(key)


@pytest.fixture
def job_id():
    return f"test-{uuid.uuid4().hex[:8]}"


def test_create_job_queue(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    progress = cm.get_progress(job_id)
    assert progress["total"] == 3
    assert progress["pending"] == 3
    assert progress["inflight"] == 0
    assert progress["completed"] == 0
    assert progress["all_done"] is False

    order = cm.get_output_order(job_id)
    assert order == ["c0000", "c0001", "c0002"]


def test_pull_chunk_fifo(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    c0 = cm.pull_chunk(job_id, "replica-0")
    assert c0["chunk_id"] == "c0000"

    c1 = cm.pull_chunk(job_id, "replica-0")
    assert c1["chunk_id"] == "c0001"

    c2 = cm.pull_chunk(job_id, "replica-0")
    assert c2["chunk_id"] == "c0002"

    c3 = cm.pull_chunk(job_id, "replica-0")
    assert c3 is None


def test_pull_chunk_concurrent_replicas(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    c0 = cm.pull_chunk(job_id, "replica-0")
    c1 = cm.pull_chunk(job_id, "replica-1")

    assert c0["chunk_id"] != c1["chunk_id"]
    assert c0["replica_id"] == "replica-0"
    assert c1["replica_id"] == "replica-1"

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 2


def test_complete_chunk_progress(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    cm.pull_chunk(job_id, "replica-0")
    cm.pull_chunk(job_id, "replica-1")
    cm.pull_chunk(job_id, "replica-0")

    result = cm.complete_chunk(job_id, "c0000", "replica-0")
    assert result["completed"] == 1
    assert result["all_done"] is False

    cm.complete_chunk(job_id, "c0001", "replica-1")
    result = cm.complete_chunk(job_id, "c0002", "replica-0")
    assert result["completed"] == 3
    assert result["all_done"] is True


def test_pull_returns_none_when_empty(cm, job_id):
    # Empty queue (never created)
    result = cm.pull_chunk(job_id, "replica-0")
    assert result is None


def test_output_order_preserved(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    # Complete in reverse order
    cm.pull_chunk(job_id, "r0")
    cm.pull_chunk(job_id, "r1")
    cm.pull_chunk(job_id, "r0")
    cm.complete_chunk(job_id, "c0002", "r0")
    cm.complete_chunk(job_id, "c0000", "r0")
    cm.complete_chunk(job_id, "c0001", "r1")

    # Order should still be original
    order = cm.get_output_order(job_id)
    assert order == ["c0000", "c0001", "c0002"]


def test_cleanup_job(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "r0")
    cm.complete_chunk(job_id, "c0000", "r0")

    cm.cleanup_job(job_id)

    assert cm.get_progress(job_id) is None
    assert cm.get_output_order(job_id) == []


def test_get_progress_nonexistent_job(cm):
    result = cm.get_progress("nonexistent-job-id")
    assert result is None


def test_chunk_info_updated_on_pull(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    info = cm.get_chunk_info(job_id, "c0000")
    assert info["status"] == "inflight"
    assert info["replica_id"] == "replica-0"
    assert float(info["started_at"]) > 0


def test_chunk_info_updated_on_complete(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")
    cm.complete_chunk(job_id, "c0000", "replica-0")

    info = cm.get_chunk_info(job_id, "c0000")
    assert info["status"] == "completed"
    assert float(info["completed_at"]) > 0


# ---------------------------------------------------------------------------
# Lease / fault tolerance tests
# ---------------------------------------------------------------------------

def test_pull_chunk_sets_lease(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    before = time.time()
    c = cm.pull_chunk(job_id, "replica-0")
    assert c is not None
    lease_until = float(c["lease_until"])
    assert lease_until > before, "lease_until should be in the future after pull"


def test_reclaim_expired_chunk(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    # Force lease expiry on c0000
    from orca_server.chunk_manager import _chunk_key
    cm._r.hset(_chunk_key(job_id, "c0000"), "lease_until", time.time() - 1)

    result = cm.reclaim_expired_chunks(job_id)
    assert result["reclaimed"] == 1
    assert result["failed"] == 0

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 0
    assert progress["pending"] == 3  # c0001, c0002 still pending + reclaimed c0000


def test_reclaim_respects_valid_lease(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    # Lease is still valid (set far in future)
    from orca_server.chunk_manager import _chunk_key
    cm._r.hset(_chunk_key(job_id, "c0000"), "lease_until", time.time() + 600)

    result = cm.reclaim_expired_chunks(job_id)
    assert result["reclaimed"] == 0
    assert result["failed"] == 0

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 1


def test_reclaim_max_retries(cm, job_id):
    from orca_server.chunk_manager import _chunk_key
    from orca_server.config import CHUNK_MAX_RETRIES

    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    # Set retry_count to max_retries - 1 and expire the lease so one more reclaim hits the limit
    cm._r.hset(_chunk_key(job_id, "c0000"), mapping={
        "lease_until": time.time() - 1,
        "retry_count": CHUNK_MAX_RETRIES - 1,
    })

    result = cm.reclaim_expired_chunks(job_id)
    assert result["failed"] == 1
    assert result["reclaimed"] == 0

    failed_ids = cm.get_failed_chunk_ids(job_id)
    assert "c0000" in failed_ids

    progress = cm.get_progress(job_id)
    assert progress["failed"] == 1
    assert progress["inflight"] == 0


def test_renew_lease_success(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    before = time.time()
    result = cm.renew_lease(job_id, "c0000", "replica-0")
    assert result["renewed"] is True
    assert result["lease_until"] > before

    from orca_server.chunk_manager import _chunk_key
    stored = float(cm._r.hget(_chunk_key(job_id, "c0000"), "lease_until"))
    assert abs(stored - result["lease_until"]) < 1.0


def test_renew_lease_wrong_replica(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    result = cm.renew_lease(job_id, "c0000", "replica-IMPOSTOR")
    assert result["renewed"] is False
    assert "lease_until" not in result


def test_complete_chunk_idempotent(cm, job_id):
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    p1 = cm.complete_chunk(job_id, "c0000", "replica-0")
    p2 = cm.complete_chunk(job_id, "c0000", "replica-0")

    assert p1["completed"] == 1
    assert p2["completed"] == 1  # not 2


def test_all_done_with_failed_chunks(cm, job_id):
    from orca_server.chunk_manager import _chunk_key
    from orca_server.config import CHUNK_MAX_RETRIES

    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")
    cm.pull_chunk(job_id, "replica-1")
    cm.pull_chunk(job_id, "replica-0")

    cm.complete_chunk(job_id, "c0000", "replica-0")
    cm.complete_chunk(job_id, "c0001", "replica-1")

    # Force c0002 to fail via max retries
    cm._r.hset(_chunk_key(job_id, "c0002"), mapping={
        "lease_until": time.time() - 1,
        "retry_count": CHUNK_MAX_RETRIES - 1,
    })
    cm.reclaim_expired_chunks(job_id)

    progress = cm.get_progress(job_id)
    assert progress["completed"] == 2
    assert progress["failed"] == 1
    assert progress["all_done"] is True


def test_complete_promotes_from_failed(cm, job_id):
    """If a chunk is failed (reclaim exhausted retries) but a slow replica actually
    finishes it, complete_chunk should promote it out of the failed set."""
    from orca_server.chunk_manager import _chunk_key
    from orca_server.config import CHUNK_MAX_RETRIES

    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    # Force c0000 into failed set
    cm._r.hset(_chunk_key(job_id, "c0000"), mapping={
        "lease_until": time.time() - 1,
        "retry_count": CHUNK_MAX_RETRIES - 1,
    })
    cm.reclaim_expired_chunks(job_id)
    assert "c0000" in cm.get_failed_chunk_ids(job_id)

    # Slow replica completes the chunk anyway
    progress = cm.complete_chunk(job_id, "c0000", "replica-0")
    assert "c0000" not in cm.get_failed_chunk_ids(job_id)
    assert progress["completed"] == 1
    assert progress["failed"] == 0  # promoted out of failed


# ---------------------------------------------------------------------------
# force_reclaim tests — immediate reclaim by replica ID (ignores lease expiry)
# ---------------------------------------------------------------------------

def test_force_reclaim_specific_replica(cm, job_id):
    """Force-reclaim only chunks owned by a specific replica; others untouched."""
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    # r0 pulls c0000 and c0002, r1 pulls c0001
    cm.pull_chunk(job_id, "replica-0")   # c0000
    cm.pull_chunk(job_id, "replica-1")   # c0001
    cm.pull_chunk(job_id, "replica-0")   # c0002

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 3

    result = cm.force_reclaim(job_id, ["replica-0"])
    assert result["reclaimed"] == 2
    assert result["failed"] == 0

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 1   # only r1's c0001 still inflight
    assert progress["pending"] == 2    # c0000 and c0002 back in pending

    # r1's chunk is untouched
    info = cm.get_chunk_info(job_id, "c0001")
    assert info["status"] == "inflight"
    assert info["replica_id"] == "replica-1"


def test_force_reclaim_no_match(cm, job_id):
    """Force-reclaim with a replica that owns nothing → 0 reclaimed."""
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")

    result = cm.force_reclaim(job_id, ["replica-NONEXISTENT"])
    assert result["reclaimed"] == 0
    assert result["failed"] == 0

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 1  # unchanged


def test_force_reclaim_respects_max_retries(cm, job_id):
    """Chunk at max retries should move to failed set, not pending."""
    from orca_server.chunk_manager import _chunk_key
    from orca_server.config import CHUNK_MAX_RETRIES

    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")  # c0000

    # Set retry_count to max - 1 so the next reclaim (which increments) hits the limit
    cm._r.hset(_chunk_key(job_id, "c0000"), "retry_count", CHUNK_MAX_RETRIES - 1)

    result = cm.force_reclaim(job_id, ["replica-0"])
    assert result["reclaimed"] == 0
    assert result["failed"] == 1

    assert "c0000" in cm.get_failed_chunk_ids(job_id)
    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 0
    assert progress["failed"] == 1


def test_force_reclaim_empty_inflight(cm, job_id):
    """Force-reclaim when nothing is inflight → 0/0."""
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    result = cm.force_reclaim(job_id, ["replica-0"])
    assert result["reclaimed"] == 0
    assert result["failed"] == 0


def test_renew_after_reclaim_returns_false(cm, job_id):
    """Simulate TOCTOU: chunk is reclaimed between ownership check and lease write.
    With the Lua-based renew, this is atomic — renew must return False."""
    from orca_server.chunk_manager import _chunk_key

    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")  # c0000 inflight, owned by replica-0

    # Expire the lease so reclaim can steal it
    cm._r.hset(_chunk_key(job_id, "c0000"), "lease_until", time.time() - 1)

    # Reclaim steals the chunk (moves from inflight → pending)
    result = cm.reclaim_expired_chunks(job_id)
    assert result["reclaimed"] == 1

    # Now replica-0 tries to renew — chunk is no longer inflight or owned by it
    renew_result = cm.renew_lease(job_id, "c0000", "replica-0")
    assert renew_result["renewed"] is False

    # Chunk should still be in pending (not corrupted by a stale renew)
    progress = cm.get_progress(job_id)
    assert progress["pending"] == 3  # c0000 reclaimed + c0001 + c0002


def test_renew_by_wrong_replica_returns_false(cm, job_id):
    """A different replica trying to renew someone else's chunk gets rejected."""
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")
    cm.pull_chunk(job_id, "replica-0")  # c0000 owned by replica-0

    result = cm.renew_lease(job_id, "c0000", "replica-IMPOSTER")
    assert result["renewed"] is False

    # Original owner can still renew
    result = cm.renew_lease(job_id, "c0000", "replica-0")
    assert result["renewed"] is True
    assert result["lease_until"] > time.time()


def test_force_reclaim_multiple_replicas(cm, job_id):
    """Force-reclaim chunks from multiple replica IDs simultaneously."""
    cm.create_job_queue(job_id, SAMPLE_CHUNKS, "test-model", "s3://bucket/output")

    cm.pull_chunk(job_id, "replica-0")   # c0000
    cm.pull_chunk(job_id, "replica-1")   # c0001
    cm.pull_chunk(job_id, "replica-2")   # c0002

    result = cm.force_reclaim(job_id, ["replica-0", "replica-1"])
    assert result["reclaimed"] == 2
    assert result["failed"] == 0

    progress = cm.get_progress(job_id)
    assert progress["inflight"] == 1   # only r2's c0002 still inflight
    assert progress["pending"] == 2    # c0000 and c0001 back in pending
