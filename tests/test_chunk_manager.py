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
