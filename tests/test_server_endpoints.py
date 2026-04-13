"""Integration tests for FastAPI server endpoints.

Uses httpx AsyncClient against the real app (no external services called).
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from server import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---- /quota/status ----

@pytest.mark.asyncio
async def test_quota_status_returns_success(client):
    resp = await client.get("/quota/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "quota_usage" in data
    assert "active_reservations" in data
    assert isinstance(data["quota_usage"], list)
    assert isinstance(data["active_reservations"], list)


# ---- /jobs ----

@pytest.mark.asyncio
async def test_list_jobs_empty(client):
    resp = await client.get("/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


# ---- /job/{job_id} ----

@pytest.mark.asyncio
async def test_get_job_not_found(client):
    resp = await client.get("/job/nonexistent-job-id")
    assert resp.status_code == 404


# ---- /test/placement (roofline solver, no cloud calls) ----

@pytest.mark.asyncio
async def test_placement_roofline(client):
    payload = {
        "user_id": "test-user",
        "input_file": "examples/workloads/demo_batch.jsonl",
        "output_file": "output.jsonl",
        "avg_output_tokens": 512,
        "description": "test",
        "task_type": "batch",
        "task_priority": "low",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "engine": "vllm",
        "slo_mode": "throughput",
        "placement": "auto",
        "placement_solver": "roofline",
    }
    resp = await client.post("/test/placement", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["solver"] == "roofline"
    assert "placements" in data
    assert len(data["placements"]) >= 1
    p = data["placements"][0]
    assert "instance_type" in p
    assert "tp_size" in p


# ---- /test/placement (user_specified) ----

@pytest.mark.asyncio
async def test_placement_user_specified(client):
    payload = {
        "user_id": "test-user",
        "input_file": "examples/workloads/demo_batch.jsonl",
        "output_file": "output.jsonl",
        "avg_output_tokens": 512,
        "description": "test",
        "task_type": "batch",
        "task_priority": "low",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "engine": "vllm",
        "slo_mode": "throughput",
        "placement": "auto",
        "placement_solver": "user_specified",
        "gpu_type": "L40S",
        "tp_size": 4,
        "pp_size": 1,
    }
    resp = await client.post("/test/placement", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["solver"] == "user_specified"
    p = data["placements"][0]
    assert p["gpu_type"] == "L40S"
    assert p["tp_size"] == 4
    assert "feasible" in p


# ---- /test/placement (llm solver) ----

@pytest.mark.asyncio
async def test_placement_llm_solver(client):
    payload = {
        "user_id": "test-user",
        "input_file": "examples/workloads/demo_batch.jsonl",
        "output_file": "output.jsonl",
        "avg_output_tokens": 512,
        "description": "test",
        "task_type": "batch",
        "task_priority": "low",
        "model_name": "meta-llama/Llama-3-8B",
        "engine": "vllm",
        "slo_mode": "throughput",
        "placement": "auto",
        "placement_solver": "llm",
    }
    resp = await client.post("/test/placement", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["solver"] == "llm"
    assert "placements" in data
    assert len(data["placements"]) >= 1


# ---- /submit/batch (invalid solver) ----

@pytest.mark.asyncio
async def test_submit_batch_invalid_solver(client):
    payload = {
        "user_id": "test-user",
        "input_file": "examples/workloads/demo_batch.jsonl",
        "output_file": "output.jsonl",
        "avg_output_tokens": 512,
        "description": "test",
        "task_type": "batch",
        "task_priority": "low",
        "model_name": "meta-llama/Llama-3-8B",
        "engine": "vllm",
        "slo_mode": "throughput",
        "placement": "auto",
        "placement_solver": "llm",
    }
    resp = await client.post("/submit/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "invalid_solver" in data["error_type"]


# ---- /submit/batch (user_specified, invalid GPU) ----

@pytest.mark.asyncio
async def test_submit_batch_invalid_gpu(client):
    payload = {
        "user_id": "test-user",
        "input_file": "examples/workloads/demo_batch.jsonl",
        "output_file": "output.jsonl",
        "avg_output_tokens": 512,
        "description": "test",
        "task_type": "batch",
        "task_priority": "low",
        "model_name": "meta-llama/Llama-3-8B",
        "engine": "vllm",
        "slo_mode": "throughput",
        "placement": "auto",
        "placement_solver": "user_specified",
        "gpu_type": "NONEXISTENT_GPU",
        "tp_size": 1,
    }
    resp = await client.post("/submit/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert data["error_type"] == "invalid_placement"
