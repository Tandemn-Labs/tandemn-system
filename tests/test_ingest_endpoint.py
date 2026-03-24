"""Unit tests for POST /job/{job_id}/metrics/ingest"""
import time
from typing import Optional

import pytest
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.testclient import TestClient

from orca_server.metrics_db import MetricsDB
from orca_server.monitoring import MetricsCollector, MetricsSnapshot, _JobCollector


# ---------------------------------------------------------------------------
# Minimal Prometheus text for ingest tests
# ---------------------------------------------------------------------------

SAMPLE_PROM = """\
vllm:avg_generation_throughput_toks_per_s{model_name="m"} 800.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.5
vllm:num_requests_running{model_name="m"} 32
vllm:num_requests_waiting{model_name="m"} 4
vllm:num_requests_swapped{model_name="m"} 0
vllm:request_success_total{model_name="m",finished_reason="stop"} 100.0
vllm:num_preemptions_total{model_name="m"} 1.0
vllm:avg_prompt_throughput_toks_per_s{model_name="m"} 100.0
"""


# ---------------------------------------------------------------------------
# Fixture — minimal FastAPI app that hosts just the ingest endpoint logic
# ---------------------------------------------------------------------------

@pytest.fixture
def ingest_app(tmp_path):
    """
    Creates a minimal test app with the ingest endpoint wired to fresh
    MetricsCollector + MetricsDB instances, plus a configurable API key.
    Returns (TestClient, mc, db, api_key).
    """
    mc = MetricsCollector()
    db = MetricsDB(db_path=str(tmp_path / "test.sqlite"))
    api_key = "test-secret"

    mini = FastAPI()
    mini.state.metrics_collector = mc
    mini.state.metrics_db = db

    @mini.post("/job/{job_id}/metrics/ingest")
    async def ingest(
        job_id: str,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Unauthorized")

        body = await request.json()
        snapshots_raw = body.get("snapshots", [])
        if not snapshots_raw:
            return {"ok": True, "ingested": 0}

        ingested = 0
        batch_for_db = []

        for item in snapshots_raw:
            ts   = item.get("timestamp", time.time())
            text = item.get("prometheus_text", "")
            if not text.strip():
                continue
            snap = MetricsSnapshot.from_prometheus_text(job_id, text, ts)
            with mc._lock:
                jc = mc._jobs.get(job_id)
            if jc:
                with jc.lock:
                    jc.buffer.append(snap)
            batch_for_db.append(snap.to_dict())
            ingested += 1

        if batch_for_db:
            try:
                db.append_timeseries(job_id, batch_for_db)
            except Exception:
                pass

        return {"ok": True, "ingested": ingested}

    return TestClient(mini), mc, db, api_key


@pytest.fixture
def no_auth_app(tmp_path):
    """Same as ingest_app but ORCA_API_KEY is empty — all requests accepted."""
    mc = MetricsCollector()
    db = MetricsDB(db_path=str(tmp_path / "noauth.sqlite"))
    api_key = ""  # empty → auth disabled

    mini = FastAPI()
    mini.state.metrics_collector = mc
    mini.state.metrics_db = db

    @mini.post("/job/{job_id}/metrics/ingest")
    async def ingest(
        job_id: str,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Unauthorized")
        body = await request.json()
        snapshots_raw = body.get("snapshots", [])
        batch_for_db = []
        ingested = 0
        for item in snapshots_raw:
            ts = item.get("timestamp", time.time())
            text = item.get("prometheus_text", "")
            if not text.strip():
                continue
            snap = MetricsSnapshot.from_prometheus_text(job_id, text, ts)
            with mc._lock:
                jc = mc._jobs.get(job_id)
            if jc:
                with jc.lock:
                    jc.buffer.append(snap)
            batch_for_db.append(snap.to_dict())
            ingested += 1
        if batch_for_db:
            db.append_timeseries(job_id, batch_for_db)
        return {"ok": True, "ingested": ingested}

    return TestClient(mini), mc, db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestAuth:
    def test_auth_rejected_with_wrong_bearer(self, ingest_app):
        client, mc, db, api_key = ingest_app
        resp = client.post(
            "/job/job-1/metrics/ingest",
            json={"snapshots": [{"timestamp": 1.0, "prometheus_text": SAMPLE_PROM}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_auth_rejected_with_no_header(self, ingest_app):
        client, mc, db, api_key = ingest_app
        resp = client.post(
            "/job/job-1/metrics/ingest",
            json={"snapshots": [{"timestamp": 1.0, "prometheus_text": SAMPLE_PROM}]},
        )
        assert resp.status_code == 401

    def test_auth_accepted_with_correct_bearer(self, ingest_app):
        client, mc, db, api_key = ingest_app
        resp = client.post(
            "/job/job-ok/metrics/ingest",
            json={"snapshots": [{"timestamp": 1.0, "prometheus_text": SAMPLE_PROM}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_no_auth_when_key_empty(self, no_auth_app):
        client, mc, db = no_auth_app
        resp = client.post(
            "/job/job-open/metrics/ingest",
            json={"snapshots": [{"timestamp": 1.0, "prometheus_text": SAMPLE_PROM}]},
        )
        assert resp.status_code == 200


class TestIngestRingBuffer:
    def test_populates_ring_buffer_for_registered_job(self, ingest_app):
        """Snapshots are added to the ring buffer when the job is registered."""
        client, mc, db, api_key = ingest_app
        job_id = "job-buf-1"

        # Register the job so ingest can find the _JobCollector
        mc.start_collecting(job_id)  # ingest-only mode
        ts_base = time.time()
        snapshots = [
            {"timestamp": ts_base + i, "prometheus_text": SAMPLE_PROM}
            for i in range(3)
        ]
        resp = client.post(
            f"/job/{job_id}/metrics/ingest",
            json={"snapshots": snapshots},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.json()["ingested"] == 3

        snap = mc.get_latest(job_id)
        assert snap is not None
        # Throughput is computed from ring buffer window, not per-snapshot.
        # With identical Prometheus text across snapshots, counter delta is 0.
        # Verify the snapshot was ingested with correct gauge/counter values.
        assert snap.gpu_cache_usage_perc == pytest.approx(0.5)
        assert snap.num_requests_running == 32

    def test_unknown_job_still_persists_to_db(self, ingest_app):
        """Jobs not registered in MetricsCollector still get DB writes (no crash)."""
        client, mc, db, api_key = ingest_app
        job_id = "job-unknown"
        resp = client.post(
            f"/job/{job_id}/metrics/ingest",
            json={"snapshots": [{"timestamp": 1.0, "prometheus_text": SAMPLE_PROM}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 1


class TestIngestTimeseries:
    def test_writes_timeseries_to_db(self, ingest_app):
        """3 snapshots → 3 rows in the timeseries table."""
        client, mc, db, api_key = ingest_app
        job_id = "job-ts-1"
        ts_base = time.time()
        snapshots = [
            {"timestamp": ts_base + i, "prometheus_text": SAMPLE_PROM}
            for i in range(3)
        ]
        resp = client.post(
            f"/job/{job_id}/metrics/ingest",
            json={"snapshots": snapshots},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.json()["ingested"] == 3

        rows = db.get_timeseries(job_id)
        assert len(rows) == 3

    def test_ingested_count_returned(self, ingest_app):
        client, mc, db, api_key = ingest_app
        snapshots = [
            {"timestamp": float(i), "prometheus_text": SAMPLE_PROM}
            for i in range(5)
        ]
        resp = client.post(
            "/job/job-count/metrics/ingest",
            json={"snapshots": snapshots},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.json() == {"ok": True, "ingested": 5}


class TestIngestEdgeCases:
    def test_empty_snapshots_returns_zero(self, ingest_app):
        client, mc, db, api_key = ingest_app
        resp = client.post(
            "/job/job-empty/metrics/ingest",
            json={"snapshots": []},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "ingested": 0}

    def test_blank_prometheus_text_skipped(self, ingest_app):
        client, mc, db, api_key = ingest_app
        resp = client.post(
            "/job/job-blank/metrics/ingest",
            json={"snapshots": [{"timestamp": 1.0, "prometheus_text": "   "}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert resp.json()["ingested"] == 0
