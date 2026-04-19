"""Tests for Orca's /health endpoint — outbox visibility.

Phase 4d of contract-hardening. Exposes:
  - outbox_enabled: whether the durable queue is up
  - outbox_pending: rows awaiting delivery
  - outbox_oldest_undelivered_age_secs: Koi-unreachability duration

A stuck outbox past ~60s is the signal an operator needs to see during
an E2E run. Without this, the CLI poll loop has no way to tell "Koi is
processing slowly" from "Koi has been down for 10 minutes."
"""

import time

import pytest
from fastapi.testclient import TestClient

from orca_server import outbox as ox_mod
from server import app


@pytest.fixture
def client_with_outbox(monkeypatch):
    """Attach a fresh in-memory OutboxDB as the singleton and yield a
    TestClient that bypasses the real lifespan (we don't want AWS /
    Redis / SkyPilot init during unit tests)."""
    # Minimal app.state so the endpoint body works.
    app.state.redis_available = False

    ox_mod._OUTBOX = ox_mod.OutboxDB(":memory:")
    client = TestClient(app)
    try:
        yield client, ox_mod._OUTBOX
    finally:
        ox_mod._OUTBOX.close()
        ox_mod._OUTBOX = None


class TestHealthWithOutbox:
    def test_reports_enabled_when_outbox_attached(self, client_with_outbox):
        client, _ = client_with_outbox
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["outbox_enabled"] is True
        assert body["outbox_pending"] == 0
        assert body["outbox_oldest_undelivered_age_secs"] == 0.0

    def test_reports_pending_count(self, client_with_outbox):
        client, outbox = client_with_outbox
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="a")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="b")
        r = client.get("/health")
        body = r.json()
        assert body["outbox_pending"] == 2

    def test_oldest_age_tracks_real_time(self, client_with_outbox):
        client, outbox = client_with_outbox
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="old")
        time.sleep(0.1)
        r = client.get("/health")
        body = r.json()
        assert body["outbox_oldest_undelivered_age_secs"] >= 0.1

    def test_delivered_row_drops_pending(self, client_with_outbox):
        client, outbox = client_with_outbox
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="a")
        outbox.mark_delivered("a")
        r = client.get("/health")
        body = r.json()
        assert body["outbox_pending"] == 0


class TestHealthWithoutOutbox:
    def test_reports_disabled_when_outbox_none(self):
        app.state.redis_available = False
        ox_mod._OUTBOX = None
        client = TestClient(app)
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert body["outbox_enabled"] is False
        assert "outbox_pending" not in body


class TestHealthReflectsRedis:
    def test_redis_flag_surfaced(self):
        ox_mod._OUTBOX = None
        app.state.redis_available = True
        client = TestClient(app)
        r = client.get("/health")
        body = r.json()
        assert body["redis_available"] is True

        app.state.redis_available = False
        r = client.get("/health")
        assert r.json()["redis_available"] is False
