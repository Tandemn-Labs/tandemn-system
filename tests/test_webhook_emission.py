"""Tests for the webhook call-site migration.

Phase 4c of contract-hardening. Before this phase, every Koi webhook was
a direct `requests.post(...)` — lost on Koi restart, doubled by
watchdog+monitor_replica, no trace in Orca. After this phase, every call
site routes through `_post_koi_webhook` which enqueues into the durable
outbox.

The two invariants here:
  1. An enqueued event is physically present in the outbox as a single row
     with the expected envelope + dedup_key.
  2. Two enqueues with the same dedup_key collapse at source — this is
     the watchdog + monitor_replica duplicate-fire case, reduced to a
     single delivered webhook via INSERT OR IGNORE on the PK.

Publisher behavior is covered separately in test_outbox_publisher.py.
"""

import json

import pytest

from orca_server import outbox as ox_mod
from orca_server.launcher import _post_koi_webhook


@pytest.fixture
def outbox(monkeypatch):
    """Install a fresh in-memory OutboxDB as the process singleton.

    KOI_SERVICE_URL is a module-level constant frozen at import time, so
    we patch the attribute directly rather than via env var.
    """
    from orca_server import config as _cfg

    monkeypatch.setattr(_cfg, "KOI_SERVICE_URL", "http://koi.test", raising=False)
    ox_mod._OUTBOX = ox_mod.OutboxDB(":memory:")
    try:
        yield ox_mod._OUTBOX
    finally:
        ox_mod._OUTBOX.close()
        ox_mod._OUTBOX = None


class TestEnqueueLandsInOutbox:
    def test_replica_failed_single_row(self, outbox):
        _post_koi_webhook(
            "/job/replica-failed",
            {"job_id": "r0", "group_id": "g0", "reason": "test"},
            "replica-failed",
            dedup_key="replica_failed:r0",
        )
        assert outbox.pending_count() == 1

    def test_envelope_injected(self, outbox):
        _post_koi_webhook(
            "/job/started",
            {"job_id": "r0", "replica_id": "r0", "tp": 4},
            "job-started",
            dedup_key="job_started:r0",
        )
        rows = outbox.claim_batch()
        payload = json.loads(rows[0]["payload"])
        assert payload["event_id"] == "job_started:r0"
        assert payload["event_type"] == "job_started"  # normalized from 'job-started'
        assert payload["tp"] == 4

    def test_job_id_resolved_from_replica_id(self, outbox):
        _post_koi_webhook(
            "/job/replica-failed",
            {"replica_id": "r0", "group_id": "g0"},
            "replica-failed",
            dedup_key="replica_failed:r0",
        )
        rows = outbox.claim_batch()
        assert rows[0]["job_id"] == "r0"


class TestWatchdogMonitorCollision:
    """The headline: watchdog and monitor_replica both detect the same
    replica death and both call _post_koi_webhook with the same dedup_key.
    Result must be a single row in the outbox (single delivery to Koi)."""

    def test_same_dedup_key_collapses_to_one_row(self, outbox):
        # First emitter: monitor_replica
        _post_koi_webhook(
            "/job/replica-failed",
            {"replica_id": "r0", "reason": "clean exit, chunks pending"},
            "replica-failed",
            dedup_key="replica_failed:r0",
        )
        # Second emitter: watchdog, 30s later
        _post_koi_webhook(
            "/job/replica-failed",
            {"replica_id": "r0", "reason": "heartbeat timeout"},
            "replica-failed",
            dedup_key="replica_failed:r0",
        )
        assert outbox.pending_count() == 1

    def test_different_replicas_are_distinct(self, outbox):
        _post_koi_webhook(
            "/job/replica-failed",
            {"replica_id": "r0"},
            "replica-failed",
            dedup_key="replica_failed:r0",
        )
        _post_koi_webhook(
            "/job/replica-failed",
            {"replica_id": "r1"},
            "replica-failed",
            dedup_key="replica_failed:r1",
        )
        assert outbox.pending_count() == 2


class TestDisabledOutboxFallback:
    """If the outbox singleton is absent (opt-out via empty DB path, or
    startup hasn't initialized it), _post_koi_webhook must not crash —
    it falls back to best-effort direct POST just like before Phase 4."""

    def test_no_outbox_no_crash(self, monkeypatch):
        """Without KOI_SERVICE_URL, _post_koi_webhook is a no-op and
        definitely must not raise."""
        from orca_server import config as _cfg

        monkeypatch.setattr(_cfg, "KOI_SERVICE_URL", "", raising=False)
        # Ensure the singleton is clear.
        ox_mod._OUTBOX = None
        _post_koi_webhook("/job/replica-failed", {"replica_id": "r0"}, "replica-failed")
        # Nothing to assert — just that it returned cleanly.


class TestEventTypeNormalization:
    """_post_koi_webhook takes human-readable `event` tags like
    'replica-failed'. The outbox envelope uses snake_case event_type to
    match the shared contract in koi_contract.py."""

    def test_dashes_become_underscores(self, outbox):
        _post_koi_webhook(
            "/job/launch-heartbeat",
            {"job_id": "r0", "phase": "provisioning"},
            "launch-heartbeat",
        )
        rows = outbox.claim_batch()
        assert json.loads(rows[0]["payload"])["event_type"] == "launch_heartbeat"
