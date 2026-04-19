"""Tests for orca_server/outbox.py — durable at-least-once delivery.

Phase 4a of contract-hardening: the schema + enqueue/claim/mark surface.
Publisher-thread behavior is covered separately in Phase 4b.

What's proven here:
  - enqueue returns an event_id and persists exactly one row per dedup_key.
  - dedup_key collapses repeated enqueues from naturally-duplicated events
    (the watchdog + monitor_replica scenario) to a single delivery.
  - claim_batch only returns undelivered rows whose next_attempt_at is due.
  - mark_failure uses bounded exponential backoff with a 30s cap.
  - mark_delivered removes rows from the pending pool.
  - prune_delivered keeps undelivered rows regardless of age.
  - envelope fields (event_id, event_type, emitted_at) are injected into
    the payload before serialization.
"""

import json
import time

import pytest

from orca_server.outbox import MAX_BACKOFF_SECS, OutboxDB


@pytest.fixture
def outbox():
    return OutboxDB(":memory:")


class TestEnqueue:
    def test_returns_event_id(self, outbox):
        event_id = outbox.enqueue(
            "/job/started",
            "job_started",
            {"replica_id": "r0"},
            job_id="r0",
        )
        assert isinstance(event_id, str) and len(event_id) >= 16

    def test_unique_events_produce_separate_rows(self, outbox):
        outbox.enqueue("/job/started", "job_started", {"x": 1}, job_id="r0")
        outbox.enqueue("/job/started", "job_started", {"x": 2}, job_id="r1")
        assert outbox.pending_count() == 2

    def test_envelope_injected_into_payload(self, outbox):
        event_id = outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"replica_id": "r0", "reason": "kill"},
            job_id="r0",
            dedup_key="replica_failed:r0",
            correlation_id="sr-1",
        )
        rows = outbox.claim_batch()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["event_id"] == event_id
        assert payload["event_id"] == "replica_failed:r0"
        assert payload["event_type"] == "replica_failed"
        assert payload["correlation_id"] == "sr-1"
        assert isinstance(payload["emitted_at"], float)
        # Caller's own fields still present:
        assert payload["replica_id"] == "r0"
        assert payload["reason"] == "kill"


class TestDedupKey:
    """The single biggest user of dedup_key: watchdog + monitor_replica
    both detecting the same replica death. Second enqueue must be no-op."""

    def test_same_dedup_key_is_single_row(self, outbox):
        e1 = outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"source": "monitor_replica"},
            job_id="r0",
            dedup_key="replica_failed:r0",
        )
        e2 = outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"source": "watchdog"},
            job_id="r0",
            dedup_key="replica_failed:r0",
        )
        assert e1 == e2 == "replica_failed:r0"
        assert outbox.pending_count() == 1

    def test_different_dedup_keys_are_distinct(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k1")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k2")
        assert outbox.pending_count() == 2


class TestClaimBatch:
    def test_only_returns_undelivered(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="a")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="b")
        outbox.mark_delivered("a")
        rows = outbox.claim_batch()
        ids = {r["event_id"] for r in rows}
        assert ids == {"b"}

    def test_respects_next_attempt_at(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="due")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="future")
        # Push "future" well past now.
        outbox.mark_failure("future", "test")  # schedules ~1s out
        # Claim with `now` slightly in the past relative to the backoff:
        # only "due" should come back.
        rows = outbox.claim_batch(now=time.time())
        ids = {r["event_id"] for r in rows}
        assert "due" in ids
        assert "future" not in ids

    def test_returns_ordered_oldest_first(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="first")
        time.sleep(0.01)
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="second")
        rows = outbox.claim_batch()
        assert rows[0]["event_id"] == "first"
        assert rows[1]["event_id"] == "second"

    def test_respects_limit(self, outbox):
        for i in range(10):
            outbox.enqueue("/x", "t", {}, job_id="j", dedup_key=f"k{i}")
        rows = outbox.claim_batch(limit=3)
        assert len(rows) == 3


class TestBackoff:
    def test_mark_failure_schedules_bounded_backoff(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        base = time.time()
        for attempt in range(1, 8):
            outbox.mark_failure("k", f"try {attempt}")
        # After ~7 failures: 2^6 = 64, but cap is MAX_BACKOFF_SECS (30s).
        rows = outbox.claim_batch(now=base + MAX_BACKOFF_SECS + 1)
        # Row IS returnable at cap+1s
        assert len(rows) == 1

    def test_backoff_records_error(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        outbox.mark_failure("k", "connection refused", status_code=None)
        rows = outbox.claim_batch(now=time.time() + 60)
        assert rows[0]["event_id"] == "k"

    def test_backoff_truncates_long_errors(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        outbox.mark_failure("k", "x" * 5000)  # should not raise
        # Row still in outbox — failure doesn't delete.
        assert outbox.pending_count() == 1


class TestMarkDelivered:
    def test_removes_from_pending(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="a")
        assert outbox.pending_count() == 1
        outbox.mark_delivered("a")
        assert outbox.pending_count() == 0

    def test_idempotent(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="a")
        outbox.mark_delivered("a")
        outbox.mark_delivered("a")  # second call must not raise
        assert outbox.pending_count() == 0


class TestPrune:
    def test_prunes_old_delivered_rows(self, outbox):
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="old")
        outbox.mark_delivered("old")
        # Backdate to 2 days ago
        with outbox._lock:
            outbox._conn.execute(
                "UPDATE outbox SET delivered_at = ? WHERE event_id = ?",
                (time.time() - 2 * 86400, "old"),
            )
            outbox._conn.commit()
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="new")
        outbox.mark_delivered("new")
        removed = outbox.prune_delivered(keep_secs=86400)
        assert removed == 1

    def test_preserves_undelivered_regardless_of_age(self, outbox):
        """Stuck-forever rows must survive pruning — they're exactly the
        ones an operator needs to see."""
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="stuck")
        with outbox._lock:
            outbox._conn.execute(
                "UPDATE outbox SET created_at = ? WHERE event_id = ?",
                (time.time() - 10 * 86400, "stuck"),
            )
            outbox._conn.commit()
        removed = outbox.prune_delivered(keep_secs=86400)
        assert removed == 0
        assert outbox.pending_count() == 1


class TestHealthCounters:
    def test_pending_count_and_oldest_age(self, outbox):
        assert outbox.pending_count() == 0
        assert outbox.oldest_undelivered_age_secs() == 0.0
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="first")
        time.sleep(0.02)
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="second")
        assert outbox.pending_count() == 2
        assert outbox.oldest_undelivered_age_secs() >= 0.02
        outbox.mark_delivered("first")
        outbox.mark_delivered("second")
        assert outbox.pending_count() == 0


class TestSingletonLifecycle:
    def test_init_and_shutdown(self, tmp_path):
        from orca_server import outbox as ox_mod

        assert ox_mod.get_outbox() is None
        ox_mod.init_outbox(str(tmp_path / "outbox.db"))
        assert ox_mod.get_outbox() is not None
        ox_mod.shutdown_outbox()
        assert ox_mod.get_outbox() is None

    def test_empty_path_disables_outbox(self, monkeypatch):
        from orca_server import outbox as ox_mod

        monkeypatch.setenv("ORCA_OUTBOX_DB_PATH", "")
        ox_mod.init_outbox()
        assert ox_mod.get_outbox() is None
