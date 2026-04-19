"""Tests for OutboxPublisher — the thread that drains the outbox.

Phase 4b of contract-hardening. Uses an injected `post_fn` seam instead
of real HTTP so tests are deterministic and fast.

Invariants proven:
  - 2xx response → row marked delivered, no retry.
  - non-2xx response → row stays undelivered, attempts incremented,
    next_attempt_at scheduled by mark_failure's backoff.
  - Network exception (connection refused, timeout) → same: row stays,
    retryable after backoff.
  - Stop is graceful: thread exits within the timeout, no lingering
    mid-flight work.
  - Prune is throttled: runs at most once per prune_interval.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from orca_server.outbox import OutboxDB, OutboxPublisher


class _FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


def _make_post_tracker(response: _FakeResponse):
    calls = []

    def post(url, json_payload, timeout):
        calls.append((url, json_payload, timeout))
        if isinstance(response, Exception):
            raise response
        return response

    return post, calls


@pytest.fixture
def outbox():
    return OutboxDB(":memory:")


@pytest.fixture
def publisher_with_mock(outbox):
    """Returns (publisher, calls_log) using a mutable response holder."""
    holder = SimpleNamespace(response=_FakeResponse(200))

    def post(url, payload, timeout):
        holder.last_call = (url, payload, timeout)
        resp = holder.response
        if isinstance(resp, Exception):
            raise resp
        return resp

    pub = OutboxPublisher(
        outbox,
        koi_base_url="http://koi.test",
        post_fn=post,
        poll_interval=0.05,
        prune_interval=3600,  # don't prune during tests unless asked
    )
    return pub, holder


class TestDrainSingleCycle:
    def test_success_marks_delivered(self, publisher_with_mock, outbox):
        pub, holder = publisher_with_mock
        outbox.enqueue("/job/started", "job_started", {"x": 1}, job_id="r0", dedup_key="a")
        delivered = pub.drain_once()
        assert delivered == 1
        assert outbox.pending_count() == 0

    def test_uses_full_url(self, publisher_with_mock, outbox):
        pub, holder = publisher_with_mock
        outbox.enqueue("/job/complete", "job_complete", {"x": 1}, job_id="g", dedup_key="b")
        pub.drain_once()
        assert holder.last_call[0] == "http://koi.test/job/complete"

    def test_sends_full_envelope(self, publisher_with_mock, outbox):
        pub, holder = publisher_with_mock
        outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"replica_id": "r0"},
            job_id="r0",
            dedup_key="replica_failed:r0",
            correlation_id="sr-9",
        )
        pub.drain_once()
        sent = holder.last_call[1]
        assert sent["event_id"] == "replica_failed:r0"
        assert sent["event_type"] == "replica_failed"
        assert sent["correlation_id"] == "sr-9"
        assert sent["replica_id"] == "r0"

    def test_empty_outbox_is_noop(self, publisher_with_mock):
        pub, _ = publisher_with_mock
        assert pub.drain_once() == 0


class TestRetryOnFailure:
    def test_5xx_keeps_row_and_bumps_attempts(self, publisher_with_mock, outbox):
        pub, holder = publisher_with_mock
        holder.response = _FakeResponse(503, text="service unavailable")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        pub.drain_once()
        assert outbox.pending_count() == 1  # row still there

        # mark_failure scheduled next_attempt_at ~1s out → not yet due
        assert outbox.claim_batch(now=time.time()) == []

    def test_4xx_also_treated_as_failure(self, publisher_with_mock, outbox):
        pub, holder = publisher_with_mock
        holder.response = _FakeResponse(422, text="bad payload")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        pub.drain_once()
        assert outbox.pending_count() == 1

    def test_network_exception_treated_as_failure(self, publisher_with_mock, outbox):
        pub, holder = publisher_with_mock
        holder.response = ConnectionError("connection refused")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        pub.drain_once()
        assert outbox.pending_count() == 1

    def test_retry_eventually_delivers(self, outbox):
        """Fail, then succeed on retry. The first drain raises → mark_failure
        schedules a backoff. Manually move next_attempt_at forward and the
        next drain succeeds → mark_delivered."""
        calls = {"n": 0}

        def post(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("still down")
            return _FakeResponse(200)

        pub = OutboxPublisher(
            outbox,
            koi_base_url="http://koi.test",
            post_fn=post,
            poll_interval=0.01,
        )
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")

        # First attempt: exception → stays pending with backoff.
        pub.drain_once()
        assert outbox.pending_count() == 1

        # Force back into the claim window; retry succeeds.
        with outbox._lock:
            outbox._conn.execute(
                "UPDATE outbox SET next_attempt_at = ? WHERE event_id = 'k'",
                (time.time(),),
            )
            outbox._conn.commit()
        pub.drain_once()
        assert outbox.pending_count() == 0
        assert calls["n"] == 2


class TestBatchBoundary:
    def test_respects_batch_size(self, outbox):
        seen = []

        def post(url, payload, timeout):
            seen.append(payload["event_id"])
            return _FakeResponse(200)

        pub = OutboxPublisher(
            outbox,
            koi_base_url="http://koi.test",
            post_fn=post,
            batch_size=3,
        )
        for i in range(10):
            outbox.enqueue("/x", "t", {}, job_id="j", dedup_key=f"k{i}")
        delivered = pub.drain_once()
        assert delivered == 3
        assert len(seen) == 3
        assert outbox.pending_count() == 7


class TestBackgroundThreadLifecycle:
    def test_thread_drains_on_its_own(self, outbox):
        def post(url, payload, timeout):
            return _FakeResponse(200)

        pub = OutboxPublisher(
            outbox,
            koi_base_url="http://koi.test",
            post_fn=post,
            poll_interval=0.02,
        )
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="a")
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="b")
        pub.start()
        deadline = time.time() + 2.0
        try:
            while outbox.pending_count() > 0 and time.time() < deadline:
                time.sleep(0.05)
            assert outbox.pending_count() == 0
        finally:
            pub.stop(timeout=1.0)

    def test_stop_is_graceful(self, outbox):
        def post(url, payload, timeout):
            return _FakeResponse(200)

        pub = OutboxPublisher(
            outbox, koi_base_url="http://koi.test", post_fn=post, poll_interval=0.02
        )
        pub.start()
        time.sleep(0.05)
        pub.stop(timeout=1.0)
        # Thread reference cleared, no zombies.
        assert pub._thread is None

    def test_start_is_idempotent(self, outbox):
        def post(url, payload, timeout):
            return _FakeResponse(200)

        pub = OutboxPublisher(
            outbox, koi_base_url="http://koi.test", post_fn=post, poll_interval=0.05
        )
        pub.start()
        t1 = pub._thread
        pub.start()
        t2 = pub._thread
        try:
            assert t1 is t2  # same thread, not a second one
        finally:
            pub.stop(timeout=1.0)


class TestPruneThrottle:
    def test_prune_runs_when_interval_elapses(self, outbox):
        def post(url, payload, timeout):
            return _FakeResponse(200)

        pub = OutboxPublisher(
            outbox,
            koi_base_url="http://koi.test",
            post_fn=post,
            poll_interval=0.01,
            prune_interval=0.05,  # tiny, so it actually prunes
            prune_keep_secs=0.0,  # "delete anything delivered"
        )
        outbox.enqueue("/x", "t", {}, job_id="j", dedup_key="k")
        pub.drain_once()  # marks delivered
        # Force _last_prune into the past so _maybe_prune fires
        pub._last_prune = 0.0
        pub._maybe_prune()
        # The row has been deleted.
        with outbox._lock:
            rows = outbox._conn.execute("SELECT * FROM outbox").fetchall()
        assert rows == []


class TestInitAttachesPublisher:
    def test_init_starts_publisher_when_koi_url_set(self, tmp_path, monkeypatch):
        from orca_server import outbox as ox_mod

        monkeypatch.setenv("KOI_SERVICE_URL", "http://koi.test")
        ox_mod.init_outbox(str(tmp_path / "ob.db"))
        try:
            assert ox_mod.get_publisher() is not None
        finally:
            ox_mod.shutdown_outbox()

    def test_init_skips_publisher_when_koi_url_missing(self, tmp_path, monkeypatch):
        from orca_server import outbox as ox_mod

        monkeypatch.delenv("KOI_SERVICE_URL", raising=False)
        ox_mod.init_outbox(str(tmp_path / "ob.db"))
        try:
            # OutboxDB exists, publisher does not.
            assert ox_mod.get_outbox() is not None
            assert ox_mod.get_publisher() is None
        finally:
            ox_mod.shutdown_outbox()
