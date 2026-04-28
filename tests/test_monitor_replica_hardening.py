"""Tests for monitor_replica guarantee-fire + terminal-phase normalization.

Phase 5 of contract-hardening. Before this phase, three paths could exit
monitor_replica without emitting /job/replica-failed, leaving Koi blind:

  1. New / unforeseen exit path (programmer didn't write a branch)
  2. Phase set to something non-terminal that fell through the explicit
     except-branches
  3. "dead" phase was already set by watchdog — fallback used to re-emit
     because the old terminal check `(completed, killed, swapped_out)`
     didn't include "dead"

After this phase:
  - `_emit_replica_failed(reason_code, detail)` is the single emit helper;
    every explicit path uses it and sets the `replica_failed_enqueued`
    flag.
  - The finally block's fallback checks that flag AND reads the current
    phase from `cm.get_replica_states(parent_job_id).get(replica_id, {})`
    (the real API — not the nonexistent `get_replica_phase`).
  - "dead" is in TERMINAL_PHASES, so watchdog-handled deaths don't get
    a redundant fallback emit.
  - Dedup via `replica_failed:{replica_id}` is the ultimate backstop;
    even if logic is wrong, the outbox collapses duplicates at source.

These tests exercise the constants and helper in isolation. The wrapped
monitor_replica closure can't be unit-tested without mocking sky.tail_logs
and a full JobTracker — integration coverage is in Phase 6.
"""

import pytest

from orca_server.koi_contract import ReasonCode, TERMINAL_PHASES


class TestTerminalPhases:
    """The blocker GPT caught: 'dead' must be in the terminal set so
    watchdog-handled deaths don't trigger redundant fallback emits."""

    def test_dead_is_terminal(self):
        assert "dead" in TERMINAL_PHASES

    def test_all_expected_phases_present(self):
        # These match launcher.py:981 and the previous hardcoded sets.
        for phase in ("completed", "failed", "dead", "killed", "swapped_out"):
            assert phase in TERMINAL_PHASES, f"{phase!r} missing from TERMINAL_PHASES"

    def test_non_terminal_phases_excluded(self):
        # Sanity — these must NOT be treated as terminal.
        for phase in ("launching", "provisioned", "running", "model_ready", "loading_model"):
            assert phase not in TERMINAL_PHASES, f"{phase!r} should not be terminal"

    def test_is_frozenset(self):
        # frozenset prevents downstream code from mutating the shared contract.
        assert isinstance(TERMINAL_PHASES, frozenset)


class TestReasonCodesAvailable:
    """Phase 5 added structured reason codes to replica-failed emissions."""

    def test_monitor_thread_exited_exists(self):
        assert ReasonCode.MONITOR_THREAD_EXITED.value == "monitor_thread_exited"

    def test_log_stream_error_exists(self):
        assert ReasonCode.LOG_STREAM_ERROR.value == "log_stream_error"

    def test_clean_exit_pending_chunks_exists(self):
        assert ReasonCode.CLEAN_EXIT_PENDING_CHUNKS.value == "clean_exit_pending_chunks"


class TestLauncherUsesSharedConstant:
    """Grep-style guard: the hardcoded terminal sets that used to drift
    across launcher.py should all be gone. If a future edit reintroduces
    one, this test catches it."""

    def test_no_hardcoded_terminal_set_in_launcher(self):
        from pathlib import Path

        src = Path("orca_server/launcher.py").read_text()
        # The old pattern was:
        #   {"completed", "failed", "dead", "killed", "swapped_out"}
        # If you see this string anywhere in launcher.py, you've drifted.
        assert '"completed", "failed", "dead", "killed", "swapped_out"' not in src, (
            "Hardcoded terminal-phase set found in launcher.py; use "
            "TERMINAL_PHASES from orca_server.koi_contract instead."
        )

    def test_launcher_imports_shared_constant(self):
        # The module must import TERMINAL_PHASES so the above grep stays meaningful.
        from orca_server import launcher

        assert launcher.TERMINAL_PHASES is TERMINAL_PHASES


class TestMonitorReplicaReasonCodeUsage:
    """launcher.py should use the structured reason codes on every emit,
    not free-text strings."""

    def test_reason_codes_appear_in_launcher(self):
        from pathlib import Path

        src = Path("orca_server/launcher.py").read_text()
        assert "ReasonCode.MONITOR_THREAD_EXITED" in src
        assert "ReasonCode.LOG_STREAM_ERROR" in src
        assert "ReasonCode.CLEAN_EXIT_PENDING_CHUNKS" in src

    def test_startup_reason_codes_routed_in_launcher(self):
        """Cold-start failures should pick STARTUP_OOM or STARTUP_CRASH
        based on the runner-reported reason text in startup_failure_reason,
        and fall back to CLEAN_EXIT_PENDING_CHUNKS only when no signal
        is present."""
        from pathlib import Path

        src = Path("orca_server/launcher.py").read_text()
        assert "ReasonCode.STARTUP_OOM" in src
        assert "ReasonCode.STARTUP_CRASH" in src
        assert "startup_failure_reason" in src


class TestStartupOOMReasonCodes:
    """The new ReasonCode values added for cold-start failure classification."""

    def test_startup_oom_value(self):
        assert ReasonCode.STARTUP_OOM.value == "startup_oom"

    def test_startup_crash_value(self):
        assert ReasonCode.STARTUP_CRASH.value == "startup_crash"


class TestClassifyAttemptFailure:
    """The Orca-side helper that mirrors Koi's _classify_failure regex.
    Keeps /job/launch-failed payload's failure_category in sync with how
    Koi will categorize the same reason text."""

    def test_oom_message_classified(self):
        from orca_server.launcher import _classify_attempt_failure

        assert _classify_attempt_failure("CUDA out of memory") == "oom"
        assert _classify_attempt_failure("torch.cuda.OutOfMemoryError") == "oom"
        assert _classify_attempt_failure("OOM during sampler warmup") == "oom"

    def test_no_capacity_classified(self):
        from orca_server.launcher import _classify_attempt_failure

        assert _classify_attempt_failure("InsufficientInstanceCapacity") == "no_capacity"
        assert _classify_attempt_failure("no capacity available") == "no_capacity"

    def test_spot_preempt_classified(self):
        from orca_server.launcher import _classify_attempt_failure

        assert _classify_attempt_failure("spot interruption notice") == "spot_preemption"
        assert _classify_attempt_failure("instance preempted") == "spot_preemption"

    def test_quota_classified(self):
        from orca_server.launcher import _classify_attempt_failure

        assert _classify_attempt_failure("quota exceeded for L40S") == "quota"

    def test_unknown_default(self):
        from orca_server.launcher import _classify_attempt_failure

        assert _classify_attempt_failure("some unrelated error") == "unknown"
        assert _classify_attempt_failure("") == "unknown"
        assert _classify_attempt_failure(None) == "unknown"
