"""Tests for orca_server/koi_contract.py — shared Orca ↔ Koi event contract."""

import pytest
from pydantic import ValidationError

from orca_server.koi_contract import EventEnvelope, ReasonCode, TERMINAL_PHASES


def test_envelope_all_fields_optional():
    env = EventEnvelope()
    assert env.event_id is None
    assert env.event_type is None
    assert env.emitted_at is None
    assert env.correlation_id is None


def test_envelope_accepts_valid_payload():
    env = EventEnvelope(
        event_id="replica_failed:mo-abc-r0",
        event_type="replica_failed",
        emitted_at=1234567890.5,
        correlation_id="sr-xyz",
    )
    assert env.event_id == "replica_failed:mo-abc-r0"
    assert env.event_type == "replica_failed"
    assert env.emitted_at == 1234567890.5
    assert env.correlation_id == "sr-xyz"


def test_envelope_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        EventEnvelope(unknown_field="oops")


def test_reason_code_stable_values():
    """Values must match Koi's koi/contract.py literal strings."""
    assert ReasonCode.HEARTBEAT_TIMEOUT.value == "heartbeat_timeout"
    assert ReasonCode.CLEAN_EXIT_PENDING_CHUNKS.value == "clean_exit_pending_chunks"
    assert ReasonCode.LOG_STREAM_ERROR.value == "log_stream_error"
    assert ReasonCode.SPOT_PREEMPTION.value == "spot_preemption"
    assert ReasonCode.LAUNCH_CAPACITY_EXHAUSTED.value == "launch_capacity_exhausted"
    assert ReasonCode.MONITOR_THREAD_EXITED.value == "monitor_thread_exited"
    assert ReasonCode.KOI_INITIATED_KILL.value == "koi_initiated_kill"
    assert ReasonCode.MODEL_LOAD_TIMEOUT.value == "model_load_timeout"
    assert ReasonCode.UNKNOWN.value == "unknown"


def test_reason_code_is_string_enum():
    assert ReasonCode.HEARTBEAT_TIMEOUT == "heartbeat_timeout"


def test_terminal_phases_contents():
    assert "completed" in TERMINAL_PHASES
    assert "failed" in TERMINAL_PHASES
    assert "dead" in TERMINAL_PHASES
    assert "killed" in TERMINAL_PHASES
    assert "swapped_out" in TERMINAL_PHASES


def test_terminal_phases_is_frozenset():
    assert isinstance(TERMINAL_PHASES, frozenset)


def test_terminal_phases_excludes_non_terminal():
    assert "launching" not in TERMINAL_PHASES
    assert "running" not in TERMINAL_PHASES
    assert "model_ready" not in TERMINAL_PHASES
