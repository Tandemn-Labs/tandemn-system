"""Shared contract for Orca → Koi webhook events.

Keep this file identical to koi/koi/contract.py.
Any change here must be mirrored on the Koi side in the same commit window.

Envelope fields are Optional during the hardening rollout so Koi can accept
both envelope-aware (new) and legacy payloads. Once Orca emits the envelope on
every event and the compat window closes, callers may treat these as required.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class EventEnvelope(BaseModel):
    """Envelope fields every Orca → Koi webhook carries."""

    model_config = ConfigDict(extra="forbid")

    event_id: Optional[str] = None
    event_type: Optional[str] = None
    emitted_at: Optional[float] = None
    correlation_id: Optional[str] = None


class ReasonCode(str, Enum):
    """Structured failure reason codes for replica/job failure webhooks."""

    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    CLEAN_EXIT_PENDING_CHUNKS = "clean_exit_pending_chunks"
    LOG_STREAM_ERROR = "log_stream_error"
    SPOT_PREEMPTION = "spot_preemption"
    LAUNCH_CAPACITY_EXHAUSTED = "launch_capacity_exhausted"
    MONITOR_THREAD_EXITED = "monitor_thread_exited"
    KOI_INITIATED_KILL = "koi_initiated_kill"
    MODEL_LOAD_TIMEOUT = "model_load_timeout"
    UNKNOWN = "unknown"


TERMINAL_PHASES: frozenset = frozenset(
    {"completed", "failed", "dead", "killed", "swapped_out"}
)
