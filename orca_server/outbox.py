"""
orca_server/outbox.py — Durable at-least-once delivery for Orca → Koi webhooks.

Orca writes events to a local SQLite outbox instead of firing webhooks
directly. A background publisher thread drains the outbox by POSTing to Koi
with bounded exponential backoff. Effectively-once processing is the
receiver's responsibility — Koi dedups by `event_id` in its inbox (see
`koi/runtime_state.py: claim_event`).

What this module guarantees:
  - **Surviving a Koi restart.** Events written while Koi is down stay
    queued and drain when Koi comes back.
  - **Dedup at source.** `enqueue(dedup_key=...)` uses INSERT OR IGNORE, so
    watchdog + monitor_replica both detecting the same replica death
    collapse to a single outbox row.
  - **Envelope injection.** Every payload gets `event_id`, `event_type`,
    `emitted_at`, `correlation_id` — the shared contract from
    `orca_server/koi_contract.py`.

Same-machine trim: Koi and Orca live on one box, so backoff is bounded
(1→2→4→8→16→30s cap) and audit retention is 24h, not 7d. This isn't
distributed-systems theater — it's local-process boundary hardening.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./state/outbox.db"
MAX_BACKOFF_SECS = 30.0


class OutboxDB:
    """SQLite-backed durable event queue for Orca → Koi webhook delivery."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    event_id         TEXT PRIMARY KEY,
                    event_type       TEXT NOT NULL,
                    webhook_path     TEXT NOT NULL,
                    payload          TEXT NOT NULL,
                    job_id           TEXT NOT NULL,
                    created_at       REAL NOT NULL,
                    next_attempt_at  REAL NOT NULL,
                    attempts         INTEGER NOT NULL DEFAULT 0,
                    delivered_at     REAL,
                    last_error       TEXT,
                    last_status_code INTEGER
                );
                CREATE INDEX IF NOT EXISTS ix_outbox_pending
                    ON outbox(next_attempt_at)
                    WHERE delivered_at IS NULL;
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Enqueue (state-change path)
    # ------------------------------------------------------------------

    def enqueue(
        self,
        path: str,
        event_type: str,
        payload: Dict[str, Any],
        *,
        job_id: str,
        dedup_key: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Append an event to the outbox, returning its event_id.

        If `dedup_key` is provided, it becomes the `event_id`, and a second
        enqueue with the same key is a no-op (INSERT OR IGNORE on the PK).
        That's how watchdog + monitor_replica both detecting the same
        replica death collapse to one delivered webhook.

        The envelope (event_id, event_type, emitted_at, correlation_id) is
        injected into the payload before serialization so Koi's handlers
        can dedup by event_id in their inbox.
        """
        now = time.time()
        event_id = dedup_key or uuid.uuid4().hex
        envelope = {
            "event_id": event_id,
            "event_type": event_type,
            "emitted_at": now,
            "correlation_id": correlation_id,
        }
        full_payload = {**envelope, **payload}
        payload_json = json.dumps(full_payload, default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO outbox
                    (event_id, event_type, webhook_path, payload, job_id,
                     created_at, next_attempt_at, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (event_id, event_type, path, payload_json, job_id, now, now),
            )
            self._conn.commit()
        return event_id

    # ------------------------------------------------------------------
    # Publisher surface (drain path)
    # ------------------------------------------------------------------

    def claim_batch(self, limit: int = 50, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return up to `limit` undelivered rows whose next_attempt_at is due.

        Rows returned are copies (dicts), safe to use outside the lock.
        """
        now = time.time() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_id, event_type, webhook_path, payload, job_id, attempts
                FROM outbox
                WHERE delivered_at IS NULL AND next_attempt_at <= ?
                ORDER BY next_attempt_at
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_delivered(self, event_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE outbox SET delivered_at = ?, last_error = NULL, last_status_code = NULL "
                "WHERE event_id = ?",
                (time.time(), event_id),
            )
            self._conn.commit()

    def mark_failure(
        self,
        event_id: str,
        error: str,
        status_code: Optional[int] = None,
    ) -> None:
        """Bump attempts and reschedule with bounded backoff: 1→2→4→8→16→30s.

        Same-machine: Koi restart is either fast (<5s) or something is
        genuinely wrong. 30s cap keeps recovery snappy without burning CPU
        on a dead target.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts FROM outbox WHERE event_id = ?", (event_id,)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            backoff = min(2 ** (attempts - 1), MAX_BACKOFF_SECS)
            self._conn.execute(
                """
                UPDATE outbox
                SET attempts = ?, next_attempt_at = ?,
                    last_error = ?, last_status_code = ?
                WHERE event_id = ?
                """,
                (attempts, time.time() + backoff, (error or "")[:2000], status_code, event_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Observability & housekeeping
    # ------------------------------------------------------------------

    def pending_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM outbox WHERE delivered_at IS NULL"
            ).fetchone()
        return int(row["n"])

    def oldest_undelivered_age_secs(self) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(created_at) AS m FROM outbox WHERE delivered_at IS NULL"
            ).fetchone()
        if row is None or row["m"] is None:
            return 0.0
        return max(0.0, time.time() - float(row["m"]))

    def prune_delivered(self, keep_secs: float = 86400.0) -> int:
        """Delete delivered rows older than keep_secs. Returns rows removed."""
        cutoff = time.time() - keep_secs
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM outbox WHERE delivered_at IS NOT NULL AND delivered_at < ?",
                (cutoff,),
            )
            self._conn.commit()
        return cur.rowcount


# ----------------------------------------------------------------------
# Module-level singleton + lifecycle hooks
# ----------------------------------------------------------------------

_OUTBOX: Optional[OutboxDB] = None


def get_outbox() -> Optional[OutboxDB]:
    """Return the process-wide OutboxDB, or None if outbox is disabled.

    Disabled (None) means either the server hasn't initialized it yet, or
    the operator set ORCA_OUTBOX_DB_PATH="" to opt out — in which case
    callers should fall back to direct HTTP POST (legacy behavior).
    """
    return _OUTBOX


def init_outbox(db_path: Optional[str] = None) -> Optional[OutboxDB]:
    """Initialize the process-wide outbox. Called once at server startup.

    An empty-string ORCA_OUTBOX_DB_PATH disables the outbox (rollback hatch).
    """
    global _OUTBOX
    resolved = db_path if db_path is not None else os.environ.get(
        "ORCA_OUTBOX_DB_PATH", DEFAULT_DB_PATH
    )
    if not resolved:
        logger.warning("[Outbox] Disabled (ORCA_OUTBOX_DB_PATH is empty)")
        _OUTBOX = None
        return None
    _OUTBOX = OutboxDB(resolved)
    logger.info("[Outbox] Initialized at %s", resolved)
    return _OUTBOX


def shutdown_outbox() -> None:
    """Close the outbox. Called once at server shutdown."""
    global _OUTBOX
    if _OUTBOX is not None:
        _OUTBOX.close()
        _OUTBOX = None
