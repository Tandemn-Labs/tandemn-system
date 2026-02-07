"""
Server-side chunk queue manager.

Used by server.py to enqueue jobs on submission and by the
background tasks (reaper, combiner) to monitor progress.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import redis.asyncio as aioredis

from chunk_queue.keys import (
    pending_queue,
    leased_set,
    completed_set,
    failed_set,
    chunk_meta,
    job_meta,
    job_workers,
    DEFAULT_LEASE_TTL_SECONDS,
    MAX_LEASE_RETRIES,
)
from chunk_queue.lua_scripts import REAP_EXPIRED_LEASES

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Describes a single chunk within a job."""
    chunk_id: str  # e.g. "000001"
    chunk_index: int  # 0-based ordering index
    input_path: str  # path relative to bucket root, e.g. "user/req_id/000001.jsonl"


@dataclass
class JobQueueInfo:
    """Summary of a job's queue state."""

    job_id: str
    total_chunks: int
    pending: int
    leased: int
    completed: int
    failed: int

    @property
    def progress_frac(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.completed / self.total_chunks

    @property
    def is_done(self) -> bool:
        return (self.completed + self.failed) >= self.total_chunks


class ChunkQueueManager:
    """Server-side manager for the Redis chunk queue."""

    def __init__(self, redis_conn: aioredis.Redis):
        self.r = redis_conn
        self._reap_script = self.r.register_script(REAP_EXPIRED_LEASES)

    async def enqueue_job(
        self,
        job_id: str,
        chunks: List[ChunkInfo],
        model_name: str,
        input_prefix: str,
        bucket: str,
        lease_ttl: int = DEFAULT_LEASE_TTL_SECONDS,
    ) -> None:
        """
        Seed Redis with all chunks for a new job.
        Called once when POST /submit/batch is processed.
        """
        pipe = self.r.pipeline(transaction=True)

        # Job metadata
        pipe.hset(
            job_meta(job_id),
            mapping={
                "total_chunks": len(chunks),
                "completed_count": 0,
                "status": "queued",
                "model_name": model_name,
                "input_prefix": input_prefix,
                "bucket": bucket,
                "lease_ttl": lease_ttl,
                "created_at": time.time(),
            },
        )

        # Populate the pending queue and per-chunk metadata
        pq_key = pending_queue(job_id)
        for chunk in chunks:
            pipe.zadd(pq_key, {chunk.chunk_id: chunk.chunk_index})
            pipe.hset(
                chunk_meta(job_id, chunk.chunk_id),
                mapping={
                    "input_path": chunk.input_path,
                    "chunk_index": chunk.chunk_index,
                    "lease_count": 0,
                },
            )

        await pipe.execute()
        logger.info(f"Enqueued {len(chunks)} chunks for job {job_id}")

    async def get_job_progress(self, job_id: str) -> JobQueueInfo:
        """Read the size of each queue to compute progress."""
        pipe = self.r.pipeline(transaction=False)
        pipe.hget(job_meta(job_id), "total_chunks")
        pipe.zcard(pending_queue(job_id))
        pipe.zcard(leased_set(job_id))
        pipe.scard(completed_set(job_id))
        pipe.scard(failed_set(job_id))
        total_raw, pend, leased, comp, fail = await pipe.execute()
        return JobQueueInfo(
            job_id=job_id,
            total_chunks=int(total_raw) if total_raw else 0,
            pending=pend,
            leased=leased,
            completed=comp,
            failed=fail,
        )

    async def update_job_status(self, job_id: str, status: str) -> None:
        await self.r.hset(job_meta(job_id), "status", status)

    async def get_completed_chunks(self, job_id: str) -> List[str]:
        """Return completed chunk IDs sorted by name (000001, 000002, ...)."""
        chunks = list(await self.r.smembers(completed_set(job_id)))
        chunks.sort()
        return chunks

    async def reap_expired_leases(self, job_id: str) -> int:
        """Re-enqueue chunks whose worker died. Returns count reaped."""
        result = await self._reap_script(
            keys=[leased_set(job_id), pending_queue(job_id)],
            args=[time.time(), job_id, MAX_LEASE_RETRIES],
        )
        reaped = int(result) if result else 0
        if reaped > 0:
            logger.info(f"Reaped {reaped} expired leases for job {job_id}")
        return reaped

    async def cleanup_job(self, job_id: str) -> None:
        """Delete all Redis keys for a completed/failed job."""
        # Collect all chunk IDs so we can delete their metadata keys
        all_chunk_ids = set()
        all_chunk_ids.update(await self.r.smembers(completed_set(job_id)))
        all_chunk_ids.update(await self.r.smembers(failed_set(job_id)))
        all_chunk_ids.update(await self.r.zrange(pending_queue(job_id), 0, -1))
        all_chunk_ids.update(await self.r.zrange(leased_set(job_id), 0, -1))

        keys_to_delete = [
            pending_queue(job_id),
            leased_set(job_id),
            completed_set(job_id),
            failed_set(job_id),
            job_meta(job_id),
            job_workers(job_id),
        ]
        for cid in all_chunk_ids:
            keys_to_delete.append(chunk_meta(job_id, cid))

        await self.r.delete(*keys_to_delete)
        logger.info(f"Cleaned up Redis keys for job {job_id}")
