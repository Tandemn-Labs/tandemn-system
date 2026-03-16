"""
Redis-backed chunk queue manager for distributed batch inference.

Each chunked job has:
  - A FIFO pending queue (replicas pull from)
  - An inflight set (chunks being processed)
  - A completed set (done chunks)
  - Per-chunk metadata hashes
  - An ordered list for output assembly
"""

import time
import logging
from typing import Optional

import redis

from orca_server.config import REDIS_URL

logger = logging.getLogger(__name__)

# Redis key prefixes
_PREFIX = "chunk:job"


def _meta_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:meta"

def _pending_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:pending"

def _inflight_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:inflight"

def _completed_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:completed"

def _chunk_key(job_id: str, chunk_id: str) -> str:
    return f"{_PREFIX}:{job_id}:chunk:{chunk_id}"

def _output_order_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:output_order"


class ChunkManager:
    """Redis-backed chunk queue for distributing work across replicas."""

    def __init__(self, redis_url: str = REDIS_URL):
        self._r = redis.from_url(redis_url, decode_responses=True)

    def create_job_queue(
        self,
        job_id: str,
        chunks: list[dict],
        model_name: str,
        s3_output_base: str,
    ) -> None:
        """Populate Redis with chunk queue for a job.

        Args:
            job_id: Parent job ID.
            chunks: List of dicts with keys: chunk_id, s3_input_path, num_lines.
            model_name: Model being served.
            s3_output_base: S3 prefix for chunk outputs.
        """
        pipe = self._r.pipeline()

        # Job metadata
        pipe.hset(_meta_key(job_id), mapping={
            "total_chunks": len(chunks),
            "model_name": model_name,
            "s3_output_base": s3_output_base,
            "created_at": time.time(),
        })

        for chunk in chunks:
            cid = chunk["chunk_id"]

            # Per-chunk hash
            s3_output_path = f"{s3_output_base}/chunks/{cid}.jsonl"
            pipe.hset(_chunk_key(job_id, cid), mapping={
                "s3_input_path": chunk["s3_input_path"],
                "s3_output_path": s3_output_path,
                "num_lines": chunk.get("num_lines", 0),
                "status": "pending",
                "replica_id": "",
                "started_at": 0,
                "completed_at": 0,
            })

            # FIFO queue
            pipe.rpush(_pending_key(job_id), cid)

            # Ordered list for assembly
            pipe.rpush(_output_order_key(job_id), cid)

        pipe.execute()
        logger.info(f"[ChunkManager] Created queue for job {job_id}: {len(chunks)} chunks")

    def pull_chunk(self, job_id: str, replica_id: str) -> Optional[dict]:
        """Pull next chunk from the pending queue.

        Returns chunk info dict, or None if queue is empty.
        """
        cid = self._r.lpop(_pending_key(job_id))
        if cid is None:
            return None

        # Move to inflight
        self._r.sadd(_inflight_key(job_id), cid)

        # Update chunk metadata
        now = time.time()
        self._r.hset(_chunk_key(job_id, cid), mapping={
            "status": "inflight",
            "replica_id": replica_id,
            "started_at": now,
        })

        # Read full chunk info
        info = self._r.hgetall(_chunk_key(job_id, cid))
        info["chunk_id"] = cid
        info["job_id"] = job_id
        return info

    def complete_chunk(self, job_id: str, chunk_id: str, replica_id: str) -> dict:
        """Mark chunk as completed and return progress.

        Returns dict with total, completed, pending, inflight, all_done.
        """
        now = time.time()
        pipe = self._r.pipeline()
        pipe.srem(_inflight_key(job_id), chunk_id)
        pipe.sadd(_completed_key(job_id), chunk_id)
        pipe.hset(_chunk_key(job_id, chunk_id), mapping={
            "status": "completed",
            "completed_at": now,
        })
        pipe.execute()

        return self.get_progress(job_id)

    def get_progress(self, job_id: str) -> Optional[dict]:
        """Get progress counts for a job."""
        meta = self._r.hgetall(_meta_key(job_id))
        if not meta:
            return None

        total = int(meta.get("total_chunks", 0))
        pending = self._r.llen(_pending_key(job_id))
        inflight = self._r.scard(_inflight_key(job_id))
        completed = self._r.scard(_completed_key(job_id))

        return {
            "total": total,
            "pending": pending,
            "inflight": inflight,
            "completed": completed,
            "all_done": completed >= total and total > 0,
        }

    def get_output_order(self, job_id: str) -> list[str]:
        """Get ordered list of chunk IDs for output assembly."""
        return self._r.lrange(_output_order_key(job_id), 0, -1)

    def get_chunk_info(self, job_id: str, chunk_id: str) -> dict:
        """Get metadata for a specific chunk."""
        info = self._r.hgetall(_chunk_key(job_id, chunk_id))
        if info:
            info["chunk_id"] = chunk_id
        return info

    def cleanup_job(self, job_id: str) -> None:
        """Remove all Redis keys for a job."""
        # Get all chunk IDs from output order
        chunk_ids = self.get_output_order(job_id)

        pipe = self._r.pipeline()
        pipe.delete(_meta_key(job_id))
        pipe.delete(_pending_key(job_id))
        pipe.delete(_inflight_key(job_id))
        pipe.delete(_completed_key(job_id))
        pipe.delete(_output_order_key(job_id))
        for cid in chunk_ids:
            pipe.delete(_chunk_key(job_id, cid))
        pipe.execute()
        logger.info(f"[ChunkManager] Cleaned up job {job_id}")


# Singleton
_chunk_manager: Optional[ChunkManager] = None


def get_chunk_manager() -> ChunkManager:
    global _chunk_manager
    if _chunk_manager is None:
        _chunk_manager = ChunkManager()
    return _chunk_manager
