"""
Redis-backed chunk queue manager for distributed batch inference.

Each chunked job has:
  - A FIFO pending queue (replicas pull from)
  - An inflight set (chunks being processed)
  - A completed set (done chunks)
  - A failed set (chunks that exhausted retries)
  - Per-chunk metadata hashes
  - An ordered list for output assembly
"""

import time
import logging
from typing import Optional

import redis

from orca_server.config import REDIS_URL, CHUNK_LEASE_TTL_SEC, CHUNK_MAX_RETRIES

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

def _failed_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:failed"

def _chunk_key(job_id: str, chunk_id: str) -> str:
    return f"{_PREFIX}:{job_id}:chunk:{chunk_id}"

def _output_order_key(job_id: str) -> str:
    return f"{_PREFIX}:{job_id}:output_order"


# Atomically scan inflight set, reclaim expired leases, move exhausted chunks to failed.
# KEYS: [inflight_key, pending_key, failed_key]
# ARGV: [job_prefix, now_float, max_retries]
# Returns: [reclaimed_count, failed_count]
_RECLAIM_LUA = """
local inflight_key = KEYS[1]
local pending_key  = KEYS[2]
local failed_key   = KEYS[3]
local job_prefix   = ARGV[1]
local now          = tonumber(ARGV[2])
local max_retries  = tonumber(ARGV[3])

local members   = redis.call('SMEMBERS', inflight_key)
local reclaimed = 0
local failed    = 0

for _, cid in ipairs(members) do
    local chunk_key   = job_prefix .. ':chunk:' .. cid
    local lease_until = tonumber(redis.call('HGET', chunk_key, 'lease_until')) or 0
    if lease_until > 0 and lease_until < now then
        local retry_count = (tonumber(redis.call('HGET', chunk_key, 'retry_count')) or 0) + 1
        if retry_count >= max_retries then
            redis.call('SREM', inflight_key, cid)
            redis.call('SADD', failed_key, cid)
            redis.call('HSET', chunk_key,
                'status', 'failed',
                'retry_count', tostring(retry_count))
            failed = failed + 1
        else
            redis.call('SREM', inflight_key, cid)
            redis.call('RPUSH', pending_key, cid)
            redis.call('HSET', chunk_key,
                'status', 'pending',
                'retry_count', tostring(retry_count),
                'lease_until', '0')
            reclaimed = reclaimed + 1
        end
    end
end

return {reclaimed, failed}
"""


# Atomically reclaim inflight chunks owned by specific replica IDs (ignoring lease expiry).
# KEYS: [inflight_key, pending_key, failed_key]
# ARGV: [job_prefix, max_retries, replica_id_1, replica_id_2, ...]
# Returns: [reclaimed_count, failed_count]
_FORCE_RECLAIM_LUA = """
local inflight_key = KEYS[1]
local pending_key  = KEYS[2]
local failed_key   = KEYS[3]
local job_prefix   = ARGV[1]
local max_retries  = tonumber(ARGV[2])

local targets = {}
for i = 3, #ARGV do targets[ARGV[i]] = true end

local members   = redis.call('SMEMBERS', inflight_key)
local reclaimed = 0
local failed    = 0

for _, cid in ipairs(members) do
    local chunk_key  = job_prefix .. ':chunk:' .. cid
    local rid        = redis.call('HGET', chunk_key, 'replica_id') or ''
    if targets[rid] then
        local retry_count = (tonumber(redis.call('HGET', chunk_key, 'retry_count')) or 0) + 1
        if retry_count >= max_retries then
            redis.call('SREM', inflight_key, cid)
            redis.call('SADD', failed_key, cid)
            redis.call('HSET', chunk_key,
                'status', 'failed',
                'retry_count', tostring(retry_count))
            failed = failed + 1
        else
            redis.call('SREM', inflight_key, cid)
            redis.call('RPUSH', pending_key, cid)
            redis.call('HSET', chunk_key,
                'status', 'pending',
                'retry_count', tostring(retry_count),
                'lease_until', '0')
            reclaimed = reclaimed + 1
        end
    end
end

return {reclaimed, failed}
"""


# Count inflight chunks owned by a specific replica.
# KEYS: [inflight_key]
# ARGV: [job_prefix, target_replica_id]
# Returns: integer count
_REPLICA_INFLIGHT_LUA = """
local inflight_key    = KEYS[1]
local job_prefix      = ARGV[1]
local target_replica  = ARGV[2]
local count = 0
local members = redis.call('SMEMBERS', inflight_key)
for _, cid in ipairs(members) do
    local chunk_key = job_prefix .. ':chunk:' .. cid
    local rid = redis.call('HGET', chunk_key, 'replica_id') or ''
    if rid == target_replica then
        count = count + 1
    end
end
return count
"""


# Atomically renew a chunk lease if the replica still owns it.
# KEYS: [chunk_key]
# ARGV: [replica_id, new_lease_until]
# Returns: [renewed (0/1), lease_until]
_RENEW_LUA = """
local chunk_key  = KEYS[1]
local replica_id = ARGV[1]
local new_lease  = ARGV[2]

local status = redis.call('HGET', chunk_key, 'status')
local owner  = redis.call('HGET', chunk_key, 'replica_id')

if status ~= 'inflight' or owner ~= replica_id then
    return {0, 0}
end

redis.call('HSET', chunk_key, 'lease_until', new_lease)
return {1, new_lease}
"""


class ChunkManager:
    """Redis-backed chunk queue for distributing work across replicas."""

    def __init__(self, redis_url: str = REDIS_URL):
        self._r = redis.from_url(redis_url, decode_responses=True)
        self._reclaim_script = self._r.register_script(_RECLAIM_LUA)
        self._force_reclaim_script = self._r.register_script(_FORCE_RECLAIM_LUA)
        self._replica_inflight_script = self._r.register_script(_REPLICA_INFLIGHT_LUA)
        self._renew_script = self._r.register_script(_RENEW_LUA)

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
                "lease_until": 0,
                "retry_count": 0,
            })

            # FIFO queue
            pipe.rpush(_pending_key(job_id), cid)

            # Ordered list for assembly
            pipe.rpush(_output_order_key(job_id), cid)

        pipe.execute()
        logger.info(f"[ChunkManager] Created queue for job {job_id}: {len(chunks)} chunks")

    def pull_chunk(self, job_id: str, replica_id: str) -> Optional[dict]:
        """Pull next chunk from the pending queue.

        If the queue is empty but inflight chunks exist, attempts passive reclaim
        of any expired leases before giving up.

        Returns chunk info dict, or None if queue is empty.
        """
        cid = self._r.lpop(_pending_key(job_id))

        if cid is None:
            # Passive reclaim: if replicas are alive, lease renewals keep chunks
            # from expiring. Only reclaim if inflight exists and leases are stale.
            if self._r.scard(_inflight_key(job_id)) > 0:
                result = self.reclaim_expired_chunks(job_id)
                if result["reclaimed"] > 0:
                    cid = self._r.lpop(_pending_key(job_id))
            if cid is None:
                return None

        # Move to inflight
        self._r.sadd(_inflight_key(job_id), cid)

        # Update chunk metadata (preserve retry_count — it accumulates across attempts)
        now = time.time()
        self._r.hset(_chunk_key(job_id, cid), mapping={
            "status": "inflight",
            "replica_id": replica_id,
            "started_at": now,
            "lease_until": now + CHUNK_LEASE_TTL_SEC,
        })

        # Read full chunk info
        info = self._r.hgetall(_chunk_key(job_id, cid))
        info["chunk_id"] = cid
        info["job_id"] = job_id
        return info

    def renew_lease(self, job_id: str, chunk_id: str, replica_id: str) -> dict:
        """Extend lease for a chunk currently owned by replica_id (atomic via Lua).

        Returns {"renewed": True, "lease_until": float} on success.
        Returns {"renewed": False} if another replica owns the chunk or it was reclaimed.
        """
        new_lease = time.time() + CHUNK_LEASE_TTL_SEC
        result = self._renew_script(
            keys=[_chunk_key(job_id, chunk_id)],
            args=[replica_id, new_lease],
        )
        renewed = int(result[0])
        if renewed:
            return {"renewed": True, "lease_until": float(result[1])}
        return {"renewed": False}

    def reclaim_expired_chunks(self, job_id: str) -> dict:
        """Atomically reclaim inflight chunks whose leases have expired.

        Chunks exceeding CHUNK_MAX_RETRIES are moved to the failed set.
        All others are re-queued in pending.

        Returns {"reclaimed": N, "failed": M}.
        """
        result = self._reclaim_script(
            keys=[_inflight_key(job_id), _pending_key(job_id), _failed_key(job_id)],
            args=[f"{_PREFIX}:{job_id}", time.time(), CHUNK_MAX_RETRIES],
        )
        return {"reclaimed": int(result[0]), "failed": int(result[1])}

    def force_reclaim(self, job_id: str, replica_ids: list[str]) -> dict:
        """Immediately reclaim all inflight chunks owned by replica_ids (ignores lease time).

        Used by ReplicaWatchdog (dead replica) and Orca Swap (old replica teardown).
        Returns {"reclaimed": N, "failed": M}.
        """
        if not replica_ids:
            return {"reclaimed": 0, "failed": 0}
        result = self._force_reclaim_script(
            keys=[_inflight_key(job_id), _pending_key(job_id), _failed_key(job_id)],
            args=[f"{_PREFIX}:{job_id}", CHUNK_MAX_RETRIES] + list(replica_ids),
        )
        return {"reclaimed": int(result[0]), "failed": int(result[1])}

    def get_replica_inflight_count(self, job_id: str, replica_id: str) -> int:
        """Count inflight chunks owned by a specific replica."""
        return self._replica_inflight_script(
            keys=[_inflight_key(job_id)],
            args=[f"{_PREFIX}:{job_id}", replica_id],
        )

    def complete_chunk(self, job_id: str, chunk_id: str, replica_id: str) -> dict:
        """Mark chunk as completed and return progress.

        Idempotent: if already completed, returns current progress without
        double-counting.

        Returns dict with total, completed, pending, inflight, failed, all_done.
        """
        # Idempotency guard — two replicas may race to complete a reclaimed chunk
        if self._r.sismember(_completed_key(job_id), chunk_id):
            return self.get_progress(job_id)

        now = time.time()
        pipe = self._r.pipeline()
        pipe.srem(_inflight_key(job_id), chunk_id)
        pipe.srem(_failed_key(job_id), chunk_id)   # promote from failed if reclaim raced us
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
        failed = self._r.scard(_failed_key(job_id))

        return {
            "total": total,
            "pending": pending,
            "inflight": inflight,
            "completed": completed,
            "failed": failed,
            "all_done": (completed + failed) >= total and total > 0,
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

    def get_failed_chunk_ids(self, job_id: str) -> set:
        """Return set of chunk IDs that exhausted retries and permanently failed."""
        return self._r.smembers(_failed_key(job_id))

    def cleanup_job(self, job_id: str) -> None:
        """Remove all Redis keys for a job."""
        # Get all chunk IDs from output order
        chunk_ids = self.get_output_order(job_id)

        pipe = self._r.pipeline()
        pipe.delete(_meta_key(job_id))
        pipe.delete(_pending_key(job_id))
        pipe.delete(_inflight_key(job_id))
        pipe.delete(_completed_key(job_id))
        pipe.delete(_failed_key(job_id))
        pipe.delete(f"{_PREFIX}:{job_id}:assembling")
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
