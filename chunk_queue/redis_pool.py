"""
Async Redis connection pool for the control plane (server.py).
Redis runs co-located on the same machine.
"""

import os
from typing import Optional

import redis.asyncio as aioredis

_pool: Optional[aioredis.Redis] = None

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
REDIS_DB = int(os.environ.get("REDIS_DB", 0))


async def get_redis() -> aioredis.Redis:
    """Return the shared async Redis connection (lazy-init with connection pool)."""
    global _pool
    if _pool is None:
        _pool = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            max_connections=32,
        )
    return _pool


async def close_redis() -> None:
    """Gracefully close the Redis connection pool. Called on server shutdown."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
