"""
Synchronous Redis client for worker processes.
Workers connect to Redis on the control plane over the network.
"""

import redis


def get_redis_sync(
    host: str,
    port: int = 6379,
    password: str = None,
    db: int = 0,
) -> redis.Redis:
    """Return a synchronous Redis client for worker processes."""
    return redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True,
        socket_timeout=10,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )
