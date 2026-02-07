"""Redis key naming conventions for the chunk queue system."""

# ---- Defaults ----
DEFAULT_LEASE_TTL_SECONDS = 600  # 10 minutes
MAX_LEASE_RETRIES = 3


# ---- Queue keys ----

def pending_queue(job_id: str) -> str:
    """Sorted set of chunk IDs not yet leased. Score = chunk index (for ordering)."""
    return f"job:{job_id}:pending"


def leased_set(job_id: str) -> str:
    """Sorted set of chunk IDs currently leased. Score = lease expiry timestamp."""
    return f"job:{job_id}:leased"


def completed_set(job_id: str) -> str:
    """Set of chunk IDs that have been fully processed."""
    return f"job:{job_id}:completed"


def failed_set(job_id: str) -> str:
    """Set of chunk IDs that failed permanently (after max retries)."""
    return f"job:{job_id}:failed"


# ---- Per-chunk metadata ----

def chunk_meta(job_id: str, chunk_id: str) -> str:
    """Hash: input_path, chunk_index, worker_id, leased_at, lease_count."""
    return f"job:{job_id}:chunk:{chunk_id}"


# ---- Job-level metadata ----

def job_meta(job_id: str) -> str:
    """Hash: total_chunks, completed_count, status, model_name, input_prefix, bucket, etc."""
    return f"job:{job_id}:meta"


def job_workers(job_id: str) -> str:
    """Set of worker IDs registered for this job."""
    return f"job:{job_id}:workers"
