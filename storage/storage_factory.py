import logging
import os

from storage.backends.base import StorageBackend
from storage.backends.s3_big import S3BigStorageBackend, S3CompatibleStorageBackend

logger = logging.getLogger(__name__)


def get_storage_backend() -> StorageBackend:
    """
    Factory function to get the appropriate storage backend based on environment variables.
    """
    storage_type = os.getenv("TD_STORAGE_BACKEND", "s3_big").lower()

    if storage_type in {"s3_big", "s3", "s3_compatible", "minio"}:
        # logger.info("Initializing S3 Big storage backend")
        if storage_type == "s3_big":
            return S3BigStorageBackend()
        return S3CompatibleStorageBackend()
    else:
        raise ValueError(f"Unknown storage backend type: {storage_type}")
