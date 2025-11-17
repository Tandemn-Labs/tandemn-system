# central_server/storage_factory.py
import os
import logging
from .backends.base import StorageBackend
from .backends.s3 import S3StorageBackend
from .backends.s3_big import S3BigStorageBackend

logger = logging.getLogger(__name__)

def get_storage_backend() -> StorageBackend:
    """
    Factory function to get the appropriate storage backend based on environment variables.
    """
    storage_type = os.getenv("STORAGE_BACKEND", "s3_big").lower()
    bucket_name = os.getenv("S3_BUCKET_NAME")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is required to initialize the storage backend")

    
    if storage_type == "s3":
        logger.info("Initializing S3 storage backend")
        return S3StorageBackend()
    elif storage_type == "s3_big":
        logger.info("Initializing S3 Big storage backend")
        return S3BigStorageBackend()
    else:
        raise ValueError(f"Unknown storage backend type: {storage_type}")

