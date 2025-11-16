# central_server/storage/factory.py
import os
import logging
from storage.base import StorageBackend
from storage.s3 import S3StorageBackend


# from .azure_backend import AzureBlobStorageBackend
# from .local_backend import LocalStorageBackend

logger = logging.getLogger(__name__)

def get_storage_backend() -> StorageBackend:
    """
    Factory function to get the appropriate storage backend based on environment variables.
    """
    storage_type = os.getenv("STORAGE_BACKEND", "s3").lower()
    
    if storage_type == "s3":
        logger.info("Initializing S3 storage backend")
        # set the environment variables
        os.environ["S3_BUCKET_NAME"] = os.getenv("S3_BUCKET_NAME")
        os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
        return S3StorageBackend()
    else:
        raise ValueError(f"Unknown storage backend type: {storage_type}")