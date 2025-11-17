# central_server/storage/s3_big.py
import os
import asyncio
import logging
from typing import AsyncIterator, List

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

from .base import StorageBackend

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler()) #? Why is this needed?

DEFAULT_MULTIPART_THRESHOLD_MB = 64
DEFAULT_MULTIPART_CHUNK_MB = 64
DEFAULT_MAX_CONCURRENCY = 16
DEFAULT_STREAM_CHUNK_MB = 8

class S3BigStorageBackend(StorageBackend):
    """
    Optimized S3 backend for large objects (multi-GB) and high concurrency.
    Uses boto3 TransferConfig to take advantage of multipart uploads/downloads
    and threaded concurrency without requiring changes to the FastAPI layer.
    """

    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.s3_client = boto3.client("s3", region_name=self.aws_region)

        multipart_threshold = int(
                os.getenv("S3_MULTIPART_THRESHOLD_MB", DEFAULT_MULTIPART_THRESHOLD_MB)
            ) * 1024 * 1024
        multipart_chunksize = int(
                os.getenv("S3_MULTIPART_CHUNK_MB", DEFAULT_MULTIPART_CHUNK_MB)
            ) * 1024 * 1024
        max_concurrency = int(
                os.getenv("S3_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
            )

        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=max_concurrency,
            use_threads=True,
        )

        self.stream_chunk_size = int(
            os.getenv("S3_STREAM_CHUNK_MB", DEFAULT_STREAM_CHUNK_MB)
        ) * 1024 * 1024

    def _get_key(self, remote_path: str, user: str) -> str:
        if remote_path.startswith("s3://"):
            return remote_path.replace(f"s3://{self.bucket_name}/", "")
        return f"{self.get_user_prefix(user)}{remote_path}"

    async def upload_file(self, local_path: str, remote_path: str, user: str) -> str:
        key = self._get_key(remote_path, user)
        try:
            await asyncio.to_thread(
                self.s3_client.upload_file,
                local_path,
                self.bucket_name,
                key,
                Config=self.transfer_config,
            )
            s3_uri = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Uploaded {local_path} to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    async def download_file(self, remote_path: str, local_path: str, user: str) -> str:
        key = self._get_key(remote_path, user)
        try:
            await asyncio.to_thread(
                self.s3_client.download_file,
                self.bucket_name,
                key,
                local_path,
                Config=self.transfer_config,
            )
            logger.info(f"Downloaded {key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise

    async def upload_data(self, data: bytes, remote_path: str, user: str) -> str:
        """
        For raw bytes we fall back to put_object (synchronous) since the payload
        is expected to be relatively small. Large payloads should be handled via upload_file.
        """
        key = self._get_key(remote_path, user)
        try:
            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
            )
            s3_uri = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Uploaded data to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading data to S3: {e}")
            raise

    async def download_data(self, remote_path: str, user: str) -> bytes:
        key = self._get_key(remote_path, user)
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object, Bucket=self.bucket_name, Key=key
            )
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"Error downloading data from S3: {e}")
            raise

    async def stream_file(
        self,
        remote_path: str,
        user: str,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        key = self._get_key(remote_path, user)
        chunk_size = chunk_size or self.stream_chunk_size
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object, Bucket=self.bucket_name, Key=key
            )
            body = response["Body"]
            while True:
                chunk = await asyncio.to_thread(body.read, chunk_size)
                if not chunk:
                    break
                yield chunk
        except ClientError as e:
            logger.error(f"Error streaming data from S3: {e}")
            raise

    async def list_files(self, prefix: str, user: str) -> List[str]:
        key_prefix = f"{self.get_user_prefix(user)}{prefix}"
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=key_prefix,
            )
            files: List[str] = []
            async for page in _paginate_async(page_iterator):
                contents = page.get("Contents", [])
                for obj in contents:
                    files.append(f"s3://{self.bucket_name}/{obj['Key']}")
            return files
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise

    async def delete_file(self, remote_path: str, user: str) -> bool:
        key = self._get_key(remote_path, user)
        try:
            await asyncio.to_thread(
                self.s3_client.delete_object,
                Bucket=self.bucket_name,
                Key=key,
            )
            logger.info(f"Deleted {key} from S3")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False

    async def file_exists(self, remote_path: str, user: str) -> bool:
        key = self._get_key(remote_path, user)
        try:
            await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=self.bucket_name,
                Key=key,
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

async def _paginate_async(page_iterator):
    """
    Helper to iterate over boto3 paginators without blocking the event loop.
    """
    iterator = iter(page_iterator)

    def _next_page():
        try:
            return next(iterator)
        except StopIteration:
            return None

    while True:
        page = await asyncio.to_thread(_next_page)
        if page is None:
            break
        yield page

