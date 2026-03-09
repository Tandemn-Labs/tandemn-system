import os
import asyncio
import logging
from typing import AsyncIterator, List

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from botocore.config import Config

from utils.utils import split_uri


from .base import StorageBackend

logger = logging.getLogger(__name__)


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
        self.aws_region = os.getenv("AWS_REGION", "us-east-2")
        boto_config = Config(signature_version="s3v4")
        # print("running in region: ", self.aws_region)
        self.s3_client = boto3.client(
            "s3", region_name=self.aws_region, config=boto_config
        )

        multipart_threshold = (
            int(os.getenv("S3_MULTIPART_THRESHOLD_MB", DEFAULT_MULTIPART_THRESHOLD_MB))
            * 1024
            * 1024
        )
        multipart_chunksize = (
            int(os.getenv("S3_MULTIPART_CHUNK_MB", DEFAULT_MULTIPART_CHUNK_MB))
            * 1024
            * 1024
        )
        max_concurrency = int(os.getenv("S3_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))

        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=max_concurrency,
            use_threads=True,
        )

        self.stream_chunk_size = (
            int(os.getenv("S3_STREAM_CHUNK_MB", DEFAULT_STREAM_CHUNK_MB)) * 1024 * 1024
        )

    def _get_bucket_and_key(self, remote_path: str) -> tuple[str, str]:
        """
        Extract bucket name and key from remote_path.
        If remote_path is a full S3 URI (s3://bucket/key), extract both.
        Otherwise, use the user prefix and remote_path as the key.
        Returns: (bucket_name, key)
        """
        if remote_path.startswith("s3://"):
            uri_bucket, key = split_uri(remote_path)
            bucket_name = uri_bucket.split("://")[1]
            return bucket_name, key
        # For non-URI paths, we need a bucket name - this should come from remote_path
        # If it's not a full URI, we can't determine the bucket, so raise an error
        raise ValueError(
            f"remote_path must be a full S3 URI (s3://bucket/key) or include bucket information. Got: {remote_path}"
        )

    def _get_key(self, remote_path: str, user: str) -> str:
        """Deprecated: Use _get_bucket_and_key instead. Kept for backward compatibility."""
        _, key = self._get_bucket_and_key(remote_path)
        return key

    async def upload_file(self, local_path: str, remote_path: str, user: str) -> str:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        try:
            await asyncio.to_thread(
                self.s3_client.upload_file,
                local_path,
                bucket_name,
                key,
                Config=self.transfer_config,
            )
            s3_uri = f"s3://{bucket_name}/{key}"
            logger.info(f"Uploaded {local_path} to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

    async def download_file(self, remote_path: str, local_path: str, user: str) -> str:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        try:
            await asyncio.to_thread(
                self.s3_client.download_file,
                bucket_name,
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
        bucket_name, key = self._get_bucket_and_key(remote_path)
        try:
            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=bucket_name,
                Key=key,
                Body=data,
            )
            s3_uri = f"s3://{bucket_name}/{key}"
            logger.info(f"Uploaded data to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading data to S3: {e}")
            raise

    async def download_data(self, remote_path: str, user: str) -> bytes:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object, Bucket=bucket_name, Key=key
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
        bucket_name, key = self._get_bucket_and_key(remote_path)
        chunk_size = chunk_size or self.stream_chunk_size
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object, Bucket=bucket_name, Key=key
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
        return ""

    # async def list_files(self, prefix: str, user: str) -> List[str]:
    #     # For list_files, prefix should be a full S3 URI or we need to handle it differently
    #     # If prefix is a full URI, extract bucket; otherwise, we can't determine bucket
    #     if prefix.startswith("s3://"):
    #         bucket_name, key_prefix = self._get_bucket_and_key(prefix, user)
    #     else:
    #         # If prefix doesn't include bucket, we can't list without knowing the bucket
    #         raise ValueError(
    #             f"prefix must be a full S3 URI (s3://bucket/prefix) when bucket_name is not configured. Got: {prefix}"
    #         )
    #     try:
    #         paginator = self.s3_client.get_paginator("list_objects_v2")
    #         page_iterator = paginator.paginate(
    #             Bucket=self.bucket_name,
    #             Prefix=key_prefix,
    #         )
    #         files: List[str] = []
    #         async for page in _paginate_async(page_iterator):
    #             contents = page.get("Contents", [])
    #             for obj in contents:
    #                 files.append(f"s3://{self.bucket_name}/{obj['Key']}")
    #         return files
    #     except ClientError as e:
    #         logger.error(f"Error  qw listing S3 objects: {e}")
    #         raise

    async def delete_file(self, remote_path: str, user: str) -> bool:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        try:
            await asyncio.to_thread(
                self.s3_client.delete_object,
                Bucket=bucket_name,
                Key=key,
            )
            logger.info(f"Deleted {key} from S3")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False

    async def file_exists(self, remote_path: str, user: str) -> bool:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        try:
            await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=bucket_name,
                Key=key,
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    # ============================== PreSigned Upload and Download Features ==============================
    # the first two are for the small and simple payloads when the files are in Kilobytes or Megabytes.
    async def presigned_upload(
        self, remote_path: str, user: str, expires_seconds: int = 600
    ) -> dict:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        params = {"Bucket": bucket_name, "Key": key}
        try:
            url = self.s3_client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=expires_seconds,
            )
            return {
                "type": "single",
                "url": url,
                "method": "put",
                "headers": {"Content-Type": "application/octet-stream"},
                "key": key,
                "s3_uri": f"s3://{bucket_name}/{key}",
            }
        except ClientError as e:
            logger.error(f"Error generating presigned upload URL: {e}")
            raise

    async def presigned_download(
        self, remote_path: str, user: str, expires_seconds: int = 600
    ) -> dict:
        """
        This coroutine should spawn a new thread, that runs in the
        thread pool background executor and just returns the URL when
        it is available.
        """
        bucket_name, key = self._get_bucket_and_key(remote_path)

        params = {"Bucket": bucket_name, "Key": key}
        try:
            url = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                "get_object",
                Params=params,
                ExpiresIn=expires_seconds,
            )
            return {
                "url": url,
                "method": "GET",
                "headers": {"Accept": "application/octet-stream"},
                "key": key,
            }
        except ClientError as e:
            logger.error(f"Error generating presigned download URL: {e}")
            raise

    # ===================== Multipart Upload and Download Features =====================
    async def multipart_upload_start(self, remote_path: str, user: str) -> dict:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        response = await asyncio.to_thread(
            self.s3_client.create_multipart_upload, Bucket=bucket_name, Key=key
        )
        return {
            "upload_id": response["UploadId"],
            "key": key,
            "bucket": bucket_name,
            "s3_uri": f"s3://{bucket_name}/{key}",
        }

    async def multipart_sign_part(
        self,
        upload_id: str,
        user: str,
        remote_path: str,
        part_number: int,
        expires_seconds: int = 600,
    ) -> dict:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        params = {
            "Bucket": bucket_name,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        }
        url = await asyncio.to_thread(
            self.s3_client.generate_presigned_url,
            "upload_part",
            Params=params,
            ExpiresIn=expires_seconds,
        )
        return {
            "url": url,
            "method": "PUT",
            "headers": {"Content-Type": "application/octet-stream"},
            "key": key,
            "upload_id": upload_id,
            "part_number": part_number,
        }

    async def multipart_complete(
        self, remote_path: str, user: str, upload_id: str, parts: list[dict]
    ) -> dict:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        await asyncio.to_thread(
            self.s3_client.complete_multipart_upload,
            Bucket=bucket_name,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        return {
            "s3_uri": f"s3://{bucket_name}/{key}",
            "key": key,
            "bucket": bucket_name,
        }

    # TODO(Hetarth): have to test it if it still works or not
    async def multipart_abort(
        self, remote_path: str, user: str, upload_id: str
    ) -> bool:
        bucket_name, key = self._get_bucket_and_key(remote_path)
        await asyncio.to_thread(
            self.s3_client.abort_multipart_upload,
            Bucket=bucket_name,
            Key=key,
            UploadId=upload_id,
        )
        return True


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
