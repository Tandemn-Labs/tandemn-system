import os
import boto3
from botocore.exceptions import ClientError
from typing import List, AsyncIterator
import logging
import asyncio
from .base import StorageBackend

logger = logging.getLogger(__name__)

class S3StorageBackend(StorageBackend):
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is required")
        
        self.s3_client = boto3.client(
            's3',
            region_name=self.aws_region
        )
    
    def _get_key(self, remote_path: str, user: str) -> str:
        """Convert remote path to S3 key"""
        if remote_path.startswith("s3://"):
            return remote_path.replace(f"s3://{self.bucket_name}/", "")
        return f"{self.get_user_prefix(user)}{remote_path}"
    
    async def upload_file(self, local_path: str, remote_path: str, user: str) -> str:
        key = self._get_key(remote_path, user)
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, key)
            s3_uri = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Uploaded {local_path} to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    async def download_file(self, remote_path: str, local_path: str, user: str) -> str:
        key = self._get_key(remote_path, user)
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, key, local_path)
            logger.info(f"Downloaded {key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            raise
    
    async def upload_data(self, data: bytes, remote_path: str, user: str) -> str:
        key = self._get_key(remote_path, user)
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=data)
            s3_uri = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Uploaded data to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Error uploading data to S3: {e}")
            raise
    
    async def download_data(self, remote_path: str, user: str) -> bytes:
        key = self._get_key(remote_path, user)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error downloading data from S3: {e}")
            raise
    
    async def stream_file(
        self,
        remote_path: str,
        user: str,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        """
        Yield file chunks so FastAPI can stream them directly to the user.
        """
        key = self._get_key(remote_path, user)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
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
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=key_prefix)
            if 'Contents' not in response:
                return []
            return [f"s3://{self.bucket_name}/{obj['Key']}" for obj in response['Contents']]
        except ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
    
    async def delete_file(self, remote_path: str, user: str) -> bool:
        key = self._get_key(remote_path, user)
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted {key} from S3")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False
    
    async def file_exists(self, remote_path: str, user: str) -> bool:
        key = self._get_key(remote_path, user)
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise