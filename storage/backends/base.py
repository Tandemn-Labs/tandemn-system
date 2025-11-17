# central_server/storage/base.py
from abc import ABC, abstractmethod
from typing import List, AsyncIterator
from pathlib import Path


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str, user: str) -> str:
        """
        Upload a file to storage.
        Returns: Storage URI/path for the uploaded file
        """
        pass
    
    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str, user: str) -> str:
        """
        Download a file from storage.
        Returns: Local path where file was downloaded
        """
        pass
    
    @abstractmethod
    async def upload_data(self, data: bytes, remote_path: str, user: str) -> str:
        """
        Upload raw bytes to storage.
        Returns: Storage URI/path for the uploaded data
        """
        pass
    
    @abstractmethod
    async def download_data(self, remote_path: str, user: str) -> bytes:
        """
        Download raw bytes from storage.
        Returns: File contents as bytes
        """
        pass
    

    @abstractmethod
    async def stream_file(
        self,
        remote_path: str,
        user: str,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        """
        Yield file contents in chunks so the API layer can stream them directly
        to the caller without writing to disk first.
        """
        pass

    @abstractmethod
    async def list_files(self, prefix: str, user: str) -> List[str]:
        """
        List files in storage with given prefix.
        Returns: List of file paths/URIs
        """
        pass
    
    @abstractmethod
    async def delete_file(self, remote_path: str, user: str) -> bool:
        """
        Delete a file from storage.
        Returns: True if successful
        """
        pass
    
    @abstractmethod
    async def file_exists(self, remote_path: str, user: str) -> bool:
        """
        Check if a file exists in storage.
        Returns: True if file exists
        """
        pass
    
    def get_user_prefix(self, user: str) -> str:
        """Get storage prefix for a user"""
        return f"users/{user}/"