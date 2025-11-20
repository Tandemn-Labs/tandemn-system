from abc import ABC, abstractmethod
from typing import List, AsyncIterator


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


    # ============================== PreSigned Upload and Download Features ==============================

    # The point we are looking at is to create multipart uploads by
    # creating presigned URLs for each chunk of the total payload.
    # The client will take care of the multipart lifecyle and we will just return the presigned URLs.
    # However such kind of mechanism is NOT NEEDED for downloada
    @abstractmethod
    async def presigned_upload(self, remote_path:str, user:str, expires_seconds:int=600) -> dict: 
        """
        Returns a presigned URL for a SINGLE CHUNK of the total payload.
        return {"type": "single", "url":".....", "method":"put", "headers":{"Content-Type":"application/octet-stream"}}
        """
        pass

    @abstractmethod
    async def presigned_download(self, remote_path:str, user:str, expires_seconds:int=600) -> dict: 
        """
        Returns a presigned URL for downloading the total payload.
        return { "url":".....", "method":"get", "headers":{"Accept":"application/octet-stream"}}
        """
        pass

    # multipart lifecycle
    @abstractmethod
    async def multipart_upload_start(self, remote_path:str, user:str) -> dict:
        """
        This basically marks the start of the multipart upload and just returns the upload Id.
        When you start uploading on AWS S3, in multiparts, you get an UploadID which
        acts as a uniqe identifier for the multipart upload.
        return {"upload_id":".....", "key":"....."}
        """
        pass

    @abstractmethod
    async def multipart_sign_part(self, upload_id:str, user:str, remote_path:str, part_number:int, expires_seconds:int=600) -> dict:
        """
        Everytime we have to upload a file in a multipart chunk, we need to sign the part's URL.
        This will be called again and again for each part of the file we are uploading. 
        return {"url":".....", "method":"put", "headers":{"Content-Type":"application/octet-stream"}}
        """
        pass

    @abstractmethod
    async def multipart_complete(self, remote_path:str, user:str, upload_id:str, part_ids:list[str]) -> dict:
        """
        This is the final step in the multipart upload.
        This just returns the final URI of the uploaded file.
        """
        pass

    @abstractmethod
    async def multipart_abort(
        self,
        remote_path: str,
        user: str,
        upload_id: str,
    ) -> bool:
        """
        This just aborts the multipart upload.
        return True if successful
        """
        pass