import uvicorn
import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import tempfile
from dotenv import load_dotenv


from storage_factory import get_storage_backend

load_dotenv()
SERVER_PORT = int(os.getenv("STORAGE_SERVER_PORT", 8002))
CHUNK_SIZE_MB = int(os.getenv("CHUNK_SIZE_MB", 8)) * 1024 * 1024 # 8MB chunks

logger = logging.getLogger(__name__)
app = FastAPI(title="Tandemn Storage Server")
storage_backend = get_storage_backend()

# ============================================================================
# STORAGE FUNCTIONS
# ============================================================================

# ### One function to upload a file to the storage backend (actual_file, file_path and a user)
# ### One function to download the file from a (user and a file_path)
# One Function to STREAM UPLOAD a file to the backend (user and a file_path)
@app.post("/storage/upload")
async def upload_file_to_storage(
    file: UploadFile = File(...),
    remote_path: str = Form(...),
    user: str = Form(...)
):
    """
    Upload a file to storage backend using streaming.
    The file is streamed directly from the client to the storage backend/
    """
    try:
        logger.info(f"Uploading file for user {user} to {remote_path}")
        # Create a temporary file to stream the upload
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Stream the file in chunks to avoid memory issues
            chunk_size = CHUNK_SIZE_MB
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        # Upload to storage backend
        storage_uri = await storage_backend.upload_file(tmp_path, remote_path, user)
        # Clean up temp file
        os.unlink(tmp_path)
        logger.info(f"Successfully uploaded file to {storage_uri}")
        return {
            "status": "success",
            "storage_uri": storage_uri,
            "remote_path": remote_path,
            "user": user,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

# One function to get a list of all files for a user
@app.get("/storage/list/{user}")
async def list_user_files(user: str, prefix: str = ""):
    """
    List all files for a user in their storage space.
    Optional prefix filter for subdirectories.
    """
    try:
        logger.info(f"Listing files for user {user} with prefix '{prefix}'")
        files = await storage_backend.list_files(prefix, user)
        
        return {
            "status": "success",
            "user": user,
            "prefix": prefix,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

# One Function to STREAM DOWNLOAD a file from the backend (user and a file_path)
@app.get("/storage/download/{user}/{file_path:path}")
async def download_file_from_storage(user: str, file_path: str):
    """
    Download a file from storage backend using streaming.
    The file is streamed directly from storage to the client
    without fully loading into server memory.
    """
    try:
        logger.info(f"Downloading file {file_path} for user {user}")
        
        # Check if file exists first
        if not await storage_backend.file_exists(file_path, user):
            raise HTTPException(status_code=404, detail=f"File {file_path} not found")
        
        # Get filename for the download
        filename = file_path.split("/")[-1] or "download"
        
        # Stream the file
        async def file_stream_iterator():
            async for chunk in storage_backend.stream_file(file_path, user):
                yield chunk
        
        return StreamingResponse(
            file_stream_iterator(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

# One Function to Delete a File from the storage backend (user and a file_path)
@app.delete("/storage/delete/{user}/{file_path:path}")
async def delete_file_from_storage(user: str, file_path: str):
    """
    Delete a file from storage backend.
    """
    try:
        logger.info(f"Deleting file {file_path} for user {user}")
        
        # Check if file exists first
        if not await storage_backend.file_exists(file_path, user):
            raise HTTPException(status_code=404, detail=f"File {file_path} not found")
        
        success = await storage_backend.delete_file(file_path, user)
        
        if success:
            return {
                "status": "success",
                "message": f"File {file_path} deleted successfully",
                "user": user,
                "file_path": file_path
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete file {file_path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# One Function to check if a file exists in the storage backend (user and a file_path)
@app.get("/storage/exists/{user}/{file_path:path}")
async def check_file_exists(user: str, file_path: str):
    """
    Check if a file exists in storage backend.
    Quick lookup without downloading the file.
    """
    try:
        exists = await storage_backend.file_exists(file_path, user)
        
        return {
            "status": "success",
            "user": user,
            "file_path": file_path,
            "exists": exists
        }
    except Exception as e:
        logger.error(f"Error checking file existence: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking file: {str(e)}")