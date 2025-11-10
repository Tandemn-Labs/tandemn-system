from enum import Enum
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from typing import List

class NodeStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEAD = "DEAD"
    AWAITING_STATUS = "AWAITING_STATUS"

class ApplicationKey(BaseModel):
    """Unique identifier for an application instance
    The priority of a user is to select a model_name, 
    task (batched/non-batched)."""
    model_name: str # llama-70b-hf
    task_mode: str # batched_inference, streaming_inference etc.
    backend: Optional[str] = None # vllm, sglang
    quantization: Optional[str] = None #awq, int4, int8, etc.
    
    def to_string(self) -> str:
        parts = [self.model_name, self.backend, self.task_mode]
        if self.quantization:
            parts.append(self.quantization)
        return ":".join(parts)

class Application(BaseModel):
    """Represents a running application instance"""
    app_id:str # unique identifier for the application instance
    key: ApplicationKey # unique identifier for the application instance
    docker_image:str # from tandemn's docker/apptainer hub
    assigned_topology:str 
    status: str # Loading, Ready, Running, Stopping, Stopped, Failed
    created_at: datetime
    running_jobs: List[str] # list of job ids


    

