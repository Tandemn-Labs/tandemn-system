from enum import Enum
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from typing import List

class NodeStatus(str, Enum):
    """This is identifier of each machine that has a GPU."""
    HEALTHY = "HEALTHY"
    DEAD = "DEAD"
    AWAITING_STATUS = "AWAITING_STATUS"

class NodeInfo(BaseModel):
    """
    This is POPULATED after the machine_runner starts to run on 
    each node, spawns the health monitor docker/apptainer; 
    installs the gpu driver, and starts communicating to the central server
    """
    node_id: str # unique identifier for the node
    ip_address: str # ip address of the node
    status: NodeStatus = NodeStatus.AWAITING_STATUS  # HEALTHY, DEAD, AWAITING_STATUS
    # the below ones are populated and updated when the health monitor is up
    gpus: Optional[dict[int, str]] = None  # GPU Info of the node (index, name)
    total_vram_gb: Optional[dict[int, float]] = None  # total vram of the node (index, total vram in GB)
    available_vram_gb: Optional[dict[int, float]] = None  # available vram of the node (index, available vram in GB)
    last_seen: Optional[datetime] = None
    current_jobs: Optional[List[str]] = None  # List of Job IDs running on this node (app_id)
    loaded_applications: Optional[List[str]] = None  # List of App IDs loaded on this node (app_id)


class JobStatus(str, Enum):
    """This is for each JOB provided to it by SLURM"""
    QUEUED = "QUEUED"
    DOCKER_LOADING = "DOCKER_LOADING"
    MODEL_LOADING = "MODEL_LOADING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ApplicationKey(BaseModel):
    """Unique identifier for an application instance
    The user has to select a model_name, 
    task (batched/non-batched)."""
    model_name: str # llama-70b-hf
    task_mode: str # batched_inference, streaming_inference etc.
    # the below modes are optional and 
    # may/may not be available for all models
    backend: Optional[str] = None # vllm, sglang
    quantization: Optional[str] = None #awq, int4, int8, etc.
    SLO: Optional[dict] = None # Hours (mostly for batched inference)
    
    def to_string(self) -> str:
        parts = [self.model_name, self.backend, self.task_mode]
        if self.quantization:
            parts.append(self.quantization)
        return ":".join(parts)

class Application(BaseModel):
    """Represents a running application instance.
    This starts ONLY AFTER the health monitor is up"""
    app_id:str # unique identifier for the application instance
    key: ApplicationKey # unique identifier for the application instance
    docker_image:str # from tandemn's docker/apptainer hub
    assigned_topology:str 
    status: str # Loading, Ready, Running, Stopping, Stopped, Failed
    created_at: datetime
    running_jobs: List[str] # list of job ids


class JobInfo(BaseModel):
    """This is for each JOB provided to it by SLURM.
    This is the config that the USER provides. 
    in the future - Tandemn CLI Will accept these configurations.
    Which model? Which task mode? which dataset? which backend etc.
    If the model is NOT DEPLOYED, it will be deployed on the fly and
    an APPLICATION will be created,  specific to that model, backend, quantization, task mode etc."""
    job_id:str # unique identifier for the job
    app_id: str # where is this job going to run?
    status: JobStatus # QUEUED, DOCKER_LOADING, MODEL_LOADING, RUNNING, COMPLETED, FAILED
    user: str # username of the user who submitted the job
    submit_time: datetime # when the job was submitted
    task_mode: str # batched_inference, streaming_inference etc.
    model_name: str # llama-70b-hf
    backend: Optional[str] = None # vllm, sglang
    quantization: Optional[str] = None #awq, int4, int8, etc.
    dataset_path: Optional[str] = None # path to the dataset
    column_names: Optional[List[str]] = None # column names of the dataset
    generation_kwargs: Optional[dict] = None # generation kwargs

