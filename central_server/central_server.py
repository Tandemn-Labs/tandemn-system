#!/usr/bin/env python3
"""
Tandemn Central Orchestrator Server
- Monitors global queue (Redis)
- Tracks GPU node health
- Schedules jobs across GPU nodes
- Manages application queues
- Coordinates docker/model loading
"""

import asyncio
import json 
import os
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
import redis
import httpx
from pymongo import MongoClient
import uvicorn
import re



# ============================================================================
# Configuration
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://choprahetarth:helloworld@demo-day.tjaxr2t.mongodb.net/?retryWrites=true&w=majority&appName=demo-day")
SERVER_PORT = int(os.getenv("CENTRAL_SERVER_PORT", 8000))

# Health monitoring
HEALTH_CHECK_INTERVAL = 1  # seconds

# Queue names
GLOBAL_QUEUE_KEY = "tandemn:global_queue"
APPLICATION_QUEUE_PREFIX = "tandemn:app_jobs:" # Redis HASH where jobs are stored as job_id -> job_json

# ============================================================================
# Data Models
# ============================================================================

from models.models import NodeInfo, Application, JobInfo, ApplicationKey, NodeStatus, JobStatus, ApplicationStatus

# ============================================================================
# API + LoggingConfiguration
# ============================================================================

app = FastAPI(title="Tandemn Central Orchestrator Server")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Connect to Global Queue (Redis)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
# check redis connection
if not redis_client.ping():
    logger.critical("Failed to connect to GLobal Queue")
    os._exit(1)

# Connect to MongoDB (for node info)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["tandemn_orca"]
nodes_collection = db["node_info"]
applications_collection = db["application_info"]

# State Variables
nodes: Dict[str, NodeInfo] = {} # [node_id: NodeInfo{}]
applications: Dict[str, Application] = {} # [app_id: Application]
jobs: Dict[str, JobInfo] = {} # [job_id: JobInfo]
# app_key_to_id: Dict[str, str] = {}  # ✅ NEW: app_key_string → app_id (reverse index)

# ============================================================================
# Helper Functions
# ============================================================================

################################# MONGODB #####################################

async def register_node_to_mongodb(node_info: NodeInfo):
    """
    Register a new node to the MongoDB Collection of NodesCollection.
    This should happen ONLY at the BOOTUP of all the machines, BEFORE the health monitor is up.
    This can ALSO HAPPEN when a node is DEAD and is being re-registered.
    This can ALSO HAPPEN when a node's IP Changes but is still AWAITING_STATUS/HEALTHY.
    """
    try:
        node_dict = node_info.model_dump(mode="json")
        # Convert datetime to ISO format string for MongoDB
        if node_dict.get("registration_time"):
            node_dict["registration_time"] = node_dict["registration_time"]
        # Set document expiration after 30 hours (in seconds)
        # MongoDB TTL works with a date, so we use a 'last_updated' field for TTL indexing.
        node_dict["last_updated"] = str(datetime.utcnow())  # always add/update
        nodes_collection.update_one(
            {"node_id": node_info.node_id},
            {"$set": node_dict},
            upsert=True
        )
        # Ensure TTL index exists (will not duplicate after creation)
        # This sets a TTL of 30 hours for a node if not updated
        nodes_collection.create_index("last_updated", expireAfterSeconds=30*60*60)
        logger.debug(f"Node {node_info.node_id} registered to MongoDB")
    except Exception as e:
        logger.error(f"Error registering node {node_info.node_id} to MongoDB: {e}")
        raise e

async def update_node_in_mongodb(node: NodeInfo):
    """
    Update a node in the MongoDB database.
    This is called by the machine runner when the health monitor is up and running.
    Also updates the 'last_updated' field for TTL expiration.
    """
    try:
        node_dict = node.model_dump(mode="json")
        node_dict["last_updated"] = str(datetime.utcnow())  # update TTL expiration
        nodes_collection.update_one({"node_id": node.node_id}, {"$set": node_dict})
        # Optionally, ensure TTL index exists (no-op if already present)
        nodes_collection.create_index("last_updated", expireAfterSeconds=30*60*60)
    except Exception as e:
        logger.error(f"Error updating node {node.node_id} in MongoDB: {e}")
        raise e
    logger.info(f"Node {node.node_id} updated in MongoDB")


async def fetch_node_map_from_mongodb() -> Dict[str, Any]:
    """
    Fetch the node map (all NodeInfo documents) from MongoDB, with their metrics and NodeInfo data.
    Returns:
        Dict[str, NodeInfo as dict] -- keyed by node_id.
    """
    try:
        node_map = {}
        # Fetch all NodeInfo docs from MongoDB (as dicts)
        all_nodes = nodes_collection.find({})
        for node_doc in all_nodes:
            node_id = node_doc.get("node_id")
            # Omit MongoDB's internal id for clarity in display
            if "_id" in node_doc:
                del node_doc["_id"]
            node_map[node_id] = node_doc
        return node_map
    except Exception as e:
        logger.error(f"Error fetching node map from MongoDB: {e}")
        raise e

################################# REDIS #####################################

async def push_job_to_global_queue(job:JobInfo):
    """
    Push a job to the global queue which is a Redis List.
    This is called by the submit_job function.
    Returns:
        True if successful, False otherwise.
    """
    try:
        job_dict = job.model_dump(mode="json")
        job_json = json.dumps(job_dict)
        redis_client.rpush(GLOBAL_QUEUE_KEY, job_json)
        return True
    except Exception as e:
        logger.error(f"Error pushing job {job.job_id} to global queue: {e}")
        raise e

async def get_application_queue_key(app_key: ApplicationKey) -> str:
    """
    Return the per-application Redis HASH key for storing jobs tied to this application.
    Example :
    app_key = ApplicationKey(model_name="llama-70b-hf", task_mode="batched_inference", backend="vllm", quantization="awq")
    returns "tandemn:app_jobs:llama-70b-hf:batched_inference:vllm:awq"
    """
    try: 
        app_key_string = app_key.to_string()
        app_queue_key = f"{APPLICATION_QUEUE_PREFIX}{app_key_string}"
        return app_queue_key
    except Exception as e:
        logger.error(f"Error getting application queue key for application key: {app_key}: {e}")
        raise e

async def add_job_to_application_queue(job:JobInfo):
    """
    Store the job in the per-application Redis HASH keyed by job_id -> job_json.
    If it is not deployed, we dont call it "TILL" its deployed.
    """
    application_queue_key = job.app_queue_key
    if application_queue_key is None:
        logger.error(f"Application queue key is not set for job {job.job_id}. Ignoring.")
        raise HTTPException(status_code=400, detail=f"Application queue key is not set for job {job.job_id}")

    try:
        job_dict = job.model_dump(mode="json")
        job_json = json.dumps(job_dict)
        redis_client.hset(application_queue_key, job.job_id, job_json)

        logger.info(f"Job {job.job_id} stored in application job map {application_queue_key}")
        return True
    except Exception as e:
        logger.error(f"Error adding job {job.job_id} to application job map {application_queue_key}: {e}")
        raise e
############################# DEPLOYMENT FUNCTIONS #####################################

class TandemnPlanner:

    def __init__(self,job: JobInfo, node_map: Dict[str, Any]):
        self.job = job
        self.node_map = node_map
        logging.info("TandemnPlanner initialized")
    
    async def get_optimal_nodes(self) -> str:
        """
        This function provide the best node for the job and return the assigned topology.
        """
        # for now it returns the entire node_map as is
        # in the future, it will return the best nodes for the job 
        # TODO - GANGMUK'S PLANNER TO BE CALLED HERE.
        return self.node_map

async def get_application_docker_image(application_key: ApplicationKey) -> str:
    """
    This function will return the docker image for the application.
    """
    # keep it a placeholder for now
    return "vllm/llama-70b-hf:latest"

# ============================================================================
# Central Server Functions
# ============================================================================

# One function to register a new node with the central server when the machine_runner starts
# and the health monitor is NOT UP yet. This is just to initialize the Node Map.
# This is for the time, when the health monitor is booting up 
# (installing nvidia drivers, python packages and shit)

# One function to update the node status when the health monitor is up. The idea is to 
# take the node map, and update the status of each of them as the health monitor says on each of the nodes.
# This also keeps on pushing the updates to the MongoDB database (asynchronously)

# PHASE 1: NODE REGISTRATION
# Function: register_node()
# - Called by machine_runner when it first starts up
# - Registers node in the node map before health monitor is ready
# - Use case: Node is provisioning (installing drivers, dependencies, etc.)

@app.post("/nodes/register")
async def register_node(request: NodeInfo, background_tasks: BackgroundTasks):
    try:
        node_id = request.node_id # get the node id from machine_runner
        ip_address = request.ip_address # get the ip address of the node from machine_runner
        machine_runner_url = request.machine_runner_url # get the machine runner url from machine_runner

        if node_id in nodes:
            logger.warning(f"Node {node_id} is already registered. Updating the Node Info")
            if nodes[node_id].status == NodeStatus.DEAD:
                # re - register a DEAD NODE
                nodes[node_id].status = NodeStatus.AWAITING_STATUS
                nodes[node_id].ip_address = ip_address
                nodes[node_id].registration_time = datetime.now() # re-registration time
            elif nodes[node_id].ip_address != ip_address:
                # in case the IP Changed but node isnt dead
                nodes[node_id].ip_address = ip_address
            else:
                pass # node is already registered and healthy (do not do anything)
        else:
            # new node registration
            registration_time = datetime.now() # get the current time

            node_info = NodeInfo(
                node_id=node_id, 
                ip_address=ip_address,
                machine_runner_url=machine_runner_url,
                status=NodeStatus.AWAITING_STATUS,
                registration_time=registration_time,
                # rest everything is None and Optional as Health Montior is not up yet.
            )
            nodes[node_id] = node_info # add it to node directory
            # add it to the MongoDB database asynchronously
            background_tasks.add_task(register_node_to_mongodb, node_info)
            logger.info(f"Registering node {node_id} with IP {ip_address} and Machine Runner URL {machine_runner_url}")
            return {
                "status": "registered",
                "node_id": node_id,
                "ip_address": ip_address,
                "registration_time": registration_time,
                "message": f"Node {node_id} registered successfully"
            }
    except Exception as e:
        logger.error(f"Error registering node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error registering node {node_id}: {e}")

# Function: update_node_status()
# - Called periodically by node's health monitor once it's operational
# - Updates node metrics (GPU availability, memory, temperature, etc.)
# - Persists updates to MongoDB asynchronously
# - Updates in-memory nodes dict for fast scheduling decisions

@app.post("/nodes/update_status")
async def update_node_status(request:NodeInfo, background_tasks: BackgroundTasks):
    """
    Called periodically by node's health monitor (once it is up and running).
    Updates node metrics (GPU, memory, etc.) in both in-memory dictionary and MongoDB database
    This IS the Heartbeat Service.
    """
    try:
        node_id = request.node_id
        if node_id not in nodes:
            logger.warning(f"Update status received for unknown node {node_id}. Ignoring.")
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        # Update selected fields for in-memory nodes dict (do not overwrite everything; only relevant keys)
        node = nodes[node_id]

        # Always update these fields if present in updated_info
        node.gpus = request.gpus
        node.total_vram_gb = request.total_vram_gb
        node.available_vram_gb = request.available_vram_gb
        node.status = request.status
        node.last_seen = datetime.now()
        node.cpu_percent = request.cpu_percent
        node.ram_percent = request.ram_percent

        # these two are optional and only will be updated when jobs are queued or applications are loaded. 
        if request.current_jobs is not None:
            node.current_jobs = request.current_jobs # this is taken from the global
        else:
            node.current_jobs = [] # if no jobs are queued, set it to an empty list
        if request.loaded_applications is not None:
            node.loaded_applications = request.loaded_applications # this is taken from the global
        else:
            node.loaded_applications = [] # if no applications are loaded, set it to an empty list

        # Persist to MongoDB asynchronously
        if background_tasks is not None:
            background_tasks.add_task(update_node_in_mongodb, node)
        else:
            await update_node_in_mongodb(node)

        logger.info(f"Updated node {node_id} status: {node.status}, GPUs: {node.gpus}, VRAM: {node.available_vram_gb}")

        return {
            "status": "success",
            "node_id": node_id,
            "message": f"Node status updated"
        }
    except Exception as e:
        logger.error(f"Error updating node status for {request.node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating node status: {e}")


# ============================================================================

# One Function to submit jobs to the GLobal Queue. This is called by TandemnCLI by the user.

# PHASE 2: JOB SUBMISSION
# Function: submit_job()
# - Called by TandemnCLI when user submits a job
# - Validates job requirements
# - Pushes job to GLOBAL_QUEUE (Redis)
# - Returns job_id to caller

@app.post("/jobs/submit")
async def submit_job(request:JobInfo):
    """
    This is either called by the TandemnCLI when the user submits a job.
    """
    try:
        job_id = request.job_id # unique identifier for the job
        user = request.user# username of the user who submitted the job
        submit_time = datetime.now() # when the job was submitted
        task_mode = request.task_mode # batched_inference, streaming_inference etc.

        if task_mode not in ["batched_inference", "online_inference", "image_generation"]: # a config file of all supported modes needs to be made for this. 
            raise HTTPException(status_code=400, detail=f"Invalid task mode: {task_mode}")

        model_name= request.model_name # llama-70b-hf

        if request.backend is not None:
            request.backend = request.backend.lower()
            if request.backend not in ["vllm", "sglang"]: # a config file of all supported backends needs to be made. 
                raise HTTPException(status_code=400, detail=f"Invalid backend: {request.backend}")
            else:
                backend = request.backend
        else:
            backend = None
        
        if request.quantization is not None:
            request.quantization = request.quantization.lower()
            if request.quantization not in ["awq", "fp4", "GGMEMMFp8", "gptq", "marlin", "exl2", "bitsandbytes", "bitsandbytes-nf4", "bitsandbytes-fp4"]: # a config file of all supported quantizations needs to be made. 
                raise HTTPException(status_code=400, detail=f"Invalid quantization: {request.quantization}")
            else:
                quantization = request.quantization
        else:
            quantization = None

        if request.slo is not None and task_mode == "batched_inference":
            slo = request.slo
            if isinstance(slo, str):
                # Accept formats like "1h", "2h", "3h", etc.
                pattern = r'^\d+\s*[h]$'
                if not re.match(pattern, slo.strip().lower()):
                    raise HTTPException(status_code=400, detail="SLO must be like '1h', '2h', '3h', etc. (h=hours)")
            else:
                raise HTTPException(status_code=400, detail="SLO must be a string")
        else:
            slo = None

        if request.dataset_path is not None and task_mode == "batched_inference":
            dataset_path = request.dataset_path
            column_names = request.column_names
        else:
            dataset_path = None
            column_names = None

        if request.generation_kwargs is not None:
            generation_kwargs = request.generation_kwargs
        else:
            generation_kwargs = None # just use the default ones provide dby the library for the model
        
        # make the application key from the job info
        application_key = ApplicationKey(
            model_name=model_name,
            task_mode=task_mode,
            backend=backend if backend is not None else None,
            quantization=quantization if quantization is not None else None
        )
        logging.info(f"Application key: {application_key}")
        application_queue_key = await get_application_queue_key(application_key)
        logging.info(f"Application queue key: {application_queue_key}")

        job_info = JobInfo(
            job_id=job_id,
            user=user,
            submit_time=submit_time,
            task_mode=task_mode,
            model_name=model_name,
            app_queue_key = application_queue_key,
            backend=backend if backend is not None else None,
            quantization=quantization if quantization is not None else None,
            dataset_path=dataset_path if dataset_path is not None else None,
            column_names=column_names if column_names is not None else None,
            slo=slo if slo is not None else None,
            generation_kwargs=generation_kwargs if generation_kwargs is not None else None
        )

        jobs[job_id] = job_info # add it to the jobs dictionary
        await push_job_to_global_queue(job_info)
        # this needs to be removed from here and added to the global_queue_polling_loop function.
        # await add_job_to_application_queue(job_info)

        return {
            "status": "success",
            "job_id": job_id,
            "message": f"Job {job_id} submitted successfully",
            "application_queue_key": application_queue_key
        }

    except Exception as e:
        logger.error(f"Error submitting job {request.job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting job {request.job_id}: {e}")


# ============================================================================

# One Function to keep on polling jobs from the Global Queue (Redis). This will, as soon as a job is queued: 
# 1 - Check which application needs to run (model+task_mode+backend+quantization)
# 2 - Checks if application is already running on any node. If yes, pop the job from the Global Queue, and add it to the Application queue, and current_jobs in the NodeInfo of those nodes. 
# 3 - If not, if batched - upload dataset + start the container with the application in parallel. 
# 3.5 - For the launching  - query tandemn_planner with the node_map (above function), and allocate nodes for the job and start container on them. 
# 4 - Pops job from Global Queue, adds it to the Application_queue, and current_jobs in the NodeInfo of those nodes. 
# Split in multiple functions if needed. The main function should be non blocking and async. Everytime a job is queued, its applicationkey should be made. 

# One function to get the docker-image from the docker/apptainer hub. This is called by the machine_runner 
# when the Application_queue gets something. The function is called, and each of those nodes are asked to pull the image, and load the application onto those GPUs

# PHASE 3: JOB SCHEDULING & DISPATCH (Main Loop)
# Function: global_queue_polling_loop()
# - Continuously polls GLOBAL_QUEUE for new jobs
# - For each job:
#   Step 1: Extract ApplicationKey (model + task_mode + backend + quantization)
#   Step 2: Check if application is already running on any node
#     - YES: Route job to that application's queue
#            Add job_id to existing node's current_jobs list
#     - NO:  Provision new application instance:
#            a) Query tandemn_planner for optimal node allocation
#            b) Send container start command to selected nodes
#            c) Wait for application readiness
#   Step 3: Pop job from GLOBAL_QUEUE
#   Step 4: Add job to APPLICATION_QUEUE_{app_key} 
#   Step 5: Update node's current_jobs list in NodeInfo
# - Non-blocking: Process jobs asynchronously

async def global_queue_polling_loop():
    """
    Continuously polls global queue and routes jobs to application queues.
    - Checks if application already exists using app_key_string mapping
    - Creates new application if needed
    - Maintains applications dict keyed by app_key_string
    """
    logger.info("🔄 Starting global queue polling loop...")
    
    while True:
        try:
            # Blocking pop from global queue (1 second timeout)
            current_job = redis_client.blpop(GLOBAL_QUEUE_KEY, timeout=1)
            if not current_job:
                logger.info("No jobs in the global queue")
                await asyncio.sleep(0.1)
                continue
            # Parse job
            _, job_json = current_job # Returns: ("tandemn:global_queue", "{"job_id":"job_001",...}") so we ignore "tandemn:global_queue"
            job_data = json.loads(job_json)
            current_job_info = JobInfo(**job_data) # Converts the job_data to a JobInfo object
            logger.info(f"📥 Processing job {current_job_info.job_id} from global queue")
            # Step 1: Extract ApplicationKey from job
            application_key = ApplicationKey(
                model_name=current_job_info.model_name,
                task_mode=current_job_info.task_mode,
                backend=current_job_info.backend,
                quantization=current_job_info.quantization
            )
            app_key_string = await get_application_queue_key(application_key)
            # Step 2: Check if application already exists (single-level, keyed by app_key_string)
            if app_key_string in applications:
                app = applications[app_key_string]
                logger.info(f"Reusing existing application: {app_key_string}")
                # Add job to application's running jobs
                app.running_jobs.append(current_job_info.job_id)
            else:
                # Need to create new application
                logger.info(f"🆕 Creating new application for {app_key_string}")
                # Query tandemn_planner for node allocation
                node_map = await fetch_node_map_from_mongodb()
                tandemn_planner = TandemnPlanner(current_job_info, node_map)
                assigned_topology = await tandemn_planner.get_optimal_nodes()
                docker_image = await get_application_docker_image(application_key)
                app_id = f"app_{uuid.uuid4().hex[:8]}"
                application = Application(
                    app_id=app_id,
                    key=application_key,
                    docker_image=docker_image,
                    assigned_topology=json.dumps(assigned_topology),
                    status=ApplicationStatus.LOADING.value,
                    created_at=datetime.now(),
                    running_jobs=[current_job_info.job_id]
                )
                applications[app_key_string] = application
                logger.info(f"✅ Created application {app_key_string}")
                # Trigger deployment to nodes
                await deploy_application(assigned_topology, docker_image, application_key)
            
            # Step 3: Store job JSON in per-application Redis HASH (unsorted)
            await add_job_to_application_queue(current_job_info)
            
            # Step 4: Update job status
            current_job_info.status = JobStatus.QUEUED
            jobs[current_job_info.job_id] = current_job_info
            
            logger.info(f"✅ Job {current_job_info.job_id} routed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in global queue polling loop: {e}")
            await asyncio.sleep(1)

# PHASE 4: APPLICATION DEPLOYMENT
# Function: deploy_application()
# - Called internally when scheduling determines new app instance needed
# - Sends deployment request to machine_runner on target nodes
# - Machine_runner pulls docker/apptainer image
# - Machine_runner starts container and loads model onto GPUs
# - Returns when application is ready to accept jobs

async def deploy_application(assigned_topology: str, docker_image: str, application_key: ApplicationKey):
    """
    This function sends a signal to the machine_runners of the 
    nodes in the assigned topology to deploy the application
    """
    for node_id, node_info in assigned_topology.items():
        try:
            machine_runner_url = node_info.get('machine_runner_url')
            if not machine_runner_url:
                logger.error(f"Missing machine_runner_url for node {node_id}, skipping deployment.")
                continue
            deploy_endpoint = f"{machine_runner_url}/deploy_application"

            payload = {
                "docker_image": docker_image,
                "application_key": application_key.to_string() if hasattr(application_key, "to_string") else str(application_key),
                "assigned_topology": assigned_topology,  # Pass full topology for context
            }

            # Use an httpx client for async POST
            async with httpx.AsyncClient() as client:
                response = await client.post(deploy_endpoint, json=payload, timeout=60)
                if response.status_code != 200:
                    logger.error(f"Deployment to node {node_id} failed: {response.status_code} - {response.text}")
                else:
                    logger.info(f"Deployment of application {application_key} to node {node_id} succeeded.")
        except Exception as e:
            logger.error(f"Exception deploying to node {node_id}: {e}")

# ============================================================================
# Monitoring Functions
# ============================================================================

# One Function to fetch the node map from the MongoDB database, with their metrics, and NodeInfos.

@app.get("/nodes/map")
async def get_node_map():
    """
    API endpoint: Returns full node map with all metrics & NodeInfo from MongoDB.
    """
    try:
        node_map = await fetch_node_map_from_mongodb()
        return {"nodes": node_map}
    except Exception as e:
        logger.error(f"Failed to return node map: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch node map")

# One Function to check job status of each of the jobs. This is called by TandemnCLI at the user level.
# It goes through all APPLICATION JOB MAPS (Redis HASHES), and checks the status of each of the jobs.

# ============================================================================
# Startup Functions
# ============================================================================

@app.on_event("startup")
async def startup():
    # Start background tasks
    asyncio.create_task(global_queue_polling_loop())
    logger.info("Tandemn Central Orchestrator Server Started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)






