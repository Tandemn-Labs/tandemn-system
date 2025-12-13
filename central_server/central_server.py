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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict

import msgpack
from fastapi import BackgroundTasks, FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel
import redis.asyncio as redis
from pymongo import AsyncMongoClient
import uvicorn
import re
import zmq.asyncio

from central_server.models.models import NodeInfo, Application, UserRequest, ApplicationKey, NodeStatus, JobStatus, ApplicationStatus
from models.deployment_pb2 import Command, MagicOutput
from models.resources_pb2 import Node, Fabric

from google.protobuf.json_format import MessageToDict

from dotenv import load_dotenv


# ============================================================================
# Global variables
# ============================================================================

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MONGODB_URI = os.getenv("MONGODB_URI")
# if MONGODB_URI is None:
#     raise ValueError("MONGODB_URI is not set. Exiting.")
SERVER_PORT = int(os.getenv("CENTRAL_SERVER_PORT", 8000))
COMPUTE_NODE_PORT = int(os.getenv("TD_LISTENING_PORT", "12345"))

# Health monitoring
HEALTH_CHECK_INTERVAL = 1  # seconds

# Queue names
REQUEST_QUEUE = "tandemn:request_queue"
DEPLOYMENT_QUEUE = "tandemn:deployment_queue"
APPLICATION_QUEUE_PREFIX = "tandemn:app_jobs:" # Redis HASH where jobs are stored as job_id -> job_json

# Webserver
MIME_PROTOBUF = "application/protobuf"

# Node map
nodes_dict: dict[Node] = {}
nodemap_total: Fabric = None
nodemap_used: Fabric = None
nodemap_free: Fabric = None

# MongoDB
mongo_client: AsyncMongoClient = None
db = None
nodes_collection = None
fabric_collection = None

# Redis
redis_client = None

# ============================================================================
# Other setup
# ============================================================================

app = FastAPI(title="Tandemn Central Orchestrator Server")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# State Variables
nodes: Dict[str, NodeInfo] = {} # [node_id: NodeInfo{}]
applications: Dict[str, Application] = {} # [app_id: Application]
jobs: Dict[str, UserRequest] = {} # [job_id: JobInfo]
# app_key_to_id: Dict[str, str] = {}  # ✅ NEW: app_key_string → app_id (reverse index)

################################# MONGODB #####################################

async def fetch_node_map_from_mongodb() -> Dict[str, Any]:
    """
    Fetch the node map (all NodeInfo documents) from MongoDB, with their metrics and NodeInfo data.
    Returns:
        Dict[str, NodeInfo as dict] -- keyed by node_id.
    """
    try:
        node_map = {}
        # Fetch all NodeInfo docs from MongoDB (as dicts)
        all_nodes = nodes_collection_old.find({})
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

async def push_job_to_global_queue(job:UserRequest):
    """
    Push a job to the global queue which is a Redis List.
    This is called by the submit_job function.
    Returns:
        True if successful, False otherwise.
    """
    try:
        job_dict = job.model_dump(mode="json")
        job_json = json.dumps(job_dict)
        await redis_client.rpush(REQUEST_QUEUE, job_json)
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

async def add_job_to_application_queue(job:UserRequest):
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



# Nodes register themselves here 
# (will probably be scrapped when we have a better health service)
@app.post("/nodes/register", response_class=Response)
async def register_node(
    request: Request,
    content_type: str = Header(default=MIME_PROTOBUF)
):  
    print(content_type)
    if MIME_PROTOBUF not in content_type:
        raise HTTPException(status_code=415)
    
    node = Node()
    raw_body = await request.body()
    node.ParseFromString(raw_body)
    print(node)

    await add_node(node)


# Register a new node, write to MongoDB and update the node maps
# A new node is always assumed to be free, declaring used is another step
async def add_node(node: Node):
    
    global nodes_collection, fabric_collection
    global nodes_dict, nodemap_total, nodemap_free

    # Add node to collection in MongoDB
    node_dict = MessageToDict(
        node, 
        always_print_fields_with_no_presence=True,
        preserving_proto_field_name=True
    ) | {"_id": node.name, "last_updated": datetime.now(timezone.utc)}
    
    await nodes_collection.replace_one(
        {"_id": node.name},
        node_dict, upsert=True
    )
    nodes_dict[node.name] = node

    # Update nodemaps (if in total, we assume it's processed and ignore it)
    if node.name in nodemap_total.nodes:
        return
    
    # Update nodemap of total resources
    nodemap_total.nodes.append(node.name)
    nodemap_total.device_counts[node.device_type] = \
        nodemap_total.device_counts.get(node.device_type, 0) + node.device_count
    device_on_node = nodemap_total.devices_on_node[node.name]
    device_on_node.device_type = node.device_type
    device_on_node.device_count = node.device_count

    await write_nodemap_to_mongo(nodemap_total)

    # Update nodemap of free resources
    nodemap_free.nodes.append(node.name)
    nodemap_free.device_counts[node.device_type] = \
        nodemap_free.device_counts.get(node.device_type, 0) + node.device_count
    device_on_node = nodemap_free.devices_on_node[node.name]
    device_on_node.device_type = node.device_type
    device_on_node.device_count = node.device_count

    await write_nodemap_to_mongo(nodemap_free)

    print(nodemap_total)
    print(nodemap_free)


# Declare the nodes in the list as used, update the free and used node maps
# Update the node index with the job and deployment id
async def use_nodes(nodes: List[Node], job_id: str, deployment_id: str):

    global nodes_collection, fabric_collection, nodemap_used, nodemap_free

    for node in nodes:
        await nodes_collection.update_one(
            {"_id": node.name},
        )


# Helper function to take in a Fabric object, convert to dict and write to Mongo
async def write_nodemap_to_mongo(nodemap: Fabric):
    
    global fabric_collection

    fabric_dict = MessageToDict(
        nodemap, 
        always_print_fields_with_no_presence=True,
        preserving_proto_field_name=True
    ) | {"_id": nodemap.name, "last_updated": datetime.now(timezone.utc)}
    
    await fabric_collection.update_one(
        {"_id": nodemap.name},
        {"$set": fabric_dict}, upsert=True
    )

# ============================================================================

# One Function to submit jobs to the GLobal Queue. This is called by TandemnCLI by the user.

# PHASE 2: JOB SUBMISSION
# Function: submit_job()
# - Called by TandemnCLI when user submits a job
# - Validates job requirements
# - Pushes job to GLOBAL_QUEUE (Redis)
# - Returns job_id to caller

@app.post("/jobs/submit")
async def submit_job(request:UserRequest):
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

        job_info = UserRequest(
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


# CLI or API gateway or whatever inserts jobs into Redis. This coro processes them
async def process_deployments():

    global redis_client

    while True:
        # Read sth from Redis
        print("Listening for jobs on the queue...")
        _, magicb = await redis_client.blpop(DEPLOYMENT_QUEUE)
        print("Dequeued from redis...")

        magic_output = MagicOutput()
        magic_output.ParseFromString(magicb)
        print(magic_output)

        # Does not wait since the launch + run two steps might take quite long
        asyncio.create_task(process_orchestrator_output(magic_output))
        print("Task created...")

    
# TODO: discuss orchestrator output type
async def process_orchestrator_output(magic: MagicOutput):

    dep, job = magic.deploy, magic.job

    node_addrs = dep.node_addrs
    ctx = zmq.asyncio.Context()

    futures = []
    sockets = []
    for addr in node_addrs:
        sock = ctx.socket(zmq.REQ)
        sockets.append(sock)
        sock_addr = f"tcp://{addr}:{COMPUTE_NODE_PORT}"
        print("Socket: ", sock_addr)
        sock.connect(sock_addr)

        command = Command(
            action=Command.Action.LAUNCH,
            deployment_info=dep
        )

        future = sock.send(command.SerializeToString())
        futures.append(future)
    
    try: # Launching the model
        await asyncio.wait_for(asyncio.gather(*futures), timeout=60) 
        print("Sent launch command to nodes!")

        futures = []
        for sock in sockets:
            futures.append(sock.recv())
        
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=120)
        print("Received reply to launch from nodes!")
        failed = False
        for i in range(len(results)):
            reply = results[i]
            reply = msgpack.unpackb(reply)
            if reply["status"] != 1:
                failed = True
                logger.error(
                    f"Failed to deploy model, error from node ({dep.node_list[i]}): {reply["error"]}"
                )
        
        if not failed:
            logger.info("Model is up!")

    except Exception as e:
        # Could be timed out
        logging.exception(f"Error in process_orch_output model launching: {e}")
        return
    
    # Invoke the job
    command = Command(
        action=Command.Action.RUN,
        job_info=job
    )

    head_sock = sockets[0]
    try:
        await asyncio.wait_for(
            head_sock.send(command.SerializeToString()), timeout=60
        )
        reply = await head_sock.recv()
        reply = msgpack.unpackb(reply)
        print(reply)
    except Exception as e:
        # Could be timed out
        logging.exception(f"Error in process_orch_output starting jobs: {e}")

    # Now teardown
    command = Command(
        action=Command.Action.TEARDOWN
    )
    await sockets[0].send(command.SerializeToString())
    reply = await asyncio.wait_for(sockets[0].recv(), timeout=120)
    reply = msgpack.unpackb(reply)
    print(reply)
    
    for sock in sockets:
        sock.close(linger=0)
    ctx.term()


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

# Setup some initial variables
async def setup():
    
    # Connect to Redis
    global redis_client
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    # Connect to MongoDB
    global mongo_client, db, nodes_collection, fabric_collection
    global nodemap_total, nodemap_free, nodemap_used

    mongo_client = AsyncMongoClient(MONGODB_URI)
    db = mongo_client["tandemn_orca"]
    nodes_collection = db["nodes"]
    await nodes_collection.create_index("last_updated", expireAfterSeconds=3600)
    fabric_collection = db["fabrics"]
    await fabric_collection.create_index(
        "last_updated", expireAfterSeconds=3600
    )

    # Populate the nodemaps
    nodemap_total = Fabric(name="total")
    if (doc := await fabric_collection.find_one({"_id": "total"})) is not None:
        nodemap_total.ParseFromString(doc["store"])

    nodemap_free = Fabric(name="free")
    if (doc := await fabric_collection.find_one({"_id": "free"})) is not None:
        nodemap_free.ParseFromString(doc["store"])

    nodemap_used = Fabric(name="used")
    if (doc := await fabric_collection.find_one({"_id": "used"})) is not None:
        nodemap_used.ParseFromString(doc["store"])

    print(nodemap_total)
    print(nodemap_used)
    print(nodemap_free)


@app.on_event("startup")
async def startup():

    await setup()

    # Start background tasks
    asyncio.create_task(process_deployments())
    logger.info("Tandemn Central Orchestrator Server Started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)

