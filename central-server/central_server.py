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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
import redis
import httpx
from pymongo import MongoClient
import uvicorn


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
# APPLICATION_QUEUE_PREFIX = "tandemn:app_queue:"

# ============================================================================
# Data Models
# ============================================================================

from models.models import NodeInfo, Application, JobInfo, ApplicationKey, NodeStatus

# ============================================================================
# API + LoggingConfiguration
# ============================================================================

app = FastAPI(title="Tandemn Central Orchestrator Server")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Connect to Redis (for global queue)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Connect to MongoDB (for node info)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client["tandemn_orca"]
nodes_collection = db["node_info"]
applications_collection = db["application_info"]

# State Variables
nodes: Dict[str, NodeInfo] = {} # [node_id: NodeInfo{}]
applications: Dict[str, Application] = {} # [app_id: Application]
jobs: Dict[str, JobInfo] = {} # [job_id: JobInfo]

# ============================================================================
# Helper Functions
# ============================================================================

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
        node_dict["last_updated"] = datetime.utcnow()  # always add/update
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

# ============================================================================

# One Function to submit jobs to the GLobal Queue. This is called by TandemnCLI/SLURM by the user.

# PHASE 2: JOB SUBMISSION
# Function: submit_job()
# - Called by TandemnCLI/SLURM when user submits a job
# - Validates job requirements
# - Pushes job to GLOBAL_QUEUE (Redis)
# - Returns job_id to caller

# ============================================================================

# One Function to keep on polling jobs from the Global Queue (Redis). This will, as soon as a job is queued: 
# 1 - Check which application needs to run (model+task_mode+backend+quantization)
# 2 - Checks if application is already running on any node. If yes, pop the job from the Global Queue, and add it to the Application queue (mongodb database), and current_jobs in the NodeInfo of those nodes. 
# 3 - If not, if batched - upload dataset + start the container with the application in parallel. 
# 3.5 - For the launching  - query tandemn_planner with the node_map (above function), and allocate nodes for the job and start container on them. 
# 4 - Pops job from Global Queue, adds it to the Application_queue (mongodb database), and current_jobs in the NodeInfo of those nodes. 
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
#            c) For dataset-based jobs: Upload dataset in parallel
#            d) Wait for application readiness
#   Step 3: Pop job from GLOBAL_QUEUE
#   Step 4: Add job to APPLICATION_QUEUE_{app_key} (MongoDB)
#   Step 5: Update node's current_jobs list in NodeInfo
# - Non-blocking: Process jobs asynchronously

# PHASE 4: APPLICATION DEPLOYMENT
# Function: deploy_application()
# - Called internally when scheduling determines new app instance needed
# - Sends deployment request to machine_runner on target nodes
# - Machine_runner pulls docker/apptainer image
# - Machine_runner starts container and loads model onto GPUs
# - Returns when application is ready to accept jobs

# ============================================================================
# Monitoring Functions
# ============================================================================

# One Function to fetch the node map from the MongoDB database, with their metrics, and NodeInfos.

# One Function to check job status of each of the jobs. This is called by the SLURM by the user.
# It goes through all APPLICATION QUEUES (mongodb database), and checks the status of each of the jobs.

# ============================================================================
# Startup Functions
# ============================================================================

@app.on_event("startup")
async def startup():
    # Start background tasks
    # asyncio.create_task(health_monitor_loop())
    # asyncio.create_task(global_queue_polling_loop())
    logger.info("Tandemn Central Orchestrator Server Started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)






