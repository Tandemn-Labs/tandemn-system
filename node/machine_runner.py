"""
Tandemn Node Machine Runner (Workload Orchestrator)

This service is responsible for:
- Registering the node with the central server (calls /nodes/register endpoint)
- Receiving deployment signals from Central Server
- Starting/stopping containers for specific applications
- Tracking current jobs and loaded applications on this node
- Exposing node state to health agent via /node_state endpoint

Health monitoring is handled by node_health_agent.py (separate service)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
import socket
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from dotenv import load_dotenv
from pathlib import Path
import sys
# ✅ Add parent directory to Python path for imports
# This allows importing from central-server and node_utils
TANDEMN_ORCA_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(TANDEMN_ORCA_ROOT))


from central_server.models.models import Request, Application, NodeStatus, DeployApplicationRequest

# ============================================================================
# Load Configuration from .env file (REQUIRED)
# ============================================================================


# Check if .env file exists
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    print("=" * 70)
    print("❌ ERROR: .env file not found!")
    print("=" * 70)
    sys.exit(1)

# Load environment variables from .env
load_dotenv(env_path)

# ============================================================================
# Configuration
# ============================================================================

CENTRAL_SERVER_HOST = os.getenv("CENTRAL_SERVER_HOST")
CENTRAL_SERVER_PORT = int(os.getenv("CENTRAL_SERVER_PORT"))

NODE_ID = os.getenv("NODE_ID")
IP_ADDRESS = os.getenv("IP_ADDRESS")

MACHINE_RUNNER_PORT = int(os.getenv("MACHINE_RUNNER_PORT"))
MACHINE_RUNNER_URL = os.getenv("MACHINE_RUNNER_URL")

# ============================================================================
# FASTAPI API + Logging Configuration
# ============================================================================

app = FastAPI(title="Tandemn Node Machine Runner")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Global State Variables
# ============================================================================

current_jobs: Dict[str, Request] = {} # [job_id: JobInfo]
current_applications: Dict[str, Application] = {} # [app_id: Application]


# ============================================================================
# Logical Functions for the Machine Runner
# ============================================================================

# One function to register the node with the central server
async def _register_with_central() -> None:
    payload = {
        "node_id": NODE_ID,
        "ip_address": IP_ADDRESS,
        "machine_runner_url": MACHINE_RUNNER_URL
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"http://{CENTRAL_SERVER_HOST}:{CENTRAL_SERVER_PORT}/nodes/register", json=payload)
            response.raise_for_status()
            logger.info("Registered with central server as %s", NODE_ID)
    except httpx.HTTPError as exc:
        logger.warning("Registration failed (%s). Retrying...", exc)
        await asyncio.sleep(2)

# ============================================================================
# Node State Endpoint (for Health Agent)
# ============================================================================

@app.get("/node_state")
async def get_node_state():
    """
    Returns current workload state of this node.
    Called by node_health_agent to include job/application info in health reports.
    """
    return {
        "node_id": NODE_ID,
        "current_jobs": list(current_jobs.keys()),
        "loaded_applications": list(current_applications.keys())
    }

# ============================================================================
# Application Deployment
# ============================================================================

@app.post("/deploy_application")
async def deploy_application(request: DeployApplicationRequest):
    """
    This function will deploy the application in the nodes in the assigned topology
    """
    if NODE_ID not in request.assigned_topology:
        return {"status": "skipped", "message": "This node is not part of the assigned topology."}
    logger.info(f"Deploying application {request.application_key} on node {NODE_ID}")
    # pull the docker image
    # start the container
    # give and set hyperparameters in the container based on the ApplicationKey
    # return the endpoint that the central server can then use to route jobs to this particular application
    await deploy_image(request.docker_image, request.application_key, request.assigned_topology)
    return {"status": "success", "message": "Application deployed successfully."}


async def deploy_image(docker_image: str, application_key: str, assigned_topology: dict):
    """
    This function will deploy the image on the nodes in the assigned topology
    """
    logging.info("Deployed!!!!!")
    pass 
# ============================================================================
# Startup Function
# ============================================================================

@app.on_event("startup")
async def startup_event():
    # register the node with the central server
    asyncio.create_task(_register_with_central())
    logger.info("Machine Runner Started")
    logger.info("Note: Start node_health_agent.py separately for health monitoring")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MACHINE_RUNNER_PORT)