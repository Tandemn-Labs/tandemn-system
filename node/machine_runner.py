"""
This is the machine runner and it is responsible for:
- Registering the node with the central server (calls /nodes/register endpoint)
- Starting the health monitor (docker/apptainer)
- Taking signal from Central Server and starts container for specific application
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from pathlib import Path
import sys
# ✅ Add parent directory to Python path for imports
# This allows importing from central-server and node_utils
TANDEMN_ORCA_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(TANDEMN_ORCA_ROOT))


from central_server.models.models import JobInfo, Application, NodeStatus
from node_utils.gpu_utils import get_gpu_info, get_total_free_vram, get_system_metrics

# ============================================================================
# Configuration
# ============================================================================

CENTRAL_SERVER_HOST = os.getenv("CENTRAL_SERVER_HOST", "localhost")  # this is the IP Address of Central Server
CENTRAL_SERVER_PORT = int(os.getenv("CENTRAL_SERVER_PORT", 8000))

NODE_ID = socket.gethostname() # get the hostname of the node
IP_ADDRESS = socket.gethostbyname(NODE_ID) # get the IP Address of the node

MACHINE_RUNNER_PORT = int(os.getenv("MACHINE_RUNNER_PORT", 8001))
MACHINE_RUNNER_URL = f"http://{IP_ADDRESS}:{MACHINE_RUNNER_PORT}"

# ============================================================================
# FASTAPI API + Logging Configuration
# ============================================================================

app = FastAPI(title="Tandemn Node Machine Runner")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Global State Variables
# ============================================================================

current_jobs: Dict[str, JobInfo] = {} # [job_id: JobInfo]
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

# One Function to start the Health Monitor Container
# The logic is this - 
# 1 - Check if the health monitor container is already running
# 2 - If not, detect the gpu type and install the appropriate nvidia/amd drivers
# 3 - Then install the necessary python packages and dependencies
# 4 - Run a python script in that container that reads gpu specific metrics
# 5 - This script will request those metrics from the container and send it to the central server

# We are making a simpler version here : The above is a placeholder for that time

async def _start_health_monitor(interval_s: float = 2.0) -> None:
    """
    Periodically collect node health metrics and send them to the central server.
    Terminates the process if it cannot contact the server for a sustained period.
    """
    consecutive_failures = 0
    max_failures = 30  # e.g., 1 minute tolerance if interval_s=2
    server_url = f"http://{CENTRAL_SERVER_HOST}:{CENTRAL_SERVER_PORT}/nodes/update_status"

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                system_metrics = await asyncio.to_thread(get_system_metrics)

                # Build GPU info dicts as required by Node model:
                gpus = {i: gpu.name for i, gpu in enumerate(system_metrics.gpu_info)}
                total_vram_gb = {i: gpu.total_vram_gb for i, gpu in enumerate(system_metrics.gpu_info)}
                available_vram_gb = {i: gpu.free_vram_gb for i, gpu in enumerate(system_metrics.gpu_info)}

                payload = {
                    "node_id": NODE_ID,
                    "ip_address": IP_ADDRESS,
                    "machine_runner_url": MACHINE_RUNNER_URL,
                    "gpus": gpus,
                    "total_vram_gb": total_vram_gb,
                    "available_vram_gb": available_vram_gb,
                    "current_jobs": list(current_jobs.keys()),
                    "loaded_applications": list(current_applications.keys()),
                    "status": NodeStatus.HEALTHY,
                    "cpu_percent": system_metrics.cpu_percent,
                    "ram_percent": system_metrics.ram_percent,
                    # Additional fields (like last_seen) can be added by the server
                }
                response = await client.post(server_url, json=payload)
                if response.status_code == 200:
                    logger.info("Health update sent to central server")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning("Health update HTTP %d", response.status_code)
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Health monitor error: {e}")

            if consecutive_failures >= max_failures:
                logger.error("Lost contact with central server, shutting down...")
                os._exit(1)
            await asyncio.sleep(interval_s)

 
 


# ============================================================================
# Startup Function
# ============================================================================

@app.on_event("startup")
async def startup_event():
    # register the node with the central server
    asyncio.create_task(_register_with_central())
    # make the infinite loop for the health monitor
    asyncio.create_task(_start_health_monitor())
    logger.info("Machine Runner Started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MACHINE_RUNNER_PORT)