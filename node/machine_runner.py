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
 


# ============================================================================
# Startup Function
# ============================================================================

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_register_with_central())
    logger.info("Machine Runner Started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MACHINE_RUNNER_PORT)