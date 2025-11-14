#!/usr/bin/env python3
"""
Tandemn Node Health Agent
- Standalone service that monitors node health metrics
- Reports GPU, CPU, RAM, VRAM status to central server
- Queries machine_runner for current jobs/applications
- Designed to run independently from workload orchestration
"""

from __future__ import annotations

import asyncio
import os
import socket
import logging
from typing import Optional
from pathlib import Path
import sys

import httpx

TANDEMN_ORCA_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(TANDEMN_ORCA_ROOT))

from central_server.models.models import NodeStatus
from node_utils.gpu_utils import get_system_metrics

# ============================================================================
# Load Configuration from .env file (REQUIRED)
# ============================================================================

from dotenv import load_dotenv

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

# Machine Runner location (to query for current jobs/applications)
MACHINE_RUNNER_HOST = os.getenv("MACHINE_RUNNER_HOST")
MACHINE_RUNNER_PORT = int(os.getenv("MACHINE_RUNNER_PORT"))

# Node identification (must match what machine_runner registered)
NODE_ID = os.getenv("NODE_ID")
IP_ADDRESS = os.getenv("IP_ADDRESS")
MACHINE_RUNNER_URL = os.getenv("MACHINE_RUNNER_URL")

# Health monitoring configuration
HEALTH_CHECK_INTERVAL = float(os.getenv("HEALTH_CHECK_INTERVAL"))

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Health Monitoring Functions
# ============================================================================

async def get_machine_runner_state() -> tuple[list[str], list[str]]:
    """
    Placeholder for the actual stats (list of prompts within a job) that
    are running on this node. 
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            #in the actual implementation, we will call the container that 
            response = await client.get(f"http://{MACHINE_RUNNER_HOST}:{MACHINE_RUNNER_PORT}/node_state")
            if response.status_code == 200:
                data = response.json()
                return data.get("current_jobs", []), data.get("loaded_applications", [])
    except Exception as e:
        logger.warning(f"Could not fetch machine_runner state: {e}")
    
    return [], [] # return empty lists if we dont have that information

async def health_monitoring_loop() -> None:
    """
    Main health monitoring loop.
    Periodically collects node health metrics and sends them to central server.
    Is an infinite loop that runs until we stop it. 
    """
    server_url = f"http://{CENTRAL_SERVER_HOST}:{CENTRAL_SERVER_PORT}/nodes/update_status"
    
    logger.info(f"🏥 Starting health monitoring for node {NODE_ID}")
    logger.info(f"Reporting to: {server_url}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                # Collect system metrics\
                system_metrics = await asyncio.to_thread(get_system_metrics)
                # Build GPU info dicts
                gpus = {i: gpu.name for i, gpu in enumerate(system_metrics.gpu_info)}
                total_vram_gb = {i: gpu.total_vram_gb for i, gpu in enumerate(system_metrics.gpu_info)}
                available_vram_gb = {i: gpu.free_vram_gb for i, gpu in enumerate(system_metrics.gpu_info)}
                # Find what jobs and applications and all are running on this node
                current_jobs, loaded_applications = await get_machine_runner_state()
                # Build payload for central server
                payload = {
                    "node_id": NODE_ID,
                    "ip_address": IP_ADDRESS,
                    "machine_runner_url": MACHINE_RUNNER_URL,
                    "gpus": gpus,
                    "total_vram_gb": total_vram_gb,
                    "available_vram_gb": available_vram_gb,
                    "current_jobs": current_jobs,
                    "loaded_applications": loaded_applications,
                    "status": NodeStatus.HEALTHY.value,
                    "cpu_percent": system_metrics.cpu_percent,
                    "ram_percent": system_metrics.ram_percent,
                }
                
                # Send health update to central server
                response = await client.post(server_url, json=payload)
                
                if response.status_code == 200:
                    logger.info(f"Health update sent")
                else:
                    logger.warning(f"Health update failed")
                    
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for the health agent."""
    logger.info("=" * 60)
    logger.info("    Tandemn Node Health Agent Starting")
    logger.info(f"   Node ID: {NODE_ID}")
    logger.info(f" Reporting to Central Server: {CENTRAL_SERVER_HOST}:{CENTRAL_SERVER_PORT}")
    logger.info("=" * 60)
    
    # Start health monitoring loop
    await health_monitoring_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Health agent stopped by user")
    except Exception as e:
        logger.critical(f"💥 Health agent crashed: {e}")
        raise

