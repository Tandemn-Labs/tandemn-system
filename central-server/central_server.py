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


# ============================================================================
# Configuration
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
SERVER_PORT = int(os.getenv("CENTRAL_SERVER_PORT", 8000))

# Health monitoring
HEALTH_CHECK_INTERVAL = 10  # seconds
NODE_TIMEOUT = 30  # seconds before marking node as DEAD
AWAITING_TIMEOUT = 60  # seconds before moving AWAITING → DEAD

# Queue names
GLOBAL_QUEUE_KEY = "tandemn:global_queue"
APPLICATION_QUEUE_PREFIX = "tandemn:app_queue:"

# ============================================================================
# Data Models
# ============================================================================
from models.models import Application, ApplicationKey, NodeStatus



