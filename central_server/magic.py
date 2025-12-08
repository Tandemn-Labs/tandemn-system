# A magic orchestrator that reads jobs, and then decides what chain to launch
# and then run it on that chain

import asyncio
import json
import logging
import os
import random
import redis.asyncio as redis

from central_server.models.models import DeploymentInfo, JobInfo
from utils.utils import dict_from_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set up Redis
REQUEST_QUEUE = "tandemn:request_queue"
DEPLOYMENT_QUEUE = "tandemn:deployment_queue"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Connect to Global Queue (Redis)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Global variables
hostname_to_address = {}

# Continuous loop getting jobs from the Redis queue that is submitted by CLI
# Writes the deployment decision to another queue
async def get_jobs():

    while True:

        _, job = await redis_client.blpop(REQUEST_QUEUE)
        print("Dequeued from redis")
        job_info = JobInfo(**(json.loads(job)))
        print(job_info)

        # Makes the decision
        deployment_info = await real_magic(job_info)
        deployment_json = json.dumps(deployment_info.model_dump(mode="json"))
        print(deployment_info)
        
        await redis_client.rpush(DEPLOYMENT_QUEUE, deployment_json)
        print("Pushed to redis!")


# Hardcoded nodes
# This function is mostly experimental and a placeholder
# Ideally, the magic orchestrator should take in request + cluster state
# Returns node list
node_addrs = []
node_list = ["test1", "test2", "test3"]
async def real_magic(job: JobInfo):
    
    global hostname_to_address

    # Randomly choose one of the 3 nodes to exclude from the chain
    exclude = random.randint(0,2)

    nodes = node_list.copy()
    del(nodes[exclude])

    node_addrs = [hostname_to_address[hostname] for hostname in nodes]

    output = DeploymentInfo(
        job_id = job.job_id,
        user = job.user,
        dataset_path = job.dataset_path,
        node_list = nodes,
        node_addrs = node_addrs,
        engine = "vllm",
        model = job.model_name,
        tp_size = 4,
        pp_size = 2
    )

    return output

def setup():
    global hostname_to_address
    hostname_to_address = dict_from_file("hostnames")

# Main event loop invocation
async def main():
    setup()
    jobs_coro = asyncio.create_task(get_jobs())
    await jobs_coro

asyncio.run(main())