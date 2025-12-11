
import asyncio
from dataclasses import dataclass
import logging
import os
import subprocess
from typing import List
import zmq
import zmq.asyncio
import msgpack
import traceback
import requests

from models.deployment_pb2 import Command, DeploymentInfo, JobInfo

# TODO: Look into cases where I should ray stop, it's not robust rn

# Stuff in this config has to be defined in envvar
# TODO: use dotenv lol
@dataclass
class GlobalConfig:
    hostname: str = ""
    listening_port: int = 12345

@dataclass
class StorageConfig:
    storage_server_url: str = "54.163.113.67"
    storage_server_port: int = 8000

@dataclass
class LaunchConfig:
    ray_head_port: int = 54321 # Head node listening port of Ray cluster
    chain_comms_port: int = 54322 # For ZMQ comms within the chain
    comms_timeout: int = 30

config = GlobalConfig()
launch_config = LaunchConfig()
storage_config = StorageConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Some global variables
worker_sockets = None # Only found on head node
launch_status = None # If an engine is up right now, this will be a string
deployment_info = None # Current launch options
chain_zmq_context = None

# `options` is a dictionary of launch options
# id and socket are because we use ZMQ ROUTER
async def launch_engine(dep: DeploymentInfo):

    global launch_status, deployment_info

    deployment_info = dep
    reply = {}

    # Some sanity checks
    if launch_status is not None: # Some deployment is already up
        reply = {"status": -1, "error": "A model is already deployed"}
        return reply

    if config.hostname not in dep.node_list:
        logger.error("Hostname %s is not included in node_list", config.hostname)
        reply = {
            "status": -1, 
            "error": f"Hostname {config.hostname} is not in node_list"
        }
        return reply
    
    match dep.WhichOneof("deployment_config"):
        case "vllm_config":
            await launch_vllm(dep)
            reply = {"status": 1}
            return reply
        case _:
            logger.error("Unable to get right deployment_config")
            reply = {
                "status": -1, 
                "error": "Unable to get right deployment_config"
            }
            return reply


# Right now we use Ray to do multi-node vLLM, probably will change in the future
async def launch_vllm(dep: DeploymentInfo):

    global launch_config, config, worker_sockets, launch_status, chain_zmq_context
    cfg, lc, opt = config, launch_config, dep
    ray_stop = ["ray", "stop"]

    # Launch Ray cluster
    ctx = zmq.asyncio.Context()
    chain_zmq_context = ctx
    if config.hostname == opt.node_list[0]: # Logic for head node
        ray_head = ["ray", "start", "--head", f"--port={lc.ray_head_port}"]
        subprocess.run(ray_head)

        # Initialize per worker socket and send ray start command
        worker_sockets = []
        send_futures = []
        for addr in opt.node_addrs[1:]:
            start_ray_sock = ctx.socket(zmq.REQ)
            start_ray_sock.connect(f"tcp://{addr}:{lc.chain_comms_port}")
            worker_sockets.append(start_ray_sock)
            
            commandb = msgpack.packb({"command": "start", "ray_head_port": lc.ray_head_port})
            send_futures.append(start_ray_sock.send(commandb))
        try:
            await asyncio.wait_for(asyncio.gather(*send_futures), timeout=lc.comms_timeout)
        except Exception as e:
            logger.error(f"Failed to send start ray command to workers. {e}")
            subprocess.run(ray_stop)
            return

        # Listen for worker's reply, which will include if start is successful
        confirmations = []
        for sock in worker_sockets:
            confirmations.append(sock.recv())
        try:
            results = await asyncio.wait_for(asyncio.gather(*confirmations), timeout=lc.comms_timeout)
            for worker in results:
                worker = msgpack.unpackb(worker)
                if worker["ray_status"] != 1:
                    logger.error(f"Failed to start ray at {worker["hostname"]}")
                    subprocess.run(ray_stop)
                    return
        except Exception as e:
            logger.error(f"Failed to receive reply to ray start. {e}")
            subprocess.run(ray_stop)
            traceback.print_exc()
            return
        
        # It's finally up, thank God
        logger.info("Ray cluster initialized.")
        launch_status = "vllm"

    else: # Logic for non-head nodes

        recv_sock = ctx.socket(zmq.REP)
        recv_sock.bind(f"tcp://*:{lc.chain_comms_port}")

        # Wait for socket handshake from head node
        ray_head_port = lc.ray_head_port
        try:
            msgb = await asyncio.wait_for(recv_sock.recv(), timeout=lc.comms_timeout)
            msg = msgpack.unpackb(msgb)
            ray_head_port = msg["ray_head_port"]
        except Exception as e:
            logger.error(f"Failed to receive ray start command: {e}")

        ray_worker = ["ray", "start", f"--address={opt.node_addrs[0]}:{ray_head_port}"]
        subprocess.run(ray_worker)
        status = {"ray_status": 1, "hostname": cfg.hostname}
        statusb = msgpack.packb(status)

        try:
            await asyncio.wait_for(recv_sock.send(statusb), timeout=lc.comms_timeout)
        except Exception as e:
            logger.error(f"Failed to send ray up status: {e}")
            subprocess.run(ray_stop)
            return
        
        # I've finally joined, thank God
        logger.info("Node has joined Ray cluster")
        launch_status = "vllm"

        # Start the coroutine to listen for more commands from Ray cluster head
        asyncio.create_task(ray_worker_listen(recv_sock))
    

# Listen for commands from Ray head node
# Loops infinitely (but not busy since async) till teardown command received
async def ray_worker_listen(rep_sock: zmq.asyncio.Socket):

    global config, launch_config, launch_status, chain_zmq_context, launch_options
    cfg, lc = config, launch_config

    teardown = False
    while not teardown: 
        commandb = await rep_sock.recv()
        command_msg = msgpack.unpackb(commandb)

        match command_msg["command"]:
            case "teardown":
                ray_teardown = ["ray", "stop"]
                subprocess.run(ray_teardown)
                teardown = True
                status = {"ray_status": 0, "hostname": cfg.hostname}

                await asyncio.wait_for(rep_sock.send(msgpack.packb(status)), timeout=lc.comms_timeout)

                # Update global variables
                launch_status, launch_options = None, None

                # Close ZMQ stuff
                rep_sock.close(linger=0)
                chain_zmq_context.term()
                return

            case _:
                status = {"ray_status": -1, "hostname": cfg.hostname}

        try:
            await asyncio.wait_for(rep_sock.send(msgpack.packb(status)), timeout=lc.comms_timeout)
        except Exception as e:
            logger.error(f"Failed to send reply to command: {e}")

# Teardown function invoked by head node
async def ray_head_teardown():

    global launch_config, worker_sockets, launch_status, chain_zmq_context, launch_options
    lc = launch_config
    reply = {}

    send_futures = []
    for sock in worker_sockets:
        cmdb = msgpack.packb({"command": "teardown"})
        send_futures.append(sock.send(cmdb))
    
    try:
        await asyncio.wait_for(asyncio.gather(*send_futures), timeout=lc.comms_timeout)
        reply = {"status": 1}
        logger.info("All the Ray kids know to kill themselves. Goodbye world.")
    except Exception as e:
        logger.error(f"Failed to send out ray teardown command: {e}")
        reply = {
            "status": -1, 
            "error": "Failed to send out ray teardown to worker nodes"
        }
    
    subprocess.run(["ray", "stop"])

    # Update global variables
    launch_status, launch_options = None, None
    
    # Close ZMQ stuff
    for sock in worker_sockets:
        sock.close(linger=0)
    chain_zmq_context.term()

    return reply


# Run a batch job
async def ray_head_run(job_info: JobInfo):

    global deployment_info
    user = job_info.user
    input = job_info.input

    # download the input file from storage server
    logger.info(f"Downloading input file from storage server: {input}")
    response_url = requests.get(f"http://{storage_config.storage_server_url}:{storage_config.storage_server_port}/storage/presign/download",
                            params={"user": user, "remote_path": input, "expires": 600})
    response_url.raise_for_status()
    presigned_url = response_url.json()["url"]

    # download the file from the presigned url
    response_file = requests.get(presigned_url)
    with open("input.jsonl", "wb") as f:
        f.write(response_file.content)

    logger.info(f"Downloaded input file to input.jsonl")

    # run the batch job
    run_batch = [
        "vllm", "run-batch", "-i", "input.jsonl", "-o", "output.jsonl",
        "--model", launch_options.model,
        "--max-model-len", "1000"
    ]
    subprocess.run(run_batch)

    # upload the output file to the storage server
    logger.info(f"Uploading output file to storage server")
    response_upload = requests.post(
        f"http://{storage_config.storage_server_url}:{storage_config.storage_server_port}/storage/presign/upload",
        data={
            "user": user,
            "remote_path": "output.jsonl",
            "expires": 600
        }
    )
    response_upload.raise_for_status()
    presigned_url = response_upload.json()["url"]

    # upload the file to the url
    with open("output.jsonl", "rb") as f:
        response_upload = requests.put(presigned_url, data=f)
    response_upload.raise_for_status()
    logger.info(f"Uploaded output file to storage server")

   # delete input and output files
   #blocking call, remove this later
    if os.path.exists("input.jsonl"):
        os.remove("input.jsonl")
    if os.path.exists("output.jsonl"):
        os.remove("output.jsonl")

    logger.info(f"Job completed successfully")
    
    reply = {"status": 1}

    return reply


# A forever alive coro that listens to model deployment and jobs commands
# Actl killing this whole process will prob be done with a container orchestration service
async def listen_to_central():

    global config, launch_status, worker_sockets
    cfg = config

    ctx = zmq.asyncio.Context()
    listen_address = f"tcp://*:{cfg.listening_port}"
    print("Listening on ", listen_address)
    central_sock = ctx.socket(zmq.REP)
    central_sock.bind(listen_address)
    
    while True:
        msgb = await central_sock.recv()
        command = Command()
        command.ParseFromString(msgb)
        print(command)

        reply = {}
        match command.action:
            case Command.Action.LAUNCH: # Launch a model
                print("matched")

                reply = await asyncio.create_task(
                    launch_engine(command.deployment_info)
                )
            
            # Only head of Ray cluster should get this msg
            case Command.Action.TEARDOWN:
                if worker_sockets is None:
                    reply  = {
                        "status": -1, 
                        "error": f"{cfg.hostname} is not head of Ray cluster"
                        }
                else:
                    reply = await ray_head_teardown()

            case Command.Action.RUN: # Run a job
                if worker_sockets is None:
                    reply  = {
                        "status": -1, 
                        "error": f"{cfg.hostname} is not head of Ray cluster"
                        }
                else:
                    reply = await ray_head_run(command.job_info)

                pass

        
        await central_sock.send(msgpack.packb(reply))
        


async def setup():
    global config, launch_config, storage_config
    lc = launch_config
    sc = storage_config
    
    config.hostname = os.environ.get("TD_HOSTNAME")
    config.listening_port = os.environ.get("TD_LISTENING_PORT")
    # need to know how to automatically get this
    # do we have the configs?
    sc.storage_server_url = os.environ.get("STORAGE_SERVER_URL")
    sc.storage_server_port = os.environ.get("STORAGE_SERVER_PORT")

    lc.ray_head_port = os.environ.get("TD_RAY_HEAD_PORT", lc.ray_head_port)


# Main event loop invocation
async def main():
    await setup()
     
    # Main loop that listens to commands from central server
    main_task = asyncio.create_task(listen_to_central())

    await main_task

asyncio.run(main())
