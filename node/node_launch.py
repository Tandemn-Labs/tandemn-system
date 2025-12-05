
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

# TODO: Look into cases where I should ray stop, it's not robust rn

# Stuff in this config has to be defined in envvar
# TODO: use dotenv lol
@dataclass
class GlobalConfig:
    hostname: str = ""
    listening_port: int = 12345

@dataclass
class LaunchConfig:
    ray_head_port: int = 54321 # Head node listening port of Ray cluster
    chain_comms_port: int = 54322 # For ZMQ comms within the chain
    comms_timeout: int = 30

config = GlobalConfig()
launch_config = LaunchConfig()

@dataclass
class LaunchOptions:
    node_list: List[str]
    node_addrs: List[str]
    engine: str
    model: str
    tp_size: int
    pp_size: int

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Some global variables
worker_sockets = None # Only found on head node
launch_status = None # If an engine is up right now, this will be a string

# `options` is a dictionary of launch options
# id and socket are because we use ZMQ ROUTER
async def launch_engine(options: LaunchOptions):

    global launch_status

    opt = options
    reply = {}

    # Some sanity checks
    if launch_status is not None: # Some deployment is already up
        reply = {"status": -1, "error": "A model is already deployed"}
        return reply

    if config.hostname not in opt.node_list:
        logger.error("Hostname %s is not included in node_list", config.hostname)
        reply = {
            "status": -1, 
            "error": f"Hostname {config.hostname} is not in node_list"
        }
        return reply
    
    match opt.engine:
        case "vllm":
            await launch_vllm(options)
            reply = {"status": 1}
            return reply
        case _:
            logger.error("Engine %s is not supported right now", opt.engine)
            reply = {
                "status": -1, 
                "error": f"Engine {opt.engine} not supported"
            }
            return reply


# Right now we use Ray to do multi-node vLLM, probably will change in the future
async def launch_vllm(options: LaunchOptions):

    global launch_config, config, worker_sockets, launch_status
    cfg, lc, opt = config, launch_config, options
    ray_stop = ["ray", "stop"]

    # Launch Ray cluster
    ctx = zmq.asyncio.Context()
    if config.hostname == options.node_list[0]: # Logic for head node
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
    
    # Only first node in chain launch vllm
    if cfg.hostname != options.node_list[0]:
        return

    #? Will probably move to a new file?
    from vllm import LLM
    llm = LLM(
        model=opt.model,
        tensor_parallel_size=opt.tp_size,
        pipeline_parallel_size=opt.pp_size,
        enforce_eager=True,
        distributed_executor_backend="ray"
    )

    # # Just for testing prompts
    # prompts = [
    #     "Tell me more about the idea of irreducible complexity!",
    #     "What is the difference between a spreadsheet and a database?",
    #     "What are your thoughts on the myth of Cupid and Psyche?"
    # ]

    # outputs = llm.generate(prompts)
    # print(outputs)

    # await ray_head_teardown()

# Listen for commands from Ray head node
# Loops infinitely (but not busy since async) till teardown command received
async def ray_worker_listen(rep_sock: zmq.asyncio.Socket):

    global config, launch_config, launch_status
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

                # Update global variables
                launch_status = None

            case _:
                status = {"ray_status": -1, "hostname": cfg.hostname}

        try:
            await asyncio.wait_for(rep_sock.send(msgpack.packb(status)), timeout=lc.comms_timeout)
        except Exception as e:
            logger.error(f"Failed to send reply to command: {e}")

# Teardown function invoked by head node
async def ray_head_teardown():

    global launch_config, worker_sockets, launch_status
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
    worker_sockets, launch_status = None, None

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
    central_sock.bind(f"tcp://*:{cfg.listening_port}")
    
    while True:
        msgb = await central_sock.recv()
        msg = msgpack.unpackb(msgb)
        print(msg)

        reply = {}
        match msg["command"]:
            case "launch": # Launch a model

                options = LaunchOptions(
                    node_list = msg["node_list"],
                    node_addrs = msg["node_addrs"],
                    engine = msg["engine"],
                    model = msg["model"],
                    tp_size = msg["tp_size"],
                    pp_size = msg["pp_size"]
                )

                reply = await asyncio.create_task(launch_engine(options))
            
            # Only head of Ray cluster should get this msg
            case "teardown":
                if worker_sockets is None:
                    reply  = {
                        "status": -1, 
                        "error": f"{cfg.hostname} is not head of Ray cluster"
                        }
                else:
                    reply = await ray_head_teardown()

            case "run": # Run a job
                pass

        
        await central_sock.send(msgpack.packb(reply))
        


def setup():
    global config, launch_config
    lc = launch_config
    
    config.hostname = os.environ.get("TD_HOSTNAME")
    config.listening_port = os.environ.get("TD_LISTENING_PORT")

    lc.ray_head_port = os.environ.get("TD_RAY_HEAD_PORT", lc.ray_head_port)


# Main event loop invocation
async def main():
    setup()
     
    # Main loop that listens to commands from central server
    main_task = asyncio.create_task(listen_to_central())

    await main_task

asyncio.run(main())
