
from dataclasses import dataclass
import logging
import os
import subprocess
import time
from typing import List
import zmq
import msgpack

@dataclass
class GlobalConfig:
    hostname: str = ""

@dataclass
class LaunchConfig:
    ray_head_port: int = 54321
    chain_comms_port: int = 54322
    ray_setup_timeout: int = 60

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

# `options` is a dictionary of launch options
def launch_engine(options):

    opt = options

    if config.hostname not in opt.node_list:
        logger.error("Hostname %s is not included in node_list", config.hostname)
        return
    
    match opt.engine:
        case "vllm":
            launch_vllm(options)
        case _:
            logger.error("Engine %s is not supported right now", opt.engine)

# Right now we use Ray to do multi-node vLLM, probably will change in the future
def launch_vllm(options: LaunchConfig):

    global launch_config, config
    cfg, lc, opt = config, launch_config, options

    # Launch Ray cluster
    ctx = zmq.Context()
    if config.hostname == options.node_list[0]: # Logic for head node
        ray_head = ["ray", "start", "--head", f"--port={lc.ray_head_port}"]
        subprocess.run(ray_head)

        listen = ctx.socket(zmq.REP)
        listen.bind(f"tcp://*:{lc.chain_comms_port}")
        confirmations = []
        start_time = time.monotonic()

        while (time.monotonic() - start_time) < lc.ray_setup_timeout and len(confirmations) < (len(opt.node_list) - 1):
            try:
                msg = listen.recv(flags=zmq.NOBLOCK)
                results = msgpack.unpackb(msg)
                if results["ray_status"]:
                    confirmations.append(results["hostname"])

            except zmq.ZMQError:
                # Ignore ZMQError since it's triggered by no message found on non-blocking
                continue
        
        # Check ray up or timeout
        if len(confirmations) != (len(opt.node_list) - 1):
            logger.error("Timeout of %d seconds reached. Ray cluster not up", lc.ray_setup_timeout)
            logger.error("List of nodes with confirmation: %s", confirmations)
            return
        else:
            logger.info("Ray cluster initialized. Confirmations: %s", confirmations)

    else: # Logic for non-head nodes
        ray_worker = ["ray", "start", f"--address={opt.node_addrs[0]}:{lc.ray_head_port}"]
        subprocess.run(ray_worker)

        send_sock = ctx.socket(zmq.REQ)
        send_sock.connect(f"tcp://{opt.node_addrs[0]}:{lc.chain_comms_port}")
        status = {"ray_status": 1, "hostname": cfg.hostname}
        status_bytes = msgpack.packb(status)
        send_sock.send(status_bytes, flags=zmq.NOBLOCK)
    
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

    # Just for testing prompts
    prompts = [
        "Tell me more about the idea of irreducible complexity!",
        "What is the difference between a spreadsheet and a database?",
        "What are your thoughts on the myth of Cupid and Psyche?"
    ]

    outputs = llm.generate(prompts)
    print(outputs)

# # Called by the head node
# def teardown_ray(options: LaunchOptions):

#     global launch_config
#     opt, lc = options, launch_config

#     ctx = zmq.Context()
#     for addr in opt.node_addrs[1:]:
#         kill_sock = ctx.socket(zmq.REQ)
#         kill_sock.connect(f"tcp://{addr}:{lc.chain_comms_port}")

#     pass

def setup():
    global config, launch_config
    lc = launch_config
    
    config.hostname = os.environ.get("TD_HOSTNAME")

    lc.ray_head_port = os.environ.get("TD_RAY_HEAD_PORT", lc.ray_head_port)
    

def main():
    setup()

    opt = LaunchOptions(
        node_list = ["test1", "test2"],
        node_addrs = ["",""],
        engine = "vllm",
        model = "facebook/opt-125m",
        tp_size = 4,
        pp_size = 2
    )

    launch_engine(opt)

main()
