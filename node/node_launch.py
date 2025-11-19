
from dataclasses import dataclass
import logging
import os
import subprocess
from typing import List

@dataclass
class GlobalConfig:
    hostname: str = ""

@dataclass
class LaunchConfig:
    ray_head_port: int = 54321

config = GlobalConfig()
launch_config = LaunchConfig()

@dataclass
class LaunchOptions:
    node_list: List[str]
    engine: str
    head_address: str
    model: str
    tp_size: int
    pp_size: int

logging.getLogger(__name__)

# `options` is a dictionary of launch options
def launch_engine(options):

    opt = options

    if config.hostname not in opt.node_list:
        logging.error("Hostname %s is not included in node_list", config.hostname)
        return
    
    match opt.engine:
        case "vllm":
            launch_vllm(options)
        case _:
            logging.error("Engine %s is not supported right now", opt.engine)

# Right now we use Ray to do multi-node vLLM, probably will change in the future
def launch_vllm(options):

    lc, opt = launch_config, options

    # Launch Ray cluster
    if config.hostname == options.node_list[0]:
        ray_head = ["ray", "start", "--head", f"--port={lc.ray_head_port}"]
        subprocess.run(ray_head)
    else:
        ray_worker = ["ray", "start", f"--address={opt.head_address}:{lc.ray_head_port}"]
        subprocess.run(ray_worker)
    
    # TODO: Does Ray let you programmatically manage cluster?

    # Only first node in chain launch vllm
    if config.hostname != options.node_list[0]:
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
        "Tell me more about the idea of irreducible complexity",
        "What is the difference between a spreadsheet and a database?",
        "What are your thoughts on the myth of Cupid and Psyche?"
    ]

    outputs = llm.generate(prompts)
    print(outputs)


def setup():
    global config, launch_config
    lc = launch_config
    
    config.hostname = os.environ.get("TD_HOSTNAME")

    lc.ray_head_port = os.environ.get("TD_RAY_HEAD_PORT", lc.ray_head_port)
    

def main():
    setup()

    opt = LaunchOptions(
        node_list = [],
        engine = "vllm",
        head_address = "",
        model = "facebook/opt-125m",
        tp_size = 4,
        pp_size = 2
    )

    launch_engine(opt)

main()
