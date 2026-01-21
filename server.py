import uuid

import requests
import sky
from typing import Union
from fastapi import FastAPI, Form
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from storage.storage_server import storage_backend
from utils.utils import split_uri, update_template, update_yaml_file
from sky import core as sky_core

##### Global variables
YAML_OUTPUT = "temp/output.yaml"

# To run the server:
# uvicorn server:app --reload
# or
# python server.py

app = FastAPI(
    title="Tandemn Server",
    description="API for receiving job requests",
)


@app.post("/submit/batch")
async def submit_batch(request: BatchedRequest):
    """
    Submit a batched inference job request.

    Receives a BatchedRequest with job configuration and returns
    confirmation of receipt.
    """

    launch_config = real_magic(request)
    print(launch_config)
    
    match launch_config.engine:
        case "vllm":
            await sp_launch_vllm_batch(request, launch_config)

@app.post("/submit/online")
async def submit_online(request: OnlineServingRequest):
    """
    Submit an online inference job request.

    Receives a OnlineServingRequest and returns a confirmation of receipt.
    """
    launch_config = real_magic(request)
    print(launch_config)

    match launch_config.engine:
        case "vllm":
            endpoint_url = await sp_launch_vllm_online(request, launch_config)
            return {
                "status": "success",
                "job_id": launch_config.decision_id,
                "endpoint": endpoint_url,
                "model": request.model_name,
                "message": f"vLLM server launched at {endpoint_url}"
            }


async def sp_launch_vllm_batch(request: BatchedRequest, config: MagicOutput):

    replace_run_dict = replace_run_vllm(request, config)
    run_string = update_template("templates/vllm_run", replace_run_dict)

    dirname, _ = split_uri(request.input_file)


    replace_yaml = {
        "name": config.decision_id,
        "num_nodes": config.num_nodes,
        "resources.instance_type": config.instances,
        "run": run_string,
        "file_mounts./data.source": dirname
    }
    update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    task = sky.Task.from_yaml(YAML_OUTPUT)
    result_id = sky.launch(task, cluster_name=config.decision_id, down=True)
    job_id, _ = sky.stream_and_get(result_id, follow=True)
    sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)


async def sp_launch_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    """Launch persistant vllm online deployment"""
    replace_run_dict = replace_run_vllm_online(request, config)
    run_string = update_template("templates/vllm_run_online", replace_run_dict)
    
    replace_yaml = {
        "name": config.decision_id,
        "num_nodes": config.num_nodes,
        "resources.instance_type": config.instances,
        "resources.ports": "8001",
        "run": run_string,
    }
    update_yaml_file("templates/vllm_online.yaml", replace_yaml, YAML_OUTPUT)
    task = sky.Task.from_yaml(YAML_OUTPUT)
    result_id = sky.launch(task,cluster_name=config.decision_id, down=False) # do not LET IT DIE!
    
    # return the public IP of the deployment
    # cluster_info = sky.status(cluster_names=[config.decision_id]) 
    # cluster_info = sky_core.status(cluster_names=[config.decision_id])
    job_id, handle = sky.stream_and_get(result_id, follow=True)
    sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)

    public_ip = handle.head_ip 

    endpoint_url = f"http://{public_ip}:8001/v1"

    print(f"vLLM server launched at {endpoint_url}")

    url = f"http://{public_ip}:8001/v1/models"
    response = requests.get(url, timeout=5)
    if response.status_code == 200: #do sth here for a valid API up and sth else otherwise
        print(f"vLLM server API is up at {endpoint_url}")
        return endpoint_url
    else:
        raise Exception(f"vLLM server API is not up at {endpoint_url}")

def replace_run_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    replace = {}
    replace["model"] = request.model_name
    replace["tp"] = config.tp_size
    replace["pp"] = config.pp_size
    replace["host"] = "0.0.0.0"  # Bind to all interfaces (allows external access)
    replace["port"] = "8001"     # hardcode the port to 8002
    
    if (
        request.vllm_specific_config is not None
        and request.vllm_specific_config.speculative_config is not None
    ):
        prefix = "--speculative-config."
        spec_config = request.vllm_specific_config.speculative_config.model_dump(
            exclude_none=True
        )

        spec_string = ""
        for key, value in spec_config.items():
            string = prefix + key + " " + str(value)
            spec_string += string + " "
        
        spec_string = spec_string.rstrip(" ")
        replace["additional_params"] = spec_string
    
    else:
        replace["additional_params"] = ""
    
    return replace

def replace_run_vllm(request: BatchedRequest, config: MagicOutput):
    replace = {}

    replace["model"] = request.model_name
    replace["tp"] = config.tp_size
    replace["pp"] = config.pp_size

    _, input_file = split_uri(request.input_file)
    output = request.output_file
    replace["input_file"] = "/data/" + input_file
    replace["output_file"] = "/data/" + output

    if (
        request.vllm_specific_config is not None
        and request.vllm_specific_config.speculative_config is not None
    ):
        prefix = "--speculative-config."
        spec_config = request.vllm_specific_config.speculative_config.model_dump(
            exclude_none=True
        )

        spec_string = ""
        for key, value in spec_config.items():
            string = prefix + key + " " + str(value)
            spec_string += string + " "
        
        spec_string = spec_string.rstrip(" ")
        replace["additional_params"] = spec_string
    
    else:
        replace["additional_params"] = ""
    
    return replace


### magic.py placeholder
def real_magic(request: Union[BatchedRequest, OnlineServingRequest]) -> MagicOutput:
    return MagicOutput(
        decision_id="mo-" + str(uuid.uuid4()),
        engine="vllm",
        instances="g6e.xlarge",
        num_nodes=1,
        tp_size=1,
        pp_size=1,
    )

##### Storage stuff #####
@app.post("/storage/presigned_upload")
async def presign_upload(
    remote_path: str = Form(...), user: str = Form(...), expires: int = Form(600)
):
    payload = await storage_backend.presigned_upload(remote_path, user, expires)
    return {"status": "success", **payload}


@app.get("/storage/presigned_download")
async def presign_download(user: str, remote_path: str, expires: int = 600):
    payload = await storage_backend.presigned_download(remote_path, user, expires)
    return {"status": "success", **payload}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=26336)
