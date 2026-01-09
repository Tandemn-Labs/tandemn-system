import uuid
import sky

from fastapi import FastAPI
from models.requests import BatchedRequest
from models.resources import MagicOutput
from utils.utils import update_template, update_yaml_file

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
            await sp_launch_vllm(request, launch_config)


async def sp_launch_vllm(request: BatchedRequest, config: MagicOutput):
    replace_run_dict = replace_run_vllm(request, config)
    run_string = update_template("templates/vllm_run", replace_run_dict)
    print(run_string)

    replace_yaml = {
        "name": config.decision_id,
        "num_nodes": config.num_nodes,
        "resources.instance_type": config.instances,
        "workdir": "files",
        "run": run_string
    }
    update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    task = sky.Task.from_yaml(YAML_OUTPUT)
    print(task)
    result = sky.launch(task, cluster_name=config.decision_id, down=True)
    print(result)


def replace_run_vllm(request: BatchedRequest, config: MagicOutput):
    replace = {}

    replace["model"] = request.model_name
    replace["input_file"] = request.input_file
    replace["output_file"] = request.output_file

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
        print(spec_string)
    
    else:
        replace["additional_params"] = ""
    
    return replace


### magic.py placeholder
def real_magic(request: BatchedRequest) -> MagicOutput:
    return MagicOutput(
        decision_id="mo-" + str(uuid.uuid4()),
        engine="vllm",
        instances="g6e.xlarge",
        num_nodes=1,
        tp_size=1,
        pp_size=1,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=26336)
