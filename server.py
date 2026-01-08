import uuid
from fastapi import FastAPI
from models.requests import BatchedRequest
from models.resources import MagicOutput
from utils.yaml import update_yaml_file

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

    match launch_config.engine:
        case "vllm":
            pass


async def sp_launch_vllm(request: BatchedRequest, config: MagicOutput):
    replace_dict = {
        "name": config.decision_id,
        "num_nodes": config.num_nodes,
        "resources.instance_type": config.instances,
    }
    update_yaml_file("yamls/vllm.yaml", replace_dict, "temp/output.yaml")

    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=26336)


### magic.py placeholder


def real_magic(request: BatchedRequest) -> MagicOutput:
    return MagicOutput("mo" + uuid.uuid4(), "vllm", "g4dn.12xlarge", 4, 4, 4)
