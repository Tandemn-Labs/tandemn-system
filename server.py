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

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple
from threading import Lock
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


@dataclass
class VPCQuotaTracker:
    quota_csv_file: str = "temp/aws_gpu_quota_by_region.csv"
    quota_df: pd.DataFrame = field(init=False)
    # Key: (region, market, family_type) → vcpu_in_use
    used_vcpu: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self.reload_quota()
    
    def reload_quota(self):
        """Reload the quota from CSV (called after Refresh)"""
        print(f"Reloading Quota from {self.quota_csv_file}")
        self.quota_df = pd.read_csv(self.quota_csv_file)
        print(f"[QuotaTracker] Loaded {len(self.quota_df)} instance types")
    
    def get_baseline_quota(self, region:str, market:str, family_type:str):
        """Get the AWS Quota limit for family type"""
        col = f"{region}_{market}"
        family_rows = self.quota_df[self.quota_df["Family_Type"] == family_type]
        if col not in family_rows:
            return 0
        return family_rows[col].iloc[0]

    def get_used_vcpu(self, region:str, market:str, family_type:str):
        """Get the vCPU in use for the given region, market, and family type"""
        # with self.lock:
        print("region", region)
        print("market", market)
        print("family_type", family_type)
        return self.used_vcpu.get((region, market, family_type), 0)
    
    def get_available(self,region:str, market:str, family_type:str):
        """combine the baseline quota and the used vCPU to get the available vCPU"""
        baseline_quota = self.get_baseline_quota(region, market, family_type)
        used_vcpu = self.get_used_vcpu(region, market, family_type)
        return baseline_quota - used_vcpu
    
    def reserve(self, region:str, market:str, family_type:str, vcpu:int):
        """Reserve vCPU, returns True if successful, False otherwise"""
        with self.lock:
            available = self.get_available(region, market, family_type)

            if vcpu > available:
                print(f"[QuotaTracker] Not enough quota for {region}, {market}, {family_type}")
                return False
            self.used_vcpu[(region, market, family_type)] = self.used_vcpu.get((region, market, family_type), 0) + vcpu
            print(f"[QuotaTracker] Reserved {vcpu} vCPU for {region}, {market}, {family_type}")
            return True
    
    def release(self, region:str, market:str, family_type:str, vcpu:int):
        """Release vCPU quota"""
        with self.lock:
            old = self.get_used_vcpu(region, market, family_type)
            self.used_vcpu[(region, market, family_type)] = max(0, old - vcpu)
            print(f"[QuotaTracker] Released {vcpu} vCPU for {region}, {market}, {family_type}")
    
    def get_family_for_instance(self, instance_type:str):
        """Get the family for the given instance example - g6e.xlarge -> G Family"""
        row = self.quota_df[self.quota_df["Instance_Type"] == instance_type]
        return row["Family_Type"].iloc[0]
    
    def reserve_for_instance(self, region:str, market:str, instance_type:str, num_instances:int):
        """Convenience: reserve by instance type (auto-looks up family + vCPU)."""
        row = self.quota_df[self.quota_df["Instance_Type"] == instance_type]
        if row.empty:
            raise ValueError(f"Instance type {instance_type} not found in quota CSV")
        family_type = row["Family_Type"].iloc[0]
        vcpu = row["vCPU"].iloc[0]
        return self.reserve(region, market, family_type, vcpu * num_instances)

    def status_summary(self) -> pd.DataFrame:
        """Get a human-readable summary of quota usage."""
        rows = []
        for (region, market, family), used in self.used_vcpu.items():
            baseline = self.get_baseline_quota(region, market, family)
            rows.append({
                "Region": region,
                "Market": market,
                "Family": family,
                "Baseline": baseline,
                "Used": used,
                "Available": baseline - used,
                "Usage %": f"{(used/baseline*100):.1f}%" if baseline > 0 else "N/A"
            })
        return pd.DataFrame(rows)


@app.on_event("startup")
async def startup_event():
    print("[QuotaTracker] Starting up")
    tracker = VPCQuotaTracker()


@app.get("/quota/status")
async def quota_status():
    """Get current quota usage summary."""
    tracker = get_quota_tracker()
    summary = tracker.status_summary()
    return {
        "status": "success",
        "quota_usage": summary.to_dict(orient="records")
    }


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

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=26336)

    # tracker = VPCQuotaTracker()

    # # Reserve some quota
    # tracker.reserve_for_instance("us-east-1", "spot", "g6e.xlarge", num_instances=10)
    # tracker.reserve_for_instance("us-east-1", "spot", "g6e.12xlarge", num_instances=2)

    # # Check status
    # print(tracker.status_summary())

    # # Release
    # tracker.release("us-east-1", "spot", "G", 40)  # g6e.xlarge has 4 vCPU
    # print(tracker.status_summary())

if __name__ == "__main__":
    print("="*60)
    print("Testing VPCQuotaTracker")
    print("="*60)
    
    tracker = VPCQuotaTracker()
    
    # Test 1: Reserve quota
    print("\n[Test 1] Reserving quota...")
    tracker.reserve_for_instance("us-east-1", "spot", "g6e.xlarge", num_instances=10)
    tracker.reserve_for_instance("us-east-1", "spot", "g6e.12xlarge", num_instances=2)
    
    # Test 2: Check status
    print("\n[Test 2] Current status:")
    print(tracker.status_summary())
    
    # Test 3: Check available
    print("\n[Test 3] Available quota for G family in us-east-1/spot:")
    avail = tracker.get_available("us-east-1", "spot", "G")
    baseline = tracker.get_baseline_quota("us-east-1", "spot", "G")
    print(f"  Baseline: {baseline} vCPU")
    print(f"  Available: {avail} vCPU")
    
    # Test 4: Release
    print("\n[Test 4] Releasing 40 vCPU...")
    tracker.release("us-east-1", "spot", "G", 40)
    print(tracker.status_summary())
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60)