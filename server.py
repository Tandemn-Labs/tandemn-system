import uuid
import requests
import sky
from fastapi import FastAPI, Form
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from storage.storage_server import storage_backend
from tracking.tracking import JobRecord, VPCQuotaTracker
from utils.utils import split_uri, update_template, update_yaml_file
from typing import Dict, Tuple, Union, Optional, Literal
from threading import Lock
from orca import JobSpec, load_all_perfdb_files, load_quota_csv, JobState
from orca import (
    get_num_params_from_text,
    enumerate_gpus_and_candidates,
    llm_choose_config_from_candidates,
)
from orca import choose_and_apply_llm_plan, load_c_pmi_model
import threading
import time
import re
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI(
    title="Tandemn Server",
    description="API for receiving job requests",
)

##### Global variables
YAML_OUTPUT = "temp/output.yaml"
OPENROUTER_API_KEY = os.environ("TD_OPENROUTER_KEY")


class JobTracker:
    def __init__(self):
        self.jobs: Dict[str, JobRecord] = {}
        self.lock = Lock()

    def build_job_state_batched(self, req: BatchedRequest, config: MagicOutput):
        # job_id = f"job-{int(time.time())}" # deprecate this
        return JobState(
            spec=JobSpec(
                job_id=config.decision_id,
                model_name=req.model_name,
                num_lines=req.num_lines or 1,
                avg_input_tokens=req.avg_input_tokens
                if req.avg_input_tokens is not None
                else 4096,
                avg_output_tokens=req.avg_output_tokens
                if req.avg_output_tokens is not None
                else 2048,
                slo_hours=req.slo_deadline_hours or 12,
                region="us-east-1",
                market="spot",
            ),
            submitted_at=time.time(),
            # add other fields when they will be decided by the Orca
        )

    def add(self, job_state: JobState):
        job_record = JobRecord(state=job_state, created_at=time.time())
        with self.lock:
            self.jobs[job_state.spec.job_id] = job_record
        return job_record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self.lock:
            return self.jobs.get(job_id)

    def update_progress(self, job_id: str, progress_frac: float):
        with self.lock:
            job_record = self.jobs.get(job_id)
            if job_record:
                job_record.state.progress_frac = progress_frac
                job_record.last_updated_at = time.time()
            return job_record

    def update_status(
        self,
        job_id: str,
        status: Literal[
            "queued", "launching", "running", "succeeded", "failed", "cancelled"
        ],
    ):
        with self.lock:
            job_record = self.jobs.get(job_id)
            if job_record:
                job_record.status = status
                job_record.last_updated_at = time.time()
            return job_record

    def set_head_ip(self, job_id: str, head_ip: str):
        with self.lock:
            rec = self.jobs.get(job_id)
            if rec:
                rec.head_ip = head_ip
                rec.last_updated_at = time.time()
            return rec

    def set_endpoint_url(self, job_id: str, endpoint_url: str):
        with self.lock:
            rec = self.jobs.get(job_id)
            if rec:
                rec.endpoint_url = endpoint_url
                rec.last_updated_at = time.time()
            return rec


class OrcaOrchestrator:
    def __init__(
        self,
        quota_tracker: VPCQuotaTracker,
        job_tracker: JobTracker,
        perfdb_dir="./perf_db",
        quota_csv="./temp/aws_gpu_quota_by_region.csv",
    ):
        self.quota_tracker = quota_tracker
        self.job_tracker = job_tracker
        self.perf_files = load_all_perfdb_files(perfdb_dir)
        self.quota_df = load_quota_csv(quota_csv)
        self.c_pmi_model, self.c_pmi_tokenizer = load_c_pmi_model()

    def decide_on_allocation(self, job_state: JobState) -> Tuple[str, int]:
        model_size_b = get_num_params_from_text(job_state.spec.model_name)
        candidates = enumerate_gpus_and_candidates(
            user_model_size_b=model_size_b,
            num_lines=job_state.spec.num_lines,
            avg_input_tokens=job_state.spec.avg_input_tokens,
            avg_output_tokens=job_state.spec.avg_output_tokens,
            slo_hours=job_state.spec.slo_hours,
            region=job_state.spec.region,
            market=job_state.spec.market,
        )
        # math solver
        # candidates_sorted = sorted(candidates, key=lambda c: (c["replicas"], c["vcpu_needed"], c["runtime_hours"]))
        math_cfg = min(
            candidates,
            key=lambda c: (c["replicas"], c["vcpu_needed"], c["runtime_hours"]),
        )
        print(
            f"[Math] tp={math_cfg['tp']} pp={math_cfg['pp']} r={math_cfg['replicas']} gpu-h={math_cfg['gpu_time']:.2f}"
        )
        # llm solver
        mimo_cfg = llm_choose_config_from_candidates(
            job=job_state,
            candidates=candidates,
            model_id="xiaomi/mimo-v2-flash:free",
            openrouter_api_key=OPENROUTER_API_KEY,
            advisor_name="XiaomiMimoAdvisor",
            top_k=20,
        )
        if mimo_cfg:
            print(
                f"[XiaomiMimo Advisor] Picks: tp={mimo_cfg['tp']} pp={mimo_cfg['pp']} r={mimo_cfg['replicas']}"
            )
        devstral_cfg = llm_choose_config_from_candidates(
            job=job_state,
            candidates=candidates,
            model_id="mistralai/devstral-2512:free",
            openrouter_api_key=OPENROUTER_API_KEY,
            advisor_name="DevstralAdvisor",
            top_k=20,
        )
        if devstral_cfg:
            print(
                f"[Devstral Advisor] Picks: tp={devstral_cfg['tp']} pp={devstral_cfg['pp']} r={devstral_cfg['replicas']}"
            )
        plans = [
            ("Math", math_cfg),
            ("XiaomiMimo", mimo_cfg),
            ("Devstral", devstral_cfg),
        ]
        plan_labels = [name for name, _ in plans]
        best_label, probs, chosen_cfg = choose_and_apply_llm_plan(
            job_state, plans, plan_labels, self.c_pmi_model, self.c_pmi_tokenizer
        )
        return chosen_cfg


###################################################################################################

##################################### vLLM Metrics Parser #########################################
# This will be replaced with a proper mechanism (prometheus), but for now, this is it.
_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{(?P<labels>[^}]*)\})?"  # optional {k="v",...}
    r"\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def _parse_labels(labels_blob: str) -> dict[str, str]:
    # Very small parser; works for the usual key="value" labels vLLM emits.
    out: dict[str, str] = {}
    if not labels_blob:
        return out
    for part in labels_blob.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def _sum_metric(
    text: str, metric_name: str, *, where: dict[str, str] | None = None
) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        if m.group("name") != metric_name:
            continue
        labels = _parse_labels(m.group("labels") or "")
        if where and any(labels.get(k) != v for k, v in where.items()):
            continue
        total += float(m.group("value"))
    return total


def get_vllm_progress(
    endpoint_url: str, *, model_name: str | None = None
) -> tuple[int, int]:
    r = requests.get(endpoint_url + "/metrics", timeout=5)
    r.raise_for_status()
    text = r.text

    # Optional: only count metrics for a specific model
    filt = {"model_name": model_name} if model_name else None

    running = _sum_metric(text, "vllm:num_requests_running", where=filt)
    waiting = _sum_metric(text, "vllm:num_requests_waiting", where=filt)

    # "done requests" = sum across finished_reason labels
    done = _sum_metric(text, "vllm:request_success_total", where=filt)

    queued = running + waiting
    return int(done), int(queued)


def poll_job_progress(
    job_id: str,
    endpoint_url: str,
    total_prompts: int,
    tracker: JobTracker,
    interval_sec=1,
):
    tracker.update_status(job_id, "installing")
    while True:
        try:
            done, queued = get_vllm_progress(endpoint_url)
            tracker.update_status(job_id, "running")
            progress = min(1.0, done / max(1, total_prompts))
            tracker.update_progress(job_id, progress)
            if done >= total_prompts:
                tracker.update_status(job_id, "succeeded")
                break
        except Exception:
            # keep polling, simple prototype
            pass
        time.sleep(interval_sec)


def jobtracker_snapshot(tracker: JobTracker) -> dict:
    with tracker.lock:
        snap = {}
        for job_id, rec in tracker.jobs.items():
            snap[job_id] = {
                "status": rec.status,
                "progress": round(rec.state.progress_frac, 4),
                "head_ip": rec.head_ip,
                "endpoint_url": rec.endpoint_url,
                "last_updated_at": rec.last_updated_at,
                "num_lines": rec.state.spec.num_lines,
            }
        return snap


def log_jobtracker_loop(tracker: JobTracker, interval_sec: int = 0.5):
    prev = None
    while True:
        snap = jobtracker_snapshot(tracker)
        print("\n[JobTracker] snapshot:")
        for job_id, d in snap.items():
            print(
                f"  - {job_id} status={d['status']} "
                f"progress={d['progress'] * 100:.1f}% "
                f"lines={d['num_lines']} head_ip={d['head_ip']} "
                f"endpoint={d['endpoint_url']}"
            )
        time.sleep(interval_sec)


###################################################################################################


@app.on_event("startup")
async def startup_event():
    print("[QuotaTracker] Starting up")
    # there is something called as app.state that shares these variables across requests
    # when the server is being run on multiple threads and multiple ppl are calling it
    app.state.quota_tracker = VPCQuotaTracker()
    app.state.job_tracker = JobTracker()
    app.state.orca_orchestrator = OrcaOrchestrator(
        app.state.quota_tracker, app.state.job_tracker
    )


def get_quota_tracker():
    return app.state.quota_tracker


def get_job_tracker():
    return app.state.job_tracker


def get_orca_orchestrator():
    return app.state.orca_orchestrator


@app.get("/quota/status")
async def quota_status():
    """Get current quota usage summary."""
    tracker = get_quota_tracker()
    summary = tracker.status_summary()
    return {"status": "success", "quota_usage": summary.to_dict(orient="records")}


@app.post("/submit/batch")
async def submit_batch(request: BatchedRequest):
    """
    Submit a batched inference job request.

    Receives a BatchedRequest with job configuration and returns
    confirmation of receipt.
    """

    launch_config = real_magic(request)
    print(launch_config)

    jt = app.state.job_tracker
    job_state = jt.build_job_state_batched(request, launch_config)
    jt.add(job_state)
    jt.update_status(launch_config.decision_id, "queued")

    # launch the job
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
                "message": f"vLLM server launched at {endpoint_url}",
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
        "file_mounts./data.source": dirname,
    }
    update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    task = sky.Task.from_yaml(YAML_OUTPUT)
    result_id = sky.launch(task, cluster_name=config.decision_id, down=True)
    job_id, handle = sky.stream_and_get(result_id, follow=True)

    # get the head IP
    head_ip = handle.head_ip
    endpoint_url = f"http://{head_ip}:8001/"

    jt = get_job_tracker()
    jt.set_head_ip(config.decision_id, head_ip)
    jt.set_endpoint_url(config.decision_id, endpoint_url)
    jt.update_status(config.decision_id, "launching")

    threading.Thread(
        target=poll_job_progress,
        args=(config.decision_id, endpoint_url.rstrip("/"), request.num_lines or 1, jt),
        daemon=False,
    ).start()

    # sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)
    threading.Thread(
        target=sky.tail_logs,
        kwargs={"cluster_name": config.decision_id, "job_id": job_id, "follow": True},
        daemon=False,
    ).start()


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
    result_id = sky.launch(
        task, cluster_name=config.decision_id, down=False
    )  # do not LET IT DIE!

    # return the public IP of the deployment
    # cluster_info = sky.status(cluster_names=[config.decision_id])
    # cluster_info = sky_core.status(cluster_names=[config.decision_id])
    job_id, handle = sky.stream_and_get(result_id, follow=True)
    sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)

    public_ip = handle.head_ip

    endpoint_url = f"http://{public_ip}:8001"

    print(f"vLLM server launched at {endpoint_url}")

    url = f"http://{public_ip}:8001/v1/models"
    response = requests.get(url, timeout=5)
    if (
        response.status_code == 200
    ):  # do sth here for a valid API up and sth else otherwise
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
    replace["port"] = "8001"  # hardcode the port to 8002

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


if __name__ == "__main__":
    import asyncio

    asyncio.run(startup_event())

    # Fake request
    req = BatchedRequest(
        user_id="test",
        input_file="s3://tandemn-orca/orange/test.jsonl",
        output_file="output_out.txt",
        num_lines=5,
        description="test run",
        task_type="batch",
        task_priority="low",
        model_name="Qwen/Qwen3-0.6B",
        engine="vllm",
        slo_mode="batch",
        placement="vpc",
    )

    asyncio.run(submit_batch(req))
    log_jobtracker_loop(app.state.job_tracker, interval_sec=0.5)
