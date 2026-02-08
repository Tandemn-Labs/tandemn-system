from contextlib import asynccontextmanager
import asyncio
import logging
import math
import uuid
import requests
import sky
from fastapi import FastAPI, Form
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from placement.aws_magic import AWSAllocation
from storage.storage_server import storage_backend
from tracking.tracking import JobRecord, JobSpec, JobState
from chunk_queue.redis_pool import get_redis, close_redis
from chunk_queue.chunk_queue import ChunkQueueManager, ChunkInfo
from utils.utils import split_uri, update_template, update_yaml_file
from typing import Dict, List, Union, Optional, Literal
from threading import Lock
import time
import re
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

##### Global variables
YAML_OUTPUT = "temp/output.yaml"
OPENROUTER_API_KEY = os.environ.get("TD_OPENROUTER_KEY")
CONTROL_PLANE_IP = os.environ.get("CONTROL_PLANE_IP", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
ENGINE_PORT = 8000
CHUNK_SIZE_LINES = 200  # must match CLI chunking convention


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup logic

    # Initialize AWSAllocation
    print("[AWSAllocation] Starting up")
    app.state.aws_allocation = AWSAllocation(
        openrouter_key=OPENROUTER_API_KEY,
        perfdb_dir="./perf_db",
        aws_quota_csv="./quotas/aws_gpu_quota_by_region.csv",
        k_nearest_model_size=5,
    )

    # Initialize Redis + chunk queue
    redis_conn = await get_redis()
    app.state.redis = redis_conn
    app.state.chunk_queue = ChunkQueueManager(redis_conn=redis_conn)
    logger.info("Redis and ChunkQueueManager initialized")

    # Start background tasks (reaper + combiner)
    from chunk_queue.background_tasks import lease_reaper_loop, combiner_loop
    app.state.bg_tasks = [
        asyncio.create_task(lease_reaper_loop(app)),
        asyncio.create_task(combiner_loop(app)),
    ]

    yield

    # Shutdown logic
    for task in app.state.bg_tasks:
        task.cancel()
    await close_redis()
    logger.info("Redis connection closed")


app = FastAPI(
    title="Tandemn Server",
    description="API for receiving job requests",
    lifespan=lifespan,
)


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
        except Exception as e:
            # keep polling, simple prototype, but log the error
            print(f"[ERROR] Polling job {job_id} failed: {e}")
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

    1. Run placement to decide instance type, TP/PP, replicas
    2. Build chunk list from input_file prefix + num_lines
    3. Enqueue all chunks in Redis
    4. Launch worker replicas as background tasks
    5. Return immediately with job_id (progress tracked via Redis)
    """
    aws_alloc_engine: AWSAllocation = app.state.aws_allocation
    launch_config = aws_alloc_engine.decide(request)
    job_id = launch_config.decision_id
    logger.info(f"Placement decided: {launch_config}")

    # Build chunk list from the deterministic naming convention
    bucket_uri, prefix = split_uri(request.input_file)
    chunks = build_chunk_list(prefix, request.num_lines)

    # Enqueue in Redis
    cq: ChunkQueueManager = app.state.chunk_queue
    await cq.enqueue_job(
        job_id=job_id,
        chunks=chunks,
        model_name=request.model_name,
        input_prefix=prefix,
        bucket=bucket_uri,
    )
    await cq.update_job_status(job_id, "launching")
    logger.info(f"Job {job_id}: enqueued {len(chunks)} chunks")

    # Launch worker replicas in the background (non-blocking)
    for replica_idx in range(launch_config.replicas):
        worker_id = f"{job_id}-w{replica_idx}"
        asyncio.create_task(
            launch_batch_worker(request, launch_config, job_id, worker_id, bucket_uri)
        )

    return {
        "status": "accepted",
        "job_id": job_id,
        "total_chunks": len(chunks),
        "replicas": launch_config.replicas,
        "model": request.model_name,
    }


# ── Chunk-based batch helpers ──────────────────────────────────────────────


def build_chunk_list(prefix: str, num_lines: int) -> List[ChunkInfo]:
    """Build the chunk list from the deterministic naming convention.

    The CLI chunks at CHUNK_SIZE_LINES lines per chunk and names them
    000001.jsonl, 000002.jsonl, etc. under the input prefix.
    No cloud API calls needed — just math.
    """
    num_chunks = math.ceil(num_lines / CHUNK_SIZE_LINES)
    prefix = prefix.rstrip("/")
    return [
        ChunkInfo(
            chunk_id=f"{i:06d}",
            chunk_index=i,
            input_path=f"{prefix}/{i:06d}.jsonl",
        )
        for i in range(1, num_chunks + 1)
    ]


def build_serve_command(request: BatchedRequest, config: MagicOutput) -> str:
    """Build the engine-specific serve command from the template.

    Selects the template based on config.engine (e.g., templates/vllm_serve).
    For a new engine, just add templates/{engine}_serve.
    """
    additional_params = build_additional_params(request)
    replace = {
        "model": request.model_name,
        "engine_port": ENGINE_PORT,
        "tp": config.tp_size,
        "pp": config.pp_size,
        "additional_params": additional_params,
    }
    template = f"templates/{config.engine}_serve"
    return update_template(template, replace)


def build_additional_params(request: BatchedRequest) -> str:
    """Extract engine-specific additional params (e.g., speculative decoding)."""
    if (
        request.vllm_specific_config is not None
        and request.vllm_specific_config.speculative_config is not None
    ):
        prefix = "--speculative-config."
        spec_config = request.vllm_specific_config.speculative_config.model_dump(
            exclude_none=True
        )
        return " ".join(f"{prefix}{key} {value}" for key, value in spec_config.items())
    return ""


async def launch_batch_worker(
    request: BatchedRequest,
    config: MagicOutput,
    job_id: str,
    worker_id: str,
    bucket_uri: str,
):
    """Launch a single SkyPilot worker cluster (runs in background task)."""
    try:
        # Build engine serve command from template
        serve_command = build_serve_command(request, config)

        # Build worker run script from template
        run_replace = {
            "redis_host": CONTROL_PLANE_IP,
            "redis_port": REDIS_PORT,
            "redis_password_flag": f"--redis-password {REDIS_PASSWORD}" if REDIS_PASSWORD else "",
            "job_id": job_id,
            "worker_id": worker_id,
            "lease_ttl": 600,
            "serve_command": serve_command,
            "engine_port": ENGINE_PORT,
            "health_endpoint": "/v1/models",
            "inference_endpoint": "/v1/chat/completions",
            "max_concurrent": 64,
        }
        run_string = update_template("templates/batch_worker_run", run_replace)

        # Build SkyPilot YAML (unique file per worker to avoid races)
        yaml_output = f"temp/{worker_id}.yaml"
        replace_yaml = {
            "name": worker_id,
            "num_nodes": config.pp_size,
            "resources.instance_type": config.instance_type,
            "run": run_string,
            "file_mounts./data.source": bucket_uri,
        }
        update_yaml_file("templates/batch_worker.yaml", replace_yaml, yaml_output)

        # Launch via SkyPilot (blocking call, run in thread to not block event loop)
        task = sky.Task.from_yaml(yaml_output)
        result_id = await asyncio.to_thread(
            sky.launch, task, cluster_name=worker_id, down=True
        )
        await asyncio.to_thread(sky.stream_and_get, result_id, follow=True)

        logger.info(f"Worker {worker_id} launched for job {job_id}")
    except Exception:
        logger.exception(f"Failed to launch worker {worker_id}")


# ── Progress endpoints ─────────────────────────────────────────────────────


@app.get("/job/{job_id}/status")
async def job_status(job_id: str):
    """Get real-time job progress from Redis."""
    cq: ChunkQueueManager = app.state.chunk_queue
    progress = await cq.get_job_progress(job_id)
    if progress.total_chunks == 0:
        return {"error": "job not found", "job_id": job_id}

    meta = await app.state.redis.hgetall(f"job:{job_id}:meta")
    return {
        "job_id": job_id,
        "status": meta.get("status", "unknown"),
        "model": meta.get("model_name", ""),
        "total_chunks": progress.total_chunks,
        "completed": progress.completed,
        "pending": progress.pending,
        "leased": progress.leased,
        "failed": progress.failed,
        "progress": round(progress.progress_frac, 4),
        "is_done": progress.is_done,
    }


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
        "num_nodes": config.pp_size * config.replicas,
        "resources.instance_type": config.instance_type,
        "run": run_string,
        "file_mounts./data.source": dirname,
    }
    update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    task = sky.Task.from_yaml(YAML_OUTPUT)
    result_id = sky.launch(task, cluster_name=config.decision_id, down=True)
    job_id, handle = sky.stream_and_get(result_id, follow=True)

    # get the head IP
    # head_ip = handle.head_ip
    # endpoint_url = f"http://{head_ip}:8001/"

    # jt = get_job_tracker()
    # jt.set_head_ip(config.decision_id, head_ip)
    # jt.set_endpoint_url(config.decision_id, endpoint_url)
    # jt.update_status(config.decision_id, "launching")

    # threading.Thread(
    #     target=poll_job_progress,
    #     args=(config.decision_id, endpoint_url.rstrip("/"), request.num_lines or 1, jt),
    #     daemon=False,
    # ).start()

    # # sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)
    # threading.Thread(
    #     target=sky.tail_logs,
    #     kwargs={"cluster_name": config.decision_id, "job_id": job_id, "follow": True},
    #     daemon=False,
    # ).start()


async def sp_launch_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    """Launch persistant vllm online deployment"""
    replace_run_dict = replace_run_vllm_online(request, config)
    run_string = update_template("templates/vllm_run_online", replace_run_dict)

    replace_yaml = {
        "name": config.decision_id,
        "num_nodes": config.pp_size * config.replicas,
        "resources.instance_type": config.instance_type,
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
        instance_type="g6e.xlarge",
        tp_size=1,
        pp_size=1,
        replicas=1,
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
