from contextlib import asynccontextmanager
import uuid
import requests
import subprocess
import signal
import atexit
import sky
from fastapi import FastAPI, Form
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from placement.aws_magic import AWSAllocation
from placement.roofline_magic import RooflineAWSAllocation
from quota.region_selector import get_ordered_regions, get_instance_family, get_cached_quotas
from storage.storage_server import storage_backend
from tracking.tracking import JobRecord, JobSpec, JobState
from utils.utils import split_uri, update_template, update_yaml_file
from typing import Dict, Union, Optional, Literal, List, Tuple
from threading import Lock
import threading
import time
import re
from dotenv import load_dotenv
import os

load_dotenv()
##### Global variables
YAML_OUTPUT = "temp/output.yaml"


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 approximation."""
    return max(1, len(text) // 4)


def extract_prompt_text(entry: dict) -> str:
    """Extract prompt text from OpenAI batch format entry."""
    try:
        messages = entry.get("body", {}).get("messages", [])
        # Concatenate all message contents
        return " ".join(msg.get("content", "") for msg in messages)
    except (KeyError, TypeError):
        return ""


def parse_input_file_stats(input_file: str) -> tuple[int, int, int]:
    """
    Parse input file to extract real stats.

    Args:
        input_file: S3 URI (s3://bucket/path) or local path

    Returns:
        (num_lines, avg_input_tokens, max_input_tokens)
    """
    import json
    import tempfile

    # Download from S3 if needed
    if input_file.startswith("s3://"):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["aws", "s3", "cp", input_file, tmp_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {input_file}: {result.stderr}")
        file_path = tmp_path
    else:
        file_path = input_file

    # Parse JSONL and calculate stats
    token_counts = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt_text = extract_prompt_text(entry)
                tokens = estimate_tokens(prompt_text)
                token_counts.append(tokens)
    finally:
        # Clean up temp file if we created one
        if input_file.startswith("s3://"):
            import os
            os.unlink(tmp_path)

    if not token_counts:
        raise ValueError(f"No valid entries found in {input_file}")

    num_lines = len(token_counts)
    avg_input_tokens = sum(token_counts) // num_lines
    max_input_tokens = max(token_counts)

    print(f"[InputParser] Parsed {num_lines} lines: avg_input={avg_input_tokens}, max_input={max_input_tokens}")

    return num_lines, avg_input_tokens, max_input_tokens
OPENROUTER_API_KEY = os.environ.get("TD_OPENROUTER_KEY")

# Solver selection: "roofline" (deterministic) or "llm" (3-advisor + C-PMI)
PLACEMENT_SOLVER = os.environ.get("TD_PLACEMENT_SOLVER", "roofline").lower()

# Optimization priority for roofline solver
PLACEMENT_PRIORITY = os.environ.get("TD_PLACEMENT_PRIORITY", "cost_first").lower()

# HuggingFace token for gated models
HF_TOKEN = os.environ.get("HF_TOKEN")


def generate_job_dirname(request: BatchedRequest, solver: str, tp_size: int, pp_size: int) -> str:
    """
    Generate an informative directory name for job outputs.
    Format: {model_short}/numreq_N-avginputlen_X-avgoutputlen_Y/{solver}-tpT-ppP-timestamp_YYYYMMDD-HHMMSS
    Example: qwen72b/numreq_100-avginputlen_50-avgoutputlen_100/roofline-tp4-pp1-timestamp_20260215-143022
    """
    from datetime import datetime

    # Shorten model name: "Qwen/Qwen2.5-72B-Instruct" -> "qwen72b"
    model_name = request.model_name or "unknown"
    # Extract key parts: remove provider prefix, get size
    model_short = model_name.split("/")[-1].lower()  # "qwen2.5-72b-instruct"
    # Extract size (e.g., "72b", "7b", "235b")
    size_match = re.search(r'(\d+\.?\d*b)', model_short, re.IGNORECASE)
    size = size_match.group(1).lower() if size_match else ""
    # Get model family (first word before numbers/dashes)
    family = re.split(r'[\d\-_.]', model_short)[0][:6]  # max 6 chars
    model_short = f"{family}{size}" if size else family[:10]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Format: model/workload/{solver}-tp{T}-pp{P}-timestamp_{ts}
    dirname = f"{model_short}/numreq_{request.num_lines}-avginputlen_{request.avg_input_tokens}-avgoutputlen_{request.avg_output_tokens}/{solver}-tp{tp_size}-pp{pp_size}-timestamp_{timestamp}"
    return dirname


class ClusterManager:
    """Manages active SkyPilot clusters to prevent zombie instances."""

    def __init__(self):
        self.active_clusters: Dict[str, dict] = {}
        self.lock = Lock()

    def register(self, cluster_name: str, job_id: str):
        with self.lock:
            self.active_clusters[cluster_name] = {"job_id": job_id, "status": "active"}
            print(f"[ClusterManager] Registered cluster: {cluster_name}")

    def unregister(self, cluster_name: str):
        with self.lock:
            if cluster_name in self.active_clusters:
                del self.active_clusters[cluster_name]
                print(f"[ClusterManager] Unregistered cluster: {cluster_name}")

    def terminate_cluster(self, cluster_name: str) -> bool:
        try:
            result = subprocess.run(
                ["sky", "down", "-y", cluster_name],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                self.unregister(cluster_name)
                return True
            return False
        except Exception as e:
            print(f"[ClusterManager] Error terminating {cluster_name}: {e}")
            return False


_cluster_manager = None


def get_cluster_manager() -> ClusterManager:
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    # Solvers are now created per-request in submit_batch()

    # Initialize cluster manager
    app.state.cluster_manager = get_cluster_manager()

    yield

    # Shutdown logic


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

    The placement solver can be:
    - "roofline": Deterministic roofline-based solver (default)
    - "llm": LLM-based 3-advisor + C-PMI solver
    """
    # Parse input file to get real stats (num_lines, avg_input_tokens, max_input_tokens)
    num_lines, avg_input_tokens, max_input_tokens = parse_input_file_stats(request.input_file)

    # Update request with parsed values (these override any user-provided values)
    request = request.model_copy(update={
        "num_lines": num_lines,
        "avg_input_tokens": avg_input_tokens,
        "max_input_tokens": max_input_tokens,
    })

    print(f"[InputStats] num_lines={num_lines}, avg_input={avg_input_tokens}, max_input={max_input_tokens}")

    # Get multiple fallback solutions for retry logic
    # Request field takes priority, then fall back to env var
    use_solver = request.placement_solver or PLACEMENT_SOLVER
    if use_solver == "roofline":
        solver = RooflineAWSAllocation(
            perfdb_dir="./perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            priority=PLACEMENT_PRIORITY,
        )
        all_configs = solver.process_batch_multi(request, top_k=5)
        launch_config = all_configs[0] if all_configs else solver._fallback_config(request)
        fallback_configs = all_configs[1:] if len(all_configs) > 1 else []
    else:
        # LLM-based solver (3-advisor + C-PMI)
        openrouter_key = request.openrouter_api_key or OPENROUTER_API_KEY
        model_tier = request.llm_advisor_tier or "free"
        solver = AWSAllocation(
            openrouter_key=openrouter_key,
            perfdb_dir="./perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            k_nearest_model_size=5,
            model_tier=model_tier,
        )
        launch_config = solver.decide(request)
        fallback_configs = []

    print(f"[Placement] Using solver: {use_solver}")
    print(f"[Placement] Primary: {launch_config.instance_type} TP={launch_config.tp_size} PP={launch_config.pp_size}")
    if fallback_configs:
        print(f"[Placement] Fallbacks: {len(fallback_configs)}")

    # Launch with fallback support
    success, used_config = await sp_launch_vllm_batch_with_fallback(
        request, launch_config, fallback_configs, solver=use_solver
    )

    if success:
        return {
            "status": "launched",
            "job_id": used_config.decision_id,
            "config": {
                "instance_type": used_config.instance_type,
                "tp_size": used_config.tp_size,
                "pp_size": used_config.pp_size,
            },
            "input_stats": {
                "num_lines": num_lines,
                "avg_input_tokens": avg_input_tokens,
                "max_input_tokens": max_input_tokens,
            },
            "message": f"Job submitted. Check progress at GET /job/{used_config.decision_id}",
        }
    else:
        return {
            "status": "error",
            "job_id": launch_config.decision_id,
            "message": "Failed to launch in all regions with all instance types",
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


def download_output_from_s3(s3_path: str, job_dirname: str) -> Optional[str]:
    """Download output file and metrics.csv from S3 to local filesystem."""
    from pathlib import Path

    # Use the informative job dirname for local storage
    local_output_dir = Path(f"outputs/{job_dirname}")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # Download output.jsonl
    filename = s3_path.split("/")[-1]
    local_path = local_output_dir / filename

    try:
        print(f"[Download] Downloading {s3_path} to {local_path}...")
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[Download] Failed to download output: {result.stderr}")
            return None
        print(f"[Download] Output saved: {local_path}")

        # Download metrics.csv from same directory
        s3_dir = "/".join(s3_path.split("/")[:-1])
        metrics_s3_path = f"{s3_dir}/metrics.csv"
        metrics_local_path = local_output_dir / "metrics.csv"

        print(f"[Download] Downloading {metrics_s3_path}...")
        metrics_result = subprocess.run(
            ["aws", "s3", "cp", metrics_s3_path, str(metrics_local_path)],
            capture_output=True, text=True, timeout=60
        )
        if metrics_result.returncode == 0:
            print(f"[Download] Metrics saved: {metrics_local_path}")
        else:
            print(f"[Download] Metrics not found (may not exist yet)")

        return str(local_path)
    except Exception as e:
        print(f"[Download] Error: {e}")
        return None


async def sp_launch_vllm_batch_with_fallback(
    request: BatchedRequest,
    primary_config: MagicOutput,
    fallback_configs: list,
    solver: str = "roofline",
) -> Tuple[bool, MagicOutput]:
    """Launch vLLM batch job with fallback to alternative instance types."""
    all_configs = [primary_config] + fallback_configs

    for i, config in enumerate(all_configs):
        print(f"[Launch] Trying config {i+1}/{len(all_configs)}: {config.instance_type} TP={config.tp_size} PP={config.pp_size}")

        try:
            await sp_launch_vllm_batch(request, config, solver)
            print(f"[Launch] Success with config {i+1}: {config.instance_type}")
            return (True, config)

        except Exception as e:
            print(f"[Launch] Config {i+1} failed: {e}")
            if i < len(all_configs) - 1:
                print("[Launch] Trying next instance type...")
                continue
            else:
                print(f"[Launch] All {len(all_configs)} configs failed")
                return (False, primary_config)

    return (False, primary_config)


async def sp_launch_vllm_batch(request: BatchedRequest, config: MagicOutput, solver: str = "roofline"):
    # Generate informative job directory name
    job_dirname = generate_job_dirname(request, solver, config.tp_size, config.pp_size)

    s3_base, _ = split_uri(request.input_file)
    # S3 output path: s3://bucket/base/job_dirname/output.jsonl
    s3_output_dir = f"{s3_base}/{job_dirname}"

    hf_token = request.hf_token or HF_TOKEN or ""
    num_nodes = config.pp_size * config.replicas

    # Select per-config template or fall back to generic
    template_path = get_vllm_config_template(
        model_name=request.model_name,
        instance_type=config.instance_type,
        tp=config.tp_size,
        pp=config.pp_size,
    )

    # Get quota-aware ordered regions
    instance_family = get_instance_family(config.instance_type)
    quotas = get_cached_quotas(instance_family)
    ordered_regions = get_ordered_regions(
        instance_type=config.instance_type,
        num_nodes=num_nodes,
        quotas=quotas,
        prefer_spot=True,
    )

    # Build resources with any_of for fallback regions
    if ordered_regions:
        any_of_resources = []
        for candidate in ordered_regions[:5]:
            any_of_resources.append({
                "region": candidate.region,
                "instance_type": config.instance_type,
                "use_spot": candidate.use_spot,
                "disk_size": "300GB",
                "ports": 8001,
            })
        print(f"[RegionSelector] Trying regions: {[(c.region, 'spot' if c.use_spot else 'on-demand') for c in ordered_regions[:5]]}")
        resources_config = {"any_of": any_of_resources}
    else:
        resources_config = {
            "infra": "aws",
            "instance_type": config.instance_type,
            "disk_size": "300GB",
            "ports": 8001,
        }

    # For per-config templates, we only need to replace minimal fields
    # The template already has model, tp, pp, max_model_len baked in
    if "vllm_configs" in template_path:
        # Per-config template - minimal substitution
        _, input_file = split_uri(request.input_file)
        output_file = request.output_file
        vllm_cfg = request.vllm_specific_config
        max_num_seqs = vllm_cfg.max_num_seqs if vllm_cfg and vllm_cfg.max_num_seqs else 32

        # Read template and substitute only dynamic fields
        from pathlib import Path
        template_content = Path(template_path).read_text()
        template_content = template_content.replace("{input_file}", f"/data/{input_file}")
        template_content = template_content.replace("{output_file}", f"/data/{job_dirname}/{output_file}")
        template_content = template_content.replace("{max_num_seqs}", str(max_num_seqs))

        # Write to temp file and parse as yaml
        import yaml
        yaml_data = yaml.safe_load(template_content)

        # Update dynamic fields
        yaml_data["name"] = config.decision_id
        yaml_data["num_nodes"] = num_nodes
        yaml_data["resources"] = resources_config
        yaml_data["file_mounts"]["/data"]["source"] = s3_base
        yaml_data["envs"]["HF_TOKEN"] = hf_token

        # Write final yaml
        from pathlib import Path
        Path(YAML_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        with open(YAML_OUTPUT, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    else:
        # Generic template - use old substitution method
        replace_run_dict = replace_run_vllm(request, config, job_dirname)
        run_string = update_template("templates/vllm_run", replace_run_dict)

        replace_yaml = {
            "name": config.decision_id,
            "num_nodes": num_nodes,
            "resources": resources_config,
            "run": run_string,
            "file_mounts./data.source": s3_base,
            "envs.HF_TOKEN": hf_token,
        }
        update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    cm = get_cluster_manager()
    cm.register(config.decision_id, config.decision_id)

    # Construct S3 output path for later download
    output_s3_path = f"{s3_output_dir}/{request.output_file}"
    print(f"[Job] Output will be saved to: {s3_output_dir}/")

    def monitor_and_download():
        """Background thread: stream logs, then download output when done."""
        try:
            sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)
            print(f"[Job] {config.decision_id} completed. Downloading output...")

            # Download output from S3 to local dir with same informative name
            local_path = download_output_from_s3(output_s3_path, job_dirname)
            if local_path:
                print(f"[Job] Output saved to: {local_path}")
            else:
                print(f"[Job] Warning: Failed to download output from {output_s3_path}")
        except Exception as e:
            print(f"[Job] Error in monitor thread: {e}")
        finally:
            cm.unregister(config.decision_id)

    try:
        task = sky.Task.from_yaml(YAML_OUTPUT)
        result_id = sky.launch(task, cluster_name=config.decision_id, down=True)
        job_id, handle = sky.stream_and_get(result_id, follow=True)

        # Stream logs in background and download when done
        threading.Thread(target=monitor_and_download, daemon=True).start()

    except Exception as e:
        cm.unregister(config.decision_id)
        raise Exception(f"Failed to launch cluster {config.decision_id}: {e}")


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


# Instance type to GPU name mapping
INSTANCE_TO_GPU = {
    # G6e instances (L40S)
    "g6e.xlarge": "L40S", "g6e.2xlarge": "L40S", "g6e.4xlarge": "L40S",
    "g6e.8xlarge": "L40S", "g6e.12xlarge": "L40S", "g6e.16xlarge": "L40S",
    "g6e.24xlarge": "L40S", "g6e.48xlarge": "L40S",
    # G6 instances (L4)
    "g6.xlarge": "L4", "g6.2xlarge": "L4", "g6.4xlarge": "L4",
    "g6.8xlarge": "L4", "g6.12xlarge": "L4", "g6.16xlarge": "L4",
    "g6.24xlarge": "L4", "g6.48xlarge": "L4",
    # G5 instances (A10G)
    "g5.xlarge": "A10G", "g5.2xlarge": "A10G", "g5.4xlarge": "A10G",
    "g5.8xlarge": "A10G", "g5.12xlarge": "A10G", "g5.16xlarge": "A10G",
    "g5.24xlarge": "A10G", "g5.48xlarge": "A10G",
    # P4 instances (A100)
    "p4d.24xlarge": "A100", "p4de.24xlarge": "A100",
    # P5 instances (H100)
    "p5.48xlarge": "H100",
    # P3 instances (V100)
    "p3.2xlarge": "V100", "p3.8xlarge": "V100", "p3.16xlarge": "V100", "p3dn.24xlarge": "V100",
}


def get_vllm_config_template(model_name: str, instance_type: str, tp: int, pp: int) -> str:
    """
    Get the per-config vLLM template path if it exists.

    Template naming: vllm_{model}-{gpu}-tp{tp}-pp{pp}.yaml
    Example: vllm_qwen2.5-72b-A100-tp8-pp1.yaml

    Returns path to specific template if exists, otherwise returns generic template.
    """
    import os

    # Normalize model name (e.g., "Qwen/Qwen2.5-72B-Instruct" -> "qwen2.5-72b")
    model_short = model_name.lower()
    if "/" in model_short:
        model_short = model_short.split("/")[-1]
    # Remove common suffixes
    for suffix in ["-instruct", "-chat", "-base", "-hf"]:
        model_short = model_short.replace(suffix, "")

    # Get GPU name from instance type
    gpu_name = INSTANCE_TO_GPU.get(instance_type, "unknown")

    # Build template filename
    template_name = f"vllm_{model_short}-{gpu_name}-tp{tp}-pp{pp}.yaml"
    template_path = f"templates/vllm_configs/{template_name}"

    if os.path.exists(template_path):
        print(f"[Template] Using per-config template: {template_path}")
        return template_path
    else:
        print(f"[Template] No specific template for {template_name}, using generic")
        return "templates/vllm.yaml"


def replace_run_vllm(request: BatchedRequest, config: MagicOutput, job_dirname: str = "output"):
    replace = {}

    replace["model"] = request.model_name
    replace["tp"] = config.tp_size
    replace["pp"] = config.pp_size

    # Calculate max_model_len:
    # 1. If solver provides it (roofline), use it
    # 2. Else if request has max_input/output_tokens, calculate from those
    # 3. Else use "auto" (let vLLM figure it out)
    if config.max_model_len:
        replace["max_model_len"] = config.max_model_len
    elif request.max_input_tokens and request.max_output_tokens:
        # Use actual max lengths from request + 10% safety margin
        calculated_len = int((request.max_input_tokens + request.max_output_tokens) * 1.1)
        # Round up to nearest power of 2 for efficiency, clamp to reasonable range
        calculated_len = max(1024, min(calculated_len, 131072))
        replace["max_model_len"] = calculated_len
        print(f"[Config] Calculated max_model_len={calculated_len} from max_input={request.max_input_tokens} + max_output={request.max_output_tokens}")
    else:
        replace["max_model_len"] = "auto"

    _, input_file = split_uri(request.input_file)
    output = request.output_file
    replace["input_file"] = "/data/" + input_file
    # Output goes to informative subdirectory: /data/{job_dirname}/output.jsonl
    replace["output_file"] = f"/data/{job_dirname}/{output}"

    # Infrastructure configuration for metrics tracking
    replace["cloud"] = "aws"  # Currently only AWS supported
    replace["instance_type"] = config.instance_type
    replace["gpu_name"] = INSTANCE_TO_GPU.get(config.instance_type, "unknown")
    replace["engine"] = request.engine or "vllm"
    replace["quantization"] = request.quantization_bits or "none"

    # Get vLLM-specific configs
    vllm_cfg = request.vllm_specific_config
    replace["max_num_seqs"] = vllm_cfg.max_num_seqs if vllm_cfg and vllm_cfg.max_num_seqs else 32
    replace["dtype"] = "auto"  # vLLM auto-detects based on model
    replace["kv_cache_dtype"] = vllm_cfg.kv_cache_dtype if vllm_cfg and vllm_cfg.kv_cache_dtype else "auto"

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
        num_inst=1,
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
