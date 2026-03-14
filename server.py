from contextlib import asynccontextmanager
import math
import uuid
import requests
import subprocess
import sky
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from placement.aws_magic import AWSAllocation
from placement.roofline_magic import (
    RooflineAWSAllocation,
    resolve_gpu_type_to_instance,
    check_user_specified_feasibility,
)
from quota.region_selector import (
    get_ordered_regions,
    get_instance_family,
    get_cached_quotas,
)
from storage.storage_factory import get_storage_backend
from tracking.tracking import JobRecord, JobSpec, JobState
from utils.utils import split_uri, update_template, update_yaml_file
from typing import Dict, List, Union, Optional, Literal, Tuple
from threading import Lock
import threading
import time
import re
import logging
from dotenv import load_dotenv
import os
import tempfile
import json

load_dotenv()
##### Global variables
YAML_OUTPUT = "temp/output.yaml"


def setup_job_logger(job_id: str, log_file_path: str) -> logging.Logger:
    """Create a named logger that writes to both console and a job-specific log file."""
    logger = logging.getLogger(f"orca.job.{job_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Console handler — INFO level, simple format (matches print() behavior)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File handler — DEBUG level, with timestamps
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    return logger


def close_job_logger(logger: logging.Logger):
    """Flush and close all handlers on the logger."""
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


# S3 model storage defaults (used with s3_models flag)
S3_MODEL_BUCKET = "tandemn-model-shards"
S3_MODEL_PREFIX = "hf-models"


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


# A: Downloads the file once from S3 to wherever server is running + timeout
# A: All prompts are extracted into program memory
# A: tokenizer
# D: aws CLI, S3 credentials
# D: python - json, tempfile, transformers AutoTokenizer
def parse_input_file_stats(
    input_file: str, model_name: str = None, top_k_tokenize: int = 100
) -> tuple[int, int, int]:
    """
    Parse input file to extract real stats.

    Uses chars/4 for fast average estimation, then tokenizes the top-K longest
    prompts (by character count) with the model's actual tokenizer for an
    accurate max_input_tokens.

    Args:
        input_file: S3 URI (s3://bucket/path) or local path
        model_name: HuggingFace model name for tokenizer (e.g. "Qwen/Qwen2.5-72B-Instruct")
        top_k_tokenize: Number of longest prompts to tokenize (default 100)

    Returns:
        (num_lines, avg_input_tokens, max_input_tokens)
    """
    # Download from S3 if needed
    if input_file.startswith("s3://"):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["aws", "s3", "cp", input_file, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {input_file}: {result.stderr}")
        file_path = tmp_path
    else:
        file_path = input_file

    # Parse JSONL: collect prompt texts and chars/4 estimates
    prompt_texts = []
    char_estimates = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt_text = extract_prompt_text(entry)
                prompt_texts.append(prompt_text)
                char_estimates.append(estimate_tokens(prompt_text))
    finally:
        if input_file.startswith("s3://"):
            import os

            os.unlink(tmp_path)

    if not prompt_texts:
        raise ValueError(f"No valid entries found in {input_file}")

    num_lines = len(prompt_texts)
    avg_input_tokens = sum(char_estimates) // num_lines

    # Tokenize top-K longest prompts for accurate max_input_tokens
    max_input_tokens = max(char_estimates)  # fallback if tokenizer unavailable
    if model_name and top_k_tokenize > 0:
        try:
            from transformers import AutoTokenizer, logging as hf_logging

            hf_logging.set_verbosity_error()  # Suppress warn msgs about hf key
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Sort indices by character length descending, take top-K
            sorted_indices = sorted(
                range(num_lines), key=lambda i: len(prompt_texts[i]), reverse=True
            )
            top_indices = sorted_indices[:top_k_tokenize]

            max_tokenized = 0
            for idx in top_indices:
                token_count = len(tokenizer.encode(prompt_texts[idx]))
                max_tokenized = max(max_tokenized, token_count)

            print(
                f"[InputParser] Tokenized top-{len(top_indices)} longest prompts: "
                f"max_input={max_tokenized} tokens (chars/4 estimate was {max(char_estimates)})"
            )
            max_input_tokens = max_tokenized
        except Exception as e:
            print(
                f"[InputParser] WARNING: Tokenizer failed ({e}), using chars/4 estimate for max_input_tokens"
            )

    print(
        f"[InputParser] Parsed {num_lines} lines: avg_input={avg_input_tokens}, max_input={max_input_tokens}"
    )

    return num_lines, avg_input_tokens, max_input_tokens


OPENROUTER_API_KEY = os.environ.get("TD_OPENROUTER_KEY")

# Solver selection: "roofline" (deterministic) or "llm" (3-advisor + C-PMI)
PLACEMENT_SOLVER = os.environ.get("TD_PLACEMENT_SOLVER", "roofline").lower()

# Optimization priority for roofline solver
PLACEMENT_PRIORITY = os.environ.get("TD_PLACEMENT_PRIORITY", "cost_first").lower()

# HuggingFace token for gated models
HF_TOKEN = os.environ.get("HF_TOKEN")


def prefix_job_dirname(job_dirname: str, status: str) -> str:
    """Prepend 'success-' or 'failed-' to the leaf directory of a job dirname."""
    if "/" in job_dirname:
        parent, leaf = job_dirname.rsplit("/", 1)
        return f"{parent}/{status}-{leaf}"
    return f"{status}-{job_dirname}"


def generate_job_dirname(
    request: BatchedRequest, solver: str, tp_size: int, pp_size: int, instance_type: str
) -> str:
    """
    Generate an informative directory name for job outputs.
    Base format: {model_short}/numreq_N-avginputlen_X-avgoutputlen_Y/{solver}-{gpu}-tpT-ppP-YYYYMMDD_HHMMSS
    After completion the leaf dir is prefixed with 'success-' or 'failed-'.
    Example: qwen72b/numreq_100-avginputlen_50-avgoutputlen_100/success-roofline-A100-tp4-pp1-20260215_143022
    """
    from datetime import datetime

    # Shorten model name: "Qwen/Qwen2.5-72B-Instruct" -> "qwen72b"
    model_name = request.model_name or "unknown"
    # Extract key parts: remove provider prefix, get size
    model_short = model_name.split("/")[-1].lower()  # "qwen2.5-72b-instruct"
    # Extract size (e.g., "72b", "7b", "235b")
    size_match = re.search(r"(\d+\.?\d*b)", model_short, re.IGNORECASE)
    size = size_match.group(1).lower() if size_match else ""
    # Get model family (first word before numbers/dashes)
    family = re.split(r"[\d\-_.]", model_short)[0][:6]  # max 6 chars
    model_short = f"{family}{size}" if size else family[:10]

    # Get GPU name from instance type
    gpu_name = INSTANCE_TO_GPU.get(instance_type, "unknown")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Format: model/workload/{solver}-{gpu}-tp{T}-pp{P}-{timestamp}
    dirname = f"{model_short}/numreq_{request.num_lines}-avginputlen_{request.avg_input_tokens}-avgoutputlen_{request.avg_output_tokens}/{solver}-{gpu_name}-tp{tp_size}-pp{pp_size}-{timestamp}"
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
                capture_output=True,
                text=True,
                timeout=300,
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

storage_backend = get_storage_backend()
CHUNK_SIZE_MB = int(os.getenv("CHUNK_SIZE_MB", 8)) * 1024 * 1024


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


# A: Stats on input file computed instead of from request
# A: When to respond to user request (prob return early + observability)
@app.post("/submit/batch")
async def submit_batch(request: BatchedRequest):
    """
    Submit a batched inference job request.

    The placement solver can be:
    - "roofline": Deterministic roofline-based solver (default)
    - "llm": LLM-based 3-advisor + C-PMI solver
    """
    # Parse input file to get real stats (num_lines, avg_input_tokens, max_input_tokens)
    num_lines, avg_input_tokens, max_input_tokens = parse_input_file_stats(
        request.input_file, model_name=request.model_name
    )

    # Update request with parsed values (these override any user-provided values)
    request = request.model_copy(
        update={
            "num_lines": num_lines,
            "avg_input_tokens": avg_input_tokens,
            "max_input_tokens": max_input_tokens,
        }
    )

    # Collect log messages before the job logger is created
    early_messages = []

    msg = f"[InputStats] num_lines={num_lines}, avg_input={avg_input_tokens}, max_input={max_input_tokens}"
    print(msg)
    early_messages.append(("INFO", msg))

    # Get multiple fallback solutions for retry logic
    # Request field takes priority, then fall back to env var
    use_solver = request.placement_solver or PLACEMENT_SOLVER

    if use_solver == "user_specified":
        # ---------- User-specified path ----------
        gpu_type = request.gpu_type
        tp = request.tp_size or 1
        pp = request.pp_size or 1

        # Resolve GPU type to the smallest AWS instance with enough GPUs for TP
        try:
            instance_type, gpu_count = resolve_gpu_type_to_instance(gpu_type, tp)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "invalid_placement",
                "message": str(e),
            }

        # Run feasibility check via LLM_placement_solver
        result = check_user_specified_feasibility(
            model_name=request.model_name,
            instance_type=instance_type,
            gpu_count=gpu_count,
            tp=tp,
            pp=pp,
            avg_input_tokens=request.avg_input_tokens,
            avg_output_tokens=request.avg_output_tokens,
            max_input_tokens=request.max_input_tokens or 0,
            max_output_tokens=request.max_output_tokens or 0,
        )

        if not result["feasible"]:
            return {
                "status": "error",
                "error_type": "infeasible_placement",
                "message": f"Placement is not feasible: {result['reason']}",
                "detail": {
                    "gpu_type": gpu_type,
                    "tp": tp,
                    "pp": pp,
                    "instance_type": instance_type,
                },
            }

        # Build MagicOutput directly
        partitions_per_inst = gpu_count // tp
        num_instances = math.ceil(pp / partitions_per_inst)
        configs = [
            MagicOutput(
                decision_id=f"mo-{uuid.uuid4()}",
                engine=request.engine or "vllm",
                instance_type=instance_type,
                tp_size=tp,
                pp_size=pp,
                replicas=1,
                max_model_len=result["max_model_len"],
                num_instances=num_instances,
            )
        ]

        sol = result.get("solution") or {}
        msg = (
            f"[Placement] user_specified: {instance_type} TP={tp} PP={pp} "
            f"max_model_len={result['max_model_len']} "
            f"throughput={sol.get('throughput_tokens_per_sec', 'N/A')} tok/s "
            f"cost=${sol.get('cost_per_hour', 'N/A')}/hr"
        )
        print(msg)
        early_messages.append(("INFO", msg))

    elif use_solver == "roofline":
        solver = RooflineAWSAllocation(
            perfdb_dir="./perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            priority=PLACEMENT_PRIORITY,
        )
        configs = solver.process_batch_multi(request, top_k=5)
        if not configs:
            configs = [solver._fallback_config(request)]
        # Capture solver log for the job log file (always, regardless of success)
        if getattr(solver, "last_solve_log", ""):
            for line in solver.last_solve_log.splitlines():
                early_messages.append(("INFO", f"[Solver] {line}"))
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
        configs = [solver.decide(request)]

    msg = f"[Placement] Using solver: {use_solver}"
    print(msg)
    early_messages.append(("INFO", msg))
    msg = f"[Placement] Primary: {configs[0].instance_type} TP={configs[0].tp_size} PP={configs[0].pp_size}"
    print(msg)
    early_messages.append(("INFO", msg))
    if len(configs) > 1:
        msg = f"[Placement] Fallbacks: {len(configs) - 1}"
        print(msg)
        early_messages.append(("INFO", msg))

    # Pre-launch check: ensure max_model_len can accommodate the longest prompt
    max_output = request.max_output_tokens or request.avg_output_tokens
    if configs[0].max_model_len is not None and max_input_tokens is not None:
        required_context = max_input_tokens + max_output
        if required_context > configs[0].max_model_len:
            return {
                "status": "error",
                "error_type": "context_length_exceeded",
                "message": (
                    f"Longest prompt ({max_input_tokens} tokens) + max_output ({max_output}) = "
                    f"{required_context} exceeds max_model_len ({configs[0].max_model_len}) "
                    f"for {configs[0].instance_type} TP={configs[0].tp_size} PP={configs[0].pp_size}. "
                    f"Some requests would be skipped at runtime."
                ),
                "detail": {
                    "max_input_tokens": max_input_tokens,
                    "max_output_tokens": max_output,
                    "required_context": required_context,
                    "max_model_len": configs[0].max_model_len,
                    "instance_type": configs[0].instance_type,
                    "tp_size": configs[0].tp_size,
                    "pp_size": configs[0].pp_size,
                },
            }

    # Launch with fallback support
    success, used_config = await sp_launch_vllm_batch_with_fallback(
        request, configs, solver=use_solver, early_messages=early_messages
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
            "job_id": configs[0].decision_id,
            "message": "Failed to launch in all regions with all instance types",
        }


@app.post("/test/placement")
async def test_placement(request: BatchedRequest):
    """
    Run placement logic only (no cloud deployment).

    Accepts the same BatchedRequest as /submit/batch but only runs the solver
    and returns the placement decision(s) with performance/cost estimates.
    """
    # Parse input file to get real stats
    num_lines, avg_input_tokens, max_input_tokens = parse_input_file_stats(
        request.input_file, model_name=request.model_name
    )

    request = request.model_copy(
        update={
            "num_lines": num_lines,
            "avg_input_tokens": avg_input_tokens,
            "max_input_tokens": max_input_tokens,
        }
    )

    use_solver = request.placement_solver or PLACEMENT_SOLVER
    solver_log = ""

    if use_solver == "user_specified":
        gpu_type = request.gpu_type
        tp = request.tp_size or 1
        pp = request.pp_size or 1

        try:
            instance_type, gpu_count = resolve_gpu_type_to_instance(gpu_type, tp)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "invalid_placement",
                "message": str(e),
            }

        result = check_user_specified_feasibility(
            model_name=request.model_name,
            instance_type=instance_type,
            gpu_count=gpu_count,
            tp=tp,
            pp=pp,
            avg_input_tokens=request.avg_input_tokens,
            avg_output_tokens=request.avg_output_tokens,
            max_input_tokens=request.max_input_tokens or 0,
            max_output_tokens=request.max_output_tokens or 0,
        )

        if not result["feasible"]:
            return {
                "status": "error",
                "error_type": "infeasible_placement",
                "message": f"Placement is not feasible: {result['reason']}",
                "detail": {
                    "gpu_type": gpu_type,
                    "tp": tp,
                    "pp": pp,
                    "instance_type": instance_type,
                },
            }

        partitions_per_inst = gpu_count // tp
        num_instances = math.ceil(pp / partitions_per_inst)
        sol = result.get("solution") or {}
        configs = [
            {
                "instance_type": instance_type,
                "gpu_type": gpu_type,
                "tp_size": tp,
                "pp_size": pp,
                "num_instances": num_instances,
                "max_model_len": result["max_model_len"],
                "throughput_tokens_per_sec": sol.get("throughput_tokens_per_sec"),
                "cost_per_hour": sol.get("cost_per_hour"),
                "cost_per_million_tokens": sol.get("cost_per_million_tokens"),
            }
        ]

    elif use_solver == "roofline":
        solver = RooflineAWSAllocation(
            perfdb_dir="./perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            priority=PLACEMENT_PRIORITY,
        )
        magic_outputs = solver.process_batch_multi(request, top_k=5)
        solver_log = getattr(solver, "last_solve_log", "")
        if not magic_outputs:
            magic_outputs = [solver._fallback_config(request)]

        configs = []
        for mo in magic_outputs:
            configs.append(
                {
                    "instance_type": mo.instance_type,
                    "gpu_type": INSTANCE_TO_GPU.get(mo.instance_type, "unknown"),
                    "tp_size": mo.tp_size,
                    "pp_size": mo.pp_size,
                    "num_instances": mo.num_instances or mo.num_nodes,
                    "max_model_len": mo.max_model_len,
                }
            )
    else:
        # LLM-based solver
        openrouter_key = request.openrouter_api_key or OPENROUTER_API_KEY
        model_tier = request.llm_advisor_tier or "free"
        solver = AWSAllocation(
            openrouter_key=openrouter_key,
            perfdb_dir="./perf_db",
            aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
            k_nearest_model_size=5,
            model_tier=model_tier,
        )
        mo = solver.decide(request)
        configs = [
            {
                "instance_type": mo.instance_type,
                "gpu_type": INSTANCE_TO_GPU.get(mo.instance_type, "unknown"),
                "tp_size": mo.tp_size,
                "pp_size": mo.pp_size,
                "num_instances": mo.num_instances or mo.num_nodes,
                "max_model_len": mo.max_model_len,
            }
        ]

    # Context length check
    max_output = request.max_output_tokens or request.avg_output_tokens
    context_warning = None
    if configs and configs[0].get("max_model_len") and max_input_tokens:
        required_context = max_input_tokens + max_output
        if required_context > configs[0]["max_model_len"]:
            context_warning = (
                f"Longest prompt ({max_input_tokens} tokens) + max_output ({max_output}) = "
                f"{required_context} exceeds max_model_len ({configs[0]['max_model_len']})"
            )

    response = {
        "status": "ok",
        "solver": use_solver,
        "input_stats": {
            "num_lines": num_lines,
            "avg_input_tokens": avg_input_tokens,
            "max_input_tokens": max_input_tokens,
        },
        "placements": configs,
    }
    if context_warning:
        response["context_warning"] = context_warning
    if solver_log:
        response["solver_log"] = solver_log
        with open("solver.log", "w") as f:
            f.write(solver_log)

    return response


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


def download_output_from_s3(
    s3_path: str, job_dirname: str, logger=None
) -> Optional[str]:
    """Download output file and metrics.csv from S3 to local filesystem."""
    from pathlib import Path

    _log = logger.info if logger else print
    _log_err = logger.error if logger else print

    # Use the informative job dirname for local storage
    local_output_dir = Path(f"outputs/{job_dirname}")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # Download output.jsonl
    filename = s3_path.split("/")[-1]
    local_path = local_output_dir / filename

    try:
        _log(f"[Download] Downloading {s3_path} to {local_path}...")
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            _log_err(f"[Download] Failed to download output: {result.stderr}")
            return None
        _log(f"[Download] Output saved: {local_path}")

        # Download metrics.csv from same directory
        s3_dir = "/".join(s3_path.split("/")[:-1])
        metrics_s3_path = f"{s3_dir}/metrics.csv"
        metrics_local_path = local_output_dir / "metrics.csv"

        _log(f"[Download] Downloading {metrics_s3_path}...")
        metrics_result = subprocess.run(
            ["aws", "s3", "cp", metrics_s3_path, str(metrics_local_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if metrics_result.returncode == 0:
            _log(f"[Download] Metrics saved: {metrics_local_path}")
        else:
            _log("[Download] Metrics not found (may not exist yet)")

        return str(local_path)
    except Exception as e:
        _log_err(f"[Download] Error: {e}")
        return None


async def sp_launch_vllm_batch_with_fallback(
    request: BatchedRequest,
    configs: List[MagicOutput],
    solver: str = "roofline",
    early_messages: list = None,
) -> Tuple[bool, MagicOutput]:
    """Launch vLLM batch job with fallback to alternative instance types."""
    if early_messages is None:
        early_messages = []

    for i, config in enumerate(configs):
        msg = f"[Launch] Trying config {i + 1}/{len(configs)}: {config.instance_type} TP={config.tp_size} PP={config.pp_size}"
        print(msg)
        early_messages.append(("INFO", msg))

        try:
            await sp_launch_vllm_batch(
                request, config, solver, early_messages=early_messages
            )
            print(f"[Launch] Success with config {i + 1}: {config.instance_type}")
            return (True, config)

        except Exception as e:
            print(f"[Launch] Config {i + 1} failed: {e}")
            if i < len(configs) - 1:
                print("[Launch] Trying next instance type...")
                continue
            else:
                print(f"[Launch] All {len(configs)} configs failed")
                return (False, configs[0])


async def sp_launch_vllm_batch(
    request: BatchedRequest,
    config: MagicOutput,
    solver: str = "roofline",
    early_messages: list = None,
):
    from pathlib import Path

    # Generate informative job directory name
    job_dirname = generate_job_dirname(
        request, solver, config.tp_size, config.pp_size, config.instance_type
    )

    # Create output dir and job logger early
    output_dir = Path(f"outputs/{job_dirname}")
    output_dir.mkdir(parents=True, exist_ok=True)
    job_logger = setup_job_logger(config.decision_id, str(output_dir / "job.log"))

    # Flush early messages collected before logger existed
    if early_messages:
        for level, msg in early_messages:
            getattr(job_logger, level.lower(), job_logger.info)(msg)

    s3_base, _ = split_uri(request.input_file)
    # S3 output path: s3://bucket/base/job_dirname/output.jsonl
    s3_output_dir = f"{s3_base}/{job_dirname}"

    # Verify model exists in S3; fall back to HuggingFace if unavailable
    if request.s3_models:
        s3_model_path = (
            f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{request.model_name}/"
        )
        s3_check = subprocess.run(
            ["aws", "s3", "ls", s3_model_path],
            capture_output=True,
            text=True,
        )
        if s3_check.returncode != 0 or not s3_check.stdout.strip():
            job_logger.warning(
                f"[S3] Model '{request.model_name}' not found at {s3_model_path}. "
                f"Falling back to HuggingFace download."
            )
            request.s3_models = False
        else:
            job_logger.info(f"[S3] Verified model exists at {s3_model_path}")

    hf_token = request.hf_token or HF_TOKEN or ""
    num_nodes = config.num_nodes

    # Select per-config template or fall back to generic
    template_path = get_vllm_config_template(
        model_name=request.model_name,
        instance_type=config.instance_type,
        tp=config.tp_size,
        pp=config.pp_size,
        logger=job_logger,
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
            any_of_resources.append(
                {
                    "region": candidate.region,
                    "instance_type": config.instance_type,
                    "use_spot": candidate.use_spot,
                    "disk_size": "300GB",
                    "ports": 8001,
                }
            )
        job_logger.info(
            f"[RegionSelector] Trying regions: {[(c.region, 'spot' if c.use_spot else 'on-demand') for c in ordered_regions[:5]]}"
        )
        resources_config = {"any_of": any_of_resources}
    else:
        resources_config = {
            "infra": "aws",
            "instance_type": config.instance_type,
            "disk_size": "300GB",
            "ports": 8001,
        }

    # For per-config templates, substitute all placeholders
    if "vllm_configs" in template_path:
        # Build substitution dict (same as generic template)
        replace_dict = replace_run_vllm(request, config, job_dirname, logger=job_logger)

        template_content = Path(template_path).read_text()
        for key, value in replace_dict.items():
            template_content = template_content.replace("{" + key + "}", str(value))

        # Write to temp file and parse as yaml
        import yaml

        yaml_data = yaml.safe_load(template_content)

        # Preserve image_id from template if specified (e.g., custom AMI for A100)
        template_image_id = yaml_data.get("resources", {}).get("image_id")
        template_region = yaml_data.get("resources", {}).get("region")

        # If template has a specific image_id, use its region and don't do quota-based fallback
        if template_image_id:
            job_logger.info(
                f"[Template] Using custom AMI: {template_image_id} in {template_region}"
            )
            resources_config = {
                "cloud": "aws",
                "accelerators": yaml_data["resources"].get("accelerators", "A100:8"),
                "disk_size": yaml_data["resources"].get("disk_size", "300GB"),
                "ports": yaml_data["resources"].get("ports", 8001),
                "image_id": template_image_id,
                "region": template_region,
            }

        # Update dynamic fields
        yaml_data["name"] = config.decision_id
        yaml_data["num_nodes"] = num_nodes
        yaml_data["resources"] = resources_config
        yaml_data["file_mounts"]["/data"]["source"] = s3_base
        yaml_data["envs"]["HF_TOKEN"] = hf_token

        # Add S3 model weight mount if requested
        if request.s3_models:
            model_mount_path = f"/models/{request.model_name}"
            yaml_data["file_mounts"][model_mount_path] = {
                "source": f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{request.model_name}",
                "mode": "COPY",
            }

        # Write final yaml
        Path(YAML_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        with open(YAML_OUTPUT, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    else:
        # Generic template - use old substitution method
        replace_run_dict = replace_run_vllm(
            request, config, job_dirname, logger=job_logger
        )
        run_string = update_template("templates/vllm_run", replace_run_dict)

        replace_yaml = {
            "name": config.decision_id,
            "num_nodes": num_nodes,
            "resources": resources_config,
            "run": run_string,
            "file_mounts./data.source": s3_base,
            "envs.HF_TOKEN": hf_token,
        }

        # Add S3 model weight mount if requested
        if request.s3_models:
            model_mount_path = f"/models/{request.model_name}"
            replace_yaml[f"file_mounts.{model_mount_path}.source"] = (
                f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/{request.model_name}"
            )
            replace_yaml[f"file_mounts.{model_mount_path}.mode"] = "COPY"

        update_yaml_file("templates/vllm.yaml", replace_yaml, YAML_OUTPUT)

    cm = get_cluster_manager()
    cm.register(config.decision_id, config.decision_id)

    # Construct S3 output path for later download
    output_s3_path = f"{s3_output_dir}/{request.output_file}"
    job_logger.info(f"[Job] Output will be saved to: {s3_output_dir}/")

    def monitor_and_download():
        """Background thread: stream logs, then download output when done."""
        try:
            sky.tail_logs(cluster_name=config.decision_id, job_id=job_id, follow=True)
            job_logger.info(
                f"[Job] {config.decision_id} completed. Downloading output..."
            )

            # Download output from S3 to local dir (base name, no prefix yet)
            local_path = download_output_from_s3(
                output_s3_path, job_dirname, logger=job_logger
            )

            # Determine success: both output file and metrics.csv must exist
            base_dir = Path(f"outputs/{job_dirname}")
            is_success = (
                local_path is not None
                and base_dir.exists()
                and (base_dir / "metrics.csv").exists()
            )

            # Rename dir with success-/failed- prefix
            status = "success" if is_success else "failed"
            prefixed_dirname = prefix_job_dirname(job_dirname, status)
            target_dir = Path(f"outputs/{prefixed_dirname}")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            job_logger.info(f"[Job] {status.upper()}: outputs/{prefixed_dirname}")
            close_job_logger(job_logger)
            base_dir.rename(target_dir)

        except Exception as e:
            job_logger.error(f"[Job] Error in monitor thread: {e}")
            # Ensure a failed- dir exists even if everything blew up
            failed_dirname = prefix_job_dirname(job_dirname, "failed")
            failed_dir = Path(f"outputs/{failed_dirname}")
            base_dir = Path(f"outputs/{job_dirname}")
            job_logger.info(f"[Job] FAILED: outputs/{failed_dirname}")
            close_job_logger(job_logger)
            if base_dir.exists() and not failed_dir.exists():
                failed_dir.parent.mkdir(parents=True, exist_ok=True)
                base_dir.rename(failed_dir)
            elif not failed_dir.exists():
                failed_dir.mkdir(parents=True, exist_ok=True)
        finally:
            cm.unregister(config.decision_id)

    try:
        job_logger.info(f"[SkyPilot] Launching cluster {config.decision_id}...")
        task = sky.Task.from_yaml(YAML_OUTPUT)
        result_id = sky.launch(task, cluster_name=config.decision_id, down=True, idle_minutes_to_autostop=10)
        job_id, handle = sky.stream_and_get(result_id, follow=True)
        job_logger.info(f"[SkyPilot] Launch complete. job_id={job_id}")

        # Stream logs in background and download when done
        threading.Thread(target=monitor_and_download, daemon=True).start()

    except Exception as e:
        job_logger.error(
            f"[SkyPilot] Failed to launch cluster {config.decision_id}: {e}"
        )
        close_job_logger(job_logger)
        cm.unregister(config.decision_id)
        raise Exception(f"Failed to launch cluster {config.decision_id}: {e}")


async def sp_launch_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    """Launch persistant vllm online deployment"""
    replace_run_dict = replace_run_vllm_online(request, config)
    run_string = update_template("templates/vllm_run_online", replace_run_dict)

    replace_yaml = {
        "name": config.decision_id,
        "num_nodes": config.num_nodes,
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
    "g6e.xlarge": "L40S",
    "g6e.2xlarge": "L40S",
    "g6e.4xlarge": "L40S",
    "g6e.8xlarge": "L40S",
    "g6e.12xlarge": "L40S",
    "g6e.16xlarge": "L40S",
    "g6e.24xlarge": "L40S",
    "g6e.48xlarge": "L40S",
    # G6 instances (L4)
    "g6.xlarge": "L4",
    "g6.2xlarge": "L4",
    "g6.4xlarge": "L4",
    "g6.8xlarge": "L4",
    "g6.12xlarge": "L4",
    "g6.16xlarge": "L4",
    "g6.24xlarge": "L4",
    "g6.48xlarge": "L4",
    # G5 instances (A10G)
    "g5.xlarge": "A10G",
    "g5.2xlarge": "A10G",
    "g5.4xlarge": "A10G",
    "g5.8xlarge": "A10G",
    "g5.12xlarge": "A10G",
    "g5.16xlarge": "A10G",
    "g5.24xlarge": "A10G",
    "g5.48xlarge": "A10G",
    # P4 instances (A100)
    "p4d.24xlarge": "A100",
    "p4de.24xlarge": "A100",
    # P5 instances (H100)
    "p5.48xlarge": "H100",
    # P3 instances (V100)
    "p3.2xlarge": "V100",
    "p3.8xlarge": "V100",
    "p3.16xlarge": "V100",
    "p3dn.24xlarge": "V100",
}


def get_vllm_config_template(
    model_name: str, instance_type: str, tp: int, pp: int, logger=None
) -> str:
    """
    Get the per-config vLLM template path if it exists.

    Template lookup order:
    1. vllm_{model}-{gpu}-tp{tp}-pp{pp}.yaml  (specific model+config)
    2. vllm_{gpu}.yaml                         (GPU-specific generic)
    3. vllm.yaml                               (generic fallback)

    Example: vllm_qwen2.5-72b-A100-tp8-pp1.yaml

    Returns path to specific template if exists, otherwise returns generic template.
    """
    import os

    _log = logger.info if logger else print

    # Normalize model name (e.g., "Qwen/Qwen2.5-72B-Instruct" -> "qwen2.5-72b")
    model_short = model_name.lower()
    if "/" in model_short:
        model_short = model_short.split("/")[-1]
    # Remove common suffixes
    for suffix in ["-instruct", "-chat", "-base", "-hf"]:
        model_short = model_short.replace(suffix, "")

    # Get GPU name from instance type
    gpu_name = INSTANCE_TO_GPU.get(instance_type, "unknown")

    # 1. Check for specific model+config template
    template_name = f"vllm_{model_short}-{gpu_name}-tp{tp}-pp{pp}.yaml"
    template_path = f"templates/vllm_configs/{template_name}"
    if os.path.exists(template_path):
        _log(f"[Template] Using per-config template: {template_path}")
        return template_path

    # 2. Check for GPU-specific generic template (e.g., vllm_A100.yaml)
    gpu_template_name = f"vllm_{gpu_name}.yaml"
    gpu_template_path = f"templates/vllm_configs/{gpu_template_name}"
    if os.path.exists(gpu_template_path):
        _log(f"[Template] Using GPU-specific template: {gpu_template_path}")
        return gpu_template_path

    # 3. Fall back to generic template
    _log(f"[Template] No specific template for {template_name}, using generic")
    return "templates/vllm.yaml"


def replace_run_vllm(
    request: BatchedRequest,
    config: MagicOutput,
    job_dirname: str = "output",
    logger=None,
):
    replace = {}

    if request.s3_models:
        replace["model"] = f"/models/{request.model_name}"
    else:
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
        calculated_len = int(
            (request.max_input_tokens + request.max_output_tokens) * 1.1
        )
        # Round up to nearest power of 2 for efficiency, clamp to reasonable range
        calculated_len = max(1024, min(calculated_len, 131072))
        replace["max_model_len"] = calculated_len
        _log = logger.info if logger else print
        _log(
            f"[Config] Calculated max_model_len={calculated_len} from max_input={request.max_input_tokens} + max_output={request.max_output_tokens}"
        )
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
    replace["max_num_seqs"] = (
        vllm_cfg.max_num_seqs if vllm_cfg and vllm_cfg.max_num_seqs else 256
    )
    replace["dtype"] = "auto"  # vLLM auto-detects based on model
    replace["kv_cache_dtype"] = (
        vllm_cfg.kv_cache_dtype if vllm_cfg and vllm_cfg.kv_cache_dtype else "auto"
    )

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


@app.post("/storage/upload")
async def upload_file_to_storage(
    file: UploadFile = File(...), remote_path: str = Form(...), user: str = Form(...)
):
    """Upload a file to storage backend using streaming via temp file."""
    try:
        logger.info(f"Uploading file for user {user} to {remote_path}")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            chunk_size = CHUNK_SIZE_MB
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        storage_uri = await storage_backend.upload_file(tmp_path, remote_path, user)
        os.unlink(tmp_path)
        logger.info(f"Successfully uploaded file to {storage_uri}")
        return {
            "status": "success",
            "storage_uri": storage_uri,
            "remote_path": remote_path,
            "user": user,
            "filename": file.filename,
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/storage/list/{user}")
async def list_user_files(user: str, prefix: str = ""):
    """List all files for a user in their storage space."""
    try:
        logger.info(f"Listing files for user {user} with prefix '{prefix}'")
        files = await storage_backend.list_files(prefix, user)
        return {
            "status": "success",
            "user": user,
            "prefix": prefix,
            "files": files,
            "count": len(files),
        }
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.get("/storage/download/{user}/{file_path:path}")
async def download_file_from_storage(user: str, file_path: str):
    """Download a file from storage backend using streaming."""
    try:
        logger.info(f"Downloading file {file_path} for user {user}")
        filename = file_path.split("/")[-1] or "download"

        async def file_stream_iterator():
            async for chunk in storage_backend.stream_file(file_path, user):
                yield chunk

        return StreamingResponse(
            file_stream_iterator(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.delete("/storage/delete/{user}/{file_path:path}")
async def delete_file_from_storage(user: str, file_path: str):
    """Delete a file from storage backend."""
    try:
        logger.info(f"Deleting file {file_path} for user {user}")
        success = await storage_backend.delete_file(file_path, user)
        if success:
            return {
                "status": "success",
                "message": f"File {file_path} deleted successfully",
                "user": user,
                "file_path": file_path,
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete file {file_path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.get("/storage/exists/{user}/{file_path:path}")
async def check_file_exists(user: str, file_path: str):
    """Check if a file exists in storage backend."""
    try:
        exists = await storage_backend.file_exists(file_path, user)
        return {
            "status": "success",
            "user": user,
            "file_path": file_path,
            "exists": exists,
        }
    except Exception as e:
        logger.error(f"Error checking file existence: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking file: {str(e)}")


@app.post("/storage/multipart/start")
async def multipart_start(remote_path: str = Form(...), user: str = Form(...)):
    return await storage_backend.multipart_upload_start(remote_path, user)


@app.post("/storage/multipart/sign-part")
async def multipart_sign_part(
    upload_id: str = Form(...),
    user: str = Form(...),
    remote_path: str = Form(...),
    part_number: int = Form(...),
    expires: int = Form(600),
):
    return await storage_backend.multipart_sign_part(
        upload_id, user, remote_path, part_number, expires
    )


@app.post("/storage/multipart/complete")
async def multipart_complete(
    user: str = Form(...),
    remote_path: str = Form(...),
    upload_id: str = Form(...),
    parts: str = Form(...),
):
    parts_list = json.loads(parts)
    if not isinstance(parts_list, list):
        raise ValueError("parts is not a list")
    return await storage_backend.multipart_complete(
        remote_path, user, upload_id, parts_list
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=26336)
