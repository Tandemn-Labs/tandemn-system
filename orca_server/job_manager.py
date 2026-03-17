"""
Job lifecycle management: logging, tracking, metrics, and output download.
"""

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Literal, Optional

import requests

from orca_server.config import INSTANCE_TO_GPU

logger = logging.getLogger(__name__)
from models.requests import BatchedRequest
from models.resources import MagicOutput
from quota.tracker import JobSpec, JobState

# Extended JobRecord that supports chunked jobs
@dataclass
class JobRecord:
    state: JobState
    status: Literal[
        "queued", "launching", "running", "succeeded", "failed", "cancelled",
        "loading_model", "model_ready", "generating",
    ] = "queued"
    endpoint_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    head_ip: Optional[str] = None
    output_s3_path: Optional[str] = None
    # Chunked job extensions
    is_chunked: bool = False
    total_chunks: Optional[int] = None
    num_replicas: int = 1
    _job_dirname: Optional[str] = None


# --------------------------------------------------------------------------- #
# Job logging
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Job directory naming
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Cluster manager
# --------------------------------------------------------------------------- #

class ClusterManager:
    """Manages active SkyPilot clusters to prevent zombie instances."""

    def __init__(self):
        self.active_clusters: Dict[str, dict] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._persist_set: set = set()
        self._job_clusters: Dict[str, list] = {}  # job_id → [cluster_names]
        self._replica_states: Dict[str, Dict[str, dict]] = {}  # job_id → {replica_id → state}
        self.lock = Lock()

    def register(self, cluster_name: str, job_id: str, region: str = None,
                 market: str = None, instance_type: str = None, num_instances: int = None):
        with self.lock:
            self.active_clusters[cluster_name] = {
                "job_id": job_id, "status": "active",
                "region": region, "market": market,
                "instance_type": instance_type, "num_instances": num_instances,
            }
            logger.info(f"[ClusterManager] Registered cluster: {cluster_name}")

    def unregister(self, cluster_name: str):
        with self.lock:
            if cluster_name in self.active_clusters:
                del self.active_clusters[cluster_name]
                logger.info(f"[ClusterManager] Unregistered cluster: {cluster_name}")

    def register_thread(self, cluster_name: str, thread: threading.Thread):
        with self.lock:
            self._threads[cluster_name] = thread

    def unregister_thread(self, cluster_name: str):
        with self.lock:
            self._threads.pop(cluster_name, None)

    def mark_persist(self, cluster_name: str):
        with self.lock:
            self._persist_set.add(cluster_name)

    def register_for_job(self, job_id: str, cluster_name: str):
        with self.lock:
            if job_id not in self._job_clusters:
                self._job_clusters[job_id] = []
            if cluster_name not in self._job_clusters[job_id]:
                self._job_clusters[job_id].append(cluster_name)

    def get_job_clusters(self, job_id: str) -> list:
        with self.lock:
            return list(self._job_clusters.get(job_id, []))

    def set_replica_state(self, job_id: str, replica_id: str, **kwargs):
        with self.lock:
            if job_id not in self._replica_states:
                self._replica_states[job_id] = {}
            if replica_id not in self._replica_states[job_id]:
                self._replica_states[job_id][replica_id] = {}
            self._replica_states[job_id][replica_id].update(kwargs)

    def get_replica_states(self, job_id: str) -> Dict[str, dict]:
        with self.lock:
            return dict(self._replica_states.get(job_id, {}))

    def get_active_threads(self) -> Dict[str, threading.Thread]:
        with self.lock:
            return dict(self._threads)

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
            logger.error(f"[ClusterManager] Error terminating {cluster_name}: {e}")
            return False


_cluster_manager = None


def get_cluster_manager() -> ClusterManager:
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


def sky_down_with_retry(cluster_name: str, max_attempts: int = 3, delay: float = 10.0) -> bool:
    """Tear down a SkyPilot cluster with retries (handles security-group race)."""
    import sky
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"[Teardown] sky.down({cluster_name}) attempt {attempt}/{max_attempts}")
            sky.down(cluster_name)
            logger.info(f"[Teardown] Cluster {cluster_name} destroyed.")
            return True
        except Exception as e:
            logger.warning(f"[Teardown] Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay)
    # Final fallback: subprocess
    logger.info(f"[Teardown] SDK retries exhausted, falling back to subprocess sky down -y {cluster_name}")
    try:
        result = subprocess.run(
            ["sky", "down", "-y", cluster_name],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"[Teardown] Subprocess fallback succeeded for {cluster_name}.")
            return True
        logger.error(f"[Teardown] Subprocess fallback failed: {result.stderr}")
    except Exception as e:
        logger.error(f"[Teardown] Subprocess fallback error: {e}")
    return False


# --------------------------------------------------------------------------- #
# Job tracker
# --------------------------------------------------------------------------- #

_job_tracker = None


def get_job_tracker() -> "JobTracker":
    global _job_tracker
    if _job_tracker is None:
        _job_tracker = JobTracker()
    return _job_tracker


class JobTracker:
    def __init__(self):
        self.jobs: Dict[str, JobRecord] = {}
        self.lock = Lock()

    def build_job_state_batched(self, req: BatchedRequest, config: MagicOutput):
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
                slo_hours=req.slo_deadline_hours or 4,
                region="us-east-1",
                market="spot",
            ),
            submitted_at=time.time(),
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

    def set_chunked_info(self, job_id: str, total_chunks: int, num_replicas: int):
        with self.lock:
            rec = self.jobs.get(job_id)
            if rec:
                rec.is_chunked = True
                rec.total_chunks = total_chunks
                rec.num_replicas = num_replicas
                rec.last_updated_at = time.time()

    def update_status(
        self,
        job_id: str,
        status: Literal[
            "queued", "launching", "loading_model", "model_ready",
            "generating", "running", "succeeded", "failed", "cancelled",
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


# --------------------------------------------------------------------------- #
# vLLM metrics parser
# --------------------------------------------------------------------------- #

METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{(?P<labels>[^}]*)\})?"  # optional {k="v",...}
    r"\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def parse_labels(labels_blob: str) -> dict[str, str]:
    # Very small parser; works for the usual key="value" labels vLLM emits.
    out: dict[str, str] = {}
    if not labels_blob:
        return out
    for part in labels_blob.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def sum_metric(
    text: str, metric_name: str, *, where: dict[str, str] | None = None
) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        if m.group("name") != metric_name:
            continue
        labels = parse_labels(m.group("labels") or "")
        if where and any(labels.get(k) != v for k, v in where.items()):
            continue
        total += float(m.group("value"))
    return total


_METRIC_ALIASES = {
    "vllm:generation_tokens_total": "vllm:generation_tokens",
    "vllm:prompt_tokens_total":     "vllm:prompt_tokens",
    "vllm:request_success_total":   "vllm:request_success",
    "vllm:num_preemptions_total":   "vllm:num_preemptions",
    "vllm:gpu_cache_usage_perc":    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total": "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits_total":    "vllm:prefix_cache_hits",
}


def sum_metric_compat(text: str, name: str, **kw) -> float:
    """sum_metric with fallback to v0.10.0 metric names."""
    val = sum_metric(text, name, **kw)
    if val == 0.0 and name in _METRIC_ALIASES:
        val = sum_metric(text, _METRIC_ALIASES[name], **kw)
    return val


def get_vllm_progress(
    endpoint_url: str, *, model_name: str | None = None
) -> tuple[int, int]:
    r = requests.get(endpoint_url + "/metrics", timeout=5)
    r.raise_for_status()
    text = r.text

    # Optional: only count metrics for a specific model
    filt = {"model_name": model_name} if model_name else None

    running = sum_metric(text, "vllm:num_requests_running", where=filt)
    waiting = sum_metric(text, "vllm:num_requests_waiting", where=filt)

    # "done requests" = sum across finished_reason labels
    done = sum_metric(text, "vllm:request_success_total", where=filt)

    queued = running + waiting
    return int(done), int(queued)


# Future: wired up when chunked batch queue is implemented
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


# --------------------------------------------------------------------------- #
# S3 output download
# --------------------------------------------------------------------------- #

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
