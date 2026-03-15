#!/usr/bin/env python3
"""
vLLM batch runner — HTTP server mode with full profiling-equivalent observability.

Starts vLLM as an OpenAI-compatible HTTP server, sends requests via SSE streaming,
collects GPU hardware metrics (pynvml), scheduler timeseries (Prometheus scraping),
and writes profiling-equivalent metrics.csv + output.jsonl.

Matches tandemn-profiling methodology: warmup → pre-scrape → benchmark → post-scrape → delta.
"""

try:
    import vllm_compat_patch
except ImportError:
    pass

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiohttp
import requests

# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------
SERVER_PORT = int(os.getenv("VLLM_PORT", "8001"))
BASE_URL = f"http://localhost:{SERVER_PORT}"

SIDECAR_INTERVAL_SEC = 5
MAX_CONCURRENT = 256
PROGRESS_FILE = "/tmp/vllm_progress.json"
WARMUP_REQUESTS = 0  # disabled — batch jobs maximize utilization, no spare requests


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def write_progress(done: int, total: int, status: str = "running"):
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"done": done, "total": total, "status": status, "timestamp": time.time()}, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prometheus helpers — inline, no orca_server imports on cluster
# ---------------------------------------------------------------------------

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def _parse_labels(labels_blob: str) -> dict:
    out = {}
    if not labels_blob:
        return out
    for part in labels_blob.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def sum_metric(text: str, metric_name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m or m.group("name") != metric_name:
            continue
        total += float(m.group("value"))
    return total


def _parse_histogram_buckets(text: str, metric_name: str) -> list:
    bucket_name = f"{metric_name}_bucket"
    buckets: dict = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m or m.group("name") != bucket_name:
            continue
        labels = _parse_labels(m.group("labels") or "")
        le_str = labels.get("le")
        if le_str is None:
            continue
        le = float("inf") if le_str == "+Inf" else float(le_str)
        buckets[le] = float(m.group("value"))
    return sorted(buckets.items())


def histogram_quantile_from_buckets(buckets: list, quantile: float) -> Optional[float]:
    """Compute quantile from [(le, count), ...] pairs. Returns seconds or None."""
    if not buckets or buckets[-1][1] == 0:
        return None
    total = buckets[-1][1]
    target = quantile * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if le == float("inf"):
            if count >= target and prev_count < target:
                return prev_le
            continue
        if count >= target:
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return prev_le


def histogram_quantile(text: str, metric_name: str, quantile: float) -> Optional[float]:
    """Compute a quantile from a Prometheus histogram. Returns ms or None if no data."""
    buckets = _parse_histogram_buckets(text, metric_name)
    val = histogram_quantile_from_buckets(buckets, quantile)
    return val * 1000.0 if val is not None else None


def _percentile(lst: list, p: int) -> Optional[float]:
    """Compute percentile p (0-100) of a list of floats. Returns None if empty."""
    if not lst:
        return None
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    idx = (n - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_lst[lower] * (1 - weight) + sorted_lst[upper] * weight


# ---------------------------------------------------------------------------
# Prometheus delta helpers — warmup subtraction (matches profiling repo)
# ---------------------------------------------------------------------------

def _scrape_prom() -> str:
    """Scrape the local vLLM /metrics endpoint. Returns raw text or empty string."""
    try:
        return requests.get(f"{BASE_URL}/metrics", timeout=10).text
    except Exception:
        return ""


def _parse_prom_counters(text: str) -> Dict[str, float]:
    """Extract all counter/gauge values from Prometheus text."""
    counters = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        # Skip histogram bucket/sum/count lines — handled separately
        if "_bucket" in name:
            continue
        counters[name] = float(m.group("value"))
    return counters


def _delta_histogram_buckets(pre_text: str, post_text: str, metric_name: str) -> list:
    """Compute bucket-level deltas between pre and post scrapes."""
    pre = dict(_parse_histogram_buckets(pre_text, metric_name))
    post = dict(_parse_histogram_buckets(post_text, metric_name))
    delta = []
    for le in sorted(post.keys()):
        delta.append((le, post[le] - pre.get(le, 0.0)))
    return delta


def _delta_counter(pre_text: str, post_text: str, metric_name: str) -> float:
    """Compute delta for a single counter between pre and post scrapes."""
    return sum_metric(post_text, metric_name) - sum_metric(pre_text, metric_name)


# ---------------------------------------------------------------------------
# GPU hardware monitor (pynvml) — matches profiling GPUMonitorActor
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Sample GPU SM%, memory bandwidth%, memory usage via pynvml every 0.5s."""

    def __init__(self, sample_interval: float = 0.5):
        self._interval = sample_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._timeseries: list = []
        self._pynvml = None
        self._device_count = 0
        self._start_time = 0.0

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._device_count = pynvml.nvmlDeviceGetCount()
            print(f"[GPUMonitor] Initialized with {self._device_count} GPUs")
        except Exception as e:
            print(f"[GPUMonitor] pynvml not available: {e}")

    def start(self):
        if not self._pynvml:
            return
        self._timeseries = []
        self._start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        pynvml = self._pynvml
        while not self._stop.is_set():
            sample = {"t": round(time.time() - self._start_time, 3)}
            for i in range(self._device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sample[f"gpu{i}_sm_pct"] = util.gpu
                    sample[f"gpu{i}_membw_pct"] = util.memory
                    sample[f"gpu{i}_mem_gb"] = round(mem.used / (1024 ** 3), 2)
                    sample[f"gpu{i}_mem_pct"] = round((mem.used / mem.total) * 100, 1) if mem.total else 0
                except Exception:
                    pass
            if len(sample) > 1:
                self._timeseries.append(sample)
            self._stop.wait(self._interval)

    def get_timeseries(self) -> list:
        return self._timeseries

    def get_summary(self) -> dict:
        if not self._timeseries:
            return {}
        all_sm, all_membw, all_mem = [], [], []
        for s in self._timeseries:
            for k, v in s.items():
                if "_sm_pct" in k:
                    all_sm.append(v)
                elif "_membw_pct" in k:
                    all_membw.append(v)
                elif "_mem_pct" in k:
                    all_mem.append(v)
        summary = {"gpu_samples": len(self._timeseries)}
        if all_sm:
            summary["avg_sm_util_pct"] = round(sum(all_sm) / len(all_sm), 2)
            summary["max_sm_util_pct"] = round(max(all_sm), 2)
        if all_membw:
            summary["avg_mem_bw_util_pct"] = round(sum(all_membw) / len(all_membw), 2)
            summary["max_mem_bw_util_pct"] = round(max(all_membw), 2)
        if all_mem:
            summary["avg_mem_util_pct"] = round(sum(all_mem) / len(all_mem), 2)
            summary["max_mem_util_pct"] = round(max(all_mem), 2)
        return summary


# ---------------------------------------------------------------------------
# Scheduler / Prometheus metrics poller — local timeseries (matches profiling)
# ---------------------------------------------------------------------------

class MetricsPoller:
    """Poll /metrics every 0.5s for scheduler state + KV cache utilization."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._timeseries: list = []
        self._start_time = 0.0

    def start(self):
        self._timeseries = []
        self._start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="metrics-poller")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while not self._stop.is_set():
            try:
                text = requests.get(f"{BASE_URL}/metrics", timeout=4).text
                sample = {
                    "t": round(time.time() - self._start_time, 3),
                    "running": int(sum_metric(text, "vllm:num_requests_running")),
                    "waiting": int(sum_metric(text, "vllm:num_requests_waiting")),
                    "swapped": int(sum_metric(text, "vllm:num_requests_swapped")),
                    "kv_cache_util_pct": round(sum_metric(text, "vllm:gpu_cache_usage_perc") * 100, 1),
                }
                self._timeseries.append(sample)
            except Exception:
                pass
            self._stop.wait(self._interval)

    def get_timeseries(self) -> list:
        return self._timeseries

    def get_summary(self) -> dict:
        if not self._timeseries:
            return {}
        # Filter to active samples (running > 0) to avoid diluting with idle periods
        active = [s for s in self._timeseries if s.get("running", 0) > 0]
        source = active if active else self._timeseries
        summary = {}
        for key in ("running", "waiting", "swapped", "kv_cache_util_pct"):
            vals = [s.get(key, 0) for s in source if key in s]
            if vals:
                summary[f"{key}_avg"] = round(sum(vals) / len(vals), 2)
                summary[f"{key}_max"] = max(vals)
        summary["scheduler_samples"] = len(self._timeseries)
        summary["scheduler_active_samples"] = len(active)
        return summary


# ---------------------------------------------------------------------------
# Sidecar — push to control plane (bonus, works when reachable)
# ---------------------------------------------------------------------------

def _sidecar_loop(stop_event: threading.Event, orca_url: str, orca_key: str, job_id: str):
    if not orca_url:
        return

    ingest_url = f"{orca_url}/job/{job_id}/metrics/ingest"
    headers = {"Content-Type": "application/json"}
    if orca_key:
        headers["Authorization"] = f"Bearer {orca_key}"

    buffer = []
    last_push = time.time()

    while not stop_event.is_set():
        try:
            prom_text = requests.get(f"{BASE_URL}/metrics", timeout=4).text
            buffer.append({"timestamp": time.time(), "prometheus_text": prom_text})
        except Exception:
            pass

        if time.time() - last_push >= SIDECAR_INTERVAL_SEC and buffer:
            try:
                payload: dict = {"snapshots": buffer}
                try:
                    with open(PROGRESS_FILE) as pf:
                        prog = json.load(pf)
                    payload["done"] = prog.get("done", 0)
                    payload["total"] = prog.get("total", 0)
                except Exception:
                    pass
                requests.post(ingest_url, json=payload, headers=headers, timeout=5)
                buffer = []
                last_push = time.time()
            except Exception:
                pass

        stop_event.wait(timeout=1.0)

    if buffer:
        try:
            requests.post(ingest_url, json={"snapshots": buffer}, headers=headers, timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CSV / JSONL helpers
# ---------------------------------------------------------------------------

def write_metrics_csv(output_path: str, metrics: Dict[str, Any]):
    output_dir = os.path.dirname(output_path)
    metrics_file = os.path.join(output_dir, "metrics.csv") if output_dir else "metrics.csv"
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for metric, value in metrics.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                writer.writerow([metric, value])
        print(f"[Runner] Wrote metrics to {metrics_file}")
    except Exception as e:
        print(f"[Runner] Warning: Failed to write metrics: {e}")


def calculate_percentiles(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"p50": 0, "p90": 0, "p99": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = (n - 1) * p / 100
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

    return {"p50": percentile(50), "p90": percentile(90), "p99": percentile(99)}


def load_requests(input_file: str) -> List[Dict[str, Any]]:
    reqs = []
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    reqs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Runner] Warning: Skipping invalid JSON on line {line_num}: {e}")
    return reqs


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm_server(args) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", str(SERVER_PORT),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--pipeline-parallel-size", str(args.pipeline_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs),
        "--dtype", args.dtype,
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--disable-log-requests",
        "--trust-remote-code",
    ]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def wait_for_server(timeout_sec: int = 600) -> float:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            if requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200:
                return time.time() - start
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM not ready after {timeout_sec}s")


def shutdown_server(proc: subprocess.Popen, timeout: int = 60):
    print("[Runner] Shutting down vLLM server...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        print("[Runner] vLLM server exited cleanly.")
    except subprocess.TimeoutExpired:
        print("[Runner] Graceful shutdown timed out, sending SIGKILL.")
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Async benchmark client — per-request TTFT / E2E / client-side TPOT via SSE
# ---------------------------------------------------------------------------

async def send_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    req_dict: dict,
    model_name: str,
) -> dict:
    body = req_dict.get("body", {})
    payload = {
        "model": model_name,
        "messages": body.get("messages", []),
        "max_tokens": body.get("max_tokens", 256),
        "temperature": body.get("temperature", 0.7),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft = None
    output_tokens = 0
    prompt_tokens = 0
    finish_reason = "stop"
    content_parts = []

    async with semaphore:
        try:
            async with session.post(
                f"{BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status == 400:
                    text = await resp.text()
                    status = "skipped" if "context_length_exceeded" in text.lower() else "error"
                    return {
                        "status": status,
                        "custom_id": req_dict.get("custom_id"),
                        "error": text,
                    }
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        delta_content = delta.get("content", "")
                        if ttft is None and delta_content:
                            ttft = time.time() - t0
                        if delta_content:
                            content_parts.append(delta_content)
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", 0)
                        prompt_tokens = usage.get("prompt_tokens", 0)
        except Exception as e:
            return {
                "status": "error",
                "custom_id": req_dict.get("custom_id"),
                "error": str(e),
            }

    e2e = time.time() - t0
    # Client-side TPOT: (e2e - ttft) / (output_tokens - 1), matching profiling repo
    tpot_client_s = None
    if ttft is not None and output_tokens > 1:
        tpot_client_s = (e2e - ttft) / (output_tokens - 1)

    return {
        "status": "success",
        "custom_id": req_dict.get("custom_id"),
        "ttft_s": ttft,
        "e2e_s": e2e,
        "tpot_client_s": tpot_client_s,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
        "content": "".join(content_parts),
    }


async def run_benchmark(requests_list: List[Dict], model_name: str) -> List[Dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    total = len(requests_list)
    results = [None] * total
    done_count = 0

    async def _tracked(idx, req):
        nonlocal done_count
        r = await send_one(session, semaphore, req, model_name)
        done_count += 1
        write_progress(done_count, total)
        results[idx] = r
        return r

    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[_tracked(i, req) for i, req in enumerate(requests_list)])

    return results


async def run_warmup(requests_list: List[Dict], model_name: str, n: int = 5):
    """Send a few warmup requests so Prometheus histograms don't include cold-start latency."""
    if not requests_list or n <= 0:
        return
    semaphore = asyncio.Semaphore(2)
    connector = aiohttp.TCPConnector(limit=4)
    warmup_batch = requests_list[:n]
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_one(session, semaphore, req, model_name) for req in warmup_batch]
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Output JSONL writer
# ---------------------------------------------------------------------------

def write_output_jsonl(results: List[Dict], output_path: str, model_name: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, r in enumerate(results):
            custom_id = r.get("custom_id", f"req-{i}")
            if r["status"] == "success":
                record = {
                    "id": f"vllm-{i}",
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": f"vllm-batch-{i}",
                        "body": {
                            "id": f"chatcmpl-{i}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": r.get("content", ""),
                                },
                                "finish_reason": r.get("finish_reason", "stop"),
                            }],
                            "usage": {
                                "prompt_tokens": r.get("prompt_tokens", 0),
                                "completion_tokens": r.get("output_tokens", 0),
                                "total_tokens": r.get("prompt_tokens", 0) + r.get("output_tokens", 0),
                            },
                        },
                    },
                    "error": None,
                }
            else:
                record = {
                    "id": f"vllm-skipped-{custom_id}",
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 400,
                        "request_id": f"vllm-skipped-{custom_id}",
                        "body": None,
                    },
                    "error": {
                        "type": "context_length_exceeded" if r["status"] == "skipped" else "inference_error",
                        "message": r.get("error", ""),
                    },
                }
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Metrics builder — profiling-equivalent CSV
# ---------------------------------------------------------------------------

def build_metrics(
    results: List[Dict],
    pre_prom_text: str,
    post_prom_text: str,
    args,
    job_start_time: float,
    job_start_timestamp: str,
    model_load_sec: float,
    generation_time: float,
    gpu_summary: dict,
    scheduler_summary: dict,
    model_info: dict | None = None,
) -> Dict[str, Any]:
    job_end_time = time.time()
    total_runtime = job_end_time - job_start_time

    successes = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] != "success"]

    # --- Token stats from client-side usage ---
    prompt_counts = [r.get("prompt_tokens", 0) for r in successes]
    output_counts = [r.get("output_tokens", 0) for r in successes]
    total_prompt_client = sum(prompt_counts)
    total_output_client = sum(output_counts)

    prompt_pct = calculate_percentiles(prompt_counts)
    output_pct = calculate_percentiles(output_counts)

    # --- Server-side throughput from Prometheus counter deltas (ground truth) ---
    server_prompt_toks = _delta_counter(pre_prom_text, post_prom_text, "vllm:prompt_tokens_total")
    server_gen_toks = _delta_counter(pre_prom_text, post_prom_text, "vllm:generation_tokens_total")
    server_total_toks = server_prompt_toks + server_gen_toks
    # Fall back to client-side if Prometheus counters are zero (e.g., old vLLM version)
    total_prompt = server_prompt_toks if server_prompt_toks > 0 else total_prompt_client
    total_output = server_gen_toks if server_gen_toks > 0 else total_output_client
    total_tokens = total_prompt + total_output

    # --- Throughput ---
    req_per_sec = len(successes) / generation_time if generation_time > 0 else 0.0
    tok_per_sec = total_tokens / generation_time if generation_time > 0 else 0.0
    out_tok_per_sec = total_output / generation_time if generation_time > 0 else 0.0
    in_tok_per_sec = total_prompt / generation_time if generation_time > 0 else 0.0

    # --- Client-side latency (s → ms) ---
    ttfts_ms = [r["ttft_s"] * 1000 for r in successes if r.get("ttft_s") is not None]
    e2es_ms = [r["e2e_s"] * 1000 for r in successes if r.get("e2e_s") is not None]
    tpots_client_ms = [r["tpot_client_s"] * 1000 for r in successes if r.get("tpot_client_s") is not None]

    # --- Server-side latency from Prometheus histogram DELTAS ---
    delta_ttft = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:time_to_first_token_seconds")
    delta_tpot = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:time_per_output_token_seconds")
    delta_e2e = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:e2e_request_latency_seconds")

    def _hq(buckets, q):
        v = histogram_quantile_from_buckets(buckets, q)
        return v * 1000.0 if v is not None else None

    # --- Preemptions ---
    num_preemptions = int(_delta_counter(pre_prom_text, post_prom_text, "vllm:num_preemptions_total"))

    # --- Cost efficiency ---
    price_per_hour = _get_price_per_hour(args.instance_type)
    elapsed_hours = generation_time / 3600
    cost_for_run = price_per_hour * elapsed_hours if price_per_hour else None
    tokens_per_dollar = round(total_tokens / cost_for_run, 2) if cost_for_run and cost_for_run > 0 else None

    metrics = {
        # === TIMESTAMPS ===
        "job_start_timestamp": job_start_timestamp,
        "job_end_timestamp": datetime.now().isoformat(),

        # === RUNTIME ===
        "total_runtime_sec": total_runtime,
        "model_load_time_sec": model_load_sec,
        "generation_time_sec": generation_time,

        # === WORKLOAD ===
        "num_requests_total": len(results),
        "num_requests_completed": len(successes),
        "num_requests_skipped": len(skipped),
        "total_input_tokens": total_prompt,
        "avg_input_tokens": total_prompt / len(prompt_counts) if prompt_counts else 0,
        "p50_input_tokens": prompt_pct["p50"],
        "p90_input_tokens": prompt_pct["p90"],
        "p99_input_tokens": prompt_pct["p99"],
        "min_input_tokens": min(prompt_counts) if prompt_counts else 0,
        "max_input_tokens": max(prompt_counts) if prompt_counts else 0,
        "total_output_tokens": total_output,
        "avg_output_tokens": total_output / len(output_counts) if output_counts else 0,
        "p50_output_tokens": output_pct["p50"],
        "p90_output_tokens": output_pct["p90"],
        "p99_output_tokens": output_pct["p99"],
        "min_output_tokens": min(output_counts) if output_counts else 0,
        "max_output_tokens": max(output_counts) if output_counts else 0,
        "total_tokens": total_tokens,

        # === THROUGHPUT (server-side ground truth) ===
        "throughput_requests_per_sec": req_per_sec,
        "throughput_tokens_per_sec": tok_per_sec,
        "throughput_output_tokens_per_sec": out_tok_per_sec,
        "throughput_input_tokens_per_sec": in_tok_per_sec,

        # === CLIENT-SIDE LATENCY (ms) ===
        "ttft_ms_p50": _percentile(ttfts_ms, 50),
        "ttft_ms_p95": _percentile(ttfts_ms, 95),
        "ttft_ms_p99": _percentile(ttfts_ms, 99),
        "e2e_ms_p50": _percentile(e2es_ms, 50),
        "e2e_ms_p95": _percentile(e2es_ms, 95),
        "e2e_ms_p99": _percentile(e2es_ms, 99),
        "tpot_client_ms_p50": _percentile(tpots_client_ms, 50),
        "tpot_client_ms_p95": _percentile(tpots_client_ms, 95),
        "tpot_client_ms_p99": _percentile(tpots_client_ms, 99),

        # === SERVER-SIDE LATENCY FROM PROMETHEUS HISTOGRAM DELTAS (ms) ===
        "tpot_ms_p50": _hq(delta_tpot, 0.50),
        "tpot_ms_p95": _hq(delta_tpot, 0.95),
        "tpot_ms_p99": _hq(delta_tpot, 0.99),
        "ttft_server_ms_p50": _hq(delta_ttft, 0.50),
        "ttft_server_ms_p95": _hq(delta_ttft, 0.95),
        "ttft_server_ms_p99": _hq(delta_ttft, 0.99),
        "e2e_server_ms_p50": _hq(delta_e2e, 0.50),
        "e2e_server_ms_p95": _hq(delta_e2e, 0.95),
        "e2e_server_ms_p99": _hq(delta_e2e, 0.99),

        # === PREEMPTIONS ===
        "num_preemptions": num_preemptions,

        # === GPU HARDWARE UTILIZATION (pynvml) ===
        "avg_sm_util_pct": gpu_summary.get("avg_sm_util_pct"),
        "max_sm_util_pct": gpu_summary.get("max_sm_util_pct"),
        "avg_mem_bw_util_pct": gpu_summary.get("avg_mem_bw_util_pct"),
        "max_mem_bw_util_pct": gpu_summary.get("max_mem_bw_util_pct"),
        "avg_mem_util_pct": gpu_summary.get("avg_mem_util_pct"),
        "max_mem_util_pct": gpu_summary.get("max_mem_util_pct"),
        "gpu_samples": gpu_summary.get("gpu_samples", 0),

        # === SCHEDULER TIMESERIES (from local MetricsPoller) ===
        "running_avg": scheduler_summary.get("running_avg"),
        "running_max": scheduler_summary.get("running_max"),
        "waiting_avg": scheduler_summary.get("waiting_avg"),
        "waiting_max": scheduler_summary.get("waiting_max"),
        "kv_cache_util_pct_avg": scheduler_summary.get("kv_cache_util_pct_avg"),
        "kv_cache_util_pct_max": scheduler_summary.get("kv_cache_util_pct_max"),
        "scheduler_samples": scheduler_summary.get("scheduler_samples", 0),

        # === COST EFFICIENCY ===
        "price_per_hour": price_per_hour,
        "cost_for_run_usd": cost_for_run,
        "tokens_per_dollar": tokens_per_dollar,

        # === MODEL CONFIG ===
        "model_name": args.model,
        "quantization": args.quantization,

        # === INFRASTRUCTURE ===
        "cloud_provider": args.cloud,
        "instance_type": args.instance_type,
        "gpu_name": args.gpu_name,

        # === ENGINE CONFIG ===
        "engine": args.engine,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
    }

    # === MODEL METADATA (from HuggingFace AutoConfig) ===
    if model_info:
        for k in ("model_architecture", "params_billion", "is_moe",
                   "num_experts_active", "vocab_size"):
            if k in model_info:
                metrics[k] = model_info[k]

    # === GPU HARDWARE SPECS ===
    gpu_spec = GPU_SPECS.get(args.gpu_name, {})
    if gpu_spec:
        metrics["gpu_mem_gb"] = gpu_spec["gpu_mem_gb"]
        metrics["gpu_tflops_fp16"] = gpu_spec["gpu_tflops_fp16"]
        metrics["gpu_bandwidth_gbps"] = gpu_spec["gpu_bandwidth_gbps"]
        metrics["gpu_model"] = gpu_spec.get("model", args.gpu_name)
        metrics["gpu_generation"] = gpu_spec.get("generation")
        metrics["interconnect"] = gpu_spec.get("interconnect")

    # === STATIC METADATA ===
    metrics["num_nodes"] = int(os.getenv("SKYPILOT_NUM_NODES", "1"))
    metrics["precision"] = "bfloat16" if args.dtype == "auto" else args.dtype
    metrics["runtime_stack"] = "vllm 0.10.0"

    # === DERIVED (canonical schema parity) ===
    mi = model_info or {}
    params_b = mi.get("params_billion")
    tp = args.tensor_parallel_size
    num_kv_heads = mi.get("num_key_value_heads")
    num_attn_heads = mi.get("num_attention_heads")
    gpu_count = tp * args.pipeline_parallel_size

    # Model sizing
    model_size_gb = round(params_b * 2, 2) if params_b else None  # bf16
    metrics["model_size_gb"] = model_size_gb
    if params_b and gpu_count:
        metrics["params_per_gpu"] = round(params_b / gpu_count, 2)
    if model_size_gb and gpu_spec:
        metrics["vram_headroom_gb"] = round(gpu_spec["gpu_mem_gb"] * gpu_count - model_size_gb, 1)
        metrics["model_fits_single_gpu"] = 1 if model_size_gb <= gpu_spec["gpu_mem_gb"] else 0

    # Attention / GQA
    if num_attn_heads and num_kv_heads and num_kv_heads > 0:
        metrics["attention_heads_per_kv_head"] = round(num_attn_heads / num_kv_heads, 2)
    if num_kv_heads and tp:
        metrics["kv_heads_per_tp"] = round(num_kv_heads / tp, 2)

    # Hardware efficiency ratios
    if params_b and params_b > 0 and gpu_spec:
        metrics["bandwidth_per_param"] = round(gpu_spec["gpu_bandwidth_gbps"] * tp / params_b, 2)
        metrics["flops_per_param"] = round(gpu_spec["gpu_tflops_fp16"] * tp / params_b, 2)

    # Topology
    metrics["crosses_node_boundary"] = 1 if args.pipeline_parallel_size > 1 else 0

    # Prefill / decode cost split
    if price_per_hour and in_tok_per_sec > 0:
        metrics["cost_per_1m_tokens_prefill_usd"] = round(price_per_hour / (in_tok_per_sec * 3.6), 4)
    if price_per_hour and out_tok_per_sec > 0:
        metrics["cost_per_1m_tokens_decode_usd"] = round(price_per_hour / (out_tok_per_sec * 3.6), 4)

    # Feature flags (static for Orca batch jobs)
    metrics["kv_offload_target"] = "None"

    # Full model config JSON
    if mi.get("model_config_json"):
        metrics["model_config_json"] = mi["model_config_json"]

    return metrics


# ---------------------------------------------------------------------------
# Model introspection — static metadata from HuggingFace config
# ---------------------------------------------------------------------------

# Confirmed against tandemn-profiling/roofline/vllm/automatic_launch_1.py GPU_CONFIGS
GPU_SPECS = {
    "L40S":     {"gpu_mem_gb": 48, "gpu_tflops_fp16": 362, "gpu_bandwidth_gbps": 864, "generation": "Ada Lovelace", "interconnect": "PCIe", "model": "NVIDIA L40S"},
    "L4":       {"gpu_mem_gb": 24, "gpu_tflops_fp16": 121, "gpu_bandwidth_gbps": 300, "generation": "Ada Lovelace", "interconnect": "PCIe", "model": "NVIDIA L4"},
    "A10G":     {"gpu_mem_gb": 24, "gpu_tflops_fp16": 125, "gpu_bandwidth_gbps": 600, "generation": "Ampere", "interconnect": "PCIe", "model": "NVIDIA A10G"},
    "A100":     {"gpu_mem_gb": 80, "gpu_tflops_fp16": 312, "gpu_bandwidth_gbps": 2039, "generation": "Ampere", "interconnect": "NVLink", "model": "NVIDIA A100 80GB"},  # p4de
    "A100-40GB":{"gpu_mem_gb": 40, "gpu_tflops_fp16": 312, "gpu_bandwidth_gbps": 1555, "generation": "Ampere", "interconnect": "NVLink", "model": "NVIDIA A100 40GB"},  # p4d
    "H100":     {"gpu_mem_gb": 80, "gpu_tflops_fp16": 989, "gpu_bandwidth_gbps": 3350, "generation": "Hopper", "interconnect": "NVLink", "model": "NVIDIA H100 SXM"},
    "V100":     {"gpu_mem_gb": 16, "gpu_tflops_fp16": 125, "gpu_bandwidth_gbps": 900, "generation": "Volta", "interconnect": "NVLink", "model": "NVIDIA V100"},
}


def _get_model_info(model_name: str) -> dict:
    """Model metadata from HuggingFace AutoConfig.

    Mirrors tandemn-profiling/roofline/vllm/automatic_launch_1.py get_model_config_info().
    """
    info: dict = {}
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Architecture
        archs = getattr(cfg, "architectures", None)
        if archs:
            info["model_architecture"] = archs[0]

        hidden = getattr(cfg, "hidden_size", None)
        layers = getattr(cfg, "num_hidden_layers", None)
        vocab = getattr(cfg, "vocab_size", None)
        intermediate = getattr(cfg, "intermediate_size", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)

        info["vocab_size"] = vocab
        info["num_attention_heads"] = num_heads
        info["num_key_value_heads"] = num_kv_heads

        # Full config dump for canonical schema
        try:
            info["model_config_json"] = json.dumps(cfg.to_dict(), default=str)
        except Exception:
            pass

        # MoE
        num_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
        num_experts_active = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_selected_experts", None)
        info["is_moe"] = 1 if (num_experts and num_experts > 1) else 0
        info["num_experts_active"] = num_experts_active

        # params_billion — try safetensors index first (exact), then formula
        params_b = None

        # Method 1: safetensors index (works for local/S3 model paths)
        try:
            import os as _os
            if _os.path.isdir(model_name):
                idx_path = _os.path.join(model_name, "model.safetensors.index.json")
                if _os.path.exists(idx_path):
                    with open(idx_path) as f:
                        idx = json.load(f)
                    total_bytes = idx.get("metadata", {}).get("total_size")
                    if total_bytes:
                        params_b = round(int(total_bytes) / 2 / 1e9, 2)
        except Exception:
            pass

        # Method 2: formula with proper GQA + MoE handling
        if params_b is None and all(v is not None for v in [hidden, layers, vocab, intermediate]):
            kv_dim = (hidden // num_heads * num_kv_heads) if (num_heads and num_kv_heads) else hidden
            embed_params = vocab * hidden * 2  # input + output embeddings
            attn_params = layers * (hidden * hidden + 2 * hidden * kv_dim + hidden * hidden)
            ffn_params = layers * 3 * hidden * intermediate
            if num_experts and num_experts > 1:
                ffn_params = ffn_params * num_experts
            params_b = round((embed_params + attn_params + ffn_params) / 1e9, 2)

        info["params_billion"] = params_b

    except Exception as e:
        print(f"[Runner] Model introspection skipped: {e}")
    return info


def _get_price_per_hour(instance_type: str) -> Optional[float]:
    """Best-effort pricing lookup. Returns None if unavailable."""
    try:
        from sky import catalog
        return catalog.get_hourly_cost(
            instance_type=instance_type, use_spot=True,
            region=None, zone=None, clouds="aws",
        )
    except Exception:
        pass
    # Fallback table for common GPU instances
    PRICES = {
        "g6e.xlarge": 0.54, "g6e.2xlarge": 0.99, "g6e.12xlarge": 4.68,
        "g6e.48xlarge": 13.35, "g5.xlarge": 0.50, "g5.2xlarge": 1.01,
        "g5.12xlarge": 4.10, "g5.48xlarge": 16.38, "p4d.24xlarge": 32.77,
        "p5.48xlarge": 98.32,
    }
    return PRICES.get(instance_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM batch inference via HTTP server mode")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file (OpenAI batch format)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("-pp", "--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--cloud", default="aws")
    parser.add_argument("--instance-type", default="unknown")
    parser.add_argument("--gpu-name", default="unknown")
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--port", type=int, default=8001, help="Compatibility stub")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    job_start_time = time.time()
    job_start_timestamp = datetime.now().isoformat()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"[Runner] Model: {args.model}")
    print(f"[Runner] TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}")
    print(f"[Runner] max_model_len={args.max_model_len or 'auto'}, gpu_util={args.gpu_memory_utilization}")

    # 1. Start vLLM HTTP server
    proc = start_vllm_server(args)
    try:
        # 2. Wait until healthy
        print("[Runner] Waiting for vLLM server to be ready...")
        model_load_sec = wait_for_server()
        print(f"[Runner] Server ready in {model_load_sec:.2f}s")

        # 3. Start sidecar (bonus push to control plane)
        orca_url = os.getenv("ORCA_SERVER_URL", "")
        orca_key = os.getenv("ORCA_API_KEY", "")
        job_id = os.getenv("JOB_ID", "unknown")
        stop_sidecar = threading.Event()
        sidecar_thread = threading.Thread(
            target=_sidecar_loop,
            args=(stop_sidecar, orca_url, orca_key, job_id),
            daemon=True, name="orca-sidecar",
        )
        sidecar_thread.start()

        # 4. Introspect model metadata (while server warms up)
        model_info = _get_model_info(args.model)
        if model_info.get("params_billion"):
            print(f"[Runner] Model: {model_info.get('model_architecture', '?')} "
                  f"{model_info['params_billion']}B "
                  f"{'MoE' if model_info.get('is_moe') else 'dense'}")

        # 5. Load requests
        all_requests = load_requests(args.input)
        print(f"[Runner] Loaded {len(all_requests)} requests from {args.input}")
        write_progress(0, len(all_requests), "running")

        # 6. Pre-benchmark Prometheus scrape (baseline for delta computation)
        #    No warmup — batch jobs maximize utilization with all user requests.
        pre_prom_text = _scrape_prom()

        # 7. Start GPU monitor + scheduler poller (0.5s sampling, matches profiling repo)
        gpu_monitor = GPUMonitor(sample_interval=0.5)
        gpu_monitor.start()
        metrics_poller = MetricsPoller(interval=0.5)
        metrics_poller.start()

        # 8. Run benchmark
        generation_start_time = time.time()
        results = asyncio.run(run_benchmark(all_requests, args.model))
        generation_time = time.time() - generation_start_time

        num_ok = sum(1 for r in results if r["status"] == "success")
        num_skip = sum(1 for r in results if r["status"] != "success")
        print(f"[Runner] Benchmark complete in {generation_time:.2f}s ({num_ok} ok, {num_skip} skipped)")
        write_progress(len(results), len(results), "completed")

        # 9. Post-benchmark Prometheus scrape
        post_prom_text = _scrape_prom()

        # 10. Stop monitors
        gpu_monitor.stop()
        metrics_poller.stop()
        stop_sidecar.set()
        sidecar_thread.join(timeout=10)

        gpu_summary = gpu_monitor.get_summary()
        scheduler_summary = metrics_poller.get_summary()

        if gpu_summary:
            print(f"[Runner] GPU: SM={gpu_summary.get('avg_sm_util_pct', '?')}% avg, "
                  f"MemBW={gpu_summary.get('avg_mem_bw_util_pct', '?')}% avg "
                  f"({gpu_summary.get('gpu_samples', 0)} samples)")
        if scheduler_summary:
            print(f"[Runner] Scheduler: running={scheduler_summary.get('running_avg', '?')} avg, "
                  f"KV={scheduler_summary.get('kv_cache_util_pct_avg', '?')}% avg "
                  f"({scheduler_summary.get('scheduler_samples', 0)} samples)")

        # 11. Write output JSONL
        write_output_jsonl(results, args.output, args.model)
        print(f"[Runner] Wrote {len(results)} results to {args.output}")

        # 12. Build and write metrics
        metrics = build_metrics(
            results, pre_prom_text, post_prom_text, args,
            job_start_time, job_start_timestamp, model_load_sec, generation_time,
            gpu_summary, scheduler_summary, model_info,
        )
        write_metrics_csv(args.output, metrics)

        # Summary
        gen_t = metrics["generation_time_sec"]
        total_tok = metrics["total_tokens"]
        print(f"\n[Runner] === Performance Summary ===")
        print(f"  Total runtime: {metrics['total_runtime_sec']:.2f}s")
        print(f"  Model load: {model_load_sec:.2f}s")
        print(f"  Generation: {gen_t:.2f}s")
        print(f"  Requests: {len(results)} ({num_ok} ok, {num_skip} skipped)")
        print(f"  Total tokens: {total_tok:,}")
        if gen_t > 0:
            print(f"  Throughput: {num_ok/gen_t:.2f} req/s, {total_tok/gen_t:.2f} tok/s")
        if metrics.get("ttft_ms_p50"):
            print(f"  TTFT p50/p95/p99: {metrics['ttft_ms_p50']:.1f} / "
                  f"{metrics['ttft_ms_p95']:.1f} / {metrics['ttft_ms_p99']:.1f} ms (client)")
        if metrics.get("tpot_client_ms_p50"):
            print(f"  TPOT p50/p95/p99: {metrics['tpot_client_ms_p50']:.1f} / "
                  f"{metrics['tpot_client_ms_p95']:.1f} / {metrics['tpot_client_ms_p99']:.1f} ms (client)")
        if metrics.get("tpot_ms_p50"):
            print(f"  TPOT p50/p95/p99: {metrics['tpot_ms_p50']:.1f} / "
                  f"{metrics['tpot_ms_p95']:.1f} / {metrics['tpot_ms_p99']:.1f} ms (server Prom)")
        if metrics.get("avg_sm_util_pct"):
            print(f"  GPU SM: {metrics['avg_sm_util_pct']:.1f}% avg, {metrics['max_sm_util_pct']:.1f}% max")
        if metrics.get("cost_for_run_usd"):
            print(f"  Cost: ${metrics['cost_for_run_usd']:.4f} ({metrics['tokens_per_dollar']:,.0f} tok/$)")
        if metrics.get("num_preemptions", 0) > 0:
            print(f"  ⚠️  {metrics['num_preemptions']} preemptions detected — KV cache was thrashing")

    finally:
        shutdown_server(proc)


if __name__ == "__main__":
    main()
