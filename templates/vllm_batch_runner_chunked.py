#!/usr/bin/env python3
"""
vLLM chunked batch runner — pulls chunks from Orca control plane.

Each replica:
  1. Starts vLLM as an OpenAI-compatible HTTP server
  2. Queries KV cache for max_concurrency
  3. Pulls chunks from the control plane Redis queue
  4. Processes each chunk with rate-limited injection
  5. Uploads per-chunk output to S3
  6. Reports chunk completion
  7. Prefetches next chunk while GPU processes current one

Reuses core infrastructure from vllm_batch_runner_server.py:
  - vLLM server lifecycle (start, wait, shutdown)
  - SSE streaming client (send_one)
  - Prometheus helpers (sum_metric, histogram_quantile, etc.)
  - GPU monitor, sidecar loop
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
from concurrent.futures import ThreadPoolExecutor
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
PROGRESS_FILE = "/tmp/vllm_progress.json"

# Control plane config
ORCA_URL = os.getenv("ORCA_SERVER_URL", "")
ORCA_KEY = os.getenv("ORCA_API_KEY", "")
JOB_ID = os.getenv("JOB_ID", "")
REPLICA_ID = os.getenv("REPLICA_ID", "")


# ---------------------------------------------------------------------------
# Progress + phase reporting (same as single-cluster runner)
# ---------------------------------------------------------------------------

def write_progress(done: int, total: int, status: str = "running"):
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"done": done, "total": total, "status": status, "timestamp": time.time()}, f)
    except Exception:
        pass


def _report_phase(phase: str):
    if not ORCA_URL or not JOB_ID:
        return
    try:
        headers = {"Content-Type": "application/json"}
        if ORCA_KEY:
            headers["Authorization"] = f"Bearer {ORCA_KEY}"
        requests.post(f"{ORCA_URL}/job/{JOB_ID}/phase",
                      json={"phase": phase}, headers=headers, timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prometheus helpers (inline, no orca_server imports on cluster)
# ---------------------------------------------------------------------------

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


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


_METRIC_ALIASES = {
    "vllm:num_gpu_blocks":  "vllm:num_gpu_blocks_total",
    "vllm:generation_tokens_total": "vllm:generation_tokens",
    "vllm:prompt_tokens_total":     "vllm:prompt_tokens",
    "vllm:request_success_total":   "vllm:request_success",
    "vllm:num_preemptions_total":   "vllm:num_preemptions",
    "vllm:gpu_cache_usage_perc":    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_queries_total": "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits_total":    "vllm:prefix_cache_hits",
}

def sum_metric_compat(text: str, name: str) -> float:
    val = sum_metric(text, name)
    if val == 0.0 and name in _METRIC_ALIASES:
        val = sum_metric(text, _METRIC_ALIASES[name])
    return val


def _parse_labels(labels_blob: str) -> dict:
    out = {}
    if not labels_blob:
        return out
    for part in labels_blob.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


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
    buckets = _parse_histogram_buckets(text, metric_name)
    val = histogram_quantile_from_buckets(buckets, quantile)
    return val * 1000.0 if val is not None else None


def _percentile(lst: list, p: int) -> Optional[float]:
    if not lst:
        return None
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    idx = (n - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_lst[lower] * (1 - weight) + sorted_lst[upper] * weight


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


# ---------------------------------------------------------------------------
# Prometheus delta helpers — warmup subtraction
# ---------------------------------------------------------------------------

def _scrape_prom() -> str:
    try:
        return requests.get(f"{BASE_URL}/metrics", timeout=10).text
    except Exception:
        return ""


def _parse_prom_counters(text: str) -> Dict[str, float]:
    counters = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if "_bucket" in name:
            continue
        counters[name] = float(m.group("value"))
    return counters


def _delta_histogram_buckets(pre_text: str, post_text: str, metric_name: str) -> list:
    pre = dict(_parse_histogram_buckets(pre_text, metric_name))
    post = dict(_parse_histogram_buckets(post_text, metric_name))
    delta = []
    for le in sorted(post.keys()):
        delta.append((le, post[le] - pre.get(le, 0.0)))
    return delta


def _delta_counter(pre_text: str, post_text: str, metric_name: str) -> float:
    return sum_metric_compat(post_text, metric_name) - sum_metric_compat(pre_text, metric_name)


# ---------------------------------------------------------------------------
# GPU hardware monitor (pynvml)
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
        self._latest_sample: dict = {}
        self._latest_lock = threading.Lock()

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
                with self._latest_lock:
                    self._latest_sample = sample
            self._stop.wait(self._interval)

    def get_timeseries(self) -> list:
        return self._timeseries

    def get_latest(self) -> dict:
        """Get the most recent GPU sample (for sidecar injection)."""
        with self._latest_lock:
            return dict(self._latest_sample)

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
# Scheduler / Prometheus metrics poller
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
                    "kv_cache_util_pct": round(sum_metric_compat(text, "vllm:gpu_cache_usage_perc") * 100, 1),
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
# GPU specs, model info, pricing — verbatim from server runner
# ---------------------------------------------------------------------------

GPU_SPECS = {
    "L40S":     {"gpu_mem_gb": 48, "gpu_tflops_fp16": 362, "gpu_bandwidth_gbps": 864, "generation": "Ada Lovelace", "interconnect": "PCIe", "model": "NVIDIA L40S"},
    "L4":       {"gpu_mem_gb": 24, "gpu_tflops_fp16": 121, "gpu_bandwidth_gbps": 300, "generation": "Ada Lovelace", "interconnect": "PCIe", "model": "NVIDIA L4"},
    "A10G":     {"gpu_mem_gb": 24, "gpu_tflops_fp16": 125, "gpu_bandwidth_gbps": 600, "generation": "Ampere", "interconnect": "PCIe", "model": "NVIDIA A10G"},
    "A100":     {"gpu_mem_gb": 80, "gpu_tflops_fp16": 312, "gpu_bandwidth_gbps": 2039, "generation": "Ampere", "interconnect": "NVLink", "model": "NVIDIA A100 80GB"},
    "A100-40GB":{"gpu_mem_gb": 40, "gpu_tflops_fp16": 312, "gpu_bandwidth_gbps": 1555, "generation": "Ampere", "interconnect": "NVLink", "model": "NVIDIA A100 40GB"},
    "H100":     {"gpu_mem_gb": 80, "gpu_tflops_fp16": 989, "gpu_bandwidth_gbps": 3350, "generation": "Hopper", "interconnect": "NVLink", "model": "NVIDIA H100 SXM"},
    "V100":     {"gpu_mem_gb": 16, "gpu_tflops_fp16": 125, "gpu_bandwidth_gbps": 900, "generation": "Volta", "interconnect": "NVLink", "model": "NVIDIA V100"},
}


def _get_model_info(model_name: str) -> dict:
    info: dict = {}
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

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

        try:
            info["model_config_json"] = json.dumps(cfg.to_dict(), default=str)
        except Exception:
            pass

        num_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
        num_experts_active = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_selected_experts", None)
        info["is_moe"] = 1 if (num_experts and num_experts > 1) else 0
        info["num_experts_active"] = num_experts_active

        params_b = None
        try:
            if os.path.isdir(model_name):
                idx_path = os.path.join(model_name, "model.safetensors.index.json")
                if os.path.exists(idx_path):
                    with open(idx_path) as f:
                        idx = json.load(f)
                    total_bytes = idx.get("metadata", {}).get("total_size")
                    if total_bytes:
                        params_b = round(int(total_bytes) / 2 / 1e9, 2)
        except Exception:
            pass

        if params_b is None and all(v is not None for v in [hidden, layers, vocab, intermediate]):
            kv_dim = (hidden // num_heads * num_kv_heads) if (num_heads and num_kv_heads) else hidden
            embed_params = vocab * hidden * 2
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
    try:
        from sky import catalog
        return catalog.get_hourly_cost(
            instance_type=instance_type, use_spot=True,
            region=None, zone=None, clouds="aws",
        )
    except Exception:
        pass
    PRICES = {
        "g6e.xlarge": 0.54, "g6e.2xlarge": 0.99, "g6e.12xlarge": 4.68,
        "g6e.48xlarge": 13.35, "g5.xlarge": 0.50, "g5.2xlarge": 1.01,
        "g5.12xlarge": 4.10, "g5.48xlarge": 16.38, "p4d.24xlarge": 32.77,
        "p5.48xlarge": 98.32,
    }
    return PRICES.get(instance_type)


# ---------------------------------------------------------------------------
# CSV + metrics builder
# ---------------------------------------------------------------------------

def write_metrics_csv(output_path: str, metrics: Dict[str, Any]):
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for metric, value in metrics.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                writer.writerow([metric, value])
        print(f"[Runner] Wrote metrics to {output_path}")
    except Exception as e:
        print(f"[Runner] Warning: Failed to write metrics: {e}")


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

    successes = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") != "success"]

    # Token stats from client-side usage
    prompt_counts = [r.get("input_tokens", 0) for r in successes]
    output_counts = [r.get("output_tokens", 0) for r in successes]
    total_prompt_client = sum(prompt_counts)
    total_output_client = sum(output_counts)

    prompt_pct = calculate_percentiles(prompt_counts)
    output_pct = calculate_percentiles(output_counts)

    # Server-side throughput from Prometheus counter deltas
    server_prompt_toks = _delta_counter(pre_prom_text, post_prom_text, "vllm:prompt_tokens_total")
    server_gen_toks = _delta_counter(pre_prom_text, post_prom_text, "vllm:generation_tokens_total")
    total_prompt = server_prompt_toks if server_prompt_toks > 0 else total_prompt_client
    total_output = server_gen_toks if server_gen_toks > 0 else total_output_client
    total_tokens = total_prompt + total_output

    # Throughput
    req_per_sec = len(successes) / generation_time if generation_time > 0 else 0.0
    tok_per_sec = total_tokens / generation_time if generation_time > 0 else 0.0
    out_tok_per_sec = total_output / generation_time if generation_time > 0 else 0.0
    in_tok_per_sec = total_prompt / generation_time if generation_time > 0 else 0.0

    # Client-side latency
    ttfts_ms = [r["ttft_ms"] for r in successes if r.get("ttft_ms") is not None]
    e2es_ms = [r["e2e_ms"] for r in successes if r.get("e2e_ms") is not None]
    tpots_client_ms = [
        (r["e2e_ms"] - r["ttft_ms"]) / (r["output_tokens"] - 1)
        for r in successes
        if r.get("output_tokens", 0) > 1 and r.get("ttft_ms") is not None and r.get("e2e_ms") is not None
    ]

    # Server-side latency from Prometheus histogram deltas
    delta_ttft = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:time_to_first_token_seconds")
    delta_tpot = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:time_per_output_token_seconds")
    delta_e2e = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:e2e_request_latency_seconds")
    delta_queue = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:request_queue_time_seconds")
    delta_prefill = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:request_prefill_time_seconds")
    delta_decode = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:request_decode_time_seconds")
    delta_inference = _delta_histogram_buckets(pre_prom_text, post_prom_text, "vllm:request_inference_time_seconds")

    def _hq(buckets, q):
        v = histogram_quantile_from_buckets(buckets, q)
        return v * 1000.0 if v is not None else None

    num_preemptions = int(_delta_counter(pre_prom_text, post_prom_text, "vllm:num_preemptions_total"))

    # Prefix cache hit rate
    pre_cq = sum_metric_compat(pre_prom_text, "vllm:prefix_cache_queries_total")
    post_cq = sum_metric_compat(post_prom_text, "vllm:prefix_cache_queries_total")
    pre_ch = sum_metric_compat(pre_prom_text, "vllm:prefix_cache_hits_total")
    post_ch = sum_metric_compat(post_prom_text, "vllm:prefix_cache_hits_total")
    delta_cq = post_cq - pre_cq
    delta_ch = post_ch - pre_ch
    prefix_cache_hit_rate = round(delta_ch / delta_cq, 4) if delta_cq > 0 else None

    # Cost
    price_per_hour = _get_price_per_hour(args.instance_type)
    elapsed_hours = generation_time / 3600
    cost_for_run = price_per_hour * elapsed_hours if price_per_hour else None
    tokens_per_dollar = round(total_tokens / cost_for_run, 2) if cost_for_run and cost_for_run > 0 else None

    metrics = {
        "job_start_timestamp": job_start_timestamp,
        "job_end_timestamp": datetime.now().isoformat(),
        "total_runtime_sec": total_runtime,
        "model_load_time_sec": model_load_sec,
        "generation_time_sec": generation_time,
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
        "throughput_requests_per_sec": req_per_sec,
        "throughput_tokens_per_sec": tok_per_sec,
        "throughput_output_tokens_per_sec": out_tok_per_sec,
        "throughput_input_tokens_per_sec": in_tok_per_sec,
        "ttft_ms_p50": _percentile(ttfts_ms, 50),
        "ttft_ms_p95": _percentile(ttfts_ms, 95),
        "ttft_ms_p99": _percentile(ttfts_ms, 99),
        "e2e_ms_p50": _percentile(e2es_ms, 50),
        "e2e_ms_p95": _percentile(e2es_ms, 95),
        "e2e_ms_p99": _percentile(e2es_ms, 99),
        "tpot_client_ms_p50": _percentile(tpots_client_ms, 50),
        "tpot_client_ms_p95": _percentile(tpots_client_ms, 95),
        "tpot_client_ms_p99": _percentile(tpots_client_ms, 99),
        "tpot_ms_p50": _hq(delta_tpot, 0.50),
        "tpot_ms_p95": _hq(delta_tpot, 0.95),
        "tpot_ms_p99": _hq(delta_tpot, 0.99),
        "ttft_server_ms_p50": _hq(delta_ttft, 0.50),
        "ttft_server_ms_p95": _hq(delta_ttft, 0.95),
        "ttft_server_ms_p99": _hq(delta_ttft, 0.99),
        "e2e_server_ms_p50": _hq(delta_e2e, 0.50),
        "e2e_server_ms_p95": _hq(delta_e2e, 0.95),
        "e2e_server_ms_p99": _hq(delta_e2e, 0.99),
        "queue_time_ms_p50": _hq(delta_queue, 0.50),
        "queue_time_ms_p95": _hq(delta_queue, 0.95),
        "queue_time_ms_p99": _hq(delta_queue, 0.99),
        "prefill_time_ms_p50": _hq(delta_prefill, 0.50),
        "prefill_time_ms_p95": _hq(delta_prefill, 0.95),
        "prefill_time_ms_p99": _hq(delta_prefill, 0.99),
        "decode_time_ms_p50": _hq(delta_decode, 0.50),
        "decode_time_ms_p95": _hq(delta_decode, 0.95),
        "decode_time_ms_p99": _hq(delta_decode, 0.99),
        "inference_time_ms_p50": _hq(delta_inference, 0.50),
        "inference_time_ms_p95": _hq(delta_inference, 0.95),
        "inference_time_ms_p99": _hq(delta_inference, 0.99),
        "prefix_cache_hit_rate": prefix_cache_hit_rate,
        "num_preemptions": num_preemptions,
        "avg_sm_util_pct": gpu_summary.get("avg_sm_util_pct"),
        "max_sm_util_pct": gpu_summary.get("max_sm_util_pct"),
        "avg_mem_bw_util_pct": gpu_summary.get("avg_mem_bw_util_pct"),
        "max_mem_bw_util_pct": gpu_summary.get("max_mem_bw_util_pct"),
        "avg_mem_util_pct": gpu_summary.get("avg_mem_util_pct"),
        "max_mem_util_pct": gpu_summary.get("max_mem_util_pct"),
        "gpu_samples": gpu_summary.get("gpu_samples", 0),
        "running_avg": scheduler_summary.get("running_avg"),
        "running_max": scheduler_summary.get("running_max"),
        "waiting_avg": scheduler_summary.get("waiting_avg"),
        "waiting_max": scheduler_summary.get("waiting_max"),
        "swapped_avg": scheduler_summary.get("swapped_avg"),
        "swapped_max": scheduler_summary.get("swapped_max"),
        "kv_cache_util_pct_avg": scheduler_summary.get("kv_cache_util_pct_avg"),
        "kv_cache_util_pct_max": scheduler_summary.get("kv_cache_util_pct_max"),
        "scheduler_samples": scheduler_summary.get("scheduler_samples", 0),
        "price_per_hour": price_per_hour,
        "cost_for_run_usd": cost_for_run,
        "tokens_per_dollar": tokens_per_dollar,
        "model_name": args.model,
        "quantization": args.quantization,
        "cloud_provider": args.cloud,
        "instance_type": args.instance_type,
        "gpu_name": args.gpu_name,
        "engine": args.engine,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
    }

    if model_info:
        for k in ("model_architecture", "params_billion", "is_moe",
                   "num_experts_active", "vocab_size"):
            if k in model_info:
                metrics[k] = model_info[k]

    gpu_spec = GPU_SPECS.get(args.gpu_name, {})
    if gpu_spec:
        metrics["gpu_mem_gb"] = gpu_spec["gpu_mem_gb"]
        metrics["gpu_tflops_fp16"] = gpu_spec["gpu_tflops_fp16"]
        metrics["gpu_bandwidth_gbps"] = gpu_spec["gpu_bandwidth_gbps"]
        metrics["gpu_model"] = gpu_spec.get("model", args.gpu_name)
        metrics["gpu_generation"] = gpu_spec.get("generation")
        metrics["interconnect"] = gpu_spec.get("interconnect")

    metrics["num_nodes"] = int(os.getenv("SKYPILOT_NUM_NODES", "1"))
    metrics["precision"] = "bfloat16" if args.dtype == "auto" else args.dtype
    metrics["runtime_stack"] = "vllm 0.10.0"

    mi = model_info or {}
    params_b = mi.get("params_billion")
    tp = args.tensor_parallel_size
    num_kv_heads = mi.get("num_key_value_heads")
    num_attn_heads = mi.get("num_attention_heads")
    gpu_count = tp * args.pipeline_parallel_size

    model_size_gb = round(params_b * 2, 2) if params_b else None
    metrics["model_size_gb"] = model_size_gb
    if params_b and gpu_count:
        metrics["params_per_gpu"] = round(params_b / gpu_count, 2)
    if model_size_gb and gpu_spec:
        metrics["vram_headroom_gb"] = round(gpu_spec["gpu_mem_gb"] * gpu_count - model_size_gb, 1)
        metrics["model_fits_single_gpu"] = 1 if model_size_gb <= gpu_spec["gpu_mem_gb"] else 0

    if num_attn_heads and num_kv_heads and num_kv_heads > 0:
        metrics["attention_heads_per_kv_head"] = round(num_attn_heads / num_kv_heads, 2)
    if num_kv_heads and tp:
        metrics["kv_heads_per_tp"] = round(num_kv_heads / tp, 2)

    if params_b and params_b > 0 and gpu_spec:
        metrics["bandwidth_per_param"] = round(gpu_spec["gpu_bandwidth_gbps"] * tp / params_b, 2)
        metrics["flops_per_param"] = round(gpu_spec["gpu_tflops_fp16"] * tp / params_b, 2)

    metrics["crosses_node_boundary"] = 1 if args.pipeline_parallel_size > 1 else 0

    if price_per_hour and in_tok_per_sec > 0:
        metrics["cost_per_1m_tokens_prefill_usd"] = round(price_per_hour / (in_tok_per_sec * 3.6), 4)
    if price_per_hour and out_tok_per_sec > 0:
        metrics["cost_per_1m_tokens_decode_usd"] = round(price_per_hour / (out_tok_per_sec * 3.6), 4)

    metrics["kv_offload_target"] = "None"

    if mi.get("model_config_json"):
        metrics["model_config_json"] = mi["model_config_json"]

    return metrics


# ---------------------------------------------------------------------------
# Sidecar (same as single-cluster runner)
# ---------------------------------------------------------------------------

def _sidecar_loop(stop_event: threading.Event, gpu_monitor: Optional["GPUMonitor"] = None):
    if not ORCA_URL:
        return

    ingest_url = f"{ORCA_URL}/job/{JOB_ID}/metrics/ingest"
    headers = {"Content-Type": "application/json"}
    if ORCA_KEY:
        headers["Authorization"] = f"Bearer {ORCA_KEY}"

    buffer = []
    last_push = time.time()

    while not stop_event.is_set():
        try:
            prom_text = requests.get(f"{BASE_URL}/metrics", timeout=4).text
            snap = {"timestamp": time.time(), "prometheus_text": prom_text}
            # Inject latest GPU hardware utilization from pynvml
            if gpu_monitor:
                latest = gpu_monitor.get_latest()
                if latest:
                    sm_vals = [v for k, v in latest.items() if "_sm_pct" in k]
                    bw_vals = [v for k, v in latest.items() if "_membw_pct" in k]
                    if sm_vals:
                        snap["gpu_sm_util_pct"] = round(sum(sm_vals) / len(sm_vals), 1)
                    if bw_vals:
                        snap["gpu_mem_bw_util_pct"] = round(sum(bw_vals) / len(bw_vals), 1)
            buffer.append(snap)
        except Exception:
            pass

        if time.time() - last_push >= SIDECAR_INTERVAL_SEC and buffer:
            try:
                payload: dict = {"snapshots": buffer, "replica_id": REPLICA_ID}
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
            requests.post(ingest_url, json={"snapshots": buffer, "replica_id": REPLICA_ID}, headers=headers, timeout=5)
        except Exception:
            pass


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
    env = os.environ.copy()
    env.setdefault("VLLM_LOG_STATS_INTERVAL", "1")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)


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
# max_concurrency from KV cache
# ---------------------------------------------------------------------------

def determine_max_concurrency() -> int:
    """Get max_concurrency from Prometheus num_gpu_blocks, with env var fallback."""
    prom_value = _estimate_max_concurrency_from_metrics()

    result = prom_value or int(os.getenv("MAX_NUM_SEQS", "256"))
    source = "prometheus" if prom_value else "fallback"
    print(f"[Runner] max_concurrency={result} (source={source})")
    return min(result, 512)


def _get_info_label(text: str, metric_name: str, label_key: str) -> Optional[str]:
    """Extract a label value from a Prometheus info/gauge metric."""
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m or m.group("name") != metric_name:
            continue
        for part in (m.group("labels") or "").split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            if k.strip() == label_key:
                return v.strip().strip('"')
    return None


def _estimate_max_concurrency_from_metrics() -> Optional[int]:
    """Estimate from Prometheus num_gpu_blocks metric.

    vLLM V1 moved num_gpu_blocks from a gauge to a label on cache_config_info.
    Try the V1 info metric first, fall back to V0 gauge.
    """
    try:
        text = requests.get(f"{BASE_URL}/metrics", timeout=10).text

        # V1: label on cache_config_info
        blocks_str = _get_info_label(text, "vllm:cache_config_info", "num_gpu_blocks")
        if blocks_str and blocks_str != "None":
            num_gpu_blocks = int(blocks_str)
            bs = _get_info_label(text, "vllm:cache_config_info", "block_size")
            block_size = int(bs) if bs and bs != "None" else 16
        else:
            # V0: gauge metric
            num_gpu_blocks = int(sum_metric_compat(text, "vllm:num_gpu_blocks"))
            block_size = 16

        if num_gpu_blocks == 0:
            return None
        avg_seq_len = (
            int(os.getenv("AVG_INPUT_TOKENS", "2000"))
            + int(os.getenv("AVG_OUTPUT_TOKENS", "500"))
        )
        blocks_per_request = max(1, avg_seq_len // block_size)
        result = max(1, int(num_gpu_blocks // blocks_per_request))
        print(f"[Runner] KV cache: {num_gpu_blocks} blocks × {block_size} tokens, "
              f"avg_seq={avg_seq_len} → {blocks_per_request} blocks/req → max_concurrency={result}")
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chunk pull / complete / upload
# ---------------------------------------------------------------------------

_SENTINEL_QUEUE_EMPTY = "QUEUE_EMPTY"
_SENTINEL_TRANSIENT_ERROR = "TRANSIENT_ERROR"


def _auth_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if ORCA_KEY:
        headers["Authorization"] = f"Bearer {ORCA_KEY}"
    return headers


def _retry(fn, description: str, max_attempts: int = 5, base_delay: float = 2.0):
    """Retry with exponential backoff. Returns fn() result or raises on exhaustion."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"[Runner] {description} failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                print(f"[Runner] Retrying in {delay:.0f}s...")
                time.sleep(delay)
    raise RuntimeError(f"{description} failed after {max_attempts} attempts")


def pull_chunk_from_server():
    """Pull next chunk. Returns chunk dict, _SENTINEL_QUEUE_EMPTY, or _SENTINEL_TRANSIENT_ERROR."""
    def _pull():
        resp = requests.post(
            f"{ORCA_URL}/job/{JOB_ID}/chunks/pull",
            json={"replica_id": REPLICA_ID},
            headers=_auth_headers(),
            timeout=30,
        )
        if resp.status_code == 204:
            return _SENTINEL_QUEUE_EMPTY
        resp.raise_for_status()
        return resp.json()

    try:
        return _retry(_pull, "Chunk pull", max_attempts=5)
    except RuntimeError:
        return _SENTINEL_TRANSIENT_ERROR


def check_chunk_progress() -> dict:
    """Check chunk-level progress from control plane."""
    try:
        resp = requests.get(
            f"{ORCA_URL}/job/{JOB_ID}/chunks/progress",
            headers=_auth_headers(), timeout=10,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {}


def report_chunk_complete(chunk_id: str) -> dict:
    """Report chunk completion to control plane (with retries)."""
    def _complete():
        resp = requests.post(
            f"{ORCA_URL}/job/{JOB_ID}/chunks/complete",
            json={"chunk_id": chunk_id, "replica_id": REPLICA_ID},
            headers=_auth_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        return _retry(_complete, f"Complete chunk {chunk_id}", max_attempts=5)
    except RuntimeError:
        return {}


def download_chunk(s3_input_path: str, local_path: str) -> bool:
    """Download a chunk via the control plane's S3 download endpoint (with retries)."""
    def _download():
        resp = requests.get(
            f"{ORCA_URL}/storage/download_s3",
            params={"path": s3_input_path, "user": "chunk-runner"},
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for data in resp.iter_content(chunk_size=8192):
                f.write(data)

    try:
        _retry(_download, f"Download {s3_input_path}", max_attempts=3, base_delay=3.0)
        return True
    except RuntimeError:
        return False


def upload_chunk_output(local_path: str, s3_output_path: str) -> bool:
    """Upload chunk output via the control plane's storage endpoint (with retries)."""
    def _upload():
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{ORCA_URL}/storage/upload",
                files={"file": (os.path.basename(local_path), f)},
                data={"user": "chunk-runner", "remote_path": s3_output_path},
                timeout=120,
            )
        resp.raise_for_status()

    try:
        _retry(_upload, f"Upload {s3_output_path}", max_attempts=3, base_delay=3.0)
        return True
    except RuntimeError:
        return False


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


def _renew_lease_loop(chunk_id: str, stop_event: threading.Event, stolen_flag: list):
    """Background thread: renews chunk lease every CHUNK_RENEW_INTERVAL_SEC seconds.

    Renews immediately on start (covers the prefetch gap — the chunk may have
    been sitting in inflight since the prefetch, and its lease could be close
    to expiry by the time the main loop starts processing it).
    """
    interval = int(os.getenv("CHUNK_RENEW_INTERVAL_SEC", "30"))
    first = True
    while not stop_event.is_set():
        if not first:
            stop_event.wait(timeout=interval)
            if stop_event.is_set():
                break
        first = False
        try:
            resp = requests.post(
                f"{ORCA_URL}/job/{JOB_ID}/chunks/renew",
                json={"chunk_id": chunk_id, "replica_id": REPLICA_ID},
                headers=_auth_headers(),
                timeout=10,
            )
            if resp.ok and not resp.json().get("renewed", True):
                print(f"[Runner] Lease for chunk {chunk_id} was reclaimed — stopping renewal")
                stolen_flag[0] = True
                stop_event.set()
        except Exception:
            pass  # transient — renewal failure is non-fatal


def pull_and_download_chunk() -> tuple:
    """Pull a chunk, start its lease renewal, and download it.

    Renewal begins immediately after pull — this keeps the lease alive while
    the chunk sits in a prefetch Future waiting for the main loop to consume it.

    Returns:
        (chunk_info, requests, renewal_ctx) — normal chunk
        (_SENTINEL_QUEUE_EMPTY, None, None) — queue is definitively empty
        (_SENTINEL_TRANSIENT_ERROR, None, None) — couldn't reach server
        (chunk_info, [], renewal_ctx) — pulled but download failed

    renewal_ctx is (stop_event, stolen_flag, thread) — caller takes ownership.
    """
    result = pull_chunk_from_server()
    if result == _SENTINEL_QUEUE_EMPTY:
        return _SENTINEL_QUEUE_EMPTY, None, None
    if result == _SENTINEL_TRANSIENT_ERROR:
        return _SENTINEL_TRANSIENT_ERROR, None, None

    chunk_info = result
    cid = chunk_info.get("chunk_id", "unknown")

    # Start renewal immediately — keeps lease alive during prefetch wait
    renewal_stop = threading.Event()
    lease_stolen = [False]
    renewal_thread = threading.Thread(
        target=_renew_lease_loop,
        args=(cid, renewal_stop, lease_stolen),
        daemon=True,
        name=f"lease-renew-{cid}",
    )
    renewal_thread.start()
    renewal_ctx = (renewal_stop, lease_stolen, renewal_thread)

    s3_path = chunk_info.get("s3_input_path", "")
    local_path = f"/tmp/chunk_{JOB_ID}_{cid}.jsonl"

    print(f"[Runner] Downloading chunk {cid} from {s3_path}")
    if not download_chunk(s3_path, local_path):
        print(f"[Runner] Failed to download chunk {cid} after retries")
        return chunk_info, [], renewal_ctx

    reqs = load_requests(local_path)
    os.unlink(local_path)
    print(f"[Runner] Chunk {cid}: {len(reqs)} requests")
    return chunk_info, reqs, renewal_ctx


# ---------------------------------------------------------------------------
# SSE streaming client (same as single-cluster runner)
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
                        "error": text[:200],
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "ttft_ms": 0,
                        "e2e_ms": (time.time() - t0) * 1000,
                    }
                if resp.status != 200:
                    return {
                        "status": "error",
                        "error": f"HTTP {resp.status}",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "ttft_ms": 0,
                        "e2e_ms": (time.time() - t0) * 1000,
                    }
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft is None:
                                ttft = (time.time() - t0) * 1000
                            content_parts.append(content)
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)[:200],
                "input_tokens": 0,
                "output_tokens": 0,
                "ttft_ms": 0,
                "e2e_ms": (time.time() - t0) * 1000,
            }

    e2e_ms = (time.time() - t0) * 1000
    if output_tokens == 0:
        output_tokens = len("".join(content_parts)) // 4

    return {
        "status": "success",
        "finish_reason": finish_reason,
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft or e2e_ms,
        "e2e_ms": e2e_ms,
        "text": "".join(content_parts),
    }


async def run_chunk(requests_list: List[Dict], model_name: str, max_concurrency: int) -> List[Dict]:
    """Process a single chunk of requests."""
    semaphore = asyncio.Semaphore(max_concurrency)
    connector = aiohttp.TCPConnector(limit=max_concurrency)
    total = len(requests_list)
    results = [None] * total

    async def _tracked(idx, req):
        r = await send_one(session, semaphore, req, model_name)
        results[idx] = r
        return r

    async with aiohttp.ClientSession(connector=connector) as session:
        await asyncio.gather(*[_tracked(i, req) for i, req in enumerate(requests_list)])

    return results


def write_output_jsonl(results: List[Dict], output_path: str, model_name: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, result in enumerate(results):
            entry = {
                "id": f"batch_req_{i}",
                "custom_id": f"request-{i}",
                "response": {
                    "status_code": 200 if result and result.get("status") == "success" else 400,
                    "body": {
                        "model": model_name,
                        "choices": [{
                            "message": {"role": "assistant", "content": result.get("text", "") if result else ""},
                            "finish_reason": result.get("finish_reason", "stop") if result else "error",
                        }],
                        "usage": {
                            "prompt_tokens": result.get("input_tokens", 0) if result else 0,
                            "completion_tokens": result.get("output_tokens", 0) if result else 0,
                        },
                    },
                },
            }
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM chunked batch inference")
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
    parser.add_argument("--chunked", action="store_true", help="Enable chunked mode")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main — chunked loop with prefetch
# ---------------------------------------------------------------------------

def _post_summary(metrics_dict: dict):
    """POST per-replica metrics summary to the control plane."""
    if not ORCA_URL or not JOB_ID:
        return
    try:
        resp = requests.post(
            f"{ORCA_URL}/job/{JOB_ID}/metrics/summary",
            json={"replica_id": REPLICA_ID, "metrics": metrics_dict},
            headers=_auth_headers(),
            timeout=30,
        )
        if resp.ok:
            print("[Runner] Posted metrics summary to control plane")
        else:
            print(f"[Runner] Warning: summary POST returned {resp.status_code}")
    except Exception as e:
        print(f"[Runner] Warning: Failed to POST metrics summary: {e}")


def main():
    args = parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"[Runner] Chunked mode — Replica: {REPLICA_ID}")
    print(f"[Runner] Model: {args.model}")
    print(f"[Runner] TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}")
    print(f"[Runner] Control plane: {ORCA_URL}, Job: {JOB_ID}")

    job_start_time = time.time()
    job_start_timestamp = datetime.now().isoformat()

    # 1. Start vLLM server
    _report_phase("loading_model")
    proc = start_vllm_server(args)
    try:
        print("[Runner] Waiting for vLLM server to be ready...")
        model_load_sec = wait_for_server()
        print(f"[Runner] Server ready in {model_load_sec:.2f}s")
        _report_phase("model_ready")

        # 2. Determine max_concurrency from KV cache
        max_concurrency = determine_max_concurrency()

        # 3. Model introspection
        print("[Runner] Introspecting model metadata...")
        model_info = _get_model_info(args.model)
        if model_info.get("params_billion"):
            print(f"[Runner] Model: {model_info.get('model_architecture', '?')} "
                  f"({model_info['params_billion']}B params)")

        # 4. Start GPU monitor + metrics poller
        gpu_monitor = GPUMonitor()
        metrics_poller = MetricsPoller()

        # 5. Start sidecar (with GPU monitor reference for live util)
        stop_sidecar = threading.Event()
        sidecar_thread = threading.Thread(
            target=_sidecar_loop, args=(stop_sidecar, gpu_monitor),
            daemon=True, name="orca-sidecar",
        )
        sidecar_thread.start()

        # 6. Pre-scrape Prometheus + start monitors
        pre_prom_text = _scrape_prom()
        gpu_monitor.start()
        metrics_poller.start()
        generation_start = time.time()

        # 7. Chunk processing loop with prefetch
        _report_phase("generating")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-prefetch")

        chunks_processed = 0
        total_requests = 0
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10
        all_results: List[Dict] = []
        s3_output_base = ""

        # Prefetch first chunk (renewal starts immediately inside)
        prefetch_future = executor.submit(pull_and_download_chunk)

        while True:
            chunk_info, chunk_requests, renewal_ctx = prefetch_future.result()

            # Queue empty — but maybe other replicas' chunks will be reclaimed
            if chunk_info == _SENTINEL_QUEUE_EMPTY:
                progress = check_chunk_progress()
                if progress.get("all_done"):
                    print("[Runner] All chunks done — exiting")
                    break
                inflight = progress.get("inflight", 0)
                if inflight > 0:
                    print(f"[Runner] Queue empty but {inflight} chunk(s) still inflight — waiting for possible reclaim...")
                    time.sleep(15)
                    prefetch_future = executor.submit(pull_and_download_chunk)
                    continue
                print("[Runner] Queue empty — all chunks consumed")
                break

            # Transient error — server unreachable, back off and retry
            if chunk_info == _SENTINEL_TRANSIENT_ERROR:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"[Runner] {MAX_CONSECUTIVE_ERRORS} consecutive errors — giving up")
                    break
                delay = min(30, 2 ** consecutive_errors)
                print(f"[Runner] Server unreachable ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}), retrying in {delay}s...")
                time.sleep(delay)
                prefetch_future = executor.submit(pull_and_download_chunk)
                continue

            consecutive_errors = 0  # reset on success
            cid = chunk_info.get("chunk_id", "unknown")
            s3_output_path = chunk_info.get("s3_output_path", "")

            # Extract s3_output_base from first chunk's output path
            if not s3_output_base and s3_output_path:
                # e.g. "s3://bucket/prefix/chunks/c0001.jsonl" → "s3://bucket/prefix"
                parts = s3_output_path.rsplit("/chunks/", 1)
                if len(parts) == 2:
                    s3_output_base = parts[0]

            # Take ownership of renewal context from prefetch
            renewal_stop, lease_stolen, renewal_thread = renewal_ctx

            # If lease was already stolen during prefetch wait, skip this chunk
            if lease_stolen[0]:
                print(f"[Runner] Chunk {cid} lease was reclaimed during prefetch — skipping")
                prefetch_future = executor.submit(pull_and_download_chunk)
                continue

            # Prefetch next chunk while GPU works (renewal starts inside)
            prefetch_future = executor.submit(pull_and_download_chunk)

            if not chunk_requests:
                print(f"[Runner] Chunk {cid}: download failed, reporting complete (empty)")
                renewal_stop.set()
                renewal_thread.join(timeout=5)
                if not lease_stolen[0]:
                    report_chunk_complete(cid)
                continue

            # Process chunk
            print(f"[Runner] Processing chunk {cid}: {len(chunk_requests)} requests (concurrency={max_concurrency})")
            results = asyncio.run(run_chunk(chunk_requests, args.model, max_concurrency))

            num_ok = sum(1 for r in results if r and r.get("status") == "success")
            print(f"[Runner] Chunk {cid} done: {num_ok}/{len(results)} ok")

            # Accumulate results for final metrics
            all_results.extend(r for r in results if r)

            # Write and upload output
            local_output = f"/tmp/chunk_output_{JOB_ID}_{cid}.jsonl"
            write_output_jsonl(results, local_output, args.model)

            upload_ok = upload_chunk_output(local_output, s3_output_path)
            if upload_ok:
                print(f"[Runner] Uploaded chunk {cid} output to {s3_output_path}")
            else:
                print(f"[Runner] FAILED to upload chunk {cid} — will NOT report complete")
            try:
                os.unlink(local_output)
            except OSError:
                pass

            # Stop renewal and check if lease was stolen while we were processing
            renewal_stop.set()
            renewal_thread.join(timeout=5)
            if lease_stolen[0]:
                print(f"[Runner] Chunk {cid} lease was reclaimed, skipping complete")
                continue

            if not upload_ok:
                print(f"[Runner] Skipping complete for {cid} due to upload failure")
                continue

            # Report completion
            progress = report_chunk_complete(cid)
            chunks_processed += 1
            total_requests += len(results)

            completed = progress.get("completed", chunks_processed)
            total_chunks = progress.get("total", "?")
            print(f"[Runner] Progress: {completed}/{total_chunks} chunks done")
            write_progress(completed, total_chunks if isinstance(total_chunks, int) else 0)

        executor.shutdown(wait=False)
        generation_time = time.time() - generation_start

        # 8. Post-scrape Prometheus + stop monitors
        post_prom_text = _scrape_prom()
        gpu_monitor.stop()
        metrics_poller.stop()

        # 9. Build metrics
        gpu_summary = gpu_monitor.get_summary()
        scheduler_summary = metrics_poller.get_summary()

        metrics_dict = build_metrics(
            results=all_results,
            pre_prom_text=pre_prom_text,
            post_prom_text=post_prom_text,
            args=args,
            job_start_time=job_start_time,
            job_start_timestamp=job_start_timestamp,
            model_load_sec=model_load_sec,
            generation_time=generation_time,
            gpu_summary=gpu_summary,
            scheduler_summary=scheduler_summary,
            model_info=model_info,
        )

        # 10. Write metrics.csv locally and upload
        local_metrics_path = f"/tmp/metrics_{JOB_ID}_{REPLICA_ID}.csv"
        write_metrics_csv(local_metrics_path, metrics_dict)

        if s3_output_base:
            metrics_s3_path = f"{s3_output_base}/replicas/{REPLICA_ID}/metrics.csv"
            if upload_chunk_output(local_metrics_path, metrics_s3_path):
                print(f"[Runner] Uploaded metrics.csv to {metrics_s3_path}")
            else:
                print("[Runner] Warning: Failed to upload metrics.csv")

        try:
            os.unlink(local_metrics_path)
        except OSError:
            pass

        # 11. POST summary to control plane
        _post_summary(metrics_dict)

        # 12. Stop sidecar
        stop_sidecar.set()
        sidecar_thread.join(timeout=10)

        # 13. Enhanced summary
        print(f"\n[Runner] === Chunked Run Summary ===")
        print(f"  Replica: {REPLICA_ID}")
        print(f"  Chunks processed: {chunks_processed}")
        print(f"  Total requests: {total_requests}")
        print(f"  Model load: {model_load_sec:.1f}s")
        print(f"  Generation: {generation_time:.1f}s")
        if metrics_dict.get("throughput_tokens_per_sec"):
            print(f"  Throughput: {metrics_dict['throughput_tokens_per_sec']:.0f} tok/s")
        if gpu_summary.get("avg_sm_util_pct") is not None:
            print(f"  GPU SM util: avg={gpu_summary['avg_sm_util_pct']:.1f}% max={gpu_summary['max_sm_util_pct']:.1f}%")
        if metrics_dict.get("ttft_ms_p50") is not None:
            print(f"  TTFT p50: {metrics_dict['ttft_ms_p50']:.1f}ms")
        if metrics_dict.get("cost_for_run_usd") is not None:
            print(f"  Cost: ${metrics_dict['cost_for_run_usd']:.4f}")

    finally:
        shutdown_server(proc)


if __name__ == "__main__":
    main()
