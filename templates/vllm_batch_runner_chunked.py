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
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
}

def sum_metric_compat(text: str, name: str) -> float:
    val = sum_metric(text, name)
    if val == 0.0 and name in _METRIC_ALIASES:
        val = sum_metric(text, _METRIC_ALIASES[name])
    return val


# ---------------------------------------------------------------------------
# Sidecar (same as single-cluster runner)
# ---------------------------------------------------------------------------

def _sidecar_loop(stop_event: threading.Event):
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


def _estimate_max_concurrency_from_metrics() -> Optional[int]:
    """Estimate from Prometheus num_gpu_blocks metric."""
    try:
        text = requests.get(f"{BASE_URL}/metrics", timeout=10).text
        num_gpu_blocks = sum_metric_compat(text, "vllm:num_gpu_blocks")
        if num_gpu_blocks == 0:
            return None
        block_size = 16  # vLLM V1 default
        avg_seq_len = (
            int(os.getenv("AVG_INPUT_TOKENS", "2000"))
            + int(os.getenv("AVG_OUTPUT_TOKENS", "500"))
        )
        blocks_per_request = max(1, avg_seq_len // block_size)
        return max(1, int(num_gpu_blocks // blocks_per_request))
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


def pull_and_download_chunk() -> tuple:
    """Pull a chunk and download it.

    Returns:
        (chunk_info, requests) — normal chunk
        (_SENTINEL_QUEUE_EMPTY, None) — queue is definitively empty
        (_SENTINEL_TRANSIENT_ERROR, None) — couldn't reach server, should retry later
        (chunk_info, []) — pulled but download failed
    """
    result = pull_chunk_from_server()
    if result == _SENTINEL_QUEUE_EMPTY:
        return _SENTINEL_QUEUE_EMPTY, None
    if result == _SENTINEL_TRANSIENT_ERROR:
        return _SENTINEL_TRANSIENT_ERROR, None

    chunk_info = result
    cid = chunk_info.get("chunk_id", "unknown")
    s3_path = chunk_info.get("s3_input_path", "")
    local_path = f"/tmp/chunk_{JOB_ID}_{cid}.jsonl"

    print(f"[Runner] Downloading chunk {cid} from {s3_path}")
    if not download_chunk(s3_path, local_path):
        print(f"[Runner] Failed to download chunk {cid} after retries")
        return chunk_info, []

    reqs = load_requests(local_path)
    os.unlink(local_path)
    print(f"[Runner] Chunk {cid}: {len(reqs)} requests")
    return chunk_info, reqs


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

def main():
    args = parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"[Runner] Chunked mode — Replica: {REPLICA_ID}")
    print(f"[Runner] Model: {args.model}")
    print(f"[Runner] TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}")
    print(f"[Runner] Control plane: {ORCA_URL}, Job: {JOB_ID}")

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

        # 3. Start sidecar
        stop_sidecar = threading.Event()
        sidecar_thread = threading.Thread(
            target=_sidecar_loop, args=(stop_sidecar,),
            daemon=True, name="orca-sidecar",
        )
        sidecar_thread.start()

        # 4. Chunk processing loop with prefetch
        _report_phase("generating")
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chunk-prefetch")

        chunks_processed = 0
        total_requests = 0
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10

        # Prefetch first chunk
        prefetch_future = executor.submit(pull_and_download_chunk)

        while True:
            chunk_info, chunk_requests = prefetch_future.result()

            # Definitively empty — all chunks consumed
            if chunk_info == _SENTINEL_QUEUE_EMPTY:
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

            # Immediately prefetch next chunk while GPU works
            prefetch_future = executor.submit(pull_and_download_chunk)

            if not chunk_requests:
                print(f"[Runner] Chunk {cid}: download failed, reporting complete (empty)")
                report_chunk_complete(cid)
                continue

            # Process chunk
            print(f"[Runner] Processing chunk {cid}: {len(chunk_requests)} requests (concurrency={max_concurrency})")
            results = asyncio.run(run_chunk(chunk_requests, args.model, max_concurrency))

            num_ok = sum(1 for r in results if r and r.get("status") == "success")
            print(f"[Runner] Chunk {cid} done: {num_ok}/{len(results)} ok")

            # Write and upload output
            local_output = f"/tmp/chunk_output_{JOB_ID}_{cid}.jsonl"
            write_output_jsonl(results, local_output, args.model)

            if upload_chunk_output(local_output, s3_output_path):
                print(f"[Runner] Uploaded chunk {cid} output to {s3_output_path}")
            else:
                print(f"[Runner] Failed to upload chunk {cid} output")
            os.unlink(local_output)

            # Report completion
            progress = report_chunk_complete(cid)
            chunks_processed += 1
            total_requests += len(results)

            completed = progress.get("completed", chunks_processed)
            total_chunks = progress.get("total", "?")
            print(f"[Runner] Progress: {completed}/{total_chunks} chunks done")
            write_progress(completed, total_chunks if isinstance(total_chunks, int) else 0)

        executor.shutdown(wait=False)

        # 5. Done
        stop_sidecar.set()
        sidecar_thread.join(timeout=10)

        print(f"\n[Runner] === Chunked Run Summary ===")
        print(f"  Replica: {REPLICA_ID}")
        print(f"  Chunks processed: {chunks_processed}")
        print(f"  Total requests: {total_requests}")

    finally:
        shutdown_server(proc)


if __name__ == "__main__":
    main()
