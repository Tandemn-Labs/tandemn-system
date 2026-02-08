"""
Batch worker process for distributed chunk-based inference.

Runs on each SkyPilot GPU worker node. Starts a local inference engine
(e.g., vLLM), leases chunks from the Redis queue, processes them, and
writes output back via the S3 FUSE mount.

Engine-agnostic: the serve command, health endpoint, and inference endpoint
are all injected via CLI args (built from templates on the server side).

Usage:
    python batch_worker.py \
        --redis-host 10.0.0.1 --redis-port 6379 \
        --job-id job-abc123 --worker-id job-abc123-w0 \
        --serve-command "vllm serve meta-llama/Llama-3-70B -tp 4 --host 0.0.0.0 --port 8000" \
        --engine-port 8000 \
        --health-endpoint /v1/models \
        --inference-endpoint /v1/chat/completions \
        --data-dir /data
"""

import argparse
import json
import logging
import os
import signal
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import redis
import requests as http_requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_worker")


# ---------------------------------------------------------------------------
# Redis key helpers (duplicated from chunk_queue/keys.py for standalone use)
# ---------------------------------------------------------------------------

def _pending_queue(job_id: str) -> str:
    return f"job:{job_id}:pending"

def _leased_set(job_id: str) -> str:
    return f"job:{job_id}:leased"

def _completed_set(job_id: str) -> str:
    return f"job:{job_id}:completed"

def _chunk_meta(job_id: str, chunk_id: str) -> str:
    return f"job:{job_id}:chunk:{chunk_id}"

def _job_meta(job_id: str) -> str:
    return f"job:{job_id}:meta"

def _job_workers(job_id: str) -> str:
    return f"job:{job_id}:workers"


# ---------------------------------------------------------------------------
# Lua scripts (duplicated from chunk_queue/lua_scripts.py for standalone use)
# ---------------------------------------------------------------------------

_LEASE_CHUNK = """
local chunk = redis.call('ZPOPMIN', KEYS[1])
if #chunk == 0 then
    return nil
end
local chunk_id = chunk[1]
local expiry = tonumber(ARGV[3]) + tonumber(ARGV[2])
redis.call('ZADD', KEYS[2], expiry, chunk_id)
local meta_key = 'job:' .. ARGV[4] .. ':chunk:' .. chunk_id
redis.call('HSET', meta_key, 'worker_id', ARGV[1], 'leased_at', ARGV[3])
redis.call('HINCRBY', meta_key, 'lease_count', 1)
return chunk_id
"""

_COMPLETE_CHUNK = """
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then
    return 0
end
redis.call('SADD', KEYS[2], ARGV[1])
redis.call('HINCRBY', KEYS[3], 'completed_count', 1)
return 1
"""

_RELEASE_CHUNK = """
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
if removed == 0 then
    return 0
end
redis.call('ZADD', KEYS[2], ARGV[2], ARGV[1])
return 1
"""

# Worker-only: renew lease expiry in the leased sorted set.
# KEYS[1] = leased_set
# ARGV[1] = chunk_id
# ARGV[2] = new_expiry
# Returns: 1 if updated, 0 if chunk not in leased set (already reaped)
_RENEW_LEASE = """
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
if score == false then
    return 0
end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
return 1
"""


# ---------------------------------------------------------------------------
# GracefulShutdown
# ---------------------------------------------------------------------------

class GracefulShutdown:
    """Thread-safe shutdown flag using threading.Event.

    Signal handlers call set() on the event, which is safe because
    Event.set() is atomic and never requires the caller to hold a lock.
    (Using a Lock here would deadlock if a signal fires while the main
    thread already holds the lock.)
    """

    def __init__(self):
        self._stop_event = threading.Event()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def trigger(self):
        self._stop_event.set()

    def wait(self, timeout: float = None) -> bool:
        """Block until shutdown is triggered or timeout expires.
        Returns True if the event was set (shutdown), False on timeout.
        """
        return self._stop_event.wait(timeout=timeout)


# ---------------------------------------------------------------------------
# EngineServer
# ---------------------------------------------------------------------------

class EngineServer:
    """Manages the local inference engine subprocess.

    Engine-agnostic: takes the full serve command as a string,
    polls a configurable health endpoint to know when it's ready.
    """

    def __init__(self, serve_command: str, port: int, health_endpoint: str):
        self.serve_command = serve_command
        self.port = port
        self.health_endpoint = health_endpoint
        self._process: Optional[subprocess.Popen] = None

    def start(self):
        """Launch the inference engine as a subprocess."""
        logger.info(f"Starting engine: {self.serve_command}")
        self._process = subprocess.Popen(
            shlex.split(self.serve_command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        threading.Thread(
            target=self._drain_output, daemon=True, name="engine-stdout"
        ).start()

    def _drain_output(self):
        """Read engine stdout and log it so the pipe buffer doesn't fill."""
        for line in self._process.stdout:
            logger.info(f"[engine] {line.decode('utf-8', errors='replace').rstrip()}")

    def wait_until_ready(self, timeout: int = 600, poll_interval: int = 5,
                         shutdown: Optional["GracefulShutdown"] = None):
        """Poll the health endpoint until the engine is serving."""
        url = f"http://localhost:{self.port}{self.health_endpoint}"
        deadline = time.time() + timeout
        logger.info(f"Waiting for engine at {url} (timeout={timeout}s)")

        while time.time() < deadline:
            if shutdown and shutdown.should_stop:
                raise RuntimeError("Shutdown requested during engine startup")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"Engine process exited with code {self._process.returncode}"
                )
            try:
                resp = http_requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info("Engine is ready")
                    return
            except (http_requests.ConnectionError, http_requests.Timeout):
                pass
            time.sleep(poll_interval)

        raise TimeoutError(f"Engine not ready after {timeout}s")

    def stop(self):
        """Gracefully stop the engine (SIGTERM, then SIGKILL if needed)."""
        if self._process is None:
            return
        logger.info("Stopping engine (SIGTERM)")
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Engine didn't stop, sending SIGKILL")
            self._process.kill()
            self._process.wait(timeout=10)


# ---------------------------------------------------------------------------
# ChunkProcessor
# ---------------------------------------------------------------------------

class ChunkProcessor:
    """Reads a JSONL chunk, sends all requests concurrently to the engine,
    and writes the output JSONL.

    vLLM (and similar engines) handle internal batching via continuous
    batching — we fire all requests at once and let the engine schedule.
    """

    def __init__(self, engine_port: int, inference_endpoint: str, max_concurrent: int = 64):
        self.engine_url = f"http://localhost:{engine_port}{inference_endpoint}"
        self.max_concurrent = max_concurrent

    def process_chunk(self, input_path: str, output_path: str) -> int:
        """Process all lines in a chunk. Returns number of lines processed."""
        lines = Path(input_path).read_text().strip().splitlines()
        if not lines:
            logger.warning(f"Empty chunk: {input_path}")
            Path(output_path).write_text("")
            return 0

        # Parse all requests
        indexed_requests = [(i, json.loads(line)) for i, line in enumerate(lines)]

        # Fire all concurrently, let the engine's continuous batching handle scheduling
        results = [None] * len(indexed_requests)
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent, len(indexed_requests))) as pool:
            futures = {
                pool.submit(self._send_request, body): idx
                for idx, body in indexed_requests
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        # Write output atomically (temp file + rename) to prevent
        # corruption if the worker dies mid-write or two workers race.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        os.replace(tmp_path, output_path)

        return len(results)

    def _send_request(self, request_body: dict) -> dict:
        """Send a single inference request to the local engine."""
        resp = http_requests.post(self.engine_url, json=request_body, timeout=300)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# DoubleBufferedWorker
# ---------------------------------------------------------------------------

class DoubleBufferedWorker:
    """Main worker loop with double buffering and lease heartbeats.

    While the GPU processes chunk A, chunk B is pre-read from the FUSE
    mount into page cache in a background thread. This hides I/O latency.
    """

    def __init__(
        self,
        r: redis.Redis,
        job_id: str,
        worker_id: str,
        data_dir: str,
        lease_ttl: int,
        processor: ChunkProcessor,
        shutdown: GracefulShutdown,
    ):
        self.r = r
        self.job_id = job_id
        self.worker_id = worker_id
        self.data_dir = data_dir
        self.lease_ttl = lease_ttl
        self.processor = processor
        self.shutdown = shutdown

        # Register Lua scripts once
        self._lease_script = self.r.register_script(_LEASE_CHUNK)
        self._complete_script = self.r.register_script(_COMPLETE_CHUNK)
        self._release_script = self.r.register_script(_RELEASE_CHUNK)
        self._renew_script = self.r.register_script(_RENEW_LEASE)

        # Chunks currently leased by this worker (for heartbeat renewal)
        self._held_lock = threading.Lock()
        self._held_chunks: List[str] = []

        # Single-thread pool for pre-reading the next chunk from FUSE
        self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")

    def run(self):
        """Entry point: start heartbeat, process chunks, clean up."""
        self._start_heartbeat()
        try:
            self._process_loop()
        finally:
            self._prefetch_pool.shutdown(wait=False)

    def _process_loop(self):
        """Lease -> process -> complete, with double buffering."""
        inflight = self._lease_chunk()
        if inflight is None:
            logger.info("Queue empty, nothing to do")
            return

        while not self.shutdown.should_stop:
            chunk_id, chunk_meta = inflight

            # Kick off a background lease + pre-read for the next chunk
            buffer_future = self._prefetch_pool.submit(self._lease_and_prefetch)

            # Resolve paths
            input_path = self._resolve_input_path(chunk_meta)
            output_path = self._derive_output_path(input_path)

            # Process current chunk on the GPU
            logger.info(f"Processing chunk {chunk_id}")
            try:
                n_lines = self.processor.process_chunk(input_path, output_path)
                logger.info(f"Chunk {chunk_id}: {n_lines} lines done")
            except Exception:
                logger.exception(f"Failed processing chunk {chunk_id}")
                self._release_chunk(chunk_id, chunk_meta)
                inflight = buffer_future.result()
                if inflight is None:
                    break
                continue

            # Mark complete in Redis
            self._complete_chunk(chunk_id)

            # Promote buffer -> inflight
            inflight = buffer_future.result()
            if inflight is None:
                logger.info("Queue empty, all done")
                break

        # Exiting loop — any still-held chunks get released by release_all_held()

    # ── Redis operations ───────────────────────────────────────────────

    def _lease_chunk(self) -> Optional[Tuple[str, dict]]:
        """Lease one chunk from Redis. Returns (chunk_id, meta) or None."""
        chunk_id = self._lease_script(
            keys=[_pending_queue(self.job_id), _leased_set(self.job_id)],
            args=[self.worker_id, self.lease_ttl, time.time(), self.job_id],
        )
        if chunk_id is None:
            return None

        meta = self.r.hgetall(_chunk_meta(self.job_id, chunk_id))
        with self._held_lock:
            self._held_chunks.append(chunk_id)

        logger.info(f"Leased chunk {chunk_id}")
        return (chunk_id, meta)

    def _complete_chunk(self, chunk_id: str):
        """Mark a chunk as completed in Redis."""
        result = self._complete_script(
            keys=[
                _leased_set(self.job_id),
                _completed_set(self.job_id),
                _job_meta(self.job_id),
            ],
            args=[chunk_id, self.worker_id],
        )
        with self._held_lock:
            if chunk_id in self._held_chunks:
                self._held_chunks.remove(chunk_id)

        if int(result) == 1:
            logger.info(f"Completed chunk {chunk_id}")
        else:
            logger.warning(f"Complete returned 0 for {chunk_id} (already reaped?)")

    def _release_chunk(self, chunk_id: str, chunk_meta: dict):
        """Return a chunk to the pending queue (graceful shutdown / failure)."""
        chunk_index = int(chunk_meta.get("chunk_index", 0))
        self._release_script(
            keys=[_leased_set(self.job_id), _pending_queue(self.job_id)],
            args=[chunk_id, chunk_index],
        )
        with self._held_lock:
            if chunk_id in self._held_chunks:
                self._held_chunks.remove(chunk_id)

        logger.info(f"Released chunk {chunk_id} back to pending")

    # ── Double-buffer prefetch ─────────────────────────────────────────

    def _lease_and_prefetch(self) -> Optional[Tuple[str, dict]]:
        """Lease next chunk and pre-read its file into page cache."""
        result = self._lease_chunk()
        if result is None:
            return None

        chunk_id, chunk_meta = result
        input_path = self._resolve_input_path(chunk_meta)
        try:
            Path(input_path).read_bytes()
            logger.info(f"Pre-fetched chunk {chunk_id} into page cache")
        except FileNotFoundError:
            logger.warning(f"Pre-fetch: file not found {input_path}")

        return result

    # ── Path helpers ───────────────────────────────────────────────────

    def _resolve_input_path(self, chunk_meta: dict) -> str:
        """Build full input path from chunk metadata and data_dir."""
        return os.path.join(self.data_dir, chunk_meta["input_path"])

    def _derive_output_path(self, input_path: str) -> str:
        """Insert /output/ before the filename.
        /data/user/req_id/000001.jsonl -> /data/user/req_id/output/000001.jsonl
        """
        parent = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        return os.path.join(parent, "output", filename)

    # ── Heartbeat ──────────────────────────────────────────────────────

    def _start_heartbeat(self):
        """Start a daemon thread that renews leases every TTL/3."""
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Periodically renew leases. Exits when shutdown is triggered."""
        interval = self.lease_ttl / 3
        while not self.shutdown.wait(timeout=interval):
            self._renew_leases()

    def _renew_leases(self):
        """Renew lease expiry for all held chunks.

        Copies the list under the lock, then does Redis calls outside
        the lock to avoid blocking lease/complete operations.
        """
        with self._held_lock:
            chunks_to_renew = list(self._held_chunks)

        if not chunks_to_renew:
            return

        new_expiry = time.time() + self.lease_ttl
        for chunk_id in chunks_to_renew:
            try:
                result = self._renew_script(
                    keys=[_leased_set(self.job_id)],
                    args=[chunk_id, new_expiry],
                )
                if int(result) == 0:
                    logger.warning(
                        f"Lease for chunk {chunk_id} was lost (reaped by server). "
                        f"Another worker may have re-leased it."
                    )
            except redis.RedisError:
                logger.warning(f"Failed to renew lease for {chunk_id}", exc_info=True)

    # ── Graceful release ───────────────────────────────────────────────

    def release_all_held(self):
        """Release any un-completed chunks back to pending (shutdown cleanup)."""
        with self._held_lock:
            chunks = list(self._held_chunks)

        for chunk_id in chunks:
            meta = self.r.hgetall(_chunk_meta(self.job_id, chunk_id))
            # Fall back to chunk_index=0 if metadata is gone — still ZREM from
            # leased so the chunk doesn't stay stranded.
            if not meta:
                meta = {"chunk_index": "0"}
            self._release_chunk(chunk_id, meta)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch inference worker")
    parser.add_argument("--redis-host", required=True)
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-password", default=None)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--lease-ttl", type=int, default=600)
    parser.add_argument("--serve-command", required=True,
                        help="Full command to start the inference engine")
    parser.add_argument("--engine-port", type=int, default=8000)
    parser.add_argument("--health-endpoint", default="/v1/models")
    parser.add_argument("--inference-endpoint", default="/v1/chat/completions")
    parser.add_argument("--max-concurrent", type=int, default=64,
                        help="Max concurrent requests to engine per chunk")
    args = parser.parse_args()

    # Graceful shutdown via signals
    shutdown = GracefulShutdown()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.trigger())
    signal.signal(signal.SIGINT, lambda *_: shutdown.trigger())

    # Connect to Redis on the control plane
    r = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        password=args.redis_password,
        db=0,
        decode_responses=True,
        socket_timeout=10,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )
    r.ping()
    logger.info(f"Connected to Redis at {args.redis_host}:{args.redis_port}")

    # Register this worker with the job
    r.sadd(_job_workers(args.job_id), args.worker_id)

    # Start the inference engine
    engine = EngineServer(
        serve_command=args.serve_command,
        port=args.engine_port,
        health_endpoint=args.health_endpoint,
    )
    engine.start()

    try:
        engine.wait_until_ready(shutdown=shutdown)
    except (TimeoutError, RuntimeError) as e:
        logger.error(f"Engine failed to start: {e}")
        engine.stop()
        sys.exit(1)

    # Build processor and worker
    processor = ChunkProcessor(
        engine_port=args.engine_port,
        inference_endpoint=args.inference_endpoint,
        max_concurrent=args.max_concurrent,
    )
    worker = DoubleBufferedWorker(
        r=r,
        job_id=args.job_id,
        worker_id=args.worker_id,
        data_dir=args.data_dir,
        lease_ttl=args.lease_ttl,
        processor=processor,
        shutdown=shutdown,
    )

    try:
        worker.run()
    except Exception:
        logger.exception("Worker crashed")
    finally:
        worker.release_all_held()
        engine.stop()
        logger.info("Worker shutdown complete")


if __name__ == "__main__":
    main()
