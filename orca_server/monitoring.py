"""Live vLLM metrics collection: per-job daemon threads, ring buffer, SSE, Prometheus."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Generator

import requests as _requests

from orca_server.job_manager import METRIC_LINE_RE, parse_labels, sum_metric, sum_metric_compat

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC: float = 1.0
RING_BUFFER_SIZE: int = 120
FLUSH_INTERVAL_SEC: int = 60


def _parse_histogram_buckets(text: str, metric_name: str) -> list[tuple[float, float]]:
    """Return sorted [(le, cumulative_count), ...] for a histogram metric."""
    bucket_name = f"{metric_name}_bucket"
    buckets: dict[float, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = METRIC_LINE_RE.match(line)
        if not m or m.group("name") != bucket_name:
            continue
        labels = parse_labels(m.group("labels") or "")
        le_str = labels.get("le")
        if le_str is None:
            continue
        le = float("inf") if le_str == "+Inf" else float(le_str)
        buckets[le] = float(m.group("value"))
    return sorted(buckets.items())


def histogram_quantile(text: str, metric_name: str, quantile: float) -> float | None:
    """Compute a quantile from a Prometheus histogram. Returns ms or None if no data."""
    buckets = _parse_histogram_buckets(text, metric_name)
    if not buckets or buckets[-1][1] == 0:
        return None
    total = buckets[-1][1]
    target = quantile * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if le == float("inf"):
            break  # don't interpolate into +Inf
        if count >= target:
            if count == prev_count:
                result_s = prev_le
            else:
                frac = (target - prev_count) / (count - prev_count)
                result_s = prev_le + frac * (le - prev_le)
            return result_s * 1000.0  # convert to ms
        prev_le, prev_count = le, count
    # All finite buckets exhausted — return the largest finite boundary
    return prev_le * 1000.0 if prev_le > 0.0 else None


@dataclass
class MetricsSnapshot:
    job_id: str
    timestamp: float
    replica_id: str | None = None
    # vLLM gauges
    avg_generation_throughput_toks_per_s: float = 0.0
    avg_prompt_throughput_toks_per_s: float = 0.0
    gpu_cache_usage_perc: float = 0.0       # 0.0–1.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    num_requests_swapped: int = 0
    request_success_total: float = 0.0
    num_preemptions_total: float = 0.0
    # vLLM counters (for computing throughput via deltas on vLLM >=0.10)
    generation_tokens_total: float = 0.0
    prompt_tokens_total: float = 0.0
    # Latency histograms (None until first requests complete)
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    ttft_ms_p99: float | None = None
    tpot_ms_p50: float | None = None
    tpot_ms_p95: float | None = None
    tpot_ms_p99: float | None = None
    e2e_ms_p50: float | None = None
    e2e_ms_p95: float | None = None
    e2e_ms_p99: float | None = None
    # Per-phase latency (vLLM v0.10.0+ only, None on v0)
    queue_time_ms_p50: float | None = None
    queue_time_ms_p95: float | None = None
    queue_time_ms_p99: float | None = None
    prefill_time_ms_p50: float | None = None
    prefill_time_ms_p95: float | None = None
    prefill_time_ms_p99: float | None = None
    decode_time_ms_p50: float | None = None
    decode_time_ms_p95: float | None = None
    decode_time_ms_p99: float | None = None
    # Inference time = total RUNNING phase (prefill + decode), V1 only
    inference_time_ms_p50: float | None = None
    inference_time_ms_p95: float | None = None
    inference_time_ms_p99: float | None = None
    # Prefix cache (vLLM v0.10.0+ only)
    prefix_cache_hit_rate: float | None = None

    @classmethod
    def from_prometheus_text(cls, job_id: str, text: str, timestamp: float) -> "MetricsSnapshot":
        snap = cls(job_id=job_id, timestamp=timestamp)
        # Throughput gauges (vLLM <0.10 — removed in 0.10.0, will be 0)
        snap.avg_generation_throughput_toks_per_s = sum_metric(
            text, "vllm:avg_generation_throughput_toks_per_s"
        )
        snap.avg_prompt_throughput_toks_per_s = sum_metric(
            text, "vllm:avg_prompt_throughput_toks_per_s"
        )
        # Token counters (vLLM >=0.10 — used to compute throughput via deltas)
        snap.generation_tokens_total = sum_metric_compat(text, "vllm:generation_tokens_total")
        snap.prompt_tokens_total = sum_metric_compat(text, "vllm:prompt_tokens_total")
        snap.gpu_cache_usage_perc = sum_metric_compat(text, "vllm:gpu_cache_usage_perc")
        snap.num_requests_running = int(sum_metric(text, "vllm:num_requests_running"))
        snap.num_requests_waiting = int(sum_metric(text, "vllm:num_requests_waiting"))
        snap.num_requests_swapped = int(sum_metric(text, "vllm:num_requests_swapped"))
        snap.request_success_total = sum_metric_compat(text, "vllm:request_success_total")
        snap.num_preemptions_total = sum_metric_compat(text, "vllm:num_preemptions_total")
        # Latency histograms
        snap.ttft_ms_p50 = histogram_quantile(text, "vllm:time_to_first_token_seconds", 0.50)
        snap.ttft_ms_p95 = histogram_quantile(text, "vllm:time_to_first_token_seconds", 0.95)
        snap.ttft_ms_p99 = histogram_quantile(text, "vllm:time_to_first_token_seconds", 0.99)
        snap.tpot_ms_p50 = histogram_quantile(text, "vllm:time_per_output_token_seconds", 0.50)
        snap.tpot_ms_p95 = histogram_quantile(text, "vllm:time_per_output_token_seconds", 0.95)
        snap.tpot_ms_p99 = histogram_quantile(text, "vllm:time_per_output_token_seconds", 0.99)
        snap.e2e_ms_p50 = histogram_quantile(text, "vllm:e2e_request_latency_seconds", 0.50)
        snap.e2e_ms_p95 = histogram_quantile(text, "vllm:e2e_request_latency_seconds", 0.95)
        snap.e2e_ms_p99 = histogram_quantile(text, "vllm:e2e_request_latency_seconds", 0.99)
        # Per-phase latency (vLLM v0.10.0+)
        snap.queue_time_ms_p50 = histogram_quantile(text, "vllm:request_queue_time_seconds", 0.50)
        snap.queue_time_ms_p95 = histogram_quantile(text, "vllm:request_queue_time_seconds", 0.95)
        snap.queue_time_ms_p99 = histogram_quantile(text, "vllm:request_queue_time_seconds", 0.99)
        snap.prefill_time_ms_p50 = histogram_quantile(text, "vllm:request_prefill_time_seconds", 0.50)
        snap.prefill_time_ms_p95 = histogram_quantile(text, "vllm:request_prefill_time_seconds", 0.95)
        snap.prefill_time_ms_p99 = histogram_quantile(text, "vllm:request_prefill_time_seconds", 0.99)
        snap.decode_time_ms_p50 = histogram_quantile(text, "vllm:request_decode_time_seconds", 0.50)
        snap.decode_time_ms_p95 = histogram_quantile(text, "vllm:request_decode_time_seconds", 0.95)
        snap.decode_time_ms_p99 = histogram_quantile(text, "vllm:request_decode_time_seconds", 0.99)
        # Inference time = total RUNNING phase (vLLM v0.10.0+)
        snap.inference_time_ms_p50 = histogram_quantile(text, "vllm:request_inference_time_seconds", 0.50)
        snap.inference_time_ms_p95 = histogram_quantile(text, "vllm:request_inference_time_seconds", 0.95)
        snap.inference_time_ms_p99 = histogram_quantile(text, "vllm:request_inference_time_seconds", 0.99)
        # Prefix cache hit rate (vLLM v0.10.0+)
        queries = sum_metric_compat(text, "vllm:prefix_cache_queries_total")
        hits = sum_metric_compat(text, "vllm:prefix_cache_hits_total")
        snap.prefix_cache_hit_rate = round(hits / queries, 4) if queries > 0 else None
        return snap

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "replica_id": self.replica_id,
            "avg_generation_throughput_toks_per_s": self.avg_generation_throughput_toks_per_s,
            "avg_prompt_throughput_toks_per_s": self.avg_prompt_throughput_toks_per_s,
            "gpu_cache_usage_perc": self.gpu_cache_usage_perc,
            "num_requests_running": self.num_requests_running,
            "num_requests_waiting": self.num_requests_waiting,
            "num_requests_swapped": self.num_requests_swapped,
            "request_success_total": self.request_success_total,
            "num_preemptions_total": self.num_preemptions_total,
            "generation_tokens_total": self.generation_tokens_total,
            "prompt_tokens_total": self.prompt_tokens_total,
            "ttft_ms_p50": self.ttft_ms_p50,
            "ttft_ms_p95": self.ttft_ms_p95,
            "ttft_ms_p99": self.ttft_ms_p99,
            "tpot_ms_p50": self.tpot_ms_p50,
            "tpot_ms_p95": self.tpot_ms_p95,
            "tpot_ms_p99": self.tpot_ms_p99,
            "e2e_ms_p50": self.e2e_ms_p50,
            "e2e_ms_p95": self.e2e_ms_p95,
            "e2e_ms_p99": self.e2e_ms_p99,
            "queue_time_ms_p50": self.queue_time_ms_p50,
            "queue_time_ms_p95": self.queue_time_ms_p95,
            "queue_time_ms_p99": self.queue_time_ms_p99,
            "prefill_time_ms_p50": self.prefill_time_ms_p50,
            "prefill_time_ms_p95": self.prefill_time_ms_p95,
            "prefill_time_ms_p99": self.prefill_time_ms_p99,
            "decode_time_ms_p50": self.decode_time_ms_p50,
            "decode_time_ms_p95": self.decode_time_ms_p95,
            "decode_time_ms_p99": self.decode_time_ms_p99,
            "inference_time_ms_p50": self.inference_time_ms_p50,
            "inference_time_ms_p95": self.inference_time_ms_p95,
            "inference_time_ms_p99": self.inference_time_ms_p99,
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
        }


_SUM_FIELDS = {
    "avg_generation_throughput_toks_per_s",
    "avg_prompt_throughput_toks_per_s",
    "generation_tokens_total",
    "prompt_tokens_total",
    "request_success_total",
    "num_preemptions_total",
    "num_requests_running",
    "num_requests_waiting",
    "num_requests_swapped",
}

_AVG_FIELDS = {
    "gpu_cache_usage_perc",
    "prefix_cache_hit_rate",
    "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
    "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
    "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99",
    "queue_time_ms_p50", "queue_time_ms_p95", "queue_time_ms_p99",
    "prefill_time_ms_p50", "prefill_time_ms_p95", "prefill_time_ms_p99",
    "decode_time_ms_p50", "decode_time_ms_p95", "decode_time_ms_p99",
    "inference_time_ms_p50", "inference_time_ms_p95", "inference_time_ms_p99",
}


def _merge_snapshots(job_id: str, snaps: list[MetricsSnapshot]) -> MetricsSnapshot:
    """Merge per-replica snapshots into a single aggregated view."""
    merged = MetricsSnapshot(job_id=job_id, timestamp=max(s.timestamp for s in snaps))
    for f in _SUM_FIELDS:
        setattr(merged, f, sum(getattr(s, f, 0) or 0 for s in snaps))
    for f in _AVG_FIELDS:
        vals = [v for s in snaps if (v := getattr(s, f, None)) is not None]
        setattr(merged, f, sum(vals) / len(vals) if vals else None)
    return merged


class _JobCollector:
    def __init__(self, job_id: str, endpoint_url: str | None):
        self.job_id = job_id
        self.endpoint_url = endpoint_url
        self.buffer: deque = deque(maxlen=RING_BUFFER_SIZE)
        self._unflushed: list = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll = endpoint_url is not None
        self._baseline_snap: MetricsSnapshot | None = None
        self._baseline_time: float | None = None
        if self._poll:
            self._thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
                name=f"orca-metrics-{job_id[:8]}",
            )

    def start(self) -> None:
        if self._poll:
            self._thread.start()
        # ingest-only: no thread; buffer ready immediately for ingest endpoint writes

    def stop(self) -> None:
        if self._poll:
            self._stop_event.set()
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("[MetricsCollector] flush thread %s did not exit cleanly", self.job_id)
        self._flush()  # always flush on stop regardless of mode

    def latest(self) -> MetricsSnapshot | None:
        with self.lock:
            return self.buffer[-1] if self.buffer else None

    def set_baseline(self) -> None:
        with self.lock:
            if self.buffer:
                self._baseline_snap = self.buffer[-1]
                self._baseline_time = self._baseline_snap.timestamp
            else:
                self._baseline_time = time.time()

    def get_sustained_throughput(self, window_sec: float = 60.0) -> dict | None:
        with self.lock:
            if len(self.buffer) < 2:
                return None
            current = self.buffer[-1]
            earliest_allowed = self._baseline_time or self.buffer[0].timestamp
            window_start = max(current.timestamp - window_sec, earliest_allowed)
            oldest = None
            for snap in self.buffer:
                if snap.timestamp >= window_start:
                    oldest = snap
                    break
            if oldest is None or oldest is current:
                return None
            dt = current.timestamp - oldest.timestamp
            if dt < 1.0:
                return None

        gen_tps = (current.generation_tokens_total - oldest.generation_tokens_total) / dt
        prompt_tps = (current.prompt_tokens_total - oldest.prompt_tokens_total) / dt

        result = {
            "generation_toks_per_s": round(gen_tps, 2),
            "prompt_toks_per_s": round(prompt_tps, 2),
            "window_actual_sec": round(dt, 2),
            "kv_cache_usage_perc": current.gpu_cache_usage_perc,
            "queue_time_ms_p50": current.queue_time_ms_p50,
            "queue_time_ms_p95": current.queue_time_ms_p95,
        }

        if self._baseline_snap is not None:
            epoch_dt = current.timestamp - self._baseline_snap.timestamp
            if epoch_dt > 1.0:
                result["epoch_generation_toks_per_s"] = round(
                    (current.generation_tokens_total - self._baseline_snap.generation_tokens_total) / epoch_dt, 2
                )
                result["epoch_prompt_toks_per_s"] = round(
                    (current.prompt_tokens_total - self._baseline_snap.prompt_tokens_total) / epoch_dt, 2
                )
                result["since_baseline_sec"] = round(epoch_dt, 2)

        return result

    def _poll_loop(self) -> None:
        last_flush = time.time()
        _prev_snap: MetricsSnapshot | None = None
        _generating = False       # requests are in-flight
        while not self._stop_event.is_set():
            try:
                r = _requests.get(self.endpoint_url + "/metrics", timeout=4)
                r.raise_for_status()
                snap = MetricsSnapshot.from_prometheus_text(self.job_id, r.text, time.time())

                # Always compute throughput from counter deltas (more accurate than
                # the gauge, which was a moving average removed in V1 anyway)
                if _prev_snap is not None:
                    dt = snap.timestamp - _prev_snap.timestamp
                    if dt > 0:
                        snap.avg_generation_throughput_toks_per_s = (
                            snap.generation_tokens_total - _prev_snap.generation_tokens_total
                        ) / dt
                        snap.avg_prompt_throughput_toks_per_s = (
                            snap.prompt_tokens_total - _prev_snap.prompt_tokens_total
                        ) / dt
                _prev_snap = snap

                with self.lock:
                    self.buffer.append(snap)
                    self._unflushed.append(snap.to_dict())

                # Update job status + progress based on what we observe
                try:
                    from orca_server.job_manager import get_job_tracker
                    jt = get_job_tracker()
                    rec = jt.get(self.job_id)
                    if rec:
                        # Requests in-flight → generating (fallback if runner didn't report)
                        if snap.num_requests_running > 0 and not _generating:
                            _generating = True
                            if rec.status not in ("generating",):
                                jt.update_status(self.job_id, "generating")

                        # Progress from completed requests
                        if snap.request_success_total > 0 and rec.state.spec.num_lines > 0:
                            frac = min(snap.request_success_total / rec.state.spec.num_lines, 0.99)
                            jt.update_progress(self.job_id, frac)
                except Exception:
                    pass
            except Exception as e:
                logger.debug("[MetricsCollector] poll error job=%s: %s", self.job_id, e)

            if time.time() - last_flush >= FLUSH_INTERVAL_SEC:
                self._flush()
                last_flush = time.time()

            self._stop_event.wait(timeout=POLL_INTERVAL_SEC)

        self._flush()  # final flush on stop

    def _flush(self) -> None:
        with self.lock:
            batch, self._unflushed = self._unflushed, []
        if not batch:
            return
        try:
            from orca_server.metrics_db import get_metrics_db
            get_metrics_db().append_timeseries(self.job_id, batch)
        except Exception as e:
            logger.warning(
                "[MetricsCollector] timeseries flush failed for %s: %s — data retained",
                self.job_id, e,
            )
            with self.lock:
                self._unflushed = batch + self._unflushed


class MetricsCollector:
    def __init__(self):
        self._jobs: dict[str, _JobCollector] = {}
        self._replicas: dict[str, _JobCollector] = {}  # "job_id:replica_id" → collector
        self._lock = threading.Lock()

    def start_collecting(self, job_id: str, endpoint_url: str | None = None) -> None:
        with self._lock:
            if job_id in self._jobs:
                return
            jc = _JobCollector(job_id, endpoint_url)
            self._jobs[job_id] = jc
        jc.start()

    def start_replica_collecting(self, job_id: str, replica_id: str) -> None:
        key = f"{job_id}:{replica_id}"
        with self._lock:
            if key in self._replicas:
                return
            jc = _JobCollector(job_id, None)  # ingest-only, no polling
            self._replicas[key] = jc
        jc.start()

    def stop_collecting(self, job_id: str) -> None:
        prefix = f"{job_id}:"
        with self._lock:
            jc = self._jobs.pop(job_id, None)
            replica_keys = [k for k in self._replicas if k.startswith(prefix)]
            replica_collectors = [self._replicas.pop(k) for k in replica_keys]
        if jc:
            jc.stop()
        for rc in replica_collectors:
            rc.stop()

    def get_latest(self, job_id: str) -> MetricsSnapshot | None:
        with self._lock:
            jc = self._jobs.get(job_id)
        return jc.latest() if jc else None

    def get_replica_latest(self, job_id: str, replica_id: str) -> MetricsSnapshot | None:
        key = f"{job_id}:{replica_id}"
        with self._lock:
            jc = self._replicas.get(key)
        return jc.latest() if jc else None

    def list_replica_ids(self, job_id: str) -> list[str]:
        prefix = f"{job_id}:"
        with self._lock:
            return [k[len(prefix):] for k in self._replicas if k.startswith(prefix)]

    def get_aggregated(self, job_id: str) -> MetricsSnapshot | None:
        """Return a merged view across all replicas, or fall back to get_latest."""
        replica_ids = self.list_replica_ids(job_id)
        if not replica_ids:
            return self.get_latest(job_id)
        snaps = [s for rid in replica_ids
                 if (s := self.get_replica_latest(job_id, rid)) is not None]
        if not snaps:
            return self.get_latest(job_id)
        return _merge_snapshots(job_id, snaps)

    def get_recent(self, job_id: str, n: int = 60) -> list[dict]:
        with self._lock:
            jc = self._jobs.get(job_id)
        if not jc:
            return []
        with jc.lock:
            snaps = list(jc.buffer)[-n:]
        return [s.to_dict() for s in snaps]

    def set_baseline(self, job_id: str) -> None:
        with self._lock:
            jc = self._jobs.get(job_id)
        if jc:
            jc.set_baseline()

    def get_sustained_throughput(self, job_id: str, window_sec: float = 60.0) -> dict | None:
        with self._lock:
            jc = self._jobs.get(job_id)
        return jc.get_sustained_throughput(window_sec) if jc else None

    def active_job_ids(self) -> list[str]:
        with self._lock:
            return list(self._jobs.keys())

    def prometheus_exposition(self) -> str:
        lines = [
            "# HELP orca_job_throughput_toks_per_s Generation throughput tokens/sec",
            "# TYPE orca_job_throughput_toks_per_s gauge",
        ]
        with self._lock:
            items = list(self._jobs.items())
        agg_snaps = {}
        for job_id, _jc in items:
            agg_snaps[job_id] = self.get_aggregated(job_id)
        for job_id, _jc in items:
            snap = agg_snaps[job_id]
            if snap is None:
                continue
            lines.append(
                f'orca_job_throughput_toks_per_s{{job_id="{job_id}"}} '
                f"{snap.avg_generation_throughput_toks_per_s}"
            )
        lines += [
            "# HELP orca_job_kv_cache_util KV cache utilization (0-1)",
            "# TYPE orca_job_kv_cache_util gauge",
        ]
        for job_id, _jc in items:
            snap = agg_snaps[job_id]
            if snap is None:
                continue
            lines.append(f'orca_job_kv_cache_util{{job_id="{job_id}"}} {snap.gpu_cache_usage_perc}')
        lines += [
            "# HELP orca_job_requests_running Running requests",
            "# TYPE orca_job_requests_running gauge",
        ]
        for job_id, _jc in items:
            snap = agg_snaps[job_id]
            if snap is None:
                continue
            lines.append(f'orca_job_requests_running{{job_id="{job_id}"}} {snap.num_requests_running}')
        lines += [
            "# HELP orca_job_requests_waiting Waiting requests",
            "# TYPE orca_job_requests_waiting gauge",
        ]
        for job_id, _jc in items:
            snap = agg_snaps[job_id]
            if snap is None:
                continue
            lines.append(f'orca_job_requests_waiting{{job_id="{job_id}"}} {snap.num_requests_waiting}')
        return "\n".join(lines) + "\n"

    def sse_generator(self, job_id: str) -> Generator[str, None, None]:
        yield "retry: 5000\n\n"
        while True:
            with self._lock:
                still_active = job_id in self._jobs
            snap = self.get_aggregated(job_id)
            if snap:
                data = snap.to_dict()
                sustained = self.get_sustained_throughput(job_id)
                if sustained:
                    data["sustained"] = sustained
                yield f"data: {json.dumps(data)}\n\n"
            elif not still_active:
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                return
            time.sleep(POLL_INTERVAL_SEC)


_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
