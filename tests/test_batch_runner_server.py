"""Unit tests for templates/vllm_batch_runner_server.py"""
import asyncio
import json
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import importlib
import os

_runner_path = os.path.join(os.path.dirname(__file__), "..", "templates", "vllm_batch_runner_server.py")
_spec = importlib.util.spec_from_file_location("vllm_batch_runner_server", _runner_path)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HISTOGRAM_TEXT = """\
vllm:time_per_output_token_seconds_bucket{le="0.001",model_name="m"} 0
vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="m"} 20
vllm:time_per_output_token_seconds_bucket{le="0.1",model_name="m"} 80
vllm:time_per_output_token_seconds_bucket{le="1.0",model_name="m"} 98
vllm:time_per_output_token_seconds_bucket{le="+Inf",model_name="m"} 100
"""

PREEMPTION_TEXT = "vllm:num_preemptions_total{model_name=\"m\"} 7.0\n"

COUNTER_TEXT_PRE = """\
vllm:prompt_tokens_total{model_name="m"} 100.0
vllm:generation_tokens_total{model_name="m"} 50.0
vllm:num_preemptions_total{model_name="m"} 0.0
"""

COUNTER_TEXT_POST = """\
vllm:prompt_tokens_total{model_name="m"} 10100.0
vllm:generation_tokens_total{model_name="m"} 5050.0
vllm:num_preemptions_total{model_name="m"} 3.0
"""


def _make_args(**kwargs):
    defaults = dict(
        model="Qwen/Qwen2.5-7B", tensor_parallel_size=1, pipeline_parallel_size=1,
        gpu_memory_utilization=0.90, max_num_seqs=32, max_model_len=None,
        dtype="auto", kv_cache_dtype="auto", quantization="none",
        cloud="aws", instance_type="g6e.12xlarge", gpu_name="L40S", engine="vllm",
        hf_token=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# wait_for_server
# ---------------------------------------------------------------------------

class TestWaitForServer:
    def test_ready_after_retries(self):
        responses = [MagicMock(status_code=503), MagicMock(status_code=200)]
        call_count = {"n": 0}

        def fake_get(url, timeout):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        with patch.object(runner.requests, "get", side_effect=fake_get), \
             patch.object(runner.time, "sleep"):
            elapsed = runner.wait_for_server(timeout_sec=60)
        assert isinstance(elapsed, float)

    def test_raises_timeout(self):
        with patch.object(runner.requests, "get", return_value=MagicMock(status_code=503)), \
             patch.object(runner.time, "sleep"), \
             patch.object(runner.time, "time", side_effect=[0.0, 61.0]):
            with pytest.raises(TimeoutError):
                runner.wait_for_server(timeout_sec=60)


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

class TestSidecar:
    def test_silent_when_no_url(self):
        with patch.object(runner.requests, "get") as mg, patch.object(runner.requests, "post") as mp:
            stop = threading.Event()
            stop.set()
            runner._sidecar_loop(stop, "", "key", "j")
        mg.assert_not_called()
        mp.assert_not_called()


# ---------------------------------------------------------------------------
# send_one
# ---------------------------------------------------------------------------

SSE_LINES = [
    b'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n',
    b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n',
    b'data: [DONE]\n',
]


class _AsyncContentIter:
    def __init__(self, lines):
        self._iter = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _MockResponseCM:
    def __init__(self, status, lines=None, text=None):
        self.status = status
        self.content = _AsyncContentIter(lines or [])
        self._text = text or ""

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestSendOne:
    def test_ttft_and_tpot_client(self):
        mock_cm = _MockResponseCM(status=200, lines=SSE_LINES)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        sem = asyncio.Semaphore(1)
        req = {"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "hi"}]}}
        times = iter([100.0, 100.1, 100.5])
        with patch.object(runner.time, "time", side_effect=times):
            result = asyncio.run(runner.send_one(mock_session, sem, req, "model"))
        assert result["status"] == "success"
        assert result["ttft_s"] == pytest.approx(0.1, abs=0.01)
        # tpot_client_s = (e2e - ttft) / (output_tokens - 1) = (0.5 - 0.1) / 4 = 0.1
        assert result["tpot_client_s"] == pytest.approx(0.1, abs=0.02)

    def test_context_length_exceeded(self):
        mock_cm = _MockResponseCM(status=400, text="context_length_exceeded: too long")
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        sem = asyncio.Semaphore(1)
        with patch.object(runner.time, "time", return_value=0.0):
            result = asyncio.run(runner.send_one(mock_session, sem, {"custom_id": "s", "body": {}}, "m"))
        assert result["status"] == "skipped"

    def test_usage_tokens(self):
        mock_cm = _MockResponseCM(status=200, lines=SSE_LINES)
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        sem = asyncio.Semaphore(1)
        with patch.object(runner.time, "time", return_value=0.0):
            result = asyncio.run(runner.send_one(mock_session, sem, {"custom_id": "t", "body": {}}, "m"))
        assert result["prompt_tokens"] == 10
        assert result["output_tokens"] == 5


# ---------------------------------------------------------------------------
# Prometheus delta helpers
# ---------------------------------------------------------------------------

class TestPromDelta:
    def test_delta_counter(self):
        delta = runner._delta_counter(COUNTER_TEXT_PRE, COUNTER_TEXT_POST, "vllm:prompt_tokens_total")
        assert delta == pytest.approx(10000.0)

    def test_delta_preemptions(self):
        delta = runner._delta_counter(COUNTER_TEXT_PRE, COUNTER_TEXT_POST, "vllm:num_preemptions_total")
        assert delta == pytest.approx(3.0)

    def test_delta_histogram_buckets(self):
        pre = 'vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="m"} 10\n'
        post = 'vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="m"} 30\n'
        delta = runner._delta_histogram_buckets(pre, post, "vllm:time_per_output_token_seconds")
        assert len(delta) == 1
        assert delta[0] == (0.01, 20.0)


# ---------------------------------------------------------------------------
# histogram_quantile
# ---------------------------------------------------------------------------

class TestHistogramQuantile:
    def test_p50_in_correct_range(self):
        result = runner.histogram_quantile(HISTOGRAM_TEXT, "vllm:time_per_output_token_seconds", 0.50)
        assert result is not None
        assert 10.0 < result < 100.0  # between 0.01s and 0.1s buckets → 10-100ms

    def test_p50_le_p95_le_p99(self):
        p50 = runner.histogram_quantile(HISTOGRAM_TEXT, "vllm:time_per_output_token_seconds", 0.50)
        p95 = runner.histogram_quantile(HISTOGRAM_TEXT, "vllm:time_per_output_token_seconds", 0.95)
        p99 = runner.histogram_quantile(HISTOGRAM_TEXT, "vllm:time_per_output_token_seconds", 0.99)
        assert p50 <= p95 <= p99

    def test_empty_returns_none(self):
        assert runner.histogram_quantile("", "vllm:time_per_output_token_seconds", 0.50) is None


# ---------------------------------------------------------------------------
# build_metrics
# ---------------------------------------------------------------------------

class TestBuildMetrics:
    def _sample_results(self):
        return [
            {"status": "success", "custom_id": "r1", "ttft_s": 0.05, "e2e_s": 0.5,
             "tpot_client_s": 0.009, "prompt_tokens": 100, "output_tokens": 50, "content": "hi"},
            {"status": "success", "custom_id": "r2", "ttft_s": 0.08, "e2e_s": 0.8,
             "tpot_client_s": 0.012, "prompt_tokens": 120, "output_tokens": 60, "content": "hello"},
            {"status": "skipped", "custom_id": "r3", "error": "context_length_exceeded"},
        ]

    def test_all_key_fields_present(self):
        args = _make_args()
        with patch.object(runner, "_get_price_per_hour", return_value=4.68):
            metrics = runner.build_metrics(
                self._sample_results(), COUNTER_TEXT_PRE, COUNTER_TEXT_POST, args,
                time.time() - 100, "2026-01-01T00:00:00", 10.0, 50.0, {}, {},
            )
        for key in [
            "ttft_ms_p50", "tpot_client_ms_p50", "tpot_ms_p50",
            "avg_sm_util_pct", "running_avg", "kv_cache_util_pct_avg",
            "cost_for_run_usd", "tokens_per_dollar",
            "throughput_input_tokens_per_sec", "num_preemptions",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_server_side_throughput_from_delta(self):
        args = _make_args()
        with patch.object(runner, "_get_price_per_hour", return_value=None):
            metrics = runner.build_metrics(
                self._sample_results(), COUNTER_TEXT_PRE, COUNTER_TEXT_POST, args,
                time.time() - 100, "ts", 10.0, 50.0, {}, {},
            )
        # Server-side: 10000 prompt + 5000 gen = 15000 tokens in 50s = 300 tok/s
        assert metrics["throughput_tokens_per_sec"] == pytest.approx(300.0)
        assert metrics["total_tokens"] == 15000

    def test_preemptions_from_delta(self):
        args = _make_args()
        with patch.object(runner, "_get_price_per_hour", return_value=None):
            metrics = runner.build_metrics(
                self._sample_results(), COUNTER_TEXT_PRE, COUNTER_TEXT_POST, args,
                time.time() - 100, "ts", 10.0, 50.0, {}, {},
            )
        assert metrics["num_preemptions"] == 3

    def test_cost_computed(self):
        args = _make_args()
        with patch.object(runner, "_get_price_per_hour", return_value=10.0):
            metrics = runner.build_metrics(
                self._sample_results(), "", "", args,
                time.time() - 100, "ts", 10.0, 3600.0, {}, {},
            )
        assert metrics["cost_for_run_usd"] == pytest.approx(10.0)

    def test_gpu_and_scheduler_summaries_passed_through(self):
        args = _make_args()
        gpu = {"avg_sm_util_pct": 75.0, "avg_mem_bw_util_pct": 40.0, "gpu_samples": 100}
        sched = {"running_avg": 32.0, "running_max": 64, "kv_cache_util_pct_avg": 45.0,
                 "scheduler_samples": 200}
        with patch.object(runner, "_get_price_per_hour", return_value=None):
            metrics = runner.build_metrics(
                self._sample_results(), "", "", args,
                time.time() - 100, "ts", 10.0, 50.0, gpu, sched,
            )
        assert metrics["avg_sm_util_pct"] == 75.0
        assert metrics["running_avg"] == 32.0
        assert metrics["kv_cache_util_pct_avg"] == 45.0
        assert metrics["scheduler_samples"] == 200


# ---------------------------------------------------------------------------
# GPUMonitor
# ---------------------------------------------------------------------------

class TestGPUMonitor:
    def test_summary_empty_when_no_pynvml(self):
        mon = runner.GPUMonitor()
        mon._pynvml = None
        mon.start()
        mon.stop()
        assert mon.get_summary() == {}


# ---------------------------------------------------------------------------
# MetricsPoller
# ---------------------------------------------------------------------------

class TestMetricsPoller:
    def test_summary_empty_when_no_data(self):
        poller = runner.MetricsPoller()
        assert poller.get_summary() == {}


# ---------------------------------------------------------------------------
# shutdown_server
# ---------------------------------------------------------------------------

class TestShutdownServer:
    def test_sigterm_and_wait(self):
        proc = MagicMock(spec=subprocess.Popen)
        runner.shutdown_server(proc, timeout=5)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()

    def test_sigkill_on_timeout(self):
        proc = MagicMock(spec=subprocess.Popen)
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd=[], timeout=5), None]
        runner.shutdown_server(proc, timeout=1)
        proc.kill.assert_called_once()
