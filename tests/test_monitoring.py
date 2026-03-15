"""Unit tests for orca_server.monitoring."""
import threading
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from orca_server.monitoring import (
    MetricsCollector,
    MetricsSnapshot,
    _JobCollector,
    histogram_quantile,
    POLL_INTERVAL_SEC,
    RING_BUFFER_SIZE,
)

# ---------------------------------------------------------------------------
# Sample Prometheus text mimicking vLLM output
# ---------------------------------------------------------------------------

SAMPLE_PROM_TEXT = """\
# HELP vllm:avg_generation_throughput_toks_per_s Average generation throughput (tokens/s).
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s{model_name="Qwen/Qwen2.5-72B"} 853.6
# HELP vllm:avg_prompt_throughput_toks_per_s Average prompt throughput (tokens/s).
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s{model_name="Qwen/Qwen2.5-72B"} 120.5
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-72B"} 0.378
# HELP vllm:num_requests_running Running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-72B"} 194
# HELP vllm:num_requests_waiting Waiting requests.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-72B"} 12
# HELP vllm:num_requests_swapped Swapped requests.
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped{model_name="Qwen/Qwen2.5-72B"} 0
# HELP vllm:request_success_total Success requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="Qwen/Qwen2.5-72B",finished_reason="stop"} 500.0
# HELP vllm:num_preemptions_total Preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="Qwen/Qwen2.5-72B"} 2.0
"""

HISTOGRAM_TEXT = """\
# HELP vllm:time_to_first_token_seconds Time to first token.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="m"} 0
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="m"} 10
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 80
vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="m"} 98
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 100
vllm:time_to_first_token_seconds_count{model_name="m"} 100
vllm:time_to_first_token_seconds_sum{model_name="m"} 4.2
"""


# ---------------------------------------------------------------------------
# TestMetricsSnapshot
# ---------------------------------------------------------------------------

class TestMetricsSnapshot:
    def test_parse_all_gauges(self):
        snap = MetricsSnapshot.from_prometheus_text("job1", SAMPLE_PROM_TEXT, 0.0)
        assert snap.avg_generation_throughput_toks_per_s == pytest.approx(853.6)
        assert snap.avg_prompt_throughput_toks_per_s == pytest.approx(120.5)
        assert snap.gpu_cache_usage_perc == pytest.approx(0.378)
        assert snap.num_requests_running == 194
        assert snap.num_requests_waiting == 12
        assert snap.num_requests_swapped == 0
        assert snap.request_success_total == pytest.approx(500.0)
        assert snap.num_preemptions_total == pytest.approx(2.0)

    def test_empty_text_returns_zeros(self):
        snap = MetricsSnapshot.from_prometheus_text("job1", "", 0.0)
        assert snap.avg_generation_throughput_toks_per_s == 0.0
        assert snap.num_requests_running == 0
        assert snap.gpu_cache_usage_perc == 0.0

    def test_to_dict_has_expected_keys(self):
        snap = MetricsSnapshot.from_prometheus_text("job1", SAMPLE_PROM_TEXT, 1234567.0)
        d = snap.to_dict()
        expected_keys = {
            "job_id", "timestamp",
            "avg_generation_throughput_toks_per_s", "avg_prompt_throughput_toks_per_s",
            "gpu_cache_usage_perc", "num_requests_running", "num_requests_waiting",
            "num_requests_swapped", "request_success_total", "num_preemptions_total",
            "generation_tokens_total", "prompt_tokens_total",
            "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
            "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
            "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99",
        }
        assert set(d.keys()) == expected_keys
        assert d["job_id"] == "job1"
        assert d["timestamp"] == pytest.approx(1234567.0)

    def test_histogram_quantile_returns_float_ms(self):
        result = histogram_quantile(HISTOGRAM_TEXT, "vllm:time_to_first_token_seconds", 0.50)
        assert result is not None
        assert isinstance(result, float)
        assert result > 0  # should be in ms (not seconds)

    def test_histogram_empty_bucket_total_returns_none(self):
        empty = 'vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 0\n'
        result = histogram_quantile(empty, "vllm:time_to_first_token_seconds", 0.50)
        assert result is None

    def test_histogram_missing_returns_none(self):
        result = histogram_quantile("", "vllm:time_to_first_token_seconds", 0.50)
        assert result is None

    def test_histogram_p50_p95_p99_ordering(self):
        p50 = histogram_quantile(HISTOGRAM_TEXT, "vllm:time_to_first_token_seconds", 0.50)
        p95 = histogram_quantile(HISTOGRAM_TEXT, "vllm:time_to_first_token_seconds", 0.95)
        p99 = histogram_quantile(HISTOGRAM_TEXT, "vllm:time_to_first_token_seconds", 0.99)
        assert p50 <= p95 <= p99


# ---------------------------------------------------------------------------
# TestJobCollector
# ---------------------------------------------------------------------------

class TestJobCollector:
    def test_poll_loop_appends_snapshot(self):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_PROM_TEXT
        mock_resp.raise_for_status = MagicMock()

        with patch("orca_server.monitoring._requests.get", return_value=mock_resp):
            jc = _JobCollector("test-job-123", "http://fake:8001")
            jc.start()
            time.sleep(0.15)
            jc.stop()

        with jc.lock:
            assert len(jc.buffer) >= 1

        snap = jc.buffer[-1]
        assert snap.avg_generation_throughput_toks_per_s == pytest.approx(853.6)

    def test_ring_buffer_respects_maxlen(self):
        assert deque(maxlen=RING_BUFFER_SIZE).maxlen == RING_BUFFER_SIZE
        jc = _JobCollector("test-job", "http://fake:8001")
        for i in range(RING_BUFFER_SIZE + 50):
            jc.buffer.append(MetricsSnapshot("j", float(i)))
        assert len(jc.buffer) == RING_BUFFER_SIZE

    def test_http_errors_dont_crash_loop(self):
        with patch(
            "orca_server.monitoring._requests.get",
            side_effect=Exception("connection refused"),
        ):
            jc = _JobCollector("test-job-err", "http://fake:8001")
            jc.start()
            time.sleep(0.15)
            jc.stop()

        # No exception raised; buffer should be empty since all polls failed
        assert len(jc.buffer) == 0

    def test_latest_returns_none_when_empty(self):
        jc = _JobCollector("test-job", "http://fake:8001")
        assert jc.latest() is None

    def test_latest_returns_last_snapshot(self):
        jc = _JobCollector("test-job", "http://fake:8001")
        snap = MetricsSnapshot("test-job", 1.0)
        snap.avg_generation_throughput_toks_per_s = 999.0
        with jc.lock:
            jc.buffer.append(snap)
        result = jc.latest()
        assert result is snap


# ---------------------------------------------------------------------------
# TestMetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def setup_method(self):
        self.mc = MetricsCollector()

    def test_start_stop_idempotent(self):
        with patch("orca_server.monitoring._requests.get", side_effect=Exception("no server")):
            self.mc.start_collecting("job-a", "http://fake:8001")
            self.mc.start_collecting("job-a", "http://fake:8001")  # idempotent
            assert "job-a" in self.mc.active_job_ids()
            self.mc.stop_collecting("job-a")
            self.mc.stop_collecting("job-a")  # no-op, no raise

    def test_stop_unknown_job_is_noop(self):
        self.mc.stop_collecting("nonexistent-xyz")  # should not raise

    def test_get_latest_none_for_unknown(self):
        assert self.mc.get_latest("nonexistent-job") is None

    def test_get_recent_empty_for_unknown(self):
        assert self.mc.get_recent("nonexistent-job") == []

    def test_prometheus_exposition_format(self):
        jc = _JobCollector("fake-job-id", "http://x:8001")
        snap = MetricsSnapshot("fake-job-id", time.time())
        snap.avg_generation_throughput_toks_per_s = 500.0
        snap.gpu_cache_usage_perc = 0.5
        snap.num_requests_running = 10
        snap.num_requests_waiting = 2
        with jc.lock:
            jc.buffer.append(snap)
        with self.mc._lock:
            self.mc._jobs["fake-job-id"] = jc

        text = self.mc.prometheus_exposition()
        assert "orca_job_throughput_toks_per_s" in text
        assert "fake-job-id" in text
        assert "500.0" in text
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_active_job_ids_tracks_correctly(self):
        with patch("orca_server.monitoring._requests.get", side_effect=Exception("no server")):
            self.mc.start_collecting("job-x", "http://fake:8001")
            self.mc.start_collecting("job-y", "http://fake:8001")
            ids = self.mc.active_job_ids()
            assert "job-x" in ids
            assert "job-y" in ids
            self.mc.stop_collecting("job-x")
            ids2 = self.mc.active_job_ids()
            assert "job-x" not in ids2
            assert "job-y" in ids2
            self.mc.stop_collecting("job-y")


# ---------------------------------------------------------------------------
# TestSustainedThroughput
# ---------------------------------------------------------------------------

def _make_snap(job_id: str, ts: float, gen_total: float, prompt_total: float) -> MetricsSnapshot:
    snap = MetricsSnapshot(job_id=job_id, timestamp=ts)
    snap.generation_tokens_total = gen_total
    snap.prompt_tokens_total = prompt_total
    return snap


class TestSustainedThroughput:
    def test_insufficient_data(self):
        jc = _JobCollector("job-sus", None)
        assert jc.get_sustained_throughput() is None
        # Single snapshot still insufficient
        jc.buffer.append(_make_snap("job-sus", 100.0, 0, 0))
        assert jc.get_sustained_throughput() is None

    def test_basic_computation(self):
        jc = _JobCollector("job-sus", None)
        jc.buffer.append(_make_snap("job-sus", 100.0, 1000, 200))
        jc.buffer.append(_make_snap("job-sus", 110.0, 9500, 1300))
        result = jc.get_sustained_throughput(window_sec=60.0)
        assert result is not None
        assert result["generation_toks_per_s"] == pytest.approx(850.0)
        assert result["prompt_toks_per_s"] == pytest.approx(110.0)
        assert result["window_actual_sec"] == pytest.approx(10.0)

    def test_baseline_clips_window(self):
        jc = _JobCollector("job-sus", None)
        # t=0: pre-generation data (should be excluded)
        jc.buffer.append(_make_snap("job-sus", 100.0, 0, 0))
        jc.buffer.append(_make_snap("job-sus", 101.0, 100, 10))
        # baseline set at t=102 (generating starts)
        jc.buffer.append(_make_snap("job-sus", 102.0, 200, 20))
        jc.set_baseline()
        # t=103: generating data
        jc.buffer.append(_make_snap("job-sus", 103.0, 1200, 120))
        jc.buffer.append(_make_snap("job-sus", 110.0, 9200, 920))

        result = jc.get_sustained_throughput(window_sec=600.0)
        assert result is not None
        # Window should start at baseline (t=102), not t=100
        assert result["window_actual_sec"] == pytest.approx(8.0)
        # (9200 - 200) / 8 = 1125
        assert result["generation_toks_per_s"] == pytest.approx(1125.0)

    def test_epoch_throughput(self):
        jc = _JobCollector("job-sus", None)
        jc.buffer.append(_make_snap("job-sus", 100.0, 500, 50))
        jc.set_baseline()
        jc.buffer.append(_make_snap("job-sus", 105.0, 5500, 550))
        jc.buffer.append(_make_snap("job-sus", 110.0, 10500, 1050))

        result = jc.get_sustained_throughput(window_sec=60.0)
        assert result is not None
        # Epoch: (10500 - 500) / 10 = 1000
        assert result["epoch_generation_toks_per_s"] == pytest.approx(1000.0)
        assert result["epoch_prompt_toks_per_s"] == pytest.approx(100.0)
        assert result["since_baseline_sec"] == pytest.approx(10.0)

    def test_collector_delegation(self):
        mc = MetricsCollector()
        jc = _JobCollector("job-del", None)
        jc.buffer.append(_make_snap("job-del", 100.0, 1000, 100))
        with mc._lock:
            mc._jobs["job-del"] = jc

        # set_baseline through MetricsCollector (at t=100)
        mc.set_baseline("job-del")
        assert jc._baseline_snap is not None

        # More data after baseline
        jc.buffer.append(_make_snap("job-del", 110.0, 9000, 900))

        result = mc.get_sustained_throughput("job-del", window_sec=60.0)
        assert result is not None
        assert result["generation_toks_per_s"] == pytest.approx(800.0)

        # Unknown job returns None
        assert mc.get_sustained_throughput("nonexistent") is None
