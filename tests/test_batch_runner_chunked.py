"""Unit tests for templates/vllm_batch_runner_chunked.py.

Currently focused on the cold-start failure classification path —
``_classify_startup_exit`` and the ``wait_for_server`` integration that
drains the LogCollector buffer to decide whether the vLLM subprocess
crashed with a CUDA OOM signature or some other startup error.
"""
import importlib.util
import os
import time
from unittest.mock import MagicMock, patch

import pytest

_runner_path = os.path.join(
    os.path.dirname(__file__), "..", "templates", "vllm_batch_runner_chunked.py"
)
_spec = importlib.util.spec_from_file_location(
    "vllm_batch_runner_chunked", _runner_path
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


# ---------------------------------------------------------------------------
# _classify_startup_exit
# ---------------------------------------------------------------------------


class TestClassifyStartupExit:
    def _collector_with(self, lines):
        col = MagicMock()
        col.drain.return_value = [{"ts": time.time(), "msg": m} for m in lines]
        return col

    def test_no_collector_returns_basic_message(self):
        out = runner._classify_startup_exit(1, None)
        assert "vLLM exited with code 1 during startup" in out
        assert "CUDA out of memory" not in out

    def test_oom_lower_case_in_log_tail(self):
        col = self._collector_with(["RuntimeError: out of memory while warming up"])
        out = runner._classify_startup_exit(1, col)
        assert "CUDA out of memory" in out
        assert "exit" in out.lower() and "1" in out

    def test_cuda_oom_marker_matched(self):
        col = self._collector_with(["[CUDA OOM] failed allocation"])
        out = runner._classify_startup_exit(2, col)
        assert "CUDA out of memory" in out

    def test_non_oom_crash_no_oom_marker(self):
        col = self._collector_with(["AssertionError: model arch mismatch"])
        out = runner._classify_startup_exit(1, col)
        assert "CUDA out of memory" not in out
        assert "during startup" in out

    def test_empty_log_tail_no_oom(self):
        col = self._collector_with([])
        out = runner._classify_startup_exit(1, col)
        assert "CUDA out of memory" not in out


# ---------------------------------------------------------------------------
# wait_for_server (chunked variant)
# ---------------------------------------------------------------------------


class TestWaitForServerChunked:
    @staticmethod
    def _live_proc():
        p = MagicMock()
        p.poll.return_value = None
        return p

    def test_health_ok_returns_elapsed(self):
        with patch.object(runner.requests, "get", return_value=MagicMock(status_code=200)), \
             patch.object(runner.time, "sleep"):
            elapsed = runner.wait_for_server(self._live_proc(), timeout_sec=60)
        assert isinstance(elapsed, float)

    def test_proc_exit_with_oom_in_log_raises_oom_message(self):
        """The runner's wait_for_server should drain its LogCollector buffer
        when the subprocess dies and surface a reason string Koi's
        _classify_failure regex tags as OOM."""
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        col = MagicMock()
        col.drain.return_value = [
            {"ts": time.time(), "msg": "torch.cuda.OutOfMemoryError: CUDA out of memory"}
        ]
        with patch.object(runner.requests, "get", side_effect=ConnectionRefusedError), \
             patch.object(runner.time, "sleep"):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                runner.wait_for_server(dead_proc, timeout_sec=1200, log_collector=col)

    def test_proc_exit_without_oom_marker_raises_generic_startup_message(self):
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        col = MagicMock()
        col.drain.return_value = [
            {"ts": time.time(), "msg": "ValueError: unsupported architecture qwen3_5"}
        ]
        with patch.object(runner.requests, "get", side_effect=ConnectionRefusedError), \
             patch.object(runner.time, "sleep"):
            with pytest.raises(RuntimeError) as excinfo:
                runner.wait_for_server(dead_proc, timeout_sec=1200, log_collector=col)
        assert "during startup" in str(excinfo.value)
        assert "CUDA out of memory" not in str(excinfo.value)

    def test_proc_exit_without_log_collector_uses_basic_message(self):
        """Backward-compatible path: callers that don't pass a log_collector
        still get a sensible RuntimeError, just without OOM classification."""
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 137
        with patch.object(runner.requests, "get", side_effect=ConnectionRefusedError), \
             patch.object(runner.time, "sleep"):
            with pytest.raises(RuntimeError, match="exited with code 137"):
                runner.wait_for_server(dead_proc, timeout_sec=1200)

    def test_timeout_when_proc_alive_but_health_never_ready(self):
        with patch.object(runner.requests, "get", return_value=MagicMock(status_code=503)), \
             patch.object(runner.time, "sleep"), \
             patch.object(runner.time, "time", side_effect=[0.0, 61.0]):
            with pytest.raises(TimeoutError):
                runner.wait_for_server(self._live_proc(), timeout_sec=60)
