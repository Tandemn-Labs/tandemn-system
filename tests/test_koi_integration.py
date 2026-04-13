"""Tests for Phase 1 Koi + Orca integration.

Covers:
  - Task 1: KOI_SERVICE_URL config
  - Task 2: GET /resources endpoint
  - Task 3: cmd_plan Koi parallel path
  - Task 4: cmd_deploy Koi 3-option prompt
  - Task 5: --skip-dangerously flag
"""

import importlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ──────────────────────────────────────────────────────────────────────────────
# Task 1: KOI_SERVICE_URL in config
# ──────────────────────────────────────────────────────────────────────────────

class TestKoiConfig:
    def test_koi_service_url_exists(self):
        from orca_server.config import KOI_SERVICE_URL
        assert isinstance(KOI_SERVICE_URL, str)

    def test_koi_service_url_default_empty(self):
        """Default is empty string when env var not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KOI_SERVICE_URL", None)
            # Re-import to get fresh value
            import importlib
            import orca_server.config as cfg
            importlib.reload(cfg)
            assert cfg.KOI_SERVICE_URL == ""

    def test_koi_service_url_from_env(self):
        """Reads from KOI_SERVICE_URL env var."""
        with patch.dict(os.environ, {"KOI_SERVICE_URL": "http://koi:8090"}):
            import importlib
            import orca_server.config as cfg
            importlib.reload(cfg)
            assert cfg.KOI_SERVICE_URL == "http://koi:8090"


# ──────────────────────────────────────────────────────────────────────────────
# Task 2: GET /resources endpoint
# ──────────────────────────────────────────────────────────────────────────────

from server import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_resources_returns_200(client):
    resp = await client.get("/resources")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_resources_top_level_fields(client):
    data = (await client.get("/resources")).json()
    assert data["vpc_id"] == "orca-cluster"
    assert "snapshot_time" in data
    assert isinstance(data["instances"], list)
    assert isinstance(data["quotas"], list)


@pytest.mark.asyncio
async def test_resources_instance_schema(client):
    data = (await client.get("/resources")).json()
    assert len(data["instances"]) > 0, "Should have at least one instance"
    required = {
        "instance_type", "gpu_type", "gpus_per_instance", "vcpus",
        "quota_family", "gpu_memory_gb", "interconnect",
        "cost_per_instance_hour_usd",
    }
    for inst in data["instances"]:
        assert required.issubset(inst.keys()), f"Missing: {required - inst.keys()}"


@pytest.mark.asyncio
async def test_resources_quota_schema(client):
    data = (await client.get("/resources")).json()
    assert len(data["quotas"]) > 0, "Should have at least one quota entry"
    required = {"family", "region", "market", "baseline_vcpus", "used_vcpus"}
    for q in data["quotas"]:
        assert required.issubset(q.keys()), f"Missing: {required - q.keys()}"
        assert q["baseline_vcpus"] > 0


@pytest.mark.asyncio
async def test_resources_no_v100(client):
    """V100 instances must not appear."""
    data = (await client.get("/resources")).json()
    for inst in data["instances"]:
        assert inst["gpu_type"] != "V100", f"V100 should be filtered: {inst}"


@pytest.mark.asyncio
async def test_resources_only_multi_gpu(client):
    """Only multi-GPU instances (useful for LLM inference)."""
    data = (await client.get("/resources")).json()
    for inst in data["instances"]:
        assert inst["gpus_per_instance"] >= 2, (
            f"{inst['instance_type']} has {inst['gpus_per_instance']} GPU — should be filtered"
        )


@pytest.mark.asyncio
async def test_resources_interconnect_values(client):
    data = (await client.get("/resources")).json()
    for inst in data["instances"]:
        assert inst["interconnect"] in ("NVLink", "PCIe")
        if inst["instance_type"].startswith("p"):
            assert inst["interconnect"] == "NVLink"
        if inst["instance_type"].startswith("g"):
            assert inst["interconnect"] == "PCIe"


@pytest.mark.asyncio
async def test_resources_quota_families_match_instances(client):
    """Every instance's quota_family should appear in the quotas list."""
    data = (await client.get("/resources")).json()
    quota_families = {q["family"] for q in data["quotas"]}
    for inst in data["instances"]:
        assert inst["quota_family"] in quota_families, (
            f"{inst['instance_type']} has family {inst['quota_family']} not in quotas"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 & 4: CLI Koi helpers (unit tests, no server needed)
# ──────────────────────────────────────────────────────────────────────────────

# Import the orca CLI as a module
ORCA_CLI = os.path.join(os.path.dirname(__file__), "..", "orca")


def _load_orca_module():
    """Load the orca CLI as a Python module for testing."""
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("orca_cli", ORCA_CLI)
    spec = importlib.util.spec_from_loader("orca_cli", loader, origin=ORCA_CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def orca_mod():
    return _load_orca_module()


class TestKoiHelpers:
    def test_koi_service_url_in_cli(self, orca_mod):
        """CLI reads KOI_SERVICE_URL."""
        assert hasattr(orca_mod, "KOI_SERVICE_URL")
        assert isinstance(orca_mod.KOI_SERVICE_URL, str)

    def test_fetch_resources_returns_none_on_failure(self, orca_mod):
        """fetch_resources returns None when server is unreachable."""
        # Point at a server that doesn't exist
        original = orca_mod.ORCA_SERVER
        orca_mod.ORCA_SERVER = "http://127.0.0.1:1"
        try:
            result = orca_mod.fetch_resources()
            assert result is None
        finally:
            orca_mod.ORCA_SERVER = original

    def test_call_koi_returns_none_when_disabled(self, orca_mod):
        """call_koi returns None when KOI_SERVICE_URL is empty."""
        original = orca_mod.KOI_SERVICE_URL
        orca_mod.KOI_SERVICE_URL = ""
        try:
            result = orca_mod.call_koi({"model_name": "test"}, {"resources": []})
            assert result is None
        finally:
            orca_mod.KOI_SERVICE_URL = original

    def test_call_koi_returns_none_on_connection_error(self, orca_mod):
        """call_koi returns None when Koi service is unreachable."""
        original = orca_mod.KOI_SERVICE_URL
        orca_mod.KOI_SERVICE_URL = "http://127.0.0.1:1"
        try:
            result = orca_mod.call_koi({"model_name": "test"}, {"resources": []}, timeout=1)
            assert result is None
        finally:
            orca_mod.KOI_SERVICE_URL = original

    def test_call_koi_returns_none_on_malformed_json(self, orca_mod):
        """call_koi should fail closed if Koi returns invalid JSON."""
        original = orca_mod.KOI_SERVICE_URL
        orca_mod.KOI_SERVICE_URL = "http://koi:8090"
        bad_response = MagicMock(status_code=200)
        bad_response.json.side_effect = ValueError("invalid json")
        try:
            with patch.object(orca_mod.requests, "post", return_value=bad_response):
                result = orca_mod.call_koi({"model_name": "test"}, {"resources": []}, timeout=1)
            assert result is None
        finally:
            orca_mod.KOI_SERVICE_URL = original

    def test_call_koi_returns_none_on_timeout(self, orca_mod):
        """call_koi should fail closed if Koi times out."""
        original = orca_mod.KOI_SERVICE_URL
        orca_mod.KOI_SERVICE_URL = "http://koi:8090"
        try:
            with patch.object(orca_mod.requests, "post", side_effect=orca_mod.requests.Timeout):
                result = orca_mod.call_koi({"model_name": "test"}, {"resources": []}, timeout=1)
            assert result is None
        finally:
            orca_mod.KOI_SERVICE_URL = original

    def test_koi_summary_lines_basic(self, orca_mod):
        """_koi_summary_lines produces display lines from Koi response."""
        koi_data = {
            "config": {
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4, "pp": 1, "dp": 2,
                "num_instances": 2,
                "engine_config": {"max_model_len": 8192, "max_num_seqs": 64},
            },
            "predicted_tps": 2500.0,
            "predicted_cost_per_hour": 9.36,
            "predicted_runtime_hours": 3.2,
            "predicted_total_cost": 29.95,
            "confidence": 0.78,
        }
        lines = orca_mod._koi_summary_lines(koi_data)
        assert len(lines) > 0
        text = "\n".join(lines)
        assert "L40S" in text
        assert "g6e.12xlarge" in text
        assert "2500" in text
        assert "9.36" in text

    def test_koi_summary_lines_minimal(self, orca_mod):
        """_koi_summary_lines handles minimal Koi response."""
        koi_data = {
            "config": {"gpu_type": "A100", "instance_type": "p4d.24xlarge",
                       "tp": 8, "pp": 1, "dp": 1, "num_instances": 1,
                       "engine_config": {}},
        }
        lines = orca_mod._koi_summary_lines(koi_data)
        assert len(lines) >= 3  # instance, GPU, parallelism lines at minimum


class TestKoiWebhookNotifications:
    def test_submit_batch_chunked_passes_koi_alternatives_to_launcher(self):
        """Chunked submit should preserve Koi alternatives as ordered fallbacks."""
        import asyncio
        import server
        from models.requests import BatchedRequest

        captured = {}
        chunk_manager = MagicMock()
        job_tracker = MagicMock()

        async def fake_launch_chunked_replicas(request, configs, num_replicas, **kwargs):
            captured["request"] = request
            captured["configs"] = configs
            captured["num_replicas"] = num_replicas
            captured["kwargs"] = kwargs
            return True

        def fake_resolve_gpu_type_to_instance(gpu_type, tp):
            mapping = {
                ("L40S", 4): ("g6e.12xlarge", 4),
                ("A100", 8): ("p4de.24xlarge", 8),
            }
            return mapping[(gpu_type, tp)]

        request = BatchedRequest(
            user_id="test-user",
            input_file="s3://bucket/input.jsonl",
            output_file="output.jsonl",
            num_lines=1000,
            avg_input_tokens=953,
            max_input_tokens=2048,
            avg_output_tokens=1024,
            max_output_tokens=2048,
            description="test",
            task_type="batch",
            task_priority="low",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            engine="vllm",
            slo_mode="throughput",
            slo_deadline_hours=8.0,
            placement="auto",
            placement_solver="user_specified",
            gpu_type="L40S",
            tp_size=4,
            pp_size=1,
            replicas=1,
            chunks=[{
                "chunk_id": 0,
                "s3_input_path": "s3://bucket/job/chunk-0.jsonl",
                "num_lines": 1000,
            }],
            koi_alternatives=[{"gpu_type": "A100", "tp": 8, "pp": 1}],
        )

        feasibility = {
            "feasible": True,
            "reason": "",
            "max_model_len": 8192,
            "solution": {
                "throughput_tokens_per_sec": 1234.0,
                "cost_per_hour": 9.36,
            },
        }

        old_redis = getattr(server.app.state, "redis_available", False)
        server.app.state.redis_available = True
        try:
            with patch.object(server, "_make_job_id", return_value="job-123"), \
                 patch.object(server, "resolve_gpu_type_to_instance", side_effect=fake_resolve_gpu_type_to_instance), \
                 patch.object(server, "check_user_specified_feasibility", return_value=feasibility), \
                 patch.object(server, "get_cached_quotas", return_value=[]), \
                 patch.object(server, "get_ordered_regions", return_value=[object()]), \
                 patch.object(server, "get_chunk_manager", return_value=chunk_manager), \
                 patch.object(server, "get_job_tracker", return_value=job_tracker), \
                 patch.object(server, "get_quota_tracker", return_value=MagicMock()), \
                 patch.object(server, "launch_chunked_replicas", new=fake_launch_chunked_replicas):
                data = asyncio.run(server.submit_batch(request))
        finally:
            server.app.state.redis_available = old_redis

        assert data["status"] == "launched"
        assert data["job_id"] == "job-123"
        assert captured["num_replicas"] == 1
        assert [cfg.instance_type for cfg in captured["configs"]] == [
            "g6e.12xlarge",
            "p4de.24xlarge",
        ]
        assert [cfg.tp_size for cfg in captured["configs"]] == [4, 8]
        assert [cfg.pp_size for cfg in captured["configs"]] == [1, 1]
        assert all(cfg.decision_id == "job-123" for cfg in captured["configs"])
        chunk_manager.create_job_queue.assert_called_once()
        job_tracker.set_chunked_info.assert_called_once_with("job-123", 1, 1)

    def test_notify_koi_replica_ready_sends_actual_fallback_payload(self):
        """model_ready webhook should report actual launched config to Koi."""
        import server
        import orca_server.config as cfg

        mock_cm = MagicMock()
        mock_cm.get_replica_states.return_value = {
            "parent-job-r1": {
                "instance_type": "p4de.24xlarge",
                "region": "us-west-2",
                "market": "on_demand",
                "tp": 8,
                "pp": 1,
                "config_index": 1,
                "koi_webhook_info": {
                    "group_id": "parent-job",
                    "decision_id": "dec-123",
                    "slo_deadline_hours": 8.0,
                    "total_tokens": 6_000_000,
                    "predicted_tps": 1500.0,
                    "deploy_timestamp": 5400.0,
                },
            },
        }

        old_cm = getattr(server.app.state, "cluster_manager", None)
        server.app.state.cluster_manager = mock_cm
        try:
            with patch.object(cfg, "KOI_SERVICE_URL", "http://koi:8090"), \
                 patch.dict(cfg.INSTANCE_TO_GPU, {"p4de.24xlarge": "A100-80GB"}, clear=False), \
                 patch("time.time", return_value=7200.0), \
                 patch("requests.post") as post:
                server._notify_koi_replica_ready("parent-job", "parent-job-r1")

            post.assert_called_once()
            assert post.call_args.args[0] == "http://koi:8090/job/started"
            payload = post.call_args.kwargs["json"]
            assert payload["job_id"] == "parent-job-r1"
            assert payload["group_id"] == "parent-job"
            assert payload["decision_id"] == "dec-123"
            assert payload["gpu_type"] == "A100-80GB"
            assert payload["instance_type"] == "p4de.24xlarge"
            assert payload["region"] == "us-west-2"
            assert payload["market"] == "on_demand"
            assert payload["tp"] == 8
            assert payload["pp"] == 1
            assert payload["predicted_tps"] == 1500.0
            assert payload["is_fallback"] is True
            assert payload["slo_deadline_hours"] == pytest.approx(7.5)
        finally:
            if old_cm is None:
                delattr(server.app.state, "cluster_manager")
            else:
                server.app.state.cluster_manager = old_cm

    def test_cmd_deploy_uses_top_level_koi_predicted_tps_for_chunked_submit(self, orca_mod):
        """Chunked deploy should forward Koi predicted_tps from the top-level response."""
        from types import SimpleNamespace

        plan_response = MagicMock(status_code=200)
        plan_response.json.return_value = {
            "status": "ok",
            "placements": [{
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp_size": 4,
                "pp_size": 1,
                "cost_per_hour": 9.36,
            }],
        }
        submit_response = MagicMock(status_code=200)
        submit_response.json.return_value = {
            "status": "launched",
            "job_id": "job-123",
            "config": {
                "instance_type": "p4de.24xlarge",
                "tp": 8,
                "pp": 1,
            },
            "chunks": 1,
            "replicas": 1,
        }
        captured = {}

        def fake_api(method, path, **kwargs):
            assert method == "post"
            assert path == "/test/placement"
            return plan_response

        def fake_api_with_spinner(method, path, message="Working...", **kwargs):
            assert method == "post"
            assert path == "/submit/batch"
            captured["payload"] = kwargs["json"]
            return submit_response

        args = SimpleNamespace(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            input_file="/tmp/input.jsonl",
            output="output.jsonl",
            slo=8.0,
            max_output_tokens=1024,
            gpu=None,
            tp=None,
            pp=None,
            force=False,
            chunk_size=1000,
            replicas=None,
            persist=False,
            on_demand=False,
            log_level="INFO",
            s3_models=None,
            skip_dangerously=False,
        )
        koi_data = {
            "_decision_id": "dec-123",
            "config": {
                "gpu_type": "A100-80GB",
                "instance_type": "p4de.24xlarge",
                "tp": 8,
                "pp": 1,
                "dp": 2,
                "num_instances": 1,
                "engine_config": {},
            },
            "predicted_tps": 2500.0,
            "alternatives": [{"gpu_type": "L40S", "tp": 4, "pp": 1}],
        }

        original_koi = orca_mod.KOI_SERVICE_URL
        orca_mod.KOI_SERVICE_URL = "http://koi:8090"
        try:
            with patch.object(orca_mod, "parse_input_file", return_value={
                "num_lines": 1000,
                "avg_input_tokens": 953,
                "max_input_tokens": 2048,
                "has_explicit_max_tokens": False,
            }), \
                 patch.object(orca_mod.os.path, "exists", return_value=True), \
                 patch.object(orca_mod, "upload_to_server", return_value="s3://bucket/input.jsonl"), \
                 patch.object(orca_mod, "split_and_upload_chunks", return_value=[{
                     "chunk_id": 0,
                     "s3_input_path": "s3://bucket/chunk-0.jsonl",
                     "num_lines": 1000,
                 }]), \
                 patch.object(orca_mod, "fetch_resources", return_value={"resources": []}), \
                 patch.object(orca_mod, "call_koi", return_value=koi_data), \
                 patch.object(orca_mod, "api", side_effect=fake_api), \
                 patch.object(orca_mod, "api_with_spinner", side_effect=fake_api_with_spinner), \
                 patch.object(orca_mod, "spinner", return_value=None), \
                 patch("builtins.input", return_value="1"):
                orca_mod.cmd_deploy(args)
        finally:
            orca_mod.KOI_SERVICE_URL = original_koi

        assert captured["payload"]["koi_decision_id"] == "dec-123"
        assert captured["payload"]["koi_predicted_tps"] == 2500.0
        assert captured["payload"]["gpu_type"] == "A100-80GB"
        assert captured["payload"]["replicas"] == 2

    def test_launch_chunked_replicas_fallback_success_updates_ready_payload(self):
        """If the primary config fails, the started webhook should reflect the fallback config."""
        import asyncio
        import threading as real_threading
        import server
        import orca_server.config as cfg
        import orca_server.launcher as launcher
        from models.requests import BatchedRequest
        from models.resources import MagicOutput
        from orca_server.job_manager import ClusterManager

        RealThread = real_threading.Thread

        class InlineThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
                self._thread = RealThread(
                    target=target,
                    args=args,
                    kwargs=kwargs or {},
                    daemon=daemon,
                    name=name,
                )

            def start(self):
                self._thread.start()
                self._thread.join()

        async def fake_launch_replica(
            request,
            config,
            replica_id,
            parent_job_id,
            job_dirname,
            job_logger=None,
            quota_tracker=None,
            persist=False,
            config_index=0,
            koi_webhook_info=None,
        ):
            if config_index == 0:
                raise RuntimeError(f"{config.instance_type} failed")
            launcher._notify_koi_config_attempted(
                parent_job_id,
                koi_webhook_info,
                config.instance_type,
                region="us-east-1",
                market="on_demand",
                launched=True,
                attempt_index=config_index,
                time_to_launch=42.0,
            )
            cm.register(
                replica_id,
                parent_job_id,
                region="us-east-1",
                market="on_demand",
                instance_type=config.instance_type,
                num_instances=config.num_nodes,
            )
            cm.set_replica_state(
                parent_job_id,
                replica_id,
                phase="provisioned",
                region="us-east-1",
                market="on_demand",
                instance_type=config.instance_type,
                koi_webhook_info=koi_webhook_info,
                tp=config.tp_size,
                pp=config.pp_size,
                config_index=config_index,
            )

        jt = MagicMock()
        jt.build_job_state_batched.return_value = MagicMock()
        jt.get.return_value = MagicMock(status="launching")
        cm = ClusterManager()
        request = BatchedRequest(
            user_id="test-user",
            input_file="s3://bucket/input.jsonl",
            output_file="output.jsonl",
            num_lines=1000,
            avg_input_tokens=953,
            avg_output_tokens=1024,
            description="test",
            task_type="batch",
            task_priority="low",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            engine="vllm",
            slo_mode="throughput",
            slo_deadline_hours=8.0,
            placement="auto",
            replicas=1,
            chunks=[{"chunk_id": 0}],
            koi_decision_id="dec-123",
        )
        configs = [
            MagicOutput(
                decision_id="parent-job",
                engine="vllm",
                instance_type="g6e.12xlarge",
                tp_size=4,
                pp_size=1,
                replicas=1,
                num_instances=1,
            ),
            MagicOutput(
                decision_id="parent-job",
                engine="vllm",
                instance_type="p4de.24xlarge",
                tp_size=8,
                pp_size=1,
                replicas=1,
                num_instances=1,
            ),
        ]

        old_cm = getattr(server.app.state, "cluster_manager", None)
        server.app.state.cluster_manager = cm
        try:
            with patch.object(launcher._cfg, "ORCA_SERVER_URL", "http://orca"), \
                 patch.object(cfg, "KOI_SERVICE_URL", "http://koi:8090"), \
                 patch.dict(cfg.INSTANCE_TO_GPU, {
                     "g6e.12xlarge": "L40S",
                     "p4de.24xlarge": "A100-80GB",
                 }, clear=False), \
                 patch("orca_server.launcher.generate_job_dirname", return_value="test-jobdir"), \
                 patch("orca_server.launcher.setup_job_logger", return_value=MagicMock()), \
                 patch("orca_server.launcher.get_job_tracker", return_value=jt), \
                 patch("orca_server.launcher.get_cluster_manager", return_value=cm), \
                 patch("orca_server.launcher._launch_chunked_replica", new=fake_launch_replica), \
                 patch("orca_server.launcher._notify_koi_config_attempted") as notify_attempted, \
                 patch("orca_server.launcher.sky_down_with_retry"), \
                 patch("orca_server.launcher.threading.Thread", InlineThread):
                ok = asyncio.run(
                    launcher.launch_chunked_replicas(
                        request=request,
                        configs=configs,
                        num_replicas=1,
                    )
                )

            assert ok is True
            assert notify_attempted.call_count == 2
            assert notify_attempted.call_args_list[0].kwargs["launched"] is False
            assert notify_attempted.call_args_list[0].args[2] == "g6e.12xlarge"
            assert notify_attempted.call_args_list[1].kwargs["launched"] is True
            assert notify_attempted.call_args_list[1].args[2] == "p4de.24xlarge"

            deploy_ts = cm.get_replica_states("parent-job")["parent-job-r0"]["koi_webhook_info"]["deploy_timestamp"]
            with patch.object(cfg, "KOI_SERVICE_URL", "http://koi:8090"), \
                 patch.dict(cfg.INSTANCE_TO_GPU, {"p4de.24xlarge": "A100-80GB"}, clear=False), \
                 patch("time.time", return_value=deploy_ts + 1800), \
                 patch("requests.post") as post:
                server._notify_koi_replica_ready("parent-job", "parent-job-r0")

            post.assert_called_once()
            assert post.call_args.args[0] == "http://koi:8090/job/started"
            payload = post.call_args.kwargs["json"]
            assert payload["job_id"] == "parent-job-r0"
            assert payload["group_id"] == "parent-job"
            assert payload["decision_id"] == "dec-123"
            assert payload["gpu_type"] == "A100-80GB"
            assert payload["instance_type"] == "p4de.24xlarge"
            assert payload["region"] == "us-east-1"
            assert payload["market"] == "on_demand"
            assert payload["tp"] == 8
            assert payload["pp"] == 1
            assert payload["total_tokens"] == 1_977_000
            assert payload["is_fallback"] is True
        finally:
            if old_cm is None:
                delattr(server.app.state, "cluster_manager")
            else:
                server.app.state.cluster_manager = old_cm

    def test_notify_koi_config_attempted_sends_failure_payload(self):
        """Launch-attempt webhook should send config failure details to Koi."""
        import orca_server.config as cfg
        from orca_server.launcher import _notify_koi_config_attempted

        with patch.object(cfg, "KOI_SERVICE_URL", "http://koi:8090"), \
             patch.dict(cfg.INSTANCE_TO_GPU, {"g6e.12xlarge": "L40S"}, clear=False), \
             patch("requests.post") as post:
            _notify_koi_config_attempted(
                parent_job_id="parent-job",
                koi_webhook_info={"decision_id": "dec-123"},
                instance_type="g6e.12xlarge",
                region="us-east-1",
                market="spot",
                launched=False,
                attempt_index=1,
                failure_reason="InsufficientCapacity",
            )

        post.assert_called_once()
        assert post.call_args.args[0] == "http://koi:8090/job/config-attempted"
        payload = post.call_args.kwargs["json"]
        assert payload["job_id"] == "parent-job"
        assert payload["decision_id"] == "dec-123"
        assert payload["instance_type"] == "g6e.12xlarge"
        assert payload["gpu_type"] == "L40S"
        assert payload["region"] == "us-east-1"
        assert payload["market"] == "spot"
        assert payload["launched"] is False
        assert payload["failure_reason"] == "InsufficientCapacity"
        assert payload["attempt_index"] == 1

    def test_launch_chunked_replicas_notifies_koi_when_all_configs_fail(self):
        """All failed chunked launch attempts should emit one /job/launch-failed webhook."""
        import asyncio
        import threading as real_threading
        import orca_server.config as cfg
        import orca_server.launcher as launcher
        from models.requests import BatchedRequest
        from models.resources import MagicOutput

        RealThread = real_threading.Thread

        class InlineThread:
            def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
                self._thread = RealThread(
                    target=target,
                    args=args,
                    kwargs=kwargs or {},
                    daemon=daemon,
                    name=name,
                )

            def start(self):
                self._thread.start()
                self._thread.join()

        async def fail_launch(*args, **_kwargs):
            config = args[1]
            raise RuntimeError(f"{config.instance_type} failed")

        jt = MagicMock()
        jt.build_job_state_batched.return_value = MagicMock()
        jt.get.return_value = MagicMock(status="launching")
        cm = MagicMock()
        cm.get_replica_states.return_value = {
            "parent-job-r0": {"phase": "failed"},
        }
        request = BatchedRequest(
            user_id="test-user",
            input_file="s3://bucket/input.jsonl",
            output_file="output.jsonl",
            num_lines=1000,
            avg_input_tokens=953,
            avg_output_tokens=1024,
            description="test",
            task_type="batch",
            task_priority="low",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            engine="vllm",
            slo_mode="throughput",
            slo_deadline_hours=8.0,
            placement="auto",
            replicas=1,
            chunks=[{"chunk_id": 0}],
            koi_decision_id="dec-123",
        )
        configs = [
            MagicOutput(
                decision_id="parent-job",
                engine="vllm",
                instance_type="g6e.12xlarge",
                tp_size=4,
                pp_size=1,
                replicas=1,
                num_instances=1,
            ),
            MagicOutput(
                decision_id="parent-job",
                engine="vllm",
                instance_type="p4de.24xlarge",
                tp_size=8,
                pp_size=1,
                replicas=1,
                num_instances=1,
            ),
        ]

        with patch.object(launcher._cfg, "ORCA_SERVER_URL", "http://orca"), \
             patch.object(cfg, "KOI_SERVICE_URL", "http://koi:8090"), \
             patch.dict(cfg.INSTANCE_TO_GPU, {
                 "g6e.12xlarge": "L40S",
                 "p4de.24xlarge": "A100-80GB",
             }, clear=False), \
             patch("orca_server.launcher.generate_job_dirname", return_value="test-jobdir"), \
             patch("orca_server.launcher.setup_job_logger", return_value=MagicMock()), \
             patch("orca_server.launcher.get_job_tracker", return_value=jt), \
             patch("orca_server.launcher.get_cluster_manager", return_value=cm), \
             patch("orca_server.launcher._launch_chunked_replica", new=fail_launch), \
             patch("orca_server.launcher._notify_koi_config_attempted"), \
             patch("orca_server.launcher.sky_down_with_retry"), \
             patch("orca_server.launcher.threading.Thread", InlineThread), \
             patch("requests.post") as post:
            ok = asyncio.run(
                launcher.launch_chunked_replicas(
                    request=request,
                    configs=configs,
                    num_replicas=1,
                )
            )

        assert ok is True
        jt.update_status.assert_any_call("parent-job", "failed")
        post.assert_called_once()
        assert post.call_args.args[0] == "http://koi:8090/job/launch-failed"
        payload = post.call_args.kwargs["json"]
        assert payload["job_id"] == "parent-job"
        assert payload["decision_id"] == "dec-123"
        assert [a["instance_type"] for a in payload["configs_tried"]] == [
            "g6e.12xlarge",
            "p4de.24xlarge",
        ]
        assert payload["failure_reasons"] == [
            "g6e.12xlarge failed",
            "p4de.24xlarge failed",
        ]
        assert payload["total_time_seconds"] >= 0


# ──────────────────────────────────────────────────────────────────────────────
# Task 5: --skip-dangerously flag
# ──────────────────────────────────────────────────────────────────────────────

class TestSkipDangerouslyFlag:
    def test_flag_accepted_by_parser(self, orca_mod):
        """deploy parser accepts --skip-dangerously."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        # Re-create parser to test flag existence
        args = orca_mod.main  # just verify module loads; test via subprocess below

    def test_skip_dangerously_parsed(self):
        """--skip-dangerously is parsed as True."""
        result = subprocess.run(
            [sys.executable, ORCA_CLI, "deploy", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--skip-dangerously" in result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Integration: backward compatibility (no Koi configured)
# ──────────────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_plan_help_works(self):
        """orca plan --help still works."""
        result = subprocess.run(
            [sys.executable, ORCA_CLI, "plan", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "model_name" in result.stdout

    def test_deploy_help_works(self):
        """orca deploy --help still works."""
        result = subprocess.run(
            [sys.executable, ORCA_CLI, "deploy", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--skip-dangerously" in result.stdout
        assert "--slo" in result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# Pricing + filtering (server-side unit tests)
# ──────────────────────────────────────────────────────────────────────────────

class TestResourceFiltering:
    def test_skypilot_pricing_returns_positive(self):
        """SkyPilot catalog returns a positive price for known instances."""
        from server import _get_instance_price
        price = _get_instance_price("g6e.12xlarge", "us-east-1")
        assert price > 0

    def test_skypilot_pricing_cached(self):
        """Second call uses cache."""
        from server import _get_instance_price, _pricing_cache
        _pricing_cache.pop("g5.12xlarge:us-east-1", None)
        p1 = _get_instance_price("g5.12xlarge", "us-east-1")
        p2 = _get_instance_price("g5.12xlarge", "us-east-1")
        assert p1 == p2

    def test_no_v100_in_koi_gpu_types(self):
        """V100 is excluded from Koi-supported GPUs."""
        from server import _KOI_GPU_TYPES
        assert "V100" not in _KOI_GPU_TYPES

    def test_koi_gpu_types_coverage(self):
        """Koi GPU types includes the main inference GPUs."""
        from server import _KOI_GPU_TYPES
        assert {"H100", "A100", "L40S", "A10G", "L4"} == _KOI_GPU_TYPES

    def test_instance_prefixes_filter(self):
        """Only multi-GPU instance prefixes are allowed."""
        from server import _KOI_INSTANCE_PREFIXES
        # Should include the big instances
        assert any("g6e.12xlarge" in p for p in _KOI_INSTANCE_PREFIXES)
        assert any("p5." in p for p in _KOI_INSTANCE_PREFIXES)
        # Should NOT include single-GPU instances
        assert not any("g6e.xlarge" == p for p in _KOI_INSTANCE_PREFIXES)
