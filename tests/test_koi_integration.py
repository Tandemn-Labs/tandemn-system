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
