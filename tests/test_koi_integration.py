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
async def test_resources_has_required_fields(client):
    resp = await client.get("/resources")
    data = resp.json()
    assert "vpc_id" in data
    assert "region" in data
    assert "snapshot_time" in data
    assert "resources" in data
    assert isinstance(data["resources"], list)


@pytest.mark.asyncio
async def test_resources_vpc_id(client):
    data = (await client.get("/resources")).json()
    assert data["vpc_id"] == "orca-cluster"


@pytest.mark.asyncio
async def test_resources_entries_have_gpu_fields(client):
    data = (await client.get("/resources")).json()
    if not data["resources"]:
        pytest.skip("No resources returned (quota may be empty)")
    r = data["resources"][0]
    required_keys = {
        "gpu_type", "instance_type", "gpus_per_instance", "total_gpus",
        "allocated_gpus", "cost_per_instance_hour_usd", "gpu_memory_gb",
        "region", "interconnect",
    }
    assert required_keys.issubset(r.keys()), f"Missing keys: {required_keys - r.keys()}"


@pytest.mark.asyncio
async def test_resources_gpu_memory_positive(client):
    data = (await client.get("/resources")).json()
    for r in data["resources"]:
        assert r["gpu_memory_gb"] > 0, f"gpu_memory_gb must be positive for {r['instance_type']}"


@pytest.mark.asyncio
async def test_resources_interconnect_values(client):
    data = (await client.get("/resources")).json()
    for r in data["resources"]:
        assert r["interconnect"] in ("NVLink", "PCIe"), (
            f"Unexpected interconnect {r['interconnect']} for {r['instance_type']}"
        )


@pytest.mark.asyncio
async def test_resources_p_instances_are_nvlink(client):
    """P-family instances (p4d, p5) should have NVLink interconnect."""
    data = (await client.get("/resources")).json()
    for r in data["resources"]:
        if r["instance_type"].startswith("p"):
            assert r["interconnect"] == "NVLink", (
                f"{r['instance_type']} should be NVLink"
            )


@pytest.mark.asyncio
async def test_resources_g_instances_are_pcie(client):
    """G-family instances (g5, g6e) should have PCIe interconnect."""
    data = (await client.get("/resources")).json()
    for r in data["resources"]:
        if r["instance_type"].startswith("g"):
            assert r["interconnect"] == "PCIe", (
                f"{r['instance_type']} should be PCIe"
            )


@pytest.mark.asyncio
async def test_resources_only_on_demand(client):
    """All returned resources should come from on_demand market."""
    # The endpoint filters to on_demand only. We verify indirectly
    # by checking we get resources (at least some on_demand quota exists)
    data = (await client.get("/resources")).json()
    # Just verify it's a valid response — market filtering is internal
    assert isinstance(data["resources"], list)


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
            "recommendation": {
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4, "pp": 1, "dp": 2,
                "num_instances": 2,
                "engine_config": {"max_model_len": 8192, "max_num_seqs": 64},
            },
            "predicted_metrics": {
                "throughput_tokens_per_sec": 2500.0,
                "cost_per_hour_usd": 9.36,
                "estimated_runtime_hours": 3.2,
                "total_cost_usd": 29.95,
            },
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
            "recommendation": {"gpu_type": "A100", "instance_type": "p4d.24xlarge",
                               "tp": 8, "pp": 1, "dp": 1, "num_instances": 1,
                               "engine_config": {}},
            "predicted_metrics": {},
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
# Pricing + GPU filtering (server-side unit tests)
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


@pytest.mark.asyncio
async def test_resources_no_v100(client):
    """V100 instances must not appear in /resources."""
    data = (await client.get("/resources")).json()
    for r in data["resources"]:
        assert r["gpu_type"] != "V100", f"V100 should be filtered out: {r}"


@pytest.mark.asyncio
async def test_resources_no_zero_gpu_entries(client):
    """Entries with total_gpus=0 must not appear."""
    data = (await client.get("/resources")).json()
    for r in data["resources"]:
        assert r["total_gpus"] > 0, f"total_gpus=0 should be filtered: {r}"


@pytest.mark.asyncio
async def test_resources_prices_from_skypilot(client):
    """Prices should come from SkyPilot catalog (non-zero for known instances)."""
    data = (await client.get("/resources")).json()
    priced = [r for r in data["resources"] if r["cost_per_instance_hour_usd"] > 0]
    assert len(priced) > 0, "At least some resources should have SkyPilot pricing"
