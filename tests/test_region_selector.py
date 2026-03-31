import pytest
from quota.region_selector import (
    get_instance_family,
    get_quota_code,
    get_ordered_regions,
    RegionCandidate,
    RegionQuota,
    INSTANCE_VCPUS,
    QUOTA_CODES,
)
from placement.roofline_magic import AWS_INSTANCE_TO_GPU


# --- get_instance_family ---


def test_instance_family_g6e():
    assert get_instance_family("g6e.12xlarge") == "g"


def test_instance_family_p4d():
    assert get_instance_family("p4d.24xlarge") == "p"


def test_instance_family_g5():
    assert get_instance_family("g5.48xlarge") == "g"


def test_instance_family_p5():
    assert get_instance_family("p5.48xlarge") == "p"


# --- get_quota_code ---


def test_quota_code_g_on_demand():
    assert get_quota_code("g", False) == "L-DB2E81BA"


def test_quota_code_p_spot():
    assert get_quota_code("p", True) == "L-7212CCBC"


# --- RegionCandidate.to_skypilot_resources ---


def test_region_candidate_to_skypilot():
    c = RegionCandidate(region="us-east-1", use_spot=True, available_quota=192)
    d = c.to_skypilot_resources("g6e.12xlarge")
    assert d["region"] == "us-east-1"
    assert d["instance_type"] == "g6e.12xlarge"
    assert d["use_spot"] is True
    assert "disk_size" in d
    assert "ports" in d


# --- get_ordered_regions (injected quotas, no AWS calls) ---


def test_get_ordered_regions_filters_insufficient(region_quotas):
    # g6e.48xlarge = 192 vCPU/node. eu-west-1 has 0 quota → excluded
    candidates = get_ordered_regions("g6e.48xlarge", num_nodes=1, quotas=region_quotas)
    regions = [c.region for c in candidates]
    assert "eu-west-1" not in regions


def test_get_ordered_regions_sort_by_quota(region_quotas):
    candidates = get_ordered_regions("g6e.48xlarge", num_nodes=1, quotas=region_quotas)
    # us-east-1 spot has 384 vCPU (highest) → should come first
    assert candidates[0].region == "us-east-1"
    assert candidates[0].use_spot is True


def test_get_ordered_regions_spot_preferred(region_quotas):
    candidates = get_ordered_regions(
        "g6e.48xlarge", num_nodes=1, quotas=region_quotas, prefer_spot=True
    )
    # At same quota level, spot should come before on-demand
    spot_indices = [i for i, c in enumerate(candidates) if c.use_spot and c.region == "us-east-1"]
    od_indices = [i for i, c in enumerate(candidates) if not c.use_spot and c.region == "us-east-1"]
    if spot_indices and od_indices:
        assert spot_indices[0] < od_indices[0]


def test_get_ordered_regions_no_viable():
    empty_quotas = {
        "us-east-1": RegionQuota(region="us-east-1", on_demand_vcpus=0, spot_vcpus=0),
    }
    candidates = get_ordered_regions("g6e.48xlarge", num_nodes=1, quotas=empty_quotas)
    assert candidates == []


# --- Cross-table consistency ---


def test_instance_vcpus_covers_roofline_table():
    """Every instance in roofline's AWS_INSTANCE_TO_GPU should have a vCPU entry."""
    missing = [inst for inst in AWS_INSTANCE_TO_GPU if inst not in INSTANCE_VCPUS]
    assert missing == [], f"Instances missing from INSTANCE_VCPUS: {missing}"
