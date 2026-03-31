"""
Quota-aware region selection for SkyPilot cluster creation.

This module queries AWS quotas and returns an ordered list of regions
to try, filtering out regions without sufficient quota.

Algorithm:
1. Get required vCPUs for the instance type
2. Query quotas for all regions (both spot and on-demand)
3. Filter out regions where quota < required vCPUs
4. Sort by: quota amount (desc), then spot before on-demand (cost savings)
"""

import subprocess
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from orca_server.config import INSTANCE_VCPUS, VLLM_PORT

logger = logging.getLogger(__name__)

# AWS regions to consider (major GPU-available regions)
AWS_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
    "ap-northeast-1",  # Tokyo
    "ap-southeast-1",  # Singapore
]

# Instance family to quota code mapping
# G and VT instances share one quota, P instances have their own
QUOTA_CODES = {
    "g": "L-DB2E81BA",       # Running On-Demand G and VT instances
    "g_spot": "L-3819A6DF",  # All G and VT Spot Instance Requests
    "p": "L-417A185B",       # Running On-Demand P instances (all P families share this)
    "p_spot": "L-7212CCBC",  # All P Spot Instance Requests
}


def get_instance_family(instance_type: str) -> str:
    """Get the instance family (g or p) from instance type."""
    if instance_type.startswith(("g5", "g6", "g6e")):
        return "g"
    elif instance_type.startswith(("p3", "p4", "p5")):
        return "p"
    else:
        return "g"  # default


def get_quota_code(instance_family: str, use_spot: bool) -> str:
    """Get AWS quota code for the instance family and market."""
    key = f"{instance_family}_spot" if use_spot else instance_family
    return QUOTA_CODES.get(key, QUOTA_CODES["g"])


@dataclass
class RegionQuota:
    """Quota information for a specific region."""

    region: str
    on_demand_vcpus: int
    spot_vcpus: int


def query_quota(region: str, quota_code: str) -> int:
    """
    Query AWS quota for a specific region and quota code.

    Returns the quota value in vCPUs, or 0 if query fails.
    """
    try:
        result = subprocess.run(
            [
                "aws",
                "service-quotas",
                "get-service-quota",
                "--service-code",
                "ec2",
                "--quota-code",
                quota_code,
                "--region",
                region,
                "--query",
                "Quota.Value",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            value = result.stdout.strip()
            return int(float(value)) if value and value != "None" else 0
    except Exception as e:
        logger.warning(f"Failed to query quota for {region}: {e}")
    return 0


def get_all_quotas(instance_family: str = "g") -> Dict[str, RegionQuota]:
    """
    Query quotas for all regions for a given instance family.

    Returns dict mapping region -> RegionQuota
    """
    on_demand_code = get_quota_code(instance_family, use_spot=False)
    spot_code = get_quota_code(instance_family, use_spot=True)

    quotas = {}
    for region in AWS_REGIONS:
        on_demand = query_quota(region, on_demand_code)
        spot = query_quota(region, spot_code)
        quotas[region] = RegionQuota(
            region=region, on_demand_vcpus=on_demand, spot_vcpus=spot
        )
        logger.info(f"[Quota] {region}: on-demand={on_demand}, spot={spot}")

    return quotas


@dataclass
class RegionCandidate:
    """A candidate region for cluster placement."""

    region: str
    use_spot: bool
    available_quota: int

    def to_skypilot_resources(
        self, instance_type: str, disk_size: str = "300GB", ports: int = VLLM_PORT
    ) -> dict:
        """Convert to SkyPilot resources dict for any_of.

        Note: Don't specify 'infra' when using 'region' - region implies cloud.
        """
        return {
            "region": self.region,
            "instance_type": instance_type,
            "use_spot": self.use_spot,
            "disk_size": disk_size,
            "ports": ports,
        }


def get_ordered_regions(
    instance_type: str,
    num_nodes: int = 1,
    quotas: Optional[Dict[str, RegionQuota]] = None,
    prefer_spot: bool = True,
) -> List[RegionCandidate]:
    """
    Get ordered list of regions to try for cluster creation.

    Args:
        instance_type: AWS instance type (e.g., "g6e.12xlarge")
        num_nodes: Number of nodes needed (for multi-node clusters)
        quotas: Pre-fetched quotas, or None to query fresh
        prefer_spot: If True, prioritize spot over on-demand at same quota level

    Returns:
        List of RegionCandidate sorted by:
        1. Available quota (descending) - regions with more quota are more likely to have capacity
        2. Spot vs on-demand (spot first if prefer_spot=True)

        Regions with insufficient quota are filtered out.
    """
    # Get required vCPUs
    vcpus_per_instance = INSTANCE_VCPUS.get(instance_type, 96)
    required_vcpus = vcpus_per_instance * num_nodes

    logger.info(f"[RegionSelector] Instance: {instance_type}, Nodes: {num_nodes}")
    logger.info(f"[RegionSelector] Required vCPUs: {required_vcpus}")

    # Get instance family for quota lookup
    family = get_instance_family(instance_type)

    # Query quotas if not provided
    if quotas is None:
        logger.info(f"[RegionSelector] Querying {family.upper()} instance quotas...")
        quotas = get_all_quotas(family)

    # Build candidate list
    candidates: List[RegionCandidate] = []

    for region, quota in quotas.items():
        # Check spot quota
        if quota.spot_vcpus >= required_vcpus:
            candidates.append(
                RegionCandidate(
                    region=region, use_spot=True, available_quota=quota.spot_vcpus
                )
            )

        # Check on-demand quota
        if quota.on_demand_vcpus >= required_vcpus:
            candidates.append(
                RegionCandidate(
                    region=region, use_spot=False, available_quota=quota.on_demand_vcpus
                )
            )

    if not candidates:
        logger.warning(
            f"[RegionSelector] No regions have sufficient quota for {required_vcpus} vCPUs!"
        )
        logger.warning("[RegionSelector] Consider requesting quota increases.")
        return []

    # Sort: highest quota first, then spot before on-demand (if prefer_spot)
    def sort_key(c: RegionCandidate) -> tuple:
        # Higher quota = lower sort key (so it comes first)
        # prefer_spot: spot (True) comes before on-demand (False)
        spot_priority = 0 if (c.use_spot and prefer_spot) else 1
        return (-c.available_quota, spot_priority, c.region)

    candidates.sort(key=sort_key)

    logger.info(
        f"[RegionSelector] Found {len(candidates)} viable region/market combinations:"
    )
    for i, c in enumerate(candidates[:5]):  # Show top 5
        market = "spot" if c.use_spot else "on-demand"
        logger.info(f"  {i + 1}. {c.region} ({market}) - {c.available_quota} vCPUs")

    return candidates


def build_skypilot_any_of(
    instance_type: str,
    num_nodes: int = 1,
    max_candidates: int = 5,
    prefer_spot: bool = True,
    disk_size: str = "300GB",
    ports: int = VLLM_PORT,
) -> List[dict]:
    """
    Build SkyPilot any_of resources list for fallback region selection.

    Args:
        instance_type: AWS instance type
        num_nodes: Number of nodes needed
        max_candidates: Maximum number of fallback options
        prefer_spot: Prefer spot instances
        disk_size: Disk size for instances
        ports: Port to expose

    Returns:
        List of resource dicts for SkyPilot any_of, or empty list if no viable regions
    """
    candidates = get_ordered_regions(instance_type, num_nodes, prefer_spot=prefer_spot)

    if not candidates:
        return []

    # Convert to SkyPilot format
    any_of = []
    for c in candidates[:max_candidates]:
        any_of.append(c.to_skypilot_resources(instance_type, disk_size, ports))

    return any_of


# Cache for quota queries (refresh every 5 minutes)
_quota_cache: Dict[str, Tuple[float, Dict[str, RegionQuota]]] = {}
_CACHE_TTL = 300  # 5 minutes


def get_cached_quotas(instance_family: str = "g") -> Dict[str, RegionQuota]:
    """Get quotas with caching to avoid repeated AWS API calls."""
    import time

    cache_key = instance_family
    now = time.time()

    if cache_key in _quota_cache:
        cached_time, cached_quotas = _quota_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            logger.debug(f"[RegionSelector] Using cached quotas for {instance_family}")
            return cached_quotas

    # Fetch fresh quotas
    quotas = get_all_quotas(instance_family)
    _quota_cache[cache_key] = (now, quotas)
    return quotas


def print_quota_summary():
    """Print a summary of all GPU quotas across regions."""
    print("\n=== AWS GPU Instance Quotas ===\n")

    # Get both G/VT and P quotas
    g_quotas = get_all_quotas("g")
    p_quotas = get_all_quotas("p")

    print(
        f"{'Region':<15} {'G/VT On-Demand':>15} {'G/VT Spot':>12} {'P On-Demand':>12} {'P Spot':>10}"
    )
    print("-" * 70)

    for region in AWS_REGIONS:
        g = g_quotas.get(region, RegionQuota(region, 0, 0))
        p = p_quotas.get(region, RegionQuota(region, 0, 0))
        print(
            f"{region:<15} {g.on_demand_vcpus:>15} {g.spot_vcpus:>12} {p.on_demand_vcpus:>12} {p.spot_vcpus:>10}"
        )

    print()


# ---------------------------------------------------------------------------
# Automatic quota discovery from AWS Service Quotas API
# ---------------------------------------------------------------------------

# All AWS regions to scan (superset of AWS_REGIONS used for launch)
ALL_AWS_REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
    "ap-south-1", "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
    "ap-southeast-1", "ap-southeast-2",
    "ca-central-1", "sa-east-1",
]

# Family-level quota codes (source: https://spenserpothier.github.io/aws-quota-code-list/)
FAMILY_QUOTA_CODES_MAP = {
    "G":         {"on_demand": "L-DB2E81BA", "spot": "L-3819A6DF"},
    "P4_P3_P2":  {"on_demand": "L-417A185B", "spot": "L-7212CCBC"},
    "P5":        {"on_demand": "L-417A185B", "spot": "L-7212CCBC"},  # same as P4/P3/P2
}

GPU_VRAM_GB = {
    "H100": 80, "A100": 80, "V100": 32, "L40S": 48,
    "L4": 24, "A10G": 24, "T4": 16, "M60": 8, "Radeon Pro V520": 8,
}


def _boto3_get_quota(quota_code: str, region: str) -> int:
    """Fetch a single quota value via boto3."""
    try:
        from sky.adaptors import aws as aws_adaptor
        client = aws_adaptor.client("service-quotas", region_name=region)
        resp = client.get_service_quota(ServiceCode="ec2", QuotaCode=quota_code)
        return int(resp["Quota"]["Value"])
    except Exception:
        # Fallback to CLI
        return query_quota(region, quota_code)


def _fetch_family_region(family_type: str, region: str):
    """Fetch on-demand + spot quota for one (family, region) pair."""
    codes = FAMILY_QUOTA_CODES_MAP.get(family_type, FAMILY_QUOTA_CODES_MAP["G"])
    on_demand = _boto3_get_quota(codes["on_demand"], region)
    spot = _boto3_get_quota(codes["spot"], region)
    return (family_type, region, on_demand, spot)


def refresh_quotas_from_aws(
    csv_path: str = "quota/aws_gpu_quota_by_region.csv",
    quota_tracker=None,
):
    """Query AWS Service Quotas API and write/overwrite the quota CSV.

    Path A: CSV exists; keep hardware specs, overwrite quota columns.
    Path B: No CSV (fresh install); build specs from config.py AWS_INSTANCES.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import pandas as pd

    logger.info("[QuotaRefresh] Starting AWS quota discovery...")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logger.info("[QuotaRefresh] Updating existing CSV (%d instance types)", len(df))
    else:
        from orca_server.config import AWS_INSTANCES
        rows = []
        for inst, (gpu_name, gpu_count, vcpus) in sorted(AWS_INSTANCES.items()):
            family_raw = inst.split(".")[0]
            family_display = family_raw[0].upper() + family_raw[1:]
            if family_raw.startswith("g"):
                family_type = "G"
            elif family_raw.startswith("p5"):
                family_type = "P5"
            else:
                family_type = "P4_P3_P2"
            vram = GPU_VRAM_GB.get(gpu_name, 0)
            gpu_type_str = f"{gpu_count}x {gpu_name}" if gpu_count > 1 else f"1x {gpu_name}"
            rows.append({
                "Family": family_display,
                "Instance_Type": inst,
                "vCPU": vcpus,
                "GPU_Type": gpu_type_str,
                "VRAM_per_GPU": float(vram),
                "Total_VRAM": float(vram * gpu_count),
                "Family_Type": family_type,
            })
        df = pd.DataFrame(rows)
        logger.info("[QuotaRefresh] Fresh CSV from %d instances in AWS_INSTANCES", len(df))

    # Parallel fetch across all (family_type, region) combos
    family_types = sorted(df["Family_Type"].unique())
    tasks = [(ft, region) for ft in family_types for region in ALL_AWS_REGIONS]
    quotas = {}

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_fetch_family_region, ft, r): (ft, r) for ft, r in tasks}
        for future in as_completed(futures):
            try:
                ft, region, on_demand, spot = future.result()
                quotas[(ft, region)] = {"on_demand": on_demand, "spot": spot}
            except Exception as e:
                ft, r = futures[future]
                logger.warning("[QuotaRefresh] Failed %s/%s: %s", ft, r, e)
                quotas[(ft, r)] = {"on_demand": 0, "spot": 0}

    logger.info("[QuotaRefresh] Fetched quotas for %d (family, region) pairs", len(quotas))

    # Build quota columns
    for region in ALL_AWS_REGIONS:
        for market in ("on_demand", "spot"):
            col = f"{region}_{market}"
            df[col] = df["Family_Type"].map(
                lambda ft, r=region, m=market: quotas.get((ft, r), {}).get(m, 0)
            )

    # Ensure column order: spec cols first, then quota cols sorted
    spec_cols = ["Family", "Instance_Type", "vCPU", "GPU_Type", "VRAM_per_GPU", "Total_VRAM", "Family_Type"]
    quota_col_list = sorted([c for c in df.columns if c not in spec_cols])
    df = df[spec_cols + quota_col_list]

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("[QuotaRefresh] Wrote %d rows to %s", len(df), csv_path)

    if quota_tracker is not None:
        quota_tracker.reload_quota()
        logger.info("[QuotaRefresh] Reloaded quota tracker")


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print_quota_summary()

    # Test region selection for g6e.12xlarge (48 vCPUs)
    print("\n=== Testing region selection for g6e.12xlarge (1 node) ===")
    candidates = get_ordered_regions("g6e.12xlarge", num_nodes=1)

    print("\n=== Testing region selection for g6e.48xlarge (2 nodes = 384 vCPUs) ===")
    candidates = get_ordered_regions("g6e.48xlarge", num_nodes=2)

    print("\n=== SkyPilot any_of output ===")
    any_of = build_skypilot_any_of("g6e.12xlarge", num_nodes=1)
    print(json.dumps(any_of, indent=2))
