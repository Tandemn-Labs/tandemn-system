"""
Canonical configuration for the Orca server.

This module MUST be imported before any other project module that reads
env vars, because it calls ``load_dotenv()`` at import time.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VLLM_PORT = 8001
YAML_OUTPUT = "temp/output.yaml"

# --------------------------------------------------------------------------- #
# Environment variables
# --------------------------------------------------------------------------- #
S3_MODEL_BUCKET = "tandemn-model-shards"
S3_MODEL_PREFIX = "hf-models"

S3_UPLOAD_BUCKET = os.getenv("S3_UPLOAD_BUCKET", "tandemn-orca")
S3_UPLOAD_PREFIX = os.getenv("S3_UPLOAD_PREFIX", "uploads")

# Solver selection: "roofline" (deterministic) or "user_specified"
PLACEMENT_SOLVER = os.environ.get("TD_PLACEMENT_SOLVER", "roofline").lower()

# Optimization priority for roofline solver
PLACEMENT_PRIORITY = os.environ.get("TD_PLACEMENT_PRIORITY", "cost_first").lower()

# HuggingFace token for gated models
HF_TOKEN = os.environ.get("HF_TOKEN")

CHUNK_SIZE_BYTES = int(os.getenv("CHUNK_SIZE_MB", 8)) * 1024 * 1024

# Control plane URL reachable from EC2 clusters (empty = local dev, sidecar disabled)
TD_SERVER_URL = os.getenv("TD_SERVER_URL", "")
ORCA_API_KEY = os.getenv("ORCA_API_KEY", "")

# Koi placement service URL (if set, Orca POSTs job completion events to Koi)
KOI_SERVICE_URL = os.getenv("KOI_SERVICE_URL", "")

# Redis (for chunked distributed batch)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_CHUNK_SIZE_LINES = int(os.getenv("CHUNK_SIZE_LINES", "1000"))
CHUNK_LEASE_TTL_SEC = int(os.getenv("CHUNK_LEASE_TTL_SEC", "600"))
CHUNK_MAX_RETRIES = int(os.getenv("CHUNK_MAX_RETRIES", "3"))
CHUNK_RECLAIM_INTERVAL = int(os.getenv("CHUNK_RECLAIM_INTERVAL_SEC", "60"))
CHUNK_RENEW_INTERVAL = int(os.getenv("CHUNK_RENEW_INTERVAL_SEC", "30"))

# Replica watchdog — heartbeat-based dead replica detection
REPLICA_DEAD_THRESHOLD_SEC = int(os.getenv("REPLICA_DEAD_THRESHOLD_SEC", "45"))
WATCHDOG_POLL_INTERVAL_SEC = int(os.getenv("WATCHDOG_POLL_INTERVAL_SEC", "10"))
RECOVERY_COOLDOWN_SEC = int(os.getenv("RECOVERY_COOLDOWN_SEC", "300"))

# --------------------------------------------------------------------------- #
# Canonical AWS instance table
#
# Single source of truth — every other mapping is derived from this dict.
# GPU counts confirmed from quota/aws_gpu_quota_by_region.csv.
#
#   {instance_type: (gpu_name, gpu_count, vcpus, vram_per_gpu_gb)}
# --------------------------------------------------------------------------- #
AWS_INSTANCES = {
    # P5 instances (H100 80GB)
    "p5.48xlarge": ("H100", 8, 192, 80),
    # P4 instances (A100)
    "p4d.24xlarge": ("A100", 8, 96, 40),  # A100 40GB HBM2e
    "p4de.24xlarge": ("A100", 8, 96, 80),  # A100 80GB HBM2e
    # P3 instances (V100)
    "p3.2xlarge": ("V100", 1, 8, 16),  # V100 16GB
    "p3.8xlarge": ("V100", 4, 32, 16),
    "p3.16xlarge": ("V100", 8, 64, 32),  # V100 32GB
    "p3dn.24xlarge": ("V100", 8, 96, 32),
    # G6e instances (L40S 48GB)
    "g6e.xlarge": ("L40S", 1, 4, 48),
    "g6e.2xlarge": ("L40S", 1, 8, 48),
    "g6e.4xlarge": ("L40S", 1, 16, 48),
    "g6e.8xlarge": ("L40S", 1, 32, 48),
    "g6e.12xlarge": ("L40S", 4, 48, 48),
    "g6e.16xlarge": ("L40S", 1, 64, 48),
    "g6e.24xlarge": ("L40S", 4, 96, 48),
    "g6e.48xlarge": ("L40S", 8, 192, 48),
    # G6 instances (L4 24GB)
    "g6.xlarge": ("L4", 1, 4, 24),
    "g6.2xlarge": ("L4", 1, 8, 24),
    "g6.4xlarge": ("L4", 1, 16, 24),
    "g6.8xlarge": ("L4", 1, 32, 24),
    "g6.12xlarge": ("L4", 4, 48, 24),
    "g6.16xlarge": ("L4", 1, 64, 24),
    "g6.24xlarge": ("L4", 4, 96, 24),
    "g6.48xlarge": ("L4", 8, 192, 24),
    # G5 instances (A10G 24GB)
    "g5.xlarge": ("A10G", 1, 4, 24),
    "g5.2xlarge": ("A10G", 1, 8, 24),
    "g5.4xlarge": ("A10G", 1, 16, 24),
    "g5.8xlarge": ("A10G", 1, 32, 24),
    "g5.12xlarge": ("A10G", 4, 48, 24),
    "g5.16xlarge": ("A10G", 1, 64, 24),
    "g5.24xlarge": ("A10G", 4, 96, 24),
    "g5.48xlarge": ("A10G", 8, 192, 24),
}

# --------------------------------------------------------------------------- #
# Derived mappings (all generated from AWS_INSTANCES)
# --------------------------------------------------------------------------- #

# instance_type -> gpu_name
INSTANCE_TO_GPU = {inst: gpu for inst, (gpu, *_) in AWS_INSTANCES.items()}

# instance_type -> vcpus
INSTANCE_VCPUS = {inst: vals[2] for inst, vals in AWS_INSTANCES.items()}

# instance_type -> vram_per_gpu_gb
INSTANCE_VRAM = {inst: vals[3] for inst, vals in AWS_INSTANCES.items()}

# instance_type -> (gpu_name, gpu_count)  [used by roofline solver]
AWS_INSTANCE_TO_GPU = {
    inst: (gpu, count) for inst, (gpu, count, *_) in AWS_INSTANCES.items()
}

# GPUs with compute capability < 8.0 — vLLM V1 engine is not supported
_V1_UNSUPPORTED_GPUS = {"V100", "T4"}


def supports_vllm_v1(instance_type: str) -> bool:
    """Check if an instance type's GPU supports vLLM V1 engine (CC >= 8.0)."""
    gpu = INSTANCE_TO_GPU.get(instance_type, "")
    return gpu not in _V1_UNSUPPORTED_GPUS


# --------------------------------------------------------------------------- #
# vLLM + transformers pin
#
# vLLM 0.19.1 is the validated version for production. It works for both
# single-node and multi-node pipeline parallelism (tp4,pp3 / tp4,pp4 / tp8,pp2
# all tested healthy). vLLM 0.20.0 introduced a regression in the v1 engine's
# multi-node PP executor that breaks cross-node worker initialization, so we
# stay on 0.19.1 until upstream lands a fix.
#
# 0.19.1 requires CUDA 13.0 / driver 580+ which is only available on the
# Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) family.
# The default SkyPilot AMI ships driver 535 / CUDA 12.2 which fails with
# "RuntimeError: NVIDIA driver too old (found version 12020)".
#
# transformers must stay pinned at 4.57.6 — vLLM still uses
# `all_special_tokens_extended` which was removed in transformers 4.47+,
# and we ship a compatibility shim (templates/vllm_compat_patch.py) that
# expects this exact range.
# --------------------------------------------------------------------------- #
DEFAULT_VLLM_VERSION = os.getenv("ORCA_VLLM_VERSION", "0.19.1")
DEFAULT_TRANSFORMERS_VERSION = os.getenv("ORCA_TRANSFORMERS_VERSION", "4.57.6")


# --------------------------------------------------------------------------- #
# GPU AMI map
#
# All entries are the same DLAMI image (Deep Learning Base OSS Nvidia Driver
# GPU AMI, Ubuntu 22.04, dated 2026-04-24). Same image_id across regions but
# different AMI IDs because AWS scopes AMI IDs per region.
#
# Driver: 580.x | CUDA: 13.0 | PyTorch 2.x+cu130 (installed by vLLM 0.19.1)
# Supports: G4dn, G5, G6, G6e, P4d, P4de, P5, P5e, P5en
#
# To refresh: `aws ec2 describe-images --owners amazon --filters
#   "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu
#   22.04) <YYYYMMDD>"` for each region.
#
# When adding a new region, update this map; otherwise launcher will skip
# that region for vLLM-installable clusters.
# --------------------------------------------------------------------------- #
GPU_AMI_MAP: dict[str, str] = {
    "us-east-1": "ami-0da2c6477a5ee1361",
    "us-east-2": "ami-0f1a998148248b8fa",
    "us-west-1": "ami-0b7295d7db6ef312f",
    "us-west-2": "ami-00212c5e5c3b2ec85",
    "ap-northeast-2": "ami-0896d4f01f89b1ec8",
    "eu-west-1": "ami-014cf2391cb36896e",
    "eu-central-1": "ami-0db8c8f0d0336e45b",
}


def ami_for_region(region: str) -> str | None:
    """Return the DLAMI ID for the given region, or None if unmapped.

    Callers should treat None as "this region is not eligible for vLLM
    clusters" — the default AMI ships an older NVIDIA driver that fails
    with vLLM 0.19.1.
    """
    return GPU_AMI_MAP.get(region)
