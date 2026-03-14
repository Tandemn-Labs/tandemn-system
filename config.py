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

CHUNK_SIZE_MB = int(os.getenv("CHUNK_SIZE_MB", 8)) * 1024 * 1024

# --------------------------------------------------------------------------- #
# Canonical AWS instance table
#
# Single source of truth — every other mapping is derived from this dict.
# GPU counts confirmed from quota/aws_gpu_quota_by_region.csv.
#
#   {instance_type: (gpu_name, gpu_count, vcpus)}
# --------------------------------------------------------------------------- #
AWS_INSTANCES = {
    # P5 instances (H100)
    "p5.48xlarge": ("H100", 8, 192),
    # P4 instances (A100)
    "p4d.24xlarge": ("A100", 8, 96),
    "p4de.24xlarge": ("A100", 8, 96),
    # P3 instances (V100)
    "p3.2xlarge": ("V100", 1, 8),
    "p3.8xlarge": ("V100", 4, 32),
    "p3.16xlarge": ("V100", 8, 64),
    "p3dn.24xlarge": ("V100", 8, 96),
    # G6e instances (L40S)
    "g6e.xlarge": ("L40S", 1, 4),
    "g6e.2xlarge": ("L40S", 1, 8),
    "g6e.4xlarge": ("L40S", 1, 16),
    "g6e.8xlarge": ("L40S", 1, 32),
    "g6e.12xlarge": ("L40S", 4, 48),
    "g6e.16xlarge": ("L40S", 1, 64),
    "g6e.24xlarge": ("L40S", 4, 96),
    "g6e.48xlarge": ("L40S", 8, 192),
    # G6 instances (L4)
    "g6.xlarge": ("L4", 1, 4),
    "g6.2xlarge": ("L4", 1, 8),
    "g6.4xlarge": ("L4", 1, 16),
    "g6.8xlarge": ("L4", 1, 32),
    "g6.12xlarge": ("L4", 4, 48),
    "g6.16xlarge": ("L4", 1, 64),
    "g6.24xlarge": ("L4", 4, 96),
    "g6.48xlarge": ("L4", 8, 192),
    # G5 instances (A10G)
    "g5.xlarge": ("A10G", 1, 4),
    "g5.2xlarge": ("A10G", 1, 8),
    "g5.4xlarge": ("A10G", 1, 16),
    "g5.8xlarge": ("A10G", 1, 32),
    "g5.12xlarge": ("A10G", 4, 48),
    "g5.16xlarge": ("A10G", 1, 64),
    "g5.24xlarge": ("A10G", 4, 96),
    "g5.48xlarge": ("A10G", 8, 192),
}

# --------------------------------------------------------------------------- #
# Derived mappings (all generated from AWS_INSTANCES)
# --------------------------------------------------------------------------- #

# instance_type -> gpu_name
INSTANCE_TO_GPU = {inst: gpu for inst, (gpu, _, _) in AWS_INSTANCES.items()}

# instance_type -> vcpus
INSTANCE_VCPUS = {inst: vcpus for inst, (_, _, vcpus) in AWS_INSTANCES.items()}

# instance_type -> (gpu_name, gpu_count)  [used by roofline solver]
AWS_INSTANCE_TO_GPU = {
    inst: (gpu, count) for inst, (gpu, count, _) in AWS_INSTANCES.items()
}
