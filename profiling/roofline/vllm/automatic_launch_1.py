import csv
import json
import os
import time
import select
from dataclasses import dataclass
import subprocess
from collections import defaultdict
import sys
import argparse
from pathlib import Path
import requests
import signal
import atexit
import logging
from datetime import datetime

# GPU type configurations
# Each GPU type maps to: available GPU counts per instance, instance family, and pricing
GPU_CONFIGS = {
    "L40S": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g6e",
        "pricing": {
            1: {"instance_type": "g6e.2xlarge", "price_per_hour": 0.99},
            4: {"instance_type": "g6e.12xlarge", "price_per_hour": 4.68},
            8: {"instance_type": "g6e.48xlarge", "price_per_hour": 13.35},
        },
        "canonical_gpu_name": "L40S",
        "gpu_mem_gb": 48,
        "gpu_tflops_fp16": 362,
        "gpu_bandwidth_gbps": 864,
        "gpu_generation": "Ada Lovelace",
        "interconnect": "PCIe",
    },
    "L4": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g6",
        "pricing": {
            1: {"instance_type": "g6.2xlarge", "price_per_hour": 0.526},
            4: {"instance_type": "g6.12xlarge", "price_per_hour": 0.752},
            8: {"instance_type": "g6.48xlarge", "price_per_hour": 1.204},
        },
        "canonical_gpu_name": "L4",
        "gpu_mem_gb": 24,
        "gpu_tflops_fp16": 121,
        "gpu_bandwidth_gbps": 300,
        "gpu_generation": "Ada Lovelace",
        "interconnect": "PCIe",
    },
    "A10G": {
        "available_gpus": [1, 4, 8],
        "instance_family": "g5",
        "pricing": {
            1: {"instance_type": "g5.2xlarge", "price_per_hour": 1.006},
            4: {"instance_type": "g5.12xlarge", "price_per_hour": 4.096},
            8: {"instance_type": "g5.48xlarge", "price_per_hour": 16.384},
        },
        "canonical_gpu_name": "A10G",
        "gpu_mem_gb": 24,
        "gpu_tflops_fp16": 125,
        "gpu_bandwidth_gbps": 600,
        "gpu_generation": "Ampere",
        "interconnect": "PCIe",
    },
    "A100_40gb": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p4d",
        "pricing": {
            1: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs, but can use 1
            4: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs, but can use 4
            8: {"instance_type": "p4d.24xlarge", "price_per_hour": 32.77},  # 8 GPUs (40GB per GPU)
        },
        "canonical_gpu_name": "A100-40GB",
        "gpu_mem_gb": 40,
        "gpu_tflops_fp16": 312,
        "gpu_bandwidth_gbps": 1555,
        "gpu_generation": "Ampere",
        "interconnect": "NVLink",
    },
    "A100_80gb": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p4de",
        "pricing": {
            1: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs, but can use 1
            4: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs, but can use 4
            8: {"instance_type": "p4de.24xlarge", "price_per_hour": 40.96},  # 8 GPUs (80GB per GPU)
        },
        "canonical_gpu_name": "A100-80GB",
        "gpu_mem_gb": 80,
        "gpu_tflops_fp16": 312,
        "gpu_bandwidth_gbps": 2039,
        "gpu_generation": "Ampere",
        "interconnect": "NVLink",
    },
    "H100": {
        "available_gpus": [1, 4, 8],
        "instance_family": "p5",
        "pricing": {
            1: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs, but can use 1
            4: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs, but can use 4
            8: {"instance_type": "p5.48xlarge", "price_per_hour": 98.32},  # 8 GPUs
        },
        "canonical_gpu_name": "H100",
        "gpu_mem_gb": 80,
        "gpu_tflops_fp16": 989,
        "gpu_bandwidth_gbps": 3350,
        "gpu_generation": "Hopper",
        "interconnect": "NVLink",
    },
}

# Default GPU type (for backward compatibility)
DEFAULT_GPU_TYPE = "L40S"

# S3 model storage defaults (used with --s3-models flag)
DEFAULT_S3_BUCKET = "tandemn-model-shards"
DEFAULT_S3_PREFIX = "hf-models"

# VM selection strategies for mapping (TP, PP) -> (gpus_per_node, num_nodes)
# - prefer_single_node: Strategy 1 then fallback to Strategy 2 (current/default behavior)
# - fit_tp_then_scale: Always use Strategy 2 (fit TP within a node, then scale nodes)
VM_SELECTION_STRATEGIES = {
    "prefer_single_node": "Prefer single-node when TP×PP fits; otherwise fit TP per node and scale.",
    "fit_tp_then_scale": "Always fit TP per node (smallest instance >= TP) and scale nodes.",
}

# Global logger instance
logger = logging.getLogger("benchmark")

def setup_logger(log_file_path: Path):
    """Set up logger to write to both console and file."""
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler - prints to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)

    # File handler - writes to log file
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Track active cluster for cleanup on unexpected exit
# Uses PID-based files so parallel processes don't stomp each other
_active_cluster = None
_active_cluster_dir = Path("/tmp/.benchmark_clusters")
_active_cluster_dir.mkdir(exist_ok=True)
_active_cluster_file = _active_cluster_dir / f"cluster_pid{os.getpid()}"

def set_active_cluster(cluster_name):
    """Set active cluster and persist to PID-specific file for crash recovery."""
    global _active_cluster
    _active_cluster = cluster_name
    if cluster_name:
        _active_cluster_file.write_text(cluster_name)
    elif _active_cluster_file.exists():
        _active_cluster_file.unlink()

def get_active_cluster():
    """Get active cluster from memory or file."""
    global _active_cluster
    if _active_cluster:
        return _active_cluster
    if _active_cluster_file.exists():
        return _active_cluster_file.read_text().strip()
    return None

def cleanup_on_exit():
    """Cleanup handler for unexpected exits."""
    cluster = get_active_cluster()
    if cluster:
        print(f"\n⚠️  Cleanup triggered. Tearing down cluster: {cluster}")
        try:
            subprocess.run(["sky", "down", "-y", cluster], timeout=300)
            print(f"✅ Cluster {cluster} terminated.")
        except Exception as e:
            print(f"❌ Failed to terminate cluster {cluster}: {e}")
            print(f"   Run manually: sky down -y {cluster}")
        finally:
            if _active_cluster_file.exists():
                _active_cluster_file.unlink()

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals."""
    print(f"\n🛑 Received signal {signum}. Cleaning up...")
    cleanup_on_exit()
    sys.exit(1)

def check_orphaned_cluster():
    """Check for orphaned clusters from crashed runs (dead PIDs only).

    Only cleans up clusters whose owning process is no longer running.
    Never touches clusters owned by live processes (safe for parallel runs).
    """
    if not _active_cluster_dir.exists():
        return
    for f in _active_cluster_dir.glob("cluster_pid*"):
        try:
            pid = int(f.name.split("pid")[1])
            # Check if the owning process is still alive
            os.kill(pid, 0)  # signal 0 = check existence, doesn't kill
            # Process is alive — skip this cluster (it's being used)
            cluster = f.read_text().strip()
            print(f"ℹ️  Cluster {cluster} is owned by active process {pid}, skipping")
        except ProcessLookupError:
            # Process is dead — this is truly orphaned
            cluster = f.read_text().strip()
            print(f"⚠️  Found orphaned cluster {cluster} (owner pid {pid} is dead)")
            print(f"   Terminating...")
            try:
                subprocess.run(["sky", "down", "-y", cluster], timeout=300)
                print(f"   ✅ Terminated {cluster}")
            except Exception as e:
                print(f"   ❌ Failed: {e}. Run: sky down -y {cluster}")
            f.unlink()
        except (ValueError, PermissionError):
            # Bad file or permission issue — clean up
            f.unlink()

# Register cleanup handlers
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_cluster_config(tp, pp, gpu_type=DEFAULT_GPU_TYPE, vm_strategy="prefer_single_node"):
    """
    Calculate the best instance configuration for given TP/PP and GPU type.

    Returns (gpus_per_node, num_nodes) that can accommodate TP×PP GPUs.

    Strategy:
    1. Prefer single-node configurations when possible (better for PP communication)
    2. Otherwise, use smallest instance that fits TP GPUs per node

    Examples:
        TP=4, PP=2 → 8 GPUs/node × 1 node = 8 GPUs (single node preferred, avoids inter-node latency)
        TP=2, PP=4 → 8 GPUs/node × 1 node = 8 GPUs (single node preferred if available)
        TP=8, PP=1 → 8 GPUs/node × 1 node = 8 GPUs
        TP=4, PP=4 → 4 GPUs/node × 4 nodes = 16 GPUs (no single instance fits 16 GPUs, falls back to multi-node)
    """
    if pp < 1:
        raise ValueError(f"Invalid PP={pp}. Must be >= 1")
    if tp < 1:
        raise ValueError(f"Invalid TP={tp}. Must be >= 1")
    if gpu_type not in GPU_CONFIGS:
        raise ValueError(f"Invalid GPU type: {gpu_type}. Must be one of {list(GPU_CONFIGS.keys())}")

    if vm_strategy not in VM_SELECTION_STRATEGIES:
        raise ValueError(f"Invalid vm_strategy: {vm_strategy}. Must be one of {list(VM_SELECTION_STRATEGIES.keys())}")

    total_gpus = tp * pp
    available_gpus = GPU_CONFIGS[gpu_type]["available_gpus"]

    # Strategy 1: Prefer single-node configuration if possible (better for PP communication)
    # Only used when vm_strategy == "prefer_single_node".
    if vm_strategy == "prefer_single_node":
        # Check if there's an instance type that can fit all GPUs in one node
        single_node_gpus = None
        for gpu_count in available_gpus:
            if gpu_count >= total_gpus:
                single_node_gpus = gpu_count
                break

        if single_node_gpus is not None:
            # Single node is possible and preferred (avoids inter-node network latency)
            return single_node_gpus, 1

    # Strategy 2: Multi-node fallback - find smallest instance that fits TP GPUs
    # This ensures TP communication stays within a node (fast)
    gpus_per_node = None
    for gpu_count in available_gpus:
        if gpu_count >= tp:
            gpus_per_node = gpu_count
            break

    if gpus_per_node is None:
        raise ValueError(f"TP={tp} exceeds max GPUs per instance ({max(available_gpus)}) for {gpu_type}")

    # Calculate how many nodes we need
    num_nodes = (total_gpus + gpus_per_node - 1) // gpus_per_node  # Ceiling division

    # Make sure we have enough total GPUs
    actual_gpus = gpus_per_node * num_nodes
    if actual_gpus < total_gpus:
        raise ValueError(f"Cannot fit TP={tp}, PP={pp} ({total_gpus} GPUs) into available {gpu_type} instances")

    return gpus_per_node, num_nodes


def load_experiments(csv_path):
    experiments = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            experiments.append({
                'tp': int(row['tensor_degree']),
                'pp': int(row['pipeline_degree']),
                'max_input_length': int(row['max_input_length']),
                'max_output_length': int(row['max_output_length']),
                'model': row['model']
            })
    return experiments

def group_by_cluster_then_io(experiments, gpu_type=DEFAULT_GPU_TYPE, vm_strategy="prefer_single_node"):
    """Group experiments: same cluster config runs multiple I/O shapes.
    
    Groups by (gpus_per_node, num_nodes, tp, pp, model) so that a single
    server instance can serve all I/O shapes for the same model/parallelism config.
    
    Returns: {(gpus_per_node, num_nodes, tp, pp, model): [experiments]}
    """
    groups = defaultdict(list)
    for exp in experiments:
        gpus_per_node, num_nodes = get_cluster_config(exp['tp'], exp['pp'], gpu_type, vm_strategy=vm_strategy)
        key = (gpus_per_node, num_nodes, exp['tp'], exp['pp'], exp['model'])
        groups[key].append(exp)
    return dict(groups)

def cleanup_old_benchmark_files(work_dir=None):
    """
    Remove old benchmark files that don't have GPU type in their names.
    These files are from the old naming scheme and are now obsolete.
    
    Old naming pattern: roofline-tp{TP}-pp{PP}[-{input}in-{output}out].yaml
    New naming pattern: roofline-tp{TP}-pp{PP}-{input}in-{output}out-{GPU_TYPE}.yaml
    
    This function identifies old files by checking if they don't end with any known GPU type.
    """
    if work_dir is None:
        work_dir = Path("roofline_benchmarks")
    
    if not work_dir.exists():
        print("ℹ️  roofline_benchmarks directory doesn't exist")
        return 0
    
    # List of known GPU types to check against
    known_gpu_types = list(GPU_CONFIGS.keys())
    gpu_type_suffixes = [gpu.replace("_", "-").replace("/", "-") for gpu in known_gpu_types]
    
    removed_count = 0
    removed_files = []
    
    # Check YAML files
    for file_path in work_dir.glob("roofline-*.yaml"):
        # Check if filename doesn't end with any GPU type suffix
        filename = file_path.stem  # without .yaml extension
        has_gpu_type = any(filename.endswith(f"-{suffix}") for suffix in gpu_type_suffixes)
        
        if not has_gpu_type:
            removed_files.append(file_path.name)
            file_path.unlink()
            removed_count += 1
    
    # Check Python benchmark scripts
    for file_path in work_dir.glob("benchmark_roofline-*.py"):
        # Check if filename doesn't end with any GPU type suffix
        filename = file_path.stem.replace("benchmark_roofline-", "")  # remove prefix
        has_gpu_type = any(filename.endswith(f"-{suffix}") for suffix in gpu_type_suffixes)
        
        if not has_gpu_type:
            removed_files.append(file_path.name)
            file_path.unlink()
            removed_count += 1
    
    if removed_count > 0:
        print(f"🗑️  Removed {removed_count} old benchmark files:")
        for fname in sorted(removed_files):
            print(f"   - {fname}")
        print(f"✅ Cleaned up {removed_count} old benchmark files")
    else:
        print("ℹ️  No old benchmark files to clean up")
    
    return removed_count

def generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type=DEFAULT_GPU_TYPE, s3_models=False, cloud="aws", use_spot=False):
    # Determine if this GPU type uses EFA-capable instances (A100/H100 on AWS)
    is_efa_capable = gpu_type.upper().startswith("A100") or gpu_type.upper() == "H100"

    env_exports = """
  # Ensure CUDA libraries are in LD_LIBRARY_PATH for PyTorch
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

  # NCCL settings for multi-GPU/multi-node
  export NCCL_P2P_DISABLE=0
  export NCCL_TIMEOUT=3600
  export TORCH_NCCL_BLOCKING_WAIT=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_DEBUG=INFO
"""
    if is_efa_capable and cloud == "aws":
        env_exports += """
  # EFA + NCCL OFI (auto-configured when using network_tier=best + DLAMI)
  export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"
  export NCCL_NET_PLUGIN=ofi
  export FI_EFA_USE_DEVICE_RDMA=1
  export FI_PROVIDER=efa
  export NCCL_DEBUG_SUBSYS=NET
"""

    # Check if any experiment needs Ray (PP > 1)
    needs_ray = any(exp['pp'] > 1 for exp in experiments)

    ray_run = f"""{env_exports}
  python roofline_benchmarks/benchmark_{cluster_name}.py
    """
    if num_nodes > 1:
        # Multi-node: use Sky's Ray cluster startup script
        ray_run = f"""{env_exports}
  # Start Ray cluster across all nodes
  export RAY_CMD="uv run ray"
  export RAY_DASHBOARD_HOST=0.0.0.0
  export RAY_HEAD_PORT=6379
  export RAY_DASHBOARD_PORT=8265
  export RAY_CGRAPH_get_timeout=1800
  export RAY_CGRAPH_submit_timeout=180
  ~/sky_templates/ray/start_cluster

  if [ "${{SKYPILOT_NODE_RANK}}" = "0" ]; then
      echo "Ray cluster started on head node"
      ray status --address="127.0.0.1:${{RAY_HEAD_PORT}}" || true

      # Run benchmark here while Ray is still up
      PYTHONHASHSEED=0 python roofline_benchmarks/benchmark_{cluster_name}.py
  fi
        """
    elif needs_ray:
        # Single-node but needs Ray for pipeline parallelism
        ray_run = f"""{env_exports}
  # Start Ray on single node for pipeline parallelism
  export RAY_DASHBOARD_HOST=0.0.0.0
  export RAY_HEAD_PORT=6379
  export RAY_DASHBOARD_PORT=8265

  echo "Starting Ray for single-node pipeline parallelism..."
  uv run ray start --head --port=${{RAY_HEAD_PORT}} --dashboard-host=${{RAY_DASHBOARD_HOST}} --dashboard-port=${{RAY_DASHBOARD_PORT}} --num-cpus=0

  sleep 5
  uv run ray status --address="127.0.0.1:${{RAY_HEAD_PORT}}"

  # Run benchmark
  PYTHONHASHSEED=0 python roofline_benchmarks/benchmark_{cluster_name}.py

  # Cleanup Ray (non-fatal - benchmark may have already cleaned up vLLM workers)
  # vLLM's Ray workers for pipeline parallelism may still be active, causing Ray to crash during shutdown.
  # This is a known issue: Ray workers holding GPU resources can't be cleanly shut down.
  # Since the benchmark completed successfully, we don't fail the job if Ray cleanup has issues.
  echo "Cleaning up Ray (this may show warnings if vLLM workers are still active)..."
  sleep 2  # Give vLLM workers a moment to finish cleanup
  uv run ray stop || echo "⚠️  Ray stop completed with warnings (this is OK - benchmark succeeded)"
        """
    is_a100 = gpu_type.upper().startswith("A100")

    # EFA-capable GPU types (A100, H100) on AWS: use instance_type + network_tier=best.
    # This makes SkyPilot auto-select the DLAMI (with EFA/NCCL pre-installed) and
    # attach EFA network interfaces. No custom image_id needed.
    # Non-EFA types (L4, L40S, A10G): use accelerators spec as before.
    EFA_INSTANCE_TYPES = {
        "A100_40gb": "p4d.24xlarge",
        "A100-40gb": "p4d.24xlarge",
        "A100_80gb": "p4de.24xlarge",
        "A100-80gb": "p4de.24xlarge",
        "H100": "p5.48xlarge",
    }

    accelerator_spec = ""
    network_tier_line = ""
    cloud_line = f"  infra: {cloud}\n"

    if gpu_type in EFA_INSTANCE_TYPES and cloud == "aws":
        instance_type = EFA_INSTANCE_TYPES[gpu_type]
        accelerator_spec = f"  instance_type: {instance_type}\n"
        network_tier_line = "  network_tier: best\n"
        # No image_id — SkyPilot auto-selects DLAMI + attaches EFA interfaces
    else:
        accelerator_spec = f"  accelerators: {gpu_type}:{gpus_per_node}\n"

    # Build file_mounts block for S3 model loading
    file_mounts_block = ""
    if s3_models:
        unique_models = list(dict.fromkeys(exp['model'] for exp in experiments))
        mounts = []
        for model_name in unique_models:
            mounts.append(f"  /models/{model_name}:\n    source: s3://{DEFAULT_S3_BUCKET}/{DEFAULT_S3_PREFIX}/{model_name}\n    mode: COPY")
        file_mounts_block = "\nfile_mounts:\n" + "\n".join(mounts) + "\n"

    # Scale disk for large models (MoE models like 235B need ~470GB for weights alone)
    model_names = [exp['model'] for exp in experiments]
    disk_size_gb = 500
    for mn in model_names:
        mn_lower = mn.lower()
        if "235b" in mn_lower or "236b" in mn_lower or "mixtral-8x22b" in mn_lower:
            disk_size_gb = 1000
            break

    return f"""
name: {cluster_name}
resources:
{cloud_line}{accelerator_spec}{network_tier_line}  use_spot: {"true" if use_spot else "false"}
  disk_size: {disk_size_gb}GB
  memory: "64GB+"
num_nodes: {num_nodes}
workdir: .{file_mounts_block}
setup: |
  export PYTHONHASHSEED=0
  set -euxo pipefail

  # GPU type for conditional setup (A100 requires special handling)
  GPU_TYPE="{gpu_type}"
  IS_A100={"true" if is_a100 else "false"}
  echo "GPU Type: $GPU_TYPE, Is A100: $IS_A100"

  # Diagnostics: Check NVIDIA driver and CUDA availability BEFORE any setup
  echo "=== Initial System Diagnostics ==="
  echo "Checking for NVIDIA driver..."
  nvidia-smi || echo "WARNING: nvidia-smi not found - drivers may need installation!"
  echo "Checking CUDA libraries..."
  ldconfig -p | grep cuda || echo "INFO: No CUDA libraries in ldconfig yet"
  echo "CUDA toolkit path:"
  ls -la /usr/local/cuda* 2>/dev/null || echo "INFO: No /usr/local/cuda* found yet"
  echo "==================================="

  # Check if nvidia-smi exists; if not, we might need to install drivers
  # However, on AWS GPU instances, drivers should already be installed via the AMI
  # SkyPilot should handle this automatically when accelerators are specified
  if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA drivers not found! This should not happen with GPU instances."
    echo "⚠️  SkyPilot should have installed drivers. Checking if this is p4de/p4d instance..."
    # On AWS p4d/p4de, drivers are usually pre-installed in the AMI
    # If they're missing, SkyPilot setup might have failed
  fi

  # ========================================================================
  # NVIDIA Fabric Manager - ONLY for NVSwitch systems (A100-SXM4, H100)
  # On DLAMI (auto-selected by network_tier=best), driver is pre-installed
  # but Fabric Manager may need to be installed/started.
  # ========================================================================
  if [ "$IS_A100" = "true" ]; then
  echo "=== Fabric Manager for NVSwitch ==="
  if systemctl is-active --quiet nvidia-fabricmanager 2>/dev/null; then
    echo "Fabric Manager already running"
  else
    DRIVER_VERSION_FULL=$(nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader)
    DRIVER_VERSION_MAJOR=$(echo "$DRIVER_VERSION_FULL" | cut -d. -f1)
    echo "Driver: $DRIVER_VERSION_FULL, installing Fabric Manager..."
    sudo apt-get update -qq
    sudo apt-get install -y "nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}}=${{DRIVER_VERSION_FULL}}-1" 2>/dev/null \
      || sudo apt-get install -y "nvidia-fabricmanager-${{DRIVER_VERSION_MAJOR}}" 2>/dev/null
    sudo systemctl enable --now nvidia-fabricmanager
    sleep 3
    systemctl is-active --quiet nvidia-fabricmanager \
      && echo "Fabric Manager: OK" \
      || echo "WARNING: Fabric Manager failed to start"
  fi
  nvidia-smi topo -m 2>&1 | awk 'NR<=15' || true
  fi  # End of A100/NVSwitch Fabric Manager section

  # Install numactl for NUMA diagnostics
  sudo apt-get install -y numactl 2>/dev/null || echo "numactl installation skipped"

  if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"

  uv venv --python 3.12 --seed --allow-existing
  source .venv/bin/activate

  # Install dependencies
  uv pip install "datasets" "requests" "pynvml" "aiohttp"

  # vLLM 0.10.0 for all GPU types
  echo "=== Installing vLLM 0.10.0 ==="
  uv pip install "vllm==0.10.0" "ray[cgraph]"

  # Pin transformers to avoid breaking changes in 5.x
  # (vllm 0.10.0 allows transformers>=5 but it breaks tokenizer backend)
  echo "=== Pinning transformers==4.57.3 (avoid 5.x breaking changes) ==="
  uv pip install "transformers==4.57.3"

  # Verify PyTorch has CUDA support
  echo "=== PyTorch CUDA Check ==="
  python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('torch.version.cuda:', torch.version.cuda); print('torch.__version__:', torch.__version__); print('torch.cuda.device_count():', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')" || echo "PyTorch CUDA check failed!"
  echo "==========================="
  
  echo "=== Setup complete ==="

run: |
  set -euxo pipefail

  # CRITICAL: Set LD_LIBRARY_PATH for CUDA libraries FIRST
  # This must be done before any Python imports that use CUDA
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${{LD_LIBRARY_PATH:-}}"
  export CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda}}"

  # Diagnostics before running benchmark
  echo "=== Pre-run Diagnostics ==="
  nvidia-smi || echo "WARNING: nvidia-smi not available in run phase!"

  # A100-SXM4 specific diagnostics - check for Fabric Manager
  echo "--- A100 SXM4 Diagnostics ---"
  echo "Checking NVIDIA Fabric Manager (required for NVSwitch on A100-SXM4)..."
  systemctl status nvidia-fabricmanager 2>&1 || echo "Fabric Manager status check failed or not present"

  echo "Checking for NVLink topology..."
  nvidia-smi topo -m 2>&1 || echo "NVLink topology check failed"

  echo "--- NUMA Topology and GPU-NUMA Mapping ---"
  echo "NUMA hardware layout:"
  numactl --hardware 2>&1 || echo "numactl not available"
  echo ""
  echo "GPU PCIe Bus IDs and NUMA node assignments:"
  gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "0")
  if [ "$gpu_count" -gt 0 ] 2>/dev/null; then
    for gpu in $(seq 0 $((gpu_count - 1))); do
      bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $gpu 2>/dev/null)
      if [ -n "$bus" ]; then
        # Convert to lowercase and proper sysfs path format (0000:XX:YY.Z)
        bus_lower=$(echo "$bus" | tr '[:upper:]' '[:lower:]')
        numa_node=$(cat /sys/bus/pci/devices/$bus_lower/numa_node 2>/dev/null || echo "N/A")
        local_cpus=$(cat /sys/bus/pci/devices/$bus_lower/local_cpulist 2>/dev/null || echo "N/A")
        echo "GPU$gpu: PCIe=$bus, NUMA_node=$numa_node, local_cpus=$local_cpus"
      else
        echo "GPU$gpu: Could not get PCIe bus ID"
      fi
    done
  else
    echo "Could not detect GPU count"
  fi
  echo "-------------------------------------------"

  echo "Checking for GPU processes and errors..."
  nvidia-smi -q -d ERRORS 2>&1 | awk 'NR<=50' || echo "Error check failed"

  echo "--- CUDA Device Query ---"
  # Try direct CUDA device query through Python (single line to avoid YAML issues)
  python3 -c "import ctypes; cudart = ctypes.CDLL('libcudart.so.12'); count = ctypes.c_int(); result = cudart.cudaGetDeviceCount(ctypes.byref(count)); print(f'cudaGetDeviceCount result: {{result}} (0=success), device count: {{count.value}}')" 2>&1 || echo "Direct CUDA query script failed"
  echo "==========================="

  source .venv/bin/activate

  # Critical: Detailed PyTorch CUDA diagnostics before running benchmark
  echo "=== DETAILED PyTorch CUDA Diagnostics (in run phase) ==="
  echo "--- Environment ---"
  echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" || true
  echo "CUDA_HOME: $CUDA_HOME" || true
  which python3
  echo "--- Torch info ---"
  python3 -c "import torch; print('torch.__version__:', torch.__version__); print('torch.__file__:', torch.__file__); print('torch.version.cuda:', torch.version.cuda); print('torch.backends.cuda.is_built():', torch.backends.cuda.is_built())"
  echo "--- CUDA check ---"
  python3 -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); print('torch.cuda.device_count():', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A - checking why...')"
  echo "--- Loading CUDA libs directly ---"
  python3 -c "import ctypes; libs=['libcuda.so.1','libcudart.so.12','libnvidia-ml.so.1']; [print(l+': OK') if ctypes.CDLL(l) else None for l in libs]" 2>&1 || echo "Some libs failed"
  echo "--- Manual CUDA init attempt ---"
  python3 -c "import torch; torch.cuda.init(); print('torch.cuda.init() OK, device_count:', torch.cuda.device_count())" 2>&1 || echo "CUDA init failed"
  echo "--- Installed torch packages ---"
  pip list | grep -i -E "torch|nvidia|cuda|triton"
  echo "========================================="
{ray_run}

  echo "Cluster ready for benchmarking"
"""

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453154642706960485/iFXIAaDTLxNO7_GHKHhXnXwFFnXziniP4TUwLUDUnXHtT9kNo08eQBjGQ4CiBr6AazY6"

def send_discord_message(message):
    try:
        payload = {"content": message}
        requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    except Exception:
        pass

def generate_benchmark_script(experiments, gpus_per_node, num_nodes, gpu_type=DEFAULT_GPU_TYPE, s3_models=False, cloud="aws"):
    """Generate Python benchmark script for a set of experiments.
    
    Generates a server-mode orchestrator that:
    1. Starts vLLM OpenAI-compatible server as subprocess
    2. Runs benchmark_client.py for each I/O shape
    3. Collects metrics from Prometheus /metrics endpoint
    """
    exp_list = json.dumps(experiments, indent=2)

    # Calculate cluster pricing info
    gpu_config = GPU_CONFIGS[gpu_type]
    price_per_node = gpu_config["pricing"][gpus_per_node]["price_per_hour"]
    total_price_per_hour = price_per_node * num_nodes
    instance_type = gpu_config["pricing"][gpus_per_node]["instance_type"]
    if num_nodes > 1:
        cluster_instance_type = f"{num_nodes}x {instance_type}"
    else:
        cluster_instance_type = instance_type
    
    # Model path expressions for the generated script
    if s3_models:
        model_path_line = 'model_path = f"/models/" + exp[\'model\']'
        global_model_path_line = 'MODEL_PATH = f"/models/" + EXPERIMENTS[0][\'model\']'
    else:
        model_path_line = "model_path = exp['model']"
        global_model_path_line = "MODEL_PATH = EXPERIMENTS[0]['model']"

    # GPU hardware specs for canonical columns
    canonical_gpu_name = gpu_config.get("canonical_gpu_name", gpu_type)
    gpu_mem_gb = gpu_config["gpu_mem_gb"]
    gpu_tflops_fp16 = gpu_config["gpu_tflops_fp16"]
    gpu_bandwidth_gbps = gpu_config["gpu_bandwidth_gbps"]
    gpu_generation = gpu_config["gpu_generation"]
    interconnect = gpu_config["interconnect"]

    return f'''#!/usr/bin/env python3
"""Server-mode benchmark orchestrator.

Starts a vLLM OpenAI-compatible server, then runs benchmark_client.py
for each I/O shape. Metrics come from Prometheus /metrics endpoint.
"""
import os
os.environ["RAY_ADDRESS"] = "127.0.0.1:6379"

import json
import time
import gc
import re
import sys
import subprocess
import threading
import urllib.request
from collections import defaultdict
from pathlib import Path

import ray
import torch
import requests as req_lib

# Prometheus parser (copied to same directory as this script)
from prometheus_parser import (
    parse_prometheus_text,
    compute_deltas,
    histogram_quantile,
    extract_throughput_metrics,
    extract_latency_percentiles,
)

# ============================================================================
# Distributed GPU Monitoring with Ray Actors
# ============================================================================

@ray.remote
class GPUMonitorActor:
    """Ray actor for GPU monitoring on a single node."""

    def __init__(self, node_id: str, sample_interval: float = 0.5):
        self.node_id = node_id
        self.sample_interval = sample_interval
        self.timeseries = []
        self._start_time = None
        self._stop_event = None
        self._pynvml_available = False
        self._pynvml = None
        self._device_count = 0

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
            self._device_count = pynvml.nvmlDeviceGetCount()
            print(f"[{{self.node_id}}] GPU monitor initialized with {{self._device_count}} GPUs")
        except Exception as e:
            print(f"[{{self.node_id}}] pynvml not available: {{e}}")
            import traceback
            traceback.print_exc()
            self._device_count = 0

    def start(self):
        """Start collecting samples."""
        if not self._pynvml_available:
            print(f"[{{self.node_id}}] Cannot start monitoring: pynvml not available")
            return
        try:
            self.timeseries = []
            self._start_time = time.time()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            print(f"[{{self.node_id}}] Monitoring thread started")
        except Exception as e:
            print(f"[{{self.node_id}}] Failed to start monitoring thread: {{e}}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop collecting samples."""
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        if hasattr(self, '_thread') and self._thread is not None:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        pynvml = self._pynvml
        while not self._stop_event.is_set():
            timestamp = time.time()
            relative_time = timestamp - self._start_time
            sample = {{'t': round(relative_time, 3)}}

            for i in range(self._device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_util_pct = (mem_info.used / mem_info.total) * 100

                    # GPU utilization (SM utilization)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                    mem_bw_util_pct = util.memory

                    # Use node-prefixed keys
                    sample[f'{{self.node_id}}_gpu{{i}}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'{{self.node_id}}_gpu{{i}}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'{{self.node_id}}_gpu{{i}}_sm_pct'] = gpu_util_pct
                    sample[f'{{self.node_id}}_gpu{{i}}_membw_pct'] = mem_bw_util_pct
                except Exception:
                    pass

            if len(sample) > 1:
                self.timeseries.append(sample)

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return collected time-series data."""
        return self.timeseries

    def get_node_id(self):
        """Return this actor's node ID."""
        return self.node_id
    
    def get_ray_node_id(self):
        """Return the Ray node ID where this actor is running."""
        try:
            import ray
            node_id = ray.get_runtime_context().get_node_id()
            return node_id
        except:
            return None

    def get_device_count(self):
        """Return number of GPUs on this node."""
        return self._device_count
    
    def health_check(self):
        """Simple health check to verify actor is responsive."""
        return {{"status": "ok", "node_id": self.node_id, "gpus": self._device_count}}


class DistributedGPUMonitor:
    """Manage GPU monitor actors across all Ray nodes."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.actors = []
        self.node_map = {{}}

    def start(self):
        """Deploy actors on all Ray nodes and start monitoring."""
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n.get('Alive', False)]

        print(f"📡 Found {{len(alive_nodes)}} Ray nodes for GPU monitoring")

        try:
            from ray.util.placement_group import placement_group, remove_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            
            bundles = [{{"CPU": 0.1}} for _ in alive_nodes]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready(), timeout=60.0)
            print(f"   Created placement group with {{len(bundles)}} bundles")
            
            for idx, node in enumerate(alive_nodes):
                node_id = f"node{{idx}}"
                node_ip = node.get('NodeManagerAddress', 'unknown')
                
                try:
                    actor = GPUMonitorActor.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=idx
                        ),
                        num_cpus=0.1,
                    ).remote(node_id, self.sample_interval)
                    
                    self.actors.append(actor)
                    self.node_map[node_id] = actor
                    print(f"  ✓ Deployed monitor on {{node_id}} ({{node_ip}}) using placement group")
                except Exception as e:
                    print(f"  ✗ Failed to deploy on {{node_id}}: {{e}}")
                    import traceback
                    traceback.print_exc()
        except Exception as pg_error:
            print(f"   Placement group approach failed: {{pg_error}}")
            print(f"   Falling back to simple actor creation (no pinning)...")
            
            for idx, node in enumerate(alive_nodes):
                node_id = f"node{{idx}}"
                node_ip = node.get('NodeManagerAddress', 'unknown')
                
                try:
                    actor = GPUMonitorActor.options(
                        num_cpus=0.1,
                    ).remote(node_id, self.sample_interval)
                    
                    self.actors.append(actor)
                    self.node_map[node_id] = actor
                    print(f"  ✓ Deployed monitor actor (will query node location)")
                    
                    try:
                        ray_node_id = ray.get(actor.get_ray_node_id.remote(), timeout=15.0)
                        print(f"     Actor running on Ray node: {{ray_node_id}}")
                    except Exception as node_check_err:
                        print(f"     Could not determine actor node: {{node_check_err}}")
                except Exception as e:
                    print(f"  ✗ Failed to deploy on {{node_id}}: {{e}}")
                    import traceback
                    traceback.print_exc()

        print(f"   Starting monitoring on {{len(self.actors)}} nodes...")
        started_count = 0
        for idx, actor in enumerate(self.actors):
            try:
                print(f"   Starting monitor on node{{idx}}...")
                future = actor.start.remote()
                ray.get(future, timeout=30.0)
                started_count += 1
                print(f"   ✓ Started monitor on node{{idx}}")
            except Exception as e:
                print(f"   ✗ Failed to start monitor on node{{idx}}: {{e}}")
                import traceback
                traceback.print_exc()
        
        if started_count > 0:
            print(f"📊 GPU monitoring started on {{started_count}}/{{len(self.actors)}} nodes")
            return True
        else:
            print(f"⚠️  Warning: No actors started successfully. Falling back to local monitoring.")
            self.actors = []
            self.node_map = {{}}
            return False

    def stop(self):
        """Stop all monitor actors."""
        if self.actors:
            ray.get([actor.stop.remote() for actor in self.actors])

    def get_timeseries(self):
        """Collect and merge time-series from all nodes."""
        if not self.actors:
            return []

        all_series = ray.get([actor.get_timeseries.remote() for actor in self.actors])

        merged = {{}}
        for series in all_series:
            for sample in series:
                t = sample['t']
                if t not in merged:
                    merged[t] = {{'t': t}}
                for key, value in sample.items():
                    if key != 't':
                        merged[t][key] = value

        return [merged[t] for t in sorted(merged.keys())]

    def get_summary(self):
        """Return summary statistics across all nodes."""
        timeseries = self.get_timeseries()
        if not timeseries:
            return {{}}

        summary = {{}}
        metrics = defaultdict(list)

        for sample in timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        for key, values in metrics.items():
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = round(max(values), 2)
                summary[f'{{key}}_min'] = round(min(values), 2)

        all_sm_util = []
        all_mem_bw_util = []
        all_mem_util = []
        for key, values in metrics.items():
            if '_sm_pct' in key:
                all_sm_util.extend(values)
            elif '_membw_pct' in key:
                all_mem_bw_util.extend(values)
            elif '_mem_pct' in key:
                all_mem_util.extend(values)

        if all_sm_util:
            summary['avg_sm_util_pct'] = round(sum(all_sm_util) / len(all_sm_util), 2)
            summary['max_sm_util_pct'] = round(max(all_sm_util), 2)
        if all_mem_bw_util:
            summary['avg_mem_bw_util_pct'] = round(sum(all_mem_bw_util) / len(all_mem_bw_util), 2)
            summary['max_mem_bw_util_pct'] = round(max(all_mem_bw_util), 2)
        if all_mem_util:
            summary['avg_mem_util_pct'] = round(sum(all_mem_util) / len(all_mem_util), 2)
            summary['max_mem_util_pct'] = round(max(all_mem_util), 2)

        summary['gpu_samples'] = len(timeseries)
        summary['num_nodes_monitored'] = len(self.actors)
        return summary


# ============================================================================
# Local GPU Monitoring (fallback for single-node)
# ============================================================================

class GPUMonitor:
    """Background GPU metrics collector using pynvml."""

    def __init__(self, sample_interval=0.5):
        self.sample_interval = sample_interval

        self.timeseries = []
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._pynvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._pynvml = pynvml
        except Exception as e:
            print(f"⚠️  pynvml not available, GPU monitoring disabled: {{e}}")

    def start(self):
        if not self._pynvml_available:
            return
        self._stop_event.clear()
        self.timeseries = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _monitor_loop(self):
        pynvml = self._pynvml
        device_count = pynvml.nvmlDeviceGetCount()

        while not self._stop_event.is_set():
            timestamp = time.time()
            relative_time = timestamp - self._start_time
            sample = {{'t': round(relative_time, 3)}}

            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                    mem_util_pct = (mem_info.used / mem_info.total) * 100

                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = util.gpu
                    mem_bw_util_pct = util.memory

                    sample[f'gpu{{i}}_mem_gb'] = round(mem_used_gb, 2)
                    sample[f'gpu{{i}}_mem_pct'] = round(mem_util_pct, 1)
                    sample[f'gpu{{i}}_sm_pct'] = gpu_util_pct
                    sample[f'gpu{{i}}_membw_pct'] = mem_bw_util_pct

                except Exception:
                    pass

            if len(sample) > 1:
                self.timeseries.append(sample)

            self._stop_event.wait(self.sample_interval)

    def get_timeseries(self):
        """Return raw time-series data for plotting."""
        return self.timeseries

    def get_summary(self):
        """Return summary statistics of GPU metrics."""
        if not self.timeseries:
            return {{}}

        summary = {{}}

        metrics = defaultdict(list)
        for sample in self.timeseries:
            for key, value in sample.items():
                if key != 't':
                    metrics[key].append(value)

        for key, values in metrics.items():
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = round(max(values), 2)
                summary[f'{{key}}_min'] = round(min(values), 2)

        all_sm_util = []
        all_mem_bw_util = []
        all_mem_util = []
        for key, values in metrics.items():
            if '_sm_pct' in key:
                all_sm_util.extend(values)
            elif '_membw_pct' in key:
                all_mem_bw_util.extend(values)
            elif '_mem_pct' in key:
                all_mem_util.extend(values)

        if all_sm_util:
            summary['avg_sm_util_pct'] = round(sum(all_sm_util) / len(all_sm_util), 2)
            summary['max_sm_util_pct'] = round(max(all_sm_util), 2)
        if all_mem_bw_util:
            summary['avg_mem_bw_util_pct'] = round(sum(all_mem_bw_util) / len(all_mem_bw_util), 2)
            summary['max_mem_bw_util_pct'] = round(max(all_mem_bw_util), 2)
        if all_mem_util:
            summary['avg_mem_util_pct'] = round(sum(all_mem_util) / len(all_mem_util), 2)
            summary['max_mem_util_pct'] = round(max(all_mem_util), 2)

        summary['gpu_samples'] = len(self.timeseries)
        return summary


# ============================================================================
def _find_gauge(gauges, base_name, default=0):
    """Find gauge value by base name, ignoring label suffixes."""
    if base_name in gauges:
        return gauges[base_name]
    for key, val in gauges.items():
        stripped = key.split("{{")[0] if "{{" in key else key
        if stripped == base_name:
            return val
    return default

# MetricsPoller — replaces SchedulerMonitor
# ============================================================================

class MetricsPoller:
    """Poll /metrics endpoint periodically for timeseries data."""

    def __init__(self, base_url="http://localhost:8000", interval=0.5):
        self.base_url = base_url
        self.interval = interval
        self.timeseries = []
        self._stop = threading.Event()
        self._thread = None
        self._start_time = None

    def start(self):
        self._stop.clear()
        self.timeseries = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                resp = urllib.request.urlopen(f"{{self.base_url}}/metrics", timeout=5)
                text = resp.read().decode()
                parsed = parse_prometheus_text(text)
                sample = {{
                    "t": round(time.time() - self._start_time, 3),
                    "running": _find_gauge(parsed["gauges"], "vllm:num_requests_running", 0),
                    "waiting": _find_gauge(parsed["gauges"], "vllm:num_requests_waiting", 0),
                    "swapped": _find_gauge(parsed["gauges"], "vllm:num_requests_swapped", 0),
                    "kv_cache_util_pct": round(_find_gauge(parsed["gauges"], "vllm:gpu_cache_usage_perc", 0) * 100, 1),
                }}
                self.timeseries.append(sample)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)

    def get_timeseries(self):
        return self.timeseries

    def get_summary(self):
        if not self.timeseries:
            return {{}}
        # Filter to only active samples (where running > 0) to avoid
        # diluting averages with idle periods before/after the benchmark.
        active = [s for s in self.timeseries if s.get('running', 0) > 0]
        source = active if active else self.timeseries
        summary = {{}}
        all_keys = set()
        for sample in source:
            all_keys.update(sample.keys())
        all_keys.discard('t')
        for key in all_keys:
            values = [s.get(key, 0) for s in source if key in s]
            if values:
                summary[f'{{key}}_avg'] = round(sum(values) / len(values), 2)
                summary[f'{{key}}_max'] = max(values)
        summary['scheduler_samples'] = len(self.timeseries)
        summary['scheduler_active_samples'] = len(active)
        return summary


# ============================================================================
# Constants
# ============================================================================

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1453154642706960485/iFXIAaDTLxNO7_GHKHhXnXwFFnXziniP4TUwLUDUnXHtT9kNo08eQBjGQ4CiBr6AazY6"

EXPERIMENTS = {exp_list}
RESULTS_FILE = "/tmp/benchmark_results.json"
MIN_NUM_SAMPLES = 50
SERVER_PORT = 8000
SERVER_URL = f"http://localhost:{{SERVER_PORT}}"
MAX_MODEL_LEN = max(e['max_input_length'] + e['max_output_length'] for e in EXPERIMENTS)

# Cluster pricing information
INSTANCE_TYPE = "{cluster_instance_type}"
PRICE_PER_HOUR = {total_price_per_hour}
NUM_NODES = {num_nodes}
GPUS_PER_NODE = {gpus_per_node}

# GPU hardware specs (from GPU_CONFIGS)
GPU_MODEL = "{canonical_gpu_name}"
GPU_MEM_GB = {gpu_mem_gb}
GPU_TFLOPS_FP16 = {gpu_tflops_fp16}
GPU_BANDWIDTH_GBPS = {gpu_bandwidth_gbps}
GPU_GENERATION = "{gpu_generation}"
INTERCONNECT = "{interconnect}"
CLOUD = "{cloud}"

# Will be set after parsing server logs
NUM_GPU_BLOCKS = 0
BLOCK_SIZE = 16

# Detect vllm version at runtime
try:
    import vllm as _vllm_mod
    RUNTIME_STACK = f"vllm {{_vllm_mod.__version__}}"
except Exception:
    RUNTIME_STACK = "vllm"


def send_discord_message(message):
    try:
        payload = {{"content": message}}
        req_lib.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    except Exception:
        pass


def get_model_config_info(model_name_or_path):
    """Load HuggingFace model config and extract architecture info."""
    result = {{
        'model_architecture': None,
        'params_billion': None,
        'model_config_json': None,
        'vocab_size': None,
        'num_attention_heads': None,
        'num_key_value_heads': None,
        'is_moe': None,
        'num_experts_active': None,
    }}
    try:
        import os as _os
        from transformers import AutoConfig

        config = None
        for path_candidate in [model_name_or_path]:
            try:
                config = AutoConfig.from_pretrained(path_candidate, trust_remote_code=True)
                break
            except Exception:
                continue

        if config is None:
            print(f"⚠️  Could not load model config for {{model_name_or_path}}")
            return result

        config_dict = config.to_dict()
        result['model_config_json'] = json.dumps(config_dict, default=str)

        if hasattr(config, 'architectures') and config.architectures:
            result['model_architecture'] = config.architectures[0]

        hidden = getattr(config, 'hidden_size', None)
        layers = getattr(config, 'num_hidden_layers', None)
        vocab = getattr(config, 'vocab_size', None)
        intermediate = getattr(config, 'intermediate_size', None)
        num_heads = getattr(config, 'num_attention_heads', None)
        num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        result['vocab_size'] = vocab
        result['num_attention_heads'] = num_heads
        result['num_key_value_heads'] = num_kv_heads

        num_experts = getattr(config, 'num_local_experts', None) or getattr(config, 'num_experts', None)
        num_experts_active = getattr(config, 'num_experts_per_tok', None) or getattr(config, 'num_selected_experts', None)
        result['is_moe'] = num_experts is not None and num_experts > 1
        result['num_experts_active'] = num_experts_active

        params_b = None
        if _os.path.isdir(model_name_or_path):
            index_file = _os.path.join(model_name_or_path, 'model.safetensors.index.json')
            if _os.path.exists(index_file):
                try:
                    with open(index_file) as _f:
                        idx = json.load(_f)
                    total_bytes = idx.get('metadata', {{}}).get('total_size')
                    if total_bytes:
                        params_b = round(int(total_bytes) / 2 / 1e9, 2)
                except Exception:
                    pass

        if params_b is None and all(v is not None for v in [hidden, layers, vocab, intermediate]):
            kv_dim = (hidden // num_heads * num_kv_heads) if (num_heads and num_kv_heads) else hidden
            embed_params = vocab * hidden * 2
            attn_params = layers * (hidden * hidden + 2 * hidden * kv_dim + hidden * hidden)
            ffn_params = layers * 3 * hidden * intermediate
            if num_experts and num_experts > 1:
                ffn_params = ffn_params * num_experts
            total_params = embed_params + attn_params + ffn_params
            params_b = round(total_params / 1e9, 2)

        result['params_billion'] = params_b

    except Exception as e:
        print(f"⚠️  get_model_config_info failed: {{e}}")
        import traceback
        traceback.print_exc()

    return result


def compute_canonical_columns(exp, model_info, vllm_config, measured_data):
    """Build all 73 canonical schema columns from experiment data."""
    gpu_count = NUM_NODES * GPUS_PER_NODE
    params_b = model_info.get('params_billion')
    model_size_gb = round(params_b * 2, 2) if params_b else None
    tps_total = measured_data.get('tokens_per_sec_total', 0)
    tps_prefill = measured_data.get('tokens_per_sec_prefill', 0)
    tps_decode = measured_data.get('tokens_per_sec_decode', 0)
    elapsed = measured_data.get('elapsed_time', 0)
    price_per_hour = PRICE_PER_HOUR
    input_len = exp.get('max_input_length', 0)
    output_len = exp.get('max_output_length', 0)
    num_heads = model_info.get('num_attention_heads')
    num_kv_heads = model_info.get('num_key_value_heads')

    def _safe_div(a, b):
        if a is None or b is None or b == 0:
            return None
        return round(a / b, 4)

    # Detect precision from model config
    _precision = 'fp16'
    _mcj = model_info.get('model_config_json')
    if _mcj:
        try:
            _mc = json.loads(_mcj)
            _dtype = _mc.get('torch_dtype') or _mc.get('dtype') or ''
            if 'bfloat16' in str(_dtype).lower():
                _precision = 'bf16'
            elif 'float32' in str(_dtype).lower():
                _precision = 'fp32'
        except Exception:
            pass

    canonical = {{
        'data_source': 'our_experiment',
        'data_source_type': 'measured',
        'precision': _precision,
        'cloud': CLOUD,
        'region': None,
        'task_type': 'batched',
        'request_pattern': 'offline_batch',
        'is_lmcache': False,
        'is_continuous_batching': True,
        'kv_offload_target': None,
        'cuda_graphs': None,
        'spec_decode': None,

        'model_name': exp.get('model'),
        'tp': exp.get('tp'),
        'pp': exp.get('pp'),

        'instance_type': INSTANCE_TYPE,
        'price_per_instance_hour_usd': price_per_hour,
        'num_nodes': NUM_NODES,
        'gpus_per_node': GPUS_PER_NODE,
        'gpu_model': GPU_MODEL,
        'gpu_mem_gb': GPU_MEM_GB,

        'interconnect': INTERCONNECT,
        'gpu_bandwidth_gbps': GPU_BANDWIDTH_GBPS,
        'gpu_tflops_fp16': GPU_TFLOPS_FP16,
        'gpu_generation': GPU_GENERATION,

        'runtime_stack': RUNTIME_STACK,

        'model_architecture': model_info.get('model_architecture'),
        'params_billion': params_b,
        'model_config_json': model_info.get('model_config_json'),
        'vocab_size': model_info.get('vocab_size'),
        'is_moe': model_info.get('is_moe'),
        'num_experts_active': model_info.get('num_experts_active'),

        'tokens_per_sec_total': tps_total,
        'tokens_per_sec_prefill': tps_prefill,
        'tokens_per_sec_decode': tps_decode,
        'total_cost_usd': round(price_per_hour * elapsed / 3600, 6) if elapsed else None,

        'max_num_seqs': vllm_config.get('max_num_seqs') if vllm_config else None,
        'batch_size': vllm_config.get('max_num_seqs') if vllm_config else None,

        'gpu_count_total': gpu_count,
        'tokens_per_sec_per_gpu': _safe_div(tps_total, gpu_count),
        'input_len_tokens_min': input_len,
        'input_len_tokens_max': input_len,
        'input_len_tokens_avg': input_len,
        'input_len_tokens_fixed': input_len,
        'output_len_tokens_min': output_len,
        'output_len_tokens_max': output_len,
        'output_len_tokens_avg': output_len,
        'output_len_tokens_fixed': output_len,
        'prefill_decode_ratio': _safe_div(input_len, output_len),
        'num_requests': measured_data.get('num_requests', MIN_NUM_SAMPLES),
        'model_size_gb': model_size_gb,
        'params_per_gpu': _safe_div(params_b, gpu_count),
        'model_fits_single_gpu': (model_size_gb <= GPU_MEM_GB) if model_size_gb else None,
        'vram_headroom_gb': round(GPU_MEM_GB * gpu_count - model_size_gb, 2) if model_size_gb else None,
        'attention_heads_per_kv_head': _safe_div(num_heads, num_kv_heads),
        'bandwidth_per_param': _safe_div(GPU_BANDWIDTH_GBPS * exp.get('tp', 1), params_b),
        'flops_per_param': _safe_div(GPU_TFLOPS_FP16 * exp.get('tp', 1), params_b),
        'kv_heads_per_tp': _safe_div(num_kv_heads, exp.get('tp', 1)),
        'crosses_node_boundary': NUM_NODES > 1,
        'price_per_gpu_hour_usd': _safe_div(price_per_hour, gpu_count),
        'cost_per_1m_tokens_total_usd': round(price_per_hour * 1e6 / (tps_total * 3600), 4) if tps_total else None,
        'cost_per_1m_tokens_prefill_usd': round(price_per_hour * 1e6 / (tps_prefill * 3600), 4) if tps_prefill else None,
        'cost_per_1m_tokens_decode_usd': round(price_per_hour * 1e6 / (tps_decode * 3600), 4) if tps_decode else None,

        'dp': None,
        'ttft_ms_p50': measured_data.get('ttft_ms_p50'),
        'ttft_ms_p95': measured_data.get('ttft_ms_p95'),
        'ttft_ms_p99': measured_data.get('ttft_ms_p99'),
        'tpot_ms_p50': measured_data.get('tpot_ms_p50'),
        'tpot_ms_p95': measured_data.get('tpot_ms_p95'),
        'tpot_ms_p99': measured_data.get('tpot_ms_p99'),
        'e2e_ms_p50': measured_data.get('e2e_ms_p50'),
        'e2e_ms_p95': measured_data.get('e2e_ms_p95'),
        'e2e_ms_p99': measured_data.get('e2e_ms_p99'),
    }}
    return canonical


# ============================================================================
# Server Management
# ============================================================================

def scrape_metrics(base_url=None):
    """Scrape /metrics endpoint from vLLM server."""
    if base_url is None:
        base_url = SERVER_URL
    resp = urllib.request.urlopen(f"{{base_url}}/metrics", timeout=10)
    return resp.read().decode()


def start_vllm_server(model_path, tp, pp, max_model_len, log_path="/tmp/vllm_server.log"):
    """Start vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--pipeline-parallel-size", str(pp),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
        "--port", str(SERVER_PORT),
    ]
    if pp > 1:
        cmd += ["--distributed-executor-backend", "ray"]
    print(f"🚀 Starting vLLM server: {{' '.join(cmd)}}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def wait_for_health(server_proc, timeout=600):
    """Wait for server /health to return 200. Returns True on success."""
    print(f"⏳ Waiting for server health (timeout={{timeout}}s)...")
    for attempt in range(timeout * 2):  # Check every 0.5s
        # Check if server process died
        if server_proc.poll() is not None:
            print(f"❌ Server process died with exit code {{server_proc.returncode}}")
            return False
        try:
            resp = urllib.request.urlopen(f"{{SERVER_URL}}/health", timeout=2)
            if resp.status == 200:
                print(f"✅ Server healthy after {{attempt * 0.5:.1f}}s")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def parse_gpu_blocks(log_path="/tmp/vllm_server.log", timeout=30):
    """Parse server log for KV cache info. Returns (num_gpu_blocks, block_size).

    vLLM 0.10.0 logs:
      GPU KV cache size: 516,896 tokens
      Maximum concurrency for 10,239 tokens per request: 50.48x
    """
    num_gpu_blocks = None
    block_size = 16  # default

    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(log_path, "r") as f:
                content = f.read()
            # Try vLLM 0.10.0 format: "GPU KV cache size: 516,896 tokens"
            m = re.search(r'GPU KV cache size:\\s*([\\d,]+)\\s*tokens', content)
            if m:
                kv_cache_tokens = int(m.group(1).replace(',', ''))
                num_gpu_blocks = kv_cache_tokens // block_size
                print(f"📊 Parsed from server log: KV cache={{kv_cache_tokens}} tokens, num_gpu_blocks={{num_gpu_blocks}}, block_size={{block_size}}")
                return num_gpu_blocks, block_size
            # Fallback: older vLLM format "# GPU blocks: XXXX"
            m = re.search(r'# GPU blocks:\\s*(\\d+)', content)
            if m:
                num_gpu_blocks = int(m.group(1))
                bs_match = re.search(r'block_size[=:\\s]+(\\d+)', content)
                if bs_match:
                    block_size = int(bs_match.group(1))
                print(f"📊 Parsed from server log: num_gpu_blocks={{num_gpu_blocks}}, block_size={{block_size}}")
                return num_gpu_blocks, block_size
        except FileNotFoundError:
            pass
        time.sleep(1.0)
    print("⚠️  Could not parse GPU blocks from server log")
    return None, block_size


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_single_benchmark(exp, target_concurrency, server_proc):
    """Run benchmark for a single I/O shape against the running server."""
    global NUM_GPU_BLOCKS, BLOCK_SIZE

    input_len = exp['max_input_length']
    output_len = exp['max_output_length']
    {model_path_line}
    exp_id = f"tp{{exp['tp']}}_pp{{exp['pp']}}_in{{input_len}}_out{{output_len}}"

    print(f"\\n{{'='*70}}")
    print(f"Running: TP={{exp['tp']}}, PP={{exp['pp']}}, input={{input_len}}, output={{output_len}}")
    print(f"Target concurrency: {{target_concurrency}}")
    print("="*70)

    # Load model config info for canonical columns
    model_info = get_model_config_info(model_path)
    if model_info.get('model_architecture') is None:
        model_info = get_model_config_info(exp['model'])

    NUM_REQUESTS = max(MIN_NUM_SAMPLES, int(4 * target_concurrency))

    # GPU monitoring setup
    gpu_monitor = None
    use_distributed = False
    backend = "ray" if exp['pp'] > 1 else None

    if backend == "ray":
        try:
            if not ray.is_initialized():
                ray.init(
                    address=os.environ.get("RAY_ADDRESS", "127.0.0.1:6379"),
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n.get('Alive', False)]
            if len(alive_nodes) > 1:
                print(f"📡 Using distributed GPU monitoring ({{len(alive_nodes)}} nodes)")
                gpu_monitor = DistributedGPUMonitor(sample_interval=0.5)
                use_distributed = True
            else:
                print(f"📊 Using local GPU monitoring (single-node Ray cluster)")
                gpu_monitor = GPUMonitor(sample_interval=0.5)
        except Exception as e:
            print(f"⚠️  Ray GPU monitor failed: {{e}}")
            gpu_monitor = GPUMonitor(sample_interval=0.5)
    else:
        print("📊 Using local GPU monitoring (single node, no Ray)")
        gpu_monitor = GPUMonitor(sample_interval=0.5)

    try:
        # Check server is still alive
        if server_proc.poll() is not None:
            raise RuntimeError(f"Server died before benchmark (exit code {{server_proc.returncode}})")

        # 1. Warmup via benchmark_client.py
        print(f"🔥 Running warmup (5 requests)...")
        warmup_cmd = [
            sys.executable, "roofline_benchmarks/benchmark_client.py",
            "--base-url", SERVER_URL,
            "--model", MODEL_PATH,
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--num-requests", "5",
            "--target-concurrency", "2",
            "--warmup-requests", "0",
            "--output", "/tmp/warmup_results.json",
        ]
        subprocess.run(warmup_cmd, check=True)
        print("✅ Warmup complete")

        # 2. Scrape /metrics after warmup (small delay to let Prometheus counters settle)
        time.sleep(2)
        warmup_text = scrape_metrics()
        warmup_parsed = parse_prometheus_text(warmup_text)

        # 3. Start GPU monitoring
        if use_distributed:
            success = gpu_monitor.start()
            if not success or (hasattr(gpu_monitor, 'actors') and len(gpu_monitor.actors) == 0):
                print("⚠️  Distributed monitoring failed, falling back to local")
                gpu_monitor = GPUMonitor(sample_interval=0.5)
                use_distributed = False
                gpu_monitor.start()
        else:
            gpu_monitor.start()

        # 4. Start MetricsPoller (replaces SchedulerMonitor)
        metrics_poller = MetricsPoller(base_url=SERVER_URL, interval=0.5)
        metrics_poller.start()

        # 5. Run real benchmark
        print(f"📊 Running benchmark: {{NUM_REQUESTS}} requests, concurrency={{target_concurrency}}")
        client_output = f"/tmp/client_results_{{exp_id}}.json"
        bench_cmd = [
            sys.executable, "roofline_benchmarks/benchmark_client.py",
            "--base-url", SERVER_URL,
            "--model", MODEL_PATH,
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--num-requests", str(NUM_REQUESTS),
            "--target-concurrency", str(target_concurrency),
            "--warmup-requests", "0",
            "--output", client_output,
        ]
        bench_start = time.perf_counter()
        subprocess.run(bench_cmd, check=True)
        elapsed = time.perf_counter() - bench_start

        # 6. Scrape /metrics after benchmark
        final_text = scrape_metrics()
        final_parsed = parse_prometheus_text(final_text)

        # 7. Stop monitoring
        gpu_monitor.stop()
        metrics_poller.stop()

        gpu_metrics = gpu_monitor.get_summary()
        scheduler_metrics = metrics_poller.get_summary()

        # 8. Compute deltas
        deltas = compute_deltas(warmup_parsed, final_parsed)

        # 9. Load client results
        with open(client_output) as f:
            client_data = json.load(f)

        wall_clock = client_data.get("wall_clock_s", elapsed)

        # 10. Extract throughput from server-side Prometheus counters (NOT client-side)
        throughput = extract_throughput_metrics(deltas, wall_clock)
        latency_pcts = extract_latency_percentiles(deltas)

        # Log monitoring info
        if use_distributed and hasattr(gpu_monitor, 'actors'):
            print(f"📊 GPU monitoring: {{len(gpu_monitor.actors)}} nodes monitored")
        else:
            print(f"📊 GPU monitoring: local (single node)")

        # 11. Build vllm_config (from server startup info)
        vllm_config = {{
            'max_num_seqs': 256,  # vLLM default
            'max_model_len': MAX_MODEL_LEN,
            'block_size': BLOCK_SIZE,
            'num_gpu_blocks': NUM_GPU_BLOCKS,
        }}

        # 12. Build measured_data for compute_canonical_columns
        measured_data = {{
            'tokens_per_sec_total': throughput['tokens_per_sec_total'],
            'tokens_per_sec_prefill': throughput['tokens_per_sec_prefill'],
            'tokens_per_sec_decode': throughput['tokens_per_sec_decode'],
            'elapsed_time': wall_clock,
            'ttft_ms_p50': latency_pcts.get('ttft_ms_p50'),
            'ttft_ms_p95': latency_pcts.get('ttft_ms_p95'),
            'ttft_ms_p99': latency_pcts.get('ttft_ms_p99'),
            'tpot_ms_p50': latency_pcts.get('tpot_ms_p50'),
            'tpot_ms_p95': latency_pcts.get('tpot_ms_p95'),
            'tpot_ms_p99': latency_pcts.get('tpot_ms_p99'),
            'e2e_ms_p50': latency_pcts.get('e2e_ms_p50'),
            'e2e_ms_p95': latency_pcts.get('e2e_ms_p95'),
            'e2e_ms_p99': latency_pcts.get('e2e_ms_p99'),
            'num_requests': NUM_REQUESTS,
        }}
        canonical = compute_canonical_columns(exp, model_info, vllm_config, measured_data)

        # 13. Save timeseries
        timeseries_file = f"/tmp/timeseries_{{exp_id}}.json"
        timeseries_data = {{
            'exp_id': exp_id,
            'config': exp,
            'elapsed_time': wall_clock,
            'gpu_timeseries': gpu_monitor.get_timeseries(),
            'scheduler_timeseries': metrics_poller.get_timeseries(),
        }}
        with open(timeseries_file, 'w') as f:
            json.dump(timeseries_data, f)
        print(f"📈 Time-series saved to {{timeseries_file}}")

        # Check for preemptions (indicates KV cache thrashing — throughput numbers suspect)
        num_preemptions = throughput.get('num_preemptions', 0)
        if num_preemptions > 0:
            print(f"⚠️  WARNING: {{num_preemptions}} preemptions detected! KV cache was thrashing. Throughput may be degraded.")
            send_discord_message(f"⚠️ tp{{exp['tp']}}-pp{{exp['pp']}} in{{input_len}}-out{{output_len}}: {{num_preemptions}} preemptions!")
        else:
            send_discord_message(f"✅ Results for tp{{exp['tp']}}-pp{{exp['pp']}} in{{input_len}}-out{{output_len}}")

        # Build result
        effective_prompt_toks = throughput['total_prompt_tokens']
        effective_gen_toks = throughput['total_generation_tokens']
        total_tokens = effective_prompt_toks + effective_gen_toks
        elapsed_hours = wall_clock / 3600
        cost_for_run = PRICE_PER_HOUR * elapsed_hours
        tokens_per_dollar = round(total_tokens / cost_for_run, 2) if cost_for_run > 0 else 0
        input_tokens_per_dollar = round(effective_prompt_toks / cost_for_run, 2) if cost_for_run > 0 else 0
        output_tokens_per_dollar = round(effective_gen_toks / cost_for_run, 2) if cost_for_run > 0 else 0

        result = {{
            **exp,
            'exp_id': exp_id,
            'elapsed_time': wall_clock,
            'total_prompt_tokens': effective_prompt_toks,
            'total_output_tokens': effective_gen_toks,
            'requests_per_sec': round(NUM_REQUESTS / wall_clock, 3) if wall_clock > 0 else 0,
            'input_tokens_per_sec': throughput['tokens_per_sec_prefill'],
            'output_tokens_per_sec': throughput['tokens_per_sec_decode'],
            'total_tokens_per_sec': throughput['tokens_per_sec_total'],
            'status': 'success',
            'timeseries_file': timeseries_file,
            # Infrastructure info
            'instance_type': INSTANCE_TYPE,
            'price_per_hour': PRICE_PER_HOUR,
            'num_nodes': NUM_NODES,
            'gpus_per_node': GPUS_PER_NODE,
            'total_gpus': NUM_NODES * GPUS_PER_NODE,
            # Cost efficiency
            'cost_for_run_usd': round(cost_for_run, 4),
            'tokens_per_dollar': tokens_per_dollar,
            'input_tokens_per_dollar': input_tokens_per_dollar,
            'output_tokens_per_dollar': output_tokens_per_dollar,
            # Benchmark config
            'benchmark_num_requests': NUM_REQUESTS,
            'benchmark_target_concurrency': target_concurrency,
            'num_preemptions': num_preemptions,
            'gpu_monitor_type': 'distributed' if use_distributed else 'local',
            'gpu_monitor_num_nodes': len(gpu_monitor.actors) if (use_distributed and hasattr(gpu_monitor, 'actors')) else 1,
            'gpu_monitor_num_nodes_reported': gpu_metrics.get('num_nodes_monitored', 1),
            # GPU utilization metrics (summary)
            **gpu_metrics,
            # Scheduler/metrics poller metrics (summary)
            **scheduler_metrics,
            # Canonical schema columns
            **canonical,
        }}
        return result

    except Exception as e:
        if gpu_monitor is not None:
            gpu_monitor.stop()

        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
            error_msg = f"OOM: {{error_msg[:200]}}"
        send_discord_message(f"❌ Benchmark {{exp_id}} FAILED: {{error_msg}}")

        error_canonical = compute_canonical_columns(exp, model_info, {{}}, {{
            'tokens_per_sec_total': 0,
            'tokens_per_sec_prefill': 0,
            'tokens_per_sec_decode': 0,
            'elapsed_time': 0,
        }})
        return {{
            **exp,
            'exp_id': exp_id,
            'status': 'error',
            'error': error_msg,
            'instance_type': INSTANCE_TYPE,
            'price_per_hour': PRICE_PER_HOUR,
            'num_nodes': NUM_NODES,
            'gpus_per_node': GPUS_PER_NODE,
            'total_gpus': NUM_NODES * GPUS_PER_NODE,
            **error_canonical,
        }}


# ============================================================================
# Main
# ============================================================================

{global_model_path_line}
TP = EXPERIMENTS[0]['tp']
PP = EXPERIMENTS[0]['pp']

# Start vLLM server
SERVER_LOG = "/tmp/vllm_server.log"
server_proc, server_log_file = start_vllm_server(MODEL_PATH, TP, PP, MAX_MODEL_LEN, SERVER_LOG)

try:
    # Wait for server health
    if not wait_for_health(server_proc, timeout=1800):  # 30 min for large models like 235B
        if server_proc.poll() is not None:
            with open(SERVER_LOG) as f:
                log_tail = f.read()[-3000:]
            print(f"Server log (last 3000 chars):\\n{{log_tail}}")
            if "OutOfMemoryError" in log_tail or "CUDA out of memory" in log_tail:
                error = f"OOM: Model too large for {{GPUS_PER_NODE * NUM_NODES}}x {{GPU_MODEL}}"
            else:
                error = f"Server crashed with exit code {{server_proc.returncode}}"
        else:
            error = "Server failed to respond within 600s"
        # Mark ALL experiments as failed
        results = []
        for exp in EXPERIMENTS:
            {model_path_line}
            model_info = get_model_config_info(model_path)
            error_canonical = compute_canonical_columns(exp, model_info, {{}}, {{
                'tokens_per_sec_total': 0,
                'tokens_per_sec_prefill': 0,
                'tokens_per_sec_decode': 0,
                'elapsed_time': 0,
            }})
            results.append({{
                **exp,
                'status': 'error',
                'error': error,
                'instance_type': INSTANCE_TYPE,
                'price_per_hour': PRICE_PER_HOUR,
                'num_nodes': NUM_NODES,
                'gpus_per_node': GPUS_PER_NODE,
                'total_gpus': NUM_NODES * GPUS_PER_NODE,
                **error_canonical,
            }})
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        send_discord_message(f"❌ Server failed to start: {{error}}")
        sys.exit(1)

    # Parse GPU blocks from server log
    NUM_GPU_BLOCKS, BLOCK_SIZE = parse_gpu_blocks(SERVER_LOG, timeout=30)
    if NUM_GPU_BLOCKS and MAX_MODEL_LEN > 0:
        target_concurrency = int((NUM_GPU_BLOCKS * BLOCK_SIZE) / MAX_MODEL_LEN)
        target_concurrency = max(1, target_concurrency)
        print(f"📊 target_concurrency = {{target_concurrency}} ({{NUM_GPU_BLOCKS}} blocks × {{BLOCK_SIZE}} / {{MAX_MODEL_LEN}})")
    else:
        target_concurrency = 32
        NUM_GPU_BLOCKS = NUM_GPU_BLOCKS or 0
        print(f"⚠️  Using default target_concurrency={{target_concurrency}}")

    # Load any previously completed results (from earlier run on same cluster)
    results = []
    skip_exp_ids = set()  # experiments to skip (completed or permanently failed)
    fail_counts = {{}}  # track how many times each experiment has failed
    MAX_RETRIES = 2
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                results = json.load(f)
            for r in results:
                eid = r.get('exp_id')
                if not eid:
                    continue
                if r.get('status') == 'success':
                    skip_exp_ids.add(eid)
                elif r.get('status') == 'error':
                    fail_counts[eid] = fail_counts.get(eid, 0) + 1
                    if fail_counts[eid] >= MAX_RETRIES:
                        skip_exp_ids.add(eid)  # permanently failed after MAX_RETRIES
            completed = sum(1 for eid in skip_exp_ids if fail_counts.get(eid, 0) < MAX_RETRIES)
            perm_failed = sum(1 for eid in skip_exp_ids if fail_counts.get(eid, 0) >= MAX_RETRIES)
            if skip_exp_ids:
                print(f"📋 Skipping {{len(skip_exp_ids)}} experiments ({{completed}} completed, {{perm_failed}} permanently failed after {{MAX_RETRIES}} retries)")
        except Exception:
            results = []

    # Run each experiment (skip completed and permanently failed)
    for i, exp in enumerate(EXPERIMENTS):
        exp_id = f"tp{{exp['tp']}}_pp{{exp['pp']}}_in{{exp['max_input_length']}}_out{{exp['max_output_length']}}"
        if exp_id in skip_exp_ids:
            reason = "already completed" if fail_counts.get(exp_id, 0) < MAX_RETRIES else f"permanently failed after {{MAX_RETRIES}} attempts"
            print(f"⏭️  Skipping {{exp_id}} ({{reason}})")
            continue
        result = run_single_benchmark(exp, target_concurrency, server_proc)
        results.append(result)
        print(f"Result: {{result.get('status')}}")
        # Save incrementally
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\\n✅ All done! Results in {{RESULTS_FILE}}")

finally:
    # Kill server
    print("🛑 Stopping vLLM server...")
    server_proc.terminate()
    try:
        server_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        server_proc.wait()
    server_log_file.close()
    print("✅ Server stopped")
'''

def run_cluster_benchmarks(cluster_config, experiments, parent_dir=None, dry_run=True, gpu_type=DEFAULT_GPU_TYPE, s3_models=False, cloud="aws", use_spot=False):
    gpus_per_node, num_nodes = cluster_config
    # Use TP/PP from the first experiment for naming (all experiments in a group have compatible TP/PP)
    tp = experiments[0]['tp']
    pp = experiments[0]['pp']
    model = experiments[0]['model']
    # Short model slug for cluster name (e.g., "Qwen/Qwen3-32B" → "qwen3-32b")
    model_slug = model.split("/")[-1].lower().replace("_", "-").replace(".", "-")[:20]
    # Normalize GPU type for use in filenames (replace special chars)
    gpu_type_safe = gpu_type.replace("_", "-").replace("/", "-")
    cluster_name = f"rf-{model_slug}-tp{tp}pp{pp}-{gpu_type_safe}"

    # Create consolidated result directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get instance family, GPU name, and instance type for directory organization
    gpu_config = GPU_CONFIGS[gpu_type]
    instance_family = gpu_config["instance_family"]
    gpu_name = gpu_type
    instance_dir = f"aws_{instance_family}_{gpu_name}"

    # Build instance name for directory (e.g., "g5_12xlarge" or "2x_g5_12xlarge")
    instance_info = gpu_config["pricing"].get(gpus_per_node, {})
    instance_type = instance_info.get("instance_type", f"{instance_family}.unknown")
    if num_nodes > 1:
        instance_name = f"{num_nodes}x_{instance_type}"
    else:
        instance_name = instance_type
    instance_name_safe = instance_name.replace(".", "_")

    subdir_name = f"tp{tp}-pp{pp}-{instance_name_safe}-{timestamp}"

    # Directory structure: results/{instance_dir}/tp{tp}-pp{pp}-{instance}-{timestamp}-{status}
    if parent_dir:
        instance_path = Path(parent_dir) / instance_dir
    else:
        instance_path = Path(instance_dir)

    # Skip if a successful run already exists for this config + model
    existing = list(instance_path.glob(f"tp{tp}-pp{pp}-{instance_name_safe}-*-success"))
    if existing and not dry_run:
        latest = sorted(existing)[-1]
        results_file = latest / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    prior = json.load(f)
                # Check model matches AND all experiment IDs are present
                prior_models = {r.get('model') for r in prior if r.get('status') == 'success'}
                if model in prior_models:
                    prior_success = {r['exp_id'] for r in prior if r.get('status') == 'success' and r.get('model') == model}
                    needed = {f"tp{e['tp']}_pp{e['pp']}_in{e['max_input_length']}_out{e['max_output_length']}" for e in experiments}
                    if needed.issubset(prior_success):
                        print(f"\n⏭️  Skipping {cluster_name} — already completed in {latest.name}")
                        return [r for r in prior if r.get('model') == model]
            except Exception:
                pass  # corrupted results, re-run

    instance_path.mkdir(parents=True, exist_ok=True)
    result_dir = instance_path / subdir_name

    print(f"\n{'='*70}")
    print(f"Cluster: {cluster_name}")
    print(f"Config: {gpus_per_node} GPUs/node × {num_nodes} nodes")
    print(f"Experiments: {len(experiments)}")
    print("="*70)

    if dry_run:
        print("[DRY RUN] Would launch and run:")
        for exp in experiments:
            print(f"  - TP={exp['tp']}, PP={exp['pp']}, "
                  f"in={exp['max_input_length']}, out={exp['max_output_length']}")
        if len(experiments) > 3:
            print(f"  ... and {len(experiments)-3} more")
        return []

    # Create result directory and set up logging
    result_dir.mkdir(exist_ok=True)
    log_file = result_dir / "benchmark.log"
    setup_logger(log_file)

    logger.info(f"Results will be saved to: {result_dir}")

    # Define paths within result directory
    local_results = result_dir / "results.json"

    # 1. Write YAML
    work_dir = Path("roofline_benchmarks")
    work_dir.mkdir(exist_ok=True)
    yaml_path = work_dir / f"{cluster_name}.yaml"
    script_path = work_dir / f"benchmark_{cluster_name}.py"
    
    yaml_content = generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type, s3_models=s3_models, cloud=cloud, use_spot=use_spot)
    yaml_path.write_text(yaml_content)

    # 2. Write benchmark script
    script_content = generate_benchmark_script(experiments, gpus_per_node, num_nodes, gpu_type, s3_models=s3_models, cloud=cloud)
    script_path.write_text(script_content)

    # 3. Copy static helper files into workdir (uploaded to cluster with sky launch)
    import shutil
    src_dir = Path(__file__).parent
    for helper_file in ["benchmark_client.py", "prometheus_parser.py"]:
        src = src_dir / helper_file
        dst = work_dir / helper_file
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"📋 Copied {helper_file} to {work_dir}/")
        else:
            logger.warning(f"⚠️  {helper_file} not found at {src}")

    # Track active cluster for cleanup on unexpected exit
    set_active_cluster(cluster_name)

    try:
        # 3. Launch cluster and capture output
        logger.info(f"🚀 Launching {cluster_name}...")
        logger.info(f"YAML to run: {yaml_path}")
        logger.info(f"Running on cluster: {cluster_name}")

        # Run sky launch and capture output (with retry-until-up to handle capacity issues)
        process = subprocess.Popen(
            ["sky", "launch", "-y", "--retry-until-up", "-c", cluster_name, str(yaml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output with watchdog: kill if no output for 30 minutes
        SKY_OUTPUT_TIMEOUT = 7200  # 2 hours with no output = hung (long-context experiments can take 30+ min)
        last_output_time = time.time()
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 60)  # check every 60s
            if ready:
                line = process.stdout.readline()
                if not line:  # EOF — process closed stdout
                    break
                logger.info(line.rstrip())
                last_output_time = time.time()
            else:
                # No output ready — check if we've exceeded the timeout
                if time.time() - last_output_time > SKY_OUTPUT_TIMEOUT:
                    logger.warning(f"⚠️  No output from sky launch for {SKY_OUTPUT_TIMEOUT}s — killing hung process")
                    process.kill()
                    process.wait()
                    raise subprocess.CalledProcessError(-9, "sky launch (watchdog timeout)")
                # Also check if process has exited
                if process.poll() is not None:
                    # Drain any remaining output
                    for remaining in process.stdout:
                        logger.info(remaining.rstrip())
                    break

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "sky launch")

        # 5. Fetch results
        logger.info(f"📥 Fetching results...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/benchmark_results.json",
            str(local_results)
        ], check=True)

        # Fetch timeseries files directly into result directory
        logger.info(f"📈 Fetching timeseries data...")
        subprocess.run([
            "scp",
            f"{cluster_name}:/tmp/timeseries_*.json",
            str(result_dir)  # Path object converts to string correctly
        ], check=False)  # Don't fail if no timeseries files

        # Update results with local timeseries paths
        with open(local_results) as f:
            results = json.load(f)

        # Check if all experiments succeeded or any failed
        all_succeeded = all(r.get('status') == 'success' for r in results)
        status_suffix = 'success' if all_succeeded else 'fail'
        
        # Rename directory to include status suffix
        new_subdir_name = f"tp{tp}-pp{pp}-{instance_name_safe}-{timestamp}-{status_suffix}"
        new_result_dir = instance_path / new_subdir_name

        if result_dir != new_result_dir:
            logger.info(f"📁 Renaming directory: {result_dir.name} -> {new_result_dir.name}")
            result_dir.rename(new_result_dir)
            result_dir = new_result_dir
            # Update local_results path (files inside directory move automatically with rename)
            local_results = result_dir / "results.json"

        for result in results:
            if 'timeseries_file' in result and result.get('status') == 'success':
                # Update path to local location
                remote_filename = Path(result['timeseries_file']).name
                local_ts_path = result_dir / remote_filename
                result['timeseries_file'] = str(local_ts_path)

        # Save updated results
        with open(local_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV in the same result directory with timestamp
        csv_filename = f"benchmark_results-{timestamp}.csv"
        csv_path = result_dir / csv_filename
        save_results_csv(results, str(csv_path))

        # Automatically plot timeseries for successful experiments
        try:
            successful_ts_files = [
                Path(r['timeseries_file'])
                for r in results
                if r.get('status') == 'success' and r.get('timeseries_file')
            ]
            if successful_ts_files:
                logger.info(f"📈 Plotting timeseries for {len(successful_ts_files)} successful experiment(s)...")
                # For now, call plot_timeseries.py once on the result directory;
                # the script will discover all timeseries_*.json files inside.
                subprocess.run(
                    [
                        "python",
                        "plot_timeseries.py",
                        str(result_dir),
                        "--output",
                        str(result_dir / "timeseries_plots.pdf"),
                    ],
                    check=False,
                )
        except Exception as plot_err:
            logger.warning(f"⚠️  Failed to generate timeseries plots: {plot_err}")
        
        logger.info(f"✅ Results saved to {result_dir}/")
        send_discord_message(f"✅ Results saved to {result_dir}")
        return results

    except Exception as e:
        # Cluster error occurred, but benchmark might have completed successfully
        logger.error(f"❌ Cluster error: {e}")
        logger.info(f"📥 Attempting to fetch results (benchmark may have completed)...")
        
        fetched_results = None
        try:
            logger.info(f"📥 Fetching results.json from cluster...")
            subprocess.run([
                "scp",
                f"{cluster_name}:/tmp/benchmark_results.json",
                str(local_results)
            ], check=True)
            
            # Best-effort fetch of timeseries files even when sky launch failed
            logger.info(f"📈 Attempting to fetch timeseries data from cluster (best-effort)...")
            subprocess.run([
                "scp",
                f"{cluster_name}:/tmp/timeseries_*.json",
                str(result_dir)
            ], check=False)  # Don't fail the cleanup path if timeseries are missing
            
            # Check if we successfully fetched valid results
            if local_results.exists():
                with open(local_results) as f:
                    fetched_results = json.load(f)
                    # Check if any results have 'success' status
                    has_success = any(r.get('status') == 'success' for r in fetched_results)
                    if has_success:
                        logger.info(f"✅ Found successful benchmark results despite cluster error!")
                        logger.info(f"   This is likely a cleanup error, not a benchmark failure.")
                        send_discord_message(f"✅ Cluster {cluster_name} completed benchmarks but had a cleanup error: {str(e)[:80]}")
                    else:
                        send_discord_message(f"❌ Cluster {cluster_name} FAILED: {str(e)[:100]}")
        except Exception as fetch_error:
            logger.warning(f"⚠️  Could not fetch results: {fetch_error}")
            send_discord_message(f"❌ Cluster {cluster_name} FAILED and results could not be fetched: {str(e)[:60]} | fetch error: {str(fetch_error)[:60]}")
        
        # Use fetched results if they contain successful benchmarks, otherwise mark as failed
        if fetched_results and any(r.get('status') == 'success' for r in fetched_results):
            results = fetched_results
            # Determine status suffix based on whether all succeeded
            all_succeeded = all(r.get('status') == 'success' for r in results)
            status_suffix = 'success' if all_succeeded else 'fail'
        else:
            # No valid results fetched, mark all as failed
            results = [{**exp, 'status': 'cluster_error', 'error': str(e)} for exp in experiments]
            status_suffix = 'fail'
        
        # Rename directory to include status suffix
        new_subdir_name = f"tp{tp}-pp{pp}-{instance_name_safe}-{timestamp}-{status_suffix}"
        new_result_dir = instance_path / new_subdir_name

        if result_dir != new_result_dir:
            logger.info(f"📁 Renaming directory: {result_dir.name} -> {new_result_dir.name}")
            result_dir.rename(new_result_dir)
            result_dir = new_result_dir
            # Update local_results path
            local_results = result_dir / "results.json"
        
        # If we have successful results and timeseries files, rewrite their paths
        if results and any(r.get('status') == 'success' for r in results):
            for result in results:
                if 'timeseries_file' in result and result.get('status') == 'success':
                    remote_filename = Path(result['timeseries_file']).name
                    local_ts_path = result_dir / remote_filename
                    result['timeseries_file'] = str(local_ts_path)
        
        # Save results (either fetched successful ones or failed markers)
        with open(local_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV in the same result directory
        csv_filename = f"benchmark_results-{timestamp}.csv"
        csv_path = result_dir / csv_filename
        save_results_csv(results, str(csv_path))

        # If we have successful results, also attempt to plot timeseries
        if results and any(r.get('status') == 'success' for r in results):
            try:
                logger.info(f"📈 Plotting timeseries for successful experiment(s) after cluster error...")
                subprocess.run(
                    [
                        "python",
                        "plot_timeseries.py",
                        str(result_dir),
                        "--output",
                        str(result_dir / "timeseries_plots.pdf"),
                    ],
                    check=False,
                )
            except Exception as plot_err:
                logger.warning(f"⚠️  Failed to generate timeseries plots after cluster error: {plot_err}")
        
        return results

    finally:
        # 6. Teardown
        logger.info(f"🗑️  Tearing down {cluster_name}...")
        subprocess.run(["sky", "down", "-y", cluster_name])

        # Clear active cluster after successful teardown
        set_active_cluster(None)




def save_results_csv(all_results, output_path="benchmark_results.csv"):
    """Save all results to CSV."""
    if not all_results:
        return
    # Collect all unique fieldnames from all results
    fieldnames = []
    for result in all_results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"📊 Saved {len(all_results)} results to {output_path}")


def main():
    # Check for orphaned clusters from previous crashed runs
    check_orphaned_cluster()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run benchmark experiments on AWS GPU instances',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Available GPU types: {', '.join(GPU_CONFIGS.keys())}
Default: {DEFAULT_GPU_TYPE}

Examples:
  python automatic_launch_1.py --gpu L40S
  python automatic_launch_1.py experiment.csv --gpu A100 --run
  python automatic_launch_1.py --gpu H100 --run
        '''
    )
    parser.add_argument('csv_file', nargs='?', default='experiment_L40_llama.csv',
                       help='CSV file with experiment configurations (default: experiment_L40_llama.csv)')
    parser.add_argument('--gpu', '--gpu-type', dest='gpu_type', 
                       choices=list(GPU_CONFIGS.keys()),
                       default=DEFAULT_GPU_TYPE,
                       help=f'GPU type to use (default: {DEFAULT_GPU_TYPE})')
    parser.add_argument('--vm-strategy', dest='vm_strategy',
                       choices=list(VM_SELECTION_STRATEGIES.keys()),
                       default='fit_tp_then_scale',
                       help='VM selection strategy for TP/PP -> cluster sizing')
    parser.add_argument('--run', action='store_true',
                       help='Actually launch clusters (default: dry run)')
    parser.add_argument('--s3-models', action='store_true',
                       help='Load models from S3 instead of HuggingFace (faster, requires prior upload via upload_model_to_s3.py)')
    parser.add_argument('--cloud', dest='cloud',
                       choices=['aws', 'gcp', 'azure'],
                       default='aws',
                       help='Cloud provider to launch on (default: aws)')
    parser.add_argument('--spot', action='store_true',
                       help='Use spot/preemptible instances (cheaper but may be interrupted)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old benchmark files without GPU type in their names')
    
    args = parser.parse_args()
    
    # Handle cleanup option
    if args.cleanup:
        print("🧹 Cleaning up old benchmark files...")
        cleanup_old_benchmark_files()
        if not args.run and args.csv_file == parser.get_default('csv_file'):
            # If only cleanup was requested (no other args), exit after cleanup
            return
    
    csv_path = args.csv_file
    gpu_type = args.gpu_type
    vm_strategy = args.vm_strategy
    dry_run = not args.run

    print(f"🔧 Using GPU type: {gpu_type}")
    print(f"📊 Loading experiments from: {csv_path}")
    if dry_run:
        print("💡 DRY RUN mode - add --run to actually launch clusters")

    experiments = load_experiments(csv_path)
    cluster_groups = group_by_cluster_then_io(experiments, gpu_type, vm_strategy=vm_strategy)

    print(f"📊 Loaded {len(experiments)} experiments")
    print(f"📦 Grouped by cluster config (server reuse across I/O shapes):")
    for (gpus, nodes, tp, pp, model), exps in sorted(cluster_groups.items()):
        io_shapes = set((e['max_input_length'], e['max_output_length']) for e in exps)
        print(f"   TP={tp}, PP={pp}, {gpus} GPU/node × {nodes} nodes: {len(exps)} experiments ({len(io_shapes)} I/O shapes)")
        for il, ol in sorted(io_shapes):
            print(f"      {il}in/{ol}out")

    if dry_run:
        print("\n💡 This is a DRY RUN. Add --run to actually launch clusters.")

    # Run each cluster group and collect all results
    all_results = []

    try:
        for (gpus, nodes, tp, pp, model), exps in sorted(cluster_groups.items()):
            # Parent directory: use results/ with all I/O shapes in one cluster
            parent_dir = "results"
            
            cluster_config = (gpus, nodes)
            results = run_cluster_benchmarks(
                cluster_config, exps, parent_dir=parent_dir,
                dry_run=dry_run, gpu_type=gpu_type,
                s3_models=args.s3_models, cloud=args.cloud,
                use_spot=args.spot,
            )
            if results:
                all_results.extend(results)
    except Exception as e:
        print(f"\n❌ Unexpected error in main: {e}")
        print("   Cleanup will be triggered automatically...")
        raise  # Re-raise to trigger atexit handler


if __name__ == "__main__":
    main()


# =============================================================================
# LMCACHE REFERENCE — Removed in Phase 5 cleanup, preserved here for future use.
# LMCache (https://docs.lmcache.ai) is a KV cache sharing/offloading layer for
# vLLM. We tested v0.2.1 and v0.3.x with vLLM 0.7–0.13 but hit persistent
# Ray DAG issues with pipeline parallelism (PP>1). Re-enable when upstream
# stabilizes.
# =============================================================================
#
# --- 1. Environment variables (add to env_exports in generate_yaml) ----------
#
#   export LMCACHE_USE_EXPERIMENTAL="True"
#   export LMCACHE_CHUNK_SIZE="256"
#   export LMCACHE_LOCAL_CPU="True"
#   export LMCACHE_MAX_LOCAL_CPU_SIZE="40.0"
#   export LMCACHE_SAVE_UNFULL_CHUNK="True"
#   export LMCACHE_ENABLE_ASYNC_LOADING="False"
#   export LMCACHE_REMOTE_SERDE="cachegen"
#   export LMCACHE_USE_LAYERWISE="True"
#   export LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR="True"
#
# Optional / experimental:
#   export LMCACHE_LOOKUP_TIMEOUT_MS="12000"
#   export LMCACHE_LOCAL_DISK="/tmp/lmcache_disk"
#   export LMCACHE_MAX_LOCAL_DISK_SIZE="100"
#   export LMCACHE_DISK_PERSISTENCE="True"
#   export LMCACHE_LOG_LEVEL="INFO"
#
# --- 2. Installation (add to setup: block in YAML) --------------------------
#
#   # Install LMCache v0.2.1 (compatible with vLLM 0.7.x)
#   # See: https://docs.lmcache.ai/getting_started/installation.html
#   # Note: v0.2.2 doesn't exist! Available v0.2.x tags: v0.2.0, v0.2.1
#   rm -rf lmcache
#   git clone https://github.com/lmcache/lmcache.git
#   cd lmcache
#   git checkout v0.2.1
#   # CUDA arch: A100=8.0, L40S=8.9, L4=7.5, H100=9.0
#   export TORCH_CUDA_ARCH_LIST="8.0;8.9;7.5;9.0"
#   export FORCE_CUDA="1"
#   uv pip install . --no-build-isolation || {
#     export TORCH_CUDA_ARCH_LIST="8.0"
#     uv pip install . --no-build-isolation
#   }
#   cd ..
#
#   # ⚠️ CRITICAL: LMCache deps overwrite torch with CPU-only version!
#   # Must force-reinstall CUDA torch AFTER LMCache. Use cu121 for A100 driver
#   # 535.x compatibility.
#   uv pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 \
#     "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
#
#   # Verify CUDA is back:
#   python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
#
#   mkdir -p /tmp/lmcache_disk
#
# --- 3. Testing notes -------------------------------------------------------
#
# - vLLM v0.10.2 without LMCache: worked for TP4/PP3
# - vLLM v0.11.0 without LMCache: worked
# - LMCache v0.3.9 + vLLM v0.11.0: one experiment worked, rest failed (Ray DAG)
# - LMCache v0.3.10 + vLLM v0.11.0: same Ray DAG issue
# - LMCache v0.3.10 + vLLM v0.13.0: "layers not found" error
# - Ray restarts did NOT help; cleanup_dist_env_and_memory() between runs DID help
# - VLLM_USE_RAY_WRAPPED_PP_COMM=0 did not help
# - enable_async_loading: True — no significant effect
# - TP4/PP4 worked only with vLLM v0.10.0, max_gpu_util=0.85, NCCL_P2P_DISABLE=1
# - LMCache with TP4/PP4: did not work
# =============================================================================