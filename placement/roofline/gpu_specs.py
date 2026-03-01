"""
GPU specifications for roofline model calculations.

Contains hardware specs (TFLOPS, memory bandwidth, efficiency) for various GPU types,
used to determine compute-bound vs memory-bound regimes and calculate throughput.

Ported from: ../LLM_placement_solver/solver.py
"""

from typing import Dict, Any

# GPU hardware specifications
# - tflops: Peak FP16/BF16 TFLOPS with tensor cores
# - mem_bw: Peak memory bandwidth in GB/s
# - efficiency: Realistic efficiency factor (0-1) accounting for real-world overhead
GPU_SPECS: Dict[str, Dict[str, float]] = {
    # Modern GPUs optimized for LLM inference (FP16/BF16 with modern tensor cores)
    'H100': {'tflops': 989, 'mem_bw': 3350, 'efficiency': 0.70},  # Best for LLMs
    'H200': {'tflops': 989, 'mem_bw': 4800, 'efficiency': 0.70},  # Higher bandwidth H100
    'A100': {'tflops': 312, 'mem_bw': 2039, 'efficiency': 0.65},  # Excellent for LLMs
    'L40S': {'tflops': 362, 'mem_bw': 864, 'efficiency': 0.58},   # Good Ada Lovelace
    'A40': {'tflops': 150, 'mem_bw': 696, 'efficiency': 0.52},    # Ampere workstation
    'L40': {'tflops': 181, 'mem_bw': 864, 'efficiency': 0.55},    # Ada Lovelace
    'L4': {'tflops': 121, 'mem_bw': 300, 'efficiency': 0.50},     # Entry Ada Lovelace

    # Older GPUs - lower efficiency for modern LLM workloads
    'V100': {'tflops': 125, 'mem_bw': 900, 'efficiency': 0.42},   # Volta - old tensor cores
    'RTX4090': {'tflops': 165, 'mem_bw': 1008, 'efficiency': 0.50},

    # Budget GPUs
    'L20': {'tflops': 119, 'mem_bw': 480, 'efficiency': 0.48},    # Budget Ada
    'A10': {'tflops': 125, 'mem_bw': 600, 'efficiency': 0.45},    # Budget Ampere
    'A10G': {'tflops': 125, 'mem_bw': 600, 'efficiency': 0.45},   # AWS A10G (same as A10)
    'T4': {'tflops': 65, 'mem_bw': 320, 'efficiency': 0.38},      # Old Turing
}

# GPU performance tiers for hierarchy constraints
# Higher tier = more powerful GPU, may warrant higher TP degree
GPU_PERFORMANCE_TIERS: Dict[str, int] = {
    'H200': 6,
    'H100': 5,
    'A100': 4,
    'L40S': 3,
    'A40': 3,
    'L40': 2,
    'L4': 2,
    'V100': 2,
    'A10': 1,
    'A10G': 1,
    'T4': 1,
    'RTX4090': 2,
    'L20': 2,
}

# GPU memory sizes (GB) - used for memory feasibility checks
GPU_MEMORY_GB: Dict[str, float] = {
    'H200': 141,
    'H100': 80,
    'A100': 80,  # 80GB variant (also 40GB exists)
    'L40S': 48,
    'A40': 48,
    'L40': 48,
    'L4': 24,
    'V100': 32,  # 32GB variant (also 16GB exists)
    'A10': 24,
    'A10G': 24,
    'T4': 16,
    'RTX4090': 24,
    'L20': 48,
}


def get_gpu_specs(gpu_type: str) -> Dict[str, float]:
    """
    Get GPU specifications for a given GPU type.

    Args:
        gpu_type: GPU type name (e.g., 'A100', 'L40S')

    Returns:
        Dictionary with 'tflops', 'mem_bw', 'efficiency' keys.
        Falls back to A100 specs if GPU type is unknown.
    """
    return GPU_SPECS.get(gpu_type, GPU_SPECS['A100'])


def get_ridge_point(gpu_type: str) -> float:
    """
    Calculate the ridge point for roofline model.

    Ridge point = Peak FLOPS / Peak Bandwidth (FLOPs per byte)

    - Above ridge point: compute-bound (limited by GPU compute)
    - Below ridge point: memory-bound (limited by memory bandwidth)

    Args:
        gpu_type: GPU type name

    Returns:
        Ridge point in FLOPs per byte
    """
    specs = get_gpu_specs(gpu_type)

    # Convert to consistent units
    peak_flops = specs['tflops'] * 1e12  # TFLOPS → FLOPS
    peak_bandwidth = specs['mem_bw'] * 1e9  # GB/s → bytes/s

    return peak_flops / peak_bandwidth


def get_gpu_memory(gpu_type: str) -> float:
    """
    Get GPU memory size in GB.

    Args:
        gpu_type: GPU type name

    Returns:
        Memory size in GB. Falls back to 48GB if unknown.
    """
    return GPU_MEMORY_GB.get(gpu_type, 48.0)


def normalize_gpu_type(gpu_name: str) -> str:
    """
    Normalize GPU type name to match GPU_SPECS keys.

    Handles variations like 'NVIDIA A100', 'A100-80GB', 'a100', etc.

    Args:
        gpu_name: Raw GPU name string

    Returns:
        Normalized GPU type name
    """
    name = gpu_name.upper().strip()

    # Remove common prefixes
    for prefix in ['NVIDIA ', 'NVIDIA-']:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Handle specific patterns
    if 'A100' in name:
        return 'A100'
    if 'H100' in name:
        return 'H100'
    if 'H200' in name:
        return 'H200'
    if 'L40S' in name:
        return 'L40S'
    if 'L40' in name and 'S' not in name:
        return 'L40'
    if 'L4' in name and '0' not in name:
        return 'L4'
    if 'A40' in name:
        return 'A40'
    if 'A10G' in name:
        return 'A10G'
    if 'A10' in name:
        return 'A10'
    if 'V100' in name:
        return 'V100'
    if 'T4' in name:
        return 'T4'
    if 'L20' in name:
        return 'L20'
    if 'RTX4090' in name or '4090' in name:
        return 'RTX4090'

    # Return as-is if no match
    return name


# AWS instance type to GPU mapping
# Maps instance type to GPU model and count
AWS_INSTANCE_GPU_MAP: Dict[str, Dict[str, Any]] = {
    # P5 instances (H100)
    "p5.48xlarge": {"gpu_model": "H100", "num_gpus": 8},
    "p5en.48xlarge": {"gpu_model": "H200", "num_gpus": 8},

    # P4 instances (A100)
    "p4d.24xlarge": {"gpu_model": "A100", "num_gpus": 8},
    "p4de.24xlarge": {"gpu_model": "A100", "num_gpus": 8},

    # P3 instances (V100)
    "p3.2xlarge": {"gpu_model": "V100", "num_gpus": 1},
    "p3.8xlarge": {"gpu_model": "V100", "num_gpus": 4},
    "p3.16xlarge": {"gpu_model": "V100", "num_gpus": 8},
    "p3dn.24xlarge": {"gpu_model": "V100", "num_gpus": 8},

    # G6e instances (L40S)
    "g6e.xlarge": {"gpu_model": "L40S", "num_gpus": 1},
    "g6e.2xlarge": {"gpu_model": "L40S", "num_gpus": 1},
    "g6e.4xlarge": {"gpu_model": "L40S", "num_gpus": 1},
    "g6e.8xlarge": {"gpu_model": "L40S", "num_gpus": 1},
    "g6e.12xlarge": {"gpu_model": "L40S", "num_gpus": 4},
    "g6e.24xlarge": {"gpu_model": "L40S", "num_gpus": 4},
    "g6e.48xlarge": {"gpu_model": "L40S", "num_gpus": 8},

    # G6 instances (L4)
    "g6.xlarge": {"gpu_model": "L4", "num_gpus": 1},
    "g6.2xlarge": {"gpu_model": "L4", "num_gpus": 1},
    "g6.4xlarge": {"gpu_model": "L4", "num_gpus": 1},
    "g6.8xlarge": {"gpu_model": "L4", "num_gpus": 1},
    "g6.12xlarge": {"gpu_model": "L4", "num_gpus": 2},
    "g6.24xlarge": {"gpu_model": "L4", "num_gpus": 4},
    "g6.48xlarge": {"gpu_model": "L4", "num_gpus": 8},

    # G5 instances (A10G)
    "g5.xlarge": {"gpu_model": "A10G", "num_gpus": 1},
    "g5.2xlarge": {"gpu_model": "A10G", "num_gpus": 1},
    "g5.4xlarge": {"gpu_model": "A10G", "num_gpus": 1},
    "g5.8xlarge": {"gpu_model": "A10G", "num_gpus": 1},
    "g5.12xlarge": {"gpu_model": "A10G", "num_gpus": 4},
    "g5.24xlarge": {"gpu_model": "A10G", "num_gpus": 4},
    "g5.48xlarge": {"gpu_model": "A10G", "num_gpus": 8},
}


def calculate_max_supported_context(
    gpu_model: str,
    num_layers: int,
    layer_weight_gb: float,
    num_kv_heads: int,
    d_model: int,
    num_attention_heads: int,
    tp_degree: int = 1,
    pp_stages: int = 1,
    batch_size: int = 32,
    bytes_per_element: int = 2,
    gpu_utilization: float = 0.85,
) -> int:
    """
    Calculate maximum context length a GPU can support given model config.

    This is critical for setting vLLM's --max-model-len to avoid OOM at startup.
    vLLM pre-allocates KV cache for max_model_len, so we need to know what fits.

    Args:
        gpu_model: GPU type (e.g., "L40S", "A100")
        num_layers: Total decoder layers in model
        layer_weight_gb: Weight memory per layer in GB
        num_kv_heads: Number of KV attention heads (for GQA/MQA)
        d_model: Hidden dimension
        num_attention_heads: Number of attention heads
        tp_degree: Tensor parallelism degree
        pp_stages: Pipeline parallelism stages (layers split across stages)
        batch_size: Batch size for KV cache calculation
        bytes_per_element: 2 for FP16, 4 for FP32
        gpu_utilization: Fraction of GPU memory to use (default 0.85 for safety)

    Returns:
        Maximum context length (sequence length) that fits in GPU memory
    """
    gpu_memory_gb = GPU_MEMORY_GB.get(gpu_model, 48)  # Default to 48GB if unknown

    # Available memory after utilization factor
    available_memory_gb = gpu_memory_gb * gpu_utilization

    # Memory used by model weights
    # - PP splits layers across stages (each GPU has num_layers / pp_stages layers)
    # - TP shards each layer across GPUs
    layers_per_gpu = num_layers / pp_stages
    weights_memory_gb = (layers_per_gpu * layer_weight_gb) / tp_degree

    # vLLM overhead estimate:
    # - Activation memory: ~1-2GB depending on batch size
    # - CUDA graphs: ~2-3GB for warmup captures
    # - Sampler buffers: vocab_size * max_num_seqs * bytes (can be 500MB+)
    # - PyTorch reserved memory and fragmentation
    vllm_overhead_gb = 4.0  # Conservative estimate for vLLM v1 overhead

    # Memory available for KV cache
    kv_available_gb = available_memory_gb - weights_memory_gb - vllm_overhead_gb

    if kv_available_gb <= 0:
        return 1024  # Minimum fallback - config is memory constrained

    # KV cache calculation:
    # Per layer on this GPU: 2 * batch * seq_len * (kv_dim / tp) * bytes_per_element
    # kv_dim = num_kv_heads * head_dim
    # head_dim = d_model / num_attention_heads
    head_dim = d_model / num_attention_heads
    kv_dim = num_kv_heads * head_dim

    # Bytes per token per layer for KV cache (K and V)
    # Note: KV cache is also sharded by TP
    kv_bytes_per_token_per_layer = 2 * batch_size * (kv_dim / tp_degree) * bytes_per_element

    # Total KV bytes per token across layers ON THIS GPU (layers_per_gpu, not all layers)
    kv_bytes_per_token = kv_bytes_per_token_per_layer * layers_per_gpu

    # Convert available GB to bytes
    kv_available_bytes = kv_available_gb * (1024 ** 3)

    # Max context length
    max_context = int(kv_available_bytes / kv_bytes_per_token)

    # Clamp to reasonable range
    max_context = max(1024, min(max_context, 131072))  # Between 1K and 128K

    return max_context
