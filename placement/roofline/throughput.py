"""
Roofline-based throughput calculation for LLM inference.

Uses the roofline model to estimate throughput based on:
- Arithmetic intensity (FLOPs per byte)
- GPU compute capacity (TFLOPS)
- GPU memory bandwidth (GB/s)
- Tensor parallelism overhead

Ported from: ../LLM_placement_solver/solver.py
"""

from typing import Optional
from dataclasses import dataclass

from .gpu_specs import get_gpu_specs, get_ridge_point


@dataclass
class ThroughputResult:
    """Result of throughput calculation."""

    throughput_tokens_per_sec: float
    regime: str  # "COMPUTE_BOUND" or "MEMORY_BOUND"
    arithmetic_intensity: float
    ridge_point: float
    bottleneck: str  # Human-readable bottleneck description


class ThroughputCalculator:
    """
    Roofline-based throughput calculator for LLM inference.

    Supports:
    - Prefill phase (O(n²) attention, processes all tokens at once)
    - Decode phase (O(n) attention, generates one token at a time)
    - Aggregated phase (combined prefill→decode with interference)
    - Tensor parallelism (TP) with communication overhead
    """

    @staticmethod
    def batch_efficiency_factor(batch_size: int) -> float:
        """
        Batch size efficiency factor - larger batches improve GPU utilization.

        Based on empirical observations:
        - Small batches (1-4): Poor GPU utilization (50-80%)
        - Medium batches (8-16): Good utilization (90-95%)
        - Large batches (32+): Near-optimal utilization (95-100%)

        Args:
            batch_size: Batch size

        Returns:
            Efficiency factor (0-1)
        """
        if batch_size >= 32:
            return 1.0  # Optimal utilization
        elif batch_size >= 16:
            return 0.95  # Very good
        elif batch_size >= 8:
            return 0.90  # Good
        elif batch_size >= 4:
            return 0.80  # Moderate
        elif batch_size >= 2:
            return 0.65  # Poor
        else:
            return 0.50  # Very poor (batch=1)

    @staticmethod
    def calculate_arithmetic_intensity(
        num_layers: int,
        batch_size: int,
        seq_len: int,
        d_model: int,
        d_hidden: int,
        bytes_per_element: int,
        tp_degree: int = 1,
        phase: str = "prefill",
        num_attention_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ) -> float:
        """
        Calculate arithmetic intensity (FLOPs / Byte) using roofline model.

        Higher AI = more compute-bound, lower AI = more memory-bound.

        PHASE-AWARE: Prefill (O(n²)) vs Decode (O(n)) have very different AI!

        Args:
            num_layers: Number of transformer layers in segment
            batch_size: Batch size
            seq_len: Sequence length (or KV cache length for decode)
            d_model: Model hidden dimension
            d_hidden: FFN intermediate dimension
            bytes_per_element: Bytes per element (2 for FP16, 4 for FP32)
            tp_degree: Tensor parallelism degree
            phase: 'prefill' or 'decode'
            num_attention_heads: Number of attention heads (defaults to d_model // 128)
            num_kv_heads: Number of KV heads for GQA (defaults to num_attention_heads)

        Returns:
            Arithmetic intensity in FLOPs per byte
        """
        if num_attention_heads is None:
            num_attention_heads = max(1, d_model // 128)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads

        d_head = d_model / num_attention_heads
        kv_dim = num_kv_heads * d_head

        # === FLOPs Calculation (PHASE-AWARE) ===
        if phase == "prefill":
            # PREFILL: Process all tokens at once (O(n²) attention)
            flops_q = 2 * batch_size * seq_len * d_model * (d_model / tp_degree)
            flops_k = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * seq_len * d_model * (d_model / tp_degree)
            flops_attn_proj = flops_q + flops_k + flops_v + flops_o
            flops_attn_scores = (
                4 * batch_size * seq_len * seq_len * (d_model / tp_degree)
            )  # O(n²)
            flops_attention = flops_attn_proj + flops_attn_scores

            # FFN for all tokens
            flops_ffn = (
                2
                * batch_size
                * seq_len
                * (
                    d_model * (d_hidden / tp_degree)
                    + d_model * (d_hidden / tp_degree)
                    + (d_hidden / tp_degree) * d_model
                )
            )
        else:  # decode
            # DECODE: Generate ONE token (O(n) attention to KV cache)
            kv_cache_len = seq_len  # seq_len represents cached context
            flops_q = 2 * batch_size * 1 * d_model * (d_model / tp_degree)
            flops_k = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * 1 * d_model * (d_model / tp_degree)
            flops_attn_proj = flops_q + flops_k + flops_v + flops_o
            flops_attn_scores = (
                4 * batch_size * 1 * kv_cache_len * (d_model / tp_degree)
            )  # O(n)
            flops_attention = flops_attn_proj + flops_attn_scores

            # FFN for 1 token
            flops_ffn = (
                2
                * batch_size
                * 1
                * (
                    d_model * (d_hidden / tp_degree)
                    + d_model * (d_hidden / tp_degree)
                    + (d_hidden / tp_degree) * d_model
                )
            )

        flops_per_layer = flops_attention + flops_ffn
        total_flops = flops_per_layer * num_layers

        # === Memory Access Calculation (PHASE-AWARE) ===
        # Weights (divided by TP): same for both phases
        bytes_weights_per_layer = (
            (
                (2 * d_model * d_model)  # W_q, W_o
                + (2 * d_model * kv_dim)  # W_k, W_v (GQA/MQA aware)
                + (3 * d_model * d_hidden)  # W_gate, W_up, W_down
            )
            * bytes_per_element
            / tp_degree
        )

        if phase == "prefill":
            # KV cache being written
            bytes_kv_cache_per_layer = (
                2 * batch_size * seq_len * kv_dim * bytes_per_element / tp_degree
            )
            # Activations for all tokens
            bytes_activations_per_layer = (
                batch_size * seq_len * d_model * bytes_per_element
            )
        else:  # decode
            # KV cache being READ (full cached context!)
            kv_cache_len = seq_len
            bytes_kv_cache_per_layer = (
                2 * batch_size * kv_cache_len * kv_dim * bytes_per_element / tp_degree
            )
            # Activations for 1 token only
            bytes_activations_per_layer = batch_size * 1 * d_model * bytes_per_element

        bytes_per_layer = (
            bytes_weights_per_layer
            + bytes_kv_cache_per_layer
            + bytes_activations_per_layer
        )
        total_bytes = bytes_per_layer * num_layers

        # Arithmetic Intensity = FLOPs / Bytes
        if total_bytes == 0:
            return float("inf")

        return total_flops / total_bytes

    @staticmethod
    def determine_regime(arithmetic_intensity: float, ridge_point: float) -> str:
        """
        Determine if workload is compute-bound or memory-bound.

        Args:
            arithmetic_intensity: FLOPs per byte
            ridge_point: Ridge point (FLOPs per byte) for the GPU

        Returns:
            "COMPUTE_BOUND" or "MEMORY_BOUND"
        """
        return "COMPUTE_BOUND" if arithmetic_intensity > ridge_point else "MEMORY_BOUND"

    @classmethod
    def calculate_throughput(
        cls,
        gpu_type: str,
        seq_len: int,
        batch_size: int,
        num_layers: int,
        d_model: int,
        d_hidden: int,
        tp_degree: int = 1,
        bytes_per_element: int = 2,
        nvlink_bw_gbps: float = 300.0,
        phase: str = "prefill",
        output_length: int = 0,
        num_attention_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        interference_factor: float = 0.80,
    ) -> ThroughputResult:
        """
        Calculate GPU throughput using roofline model with TP support.

        CRITICAL: TP is NOT data parallelism!
        - TP splits model weights across GPUs for the SAME batch
        - Each GPU processes the SAME sequences (not different ones)
        - Purpose: fit larger models in memory, NOT increase throughput
        - Throughput effect: minor speedup from reduced memory pressure, offset by communication

        Args:
            gpu_type: GPU type name (e.g., 'A100', 'L40S')
            seq_len: Sequence length (for prefill) OR KV cache length (for decode)
            batch_size: Batch size
            num_layers: Number of layers in segment
            d_model: Model hidden dimension
            d_hidden: FFN intermediate dimension
            tp_degree: Tensor parallelism degree
            bytes_per_element: Bytes per element (2 for FP16)
            nvlink_bw_gbps: NVLink bandwidth in GB/s
            phase: 'prefill', 'decode', or 'aggregated'
            output_length: Output length (for decode/aggregated phase)
            num_attention_heads: Number of attention heads
            num_kv_heads: Number of KV heads for GQA
            interference_factor: Penalty for prefill-decode mixing in aggregated mode

        Returns:
            ThroughputResult with throughput and regime information
        """
        # Handle aggregated phase: compute prefill and decode separately, then combine
        if phase == "aggregated":
            prefill_result = cls.calculate_throughput(
                gpu_type=gpu_type,
                seq_len=seq_len,
                batch_size=batch_size,
                num_layers=num_layers,
                d_model=d_model,
                d_hidden=d_hidden,
                tp_degree=tp_degree,
                bytes_per_element=bytes_per_element,
                nvlink_bw_gbps=nvlink_bw_gbps,
                phase="prefill",
                output_length=0,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
            )
            decode_result = cls.calculate_throughput(
                gpu_type=gpu_type,
                seq_len=seq_len,
                batch_size=batch_size,
                num_layers=num_layers,
                d_model=d_model,
                d_hidden=d_hidden,
                tp_degree=tp_degree,
                bytes_per_element=bytes_per_element,
                nvlink_bw_gbps=nvlink_bw_gbps,
                phase="decode",
                output_length=output_length,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
            )

            # Combine using request-level throughput model
            aggregated_tps = cls._calculate_aggregated_throughput(
                prefill_throughput=prefill_result.throughput_tokens_per_sec,
                decode_throughput=decode_result.throughput_tokens_per_sec,
                seq_len=seq_len,
                output_len=output_length,
                interference_factor=interference_factor,
            )

            return ThroughputResult(
                throughput_tokens_per_sec=aggregated_tps,
                regime="AGGREGATED",
                arithmetic_intensity=(
                    prefill_result.arithmetic_intensity
                    + decode_result.arithmetic_intensity
                )
                / 2,
                ridge_point=prefill_result.ridge_point,
                bottleneck=f"Prefill: {prefill_result.bottleneck}, Decode: {decode_result.bottleneck}",
            )

        # Get GPU specs
        specs = get_gpu_specs(gpu_type)
        ridge_point = get_ridge_point(gpu_type)

        # Default attention heads
        if num_attention_heads is None:
            num_attention_heads = max(1, d_model // 128)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads

        d_head = d_model / num_attention_heads
        kv_dim = num_kv_heads * d_head

        # Calculate arithmetic intensity
        arithmetic_intensity = cls.calculate_arithmetic_intensity(
            num_layers,
            batch_size,
            seq_len,
            d_model,
            d_hidden,
            bytes_per_element,
            tp_degree,
            phase,
            num_attention_heads,
            num_kv_heads,
        )

        regime = cls.determine_regime(arithmetic_intensity, ridge_point)

        # === Compute FLOPs PER GPU (with TP, each GPU does less compute) ===
        if phase == "prefill":
            flops_q = 2 * batch_size * seq_len * d_model * (d_model / tp_degree)
            flops_k = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * seq_len * d_model * (d_model / tp_degree)
            attn_proj_flops = flops_q + flops_k + flops_v + flops_o
            attn_score_flops = (
                4 * batch_size * seq_len * seq_len * (d_model / tp_degree)
            )

            ffn_flops = (
                2
                * batch_size
                * seq_len
                * (
                    d_model * (d_hidden / tp_degree)
                    + d_model * (d_hidden / tp_degree)
                    + (d_hidden / tp_degree) * d_model
                )
            )
        else:  # decode
            # Use average KV cache length for realistic estimation
            if output_length > 0:
                avg_kv_cache_len = seq_len + (output_length - 1) / 2.0
            else:
                avg_kv_cache_len = seq_len

            flops_q = 2 * batch_size * 1 * d_model * (d_model / tp_degree)
            flops_k = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * 1 * d_model * (d_model / tp_degree)
            attn_proj_flops = flops_q + flops_k + flops_v + flops_o
            attn_score_flops = (
                4 * batch_size * 1 * avg_kv_cache_len * (d_model / tp_degree)
            )

            ffn_flops = (
                2
                * batch_size
                * 1
                * (
                    d_model * (d_hidden / tp_degree)
                    + d_model * (d_hidden / tp_degree)
                    + (d_hidden / tp_degree) * d_model
                )
            )

        flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops
        total_flops_per_gpu = num_layers * flops_per_layer

        # === Compute Memory Access PER GPU ===
        weight_bytes = (
            num_layers
            * (
                (2 * d_model * (d_model / tp_degree))
                + (2 * d_model * (kv_dim / tp_degree))
                + (3 * d_model * (d_hidden / tp_degree))
            )
            * bytes_per_element
        )

        if phase == "prefill":
            activation_bytes = (
                num_layers * batch_size * seq_len * d_model * bytes_per_element
            )
            kv_cache_bytes = (
                num_layers
                * 2
                * batch_size
                * seq_len
                * (kv_dim / tp_degree)
                * bytes_per_element
            )
        else:
            activation_bytes = num_layers * batch_size * 1 * d_model * bytes_per_element
            if output_length > 0:
                avg_kv_cache_len = seq_len + (output_length - 1) / 2.0
            else:
                avg_kv_cache_len = seq_len
            kv_cache_bytes = (
                num_layers
                * 2
                * batch_size
                * avg_kv_cache_len
                * (kv_dim / tp_degree)
                * bytes_per_element
            )

        total_bytes_per_gpu = weight_bytes + activation_bytes + kv_cache_bytes

        # === Calculate BASE time ===
        if regime == "COMPUTE_BOUND":
            base_time_per_batch = total_flops_per_gpu / (
                specs["tflops"] * 1e12 * specs["efficiency"]
            )
            bottleneck = "GPU Compute (TFLOPS)"
        else:
            base_time_per_batch = total_bytes_per_gpu / (
                specs["mem_bw"] * 1e9 * specs["efficiency"]
            )
            bottleneck = "Memory Bandwidth (GB/s)"

        # === TP Communication Overhead ===
        if phase == "prefill":
            activation_size_bytes = batch_size * seq_len * d_model * bytes_per_element
        else:
            activation_size_bytes = batch_size * 1 * d_model * bytes_per_element

        comm_time_per_layer = (
            2
            * activation_size_bytes
            / (nvlink_bw_gbps * 1e9)
            * (tp_degree - 1)
            / tp_degree
        )
        total_comm_time = num_layers * comm_time_per_layer

        # TP Efficiency Factor
        if tp_degree == 1:
            tp_efficiency = 1.0
        else:
            additional_overhead = {1: 0.00, 2: 0.05, 4: 0.10, 8: 0.15, 16: 0.20}.get(
                tp_degree, 0.25
            )
            tp_efficiency = max(0.30, 1.0 - additional_overhead)

        # Apply TP efficiency
        time_per_batch = base_time_per_batch / tp_efficiency
        total_time = time_per_batch + total_comm_time

        # Calculate throughput
        if phase == "prefill":
            tokens_per_batch = batch_size * seq_len
        else:
            tokens_per_batch = batch_size * 1

        base_throughput = tokens_per_batch / total_time
        batch_eff = cls.batch_efficiency_factor(batch_size)
        final_throughput = base_throughput * batch_eff

        return ThroughputResult(
            throughput_tokens_per_sec=final_throughput,
            regime=regime,
            arithmetic_intensity=arithmetic_intensity,
            ridge_point=ridge_point,
            bottleneck=bottleneck,
        )

    @staticmethod
    def _calculate_aggregated_throughput(
        prefill_throughput: float,
        decode_throughput: float,
        seq_len: int,
        output_len: int,
        interference_factor: float = 0.80,
    ) -> float:
        """
        Request-level throughput model for aggregated clusters.

        Models vLLM-style continuous batching where same GPUs handle
        complete requests (prefill → decode) with interference effects.

        Args:
            prefill_throughput: Throughput for pure prefill workload (tokens/sec)
            decode_throughput: Throughput for pure decode workload (tokens/sec)
            seq_len: Input sequence length (prefill tokens)
            output_len: Output length (decode tokens to generate)
            interference_factor: Penalty for prefill-decode mixing (0.7-0.9 typical)

        Returns:
            Effective aggregated throughput in tokens/second
        """
        if prefill_throughput <= 0 or decode_throughput <= 0:
            return 0.0
        if seq_len <= 0 and output_len <= 0:
            return 0.0

        # Time to process one complete request
        prefill_time = seq_len / prefill_throughput
        decode_time = output_len / decode_throughput
        total_time = prefill_time + decode_time

        # Total tokens in one request
        total_tokens = seq_len + output_len

        # Effective throughput with interference penalty
        base_throughput = total_tokens / total_time if total_time > 0 else 0.0

        return base_throughput * interference_factor
