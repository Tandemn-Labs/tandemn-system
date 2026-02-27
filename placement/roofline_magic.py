"""
Roofline-based VPCMagic implementation for AWS.

This module provides a deterministic, roofline-model-based placement solver
that replaces the LLM-based 3-advisor + C-PMI system in aws_magic.py.

It uses the actual LLM_placement_solver for accurate throughput estimation
and cost optimization.

Usage:
    from placement.roofline_magic import RooflineAWSAllocation

    solver = RooflineAWSAllocation(
        perfdb_dir="./perf_db",
        aws_quota_csv="./quota/aws_gpu_quota_by_region.csv",
    )
    result = solver.decide(request)
"""

import uuid
import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from placement.magic import VPCMagic
from utils.utils import load_aws_quota_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reverse map: GPU type -> list of (instance_type, gpu_count)
# Built from AWS_INSTANCE_TO_GPU at module level, sorted by ascending gpu_count.
# ---------------------------------------------------------------------------
def _build_gpu_type_to_instances(
    instance_map: Dict[str, tuple],
) -> Dict[str, List[tuple]]:
    """Build reverse map from GPU name to list of (instance_type, gpu_count)."""
    rev: Dict[str, List[tuple]] = {}
    for inst, (gpu_name, gpu_count) in instance_map.items():
        rev.setdefault(gpu_name, []).append((inst, gpu_count))
    for gpu_name in rev:
        rev[gpu_name].sort(key=lambda x: x[1])
    return rev


# AWS instance type to GPU mapping
AWS_INSTANCE_TO_GPU: Dict[str, tuple] = {
    # P5 instances (H100)
    "p5.48xlarge": ("H100", 8),
    # P4 instances (A100)
    "p4d.24xlarge": ("A100", 8),
    "p4de.24xlarge": ("A100", 8),
    # P3 instances (V100)
    "p3.2xlarge": ("V100", 1),
    "p3.8xlarge": ("V100", 4),
    "p3.16xlarge": ("V100", 8),
    "p3dn.24xlarge": ("V100", 8),
    # G6e instances (L40S)
    "g6e.xlarge": ("L40S", 1),
    "g6e.2xlarge": ("L40S", 1),
    "g6e.4xlarge": ("L40S", 1),
    "g6e.12xlarge": ("L40S", 4),
    "g6e.24xlarge": ("L40S", 4),
    "g6e.48xlarge": ("L40S", 8),
    # G6 instances (L4)
    "g6.xlarge": ("L4", 1),
    "g6.2xlarge": ("L4", 1),
    "g6.4xlarge": ("L4", 1),
    "g6.12xlarge": ("L4", 2),
    "g6.24xlarge": ("L4", 4),
    "g6.48xlarge": ("L4", 8),
    # G5 instances (A10G)
    "g5.xlarge": ("A10G", 1),
    "g5.2xlarge": ("A10G", 1),
    "g5.4xlarge": ("A10G", 1),
    "g5.12xlarge": ("A10G", 4),
    "g5.24xlarge": ("A10G", 4),
    "g5.48xlarge": ("A10G", 8),
}

# Reverse map built once at import time
GPU_TYPE_TO_INSTANCES: Dict[str, List[tuple]] = _build_gpu_type_to_instances(
    AWS_INSTANCE_TO_GPU
)


def resolve_gpu_type_to_instance(gpu_type: str, min_gpus: int) -> tuple:
    """
    Find the smallest AWS instance with >= min_gpus of the given GPU type.

    Args:
        gpu_type: Normalized GPU name (e.g. "A100", "L40S")
        min_gpus: Minimum number of GPUs required (tp * pp)

    Returns:
        (instance_type, gpu_count) tuple

    Raises:
        ValueError: If no matching instance exists
    """
    from placement.roofline.gpu_specs import normalize_gpu_type

    gpu_type = normalize_gpu_type(gpu_type)

    instances = GPU_TYPE_TO_INSTANCES.get(gpu_type)
    if not instances:
        raise ValueError(
            f"No AWS instance found for GPU type '{gpu_type}'. "
            f"Available GPU types: {sorted(GPU_TYPE_TO_INSTANCES.keys())}"
        )

    for inst, gpu_count in instances:
        if gpu_count >= min_gpus:
            return inst, gpu_count

    max_count = instances[-1][1]
    raise ValueError(
        f"No AWS instance with >= {min_gpus} {gpu_type} GPUs. "
        f"Max available: {max_count} GPUs on {instances[-1][0]}"
    )


def check_user_specified_feasibility(
    model_name: str,
    instance_type: str,
    gpu_count: int,
    tp: int,
    pp: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    batch_size: int = 32,
) -> dict:
    """
    Validate a user-specified GPU/TP/PP config using LLM_placement_solver.

    Uses evaluate_manual_placement() from LLMPlacementSolverWithTP for
    accurate memory feasibility (activation memory, CUDA graphs, sampler,
    allocator fragmentation) and throughput/cost estimation.

    Returns:
        dict with keys: feasible, reason, max_model_len, solution
        On success, solution contains the full evaluate_manual_placement() output.
    """
    from placement.roofline.solver_adapter import PlacementSolverAdapter
    from placement.roofline.gpu_specs import (
        calculate_max_supported_context,
        get_gpu_memory,
        AWS_INSTANCE_GPU_MAP,
    )

    adapter = PlacementSolverAdapter(
        gpu_pool={instance_type: 1},
        cloud_provider="AWS",
    )

    try:
        config_dir = adapter._get_config_dir(model_name)
    except ValueError as e:
        return {
            "feasible": False,
            "reason": str(e),
            "max_model_len": None,
            "solution": None,
        }

    # Build temp files (gpu_pool.csv, network_bandwidth.csv) same as normal solve path
    try:
        gpu_pool_file = adapter._write_gpu_pool_csv_temp()
        network_bandwidth_file = adapter._generate_network_bandwidth_temp(gpu_pool_file)
        cloud_specs_file = adapter.SOLVER_CONFIG_BASE + "/cloud_instances_specs.csv"

        from solver import LLMPlacementSolverWithTP

        solver = LLMPlacementSolverWithTP(
            config_dir=config_dir,
            sequence_length=avg_input_tokens,
            output_length=avg_output_tokens,
            workload_phase="aggregated",
            min_batch_size=batch_size,
            max_batch_size=batch_size,
            cloud_provider="AWS",
            skip_gurobi=True,
            gpu_pool_file=gpu_pool_file,
            network_bandwidth_file=network_bandwidth_file,
            cloud_specs_file=cloud_specs_file,
        )

        # Build the gpu_type key the solver expects (instance_type#0)
        solver_gpu_type = f"{instance_type}#0"

        # Build placement dict for evaluate_manual_placement
        num_layers = solver.config.num_decoder_layers
        layers_per_stage = num_layers // pp
        stages = []
        for i in range(pp):
            start = i * layers_per_stage + 1
            end = (i + 1) * layers_per_stage
            stages.append(
                {
                    "gpu_type": solver_gpu_type,
                    "tp_degree": tp,
                    "start_layer": start,
                    "end_layer": end,
                }
            )

        placement = {
            "batch_size": batch_size,
            "stages": stages,
            "sequence_length": avg_input_tokens,
            "output_length": avg_output_tokens,
            "workload_phase": "aggregated",
        }

        solution = solver.evaluate_manual_placement(placement)

        # Calculate max_model_len using gpu_specs
        gpu_info = AWS_INSTANCE_GPU_MAP.get(instance_type, {})
        gpu_model = gpu_info.get("gpu_model", "")
        max_model_len = calculate_max_supported_context(
            gpu_model=gpu_model,
            num_layers=solver.config.num_decoder_layers,
            layer_weight_gb=solver.config.layer_weight_memory_gb,
            num_kv_heads=solver.config.num_kv_heads,
            d_model=solver.config.d_model,
            num_attention_heads=solver.config.num_attention_heads,
            tp_degree=tp,
            pp_stages=pp,
            batch_size=batch_size,
            bytes_per_element=solver.config.bytes_per_element,
        )

        return {
            "feasible": True,
            "reason": "",
            "max_model_len": max_model_len,
            "solution": solution,
        }

    except ValueError as e:
        return {
            "feasible": False,
            "reason": str(e),
            "max_model_len": None,
            "solution": None,
        }
    except Exception as e:
        return {
            "feasible": False,
            "reason": str(e),
            "max_model_len": None,
            "solution": None,
        }
    finally:
        adapter._cleanup_temp_files()


def quota_to_gpu_pool(
    quota_df: pd.DataFrame,
    region: str = "us-east-1",
    market: str = "on_demand",
    max_instances_per_type: int = 4,
) -> Dict[str, int]:
    """
    Convert Orca quota DataFrame to solver gpu_pool format.

    The quota CSV has vCPU limits per region/market. We convert this to
    instance counts that the solver expects.

    Filters out old/incapable GPUs to keep the search space manageable.

    Args:
        quota_df: DataFrame from load_aws_quota_csv()
        region: AWS region
        market: Market type (on_demand or spot)
        max_instances_per_type: Cap on instances per type to limit search space

    Returns:
        Dict mapping instance_name to count
    """
    # Only include modern GPUs capable of running large LLMs efficiently
    # Exclude: p2 (K80), p3 small, g4ad (Radeon), g4dn small, g5 small, g6 small
    ALLOWED_INSTANCE_PREFIXES = [
        # High-end instances (A100, H100)
        "p4d.",
        "p4de.",
        "p5.",
        # L40S instances (g6e) - good for LLMs
        "g6e.12xlarge",
        "g6e.24xlarge",
        "g6e.48xlarge",
        # A10G instances (g5) - only multi-GPU variants
        "g5.12xlarge",
        "g5.24xlarge",
        "g5.48xlarge",
        # V100 large instances
        "p3.16xlarge",
        "p3dn.",
    ]

    gpu_pool = {}
    col = f"{region}_{market}"

    if col not in quota_df.columns:
        logger.warning(f"[Roofline] Column {col} not found in quota CSV")
        return gpu_pool

    for _, row in quota_df.iterrows():
        instance_type = row.get("Instance_Type", "")
        if not instance_type:
            continue

        # Filter to only allowed instance types
        is_allowed = any(
            instance_type.startswith(prefix) or instance_type == prefix
            for prefix in ALLOWED_INSTANCE_PREFIXES
        )
        if not is_allowed:
            continue

        vcpu_quota = row.get(col, 0)
        if pd.isna(vcpu_quota) or vcpu_quota <= 0:
            continue

        vcpu_per_instance = row.get("vCPU", 0)
        if pd.isna(vcpu_per_instance) or vcpu_per_instance <= 0:
            continue

        # Calculate how many instances are allowed, capped to limit search space
        instance_count = min(
            int(vcpu_quota / vcpu_per_instance), max_instances_per_type
        )
        if instance_count > 0:
            gpu_pool[instance_type] = instance_count

    logger.info(
        f"[Roofline] Filtered quota pool to {len(gpu_pool)} capable instance types"
    )
    return gpu_pool


class RooflineAWSAllocation(VPCMagic):
    """
    Roofline-based AWS GPU allocation using LLM_placement_solver.

    This replaces the LLM-based 3-advisor + C-PMI system with a deterministic
    roofline model approach for GPU/TP/PP selection.
    """

    def __init__(
        self,
        perfdb_dir: str = "./perf_db",
        aws_quota_csv: str = "./quota/aws_gpu_quota_by_region.csv",
        priority: str = "cost_first",
        k_nearest_model_size: int = 1,
        use_quota_pool: bool = True,
    ):
        """
        Initialize the roofline-based allocator.

        Args:
            perfdb_dir: Directory containing performance database (for compatibility)
            aws_quota_csv: Path to AWS quota CSV file
            priority: Optimization priority (cost_first, throughput_first, balanced)
            k_nearest_model_size: Number of GPU types to consider (for compatibility)
            use_quota_pool: If True, use quota CSV to build GPU pool; if False, use solver's default
        """
        self.perfdb_dir = perfdb_dir
        self.aws_quota_csv = aws_quota_csv
        self.priority = priority
        self.k_nearest_model_size = k_nearest_model_size
        self.use_quota_pool = use_quota_pool

        # Load AWS quota data
        try:
            self.quota_df = load_aws_quota_csv(aws_quota_csv)
            logger.info(f"[Roofline] Loaded quota from {aws_quota_csv}")
        except FileNotFoundError:
            logger.warning(f"[Roofline] Quota file not found: {aws_quota_csv}")
            logger.warning("[Roofline] Using default GPU pool")
            self.quota_df = pd.DataFrame()

        # Lazy-load solver adapter
        self._solver_adapter = None

    def _get_solver_adapter(self, gpu_pool: Optional[Dict[str, int]] = None):
        """Get or create solver adapter."""
        if self._solver_adapter is None:
            from placement.roofline.solver_adapter import PlacementSolverAdapter

            self._solver_adapter = PlacementSolverAdapter(
                gpu_pool=gpu_pool,
                cloud_provider="AWS",
            )
        elif gpu_pool is not None:
            # Update GPU pool
            self._solver_adapter.gpu_pool = gpu_pool
        return self._solver_adapter

    def decide(
        self, request: Union[BatchedRequest, OnlineServingRequest]
    ) -> MagicOutput:
        """
        Make allocation decision based on request.

        Args:
            request: BatchedRequest or OnlineServingRequest

        Returns:
            MagicOutput with allocation decision
        """
        if isinstance(request, BatchedRequest):
            return self.process_batch(request)
        else:
            return self._default_online_config(request)

    def process_batch(
        self,
        req: BatchedRequest,
        region: str = "us-east-1",
        market: str = "on_demand",
    ) -> MagicOutput:
        """
        Process batch request using roofline solver.

        Args:
            req: BatchedRequest with job parameters
            region: AWS region
            market: Market type (on_demand or spot)

        Returns:
            MagicOutput with allocation decision
        """
        logger.info(f"[Roofline] Processing batch request for model: {req.model_name}")
        logger.info(
            f"[Roofline] Requests: {req.num_lines}, Tokens: {req.avg_input_tokens}in/{req.avg_output_tokens}out"
        )
        logger.info(f"[Roofline] SLO: {req.slo_deadline_hours} hours")

        # Get GPU pool from quota (if enabled)
        if self.use_quota_pool and not self.quota_df.empty:
            gpu_pool = quota_to_gpu_pool(self.quota_df, region, market)
            logger.info(
                f"[Roofline] Available instances: {list(gpu_pool.keys())[:5]}..."
            )
        else:
            # Use a small, fast default GPU pool for testing
            # Only includes modern GPUs capable of running large LLMs
            gpu_pool = {
                "g6e.48xlarge": 2,  # 8x L40S (48GB each)
                "g6e.12xlarge": 4,  # 4x L40S
                "p4d.24xlarge": 2,  # 8x A100 (40GB)
                "p4de.24xlarge": 1,  # 8x A100 (80GB)
                "p5.48xlarge": 1,  # 8x H100
            }
            logger.info(
                f"[Roofline] Using fast default GPU pool: {list(gpu_pool.keys())}"
            )

        # Create solver input
        from placement.roofline.solver_adapter import (
            PlacementSolverAdapter,
            create_solver_input_from_request,
        )

        solver_input = create_solver_input_from_request(
            model_name=req.model_name,
            avg_input_tokens=req.avg_input_tokens,
            avg_output_tokens=req.avg_output_tokens,
            num_requests=req.num_lines,
            slo_deadline_hours=req.slo_deadline_hours,
        )

        # Run solver
        adapter = self._get_solver_adapter(gpu_pool)
        result = adapter.solve(solver_input)

        if not result.success:
            logger.warning(f"[Roofline] Solver failed: {result.error_message}")
            logger.warning("[Roofline] Using fallback configuration")
            return self._fallback_config(req)

        # Log result
        logger.info("[Roofline] Solution found:")
        logger.info(f"  Instance: {result.instance_family} x {result.num_instances}")
        logger.info(
            f"  GPU: {result.gpu_model}, TP={result.tp_degree}, PP={result.pp_stages}"
        )
        logger.info(f"  Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
        logger.info(
            f"  Cost: ${result.cost_per_hour:.2f}/hour, ${result.cost_per_million_tokens:.2f}/M tokens"
        )
        logger.info(f"  Max context: {result.max_supported_context} tokens")

        # Convert to MagicOutput
        # Note: solver's num_instances is TOTAL nodes (already includes PP stages)
        # replicas = num_instances / pp_stages = number of data-parallel pipeline replicas
        # num_nodes in server.py = replicas * pp_size, so we need replicas to be correct
        data_parallel_replicas = max(1, result.num_instances // result.pp_stages)
        return MagicOutput(
            decision_id=f"gangmuk-{uuid.uuid4()}",
            engine=req.engine or "vllm",
            instance_type=result.instance_family,
            replicas=data_parallel_replicas,
            tp_size=result.tp_degree,
            pp_size=result.pp_stages,
            max_model_len=result.max_supported_context,
        )

    def _fallback_config(self, req: BatchedRequest) -> MagicOutput:
        """
        Generate fallback configuration when solver fails.

        Args:
            req: Original request

        Returns:
            MagicOutput with conservative defaults
        """
        return MagicOutput(
            decision_id=f"gangmuk-{uuid.uuid4()}",
            engine=req.engine or "vllm",
            instance_type="g6e.12xlarge",
            replicas=1,
            tp_size=4,  # TP=4 is safer - works with most attention head counts
            pp_size=1,
            max_model_len=8192,  # Conservative fallback
        )

    def _default_online_config(self, req: OnlineServingRequest) -> MagicOutput:
        """
        Generate default configuration for online serving.

        Args:
            req: Online serving request

        Returns:
            MagicOutput with online serving defaults
        """
        return MagicOutput(
            decision_id=f"gangmuk-{uuid.uuid4()}",
            engine=req.engine or "vllm",
            instance_type="g6e.12xlarge",
            replicas=1,
            tp_size=4,
            pp_size=1,
            max_model_len=8192,  # Conservative default
        )

    def process_batch_multi(
        self,
        req: BatchedRequest,
        region: str = "us-east-1",
        market: str = "on_demand",
        top_k: int = 5,
    ) -> List[MagicOutput]:
        """
        Process batch request and return multiple fallback solutions.

        Used for retry logic - if the first solution fails in all regions,
        try the next solution with a different instance type.

        Args:
            req: BatchedRequest with job parameters
            region: AWS region
            market: Market type (on_demand or spot)
            top_k: Number of solutions to return

        Returns:
            List of MagicOutput solutions sorted by cost (best first)
        """
        logger.info(
            f"[Roofline] Processing batch request (multi) for model: {req.model_name}"
        )

        # Get GPU pool from quota (if enabled)
        if self.use_quota_pool and not self.quota_df.empty:
            gpu_pool = quota_to_gpu_pool(self.quota_df, region, market)
        else:
            gpu_pool = {
                "g6e.48xlarge": 2,
                "g6e.12xlarge": 4,
                "p4d.24xlarge": 2,
                "p4de.24xlarge": 1,
                "p5.48xlarge": 1,
            }

        # Create solver input
        from placement.roofline.solver_adapter import (
            PlacementSolverAdapter,
            create_solver_input_from_request,
        )

        solver_input = create_solver_input_from_request(
            model_name=req.model_name,
            avg_input_tokens=req.avg_input_tokens,
            avg_output_tokens=req.avg_output_tokens,
            num_requests=req.num_lines,
            slo_deadline_hours=req.slo_deadline_hours,
        )

        # Run solver for multiple solutions
        adapter = self._get_solver_adapter(gpu_pool)
        results = adapter.solve_multi(solver_input, top_k=top_k)

        if not results:
            logger.warning("[Roofline] No solutions found, returning fallback")
            return [self._fallback_config(req)]

        # Convert to MagicOutput list
        outputs = []
        decision_id_base = f"gangmuk-{uuid.uuid4()}"

        for i, result in enumerate(results):
            data_parallel_replicas = max(1, result.num_instances // result.pp_stages)
            output = MagicOutput(
                decision_id=f"{decision_id_base}-opt{i + 1}"
                if i > 0
                else decision_id_base,
                engine=req.engine or "vllm",
                instance_type=result.instance_family,
                replicas=data_parallel_replicas,
                tp_size=result.tp_degree,
                pp_size=result.pp_stages,
                max_model_len=result.max_supported_context,
            )
            outputs.append(output)
            logger.info(
                f"[Roofline] Solution {i + 1}: {result.instance_family} TP={result.tp_degree} PP={result.pp_stages} ${result.cost_per_million_tokens:.2f}/M"
            )

        return outputs
