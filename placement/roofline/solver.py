"""
Roofline-based GPU placement solver for LLM inference.

This solver replaces the LLM-based 3-advisor + C-PMI system with a
deterministic, roofline-model-based approach.

Key features:
- Memory feasibility checks (weights + KV cache + activations)
- Roofline throughput estimation (compute vs memory bound)
- Cost-throughput optimization
- Tensor parallelism and pipeline parallelism support

Ported from: ../LLM_placement_solver/solver.py
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .gpu_specs import get_gpu_specs, get_gpu_memory, normalize_gpu_type, GPU_SPECS
from .throughput import ThroughputCalculator, ThroughputResult
from .model_arch import (
    ModelArchitecture,
    get_model_architecture,
    get_model_architecture_or_estimate,
)


class OptimizationPriority(Enum):
    """Optimization priority for placement decisions."""
    THROUGHPUT_FIRST = "throughput_first"  # Maximize throughput, then minimize cost
    COST_FIRST = "cost_first"              # Minimize cost while meeting SLO
    BALANCED = "balanced"                   # Balance throughput and cost


@dataclass
class PlacementConfig:
    """
    Configuration for placement solver.

    Contains workload parameters and optimization settings.
    """
    # Workload parameters
    model_name: str
    avg_input_tokens: int
    avg_output_tokens: int
    num_requests: int
    slo_deadline_hours: float

    # Optimization settings
    priority: OptimizationPriority = OptimizationPriority.COST_FIRST
    guard_factor: float = 0.10  # SLO safety margin (10%)

    # Search space limits
    max_tp: int = 8
    max_pp: int = 4
    max_replicas: int = 50
    tp_degrees: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pp_degrees: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])

    # Model parameters (auto-filled from model_name if not provided)
    model_arch: Optional[ModelArchitecture] = None

    # Phase for throughput calculation
    workload_phase: str = "aggregated"  # prefill, decode, or aggregated

    # Real-world efficiency factor (accounts for framework overhead)
    real_world_efficiency: float = 0.30

    def __post_init__(self):
        """Auto-fill model architecture if not provided."""
        if self.model_arch is None:
            self.model_arch = get_model_architecture_or_estimate(self.model_name)

    @property
    def total_tokens(self) -> int:
        """Total tokens to process."""
        return self.num_requests * (self.avg_input_tokens + self.avg_output_tokens)

    @property
    def effective_slo_hours(self) -> float:
        """SLO with guard factor applied."""
        return self.slo_deadline_hours * (1.0 - self.guard_factor)


@dataclass
class PlacementCandidate:
    """A candidate GPU configuration."""
    gpu_type: str
    tp: int
    pp: int
    replicas: int
    batch_size: int

    # Computed fields
    gpus_needed: int = 0
    throughput_tokens_per_sec: float = 0.0
    runtime_hours: float = 0.0
    gpu_hours: float = 0.0
    cost_per_hour: float = 0.0
    total_cost: float = 0.0

    # Roofline analysis
    regime: str = ""  # COMPUTE_BOUND or MEMORY_BOUND
    bottleneck: str = ""

    # Feasibility
    is_feasible: bool = True
    infeasibility_reason: str = ""

    def __post_init__(self):
        self.gpus_needed = self.tp * self.pp * self.replicas


@dataclass
class PlacementResult:
    """Result of placement solver."""
    # Selected configuration
    gpu_type: str
    tp: int
    pp: int
    replicas: int
    batch_size: int

    # Performance metrics
    gpus_needed: int
    throughput_tokens_per_sec: float
    runtime_hours: float
    gpu_hours: float

    # Cost metrics
    cost_per_hour: float
    total_cost: float
    cost_per_million_tokens: float

    # Analysis
    regime: str
    bottleneck: str
    meets_slo: bool

    # All candidates considered (for debugging)
    all_candidates: List[PlacementCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "gpu_type": self.gpu_type,
            "tp": self.tp,
            "pp": self.pp,
            "replicas": self.replicas,
            "batch_size": self.batch_size,
            "gpus_needed": self.gpus_needed,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "runtime_hours": self.runtime_hours,
            "gpu_hours": self.gpu_hours,
            "cost_per_hour": self.cost_per_hour,
            "total_cost": self.total_cost,
            "cost_per_million_tokens": self.cost_per_million_tokens,
            "regime": self.regime,
            "bottleneck": self.bottleneck,
            "meets_slo": self.meets_slo,
        }


class RooflinePlacementSolver:
    """
    Roofline-based GPU placement solver.

    Enumerates valid configurations, calculates throughput using roofline model,
    and selects optimal configuration based on cost/throughput tradeoff.
    """

    # Default GPU costs ($/hour per GPU)
    DEFAULT_GPU_COSTS: Dict[str, float] = {
        'H200': 8.00,
        'H100': 4.00,
        'A100': 2.50,
        'L40S': 1.20,
        'L40': 1.00,
        'L4': 0.50,
        'A40': 1.00,
        'A10': 0.60,
        'A10G': 0.60,
        'V100': 1.50,
        'T4': 0.35,
    }

    def __init__(
        self,
        available_gpus: Dict[str, int],
        gpu_costs: Optional[Dict[str, float]] = None,
        nvlink_bandwidth_gbps: float = 300.0,
    ):
        """
        Initialize the solver.

        Args:
            available_gpus: Dict mapping GPU type to count (e.g., {"L40S": 8, "A100": 4})
            gpu_costs: Dict mapping GPU type to cost per hour ($/hour per GPU)
            nvlink_bandwidth_gbps: NVLink bandwidth for TP communication
        """
        self.available_gpus = {
            normalize_gpu_type(k): v for k, v in available_gpus.items()
        }
        self.gpu_costs = gpu_costs or self.DEFAULT_GPU_COSTS
        self.nvlink_bandwidth_gbps = nvlink_bandwidth_gbps

    def solve(self, config: PlacementConfig) -> Optional[PlacementResult]:
        """
        Solve for optimal GPU placement.

        Args:
            config: Placement configuration with workload parameters

        Returns:
            PlacementResult with optimal configuration, or None if no feasible solution
        """
        if config.model_arch is None:
            raise ValueError(f"Unknown model architecture for: {config.model_name}")

        # Generate all candidates
        candidates = self._enumerate_candidates(config)

        if not candidates:
            return None

        # Filter feasible candidates
        feasible = [c for c in candidates if c.is_feasible]

        if not feasible:
            # Return best infeasible for debugging
            return None

        # Filter by SLO
        slo_compliant = [c for c in feasible if c.runtime_hours <= config.effective_slo_hours]

        if not slo_compliant:
            # Find closest to SLO
            closest = min(feasible, key=lambda c: c.runtime_hours)
            return self._make_result(closest, config, meets_slo=False, all_candidates=candidates)

        # Select optimal based on priority
        if config.priority == OptimizationPriority.THROUGHPUT_FIRST:
            # Maximize throughput, then minimize cost
            optimal = max(slo_compliant, key=lambda c: (c.throughput_tokens_per_sec, -c.gpu_hours))
        elif config.priority == OptimizationPriority.COST_FIRST:
            # Minimize GPU-hours (cost proxy)
            optimal = min(slo_compliant, key=lambda c: (c.gpu_hours, c.replicas, -c.throughput_tokens_per_sec))
        else:  # BALANCED
            # Balance throughput and cost
            optimal = min(slo_compliant, key=lambda c: (c.gpu_hours / max(c.throughput_tokens_per_sec, 1), c.replicas))

        return self._make_result(optimal, config, meets_slo=True, all_candidates=candidates)

    def _enumerate_candidates(self, config: PlacementConfig) -> List[PlacementCandidate]:
        """
        Enumerate all valid placement candidates.

        Args:
            config: Placement configuration

        Returns:
            List of PlacementCandidate objects
        """
        candidates = []
        arch = config.model_arch

        for gpu_type, gpu_count in self.available_gpus.items():
            if gpu_count <= 0:
                continue

            gpu_memory = get_gpu_memory(gpu_type)
            gpu_cost = self.gpu_costs.get(gpu_type, 1.0)

            for tp in config.tp_degrees:
                if tp > config.max_tp or tp > gpu_count:
                    continue

                for pp in config.pp_degrees:
                    if pp > config.max_pp:
                        continue

                    gpus_per_replica = tp * pp
                    if gpus_per_replica > gpu_count:
                        continue

                    # Check memory feasibility for this TP/PP
                    layers_per_stage = arch.num_hidden_layers // pp
                    if layers_per_stage < 1:
                        continue

                    memory_feasible, mem_reason = self._check_memory_feasibility(
                        arch, gpu_type, gpu_memory, tp, pp,
                        config.avg_input_tokens, config.avg_output_tokens,
                        config.batch_sizes[-1]  # Check with max batch size
                    )

                    for batch_size in config.batch_sizes:
                        # Calculate throughput for this configuration
                        throughput_result = ThroughputCalculator.calculate_throughput(
                            gpu_type=gpu_type,
                            seq_len=config.avg_input_tokens,
                            batch_size=batch_size,
                            num_layers=layers_per_stage,
                            d_model=arch.d_model,
                            d_hidden=arch.d_hidden,
                            tp_degree=tp,
                            bytes_per_element=2,  # FP16
                            nvlink_bw_gbps=self.nvlink_bandwidth_gbps,
                            phase=config.workload_phase,
                            output_length=config.avg_output_tokens,
                            num_attention_heads=arch.num_attention_heads,
                            num_kv_heads=arch.num_kv_heads,
                        )

                        # Apply pipeline efficiency for PP > 1
                        pipeline_efficiency = self._calculate_pipeline_efficiency(pp, batch_size)

                        # Apply real-world efficiency
                        effective_throughput = (
                            throughput_result.throughput_tokens_per_sec *
                            pipeline_efficiency *
                            config.real_world_efficiency
                        )

                        # Calculate replicas needed to meet SLO
                        for replicas in range(1, min(config.max_replicas + 1, gpu_count // gpus_per_replica + 1)):
                            total_throughput = effective_throughput * replicas
                            gpus_needed = gpus_per_replica * replicas

                            if gpus_needed > gpu_count:
                                break

                            if total_throughput <= 0:
                                continue

                            runtime_hours = config.total_tokens / (total_throughput * 3600)
                            gpu_hours = gpus_needed * runtime_hours
                            cost_per_hour = gpus_needed * gpu_cost
                            total_cost = cost_per_hour * runtime_hours

                            candidate = PlacementCandidate(
                                gpu_type=gpu_type,
                                tp=tp,
                                pp=pp,
                                replicas=replicas,
                                batch_size=batch_size,
                                gpus_needed=gpus_needed,
                                throughput_tokens_per_sec=total_throughput,
                                runtime_hours=runtime_hours,
                                gpu_hours=gpu_hours,
                                cost_per_hour=cost_per_hour,
                                total_cost=total_cost,
                                regime=throughput_result.regime,
                                bottleneck=throughput_result.bottleneck,
                                is_feasible=memory_feasible,
                                infeasibility_reason="" if memory_feasible else mem_reason,
                            )
                            candidates.append(candidate)

        return candidates

    def _check_memory_feasibility(
        self,
        arch: ModelArchitecture,
        gpu_type: str,
        gpu_memory_gb: float,
        tp: int,
        pp: int,
        input_length: int,
        output_length: int,
        batch_size: int,
    ) -> Tuple[bool, str]:
        """
        Check if configuration fits in GPU memory.

        Args:
            arch: Model architecture
            gpu_type: GPU type
            gpu_memory_gb: GPU memory in GB
            tp: Tensor parallelism degree
            pp: Pipeline parallelism degree
            input_length: Input sequence length
            output_length: Output sequence length
            batch_size: Batch size

        Returns:
            Tuple of (is_feasible, reason_if_not)
        """
        layers_per_stage = arch.num_hidden_layers // pp
        d_head = arch.d_model / arch.num_attention_heads
        kv_dim = arch.num_kv_heads * d_head

        # Weight memory per layer (sharded by TP)
        weight_per_layer_gb = arch.layer_weight_memory_gb / tp
        total_weight_gb = weight_per_layer_gb * layers_per_stage

        # KV cache memory (sharded by TP)
        kv_cache_len = input_length + output_length
        kv_cache_per_layer_bytes = 2 * batch_size * kv_cache_len * (kv_dim / tp) * 2  # 2 for K+V, 2 for FP16
        kv_cache_gb = (kv_cache_per_layer_bytes * layers_per_stage) / (1024 ** 3)

        # Activation memory (conservative estimate)
        activation_bytes = batch_size * input_length * arch.d_model * 2  # FP16
        activation_gb = (activation_bytes * 1.5) / (1024 ** 3)  # 1.5x overhead

        # Total memory needed
        total_memory_gb = total_weight_gb + kv_cache_gb + activation_gb

        # Memory buffer (20% safety margin)
        available_memory_gb = gpu_memory_gb * 0.80

        if total_memory_gb > available_memory_gb:
            return False, (
                f"Memory overflow: {total_memory_gb:.1f}GB needed > "
                f"{available_memory_gb:.1f}GB available "
                f"(weights={total_weight_gb:.1f}GB, KV={kv_cache_gb:.1f}GB, act={activation_gb:.1f}GB)"
            )

        return True, ""

    def _calculate_pipeline_efficiency(self, pp: int, batch_size: int, micro_batch_size: int = 8) -> float:
        """
        Calculate pipeline efficiency accounting for bubble overhead.

        Args:
            pp: Pipeline parallelism degree
            batch_size: Batch size
            micro_batch_size: Micro-batch size for pipeline

        Returns:
            Pipeline efficiency factor (0-1)
        """
        if pp == 1:
            return 1.0

        # Number of micro-batches
        num_micro_batches = max(1, batch_size // micro_batch_size)

        # Pipeline efficiency formula:
        # efficiency ≈ num_micro_batches / (num_micro_batches + pp - 1)
        ideal_efficiency = num_micro_batches / (num_micro_batches + pp - 1)

        # Additional scheduling overhead
        scheduling_overhead = 0.10

        return max(0.50, ideal_efficiency * (1.0 - scheduling_overhead))

    def _make_result(
        self,
        candidate: PlacementCandidate,
        config: PlacementConfig,
        meets_slo: bool,
        all_candidates: List[PlacementCandidate],
    ) -> PlacementResult:
        """Create PlacementResult from candidate."""
        cost_per_million = (
            candidate.total_cost / (config.total_tokens / 1_000_000)
            if config.total_tokens > 0 else 0.0
        )

        return PlacementResult(
            gpu_type=candidate.gpu_type,
            tp=candidate.tp,
            pp=candidate.pp,
            replicas=candidate.replicas,
            batch_size=candidate.batch_size,
            gpus_needed=candidate.gpus_needed,
            throughput_tokens_per_sec=candidate.throughput_tokens_per_sec,
            runtime_hours=candidate.runtime_hours,
            gpu_hours=candidate.gpu_hours,
            cost_per_hour=candidate.cost_per_hour,
            total_cost=candidate.total_cost,
            cost_per_million_tokens=cost_per_million,
            regime=candidate.regime,
            bottleneck=candidate.bottleneck,
            meets_slo=meets_slo,
            all_candidates=all_candidates,
        )


def solve_placement(
    model_name: str,
    num_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    slo_deadline_hours: float,
    available_gpus: Dict[str, int],
    gpu_costs: Optional[Dict[str, float]] = None,
    priority: str = "cost_first",
) -> Optional[PlacementResult]:
    """
    Convenience function for solving GPU placement.

    Args:
        model_name: Model name (e.g., "llama-3-70b")
        num_requests: Number of requests to process
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        slo_deadline_hours: SLO deadline in hours
        available_gpus: Dict mapping GPU type to count
        gpu_costs: Optional dict mapping GPU type to cost per hour
        priority: Optimization priority ("throughput_first", "cost_first", "balanced")

    Returns:
        PlacementResult or None if no feasible solution
    """
    priority_enum = {
        "throughput_first": OptimizationPriority.THROUGHPUT_FIRST,
        "cost_first": OptimizationPriority.COST_FIRST,
        "balanced": OptimizationPriority.BALANCED,
    }.get(priority.lower(), OptimizationPriority.COST_FIRST)

    config = PlacementConfig(
        model_name=model_name,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        num_requests=num_requests,
        slo_deadline_hours=slo_deadline_hours,
        priority=priority_enum,
    )

    solver = RooflinePlacementSolver(
        available_gpus=available_gpus,
        gpu_costs=gpu_costs,
    )

    return solver.solve(config)
