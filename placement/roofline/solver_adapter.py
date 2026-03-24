"""
Adapter to integrate LLM_placement_solver with Orca.

This module provides a clean interface between Orca's BatchedRequest and
the LLM_placement_solver's LLMPlacementSolverWithTP.

Flow when custom GPU pool is provided:
1. Write gpu_pool.csv to a temp file
2. Generate network_bandwidth.csv to a temp file
3. Pass paths directly to solver constructor (thread-safe, no chdir needed)
4. Clean up temp files after solve completes

The solver accepts explicit paths for config files, falling back to defaults
if not provided.
"""

import os
import sys
import subprocess
import tempfile
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Set

import pandas as pd

# Add LLM_placement_solver to path (submodule in repo root)
LLM_SOLVER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../LLM_placement_solver")
)
if LLM_SOLVER_PATH not in sys.path:
    sys.path.insert(0, LLM_SOLVER_PATH)

from solver import LLMPlacementSolverWithTP

logger = logging.getLogger(__name__)


def load_supported_instances(
    cloud_specs_path: str, cloud_provider: str = "AWS"
) -> Set[str]:
    """
    Load the set of supported instance names from cloud_instances_specs.csv.

    Args:
        cloud_specs_path: Path to cloud_instances_specs.csv
        cloud_provider: Cloud provider to filter by (default: AWS)

    Returns:
        Set of supported instance names
    """
    try:
        df = pd.read_csv(cloud_specs_path)
        # Filter by cloud provider
        provider_df = df[df["Cloud Provider"] == cloud_provider]
        return set(provider_df["Instance Name"].tolist())
    except Exception as e:
        logger.warning(f"[SolverAdapter] Failed to load cloud specs: {e}")
        return set()


@dataclass
class SolverInput:
    """Input parameters for the placement solver."""

    model_name: str
    sequence_length: int  # avg_input_tokens
    output_length: int  # avg_output_tokens
    total_tokens: int  # num_requests * (input + output)
    slo_hours: float  # slo_deadline_hours
    num_replicas: int = 1  # number of data-parallel replicas sharing the workload

    # Max token limits for feasibility checking (0 = use avg as fallback)
    max_input_tokens: int = 0
    max_output_tokens: int = 0

    # vLLM scheduler parameter: max concurrent sequences (decode batch size)
    max_num_seqs: int = 256

    # vLLM max_num_batched_tokens: max tokens per prefill iteration (0 = fall back to max_model_len)
    max_num_batched_tokens: int = 16384
    # vLLM gpu_memory_utilization: fraction of GPU memory for model/KV cache
    gpu_memory_utilization: float = 0.90

    # Solver log level (None = info)
    log_level: str = "info"

    # Fixed parameters (per user requirements)
    workload_phase: str = "aggregated"
    optimization_priority: str = "cost_first"


@dataclass
class SolverOutput:
    """Output from the placement solver."""

    success: bool

    # Placement configuration
    instance_family: str  # e.g., "p4de.24xlarge"
    num_instances: int
    tp_degree: int
    pp_stages: int
    layers_per_stage: int
    max_concurrent_sequences: int  # derived from KV pool, replaces static batch_size

    # Performance metrics
    throughput_tokens_per_sec: float
    cost_per_hour: float
    cost_per_million_tokens: float

    # GPU info
    gpu_model: str  # e.g., "A100"
    gpus_per_instance: int
    total_gpus: int

    # Memory constraints - max context length this config can actually support
    # This should be passed to vLLM as --max-model-len to avoid KV cache OOM
    max_supported_context: int = 8192

    # SLO
    estimated_runtime_hours: float = 0.0
    meets_slo: bool = True

    # Error info (if success=False)
    error_message: str = ""

    # Solver log: full solving process output (always populated)
    solve_log: str = ""

    # Rank in sorted solutions (1 = best)
    rank: int = 1


class PlacementSolverAdapter:
    """
    Adapter between Orca and LLM_placement_solver.

    Uses the actual solver's solve_homogeneous() method which:
    - Doesn't require Gurobi (fast enumeration)
    - Uses roofline model for throughput estimation
    - Considers memory constraints (KV cache, activations)
    - Optimizes for cost/throughput based on priority
    """

    # Base path for solver configs
    SOLVER_CONFIG_BASE = os.path.join(LLM_SOLVER_PATH, "config")

    # Model name normalization mapping
    MODEL_NAME_MAP = {
        # Llama 3 variants
        "llama-3-70b": "llama3-70b",
        "llama3-70b": "llama3-70b",
        "meta-llama/llama-3-70b": "llama3-70b",
        "meta-llama/Meta-Llama-3-70B": "llama3-70b",
        "meta-llama/Meta-Llama-3-70B-Instruct": "llama3-70b",
        "llama-3-8b": "llama3-8b",
        "llama3-8b": "llama3-8b",
        "meta-llama/llama-3-8b": "llama3-8b",
        "meta-llama/Meta-Llama-3-8B": "llama3-8b",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b",
        # DeepSeek (same arch as Llama-3-70B for now)
        "deepseek-r1-70b": "llama3-70b",
        "deepseek-ai/deepseek-r1-70b": "llama3-70b",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "llama3-70b",
        # Qwen 2.5 variants
        "qwen2.5-7b": "qwen2.5-7b",
        "Qwen/Qwen2.5-7B": "qwen2.5-7b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
        "qwen2.5-32b": "qwen2.5-32b",
        "Qwen/Qwen2.5-32B": "qwen2.5-32b",
        "Qwen/Qwen2.5-32B-Instruct": "qwen2.5-32b",
        "qwen2.5-72b": "qwen2.5-72b",
        "Qwen/Qwen2.5-72B": "qwen2.5-72b",
        "Qwen/Qwen2.5-72B-Instruct": "qwen2.5-72b",
        # Qwen 2.5 MoE (57B total, 14B active)
        "qwen2.5-a14b": "qwen2.5-a14b",
        "Qwen/Qwen2.5-A14B": "qwen2.5-a14b",
        "Qwen/Qwen2.5-A14B-Instruct": "qwen2.5-a14b",
        # Qwen 3 dense
        "qwen3-32b": "qwen3-32b",
        "Qwen/Qwen3-32B": "qwen3-32b",
        "Qwen/Qwen3-32B-Instruct": "qwen3-32b",
        # Qwen 3 MoE (235B total, 22B active)
        "qwen3-235b-a22b": "qwen3-235b-a22b",
        "Qwen/Qwen3-235B-A22B": "qwen3-235b-a22b",
    }

    def __init__(
        self,
        gpu_pool: Optional[Dict[str, int]] = None,
        cloud_provider: str = "AWS",
    ):
        """
        Initialize the adapter.

        Args:
            gpu_pool: Available instances as {instance_name: count}
                      If None, uses default gpu_pool.csv from solver
            cloud_provider: Cloud provider for pricing (AWS, GCP, Azure, etc.)
        """
        self.gpu_pool = gpu_pool
        self.cloud_provider = cloud_provider
        self._temp_files: List[str] = []

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to match solver config directory."""
        # Direct lookup
        if model_name in self.MODEL_NAME_MAP:
            return self.MODEL_NAME_MAP[model_name]

        # Try lowercase
        lower = model_name.lower()
        if lower in self.MODEL_NAME_MAP:
            return self.MODEL_NAME_MAP[lower]

        # Try extracting base name
        for key, value in self.MODEL_NAME_MAP.items():
            if key.lower() in lower or lower in key.lower():
                return value

        # Return as-is and hope it matches a config directory
        return model_name.replace("/", "-").lower()

    def _get_config_dir(self, model_name: str) -> str:
        """Get config directory for model."""
        normalized = self._normalize_model_name(model_name)
        config_dir = os.path.join(self.SOLVER_CONFIG_BASE, normalized)

        if not os.path.exists(config_dir):
            available = [
                d
                for d in os.listdir(self.SOLVER_CONFIG_BASE)
                if os.path.isdir(os.path.join(self.SOLVER_CONFIG_BASE, d))
                and not d.startswith(".")
                and d != "outdated"
            ]
            raise ValueError(
                f"No config found for model '{model_name}' (normalized: '{normalized}'). "
                f"Available configs: {available}"
            )

        return config_dir

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for path in self._temp_files:
            if os.path.exists(path):
                os.remove(path)
        self._temp_files.clear()

    def _write_gpu_pool_csv_temp(self) -> str:
        """
        Write gpu_pool.csv to a temp file.

        Returns:
            Path to the temp file (caller should clean up)
        """
        fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="gpu_pool_")
        self._temp_files.append(temp_path)

        with os.fdopen(fd, "w") as f:
            f.write("instance_name,count\n")
            for instance, count in self.gpu_pool.items():
                f.write(f"{instance},{count}\n")

        logger.info(
            f"[SolverAdapter] Wrote temp gpu_pool.csv with {len(self.gpu_pool)} instance types"
        )
        return temp_path

    def _generate_network_bandwidth_temp(self, gpu_pool_path: str) -> str:
        """
        Generate network_bandwidth.csv to a temp file.

        Args:
            gpu_pool_path: Path to the gpu_pool.csv file to use

        Returns:
            Path to the generated temp network_bandwidth.csv
        """
        cloud_specs_path = os.path.join(
            self.SOLVER_CONFIG_BASE, "cloud_instances_specs.csv"
        )
        generator_script = os.path.join(
            self.SOLVER_CONFIG_BASE, "generate_network_bandwidth.py"
        )

        # Create temp file for output
        fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="network_bandwidth_")
        os.close(fd)  # Close fd, generator script will write to the path
        self._temp_files.append(temp_path)

        # Run the generator script
        cmd = [
            sys.executable,
            generator_script,
            "--gpu-pool",
            gpu_pool_path,
            "--cloud-specs",
            cloud_specs_path,
            "--output",
            temp_path,
            "--cloud-provider",
            self.cloud_provider,
        ]

        logger.info(f"[SolverAdapter] Generating network_bandwidth.csv to temp file...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.SOLVER_CONFIG_BASE,
            )
            if result.returncode != 0:
                logger.error(
                    f"[SolverAdapter] generate_network_bandwidth.py failed: {result.stderr}"
                )
                raise RuntimeError(
                    f"Network bandwidth generation failed: {result.stderr}"
                )

            logger.info(f"[SolverAdapter] {result.stdout.strip()}")

        except Exception as e:
            logger.error(f"[SolverAdapter] Failed to generate network bandwidth: {e}")
            raise

        return temp_path

    def solve(self, input: SolverInput) -> SolverOutput:
        """
        Solve placement using LLM_placement_solver.

        When custom GPU pool is provided:
        1. Write gpu_pool.csv to temp file
        2. Generate network_bandwidth.csv to temp file
        3. Pass paths directly to solver (no chdir needed!)

        Args:
            input: Solver input parameters

        Returns:
            SolverOutput with placement configuration
        """
        _solver_debug_handler = None
        try:
            # Set solver log level based on request
            solver_logger = logging.getLogger("solver")
            _level_map = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
            _requested_level = _level_map.get(input.log_level.lower(), logging.INFO)
            solver_logger.setLevel(_requested_level)

            # If debug requested, add a temporary console handler so debug lines
            # are printed (the root handler from basicConfig filters at INFO).
            if _requested_level == logging.DEBUG:
                _solver_debug_handler = logging.StreamHandler()
                _solver_debug_handler.setLevel(logging.DEBUG)
                _solver_debug_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                solver_logger.addHandler(_solver_debug_handler)
                solver_logger.propagate = False  # Avoid duplicate INFO lines from root

            config_dir = self._get_config_dir(input.model_name)
            logger.info(f"[SolverAdapter] Using config: {config_dir}")

            # Prepare optional path arguments for solver
            gpu_pool_file = None
            network_bandwidth_file = None
            cloud_specs_file = os.path.join(
                self.SOLVER_CONFIG_BASE, "cloud_instances_specs.csv"
            )

            # Handle custom GPU pool
            if self.gpu_pool:
                # Filter to only supported instances (those in cloud_instances_specs.csv)
                supported = load_supported_instances(
                    cloud_specs_file, self.cloud_provider
                )

                if supported:
                    original_count = len(self.gpu_pool)
                    self.gpu_pool = {
                        k: v for k, v in self.gpu_pool.items() if k in supported
                    }
                    filtered_count = len(self.gpu_pool)
                    logger.info(
                        f"[SolverAdapter] Filtered GPU pool: {original_count} -> {filtered_count} instances (only those in cloud_instances_specs.csv)"
                    )

                if not self.gpu_pool:
                    return SolverOutput(
                        success=False,
                        instance_family="",
                        num_instances=0,
                        tp_degree=0,
                        pp_stages=0,
                        layers_per_stage=0,
                        max_concurrent_sequences=0,
                        throughput_tokens_per_sec=0,
                        cost_per_hour=0,
                        cost_per_million_tokens=0,
                        gpu_model="",
                        gpus_per_instance=0,
                        total_gpus=0,
                        error_message="No supported instances in GPU pool after filtering",
                    )

                logger.info(
                    f"[SolverAdapter] Using custom GPU pool: {list(self.gpu_pool.keys())}"
                )

                # Write gpu_pool.csv to temp file
                gpu_pool_file = self._write_gpu_pool_csv_temp()

                # Generate network_bandwidth.csv to temp file
                network_bandwidth_file = self._generate_network_bandwidth_temp(
                    gpu_pool_file
                )

            # Initialize solver with explicit paths (no chdir needed!)
            solver = LLMPlacementSolverWithTP(
                config_dir=config_dir,
                sequence_length=input.sequence_length,
                output_length=input.output_length,
                workload_phase=input.workload_phase,
                min_batch_size=1,  # batch_size no longer meaningful; kept for MILP compat
                max_batch_size=1,
                cloud_provider=self.cloud_provider,
                skip_gurobi=True,  # Use homogeneous solver (no Gurobi needed)
                # Pass explicit paths (None = use solver defaults)
                gpu_pool_file=gpu_pool_file,
                network_bandwidth_file=network_bandwidth_file,
                cloud_specs_file=cloud_specs_file,
                max_input_tokens=input.max_input_tokens,
                max_output_tokens=input.max_output_tokens,
                max_num_seqs=input.max_num_seqs,
                max_num_batched_tokens=input.max_num_batched_tokens,
                gpu_memory_utilization=input.gpu_memory_utilization,
                # SLO: pass per-replica workload and deadline so solver filters by feasibility
                # Each replica processes total_tokens/num_replicas (work is split evenly)
                total_tokens_to_process=input.total_tokens // max(1, input.num_replicas),
                max_total_runtime_hours=input.slo_hours,
            )

            # Solve using homogeneous solver (fast enumeration)
            success = solver.solve_homogeneous()

            if not success or solver.solution is None:
                return SolverOutput(
                    success=False,
                    instance_family="",
                    num_instances=0,
                    tp_degree=0,
                    pp_stages=0,
                    layers_per_stage=0,
                    max_concurrent_sequences=0,
                    throughput_tokens_per_sec=0,
                    cost_per_hour=0,
                    cost_per_million_tokens=0,
                    gpu_model="",
                    gpus_per_instance=0,
                    total_gpus=0,
                    error_message="No valid placement found",
                    solve_log=getattr(solver, 'solve_log', ''),
                )

            sol = solver.solution

            # Extract homogeneous config (nested under 'homogeneous_config' key)
            homo_cfg = sol.get("homogeneous_config", {})

            # Get GPU model from first assignment
            gpu_model = ""
            gpus_per_instance = 8
            assignments = sol.get("gpu_assignments", [])
            if assignments:
                gpu_type_key = assignments[0].get("gpu_type", "")
                # gpu_type is like "g6e.48xlarge#0", extract instance family
                instance_family = (
                    gpu_type_key.split("#")[0] if "#" in gpu_type_key else gpu_type_key
                )

                # Look up GPU model from instance specs
                from .gpu_specs import AWS_INSTANCE_GPU_MAP

                gpu_info = AWS_INSTANCE_GPU_MAP.get(instance_family, {})
                gpu_model = gpu_info.get("gpu_model", "L40S")
                gpus_per_instance = gpu_info.get("num_gpus", 8)
            else:
                instance_family = homo_cfg.get("family", "")

            # Calculate cost per million tokens
            throughput = sol.get("throughput_tokens_per_sec", 0)
            cost_per_hour = sol.get("cost_per_hour", 0)
            cost_per_million = 0
            if throughput > 0:
                cost_per_million = (cost_per_hour / (throughput * 3600)) * 1_000_000

            # Use solver's own max_model_len (consistent memory model, no dual calculation)
            tp_degree = homo_cfg.get("tp_degree", 1)
            pp_stages = homo_cfg.get("pp_stages", 1)
            max_context = min(
                sol.get('max_model_len', 8192),
                solver.config.max_position_embeddings
            )
            logger.info(
                f"[SolverAdapter] Max context from solver: {sol.get('max_model_len', 'N/A')}, "
                f"model max: {solver.config.max_position_embeddings}, using {max_context}"
            )

            return SolverOutput(
                success=True,
                instance_family=homo_cfg.get("family", instance_family),
                num_instances=homo_cfg.get("instances_used", 1),
                tp_degree=tp_degree,
                pp_stages=homo_cfg.get("pp_stages", 1),
                layers_per_stage=homo_cfg.get("layers_per_stage", 0),
                max_concurrent_sequences=sol.get("max_concurrent_sequences", 1),
                throughput_tokens_per_sec=throughput,
                cost_per_hour=cost_per_hour,
                cost_per_million_tokens=cost_per_million,
                gpu_model=gpu_model,
                gpus_per_instance=gpus_per_instance,
                total_gpus=homo_cfg.get("instances_used", 1) * gpus_per_instance,
                max_supported_context=max_context,
                estimated_runtime_hours=sol.get("estimated_runtime_hours") or 0.0,
                meets_slo=sol.get("meets_slo", True),
                solve_log=getattr(solver, 'solve_log', ''),
            )

        except Exception as e:
            logger.error(f"[SolverAdapter] Error: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            # Try to get solve_log even on exception (solver may have been partially created)
            _log = ''
            try:
                _log = getattr(solver, 'solve_log', '')
            except Exception:
                pass
            return SolverOutput(
                success=False,
                instance_family="",
                num_instances=0,
                tp_degree=0,
                pp_stages=0,
                layers_per_stage=0,
                max_concurrent_sequences=0,
                throughput_tokens_per_sec=0,
                cost_per_hour=0,
                cost_per_million_tokens=0,
                gpu_model="",
                gpus_per_instance=0,
                total_gpus=0,
                error_message=str(e),
                solve_log=_log,
            )
        finally:
            # Restore solver logger to INFO (default) so debug doesn't leak to next request
            solver_logger = logging.getLogger("solver")
            solver_logger.setLevel(logging.INFO)
            solver_logger.propagate = True
            if _solver_debug_handler:
                solver_logger.removeHandler(_solver_debug_handler)
            self._cleanup_temp_files()

    def solve_multi(self, input: SolverInput, top_k: int = 5) -> List[SolverOutput]:
        """
        Solve placement and return top K solutions sorted by cost.

        This is used for fallback - if the best solution fails in all regions,
        try the next best solution with a different instance type.

        For now, returns the single best solution from solve().
        Multi-solution support requires the underlying solver to expose
        all enumeration results.

        Args:
            input: SolverInput with model and workload parameters
            top_k: Number of top solutions to return (default 5)

        Returns:
            List of SolverOutput, sorted by cost (best first).
            Returns empty list if solver fails entirely.
        """
        # For now, just return single best solution from solve()
        # The underlying solver enumerate internally but only returns best
        result = self.solve(input)
        if result.success:
            return [result]
        return []


def create_solver_input_from_request(
    model_name: str,
    avg_input_tokens: int,
    avg_output_tokens: int,
    num_requests: int,
    slo_deadline_hours: float,
    max_input_tokens: int = 0,
    max_output_tokens: int = 0,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 16384,
    gpu_memory_utilization: float = 0.90,
    log_level: str = "info",
    num_replicas: int = 1,
) -> SolverInput:
    """
    Create SolverInput from Orca BatchedRequest fields.

    Args:
        model_name: Target LLM model name
        avg_input_tokens: Average input tokens per request
        avg_output_tokens: Average output tokens per request
        num_requests: Total number of requests (num_lines)
        slo_deadline_hours: SLO deadline in hours
        max_input_tokens: Maximum input tokens in workload (0 = use avg)
        max_output_tokens: Maximum output tokens in workload (0 = use avg)
        max_num_seqs: vLLM max_num_seqs scheduler parameter (decode batch size)
        max_num_batched_tokens: vLLM max tokens per prefill iteration (0 = fall back to max_model_len)
        gpu_memory_utilization: vLLM gpu_memory_utilization (0.0-1.0)
        log_level: Solver log level ("debug", "info", "warning")
        num_replicas: Number of data-parallel replicas sharing the workload (for SLO calculation)

    Returns:
        SolverInput ready for solver
    """
    return SolverInput(
        model_name=model_name,
        sequence_length=avg_input_tokens,
        output_length=avg_output_tokens,
        total_tokens=num_requests * (avg_input_tokens + avg_output_tokens),
        slo_hours=slo_deadline_hours,
        num_replicas=max(1, num_replicas),
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        log_level=log_level,
        # Fixed per user requirements
        workload_phase="aggregated",
        optimization_priority="cost_first",
    )
