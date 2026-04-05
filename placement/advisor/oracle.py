"""
Feasibility pruning + arch-scaled performance prediction.

Produces an OracleCandidate list over all valid (instance_type, tp, pp) configs.
The LLM ensemble reasons over this list — it never sees infeasible configs.

Interpolation tiers:
  T1 (0.85): exact model_name + gpu_model + tp + pp in RAG
  T2 (0.65): same model_architecture class + gpu_model + tp + pp
  T3 (0.50): same gpu_model + tp + pp, different arch → scale by bandwidth_per_param
  No T4:     if no DB match, candidate passes with predicted_tps=None, conf=0.0
             (LLM reasons from hardware constraints + arch features alone)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from orca_server.config import AWS_INSTANCES
from placement.roofline.gpu_specs import GPU_SPECS, GPU_MEMORY_GB
from placement.advisor.model_arch_fetcher import ModelArchFeatures
from placement.advisor.perf_rag import retrieve_multi

# Multi-GPU instances only (single-GPU not useful for large models)
_MULTI_GPU_PREFIXES = (
    "p5.", "p4d.", "p4de.",
    "g6e.12xlarge", "g6e.24xlarge", "g6e.48xlarge",
    "g5.12xlarge", "g5.24xlarge", "g5.48xlarge",
    "g6.12xlarge", "g6.24xlarge", "g6.48xlarge",
)

# GPU names available in the perf database
_PERFDB_GPUS = {"A100", "H100_SXM", "H200_SXM", "L40S", "B200", "GB200"}

# Map Orca GPU names → perf DB GPU names
_GPU_NAME_MAP = {
    "H100": "H100_SXM",
    "H200": "H200_SXM",
    "A100": "A100",
    "L40S": "L40S",
}

_TP_OPTIONS = [1, 2, 4, 8]
_PP_OPTIONS = [1, 2, 4]


@dataclass
class OracleCandidate:
    gpu_type: str                   # Orca GPU name e.g. "A100"
    instance_type: str              # AWS instance e.g. "p4de.24xlarge"
    gpu_count: int                  # GPUs per instance
    tp: int
    pp: int
    num_instances: int              # instances needed for tp*pp GPUs
    predicted_tps: Optional[float]  # None if no DB match
    predicted_cost_per_hour: float
    estimated_runtime_hours: Optional[float]
    meets_slo: Optional[bool]
    confidence: float               # 0.0 = no data, 0.50-0.85 = interpolated
    tier: str                       # "T1_exact" | "T2_arch_class" | "T3_gpu_scaling" | "none"
    nearest_db_entry: str           # human-readable for LLM context
    rag_records: List[dict] = field(default_factory=list)


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val not in ("", None, "nan", "NaN") else default
    except (ValueError, TypeError):
        return default


def _instance_price(instance_type: str) -> float:
    """Approximate on-demand hourly price. Falls back to a rough estimate."""
    try:
        import sky.catalog as _sky_catalog
        cost = _sky_catalog.get_hourly_cost(
            instance_type=instance_type, use_spot=False,
            region="us-east-1", zone=None, clouds="aws",
        )
        return cost if cost else 0.0
    except Exception:
        return 0.0


def _model_vram_gb(arch: ModelArchFeatures, precision_bytes: int = 2) -> float:
    """Estimate model weight footprint in GB."""
    # For MoE use active params fraction
    active_params = arch.params_billion
    if arch.is_moe and arch.num_experts > 0 and arch.num_experts_active > 0:
        active_params = arch.params_billion * (arch.num_experts_active / arch.num_experts)
    return active_params * precision_bytes  # bf16 = 2 bytes/param


def _kv_cache_gb_per_token(arch: ModelArchFeatures, tp: int) -> float:
    """KV cache bytes per token per layer, in GB."""
    head_dim = arch.hidden_size / max(arch.num_attention_heads, 1)
    kv_heads_per_tp = max(1, arch.num_kv_heads // tp)
    bytes_per_token_per_layer = 2 * kv_heads_per_tp * head_dim * 2  # K+V, bf16
    total_bytes = bytes_per_token_per_layer * arch.num_layers
    return total_bytes / (1024 ** 3)


def _is_feasible(
    arch: ModelArchFeatures,
    gpu_type: str,
    vram_per_gpu: float,
    tp: int,
    pp: int,
) -> tuple[bool, str]:
    """Return (feasible, reason). Hard constraint checks."""
    # TP must divide num_attention_heads and num_kv_heads
    if arch.num_attention_heads % tp != 0:
        return False, f"TP={tp} doesn't divide num_attention_heads={arch.num_attention_heads}"
    if arch.num_kv_heads % tp != 0 and arch.num_kv_heads > 1:
        return False, f"TP={tp} doesn't divide num_kv_heads={arch.num_kv_heads}"

    # PP must divide num_layers
    if arch.num_layers % pp != 0:
        return False, f"PP={pp} doesn't divide num_layers={arch.num_layers}"

    # VRAM: model shard + KV cache headroom
    model_gb = _model_vram_gb(arch)
    shard_gb = model_gb / (tp * pp)
    kv_per_token = _kv_cache_gb_per_token(arch, tp)
    # Need at least 1K tokens of KV cache headroom
    min_kv_gb = kv_per_token * 1024
    used_gb = shard_gb + min_kv_gb
    if used_gb > vram_per_gpu * 0.92:
        return False, f"VRAM: need {used_gb:.1f}GB, have {vram_per_gpu * 0.92:.1f}GB"

    return True, ""


def _scale_tps_for_io(base_tps: float, base_input: float, base_output: float,
                       target_input: float, target_output: float) -> float:
    """Scale throughput for different I/O lengths using dampened sqrt."""
    base_work = 0.3 * base_input + 1.0 * base_output
    target_work = 0.3 * target_input + 1.0 * target_output
    if base_work <= 0 or target_work <= 0:
        return base_tps
    scale = math.sqrt(base_work / target_work)
    scale = max(0.5, min(1.4, scale))
    return base_tps * scale


def _predict_from_records(
    records: List[dict],
    arch: ModelArchFeatures,
    gpu_type: str,
    tp: int,
    pp: int,
    avg_input: int,
    avg_output: int,
) -> tuple[Optional[float], float, str, str]:
    """
    Return (predicted_tps, confidence, tier, nearest_entry_description).
    Tries T1 → T2 → T3 in order, returns (None, 0.0, 'none', '') if nothing matches.
    """
    perfdb_gpu = _GPU_NAME_MAP.get(gpu_type, gpu_type)

    # Filter to matching gpu+tp+pp
    matching = [
        r for r in records
        if r.get("gpu_model") == perfdb_gpu
        and _safe_float(r.get("tp")) == tp
        and _safe_float(r.get("pp")) == pp
        and _safe_float(r.get("tokens_per_sec_total")) > 0
    ]
    if not matching:
        return None, 0.0, "none", ""

    # T1: exact model match
    exact = [r for r in matching if r.get("model_name", "").lower() == arch.__class__.__name__.lower()
             or r.get("model_name", "") != ""]
    # refine: exact model name match
    exact_model = [r for r in matching
                   if r.get("model_name", "").split("/")[-1].lower().replace("-", "").replace(".", "")
                   in arch.__class__.__name__.lower().replace("for", "").replace("causallm", "")
                   or False]
    # simpler: check if model_name in records matches the arch's model id
    for r in matching:
        mn = r.get("model_name", "")
        # direct string match on architecture class
        if r.get("model_architecture", "") == arch.architecture_class:
            pass  # handled in T2

    # Actually: T1 = same model_architecture + same params_billion range + gpu+tp+pp
    t1_records = [
        r for r in matching
        if r.get("model_architecture") == arch.architecture_class
        and abs(_safe_float(r.get("params_billion")) - arch.params_billion) < arch.params_billion * 0.15
    ]
    if t1_records:
        best = max(t1_records, key=lambda r: _safe_float(r.get("tokens_per_sec_total")))
        base_tps = _safe_float(best.get("tokens_per_sec_total"))
        base_in = _safe_float(best.get("input_len_tokens_avg") or best.get("input_len_tokens_fixed"), 512)
        base_out = _safe_float(best.get("output_len_tokens_avg") or best.get("output_len_tokens_fixed"), 256)
        tps = _scale_tps_for_io(base_tps, base_in, base_out, avg_input, avg_output)
        desc = (f"{best.get('model_name')} on {perfdb_gpu} TP={tp} PP={pp} "
                f"({best.get('input_len_tokens_avg') or best.get('input_len_tokens_fixed')} in/"
                f"{best.get('output_len_tokens_avg') or best.get('output_len_tokens_fixed')} out)")
        return tps, 0.85, "T1_exact", desc

    # T2: same architecture class, scale by params_billion ratio
    t2_records = [r for r in matching if r.get("model_architecture") == arch.architecture_class]
    if t2_records:
        best = max(t2_records, key=lambda r: _safe_float(r.get("tokens_per_sec_total")))
        base_tps = _safe_float(best.get("tokens_per_sec_total"))
        ref_params = _safe_float(best.get("params_billion"), arch.params_billion)
        # Throughput scales roughly inversely with params (memory bound decode)
        param_scale = ref_params / max(arch.params_billion, 0.1)
        param_scale = max(0.3, min(3.0, param_scale))
        base_in = _safe_float(best.get("input_len_tokens_avg") or best.get("input_len_tokens_fixed"), 512)
        base_out = _safe_float(best.get("output_len_tokens_avg") or best.get("output_len_tokens_fixed"), 256)
        tps = _scale_tps_for_io(base_tps * param_scale, base_in, base_out, avg_input, avg_output)
        desc = (f"{best.get('model_name')} on {perfdb_gpu} TP={tp} PP={pp} "
                f"(T2: same arch class {arch.architecture_class}, "
                f"{ref_params:.0f}B→{arch.params_billion:.0f}B scale)")
        return tps, 0.65, "T2_arch_class", desc

    # T3: same gpu+tp+pp, different arch — scale by bandwidth_per_param ratio
    t3_records = matching  # all have matching gpu+tp+pp
    if t3_records:
        best = max(t3_records, key=lambda r: _safe_float(r.get("tokens_per_sec_total")))
        base_tps = _safe_float(best.get("tokens_per_sec_total"))
        ref_bw = _safe_float(best.get("bandwidth_per_param"), 1.0)
        gpu_specs = GPU_SPECS.get(gpu_type, GPU_SPECS.get("A100"))
        if gpu_specs and ref_bw > 0:
            # Rough scale: target bw_per_param ∝ gpu_bw / params
            target_bw = gpu_specs["mem_bw"] / max(arch.params_billion * 1e9 / 1e6, 1.0)
            bw_scale = target_bw / ref_bw
            bw_scale = max(0.2, min(5.0, bw_scale))
        else:
            bw_scale = 1.0
        base_in = _safe_float(best.get("input_len_tokens_avg") or best.get("input_len_tokens_fixed"), 512)
        base_out = _safe_float(best.get("output_len_tokens_avg") or best.get("output_len_tokens_fixed"), 256)
        tps = _scale_tps_for_io(base_tps * bw_scale, base_in, base_out, avg_input, avg_output)
        desc = (f"{best.get('model_name')} on {perfdb_gpu} TP={tp} PP={pp} "
                f"(T3: different arch, bw_per_param scaled)")
        return tps, 0.50, "T3_gpu_scaling", desc

    return None, 0.0, "none", ""


def get_candidates(
    arch: ModelArchFeatures,
    avg_input: int,
    avg_output: int,
    num_requests: int,
    slo_hours: float,
    gpu_pool: Optional[dict] = None,  # {instance_type: available_count}, None = all AWS_INSTANCES
) -> List[OracleCandidate]:
    """
    Return all feasible OracleCandidates sorted by predicted cost (cheapest first).
    gpu_pool restricts which instance types are considered.
    """
    # Build list of candidate (instance_type, gpu_name, gpu_count, vram_per_gpu)
    candidate_instances = []
    for inst, (gpu_name, gpu_count, _vcpus, vram_per_gpu) in AWS_INSTANCES.items():
        if not any(inst.startswith(p) or inst == p for p in _MULTI_GPU_PREFIXES):
            continue
        if gpu_pool is not None and inst not in gpu_pool:
            continue
        if gpu_name in ("V100", "A10G", "L4"):  # skip non-perf-DB GPUs for advisor
            continue
        candidate_instances.append((inst, gpu_name, gpu_count, vram_per_gpu))

    if not candidate_instances:
        return []

    # Collect all feasible (instance, tp, pp) combos for RAG multi-query
    feasible_combos: list[tuple[str, int, int, str, int, int, float]] = []  # (inst, tp, pp, gpu, count, vram, price)
    for inst, gpu_name, gpu_count, vram_per_gpu in candidate_instances:
        price = _instance_price(inst)
        for tp in _TP_OPTIONS:
            if tp > gpu_count:
                continue
            for pp in _PP_OPTIONS:
                total_gpus = tp * pp
                if total_gpus > gpu_count * 8:  # max 8 instances
                    continue
                ok, _ = _is_feasible(arch, gpu_name, vram_per_gpu, tp, pp)
                if ok:
                    feasible_combos.append((inst, tp, pp, gpu_name, gpu_count, vram_per_gpu, price))

    if not feasible_combos:
        return []

    # One RAG sweep across all feasible configs
    perfdb_configs = list({
        (_GPU_NAME_MAP.get(gpu, gpu), tp, pp)
        for (_, tp, pp, gpu, _, _, _) in feasible_combos
    })
    rag_records = retrieve_multi(
        arch_class=arch.architecture_class,
        params_billion=arch.params_billion,
        gqa_ratio=arch.gqa_ratio,
        is_moe=arch.is_moe,
        num_experts_active=arch.num_experts_active,
        feasible_configs=perfdb_configs,
        avg_input=avg_input,
        avg_output=avg_output,
        k_per_config=8,
    )

    # Build candidates
    candidates: list[OracleCandidate] = []
    for inst, tp, pp, gpu_name, gpu_count, vram_per_gpu, price_per_inst in feasible_combos:
        total_gpus_needed = tp * pp
        num_instances = max(1, math.ceil(total_gpus_needed / gpu_count))
        cost_per_hour = price_per_inst * num_instances

        predicted_tps, confidence, tier, nearest_entry = _predict_from_records(
            records=rag_records,
            arch=arch,
            gpu_type=gpu_name,
            tp=tp,
            pp=pp,
            avg_input=avg_input,
            avg_output=avg_output,
        )

        total_tokens = num_requests * (avg_input + avg_output)
        if predicted_tps and predicted_tps > 0:
            runtime_h = (total_tokens / predicted_tps) / 3600.0
            meets_slo = runtime_h <= slo_hours
        else:
            runtime_h = None
            meets_slo = None

        candidates.append(OracleCandidate(
            gpu_type=gpu_name,
            instance_type=inst,
            gpu_count=gpu_count,
            tp=tp,
            pp=pp,
            num_instances=num_instances,
            predicted_tps=predicted_tps,
            predicted_cost_per_hour=cost_per_hour,
            estimated_runtime_hours=runtime_h,
            meets_slo=meets_slo,
            confidence=confidence,
            tier=tier,
            nearest_db_entry=nearest_entry,
            rag_records=rag_records[:5],  # top-5 for LLM context
        ))

    # Sort: SLO-meeting first, then by cost
    candidates.sort(key=lambda c: (
        0 if c.meets_slo else (1 if c.meets_slo is None else 2),
        c.predicted_cost_per_hour,
    ))
    return candidates[:20]  # top-20 to LLM
