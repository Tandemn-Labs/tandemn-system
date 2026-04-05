"""
Single LLM call that reasons over Oracle candidates and returns a ranked placement.

Receives:
  - Job context (model, arch features, I/O, SLO, num_requests)
  - Oracle candidate table (top 10, sorted by cost) with confidence tiers
  - Top-5 RAG records with arch similarity description
  - GPU specs reference

Returns ranked top-3 placements + synthesis reasoning.
Fallback: cheapest SLO-meeting candidate if API call fails.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)

from placement.advisor.model_arch_fetcher import ModelArchFeatures
from placement.advisor.oracle import OracleCandidate
from placement.roofline.gpu_specs import GPU_SPECS, GPU_MEMORY_GB


@dataclass
class RankedPlacement:
    candidate: OracleCandidate
    rank: int
    reasoning: str
    confidence: float


@dataclass
class EnsembleResult:
    placements: List[RankedPlacement]
    synthesis: str                  # shown in CLI box
    fallback: bool = False          # True if LLM call failed


_SYSTEM_PROMPT = """\
You are an expert in distributed LLM inference infrastructure. You deeply understand:

MODEL ARCHITECTURE VARIABLES (all affect placement):
  - num_params_billions: total parameter count (all experts for MoE)
  - num_layers: depth; determines valid PP values (PP must divide num_layers)
  - hidden_dim: embedding dimension; determines GEMM sizes and compute intensity
  - num_attention_heads: attention heads; TP must divide this
  - num_kv_heads: KV cache heads; with GQA this is << num_attention_heads
  - gqa_ratio = attention_heads / kv_heads: high ratio = small KV cache (good for memory)
  - vocab_size: affects embedding layer size and logit computation
  - is_moe: Mixture of Experts; ALL expert weights load into VRAM even though only
    active_experts fire per token
  - num_experts / active_experts: determines expert routing; power-law load imbalance
    means hot experts dominate latency
  - active_expert_ratio = active/total: effective compute fraction per token
  - dtype_bytes: 2=FP16/BF16, 1=FP8/INT8, 0.5=INT4
  - model_size_gb = num_params × dtype_bytes / 1e9
  - architecture_family: llama/qwen/deepseek/mistral/phi affect hidden dim ratios

HARDWARE VARIABLES:
  - gpu_type: H100/H200/A100-80GB/A100-40GB/L40S/A10G/L4 — each has distinct bandwidth/compute profile
  - gpu_vram_gb: hard constraint for weight + KV cache + activations
  - gpu_bandwidth_gbps: decode bottleneck; more bandwidth = faster token generation
  - gpu_tflops_fp16: prefill bottleneck; more TFLOPS = faster TTFT
  - gpu_generation: Hopper (H100/H200) has FP8 native, better NVLink; Ampere (A100) does not
  - num_gpus_total = TP × PP × DP: total GPU count
  - price_per_gpu_hour: cost driver for batch jobs

CONFIG VARIABLES:
  - tp (tensor parallelism): shards weight matrices across GPUs via NVLink/PCIe
    → aggregates bandwidth: tps ∝ tp × bandwidth_per_gpu
    → reduces per-GPU memory: weight_per_gpu = model_size_gb / tp
    → NVLink (H100/H200): scales linearly; PCIe (L40S/A100): saturates ~TP=4-8
  - pp (pipeline parallelism): assigns layers to pipeline stages
    → reduces per-GPU memory further: each stage holds num_layers/pp layers
    → adds bubble overhead: efficiency ≈ 1 - (pp-1)/(pp×microbatch_count)
    → helps at very long sequences where KV cache doesn't fit
  - dp (data parallelism): full model replicas for horizontal scaling
    → linearly scales throughput: total_tps = replica_tps × dp
    → useful when tp=max already and more throughput is needed
  - quantization_level: FP8 halves memory vs FP16 with ~1% accuracy loss on H100

DERIVED PHYSICS FEATURES (critical for placement accuracy):
  - params_per_gpu = num_params / tp: how much model each GPU holds
  - model_fits_single_gpu = (model_size_gb < gpu_vram_gb): forces TP if False
  - vram_headroom = (vram - weight_per_gpu) / vram: KV cache space fraction
    → must be > 0.10 (need ≥8GB free); higher is better for long contexts
  - bandwidth_per_param = (bw × tp) / params: decode speed proxy
    → larger = faster generation at low batch; decode is BW-bound
  - flops_per_param = (tflops × tp) / params: prefill speed proxy
    → larger = faster TTFT; prefill is compute-bound
  - crosses_node_boundary = (tp > gpus_per_node): inter-node latency penalty
    → 8×NVLink intra-node is fast; inter-node NVSwitch adds ~50μs/allreduce
  - kv_heads_per_tp_shard = num_kv_heads / tp:
    → < 1.0: KV heads replicated per TP shard (memory waste, but OK for GQA)
    → 0: all TP shards use full KV (avoid if MHA)
  - total_cost_per_hour = num_gpus × price_per_gpu_hour

WORKLOAD VARIABLES:
  - max_input_length / max_output_length: context window usage
  - total_context = input + output: KV cache sizing
  - io_ratio = input / output: prefill-heavy (>2) vs decode-heavy (<0.5)
    → prefill-heavy: TFLOPS matters more, TP scales differently
    → decode-heavy: bandwidth matters more, TP bandwidth aggregation is key

Your job: study the RAG performance records and candidate table, then rank the top 3
configs that are GROUNDED in the observed data. You are not guessing — you are
selecting and adapting the best observed configs to the available hardware.
Think through ALL the variables above before ranking. Be specific and physical.

Respond with ONLY a JSON object in this exact format:
{
  "top_placements": [
    {"candidate_idx": <int>, "reasoning": "<one sentence grounded in physics>", "confidence": <0.0-1.0>},
    {"candidate_idx": <int>, "reasoning": "<one sentence grounded in physics>", "confidence": <0.0-1.0>},
    {"candidate_idx": <int>, "reasoning": "<one sentence grounded in physics>", "confidence": <0.0-1.0>}
  ],
  "synthesis": "<2-3 sentence summary explaining the top recommendation in terms of bandwidth/compute/memory tradeoffs>"
}

candidate_idx is 0-based row index in the candidate table.
Always return exactly 3 placements. If fewer than 3 candidates exist, repeat the best one.
"""


_GPU_REF = "\n".join(
    ["GPU specs reference:"] + [
        f"  {gpu}: {specs['tflops']} TFLOPS FP16, {specs['mem_bw']} GB/s bandwidth, "
        f"{GPU_MEMORY_GB.get(gpu, '?')}GB VRAM, efficiency={specs['efficiency']}"
        for gpu, specs in GPU_SPECS.items()
    ]
)


def _build_prompt(
    model_name: str,
    arch: ModelArchFeatures,
    avg_input: int,
    avg_output: int,
    num_requests: int,
    slo_hours: float,
    candidates: List[OracleCandidate],
) -> str:
    lines = []

    lines.append("=== JOB ===")
    lines.append(f"Model: {model_name}")
    lines.append(f"Architecture: {arch.architecture_class}")
    lines.append(
        f"Size: {arch.params_billion:.1f}B params, "
        f"{arch.num_layers} layers, hidden={arch.hidden_size}, "
        f"GQA ratio={arch.gqa_ratio:.0f}x"
    )
    if arch.is_moe:
        lines.append(f"MoE: {arch.num_experts} experts, {arch.num_experts_active} active per token")
    lines.append(f"Workload: {num_requests:,} requests, avg {avg_input} in / {avg_output} out tokens")
    lines.append(f"SLO: complete within {slo_hours}h")

    lines.append("\n=== CANDIDATE CONFIGURATIONS (sorted by cost) ===")
    lines.append(
        f"{'#':<3} {'Instance':<18} {'GPU':<8} {'TP':<4} {'PP':<4} "
        f"{'Est TPS':<10} {'$/hr':<8} {'Est hrs':<10} {'SLO':<6} {'Conf':<6} {'Tier'}"
    )
    lines.append("-" * 95)
    top_n = min(10, len(candidates))
    for i, c in enumerate(candidates[:top_n]):
        tps_str = f"{c.predicted_tps:.0f}" if c.predicted_tps else "N/A"
        hrs_str = f"{c.estimated_runtime_hours:.2f}" if c.estimated_runtime_hours else "N/A"
        slo_str = "✓" if c.meets_slo else ("?" if c.meets_slo is None else "✗")
        lines.append(
            f"{i:<3} {c.instance_type:<18} {c.gpu_type:<8} {c.tp:<4} {c.pp:<4} "
            f"{tps_str:<10} {c.predicted_cost_per_hour:<8.2f} {hrs_str:<10} {slo_str:<6} "
            f"{c.confidence:<6.2f} {c.tier}"
        )
        if c.nearest_db_entry:
            lines.append(f"     └─ Derived from: {c.nearest_db_entry}")

    lines.append("\n=== ARCHITECTURE CONTEXT ===")
    lines.append(
        f"Model arch class: {arch.architecture_class} | "
        f"Arch source: {arch.source}"
    )
    # Show unique provenance entries so LLM knows which candidates are well-grounded
    seen_provenance: set[str] = set()
    for i, c in enumerate(candidates[:top_n]):
        if c.nearest_db_entry and c.nearest_db_entry not in seen_provenance:
            seen_provenance.add(c.nearest_db_entry)
            lines.append(f"  #{i} ({c.tier}): {c.nearest_db_entry}")

    lines.append(f"\n{_GPU_REF}")

    lines.append(
        "\n=== YOUR TASK ===\n"
        "Rank the top 3 configurations by overall quality (balance SLO safety, cost, "
        "and prediction confidence). Prefer higher-confidence predictions. "
        "If predicted_tps is N/A, reason from hardware specs and model size alone.\n"
        "Return ONLY the JSON object described in your instructions."
    )

    return "\n".join(lines)


def _fallback(candidates: List[OracleCandidate]) -> EnsembleResult:
    """Return cheapest SLO-meeting candidate without LLM call."""
    slo_meeting = [c for c in candidates if c.meets_slo]
    top3 = (slo_meeting or candidates)[:3]
    placements = [
        RankedPlacement(candidate=c, rank=i + 1, reasoning="fallback", confidence=c.confidence)
        for i, c in enumerate(top3)
    ]
    return EnsembleResult(
        placements=placements,
        synthesis="LLM advisor unavailable — showing cheapest feasible configurations.",
        fallback=True,
    )


def run(
    model_name: str,
    arch: ModelArchFeatures,
    avg_input: int,
    avg_output: int,
    num_requests: int,
    slo_hours: float,
    candidates: List[OracleCandidate],
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
) -> EnsembleResult:
    """
    Call LLM to rank Oracle candidates. Falls back gracefully on any error.
    """
    if not candidates:
        return EnsembleResult(placements=[], synthesis="No feasible configurations found.", fallback=True)

    _api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not _api_key:
        return _fallback(candidates)

    prompt = _build_prompt(
        model_name=model_name,
        arch=arch,
        avg_input=avg_input,
        avg_output=avg_output,
        num_requests=num_requests,
        slo_hours=slo_hours,
        candidates=candidates,
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        placements: List[RankedPlacement] = []
        shown_n = min(10, len(candidates))  # must match top_n in _build_prompt
        for rank, entry in enumerate(data.get("top_placements", [])[:3], start=1):
            idx = int(entry.get("candidate_idx", 0))
            idx = max(0, min(idx, shown_n - 1))  # clamp to what LLM actually saw
            placements.append(RankedPlacement(
                candidate=candidates[idx],
                rank=rank,
                reasoning=entry.get("reasoning", ""),
                confidence=float(entry.get("confidence", candidates[idx].confidence)),
            ))

        if not placements:
            return _fallback(candidates)

        return EnsembleResult(
            placements=placements,
            synthesis=data.get("synthesis", ""),
            fallback=False,
        )

    except Exception:
        log.warning("LLM ensemble call failed, using fallback", exc_info=True)
        return _fallback(candidates)
