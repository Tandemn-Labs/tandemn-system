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
import os
from dataclasses import dataclass
from typing import List, Optional

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
You are an expert in LLM inference infrastructure. Your job is to recommend the best \
GPU placement configuration for a batched inference job based on real profiling data \
and hardware constraints.

You will receive:
1. Job details (model, workload, SLO deadline)
2. A table of feasible configurations ranked by cost, with predicted throughput and confidence
3. Architecture context — which profiled model is closest and how predictions were derived
4. GPU hardware reference

Respond with ONLY a JSON object in this exact format:
{
  "top_placements": [
    {"candidate_idx": <int>, "reasoning": "<one sentence>", "confidence": <0.0-1.0>},
    {"candidate_idx": <int>, "reasoning": "<one sentence>", "confidence": <0.0-1.0>},
    {"candidate_idx": <int>, "reasoning": "<one sentence>", "confidence": <0.0-1.0>}
  ],
  "synthesis": "<2-3 sentence summary for the user explaining the top recommendation>"
}

candidate_idx refers to the row index (0-based) in the candidate table you receive.
Always return exactly 3 placements. If fewer than 3 candidates exist, repeat the best one.
"""


def _build_gpu_ref() -> str:
    lines = ["GPU specs reference:"]
    for gpu, specs in GPU_SPECS.items():
        vram = GPU_MEMORY_GB.get(gpu, "?")
        lines.append(
            f"  {gpu}: {specs['tflops']} TFLOPS FP16, {specs['mem_bw']} GB/s bandwidth, "
            f"{vram}GB VRAM, efficiency={specs['efficiency']}"
        )
    return "\n".join(lines)


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
    if candidates and candidates[0].nearest_db_entry:
        lines.append(f"Closest profiled match: {candidates[0].nearest_db_entry}")
    lines.append(
        f"Model arch class: {arch.architecture_class} | "
        f"Arch source: {arch.source}"
    )

    lines.append(f"\n{_build_gpu_ref()}")

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
        for rank, entry in enumerate(data.get("top_placements", [])[:3], start=1):
            idx = int(entry.get("candidate_idx", 0))
            idx = max(0, min(idx, len(candidates) - 1))
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
        return _fallback(candidates)
