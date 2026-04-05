"""
PlacementAdvisor — orchestrator for the in-repo LLM placement advisor.

Flow:
  1. Fetch model architecture features (registry → perfdb → HF Hub → estimate)
  2. Get feasible Oracle candidates with RAG-based performance predictions
  3. Call LLM ensemble to rank candidates
  4. Convert top placements → List[MagicOutput] (same shape as roofline output)

Usage:
  advisor = PlacementAdvisor()
  outputs = advisor.recommend(request)   # List[MagicOutput], sorted by cost
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from models.resources import MagicOutput
from orca_server.utils import make_job_id

from placement.advisor.model_arch_fetcher import fetch_arch_features
from placement.advisor.oracle import get_candidates, OracleCandidate
from placement.advisor import ensemble as _ensemble


def _max_model_len(arch_features, vram_per_gpu: float, tp: int) -> Optional[int]:
    """Estimate a safe max_model_len for vLLM given VRAM after weights."""
    from placement.advisor.oracle import _model_vram_gb, _kv_cache_gb_per_token

    model_shard_gb = _model_vram_gb(arch_features) / tp
    available_gb = vram_per_gpu * 0.90 - model_shard_gb - 2.0  # 2GB vLLM overhead
    if available_gb <= 0:
        return None

    kv_per_token = _kv_cache_gb_per_token(arch_features, tp)
    if kv_per_token <= 0:
        return None

    max_tokens = int(available_gb / kv_per_token)
    # Round down to nearest power of 2 or common context length
    for ctx in [131072, 65536, 32768, 16384, 8192, 4096, 2048]:
        if max_tokens >= ctx:
            return ctx
    return 2048


def _candidate_to_magic_output(
    candidate: OracleCandidate,
    vram_per_gpu: float,
    arch_features,
    num_requests: int,
    avg_input: int,
    avg_output: int,
) -> MagicOutput:
    total_tokens = num_requests * (avg_input + avg_output)
    cost_per_million = None
    if candidate.predicted_tps and candidate.predicted_tps > 0 and candidate.predicted_cost_per_hour > 0:
        runtime_h = total_tokens / candidate.predicted_tps / 3600.0
        total_cost = runtime_h * candidate.predicted_cost_per_hour
        cost_per_million = (total_cost / total_tokens) * 1_000_000 if total_tokens > 0 else None

    max_len = _max_model_len(arch_features, vram_per_gpu, candidate.tp)

    return MagicOutput(
        decision_id=make_job_id("advisor"),
        engine="vllm",
        instance_type=candidate.instance_type,
        tp_size=candidate.tp,
        pp_size=candidate.pp,
        replicas=1,
        num_instances=candidate.num_instances,
        max_model_len=max_len,
        throughput_tokens_per_sec=candidate.predicted_tps,
        cost_per_hour=candidate.predicted_cost_per_hour,
        cost_per_million_tokens=cost_per_million,
        estimated_runtime_hours=candidate.estimated_runtime_hours,
        meets_slo=candidate.meets_slo,
    )


class PlacementAdvisor:
    """
    In-repo replacement for the external Koi placement service.
    Combines architecture-aware RAG, feasibility pruning, and a single LLM
    reasoning call to recommend GPU configurations for batch inference jobs.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def recommend(
        self,
        model_name: str,
        avg_input_tokens: int,
        avg_output_tokens: int,
        num_requests: int,
        slo_hours: float,
        gpu_pool: Optional[Dict] = None,
    ) -> List[MagicOutput]:
        """
        Return up to 5 ranked GPU configurations for the given job.

        Args:
            model_name: HuggingFace model name or short name (e.g. "meta-llama/Meta-Llama-3.1-70B")
            avg_input_tokens: Average prompt length in tokens
            avg_output_tokens: Average response length in tokens
            num_requests: Total number of requests in the batch
            slo_hours: SLO deadline in hours
            gpu_pool: {instance_type: available_count} from quota tracker, or None for all instances

        Returns:
            List[MagicOutput] sorted by cost (cheapest first), up to 5 items.
        """
        # 1. Architecture features
        arch = fetch_arch_features(model_name)

        # 2. Oracle candidates
        candidates = get_candidates(
            arch=arch,
            avg_input=avg_input_tokens,
            avg_output=avg_output_tokens,
            num_requests=num_requests,
            slo_hours=slo_hours,
            gpu_pool=gpu_pool,
        )
        if not candidates:
            return []

        # 3. LLM ranking
        result = _ensemble.run(
            model_name=model_name,
            arch=arch,
            avg_input=avg_input_tokens,
            avg_output=avg_output_tokens,
            num_requests=num_requests,
            slo_hours=slo_hours,
            candidates=candidates,
            api_key=self.api_key,
            model=self.model,
        )

        # 4. Convert to MagicOutput
        from orca_server.config import AWS_INSTANCES
        outputs: List[MagicOutput] = []
        seen: set[tuple] = set()

        for placement in result.placements:
            c = placement.candidate
            key = (c.instance_type, c.tp, c.pp)
            if key in seen:
                continue
            seen.add(key)

            vram_per_gpu = AWS_INSTANCES.get(c.instance_type, (None, None, None, 24))[3]
            outputs.append(_candidate_to_magic_output(
                candidate=c,
                vram_per_gpu=vram_per_gpu,
                arch_features=arch,
                num_requests=num_requests,
                avg_input=avg_input_tokens,
                avg_output=avg_output_tokens,
            ))

        # Pad with next-best candidates if fewer than 3 LLM picks
        if len(outputs) < 3:
            for c in candidates:
                if len(outputs) >= 5:
                    break
                key = (c.instance_type, c.tp, c.pp)
                if key in seen:
                    continue
                seen.add(key)
                vram_per_gpu = AWS_INSTANCES.get(c.instance_type, (None, None, None, 24))[3]
                outputs.append(_candidate_to_magic_output(
                    candidate=c,
                    vram_per_gpu=vram_per_gpu,
                    arch_features=arch,
                    num_requests=num_requests,
                    avg_input=avg_input_tokens,
                    avg_output=avg_output_tokens,
                ))

        return outputs[:5]
