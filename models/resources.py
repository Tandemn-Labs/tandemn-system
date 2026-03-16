from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class MagicOutput(BaseModel):
    decision_id: str
    engine: str
    instance_type: str
    tp_size: int
    pp_size: int
    replicas: int
    max_model_len: Optional[int] = None  # Max context length for vLLM --max-model-len
    num_instances: Optional[int] = None  # Total instances needed (from solver). Overrides pp_size*replicas when set.
    # Solver estimates (populated by roofline solver, None for fallback/user_specified)
    throughput_tokens_per_sec: Optional[float] = None
    cost_per_hour: Optional[float] = None
    cost_per_million_tokens: Optional[float] = None

    @property
    def num_nodes(self) -> int:
        """Total SkyPilot nodes needed. Uses solver's num_instances if available,
        otherwise falls back to pp_size * replicas (1 PP stage per node)."""
        if self.num_instances is not None:
            return self.num_instances  # solver already computed total nodes
        return self.pp_size * self.replicas
