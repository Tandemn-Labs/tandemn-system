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
    planned_market: Optional[str] = None
    max_model_len: Optional[int] = None  # Max context length for vLLM --max-model-len
    num_instances: Optional[int] = (
        None  # Total instances needed (from solver). Overrides pp_size*replicas when set.
    )
    # Solver estimates (populated by roofline solver, None for fallback/user_specified)
    throughput_tokens_per_sec: Optional[float] = None
    cost_per_hour: Optional[float] = None
    cost_per_million_tokens: Optional[float] = None
    # SLO estimates (populated when slo_deadline_hours is set)
    estimated_runtime_hours: Optional[float] = None
    meets_slo: Optional[bool] = None
    is_fallback: bool = (
        False  # True when solver returned a generic fallback (unsupported model)
    )

    @property
    def num_nodes(self) -> int:
        """Configured node count for this launch topology.

        TP is assumed to stay within a node. PP may either span nodes or pack
        multiple stages onto one node when an instance has more GPUs than TP
        requires (for example TP=2, PP=4 on an 8-GPU p4d).
        """
        if self.num_instances is not None:
            return self.num_instances
        return self.pp_size
