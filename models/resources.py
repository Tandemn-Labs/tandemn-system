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
    # SLO estimates (populated when slo_deadline_hours is set)
    estimated_runtime_hours: Optional[float] = None
    meets_slo: Optional[bool] = None

    @property
    def num_nodes(self) -> int:
        """SkyPilot nodes needed per replica.
        TP stays intra-node, PP crosses node boundaries.
        Uses solver's num_instances when available (accounts for multi-PP-per-node
        on fixed-size instances like p4d.24xlarge where gpus > tp_degree)."""
        if self.num_instances is not None:
            return self.num_instances
        return self.pp_size
