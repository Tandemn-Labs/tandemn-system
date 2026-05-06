from __future__ import annotations

from pydantic import BaseModel

from orca_server.cloud.models import PlacementCandidate


class MagicOutput(BaseModel):
    decision_id: str
    engine: str
    instance_type: str
    tp_size: int
    pp_size: int
    replicas: int
    planned_market: str | None = None
    max_model_len: int | None = None  # Max context length for vLLM --max-model-len
    num_instances: int | None = None  # Total instances needed (from solver). Overrides pp_size*replicas when set.
    # Solver estimates (populated by roofline solver, None for fallback/user_specified)
    throughput_tokens_per_sec: float | None = None
    cost_per_hour: float | None = None
    cost_per_million_tokens: float | None = None
    # SLO estimates (populated when slo_deadline_hours is set)
    estimated_runtime_hours: float | None = None
    meets_slo: bool | None = None
    is_fallback: bool = False  # True when solver returned a generic fallback (unsupported model)
    placement_candidate: PlacementCandidate | None = None

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
