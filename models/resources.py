from __future__ import annotations
from pydantic import BaseModel


class MagicOutput(BaseModel):
    decision_id: str
    engine: str
    instance_type: str
    num_inst: int
    tp_size: int
    pp_size: int
    replicas: int
