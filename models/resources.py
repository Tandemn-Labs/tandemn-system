from __future__ import annotations
from pydantic import BaseModel

class MagicOutput(BaseModel):
    decision_id: str
    engine: str
    instances: str
    num_nodes: int
    tp_size: int
    pp_size: int