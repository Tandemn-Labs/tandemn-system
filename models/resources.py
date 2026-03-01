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
