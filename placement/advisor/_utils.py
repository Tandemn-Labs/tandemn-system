"""Shared helpers for placement/advisor/ — avoids duplication across modules."""

from __future__ import annotations

import json
from functools import lru_cache


def safe_float(val, default=0.0) -> float:
    """Parse a CSV value to float, handling NaN/bool strings gracefully."""
    try:
        if val in ("", None, "nan", "NaN"):
            return default
        if isinstance(val, str) and val.lower() in ("true", "false"):
            return 1.0 if val.lower() == "true" else 0.0
        v = float(val)
        if v != v:  # NaN check
            return default
        return v
    except (ValueError, TypeError):
        return default


def active_params_from_config(cfg: dict, num_experts_active: int) -> float | None:
    """Compute active params (billions) from an HF-style config dict.

    Returns None if the config doesn't describe an MoE model or lacks required fields.
    For dense models, returns None (caller should use total params).
    """
    total_experts = cfg.get("num_experts", cfg.get("num_local_experts", 0))
    if total_experts <= 1 or num_experts_active <= 0 or num_experts_active >= total_experts:
        return None
    hidden = cfg.get("hidden_size", 0)
    n_heads = cfg.get("num_attention_heads", 0)
    n_kv = cfg.get("num_key_value_heads", n_heads)
    n_layers = cfg.get("num_hidden_layers", 0)
    vocab = cfg.get("vocab_size", 0)
    if not all((hidden, n_heads, n_layers, vocab)):
        return None
    moe_intermediate = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", hidden * 4))
    shared_intermediate = cfg.get("shared_expert_intermediate_size", 0)
    d_head = hidden / max(n_heads, 1)
    kv_dim = n_kv * d_head
    attn_per_layer = 2 * hidden * hidden + 2 * hidden * kv_dim
    ffn_per_expert = 3 * hidden * moe_intermediate
    active_ffn = ffn_per_expert * num_experts_active
    shared_ffn = 3 * hidden * shared_intermediate if shared_intermediate else 0
    embed = vocab * hidden * 2
    return (embed + n_layers * (attn_per_layer + active_ffn + shared_ffn)) / 1e9


@lru_cache(maxsize=64)
def instance_price(instance_type: str) -> float:
    """On-demand hourly price via SkyPilot catalog. Cached across calls."""
    try:
        import sky.catalog as _sky_catalog
        cost = _sky_catalog.get_hourly_cost(
            instance_type=instance_type, use_spot=False,
            region="us-east-1", zone=None, clouds="aws",
        )
        return cost if cost else 0.0
    except Exception:
        return 0.0
