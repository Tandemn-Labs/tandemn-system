"""
Model architecture feature extraction for the placement advisor.

Lookup order:
  1. placement/roofline/model_arch.py registry (fast, no network)
  2. data.csv model_config_json column (for models in the perf DB)
  3. HuggingFace Hub config.json fetch (requires HF_TOKEN)
  4. Regex param-count extraction + Llama-class template fallback
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# Reuse existing registry
from placement.roofline.model_arch import (
    get_model_architecture,
    normalize_model_name,
    estimate_model_size_from_name,
)


@dataclass
class ModelArchFeatures:
    architecture_class: str        # HF class name e.g. "LlamaForCausalLM"
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    gqa_ratio: float               # num_attention_heads / num_kv_heads
    intermediate_size: int
    is_moe: bool
    num_experts: int               # 0 for dense
    num_experts_active: int        # 0 for dense
    params_billion: float
    vocab_size: int
    max_position_embeddings: int
    source: str                    # "registry" | "perfdb" | "hf_hub" | "estimate"


# Map HF architecture class → canonical arch name used in data.csv
_ARCH_CLASS_MAP = {
    "LlamaForCausalLM":      "LlamaForCausalLM",
    "MistralForCausalLM":    "LlamaForCausalLM",   # same structure
    "Qwen3ForCausalLM":      "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM":   "Qwen3MoeForCausalLM",
    "NemotronHForCausalLM":  "NemotronHForCausalLM",
    "DeciLMForCausalLM":     "DeciLMForCausalLM",
    "MixtralForCausalLM":    "Qwen3MoeForCausalLM",  # MoE, similar structure
}


def _from_registry(model_name: str) -> Optional[ModelArchFeatures]:
    arch = get_model_architecture(model_name)
    if arch is None:
        return None

    is_moe = "mixtral" in model_name.lower() or "moe" in model_name.lower()
    params = arch.total_params_billions

    # Infer architecture class from model name
    name_lower = model_name.lower()
    if "qwen3" in name_lower and is_moe:
        arch_class = "Qwen3MoeForCausalLM"
    elif "qwen" in name_lower:
        arch_class = "Qwen3ForCausalLM"
    elif "mixtral" in name_lower:
        arch_class = "Qwen3MoeForCausalLM"
    else:
        arch_class = "LlamaForCausalLM"

    return ModelArchFeatures(
        architecture_class=arch_class,
        num_layers=arch.num_hidden_layers,
        hidden_size=arch.hidden_size,
        num_attention_heads=arch.num_attention_heads,
        num_kv_heads=arch.num_kv_heads,
        gqa_ratio=arch.num_attention_heads / max(arch.num_kv_heads, 1),
        intermediate_size=arch.intermediate_size,
        is_moe=is_moe,
        num_experts=0,
        num_experts_active=0,
        params_billion=params,
        vocab_size=arch.vocab_size,
        max_position_embeddings=arch.max_position_embeddings,
        source="registry",
    )


@lru_cache(maxsize=32)
def _load_perfdb_configs() -> dict[str, dict]:
    """Load model_config_json entries from data.csv, keyed by model_name. Cached."""
    data_csv = os.path.join(
        os.path.dirname(__file__),
        "../../LLM_placement_solver/llm_advisor/data/aiconfigurator/data.csv",
    )
    data_csv = os.path.normpath(data_csv)
    if not os.path.exists(data_csv):
        return {}

    configs: dict[str, dict] = {}
    try:
        import csv
        with open(data_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model_name", "")
                cfg_json = row.get("model_config_json", "")
                if model and cfg_json and model not in configs:
                    try:
                        configs[model] = json.loads(cfg_json)
                    except (json.JSONDecodeError, ValueError):
                        pass
    except Exception:
        pass
    return configs


def _from_perfdb(model_name: str) -> Optional[ModelArchFeatures]:
    configs = _load_perfdb_configs()
    # Try exact match first, then normalised
    cfg = configs.get(model_name) or configs.get(normalize_model_name(model_name))
    if cfg is None:
        # Try case-insensitive suffix match
        lower = model_name.lower()
        for key, val in configs.items():
            if key.lower().endswith(lower.split("/")[-1]):
                cfg = val
                break
    if cfg is None:
        return None
    return _features_from_hf_config(cfg, source="perfdb")


def _from_hf_hub(model_name: str) -> Optional[ModelArchFeatures]:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        import urllib.request
        url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            cfg = json.loads(resp.read())
        return _features_from_hf_config(cfg, source="hf_hub")
    except Exception:
        return None


def _features_from_hf_config(cfg: dict, source: str) -> ModelArchFeatures:
    archs = cfg.get("architectures", ["LlamaForCausalLM"])
    arch_class = _ARCH_CLASS_MAP.get(archs[0], archs[0]) if archs else "LlamaForCausalLM"

    num_layers = cfg.get("num_hidden_layers", 32)
    hidden_size = cfg.get("hidden_size", 4096)
    num_heads = cfg.get("num_attention_heads", 32)
    num_kv = cfg.get("num_key_value_heads", num_heads)
    intermediate = cfg.get("intermediate_size", hidden_size * 4)
    vocab_size = cfg.get("vocab_size", 32000)
    max_pos = cfg.get("max_position_embeddings", 4096)

    # MoE fields
    num_experts = cfg.get("num_experts", cfg.get("num_local_experts", 0))
    num_experts_active = cfg.get(
        "num_experts_per_tok",
        cfg.get("num_selected_experts", cfg.get("top_k", 0)),
    )
    is_moe = num_experts > 1

    # Param estimate: use active params for MoE
    active_intermediate = intermediate
    if is_moe:
        moe_intermediate = cfg.get("moe_intermediate_size", intermediate)
        # Each token routes through num_experts_active experts
        active_intermediate = moe_intermediate * num_experts_active if num_experts_active else intermediate

    d_head = hidden_size / max(num_heads, 1)
    kv_dim = num_kv * d_head
    qo = 2 * hidden_size * hidden_size
    kv = 2 * hidden_size * kv_dim
    ffn = 3 * hidden_size * active_intermediate
    layer_p = qo + kv + ffn
    embed_p = vocab_size * hidden_size * 2
    params_b = (embed_p + num_layers * layer_p) / 1e9

    return ModelArchFeatures(
        architecture_class=arch_class,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv,
        gqa_ratio=num_heads / max(num_kv, 1),
        intermediate_size=intermediate,
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_active=num_experts_active,
        params_billion=round(params_b, 2),
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        source=source,
    )


def _from_estimate(model_name: str) -> Optional[ModelArchFeatures]:
    size_b = estimate_model_size_from_name(model_name)
    if size_b is None:
        return None

    # Llama-class template scaled by size
    from placement.roofline.model_arch import KNOWN_ARCHITECTURES
    if size_b >= 65:
        tmpl = KNOWN_ARCHITECTURES["llama-3-70b"]
    elif size_b >= 12:
        tmpl = KNOWN_ARCHITECTURES["llama-2-13b"]
    else:
        tmpl = KNOWN_ARCHITECTURES["llama-3-8b"]

    return ModelArchFeatures(
        architecture_class="LlamaForCausalLM",
        num_layers=tmpl.num_hidden_layers,
        hidden_size=tmpl.hidden_size,
        num_attention_heads=tmpl.num_attention_heads,
        num_kv_heads=tmpl.num_kv_heads,
        gqa_ratio=tmpl.num_attention_heads / max(tmpl.num_kv_heads, 1),
        intermediate_size=tmpl.intermediate_size,
        is_moe=False,
        num_experts=0,
        num_experts_active=0,
        params_billion=round(size_b, 2),
        vocab_size=tmpl.vocab_size,
        max_position_embeddings=tmpl.max_position_embeddings,
        source="estimate",
    )


def fetch_arch_features(model_name: str) -> ModelArchFeatures:
    """
    Return architecture features for model_name.

    Tries in order: registry → perfdb → HF Hub → size estimate.
    Never raises — falls back to a Llama-7B template if everything fails.
    """
    for fn in (_from_registry, _from_perfdb, _from_hf_hub, _from_estimate):
        result = fn(model_name)
        if result is not None:
            return result

    # Last resort: 7B Llama template
    from placement.roofline.model_arch import KNOWN_ARCHITECTURES
    tmpl = KNOWN_ARCHITECTURES["llama-3-8b"]
    return ModelArchFeatures(
        architecture_class="LlamaForCausalLM",
        num_layers=tmpl.num_hidden_layers,
        hidden_size=tmpl.hidden_size,
        num_attention_heads=tmpl.num_attention_heads,
        num_kv_heads=tmpl.num_kv_heads,
        gqa_ratio=tmpl.num_attention_heads / max(tmpl.num_kv_heads, 1),
        intermediate_size=tmpl.intermediate_size,
        is_moe=False,
        num_experts=0,
        num_experts_active=0,
        params_billion=7.0,
        vocab_size=tmpl.vocab_size,
        max_position_embeddings=tmpl.max_position_embeddings,
        source="estimate",
    )
