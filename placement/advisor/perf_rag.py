"""
Two-stage architecture-aware retrieval over the aiconfigurator performance database.

Stage 1 (FAISS): Find architecturally similar profiled models.
  Input:  model name / arch features (from user)
  Output: all profiled records for similar models, across ALL GPU/TP/PP configs
  Embedding: pure architecture features (no hardware) — 9 dims:
    [params_norm, num_layers_norm, hidden_size_norm, gqa_ratio_norm,
     intermediate_ratio_norm, is_moe, num_experts_active_norm,
     input_norm, output_norm]

Stage 2 (Oracle): Hard-filter RAG results by (gpu, tp, pp) per candidate.
  This happens in oracle.py, not here.

On first call, builds index from data.csv and caches to placement/advisor/data/.
"""

from __future__ import annotations

import csv
import os
import pickle
from typing import List

import numpy as np

from placement.advisor._utils import safe_float as _safe_float

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Normalisation denominators (based on actual data.csv ranges)
_MAX_PARAMS = 500.0
_MAX_LAYERS = 128.0
_MAX_HIDDEN = 16384.0
_MAX_GQA = 64.0
_MAX_INTERMEDIATE_RATIO = 10.0  # intermediate / hidden, typically 3-5
_MAX_EXPERTS = 128.0
_MAX_INPUT = 32768.0
_MAX_OUTPUT = 16384.0

_DATA_CSV = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "../../LLM_placement_solver/llm_advisor/data/aiconfigurator/data.csv",
    )
)
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
_INDEX_PATH = os.path.join(_CACHE_DIR, "arch_index.faiss")
_META_PATH = os.path.join(_CACHE_DIR, "arch_meta.pkl")

EMBED_DIM = 9  # pure architecture + I/O shape


# ---------------------------------------------------------------------------
# Embedding — pure architecture features, no hardware
# ---------------------------------------------------------------------------

def _embed(
    params_b: float,
    num_layers: float,
    hidden_size: float,
    gqa_ratio: float,
    intermediate_ratio: float,
    is_moe: float,
    num_experts_active: float,
    avg_input: float,
    avg_output: float,
) -> np.ndarray:
    vec = np.array([
        min(params_b / _MAX_PARAMS, 1.0),
        min(num_layers / _MAX_LAYERS, 1.0),
        min(hidden_size / _MAX_HIDDEN, 1.0),
        min(gqa_ratio / _MAX_GQA, 1.0),
        min(intermediate_ratio / _MAX_INTERMEDIATE_RATIO, 1.0),
        float(bool(is_moe)),
        min(num_experts_active / _MAX_EXPERTS, 1.0),
        min(avg_input / _MAX_INPUT, 1.0),
        min(avg_output / _MAX_OUTPUT, 1.0),
    ], dtype=np.float32)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------

def _build_index(data_csv: str):
    """Read data.csv, build FAISS flat-IP index on arch features, return (index, rows)."""
    import faiss

    rows: list[dict] = []
    vectors: list[np.ndarray] = []

    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = _safe_float(row.get("params_billion"))
            hidden = _safe_float(row.get("hidden_size") or
                                 _extract_from_config(row, "hidden_size"), 4096)
            num_layers = _safe_float(row.get("num_hidden_layers") or
                                     _extract_from_config(row, "num_hidden_layers"), 32)
            gqa = _safe_float(row.get("attention_heads_per_kv_head"), 1.0)
            intermediate = _safe_float(
                _extract_from_config(row, "intermediate_size"), hidden * 4)
            intermediate_ratio = intermediate / max(hidden, 1)
            is_moe = _safe_float(row.get("is_moe"))
            experts_active = _safe_float(row.get("num_experts_active"))
            avg_in = _safe_float(row.get("input_len_tokens_avg") or
                                 row.get("input_len_tokens_fixed"))
            avg_out = _safe_float(row.get("output_len_tokens_avg") or
                                  row.get("output_len_tokens_fixed"))

            vec = _embed(
                params_b=params,
                num_layers=num_layers,
                hidden_size=hidden,
                gqa_ratio=gqa,
                intermediate_ratio=intermediate_ratio,
                is_moe=is_moe,
                num_experts_active=experts_active,
                avg_input=avg_in,
                avg_output=avg_out,
            )
            vectors.append(vec)
            rows.append(row)

    matrix = np.stack(vectors, axis=0).astype(np.float32)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)
    return index, rows


def _extract_from_config(row: dict, key: str, default=None):
    """Try to extract a field from the model_config_json column."""
    import json
    cfg_str = row.get("model_config_json", "")
    if not cfg_str:
        return default
    try:
        cfg = json.loads(cfg_str)
        return cfg.get(key, default)
    except (json.JSONDecodeError, ValueError):
        return default


def _get_index():
    """Return (faiss_index, rows). Builds and caches on first call."""
    if not os.path.exists(_DATA_CSV):
        raise FileNotFoundError(
            f"Performance database not found at {_DATA_CSV}.\n"
            "Download it with:\n"
            "  curl -L https://github.com/Tandemn-Labs/LLM_placement_solver/"
            "releases/download/aiconfigurator-v1/data.csv \\\n"
            f"    -o {_DATA_CSV}"
        )

    os.makedirs(_CACHE_DIR, exist_ok=True)

    # Invalidate cache if CSV is newer
    if (os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH)
            and os.path.getmtime(_INDEX_PATH) >= os.path.getmtime(_DATA_CSV)):
        import faiss
        index = faiss.read_index(_INDEX_PATH)
        with open(_META_PATH, "rb") as f:
            rows = pickle.load(f)
        return index, rows

    print("[advisor] Building FAISS arch index from performance database (one-time, ~30s)...")
    index, rows = _build_index(_DATA_CSV)

    import faiss
    faiss.write_index(index, _INDEX_PATH)
    with open(_META_PATH, "wb") as f:
        pickle.dump(rows, f)
    print(f"[advisor] Arch index cached ({len(rows):,} rows)")
    return index, rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    params_billion: float,
    num_layers: int,
    hidden_size: int,
    gqa_ratio: float,
    intermediate_size: int,
    is_moe: bool,
    num_experts_active: int,
    avg_input: int,
    avg_output: int,
    k: int = 50,
) -> List[dict]:
    """
    Find the k most architecturally similar profiled records.

    Returns rows across ALL GPUs/TP/PP configs. The caller (oracle)
    hard-filters to specific (gpu, tp, pp) candidates.
    """
    index, rows = _get_index()

    intermediate_ratio = intermediate_size / max(hidden_size, 1)
    query = _embed(
        params_b=params_billion,
        num_layers=float(num_layers),
        hidden_size=float(hidden_size),
        gqa_ratio=gqa_ratio,
        intermediate_ratio=intermediate_ratio,
        is_moe=float(is_moe),
        num_experts_active=float(num_experts_active),
        avg_input=float(avg_input),
        avg_output=float(avg_output),
    ).reshape(1, -1)

    n_results = min(k, index.ntotal)
    distances, indices = index.search(query, n_results)
    return [rows[i] for i in indices[0] if i < len(rows)]
