"""
Architecture-aware FAISS retrieval over the aiconfigurator performance database.

On first call, builds a FAISS flat-IP index from data.csv and caches it to
placement/advisor/data/index.faiss + meta.pkl. Subsequent calls load from cache.

Embedding (12 dims, all in [0,1]):
  [arch_class_0..4 (one-hot 5),  params_norm, gqa_ratio_norm, is_moe,
   num_experts_norm, gpu_hash (one-hot 6 = dims 9..14... wait let's flatten)

Actually we encode as a flat float32 vector:
  arch_onehot[5] + gpu_onehot[6] + [params_norm, gqa_norm, is_moe,
   experts_norm, tp_norm, pp_norm, input_norm, output_norm,
   bandwidth_per_param_norm, vram_headroom_norm]
  = 5 + 6 + 10 = 21 dims
"""

from __future__ import annotations

import csv
import os
import pickle
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCH_CLASSES = [
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "NemotronHForCausalLM",
    "DeciLMForCausalLM",
]
_ARCH_IDX = {a: i for i, a in enumerate(_ARCH_CLASSES)}

_GPU_TYPES = ["A100", "B200", "GB200", "H100_SXM", "H200_SXM", "L40S"]
_GPU_IDX = {g: i for i, g in enumerate(_GPU_TYPES)}

# Normalisation denominators
_MAX_PARAMS = 500.0       # billion
_MAX_GQA = 64.0
_MAX_EXPERTS = 128.0
_MAX_TP = 8.0
_MAX_PP = 8.0
_MAX_INPUT = 32768.0
_MAX_OUTPUT = 16384.0
_MAX_BW_PER_PARAM = 1.0   # already small values in CSV
_MAX_VRAM_HEADROOM = 140.0

_DATA_CSV = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "../../LLM_placement_solver/llm_advisor/data/aiconfigurator/data.csv",
    )
)
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
_INDEX_PATH = os.path.join(_CACHE_DIR, "index.faiss")
_META_PATH = os.path.join(_CACHE_DIR, "meta.pkl")

EMBED_DIM = 21  # 5 arch + 6 gpu + 10 numeric


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_row(
    arch_class: str,
    gpu_model: str,
    params_b: float,
    gqa_ratio: float,
    is_moe: float,
    num_experts_active: float,
    tp: float,
    pp: float,
    avg_input: float,
    avg_output: float,
    bandwidth_per_param: float,
    vram_headroom: float,
) -> np.ndarray:
    vec = np.zeros(EMBED_DIM, dtype=np.float32)

    # One-hot arch (dims 0-4)
    arch_i = _ARCH_IDX.get(arch_class, 0)
    vec[arch_i] = 1.0

    # One-hot GPU (dims 5-10)
    gpu_i = _GPU_IDX.get(gpu_model, 0)
    vec[5 + gpu_i] = 1.0

    # Numeric (dims 11-20)
    vec[11] = min(params_b / _MAX_PARAMS, 1.0)
    vec[12] = min(gqa_ratio / _MAX_GQA, 1.0)
    vec[13] = float(bool(is_moe))
    vec[14] = min(num_experts_active / _MAX_EXPERTS, 1.0)
    vec[15] = min(tp / _MAX_TP, 1.0)
    vec[16] = min(pp / _MAX_PP, 1.0)
    vec[17] = min(avg_input / _MAX_INPUT, 1.0)
    vec[18] = min(avg_output / _MAX_OUTPUT, 1.0)
    vec[19] = min(abs(bandwidth_per_param) / _MAX_BW_PER_PARAM, 1.0)
    vec[20] = min(abs(vram_headroom) / _MAX_VRAM_HEADROOM, 1.0)

    # L2 normalise for cosine via inner product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val not in ("", None, "nan", "NaN") else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        return int(float(val)) if val not in ("", None, "nan", "NaN") else default
    except (ValueError, TypeError):
        return default


def _build_index(data_csv: str):
    """Read data.csv, build FAISS flat-IP index, return (index, rows)."""
    import faiss

    rows: list[dict] = []
    vectors: list[np.ndarray] = []

    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arch = row.get("model_architecture", "LlamaForCausalLM") or "LlamaForCausalLM"
            gpu = row.get("gpu_model", "A100") or "A100"

            # Input/output: prefer avg, fall back to fixed
            avg_in = _safe_float(row.get("input_len_tokens_avg") or row.get("input_len_tokens_fixed"))
            avg_out = _safe_float(row.get("output_len_tokens_avg") or row.get("output_len_tokens_fixed"))

            vec = _embed_row(
                arch_class=arch,
                gpu_model=gpu,
                params_b=_safe_float(row.get("params_billion")),
                gqa_ratio=_safe_float(row.get("attention_heads_per_kv_head"), 1.0),
                is_moe=_safe_float(row.get("is_moe")),
                num_experts_active=_safe_float(row.get("num_experts_active")),
                tp=_safe_float(row.get("tp"), 1.0),
                pp=_safe_float(row.get("pp"), 1.0),
                avg_input=avg_in,
                avg_output=avg_out,
                bandwidth_per_param=_safe_float(row.get("bandwidth_per_param")),
                vram_headroom=_safe_float(row.get("vram_headroom_gb")),
            )
            vectors.append(vec)
            rows.append(row)

    matrix = np.stack(vectors, axis=0).astype(np.float32)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)
    return index, rows


def _get_index():
    """Return (faiss_index, rows). Builds and caches on first call."""
    if not os.path.exists(_DATA_CSV):
        raise FileNotFoundError(
            f"Performance database not found at {_DATA_CSV}.\n"
            "Download it with:\n"
            "  curl -L https://github.com/Tandemn-Labs/LLM_placement_solver/releases/download/aiconfigurator-v1/data.csv \\\n"
            f"    -o {_DATA_CSV}"
        )

    os.makedirs(_CACHE_DIR, exist_ok=True)

    if os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH):
        import faiss
        index = faiss.read_index(_INDEX_PATH)
        with open(_META_PATH, "rb") as f:
            rows = pickle.load(f)
        return index, rows

    print("[advisor] Building FAISS index from performance database (one-time, ~30s)...")
    index, rows = _build_index(_DATA_CSV)

    import faiss
    faiss.write_index(index, _INDEX_PATH)
    with open(_META_PATH, "wb") as f:
        pickle.dump(rows, f)
    print(f"[advisor] Index cached ({len(rows):,} rows)")
    return index, rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    arch_class: str,
    params_billion: float,
    gqa_ratio: float,
    is_moe: bool,
    num_experts_active: int,
    gpu_model: str,
    tp: int,
    pp: int,
    avg_input: int,
    avg_output: int,
    k: int = 10,
) -> List[dict]:
    """
    Return the k most architecturally similar rows from data.csv for a given
    (gpu_model, tp, pp) configuration.
    """
    index, rows = _get_index()

    query = _embed_row(
        arch_class=arch_class,
        gpu_model=gpu_model,
        params_b=params_billion,
        gqa_ratio=gqa_ratio,
        is_moe=float(is_moe),
        num_experts_active=float(num_experts_active),
        tp=float(tp),
        pp=float(pp),
        avg_input=float(avg_input),
        avg_output=float(avg_output),
        bandwidth_per_param=0.0,
        vram_headroom=0.0,
    ).reshape(1, -1)

    n_results = min(k, index.ntotal)
    distances, indices = index.search(query, n_results)
    return [rows[i] for i in indices[0] if i < len(rows)]


def retrieve_multi(
    arch_class: str,
    params_billion: float,
    gqa_ratio: float,
    is_moe: bool,
    num_experts_active: int,
    feasible_configs: list[tuple[str, int, int]],  # [(gpu_model, tp, pp), ...]
    avg_input: int,
    avg_output: int,
    k_per_config: int = 5,
) -> List[dict]:
    """
    Sweep over all feasible (gpu_model, tp, pp) combos, retrieve k per combo,
    deduplicate by (model_name, gpu_model, tp, pp), return up to k_per_config * len(configs) unique rows.
    """
    seen: set[tuple] = set()
    results: list[dict] = []

    for gpu_model, tp, pp in feasible_configs:
        hits = retrieve(
            arch_class=arch_class,
            params_billion=params_billion,
            gqa_ratio=gqa_ratio,
            is_moe=is_moe,
            num_experts_active=num_experts_active,
            gpu_model=gpu_model,
            tp=tp,
            pp=pp,
            avg_input=avg_input,
            avg_output=avg_output,
            k=k_per_config,
        )
        for row in hits:
            key = (
                row.get("model_name", ""),
                row.get("gpu_model", ""),
                row.get("tp", ""),
                row.get("pp", ""),
            )
            if key not in seen:
                seen.add(key)
                results.append(row)

    return results
