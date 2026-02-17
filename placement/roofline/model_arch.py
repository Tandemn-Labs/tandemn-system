"""
Model architecture definitions and lookup for LLM placement.

Maps model names to their architectural parameters needed for
roofline throughput calculations and memory feasibility checks.

Ported from: ../LLM_placement_solver/llm_advisor/model_arch.py
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelArchitecture:
    """
    Model architecture specification.

    Contains all parameters needed for roofline throughput calculation
    and memory feasibility checks.
    """
    model_id: str
    num_hidden_layers: int
    hidden_size: int  # d_model
    intermediate_size: int  # d_hidden (FFN)
    num_attention_heads: int
    num_kv_heads: int  # For GQA/MQA models
    vocab_size: int
    max_position_embeddings: int

    @property
    def d_model(self) -> int:
        """Alias for hidden_size."""
        return self.hidden_size

    @property
    def d_hidden(self) -> int:
        """Alias for intermediate_size (FFN dimension)."""
        return self.intermediate_size

    @property
    def layer_weight_memory_gb(self) -> float:
        """
        Estimate weight memory per layer in GB (FP16).

        Includes:
        - Attention: Q, K, V, O projections
        - FFN: gate, up, down projections
        """
        d_head = self.hidden_size / self.num_attention_heads
        kv_dim = self.num_kv_heads * d_head

        # Attention weights
        qo_params = 2 * self.hidden_size * self.hidden_size  # Q and O
        kv_params = 2 * self.hidden_size * kv_dim  # K and V

        # FFN weights (LLaMA-style gated FFN)
        ffn_params = 3 * self.hidden_size * self.intermediate_size

        total_params = qo_params + kv_params + ffn_params
        bytes_per_param = 2  # FP16

        return (total_params * bytes_per_param) / (1024 ** 3)

    @property
    def total_params_billions(self) -> float:
        """Estimate total model parameters in billions."""
        d_head = self.hidden_size / self.num_attention_heads
        kv_dim = self.num_kv_heads * d_head

        # Embedding params
        embed_params = self.vocab_size * self.hidden_size * 2  # input + output

        # Per-layer params
        qo_params = 2 * self.hidden_size * self.hidden_size
        kv_params = 2 * self.hidden_size * kv_dim
        ffn_params = 3 * self.hidden_size * self.intermediate_size
        ln_params = 2 * self.hidden_size * 2  # 2 layer norms

        layer_params = qo_params + kv_params + ffn_params + ln_params
        total = embed_params + self.num_hidden_layers * layer_params

        return total / 1e9

    @property
    def fp16_memory_gb(self) -> float:
        """Total model memory in GB (FP16)."""
        return self.total_params_billions * 2  # 2 bytes per param


# Known model architectures (offline cache)
# These are used when HuggingFace lookup fails or for speed
KNOWN_ARCHITECTURES: Dict[str, ModelArchitecture] = {
    # ============ Llama 3 Family ============
    "llama-3-70b": ModelArchitecture(
        model_id="llama-3-70b",
        num_hidden_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "llama-3.1-70b": ModelArchitecture(
        model_id="llama-3.1-70b",
        num_hidden_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "llama-3.3-70b": ModelArchitecture(
        model_id="llama-3.3-70b",
        num_hidden_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "llama-3-8b": ModelArchitecture(
        model_id="llama-3-8b",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "llama-3.1-8b": ModelArchitecture(
        model_id="llama-3.1-8b",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),

    # ============ Llama 2 Family ============
    "llama-2-70b": ModelArchitecture(
        model_id="llama-2-70b",
        num_hidden_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=32000,
        max_position_embeddings=4096,
    ),
    "llama-2-13b": ModelArchitecture(
        model_id="llama-2-13b",
        num_hidden_layers=40,
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_kv_heads=40,
        vocab_size=32000,
        max_position_embeddings=4096,
    ),
    "llama-2-7b": ModelArchitecture(
        model_id="llama-2-7b",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_kv_heads=32,
        vocab_size=32000,
        max_position_embeddings=4096,
    ),

    # ============ DeepSeek Family ============
    # DeepSeek-R1-Distill-Llama uses Llama 3 architecture
    "deepseek-r1-distill-llama-70b": ModelArchitecture(
        model_id="deepseek-r1-distill-llama-70b",
        num_hidden_layers=80,
        hidden_size=8192,
        intermediate_size=28672,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),
    "deepseek-r1-distill-llama-8b": ModelArchitecture(
        model_id="deepseek-r1-distill-llama-8b",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=128256,
        max_position_embeddings=131072,
    ),

    # ============ Qwen Family ============
    "qwen2-72b": ModelArchitecture(
        model_id="qwen2-72b",
        num_hidden_layers=80,
        hidden_size=8192,
        intermediate_size=29568,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=152064,
        max_position_embeddings=131072,
    ),
    "qwen2-7b": ModelArchitecture(
        model_id="qwen2-7b",
        num_hidden_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_kv_heads=4,
        vocab_size=152064,
        max_position_embeddings=131072,
    ),

    # ============ Mistral Family ============
    "mistral-7b": ModelArchitecture(
        model_id="mistral-7b",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=32000,
        max_position_embeddings=32768,
    ),
    "mixtral-8x7b": ModelArchitecture(
        model_id="mixtral-8x7b",
        num_hidden_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=32000,
        max_position_embeddings=32768,
    ),
}


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for lookup in KNOWN_ARCHITECTURES.

    Handles variations like:
    - "llama-3-70b", "llama3-70b", "Llama-3-70B"
    - "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    Args:
        model_name: Raw model name string

    Returns:
        Normalized lowercase model name
    """
    if not model_name:
        return ""

    name = model_name.lower().strip()

    # Remove HuggingFace org prefixes
    if "/" in name:
        name = name.split("/")[-1]

    # Normalize separators
    name = name.replace("_", "-")

    # Handle specific patterns
    patterns = [
        # DeepSeek Llama distillations
        (r"deepseek.*r1.*distill.*llama.*70b", "deepseek-r1-distill-llama-70b"),
        (r"deepseek.*r1.*distill.*llama.*8b", "deepseek-r1-distill-llama-8b"),

        # Llama 3.x variants (all use same 70B arch)
        (r"llama[-.]?3\.?[13]?[-.]?70b", "llama-3-70b"),
        (r"llama[-.]?3\.?[13]?[-.]?8b", "llama-3-8b"),

        # Llama 2 variants
        (r"llama[-.]?2[-.]?70b", "llama-2-70b"),
        (r"llama[-.]?2[-.]?13b", "llama-2-13b"),
        (r"llama[-.]?2[-.]?7b", "llama-2-7b"),

        # Generic Llama (assume Llama 3)
        (r"^llama[-.]?70b", "llama-3-70b"),
        (r"^llama[-.]?8b", "llama-3-8b"),

        # Qwen
        (r"qwen2?[-.]?72b", "qwen2-72b"),
        (r"qwen2?[-.]?7b", "qwen2-7b"),

        # Mistral
        (r"mistral[-.]?7b", "mistral-7b"),
        (r"mixtral[-.]?8x7b", "mixtral-8x7b"),
    ]

    for pattern, normalized in patterns:
        if re.search(pattern, name):
            return normalized

    return name


def get_model_architecture(model_name: str) -> Optional[ModelArchitecture]:
    """
    Get model architecture by name.

    First tries to match against known architectures.
    Returns None if model is not found.

    Args:
        model_name: Model name (e.g., "llama-3-70b", "deepseek-r1-distill-llama-70b")

    Returns:
        ModelArchitecture or None if not found
    """
    normalized = normalize_model_name(model_name)

    # Direct lookup
    if normalized in KNOWN_ARCHITECTURES:
        return KNOWN_ARCHITECTURES[normalized]

    # Try without version numbers
    for key, arch in KNOWN_ARCHITECTURES.items():
        if normalized.replace(".", "-") in key or key in normalized:
            return arch

    return None


def estimate_model_size_from_name(model_name: str) -> Optional[float]:
    """
    Extract model size in billions from model name.

    Args:
        model_name: Model name containing size (e.g., "llama-70b")

    Returns:
        Model size in billions, or None if not found
    """
    if not model_name:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)[bB](?:illion)?", model_name)
    if match:
        return float(match.group(1))

    return None


def get_model_architecture_or_estimate(model_name: str) -> Optional[ModelArchitecture]:
    """
    Get model architecture, falling back to size-based estimation.

    If exact architecture is not known, estimates based on model size
    using Llama-like architecture as template.

    Args:
        model_name: Model name

    Returns:
        ModelArchitecture (exact or estimated), or None if size unknown
    """
    # Try exact lookup first
    arch = get_model_architecture(model_name)
    if arch:
        return arch

    # Estimate based on size
    size_b = estimate_model_size_from_name(model_name)
    if not size_b:
        return None

    # Use Llama architecture as template based on size
    if size_b >= 65:
        # ~70B class: use Llama-3-70B architecture
        template = KNOWN_ARCHITECTURES["llama-3-70b"]
    elif size_b >= 12:
        # ~13B class: use Llama-2-13B architecture
        template = KNOWN_ARCHITECTURES["llama-2-13b"]
    elif size_b >= 6:
        # ~7-8B class: use Llama-3-8B architecture
        template = KNOWN_ARCHITECTURES["llama-3-8b"]
    else:
        # Smaller models: scale down from 7B
        template = KNOWN_ARCHITECTURES["llama-2-7b"]

    # Return template with updated model_id
    return ModelArchitecture(
        model_id=f"{model_name}-estimated",
        num_hidden_layers=template.num_hidden_layers,
        hidden_size=template.hidden_size,
        intermediate_size=template.intermediate_size,
        num_attention_heads=template.num_attention_heads,
        num_kv_heads=template.num_kv_heads,
        vocab_size=template.vocab_size,
        max_position_embeddings=template.max_position_embeddings,
    )
