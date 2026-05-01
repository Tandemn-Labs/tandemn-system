"""
This is the pydantic schema for the cli to send to the api_gateway
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class KoiPlacementAlternative(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_type: str
    tp: int
    pp: int
    dp: int = 1
    planned_market: Literal["spot", "on_demand"] | None = None


class BatchedRequest(BaseModel):
    """Request to send batched inference job to central server."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    input_file: str  # S3/local path to the file
    output_file: str  # local path to where the output will be saved
    # Input stats - extracted from input_file by server, but can be overridden
    num_lines: int | None = None  # number of prompt lines in the file (parsed from input_file)
    avg_input_tokens: int | None = None  # avg per line (parsed from input_file)
    max_input_tokens: int | None = None  # max input length (parsed from input_file)
    # Output stats - must be provided by user (not in input file)
    avg_output_tokens: int | None  # expected avg output per line
    max_output_tokens: int | None = None  # max expected output length
    # Get some parameters directly from the JobConfig
    description: str
    task_type: str
    task_priority: str
    model_name: str | None = None
    engine: str
    quantization_bits: Literal["4", "8", "16"] | None = None
    is_speculative_decode: bool | None = None  # none means not specified
    is_PD_disaggregation: bool | None = None  # none means not specified
    slo_mode: str
    slo_deadline_hours: float = 4
    placement: str
    # Placement solver selection:
    #   "roofline"       — deterministic roofline-based solver (default)
    #   "llm"            — LLM-based 3-advisor + C-PMI solver
    #   "user_specified" — user provides gpu_type/tp_size/pp_size; validates feasibility only
    placement_solver: Literal["roofline", "llm", "user_specified"] | None = None
    # LLM advisor model tier: "free" (Gemini, DeepSeek-R1) or "paid" (Claude, DeepSeek-chat)
    llm_advisor_tier: Literal["free", "paid"] | None = "free"
    # User-specified placement (only used when placement_solver="user_specified")
    gpu_type: str | None = None  # e.g. "A100", "L40S", "H100"
    tp_size: int | None = None  # tensor parallelism (1,2,4,8)
    pp_size: int | None = None  # pipeline parallelism (1,2,3,4)

    # Skip feasibility check for user_specified placement (launch even if infeasible)
    force: bool | None = False

    # Keep cluster alive after job completes (default: destroy)
    persist: bool | None = False

    # Use on-demand instances instead of spot (default: prefer spot)
    prefer_spot: bool | None = True
    preferred_market: Literal["spot", "on_demand"] | None = None
    planned_market: Literal["spot", "on_demand"] | None = None

    # S3 URI to model weights — if set, loads from S3 instead of HuggingFace
    s3_model_path: str | None = None
    # HuggingFace token for gated models (Llama, etc.)
    hf_token: str | None = None
    # OpenRouter API key for LLM-based placement solver
    openrouter_api_key: str | None = None
    # Solver log level: "debug", "info", "warning" (default: "info")
    log_level: str | None = None
    # Chunked distributed batch inference
    replicas: int | None = None  # Replica count (from CLI --replicas or solver)
    chunk_size: int | None = None  # Lines per chunk (default: 1000)
    chunks: list[dict] | None = None  # [{chunk_id, s3_input_path, num_lines}, ...] — CLI uploads
    koi_alternatives: list[KoiPlacementAlternative] | None = None  # Koi placement alternatives for fallback retry
    koi_decision_id: str | None = None  # Koi decision ID — passed back via /job/started webhook
    koi_predicted_tps: float | None = None  # Koi top-level TPS prediction for webhook propagation

    # Only change the ModelSpecificCofig
    # right now its just vllm, but we can interject the parameters here
    vllm_specific_config: vLLMSpecificConfig | None = None
    # we will add this when we get there
    # ----- sglang_specific_config: SGLangSpecificConfig ----
    # ----- diffusers_specific_config: DiffusersSpecificConfig ----
    # ----- xDIT_specific_config: XDITSpecificConfig ----

    @model_validator(mode="after")
    def _validate_user_specified(self):
        if self.placement_solver == "user_specified":
            if self.gpu_type is None:
                raise ValueError("gpu_type is required when placement_solver='user_specified'")
            if self.tp_size is not None and self.tp_size not in {1, 2, 4, 8}:
                raise ValueError(f"tp_size must be one of {{1, 2, 4, 8}}, got {self.tp_size}")
            if self.pp_size is not None and self.pp_size not in {1, 2, 3, 4}:
                raise ValueError(f"pp_size must be one of {{1, 2, 3, 4}}, got {self.pp_size}")
        return self


class OnlineServingRequest(BaseModel):
    """Request to send online inference job to central server."""

    user_id: str
    description: str
    task_type: str
    task_priority: str | None = None
    model_name: str
    engine: str
    quantization_bits: Literal["4", "8", "16"] | None = None
    is_speculative_decode: bool | None = None
    is_PD_disaggregation: bool | None = None
    slo_mode: str | None = "online"  # the slo of online serving is always online
    placement: str
    # HuggingFace token for gated models (Llama, etc.)
    hf_token: str | None = None

    # # online-specific config
    # port: Optional[int] = 8000  # we will use the default port 8000 *(the cli user has no idea what to do?)
    # host: Optional[str] = "0.0.0.0" # we will use the default host 0.0.0.0
    vllm_specific_config: vLLMSpecificConfig | None = None


################## VLLM Specific Config #####################
#### These parameters will be filled both from the LLM parameters and with
# Rules that we will create


class SpeculativeConfig(BaseModel):
    method: (
        Literal[
            "ngram",
            "suffix",
            "eagle",
            "eagle3",
            "mlp_speculator",
            "draft_model",
            "mtp",
            "deepseek_mtp",
            "glm4_moe_mtp",
            "qwen3_next_mtp",
            "longcat_flash_mtp",
        ]
        | None
    ) = None
    model: str | None = None
    draft_tensor_parallel_size: int | None = 1  # use default for now
    num_speculative_tokens: int | None = None
    max_model_len: int | None = None
    revision: str | None = None
    prompt_lookup_max: int | None = None


class vLLMSpecificConfig(BaseModel):
    # persona 3 config
    max_model_len: int | None = None
    trust_remote_code: bool | None = True  # default is True
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    config_format: Literal["auto", "mistral"] | None = "auto"  # use mistral if mistral model is detected
    # persona 2 config
    tokenizer: str | None = None  # leave this field blank if not specified (just the path of the tokenizer)
    tokenizer_mode: Literal["auto", "deepseek", "mistral"] | None = "auto"  # Leave this to Auto if not specified
    kv_cache_dtype: Literal["auto", "bfloat16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"] | None = None
    speculative_config: SpeculativeConfig | None = None
    limit_mm_per_prompt: int | None = 20  # default is 20
