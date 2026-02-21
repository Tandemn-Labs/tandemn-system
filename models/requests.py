"""
This is the pydantic schema for the cli to send to the api_gateway
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, model_validator

class BatchedRequest(BaseModel):
    """Request to send batched inference job to central server."""
    user_id: str
    input_file: str  # S3/local path to the file
    output_file: str  # local path to where the output will be saved
    # Input stats - extracted from input_file by server, but can be overridden
    num_lines: Optional[int] = None  # number of prompt lines in the file (parsed from input_file)
    avg_input_tokens: Optional[int] = None  # avg per line (parsed from input_file)
    max_input_tokens: Optional[int] = None  # max input length (parsed from input_file)
    # Output stats - must be provided by user (not in input file)
    avg_output_tokens: int  # expected avg output per line
    max_output_tokens: Optional[int] = None  # max expected output length
    # Get some parameters directly from the JobConfig
    description : str
    task_type : str
    task_priority : str
    model_name: Optional[str] = None
    engine: str
    quantization_bits: Optional[Literal["4", "8", "16"]] = None
    is_speculative_decode: Optional[bool] = None # none means not specified
    is_PD_disaggregation: Optional[bool] = None # none means not specified
    slo_mode : str
    slo_deadline_hours: int = None
    placement : str
    # Placement solver selection:
    #   "roofline"       — deterministic roofline-based solver (default)
    #   "llm"            — LLM-based 3-advisor + C-PMI solver
    #   "user_specified" — user provides gpu_type/tp_size/pp_size; validates feasibility only
    placement_solver: Optional[Literal["roofline", "llm", "user_specified"]] = None
    # LLM advisor model tier: "free" (Gemini, DeepSeek-R1) or "paid" (Claude, DeepSeek-chat)
    llm_advisor_tier: Optional[Literal["free", "paid"]] = "free"
    # User-specified placement (only used when placement_solver="user_specified")
    gpu_type: Optional[str] = None      # e.g. "A100", "L40S", "H100"
    tp_size: Optional[int] = None       # tensor parallelism (1,2,4,8)
    pp_size: Optional[int] = None       # pipeline parallelism (1,2,3,4)

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
    # HuggingFace token for gated models (Llama, etc.)
    hf_token: Optional[str] = None
    # OpenRouter API key for LLM-based placement solver
    openrouter_api_key: Optional[str] = None
    # Only change the ModelSpecificCofig
    # right now its just vllm, but we can interject the parameters here
    vllm_specific_config: Optional[vLLMSpecificConfig] = None
    # we will add this when we get there
    # ----- sglang_specific_config: SGLangSpecificConfig ----
    # ----- diffusers_specific_config: DiffusersSpecificConfig ----
    # ----- xDIT_specific_config: XDITSpecificConfig ----

class OnlineServingRequest(BaseModel):
    """Request to send online inference job to central server."""
    user_id: str
    description: str
    task_type: str
    task_priority: Optional[str] = None
    model_name: str
    engine: str
    quantization_bits: Optional[Literal["4", "8", "16"]] = None
    is_speculative_decode: Optional[bool] = None
    is_PD_disaggregation: Optional[bool] = None
    slo_mode: Optional[str] = "online" # the slo of online serving is always online
    placement: str
    # HuggingFace token for gated models (Llama, etc.)
    hf_token: Optional[str] = None

    # # online-specific config
    # port: Optional[int] = 8000  # we will use the default port 8000 *(the cli user has no idea what to do?)
    # host: Optional[str] = "0.0.0.0" # we will use the default host 0.0.0.0 
    vllm_specific_config: Optional[vLLMSpecificConfig] = None

################## VLLM Specific Config #####################
#### These parameters will be filled both from the LLM parameters and with
# Rules that we will create
 
class SpeculativeConfig(BaseModel):
    method: Optional[Literal["ngram", "suffix", "eagle", "eagle3",  "mlp_speculator", "draft_model", "mtp", "deepseek_mtp", "glm4_moe_mtp", "qwen3_next_mtp", "longcat_flash_mtp"]] = None
    model: Optional[str] = None
    draft_tensor_parallel_size: Optional[int] = 1 # use default for now
    num_speculative_tokens: Optional[int] = None
    max_model_len: Optional[int] = None
    revision: Optional[str] = None
    prompt_lookup_max: Optional[int] = None

class vLLMSpecificConfig(BaseModel):
    # persona 3 config
    max_model_len: Optional[int] = None
    trust_remote_code: Optional[bool] = True # default is True
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    config_format: Optional[Literal["auto", "mistral"]] = "auto" # use mistral if mistral model is detected
    # persona 2 config
    tokenizer: Optional[str] = None # leave this field blank if not specified (just the path of the tokenizer)
    tokenizer_mode: Optional[Literal["auto","deepseek","mistral"]] = "auto" # Leave this to Auto if not specified
    kv_cache_dtype: Optional[Literal["auto","bfloat16","fp8","fp8_ds_mla","fp8_e4m3","fp8_e5m2","fp8_inc"]] = None
    speculative_config: Optional[SpeculativeConfig] = None
    limit_mm_per_prompt: Optional[int] = 20 # default is 20
