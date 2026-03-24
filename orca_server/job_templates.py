"""
Template generation for vLLM SkyPilot YAML configs.
"""

import os
import uuid
from typing import Union

from orca_server.config import INSTANCE_TO_GPU, VLLM_PORT
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput
from utils.utils import split_uri


def get_vllm_config_template(
    model_name: str, instance_type: str, tp: int, pp: int, logger=None
) -> str:
    """
    Get the per-config vLLM template path if it exists.

    Template lookup order:
    1. vllm_{model}-{gpu}-tp{tp}-pp{pp}.yaml  (specific model+config)
    2. vllm_{gpu}.yaml                         (GPU-specific generic)
    3. vllm.yaml                               (generic fallback)

    Example: vllm_qwen2.5-72b-A100-tp8-pp1.yaml

    Returns path to specific template if exists, otherwise returns generic template.
    """
    _log = logger.info if logger else print

    # Normalize model name (e.g., "Qwen/Qwen2.5-72B-Instruct" -> "qwen2.5-72b")
    model_short = model_name.lower()
    if "/" in model_short:
        model_short = model_short.split("/")[-1]
    # Remove common suffixes
    for suffix in ["-instruct", "-chat", "-base", "-hf"]:
        model_short = model_short.replace(suffix, "")

    # Get GPU name from instance type
    gpu_name = INSTANCE_TO_GPU.get(instance_type, "unknown")

    # 1. Check for specific model+config template
    template_name = f"vllm_{model_short}-{gpu_name}-tp{tp}-pp{pp}.yaml"
    template_path = f"templates/vllm_configs/{template_name}"
    if os.path.exists(template_path):
        _log(f"[Template] Using per-config template: {template_path}")
        return template_path

    # 2. Check for GPU-specific generic template (e.g., vllm_A100.yaml)
    gpu_template_name = f"vllm_{gpu_name}.yaml"
    gpu_template_path = f"templates/vllm_configs/{gpu_template_name}"
    if os.path.exists(gpu_template_path):
        _log(f"[Template] Using GPU-specific template: {gpu_template_path}")
        return gpu_template_path

    # 3. Fall back to generic template
    _log(f"[Template] No specific template for {template_name}, using generic")
    return "templates/vllm.yaml"


def replace_run_vllm(
    request: BatchedRequest,
    config: MagicOutput,
    job_dirname: str = "output",
    logger=None,
):
    replace = {}

    if request.s3_model_path:
        replace["model"] = f"/models/{request.model_name}"
    else:
        replace["model"] = request.model_name
    replace["tp"] = config.tp_size
    replace["pp"] = config.pp_size

    # Calculate max_model_len:
    # 1. If solver provides it (roofline), use it
    # 2. Else if request has max_input/output_tokens, calculate from those
    # 3. Else use "auto" (let vLLM figure it out)
    if config.max_model_len:
        replace["max_model_len"] = config.max_model_len
    elif request.max_input_tokens and request.max_output_tokens:
        # Use actual max lengths from request + 10% safety margin
        calculated_len = int(
            (request.max_input_tokens + request.max_output_tokens) * 1.1
        )
        # Round up to nearest power of 2 for efficiency, clamp to reasonable range
        calculated_len = max(1024, min(calculated_len, 131072))
        replace["max_model_len"] = calculated_len
        _log = logger.info if logger else print
        _log(
            f"[Config] Calculated max_model_len={calculated_len} from max_input={request.max_input_tokens} + max_output={request.max_output_tokens}"
        )
    else:
        replace["max_model_len"] = "auto"

    _, input_file = split_uri(request.input_file)
    output = request.output_file
    replace["input_file"] = "/data/" + input_file
    # Output goes to informative subdirectory: /data/{job_dirname}/output.jsonl
    replace["output_file"] = f"/data/{job_dirname}/{output}"

    # Infrastructure configuration for metrics tracking
    replace["cloud"] = "aws"  # Currently only AWS supported
    replace["instance_type"] = config.instance_type
    replace["gpu_name"] = INSTANCE_TO_GPU.get(config.instance_type, "unknown")
    replace["engine"] = request.engine or "vllm"
    replace["quantization"] = request.quantization_bits or "none"

    # Get vLLM-specific configs
    vllm_cfg = request.vllm_specific_config
    replace["max_num_seqs"] = (
        vllm_cfg.max_num_seqs if vllm_cfg and vllm_cfg.max_num_seqs else 256
    )
    replace["dtype"] = "auto"  # vLLM auto-detects based on model
    replace["kv_cache_dtype"] = (
        vllm_cfg.kv_cache_dtype if vllm_cfg and vllm_cfg.kv_cache_dtype else "auto"
    )

    if (
        request.vllm_specific_config is not None
        and request.vllm_specific_config.speculative_config is not None
    ):
        prefix = "--speculative-config."
        spec_config = request.vllm_specific_config.speculative_config.model_dump(
            exclude_none=True
        )

        spec_string = ""
        for key, value in spec_config.items():
            string = prefix + key + " " + str(value)
            spec_string += string + " "

        spec_string = spec_string.rstrip(" ")
        replace["additional_params"] = spec_string

    else:
        replace["additional_params"] = ""

    return replace


def replace_run_vllm_online(request: OnlineServingRequest, config: MagicOutput):
    replace = {}
    replace["model"] = request.model_name
    replace["tp"] = config.tp_size
    replace["pp"] = config.pp_size
    replace["host"] = "0.0.0.0"  # Bind to all interfaces (allows external access)
    replace["port"] = str(VLLM_PORT)

    if (
        request.vllm_specific_config is not None
        and request.vllm_specific_config.speculative_config is not None
    ):
        prefix = "--speculative-config."
        spec_config = request.vllm_specific_config.speculative_config.model_dump(
            exclude_none=True
        )

        spec_string = ""
        for key, value in spec_config.items():
            string = prefix + key + " " + str(value)
            spec_string += string + " "

        spec_string = spec_string.rstrip(" ")
        replace["additional_params"] = spec_string

    else:
        replace["additional_params"] = ""

    return replace


### magic.py placeholder
def real_magic(request: Union[BatchedRequest, OnlineServingRequest]) -> MagicOutput:
    return MagicOutput(
        decision_id="mo-" + str(uuid.uuid4()),
        engine="vllm",
        instance_type="g6e.xlarge",
        tp_size=1,
        pp_size=1,
        replicas=1,
    )
