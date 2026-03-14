import pytest
from pydantic import ValidationError
from models.requests import BatchedRequest, OnlineServingRequest, vLLMSpecificConfig
from models.resources import MagicOutput


BATCH_DEFAULTS = dict(
    user_id="test",
    input_file="s3://bucket/input.jsonl",
    output_file="s3://bucket/output.jsonl",
    avg_output_tokens=100,
    description="test job",
    task_type="inference",
    task_priority="low",
    engine="vllm",
    slo_mode="deadline",
    placement="aws",
)


# --- BatchedRequest ---


def test_batched_request_valid():
    req = BatchedRequest(**BATCH_DEFAULTS, placement_solver="roofline")
    assert req.user_id == "test"


def test_batched_request_user_specified_requires_gpu():
    with pytest.raises(ValidationError, match="gpu_type is required"):
        BatchedRequest(**BATCH_DEFAULTS, placement_solver="user_specified")


def test_batched_request_user_specified_valid():
    req = BatchedRequest(
        **BATCH_DEFAULTS,
        placement_solver="user_specified",
        gpu_type="A100",
        tp_size=4,
        pp_size=2,
    )
    assert req.tp_size == 4


def test_batched_request_user_specified_invalid_tp():
    with pytest.raises(ValidationError, match="tp_size must be one of"):
        BatchedRequest(
            **BATCH_DEFAULTS,
            placement_solver="user_specified",
            gpu_type="A100",
            tp_size=3,
        )


def test_batched_request_user_specified_invalid_pp():
    with pytest.raises(ValidationError, match="pp_size must be one of"):
        BatchedRequest(
            **BATCH_DEFAULTS,
            placement_solver="user_specified",
            gpu_type="A100",
            pp_size=5,
        )


def test_batched_request_roofline_no_gpu_ok():
    req = BatchedRequest(**BATCH_DEFAULTS, placement_solver="roofline")
    assert req.gpu_type is None


# --- MagicOutput ---


def test_magic_output_num_nodes_with_instances():
    mo = MagicOutput(
        decision_id="test",
        engine="vllm",
        instance_type="g6e.12xlarge",
        tp_size=4,
        pp_size=2,
        replicas=2,
        num_instances=3,
    )
    assert mo.num_nodes == 3  # solver's num_instances IS total nodes


def test_magic_output_num_nodes_fallback():
    mo = MagicOutput(
        decision_id="test",
        engine="vllm",
        instance_type="g6e.12xlarge",
        tp_size=4,
        pp_size=2,
        replicas=3,
    )
    assert mo.num_nodes == 6  # 2 * 3


def test_magic_output_num_nodes_single():
    mo = MagicOutput(
        decision_id="test",
        engine="vllm",
        instance_type="g6e.xlarge",
        tp_size=1,
        pp_size=1,
        replicas=1,
    )
    assert mo.num_nodes == 1


# --- OnlineServingRequest ---


def test_online_serving_request_valid():
    req = OnlineServingRequest(
        user_id="test",
        description="serve",
        task_type="chat",
        model_name="Qwen/Qwen3-32B",
        engine="vllm",
        placement="aws",
    )
    assert req.model_name == "Qwen/Qwen3-32B"


# --- vLLMSpecificConfig ---


def test_vllm_config_defaults():
    cfg = vLLMSpecificConfig()
    assert cfg.trust_remote_code is True
    assert cfg.config_format == "auto"
    assert cfg.tokenizer_mode == "auto"
