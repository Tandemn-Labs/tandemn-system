from config import (
    AWS_INSTANCES,
    AWS_INSTANCE_TO_GPU,
    INSTANCE_TO_GPU,
    INSTANCE_VCPUS,
    VLLM_PORT,
)


def test_derived_dicts_cover_all_instances():
    """All 3 derived dicts have same keys as AWS_INSTANCES."""
    keys = set(AWS_INSTANCES.keys())
    assert set(INSTANCE_TO_GPU.keys()) == keys
    assert set(INSTANCE_VCPUS.keys()) == keys
    assert set(AWS_INSTANCE_TO_GPU.keys()) == keys


def test_instance_to_gpu_values():
    """INSTANCE_TO_GPU maps to the gpu_name from AWS_INSTANCES."""
    for inst, gpu_name in INSTANCE_TO_GPU.items():
        assert gpu_name == AWS_INSTANCES[inst][0]


def test_instance_vcpus_values():
    """INSTANCE_VCPUS maps to vcpus, all > 0."""
    for inst, vcpus in INSTANCE_VCPUS.items():
        assert vcpus == AWS_INSTANCES[inst][2]
        assert vcpus > 0


def test_aws_instance_to_gpu_shape():
    """Tuples of (gpu_name, gpu_count), all count >= 1."""
    for inst, (gpu_name, gpu_count) in AWS_INSTANCE_TO_GPU.items():
        assert isinstance(gpu_name, str)
        assert gpu_count >= 1
        assert (gpu_name, gpu_count) == AWS_INSTANCES[inst][:2]


def test_vllm_port_is_int():
    assert isinstance(VLLM_PORT, int)
    assert VLLM_PORT > 0
