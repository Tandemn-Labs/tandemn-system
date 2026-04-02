from orca_server.config import (
    AWS_INSTANCES,
    AWS_INSTANCE_TO_GPU,
    INSTANCE_TO_GPU,
    INSTANCE_VCPUS,
    INSTANCE_VRAM,
    VLLM_PORT,
)


def test_derived_dicts_cover_all_instances():
    """All derived dicts have same keys as AWS_INSTANCES."""
    keys = set(AWS_INSTANCES.keys())
    assert set(INSTANCE_TO_GPU.keys()) == keys
    assert set(INSTANCE_VCPUS.keys()) == keys
    assert set(AWS_INSTANCE_TO_GPU.keys()) == keys
    assert set(INSTANCE_VRAM.keys()) == keys


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


def test_instance_vram_values():
    """INSTANCE_VRAM maps to vram_per_gpu, all > 0."""
    for inst, vram in INSTANCE_VRAM.items():
        assert vram == AWS_INSTANCES[inst][3]
        assert vram > 0


def test_a100_vram_differentiated():
    """p4d (40GB) and p4de (80GB) must have different VRAM."""
    assert INSTANCE_VRAM["p4d.24xlarge"] == 40
    assert INSTANCE_VRAM["p4de.24xlarge"] == 80


def test_vllm_port_is_int():
    assert isinstance(VLLM_PORT, int)
    assert VLLM_PORT > 0
