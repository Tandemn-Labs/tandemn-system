import pytest
from placement.roofline_magic import (
    resolve_gpu_type_to_instance,
    quota_to_gpu_pool,
    AWS_INSTANCE_TO_GPU,
    GPU_TYPE_TO_INSTANCES,
    _build_gpu_type_to_instances,
)


# --- resolve_gpu_type_to_instance ---


def test_resolve_l40s_1gpu():
    inst, count = resolve_gpu_type_to_instance("L40S", 1)
    assert inst == "g6e.xlarge"
    assert count == 1


def test_resolve_l40s_4gpu():
    inst, count = resolve_gpu_type_to_instance("L40S", 4)
    assert inst == "g6e.12xlarge"
    assert count == 4


def test_resolve_l40s_8gpu():
    inst, count = resolve_gpu_type_to_instance("L40S", 8)
    assert inst == "g6e.48xlarge"
    assert count == 8


def test_resolve_a100():
    inst, count = resolve_gpu_type_to_instance("A100", 8)
    assert inst == "p4d.24xlarge"
    assert count == 8


def test_resolve_h100():
    inst, count = resolve_gpu_type_to_instance("H100", 8)
    assert inst == "p5.48xlarge"
    assert count == 8


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="No AWS instance found"):
        resolve_gpu_type_to_instance("Z100", 1)


def test_resolve_too_many_gpus_raises():
    with pytest.raises(ValueError, match="No AWS instance with >="):
        resolve_gpu_type_to_instance("L40S", 16)


# --- GPU_TYPE_TO_INSTANCES ---


def test_reverse_map_sorted_ascending():
    for gpu_type, instances in GPU_TYPE_TO_INSTANCES.items():
        counts = [count for _, count in instances]
        assert counts == sorted(counts), f"{gpu_type} instances not sorted ascending"


# --- quota_to_gpu_pool ---


def test_quota_to_gpu_pool_basic(sample_quota_df):
    pool = quota_to_gpu_pool(sample_quota_df, region="us-east-1", market="on_demand")
    # g6e.12xlarge and g6e.48xlarge should be in the pool (allowed prefixes)
    assert "g6e.12xlarge" in pool
    assert "g6e.48xlarge" in pool


def test_quota_to_gpu_pool_filters_old_gpus(sample_quota_df):
    pool = quota_to_gpu_pool(sample_quota_df, region="us-east-1", market="on_demand")
    # g3 instances should be filtered out
    assert "g3.16xlarge" not in pool


def test_quota_to_gpu_pool_missing_column(sample_quota_df):
    pool = quota_to_gpu_pool(
        sample_quota_df, region="ap-south-1", market="on_demand"
    )
    assert pool == {}


def test_quota_to_gpu_pool_caps_instances(sample_quota_df):
    pool = quota_to_gpu_pool(
        sample_quota_df,
        region="us-east-1",
        market="on_demand",
        max_instances_per_type=1,
    )
    for count in pool.values():
        assert count <= 1


# --- AWS_INSTANCE_TO_GPU table ---


def test_instance_to_gpu_table_entries_valid():
    known_gpus = {"H100", "A100", "V100", "L40S", "L4", "A10G"}
    for inst, (gpu_name, gpu_count) in AWS_INSTANCE_TO_GPU.items():
        assert gpu_name in known_gpus, f"{inst} has unknown GPU: {gpu_name}"
        assert gpu_count >= 1, f"{inst} has invalid count: {gpu_count}"
