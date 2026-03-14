import pytest
import pandas as pd


@pytest.fixture
def sample_quota_csv_path(tmp_path):
    """Minimal quota CSV matching the real schema."""
    csv = tmp_path / "quota.csv"
    csv.write_text(
        "Family,Instance_Type,vCPU,GPU_Type,VRAM_per_GPU,Total_VRAM,Family_Type,"
        "us-east-1_on_demand,us-east-1_spot,us-west-2_on_demand,us-west-2_spot\n"
        "G6e,g6e.12xlarge,48,4x L40S,48.0,192.0,G,192,192,96,96\n"
        "G6e,g6e.48xlarge,192,8x L40S,48.0,384.0,G,192,384,192,192\n"
        "P4d,p4d.24xlarge,96,8x A100,80.0,640.0,P4_P3_P2,192,192,96,96\n"
        "G3,g3.16xlarge,64,4x M60,8.0,32.0,G,128,128,64,64\n"
    )
    return str(csv)


@pytest.fixture
def sample_quota_df(sample_quota_csv_path):
    """Load the sample CSV via the real loader."""
    from utils.utils import load_aws_quota_csv

    return load_aws_quota_csv(sample_quota_csv_path)


@pytest.fixture
def sample_perfdb_dir(tmp_path):
    """Minimal perf_db directory with one CSV."""
    gpu_dir = tmp_path / "L40S"
    gpu_dir.mkdir()
    csv = gpu_dir / "perfdb_l40s_llama70b.csv"
    csv.write_text(
        "Model Name,Max Input Length,Max Output Length,Total Tokens Per Second,"
        "GPU Name,TP,PP,Mem Per GPU GB\n"
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B,4096,1024,1136.87,"
        "NVIDIA L40S,4,3,48\n"
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B,4096,4096,586.77,"
        "NVIDIA L40S,4,3,48\n"
    )
    return str(tmp_path)


@pytest.fixture
def region_quotas():
    """Pre-built RegionQuota dict for testing without AWS calls."""
    from quota.region_selector import RegionQuota

    return {
        "us-east-1": RegionQuota(region="us-east-1", on_demand_vcpus=192, spot_vcpus=384),
        "us-west-2": RegionQuota(region="us-west-2", on_demand_vcpus=96, spot_vcpus=96),
        "eu-west-1": RegionQuota(region="eu-west-1", on_demand_vcpus=0, spot_vcpus=0),
    }
