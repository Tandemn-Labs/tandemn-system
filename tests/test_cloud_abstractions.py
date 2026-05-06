from orca_server.cloud import (
    AWSInstanceCatalog,
    InstanceSpec,
    PlacementCandidate,
    QuotaRequest,
    SkyPilotLaunchBackend,
)
from orca_server.config import AWS_INSTANCES
from quota.region_selector import AWSQuotaProvider, RegionCandidate
from storage.uri import is_storage_uri, parse_storage_uri


def test_aws_instance_catalog_wraps_existing_table():
    catalog = AWSInstanceCatalog(AWS_INSTANCES)

    spec = catalog.get_instance("aws", "g6e.12xlarge")

    assert spec == InstanceSpec(
        cloud="aws",
        instance_type="g6e.12xlarge",
        gpu_name="L40S",
        gpu_count=4,
        vcpus=48,
        vram_gb=48,
        supports_vllm_v1=True,
    )
    assert len(catalog.list_instances("aws")) == len(AWS_INSTANCES)
    assert catalog.list_instances("gcp") == []


def test_placement_candidate_is_provider_neutral():
    candidate = PlacementCandidate(
        cloud="aws",
        region="us-east-1",
        instance_type="p5.48xlarge",
        gpu_type="H100",
        gpus_per_node=8,
        num_nodes=2,
        tp_size=8,
        pp_size=2,
        market="spot",
        estimated_cost_per_hour=98.0,
    )

    assert candidate.cloud == "aws"
    assert candidate.market == "spot"
    assert candidate.num_nodes == 2


def test_storage_uri_parsing_is_scheme_agnostic():
    uri = parse_storage_uri("minio://bucket/path/to/file.jsonl")

    assert uri.scheme == "minio"
    assert uri.bucket == "bucket"
    assert uri.key == "path/to/file.jsonl"
    assert uri.uri == "minio://bucket/path/to/file.jsonl"
    assert is_storage_uri("gs://bucket/object") is True
    assert is_storage_uri("local/file.jsonl") is False


def test_aws_quota_provider_delegates_to_existing_selector(monkeypatch):
    def fake_get_ordered_regions(**kwargs):
        assert kwargs == {
            "instance_type": "g6e.12xlarge",
            "num_nodes": 2,
            "prefer_spot": False,
            "target_market": "on_demand",
        }
        return [
            RegionCandidate(
                region="us-east-1",
                use_spot=False,
                available_quota=192,
            )
        ]

    monkeypatch.setattr("quota.region_selector.get_ordered_regions", fake_get_ordered_regions)

    provider = AWSQuotaProvider()
    candidates = provider.get_region_candidates(
        QuotaRequest(
            cloud="aws",
            instance_type="g6e.12xlarge",
            num_nodes=2,
            prefer_spot=False,
            target_market="on_demand",
        )
    )

    assert candidates[0].region == "us-east-1"
    assert candidates[0].market == "on_demand"
    assert candidates[0].use_spot is False
    assert provider.get_region_candidates(QuotaRequest(cloud="gcp", instance_type="a3-highgpu-8g")) == []


def test_skypilot_launch_backend_builds_resources_from_candidate():
    backend = SkyPilotLaunchBackend(ports=8001)
    resources = backend.build_resources(
        PlacementCandidate(
            cloud="aws",
            region="us-west-2",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            gpus_per_node=4,
            num_nodes=1,
            tp_size=4,
            pp_size=1,
            market="on_demand",
        )
    )

    assert resources == {
        "instance_type": "g6e.12xlarge",
        "disk_size": "300GB",
        "region": "us-west-2",
        "use_spot": False,
        "ports": 8001,
    }
