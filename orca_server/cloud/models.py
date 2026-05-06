from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CloudMarket = Literal["spot", "on_demand"]


class InstanceSpec(BaseModel):
    """Provider-neutral description of a launchable accelerator instance."""

    model_config = ConfigDict(frozen=True)

    cloud: str
    instance_type: str
    gpu_name: str
    gpu_count: int = Field(ge=0)
    vcpus: int = Field(ge=1)
    vram_gb: int = Field(ge=0)
    supports_vllm_v1: bool


class PlacementCandidate(BaseModel):
    """A concrete placement option independent of a specific cloud provider API."""

    model_config = ConfigDict(frozen=True)

    cloud: str
    region: str | None = None
    zone: str | None = None
    instance_type: str
    gpu_type: str
    gpus_per_node: int = Field(ge=0)
    num_nodes: int = Field(ge=1)
    tp_size: int = Field(ge=1)
    pp_size: int = Field(ge=1)
    market: CloudMarket | None = None
    estimated_cost_per_hour: float | None = Field(default=None, ge=0)


class QuotaRequest(BaseModel):
    """Quota/capacity lookup input for a provider implementation."""

    model_config = ConfigDict(frozen=True)

    cloud: str
    instance_type: str
    num_nodes: int = Field(default=1, ge=1)
    prefer_spot: bool = True
    target_market: CloudMarket | None = None


class RegionCandidate(BaseModel):
    """A region/market option with enough reported capacity for a request."""

    model_config = ConfigDict(frozen=True)
    region: str
    market: CloudMarket
    available_quota: int = Field(ge=0)

    @property
    def use_spot(self) -> bool:
        return self.market == "spot"

    def to_skypilot_resources(
        self,
        instance_type: str,
        disk_size: str = "300GB",
        ports: int | None = None,
    ) -> dict:
        resources = {
            "region": self.region,
            "instance_type": instance_type,
            "use_spot": self.use_spot,
            "disk_size": disk_size,
        }
        if ports is not None:
            resources["ports"] = ports
        return resources


class PricingRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    cloud: str
    instance_type: str
    region: str | None = None
    market: CloudMarket | None = None


class LaunchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    job_id: str
    candidate: PlacementCandidate
    num_nodes: int = Field(ge=1)


class LaunchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    job_id: str
    cluster_name: str | None = None
    endpoint: str | None = None
    status: str
