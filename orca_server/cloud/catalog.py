from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .models import InstanceSpec

V1_UNSUPPORTED_GPUS = {"V100", "T4"}


class InstanceCatalog(ABC):
    """Catalog of launchable instances for one or more providers."""

    @abstractmethod
    def list_instances(self, cloud: str | None = None) -> list[InstanceSpec]:
        pass

    @abstractmethod
    def get_instance(self, cloud: str, instance_type: str) -> InstanceSpec:
        pass


class AWSInstanceCatalog(InstanceCatalog):
    """AWS implementation backed by Orca's existing static instance table."""

    cloud = "aws"

    def __init__(self, instances: dict[str, tuple[str, int, int, int]]):
        self._instances = dict(instances)

    def list_instances(self, cloud: str | None = None) -> list[InstanceSpec]:
        if cloud is not None and cloud.lower() != self.cloud:
            return []
        return list(self._iter_specs())

    def get_instance(self, cloud: str, instance_type: str) -> InstanceSpec:
        if cloud.lower() != self.cloud:
            raise KeyError(f"AWSInstanceCatalog does not handle cloud={cloud!r}")
        try:
            gpu_name, gpu_count, vcpus, vram_gb = self._instances[instance_type]
        except KeyError as exc:
            raise KeyError(f"Unknown AWS instance type: {instance_type}") from exc
        return InstanceSpec(
            cloud=self.cloud,
            instance_type=instance_type,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            vcpus=vcpus,
            vram_gb=vram_gb,
            supports_vllm_v1=gpu_name not in V1_UNSUPPORTED_GPUS,
        )

    def _iter_specs(self) -> Iterable[InstanceSpec]:
        for instance_type in self._instances:
            yield self.get_instance(self.cloud, instance_type)
