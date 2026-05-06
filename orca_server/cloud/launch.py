from __future__ import annotations

from abc import ABC, abstractmethod

from .models import LaunchRequest, LaunchResult, PlacementCandidate


class LaunchBackend(ABC):
    """Launch abstraction that hides SkyPilot/provider-specific resource shape."""

    @abstractmethod
    def build_resources(self, candidate: PlacementCandidate) -> dict:
        pass

    @abstractmethod
    def launch(self, request: LaunchRequest) -> LaunchResult:
        pass


class SkyPilotLaunchBackend(LaunchBackend):
    """SkyPilot adapter for provider-neutral placement candidates."""

    def __init__(self, disk_size: str = "300GB", ports: int | None = None):
        self.disk_size = disk_size
        self.ports = ports

    def build_resources(self, candidate: PlacementCandidate) -> dict:
        resources = {
            "instance_type": candidate.instance_type,
            "disk_size": self.disk_size,
        }
        if candidate.region:
            resources["region"] = candidate.region
        if candidate.zone:
            resources["zone"] = candidate.zone
        if candidate.market is not None:
            resources["use_spot"] = candidate.market == "spot"
        if self.ports is not None:
            resources["ports"] = self.ports
        return resources

    def launch(self, request: LaunchRequest) -> LaunchResult:
        raise NotImplementedError("SkyPilot launch wiring remains in orca_server.launcher")
