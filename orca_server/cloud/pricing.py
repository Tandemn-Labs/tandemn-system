from __future__ import annotations

from abc import ABC, abstractmethod

from .models import PricingRequest


class PricingProvider(ABC):
    """Provider-specific pricing lookup interface."""

    @abstractmethod
    def get_hourly_cost(self, request: PricingRequest) -> float | None:
        pass


class SkyPilotPricingProvider(PricingProvider):
    """Pricing adapter backed by SkyPilot's catalog."""

    def get_hourly_cost(self, request: PricingRequest) -> float | None:
        try:
            from sky import catalog

            return catalog.get_hourly_cost(
                request.instance_type,
                use_spot=request.market == "spot",
                region=request.region,
                zone=None,
                clouds=request.cloud,
            )
        except Exception:
            return None
