from __future__ import annotations

from abc import ABC, abstractmethod

from .models import QuotaRequest, RegionCandidate


class QuotaProvider(ABC):
    """Provider-specific quota and capacity lookup interface."""

    @abstractmethod
    def get_region_candidates(self, request: QuotaRequest) -> list[RegionCandidate]:
        pass
