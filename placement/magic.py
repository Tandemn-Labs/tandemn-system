from abc import ABC, abstractmethod
from typing import Union
from models.requests import BatchedRequest, OnlineServingRequest
from models.resources import MagicOutput


class VPCMagic(ABC):
    """Abstract base class for VPC allocation decision strategies."""

    @abstractmethod
    def decide(
        self, request: Union[BatchedRequest, OnlineServingRequest]
    ) -> MagicOutput:
        """
        Make an allocation decision based on the request.

        Args:
            request: Either a BatchedRequest or OnlineServingRequest

        Returns:
            MagicOutput containing the allocation decision
        """
        pass
