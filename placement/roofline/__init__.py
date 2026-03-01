"""
Roofline-based GPU placement solver for LLM inference.

This package provides GPU specs, model architecture data, and throughput
calculations used by the PlacementSolverAdapter to interface with
LLM_placement_solver.
"""

from .gpu_specs import GPU_SPECS, GPU_PERFORMANCE_TIERS
from .throughput import ThroughputCalculator
from .model_arch import ModelArchitecture, get_model_architecture, KNOWN_ARCHITECTURES

__all__ = [
    "GPU_SPECS",
    "GPU_PERFORMANCE_TIERS",
    "ThroughputCalculator",
    "ModelArchitecture",
    "get_model_architecture",
    "KNOWN_ARCHITECTURES",
]
