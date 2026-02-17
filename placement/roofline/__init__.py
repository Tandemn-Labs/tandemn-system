"""
Roofline-based GPU placement solver for LLM inference.

This module provides a deterministic, roofline-model-based approach for selecting
GPU configurations (instance type, TP, PP) for LLM inference workloads.

Ported from: ../LLM_placement_solver/solver.py
"""

from .gpu_specs import GPU_SPECS, GPU_PERFORMANCE_TIERS
from .throughput import ThroughputCalculator
from .model_arch import ModelArchitecture, get_model_architecture, KNOWN_ARCHITECTURES
from .solver import (
    RooflinePlacementSolver,
    PlacementConfig,
    PlacementResult,
    OptimizationPriority,
)

__all__ = [
    "GPU_SPECS",
    "GPU_PERFORMANCE_TIERS",
    "ThroughputCalculator",
    "ModelArchitecture",
    "get_model_architecture",
    "KNOWN_ARCHITECTURES",
    "RooflinePlacementSolver",
    "PlacementConfig",
    "PlacementResult",
    "OptimizationPriority",
]
