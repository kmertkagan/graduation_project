"""Use cases: optimization algorithms and their shared result contract."""

from use_cases.adam import AdamOptimizer
from use_cases.base_optimizer import BaseOptimizer, OptimizationError
from use_cases.gradient_descent import GradientDescentOptimizer
from use_cases.newton_raphson import NewtonRaphsonOptimizer
from use_cases.optimization_result import (
    OptimizationResult,
    OptimizationStatus,
)

__all__ = [
    "AdamOptimizer",
    "BaseOptimizer",
    "GradientDescentOptimizer",
    "NewtonRaphsonOptimizer",
    "OptimizationError",
    "OptimizationResult",
    "OptimizationStatus",
]
