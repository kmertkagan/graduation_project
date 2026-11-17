"""Domain layer: framework-independent mathematical entities.

Exports the loss function port and its concrete implementations so that outer
layers can depend on a small, explicit public surface.
"""

from domain.loss_function import FloatArray, LossFunction
from domain.quadratic_loss import QuadraticLoss
from domain.rosenbrock_loss import RosenbrockLoss

__all__ = [
    "FloatArray",
    "LossFunction",
    "QuadraticLoss",
    "RosenbrockLoss",
]
