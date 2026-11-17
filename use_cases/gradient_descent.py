"""First-order Gradient Descent optimizer."""

from __future__ import annotations

import numpy as np

from domain import FloatArray, LossFunction
from use_cases.base_optimizer import BaseOptimizer


class GradientDescentOptimizer(BaseOptimizer):
    r"""Standard first-order gradient descent.

    .. math::

        \mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)

    The update is cheap but ignores curvature. On ill-conditioned landscapes
    such as Rosenbrock it zig-zags across the valley and needs many iterations.

    Parameters
    ----------
    loss_function:
        Differentiable objective to minimize.
    learning_rate:
        Step size :math:`\eta > 0`.
    max_iterations:
        Maximum number of parameter updates.
    tolerance:
        Convergence threshold on the Euclidean gradient norm.
    """

    def __init__(
        self,
        loss_function: LossFunction,
        learning_rate: float = 1e-3,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        super().__init__(loss_function, max_iterations, tolerance)

        learning_rate_value = float(learning_rate)
        if not np.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
            raise ValueError(
                "learning_rate must be a finite, positive number."
            )
        self._learning_rate = learning_rate_value

    @property
    def name(self) -> str:
        return "Gradient Descent"

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def _compute_update(
        self,
        parameters: FloatArray,
        gradient: FloatArray,
        iteration: int,
    ) -> FloatArray:
        return parameters - self.learning_rate * gradient
