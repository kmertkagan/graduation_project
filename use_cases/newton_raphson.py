"""Second-order Newton-Raphson optimizer."""

from __future__ import annotations

import numpy as np

from domain import FloatArray, LossFunction
from use_cases.base_optimizer import BaseOptimizer, OptimizationError


class NewtonRaphsonOptimizer(BaseOptimizer):
    r"""Newton-Raphson method using the inverse Hessian.

    .. math::

        \mathbf{x}_{k+1} = \mathbf{x}_k
        - \eta \left[\nabla^2 f(\mathbf{x}_k)\right]^{-1}
        \nabla f(\mathbf{x}_k)

    For a quadratic loss with :math:`\eta = 1` the method reaches the exact
    minimum in a single step. On general non-convex losses the Hessian may be
    singular or indefinite, so the method can overshoot or diverge; that
    fragility is part of the comparison rather than something this class
    hides via a line search.

    The inverse is computed explicitly with :func:`numpy.linalg.inv`, which is
    :math:`O(n^3)` and becomes prohibitively expensive for large parameter
    counts.

    Parameters
    ----------
    loss_function:
        Twice-differentiable objective to minimize.
    learning_rate:
        Newton step multiplier :math:`\eta > 0` (``1.0`` is the pure method).
    max_iterations:
        Maximum number of parameter updates.
    tolerance:
        Convergence threshold on the Euclidean gradient norm.
    """

    def __init__(
        self,
        loss_function: LossFunction,
        learning_rate: float = 1.0,
        max_iterations: int = 100,
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
        return "Newton-Raphson"

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def _compute_update(
        self,
        parameters: FloatArray,
        gradient: FloatArray,
        iteration: int,
    ) -> FloatArray:
        hessian = self.loss_function.hessian(parameters)
        if not np.all(np.isfinite(hessian)):
            raise OptimizationError(
                "Hessian matrix contains non-finite values "
                f"at parameters {parameters!r}."
            )

        try:
            hessian_inverse = np.linalg.inv(hessian)
        except np.linalg.LinAlgError as error:
            raise OptimizationError(
                f"Hessian matrix is singular at parameters {parameters!r}."
            ) from error

        return parameters - self.learning_rate * (hessian_inverse @ gradient)
