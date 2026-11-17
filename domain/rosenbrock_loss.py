"""Non-convex Rosenbrock function used as the hard benchmark landscape."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from domain.loss_function import FloatArray, LossFunction


class RosenbrockLoss(LossFunction):
    r"""The non-convex Rosenbrock ("banana") function.

    .. math::

        f(x, y) = (1 - x)^2 + 100 (y - x^2)^2

    The function is unimodal in the strict sense (it has a single stationary
    point, the global minimum at :math:`(1, 1)`), but its narrow, curved valley
    makes first-order methods crawl: the gradient is tiny along the valley and
    large across it. This geometry is the standard stress test for comparing
    gradient descent, Adam and Newton-Raphson.

    Derivatives
    -----------
    .. math::

        \nabla f(x, y) =
        \begin{bmatrix}
            -2(1 - x) - 400x(y - x^2) \\
            200(y - x^2)
        \end{bmatrix}

    .. math::

        \nabla^2 f(x, y) =
        \begin{bmatrix}
            2 - 400y + 1200x^2 & -400x \\
            -400x              & 200
        \end{bmatrix}

    The Hessian is not positive definite everywhere, so the landscape is
    non-convex even though the only stationary point is the global minimum.
    """

    @property
    def name(self) -> str:
        return "Rosenbrock Loss"

    @property
    def dimension(self) -> int:
        return 2

    @property
    def known_minimizer(self) -> FloatArray:
        return np.array([1.0, 1.0], dtype=np.float64)

    @property
    def known_minimum_value(self) -> float:
        return 0.0

    def evaluate(self, parameters: ArrayLike) -> float:
        x, y = self._validate_parameters(parameters)
        return float((1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2)

    def gradient(self, parameters: ArrayLike) -> FloatArray:
        x, y = self._validate_parameters(parameters)
        return np.array(
            [
                -2.0 * (1.0 - x) - 400.0 * x * (y - x**2),
                200.0 * (y - x**2),
            ],
            dtype=np.float64,
        )

    def hessian(self, parameters: ArrayLike) -> FloatArray:
        x, y = self._validate_parameters(parameters)
        return np.array(
            [
                [2.0 - 400.0 * y + 1200.0 * x**2, -400.0 * x],
                [-400.0 * x, 200.0],
            ],
            dtype=np.float64,
        )
