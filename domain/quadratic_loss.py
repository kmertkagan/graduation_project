"""Convex quadratic loss used as the well-behaved benchmark landscape."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from domain.loss_function import FloatArray, LossFunction

_DEFAULT_MATRIX = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
_DEFAULT_CENTER = np.array([1.0, 1.0], dtype=np.float64)


class QuadraticLoss(LossFunction):
    r"""Convex quadratic loss with an adjustable curvature matrix.

    The function is defined as

    .. math::

        f(\mathbf{x}) = \frac{1}{2}
        (\mathbf{x} - \mathbf{c})^{\top} A (\mathbf{x} - \mathbf{c}) + k,

    where :math:`A` is a symmetric positive definite matrix,
    :math:`\mathbf{c}` is the center (global minimizer) and :math:`k` is a
    non-negative constant offset.

    Derivatives
    -----------
    .. math::

        \nabla f(\mathbf{x}) = A(\mathbf{x} - \mathbf{c}), \qquad
        \nabla^2 f(\mathbf{x}) = A.

    Because the Hessian is constant and positive definite, the loss is strictly
    convex and its unique minimum is exactly :math:`\mathbf{c}`. The default
    configuration uses :math:`A = [[3, 1], [1, 2]]` and
    :math:`\mathbf{c} = (1, 1)`, which shares its minimizer with the
    :class:`~domain.rosenbrock_loss.RosenbrockLoss` benchmark.

    Parameters
    ----------
    matrix:
        Symmetric positive definite curvature matrix ``A``. Defaults to a
        2x2 well-conditioned matrix.
    center:
        Location of the global minimum. Defaults to ``(1, 1)``.
    constant:
        Non-negative additive constant ``k`` (the value at the minimum).
    """

    def __init__(
        self,
        matrix: ArrayLike | None = None,
        center: ArrayLike | None = None,
        constant: float = 0.0,
    ) -> None:
        matrix_array = np.array(
            _DEFAULT_MATRIX if matrix is None else matrix,
            dtype=np.float64,
            copy=True,
        )
        if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
            raise ValueError(
                "matrix must be a square two-dimensional array, "
                f"received shape {matrix_array.shape}."
            )
        if not np.all(np.isfinite(matrix_array)):
            raise ValueError("matrix must contain only finite values.")
        if not np.allclose(matrix_array, matrix_array.T, rtol=0.0, atol=1e-12):
            raise ValueError("matrix must be symmetric.")

        eigenvalues = np.linalg.eigvalsh(matrix_array)
        if np.any(eigenvalues <= 0.0):
            raise ValueError(
                "matrix must be positive definite for the loss to be convex; "
                f"eigenvalues: {eigenvalues!r}."
            )

        center_array = np.array(
            _DEFAULT_CENTER if center is None else center,
            dtype=np.float64,
            copy=True,
        )
        if center_array.shape != (matrix_array.shape[0],):
            raise ValueError(
                f"center must have shape ({matrix_array.shape[0]},), "
                f"received {center_array.shape}."
            )
        if not np.all(np.isfinite(center_array)):
            raise ValueError("center must contain only finite values.")

        constant_value = float(constant)
        if not np.isfinite(constant_value):
            raise ValueError("constant must be a finite number.")
        if constant_value < 0.0:
            raise ValueError("constant must be non-negative.")

        # Immutable configuration: callers cannot silently alter the landscape.
        matrix_array.setflags(write=False)
        center_array.setflags(write=False)

        self._matrix = matrix_array
        self._center = center_array
        self._constant = constant_value

    @property
    def name(self) -> str:
        return "Quadratic Loss"

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[0])

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def known_minimizer(self) -> FloatArray:
        return self._center.copy()

    @property
    def known_minimum_value(self) -> float:
        return self._constant

    @property
    def matrix(self) -> FloatArray:
        """Read-only view of the curvature matrix ``A``."""
        return self._matrix

    def evaluate(self, parameters: ArrayLike) -> float:
        vector = self.validate_parameters(parameters)
        delta = vector - self._center
        return float(0.5 * delta @ self._matrix @ delta + self._constant)

    def gradient(self, parameters: ArrayLike) -> FloatArray:
        vector = self.validate_parameters(parameters)
        return self._matrix @ (vector - self._center)

    def hessian(self, parameters: ArrayLike) -> FloatArray:
        self.validate_parameters(parameters)
        return self._matrix.copy()
