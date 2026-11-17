"""Domain contracts for twice-differentiable loss functions.

This module belongs to the domain layer and deliberately knows nothing about
optimizers, benchmarking, or presentation concerns. A loss function is the
mathematical entity that every optimization algorithm consumes, so it is
modelled here as an abstract port (dependency inversion principle).

Only NumPy is used: no autodiff, deep-learning framework, or optimizer library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class LossFunction(ABC):
    r"""Port for a scalar, twice-differentiable loss function.

    Implementations expose the three mathematical primitives required by the
    optimization use cases:

    * :meth:`evaluate`  -> :math:`f(\mathbf{x})`
    * :meth:`gradient`  -> :math:`\nabla f(\mathbf{x})`
    * :meth:`hessian`   -> :math:`\nabla^2 f(\mathbf{x})`

    The parameter vector always has shape ``(dimension,)`` and dtype
    ``float64``. Inputs are converted and validated before use so that
    numerical errors surface at the domain boundary instead of inside an
    optimizer iteration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """English identifier of the function used in logs and tests."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of parameters accepted by the function (``n``)."""

    @property
    def is_convex(self) -> bool:
        """Whether the function is globally convex.

        The base implementation is conservative and returns ``False``;
        convex implementations override it with ``True``.
        """
        return False

    @property
    def known_minimizer(self) -> FloatArray | None:
        """Analytical global minimizer, if known; otherwise ``None``."""
        return None

    @property
    def known_minimum_value(self) -> float | None:
        """Loss value at :attr:`known_minimizer`, if known; otherwise ``None``."""
        return None

    @abstractmethod
    def evaluate(self, parameters: ArrayLike) -> float:
        r"""Return the loss value :math:`f(\mathbf{x})` for ``parameters``."""

    @abstractmethod
    def gradient(self, parameters: ArrayLike) -> FloatArray:
        r"""Return the gradient vector :math:`\nabla f(\mathbf{x})`."""

    @abstractmethod
    def hessian(self, parameters: ArrayLike) -> FloatArray:
        r"""Return the Hessian matrix :math:`\nabla^2 f(\mathbf{x})`."""

    def _validate_parameters(self, parameters: ArrayLike) -> FloatArray:
        """Convert ``parameters`` to a finite ``float64`` vector.

        Raises
        ------
        ValueError
            If the input shape does not match :attr:`dimension` or contains
            NaN/infinite values.
        """
        vector = np.asarray(parameters, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"{self.name} expects a parameter vector with shape "
                f"({self.dimension},), received {vector.shape}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{self.name} received non-finite parameters: {vector!r}."
            )
        return vector
