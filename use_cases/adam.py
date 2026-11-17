"""Adam optimizer with momentum and RMSProp-style scaling."""

from __future__ import annotations

import numpy as np

from domain import FloatArray, LossFunction
from use_cases.base_optimizer import BaseOptimizer


class AdamOptimizer(BaseOptimizer):
    r"""Adaptive Moment Estimation (Adam).

    Adam keeps an exponentially decaying average of past gradients
    (:math:`m_k`, first moment) and of past squared gradients
    (:math:`v_k`, second moment):

    .. math::

        m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k \\
        v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2 \\
        \hat{m}_k = \frac{m_k}{1-\beta_1^k}, \qquad
        \hat{v}_k = \frac{v_k}{1-\beta_2^k} \\
        x_{k+1} = x_k - \eta \frac{\hat{m}_k}
        {\sqrt{\hat{v}_k} + \epsilon}

    The bias-correction terms make the estimates unbiased during the first
    iterations. Per-parameter scaling removes the need for the Hessian and
    usually makes Adam far less sensitive to curvature than plain gradient
    descent.

    Parameters
    ----------
    loss_function:
        Differentiable objective to minimize.
    learning_rate:
        Base step size :math:`\eta > 0`.
    beta1:
        Decay rate for the first moment, in ``[0, 1)``.
    beta2:
        Decay rate for the second moment, in ``[0, 1)``.
    epsilon:
        Small stabilizer added to the denominator.
    max_iterations:
        Maximum number of parameter updates.
    tolerance:
        Convergence threshold on the Euclidean gradient norm.
    """

    def __init__(
        self,
        loss_function: LossFunction,
        learning_rate: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        super().__init__(loss_function, max_iterations, tolerance)

        learning_rate_value = float(learning_rate)
        if not np.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
            raise ValueError(
                "learning_rate must be a finite, positive number."
            )

        beta1_value = float(beta1)
        beta2_value = float(beta2)
        if not 0.0 <= beta1_value < 1.0:
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1.")
        if not 0.0 <= beta2_value < 1.0:
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1.")

        epsilon_value = float(epsilon)
        if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be a finite, positive number.")

        self._learning_rate = learning_rate_value
        self._beta1 = beta1_value
        self._beta2 = beta2_value
        self._epsilon = epsilon_value
        self._first_moment: FloatArray | None = None
        self._second_moment: FloatArray | None = None

    @property
    def name(self) -> str:
        return "Adam"

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def beta1(self) -> float:
        return self._beta1

    @property
    def beta2(self) -> float:
        return self._beta2

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def _on_optimization_start(self, initial_parameters: FloatArray) -> None:
        self._first_moment = np.zeros_like(initial_parameters)
        self._second_moment = np.zeros_like(initial_parameters)

    def _compute_update(
        self,
        parameters: FloatArray,
        gradient: FloatArray,
        iteration: int,
    ) -> FloatArray:
        if self._first_moment is None or self._second_moment is None:
            raise RuntimeError(
                "Adam state was not initialized; call optimize() instead of "
                "invoking the update rule directly."
            )

        self._first_moment = (
            self.beta1 * self._first_moment + (1.0 - self.beta1) * gradient
        )
        self._second_moment = (
            self.beta2 * self._second_moment
            + (1.0 - self.beta2) * np.square(gradient)
        )

        first_moment_hat = self._first_moment / (
            1.0 - self.beta1**iteration
        )
        second_moment_hat = self._second_moment / (
            1.0 - self.beta2**iteration
        )

        return parameters - self.learning_rate * first_moment_hat / (
            np.sqrt(second_moment_hat) + self.epsilon
        )
