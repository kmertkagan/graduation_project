"""Abstract optimization use case with shared bookkeeping.

Concrete optimizers only implement their update rule; timing, validation,
history recording, convergence checks and failure handling live here so that
every algorithm reports metrics in exactly the same way.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from domain import FloatArray, LossFunction
from use_cases.optimization_result import OptimizationResult, OptimizationStatus


class OptimizationError(RuntimeError):
    """Raised when a concrete optimizer cannot compute a valid update."""


def _gradient_norm(gradient: FloatArray) -> float:
    """Return the Euclidean norm without emitting overflow warnings."""
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.linalg.norm(gradient))


class BaseOptimizer(ABC):
    """Template-method base class for iterative optimization algorithms.

    Subclasses implement :meth:`_compute_update` and may reset internal state
    in :meth:`_on_optimization_start`. The public :meth:`optimize` method is
    responsible for:

    * validating the starting point,
    * measuring wall-clock and CPU time,
    * recording parameters and loss values at every step,
    * detecting convergence, divergence and numerical failures,
    * returning a uniform :class:`OptimizationResult`.
    """

    def __init__(
        self,
        loss_function: LossFunction,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        if not isinstance(loss_function, LossFunction):
            raise TypeError(
                "loss_function must be an implementation of LossFunction."
            )
        if not isinstance(max_iterations, (int, np.integer)) or isinstance(
            max_iterations, bool
        ):
            raise TypeError("max_iterations must be an integer.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")

        tolerance_value = float(tolerance)
        if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
            raise ValueError("tolerance must be a finite, non-negative number.")

        self._loss_function = loss_function
        self._max_iterations = int(max_iterations)
        self._tolerance = tolerance_value

    @property
    @abstractmethod
    def name(self) -> str:
        """English name of the optimizer."""

    @property
    def loss_function(self) -> LossFunction:
        return self._loss_function

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def tolerance(self) -> float:
        return self._tolerance

    def optimize(self, initial_parameters: ArrayLike) -> OptimizationResult:
        """Run the optimizer from ``initial_parameters``.

        Returns
        -------
        OptimizationResult
            Uniform record of the run, including its full trajectory.
        """
        current = self.loss_function.validate_parameters(
            initial_parameters
        ).copy()
        self._on_optimization_start(current)

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        trajectory: list[FloatArray] = [current.copy()]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            initial_loss = float(self.loss_function.evaluate(current))
            gradient = self.loss_function.gradient(current)
        loss_history: list[float] = [initial_loss]

        gradient_norm = _gradient_norm(gradient)
        status = OptimizationStatus.MAX_ITERATIONS
        stop_reason = "maximum number of iterations reached"
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            if not np.isfinite(gradient_norm):
                status = OptimizationStatus.DIVERGED
                stop_reason = "gradient norm became non-finite"
                break
            if gradient_norm <= self.tolerance:
                status = OptimizationStatus.CONVERGED
                stop_reason = (
                    f"gradient norm {gradient_norm:.3e} <= "
                    f"tolerance {self.tolerance:.3e}"
                )
                break

            try:
                with np.errstate(
                    over="ignore", invalid="ignore", divide="ignore"
                ):
                    candidate = self._compute_update(
                        current, gradient, iteration
                    )
            except OptimizationError as error:
                status = OptimizationStatus.FAILED
                stop_reason = str(error)
                break

            candidate = np.asarray(candidate, dtype=np.float64)
            if candidate.shape != current.shape or not np.all(
                np.isfinite(candidate)
            ):
                status = OptimizationStatus.DIVERGED
                stop_reason = "update produced invalid parameters"
                break

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                candidate_loss = float(self.loss_function.evaluate(candidate))
                candidate_gradient = self.loss_function.gradient(candidate)

            if not np.isfinite(candidate_loss):
                status = OptimizationStatus.DIVERGED
                stop_reason = "loss became non-finite"
                break
            if not np.all(np.isfinite(candidate_gradient)):
                status = OptimizationStatus.DIVERGED
                stop_reason = "gradient became non-finite"
                break

            current = candidate
            gradient = candidate_gradient
            gradient_norm = _gradient_norm(gradient)
            trajectory.append(current.copy())
            loss_history.append(candidate_loss)
            iterations = iteration

        elapsed_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start

        trajectory_array = np.vstack(trajectory)
        loss_array = np.asarray(loss_history, dtype=np.float64)
        best_index = int(np.argmin(loss_array))

        return OptimizationResult(
            algorithm=self.name,
            loss_function=self.loss_function.name,
            initial_parameters=trajectory_array[0],
            final_parameters=current,
            initial_loss=initial_loss,
            final_loss=float(loss_array[-1]),
            best_loss=float(loss_array[best_index]),
            best_parameters=trajectory_array[best_index],
            iterations=iterations,
            gradient_norm=gradient_norm,
            elapsed_seconds=elapsed_seconds,
            cpu_seconds=cpu_seconds,
            status=status,
            stop_reason=stop_reason,
            trajectory=trajectory_array,
            loss_history=loss_array,
        )

    def _on_optimization_start(self, initial_parameters: FloatArray) -> None:
        """Reset any stateful bookkeeping before a new run.

        The default implementation does nothing; stateful optimizers such as
        Adam override it so that instances can be safely reused.
        """

    @abstractmethod
    def _compute_update(
        self,
        parameters: FloatArray,
        gradient: FloatArray,
        iteration: int,
    ) -> FloatArray:
        """Return the next parameter vector.

        Parameters
        ----------
        parameters:
            Current, validated parameter vector.
        gradient:
            Gradient of the loss at ``parameters``.
        iteration:
            One-based index of the current update, used by schedules such as
            Adam's bias correction.
        """
