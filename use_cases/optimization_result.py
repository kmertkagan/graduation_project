"""Use-case level data structures produced by optimization runs.

The result object is deliberately independent of plotting and console
formatting: presentation adapters consume it, but it never imports them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from domain import FloatArray


class OptimizationStatus(str, Enum):
    """Terminal state of an optimization run."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged"
    FAILED = "failed"


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable snapshot of a single optimizer run.

    Attributes
    ----------
    algorithm:
        English name of the optimizer.
    loss_function:
        English name of the optimized loss function.
    initial_parameters:
        Starting point supplied by the caller.
    final_parameters:
        Last valid parameter vector reached by the optimizer.
    initial_loss:
        Loss value at ``initial_parameters``.
    final_loss:
        Loss value at ``final_parameters``.
    best_loss:
        Smallest loss observed along the trajectory.
    best_parameters:
        Parameters at which ``best_loss`` was observed.
    iterations:
        Number of completed parameter updates.
    gradient_norm:
        Euclidean norm of the gradient at the last valid point.
    elapsed_seconds:
        Wall-clock duration of the run in seconds.
    cpu_seconds:
        Process CPU time consumed by the run in seconds.
    status:
        Why the optimization loop terminated.
    stop_reason:
        English diagnostic message describing the termination.
    trajectory:
        Array of shape ``(iterations + 1, dimension)`` with every visited
        parameter vector, starting at ``initial_parameters``.
    loss_history:
        Array of shape ``(iterations + 1,)`` aligned with ``trajectory``.
    """

    algorithm: str
    loss_function: str
    initial_parameters: FloatArray
    final_parameters: FloatArray
    initial_loss: float
    final_loss: float
    best_loss: float
    best_parameters: FloatArray
    iterations: int
    gradient_norm: float
    elapsed_seconds: float
    cpu_seconds: float
    status: OptimizationStatus
    stop_reason: str
    trajectory: FloatArray
    loss_history: FloatArray

    def __post_init__(self) -> None:
        trajectory = np.array(self.trajectory, dtype=np.float64, copy=True)
        loss_history = np.array(self.loss_history, dtype=np.float64, copy=True)

        if trajectory.ndim != 2:
            raise ValueError("trajectory must be a two-dimensional array.")
        if loss_history.shape != (trajectory.shape[0],):
            raise ValueError(
                "loss_history must contain one value per trajectory point."
            )

        for field_name in (
            "initial_parameters",
            "final_parameters",
            "best_parameters",
        ):
            array = np.array(
                getattr(self, field_name), dtype=np.float64, copy=True
            )
            if array.shape != (trajectory.shape[1],):
                raise ValueError(
                    f"{field_name} must have shape ({trajectory.shape[1]},)."
                )
            array.setflags(write=False)
            object.__setattr__(self, field_name, array)

        trajectory.setflags(write=False)
        loss_history.setflags(write=False)
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "loss_history", loss_history)

    @property
    def converged(self) -> bool:
        """Whether the run met the gradient-norm tolerance."""
        return self.status is OptimizationStatus.CONVERGED
