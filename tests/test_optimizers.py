"""Unit tests for the Phase 2 optimization use cases."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.typing import ArrayLike

from domain import FloatArray, LossFunction, QuadraticLoss, RosenbrockLoss
from use_cases import (
    AdamOptimizer,
    GradientDescentOptimizer,
    NewtonRaphsonOptimizer,
    OptimizationStatus,
)


class _SingularHessianLoss(LossFunction):
    """Linear toy loss with a zero Hessian, used to test failure handling."""

    @property
    def name(self) -> str:
        return "Singular Hessian Test Loss"

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, parameters: ArrayLike) -> float:
        x, _ = self.validate_parameters(parameters)
        return float(x)

    def gradient(self, parameters: ArrayLike) -> FloatArray:
        self.validate_parameters(parameters)
        return np.array([1.0, 0.0], dtype=np.float64)

    def hessian(self, parameters: ArrayLike) -> FloatArray:
        self.validate_parameters(parameters)
        return np.zeros((2, 2), dtype=np.float64)


class QuadraticOptimizationTests(unittest.TestCase):
    """Behaviour of the three algorithms on the convex landscape."""

    def setUp(self) -> None:
        self.loss = QuadraticLoss()
        self.start = np.array([-1.5, 1.5], dtype=np.float64)

    def _assert_result_contract(self, result, optimizer) -> None:
        self.assertEqual(result.algorithm, optimizer.name)
        self.assertEqual(result.loss_function, self.loss.name)
        self.assertEqual(result.iterations, result.trajectory.shape[0] - 1)
        self.assertEqual(result.loss_history.shape, (result.trajectory.shape[0],))
        np.testing.assert_allclose(result.trajectory[0], self.start)
        np.testing.assert_allclose(result.final_parameters, result.trajectory[-1])
        self.assertAlmostEqual(result.initial_loss, self.loss.evaluate(self.start))
        self.assertAlmostEqual(result.final_loss, result.loss_history[-1])
        self.assertAlmostEqual(result.best_loss, result.loss_history.min())
        self.assertLessEqual(result.best_loss, result.final_loss)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertGreaterEqual(result.cpu_seconds, 0.0)
        self.assertFalse(result.trajectory.flags.writeable)
        self.assertFalse(result.loss_history.flags.writeable)

    def test_gradient_descent_converges_to_the_known_minimum(self) -> None:
        optimizer = GradientDescentOptimizer(
            self.loss,
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-8,
        )
        result = optimizer.optimize(self.start)

        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.final_parameters, [1.0, 1.0], atol=1e-5)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(np.all(np.diff(result.loss_history) <= 0.0))
        self._assert_result_contract(result, optimizer)

    def test_adam_converges_and_restarts_cleanly(self) -> None:
        optimizer = AdamOptimizer(
            self.loss,
            learning_rate=0.05,
            max_iterations=2000,
            tolerance=1e-8,
        )
        first = optimizer.optimize(self.start)
        second = optimizer.optimize(self.start)

        self.assertTrue(first.converged)
        np.testing.assert_allclose(
            first.final_parameters, [1.0, 1.0], atol=1e-5
        )
        np.testing.assert_allclose(
            first.trajectory, second.trajectory, atol=1e-12
        )
        self._assert_result_contract(first, optimizer)

    def test_newton_solves_the_quadratic_in_one_step(self) -> None:
        optimizer = NewtonRaphsonOptimizer(
            self.loss,
            learning_rate=1.0,
            max_iterations=10,
            tolerance=1e-8,
        )
        result = optimizer.optimize(self.start)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_allclose(
            result.final_parameters, [1.0, 1.0], atol=1e-12
        )
        self.assertAlmostEqual(result.final_loss, 0.0, places=12)
        self._assert_result_contract(result, optimizer)

    def test_max_iterations_status_and_history(self) -> None:
        optimizer = GradientDescentOptimizer(
            self.loss,
            learning_rate=0.01,
            max_iterations=5,
            tolerance=0.0,
        )
        result = optimizer.optimize(self.start)

        self.assertEqual(result.status, OptimizationStatus.MAX_ITERATIONS)
        self.assertEqual(result.iterations, 5)
        self.assertEqual(result.trajectory.shape, (6, 2))

    def test_divergent_run_is_reported_without_raising(self) -> None:
        optimizer = GradientDescentOptimizer(
            self.loss,
            learning_rate=1000.0,
            max_iterations=100,
            tolerance=0.0,
        )
        result = optimizer.optimize(self.start)

        self.assertEqual(result.status, OptimizationStatus.DIVERGED)
        self.assertLess(result.iterations, 100)
        self.assertTrue(np.all(np.isfinite(result.trajectory)))
        self.assertTrue(np.all(np.isfinite(result.loss_history)))


class RosenbrockOptimizationTests(unittest.TestCase):
    """Behaviour of the algorithms on the non-convex landscape."""

    def setUp(self) -> None:
        self.loss = RosenbrockLoss()
        self.start = np.array([-1.2, 1.0], dtype=np.float64)

    def test_newton_converges_in_few_iterations(self) -> None:
        optimizer = NewtonRaphsonOptimizer(
            self.loss,
            learning_rate=1.0,
            max_iterations=100,
            tolerance=1e-8,
        )
        result = optimizer.optimize(self.start)

        self.assertTrue(result.converged)
        self.assertLess(result.iterations, 20)
        self.assertLess(result.final_loss, 1e-10)
        np.testing.assert_allclose(
            result.final_parameters, [1.0, 1.0], atol=1e-5
        )

    def test_adam_navigates_the_valley(self) -> None:
        optimizer = AdamOptimizer(
            self.loss,
            learning_rate=0.05,
            max_iterations=5000,
            tolerance=1e-8,
        )
        result = optimizer.optimize(self.start)

        self.assertTrue(result.converged)
        self.assertLess(result.best_loss, 1e-10)
        np.testing.assert_allclose(
            result.best_parameters, [1.0, 1.0], atol=1e-4
        )

    def test_gradient_descent_improves_but_needs_many_iterations(self) -> None:
        optimizer = GradientDescentOptimizer(
            self.loss,
            learning_rate=2e-3,
            max_iterations=5000,
            tolerance=1e-8,
        )
        result = optimizer.optimize(self.start)

        self.assertLess(result.best_loss, result.initial_loss)
        self.assertLess(result.best_loss, 1e-2)
        self.assertGreater(result.iterations, 100)


class OptimizerValidationTests(unittest.TestCase):
    """Constructor validation and numerical failure handling."""

    def setUp(self) -> None:
        self.loss = QuadraticLoss()

    def test_rejects_invalid_learning_rates(self) -> None:
        for optimizer_class in (
            GradientDescentOptimizer,
            AdamOptimizer,
            NewtonRaphsonOptimizer,
        ):
            with self.subTest(optimizer=optimizer_class.__name__):
                with self.assertRaises(ValueError):
                    optimizer_class(self.loss, learning_rate=0.0)
                with self.assertRaises(ValueError):
                    optimizer_class(self.loss, learning_rate=float("nan"))

    def test_rejects_invalid_adam_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            AdamOptimizer(self.loss, beta1=1.0)
        with self.assertRaises(ValueError):
            AdamOptimizer(self.loss, beta2=-0.1)
        with self.assertRaises(ValueError):
            AdamOptimizer(self.loss, epsilon=0.0)

    def test_rejects_invalid_common_settings(self) -> None:
        with self.assertRaises(ValueError):
            GradientDescentOptimizer(self.loss, max_iterations=0)
        with self.assertRaises(ValueError):
            GradientDescentOptimizer(self.loss, tolerance=-1.0)
        with self.assertRaises(TypeError):
            GradientDescentOptimizer(self.loss, max_iterations=2.5)
        with self.assertRaises(TypeError):
            GradientDescentOptimizer("not a loss function")  # type: ignore[arg-type]

    def test_singular_hessian_is_reported_as_failure(self) -> None:
        optimizer = NewtonRaphsonOptimizer(
            _SingularHessianLoss(),
            max_iterations=5,
            tolerance=0.0,
        )
        result = optimizer.optimize([0.0, 0.0])

        self.assertEqual(result.status, OptimizationStatus.FAILED)
        self.assertIn("singular", result.stop_reason)
        self.assertEqual(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()
