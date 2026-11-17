"""Unit tests for the Phase 1 domain layer."""

from __future__ import annotations

import unittest

import numpy as np

from domain import LossFunction, QuadraticLoss, RosenbrockLoss


def numerical_gradient(
    loss: LossFunction,
    point: np.ndarray,
    step: float = 1e-6,
) -> np.ndarray:
    """Approximate the gradient with central finite differences."""
    gradient = np.zeros_like(point, dtype=np.float64)
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset[index] = step
        gradient[index] = (
            loss.evaluate(point + offset) - loss.evaluate(point - offset)
        ) / (2.0 * step)
    return gradient


def numerical_hessian(
    loss: LossFunction,
    point: np.ndarray,
    step: float = 1e-4,
) -> np.ndarray:
    """Approximate the Hessian with central differences of gradients."""
    dimension = point.size
    hessian = np.zeros((dimension, dimension), dtype=np.float64)
    for index in range(dimension):
        offset = np.zeros_like(point)
        offset[index] = step
        hessian[:, index] = (
            loss.gradient(point + offset) - loss.gradient(point - offset)
        ) / (2.0 * step)
    return hessian


class QuadraticLossTests(unittest.TestCase):
    """Contract and derivative checks for the convex quadratic loss."""

    def setUp(self) -> None:
        self.loss = QuadraticLoss()
        self.point = np.array([-1.5, 0.5], dtype=np.float64)

    def test_metadata_and_known_minimum(self) -> None:
        self.assertEqual(self.loss.name, "Quadratic Loss")
        self.assertEqual(self.loss.dimension, 2)
        self.assertTrue(self.loss.is_convex)

        minimizer = self.loss.known_minimizer
        np.testing.assert_allclose(minimizer, [1.0, 1.0])
        self.assertAlmostEqual(self.loss.evaluate(minimizer), 0.0)
        self.assertAlmostEqual(self.loss.known_minimum_value, 0.0)
        np.testing.assert_allclose(
            self.loss.gradient(minimizer),
            np.zeros(2),
            atol=1e-12,
        )

    def test_gradient_matches_finite_differences(self) -> None:
        np.testing.assert_allclose(
            self.loss.gradient(self.point),
            numerical_gradient(self.loss, self.point),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_hessian_is_constant_and_matches_finite_differences(self) -> None:
        expected = np.array([[3.0, 1.0], [1.0, 2.0]])
        np.testing.assert_allclose(self.loss.hessian(self.point), expected)
        np.testing.assert_allclose(
            self.loss.hessian(self.point),
            numerical_hessian(self.loss, self.point),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_custom_configuration(self) -> None:
        loss = QuadraticLoss(
            matrix=[[4.0, 0.0], [0.0, 1.0]],
            center=[2.0, -3.0],
            constant=1.5,
        )
        self.assertEqual(loss.dimension, 2)
        self.assertAlmostEqual(loss.evaluate([2.0, -3.0]), 1.5)
        np.testing.assert_allclose(loss.gradient([3.0, -3.0]), [4.0, 0.0])
        np.testing.assert_allclose(
            loss.hessian([0.0, 0.0]),
            [[4.0, 0.0], [0.0, 1.0]],
        )

    def test_hessian_is_returned_as_a_copy(self) -> None:
        hessian = self.loss.hessian(self.point)
        hessian[0, 0] = 999.0
        self.assertEqual(self.loss.hessian(self.point)[0, 0], 3.0)

    def test_rejects_invalid_curvature_matrix(self) -> None:
        with self.assertRaises(ValueError):
            QuadraticLoss(matrix=[[1.0, 2.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            QuadraticLoss(matrix=[[1.0, 0.0], [0.0, -1.0]])
        with self.assertRaises(ValueError):
            QuadraticLoss(matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            QuadraticLoss(center=[1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            QuadraticLoss(center=[np.nan, 1.0])
        with self.assertRaises(ValueError):
            QuadraticLoss(constant=-1.0)
        with self.assertRaises(ValueError):
            QuadraticLoss(constant=float("inf"))

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            self.loss.evaluate([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.loss.gradient([1.0, np.inf])
        with self.assertRaises(ValueError):
            self.loss.hessian([np.nan, 0.0])


class RosenbrockLossTests(unittest.TestCase):
    """Contract and derivative checks for the non-convex Rosenbrock loss."""

    def setUp(self) -> None:
        self.loss = RosenbrockLoss()
        self.point = np.array([-1.2, 1.0], dtype=np.float64)

    def test_metadata_and_known_minimum(self) -> None:
        self.assertEqual(self.loss.name, "Rosenbrock Loss")
        self.assertEqual(self.loss.dimension, 2)
        self.assertFalse(self.loss.is_convex)

        minimizer = self.loss.known_minimizer
        np.testing.assert_allclose(minimizer, [1.0, 1.0])
        self.assertAlmostEqual(self.loss.evaluate(minimizer), 0.0)
        self.assertAlmostEqual(self.loss.known_minimum_value, 0.0)
        np.testing.assert_allclose(
            self.loss.gradient(minimizer),
            np.zeros(2),
            atol=1e-12,
        )

    def test_gradient_matches_finite_differences(self) -> None:
        np.testing.assert_allclose(
            self.loss.gradient(self.point),
            numerical_gradient(self.loss, self.point),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_hessian_matches_finite_differences(self) -> None:
        np.testing.assert_allclose(
            self.loss.hessian(self.point),
            numerical_hessian(self.loss, self.point),
            rtol=1e-4,
            atol=1e-3,
        )

    def test_hessian_is_indefinite_on_the_landscape(self) -> None:
        # At (0, 2) the Hessian has a negative eigenvalue, proving that the
        # function is genuinely non-convex away from its global minimum.
        eigenvalues = np.linalg.eigvalsh(
            self.loss.hessian(np.array([0.0, 2.0]))
        )
        self.assertLess(eigenvalues.min(), 0.0)

    def test_hessian_is_positive_definite_at_the_minimizer(self) -> None:
        eigenvalues = np.linalg.eigvalsh(
            self.loss.hessian(self.loss.known_minimizer)
        )
        self.assertTrue(np.all(eigenvalues > 0.0))

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            self.loss.evaluate([1.0])
        with self.assertRaises(ValueError):
            self.loss.gradient([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.loss.hessian([np.nan, 1.0])


if __name__ == "__main__":
    unittest.main()
