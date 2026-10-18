"""Unit tests for numerical metrics in src.py_nn1.utils."""

import unittest

import numpy as np

from src.py_nn1 import compute_mae, compute_r2


class TestRegressionMetrics(unittest.TestCase):
    def test_compute_mae(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.5], [1.0], [4.0]])

        self.assertAlmostEqual(compute_mae(y_pred, y_true), 5.0 / 6.0)

    def test_compute_r2_perfect_prediction(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])

        self.assertAlmostEqual(compute_r2(y_true, y_true), 1.0)

    def test_compute_r2_mean_predictor(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.full_like(y_true, np.mean(y_true))

        self.assertAlmostEqual(compute_r2(y_pred, y_true), 0.0)

    def test_compute_r2_constant_targets(self):
        y_true = np.ones((4, 1))
        y_pred = np.full_like(y_true, 2.0)

        self.assertEqual(compute_r2(y_pred, y_true), 0.0)

    def test_metrics_reject_shape_mismatch(self):
        with self.assertRaises(ValueError):
            compute_mae(np.zeros((3, 1)), np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            compute_r2(np.zeros((3, 1)), np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
