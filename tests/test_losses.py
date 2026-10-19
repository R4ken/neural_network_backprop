"""Unit tests for loss functions in src.py_nn1.losses."""

import unittest

import numpy as np

from src.py_nn1 import MAELoss


class TestMAELoss(unittest.TestCase):
    def test_forward_computes_mean_absolute_error(self):
        loss = MAELoss()
        y_pred = np.array([[1.0], [2.0], [4.0]])
        y_true = np.array([[1.5], [1.0], [2.0]])

        self.assertAlmostEqual(loss.forward(y_pred, y_true), 7.0 / 6.0)

    def test_backward_returns_sign_gradient_scaled_by_size(self):
        loss = MAELoss()
        y_pred = np.array([[1.0], [3.0]])
        y_true = np.array([[2.0], [1.0]])

        gradient = loss.backward(y_pred, y_true)

        np.testing.assert_allclose(gradient, np.array([[-0.5], [0.5]]))

    def test_shape_mismatch_raises(self):
        loss = MAELoss()

        with self.assertRaises(ValueError):
            loss.forward(np.zeros((2, 1)), np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            loss.backward(np.zeros((2, 1)), np.zeros((3, 1)))


if __name__ == "__main__":
    unittest.main()
