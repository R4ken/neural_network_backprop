"""Loss functions.

A loss computes a scalar objective during the forward pass and produces the
gradient ``dL/dy_pred`` that seeds the backward pass through the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the scalar loss value.

        Args:
            y_pred: Model predictions.
            y_true: Ground-truth targets.

        Returns:
            The mean loss over the batch.
        """
        ...

    @abstractmethod
    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes the gradient of the loss with respect to the predictions.

        Args:
            y_pred: Model predictions.
            y_true: Ground-truth targets.

        Returns:
            Gradient ``dL/dy_pred`` with the same shape as ``y_pred``.
        """
        ...

    def __call__(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Convenience wrapper around :meth:`forward`."""
        return self.forward(y_pred, y_true)


class MSELoss(Loss):
    """Mean squared error loss."""

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the mean squared error."""
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes ``2 * (y_pred - y_true) / N``."""
        return 2.0 * (y_pred - y_true) / y_pred.size


class BinaryCrossEntropyLoss(Loss):
    """Binary cross-entropy loss for two-class / multi-label problems."""

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the binary cross-entropy (with clipping for numerical stability)."""
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes the gradient of binary cross-entropy with respect to predictions."""
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        return ((-y_true / p) + (1.0 - y_true) / (1.0 - p)) / y_pred.size


class CategoricalCrossEntropyLoss(Loss):
    """Categorical cross-entropy loss for multi-class classification."""

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the categorical cross-entropy (with clipping for numerical stability)."""
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes the gradient of categorical cross-entropy with respect to predictions."""
        eps = 1e-15
        p = np.clip(y_pred, eps, 1.0 - eps)
        n = y_pred.shape[0]
        return (-y_true / p) / n
