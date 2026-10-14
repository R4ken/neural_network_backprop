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
        ...


class MSELoss(Loss):
    """Mean squared error loss."""

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the mean squared error."""
        ...

    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes ``2 * (y_pred - y_true) / N``."""
        ...


class BinaryCrossEntropyLoss(Loss):
    """Binary cross-entropy loss for two-class / multi-label problems."""

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the binary cross-entropy (with clipping for numerical stability)."""
        ...

    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes the gradient of binary cross-entropy with respect to predictions."""
        ...


class CategoricalCrossEntropyLoss(Loss):
    """Categorical cross-entropy loss for multi-class classification."""

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Computes the categorical cross-entropy (with clipping for numerical stability)."""
        ...

    def backward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Computes the gradient of categorical cross-entropy with respect to predictions."""
        ...
