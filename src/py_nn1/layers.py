"""Core layer abstractions and trainable layers for the neural network.

Every layer receives its inputs during :meth:`Layer.forward` and the upstream
gradient ``dL/dY`` during :meth:`Layer.backward`, returning the gradient with
respect to its inputs ``dL/dX``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class Layer(ABC):
    """Abstract base class for all layers in the network."""

    def __init__(self) -> None:
        """Initializes the layer (subclasses set up parameters/state here)."""
        ...

    @abstractmethod
    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Computes the layer output and caches state needed for the backward pass.

        Args:
            inputs: Input array of shape ``(batch_size, in_features)``.
            training: Whether the layer runs in training mode (affects e.g. dropout).

        Returns:
            Output array of shape ``(batch_size, out_features)``.
        """
        ...

    @abstractmethod
    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Computes parameter gradients and the gradient with respect to the inputs.

        Args:
            grad_output: Upstream gradient ``dL/dY`` of shape ``(batch_size, out_features)``.

        Returns:
            Gradient with respect to the inputs ``dL/dX`` of shape ``(batch_size, in_features)``.
        """
        ...

    @property
    def params(self) -> dict[str, FloatArray]:
        """Returns the trainable parameters, e.g. ``{"W": W, "b": b}``."""
        ...

    @property
    def grads(self) -> dict[str, FloatArray]:
        """Returns the parameter gradients with the same keys as :attr:`params`."""
        ...


class Dense(Layer):
    """A fully connected (linear) layer computing ``Y = X @ W + b``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: str = "he",
        use_bias: bool = True,
    ) -> None:
        """Creates a dense layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features (neurons).
            weight_init: Weight initialization scheme (e.g. ``"he"``, ``"xavier"``).
            use_bias: Whether to include a bias term.
        """
        ...

    def _initialize_weights(self) -> None:
        """Initializes ``self.W`` and ``self.b`` using the global random generator."""
        ...

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Computes ``X @ W + b`` and caches the inputs for backpropagation."""
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Computes ``dW``, ``db`` and returns the gradient with respect to the inputs."""
        ...

    @property
    def params(self) -> dict[str, FloatArray]:
        """Returns ``{"W": self.W, "b": self.b}``."""
        ...

    @property
    def grads(self) -> dict[str, FloatArray]:
        """Returns ``{"W": self.dW, "b": self.db}``."""
        ...
