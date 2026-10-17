"""Core layer abstractions and trainable layers for the neural network.

Every layer receives its inputs during :meth:`Layer.forward` and the upstream
gradient ``dL/dY`` during :meth:`Layer.backward`, returning the gradient with
respect to its inputs ``dL/dX``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .utils import initialize_weights

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
            inputs: Input array of shape ``(batch_size, in_neruons)``.
            training: Whether the layer runs in training mode (affects e.g. dropout).

        Returns:
            Output array of shape ``(batch_size, out_neruons)``.
        """
        ...

    @abstractmethod
    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Computes parameter gradients and the gradient with respect to the inputs.

        Args:
            grad_output: Upstream gradient ``dL/dY`` of shape ``(batch_size, out_neruons)``.

        Returns:
            Gradient with respect to the inputs ``dL/dX`` of shape ``(batch_size, in_neruons)``.
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
    """A fully connected (linear) layer computing ``Y = X @ W + b`` (or ``Y = X @ W``)."""

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
            use_bias: Whether to include and learn a bias vector.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_init = weight_init
        self.use_bias = use_bias

        self.W: FloatArray = np.empty(0)
        self.b: FloatArray | None = None
        self._initialize_weights()

        # Gradient (computed during backward pass)
        self.dW: FloatArray = np.zeros_like(self.W)
        self.db: FloatArray | None = (
            np.zeros_like(self.b) if self.use_bias and self.b is not None else None
        )

        # Saving inputs during backward pass
        self._inputs: FloatArray | None = None

    def _initialize_weights(self) -> None:
        """Initializes ``self.W`` and optionally ``self.b``."""
        self.W = initialize_weights(
            (self.in_features, self.out_features), method=self.weight_init
        )
        if self.use_bias:
            self.b = np.zeros((1, self.out_features))
        else:
            self.b = None

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Computes ``X @ W + b`` (or ``X @ W``) and caches inputs for backpropagation."""
        self._inputs = inputs
        output = inputs @ self.W
        if self.use_bias and self.b is not None:
            output = output + self.b
        return output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Computes parameter gradients and returns gradient with respect to inputs."""
        self.dW = self._inputs.T @ grad_output
        if self.use_bias and self.db is not None:
            self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input

    @property
    def params(self) -> dict[str, FloatArray]:
        """Returns trainable parameters (``{"W": self.W}`` or with ``"b"``)."""
        p = {"W": self.W}
        if self.use_bias and self.b is not None:
            p["b"] = self.b
        return p

    @property
    def grads(self) -> dict[str, FloatArray]:
        """Returns parameter gradients matching keys in :attr:`params`."""
        g = {"W": self.dW}
        if self.use_bias and self.db is not None:
            g["b"] = self.db
        return g
