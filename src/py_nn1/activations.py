"""Parameter-free activation layers.

Modeling activations as standalone layers keeps backpropagation uniform: the
:class:`~py_nn1.mlp.MLP` iterates through layers in order for the forward pass
and in reverse for the backward pass, without special-casing activations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .layers import Layer

FloatArray = NDArray[np.floating]


class Activation(Layer):
    """Base class for parameter-free activation layers (no trainable parameters)."""

    def __init__(self) -> None:
        """Initializes the activation layer."""
        super().__init__()

    @property
    def params(self) -> dict[str, FloatArray]:
        """Returns an empty dictionary (activations have no parameters)."""
        return {}

    @property
    def grads(self) -> dict[str, FloatArray]:
        """Returns an empty dictionary (activations have no gradients)."""
        return {}


class ReLU(Activation):
    """Rectified Linear Unit activation: ``max(0, x)``."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies ReLU element-wise and caches the output/mask for the backward pass."""
        self._mask = inputs > 0
        return inputs * self._mask

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Multiplies the upstream gradient by the ReLU derivative mask."""
        return grad_output * self._mask


class Sigmoid(Activation):
    """Sigmoid activation: ``1 / (1 + exp(-x))``."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies the sigmoid element-wise and caches the output for the backward pass."""
        self._output = 1.0 / (1.0 + np.exp(-inputs))
        return self._output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Multiplies the upstream gradient by ``sigmoid(x) * (1 - sigmoid(x))``."""
        return grad_output * self._output * (1.0 - self._output)


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies tanh element-wise and caches the output for the backward pass."""
        self._output = np.tanh(inputs)
        return self._output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Multiplies the upstream gradient by ``1 - tanh(x)**2``."""
        return grad_output * (1.0 - self._output ** 2)


class Softmax(Activation):
    """Softmax activation turning logits into a probability distribution."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies the numerically stable softmax along the feature axis."""
        # Subtract row-max for numerical stability
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        self._output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Computes the gradient of the softmax with respect to its inputs."""
        # Jacobian of softmax row i is diag(S_i) - S_i @ S_i^T
        # The JVP is: dX = S * (dY - sum(dY * S, axis=1, keepdims=True))
        dot = np.sum(grad_output * self._output, axis=1, keepdims=True)
        return self._output * (grad_output - dot)
