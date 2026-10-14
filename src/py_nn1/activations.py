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
        ...

    @property
    def params(self) -> dict[str, FloatArray]:
        """Returns an empty dictionary (activations have no parameters)."""
        ...

    @property
    def grads(self) -> dict[str, FloatArray]:
        """Returns an empty dictionary (activations have no gradients)."""
        ...


class ReLU(Activation):
    """Rectified Linear Unit activation: ``max(0, x)``."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies ReLU element-wise and caches the output/mask for the backward pass."""
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Multiplies the upstream gradient by the ReLU derivative mask."""
        ...


class Sigmoid(Activation):
    """Sigmoid activation: ``1 / (1 + exp(-x))``."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies the sigmoid element-wise and caches the output for the backward pass."""
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Multiplies the upstream gradient by ``sigmoid(x) * (1 - sigmoid(x))``."""
        ...


class Tanh(Activation):
    """Hyperbolic tangent activation."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies tanh element-wise and caches the output for the backward pass."""
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Multiplies the upstream gradient by ``1 - tanh(x)**2``."""
        ...


class Softmax(Activation):
    """Softmax activation turning logits into a probability distribution."""

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Applies the numerically stable softmax along the feature axis."""
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Computes the gradient of the softmax with respect to its inputs."""
        ...
