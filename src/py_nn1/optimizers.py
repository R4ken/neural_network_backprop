"""Optimizers that update layer parameters from their stored gradients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .layers import Layer


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        """Stores the learning rate used for updates.

        Args:
            learning_rate: Step size applied to parameter gradients.
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers: Iterable[Layer]) -> None:
        """Updates the parameters of every layer using its stored gradients.

        Args:
            layers: The layers whose parameters should be updated.
        """
        ...

    def zero_grad(self, layers: Iterable[Layer]) -> None:
        """Resets the accumulated gradients of every layer to zero.

        Args:
            layers: The layers whose gradients should be cleared.
        """
        for layer in layers:
            for grad in layer.grads.values():
                grad.fill(0.0)


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(
        self,
        learning_rate: float = 0.01,
    ) -> None:
        """Creates an SGD optimizer.

        Args:
            learning_rate: Step size applied to parameter gradients.
        """
        super().__init__(learning_rate)

    def step(self, layers: Iterable[Layer]) -> None:
        """Performs one SGD update over the parameters of the given layers."""
        for layer in layers:
            params = layer.params
            grads = layer.grads
            for key in params:
                params[key] -= self.learning_rate * grads[key]
