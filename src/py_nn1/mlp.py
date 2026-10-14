"""Multi-Layer Perceptron model container and training loop."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .layers import Layer
from .losses import Loss
from .optimizers import Optimizer

FloatArray = NDArray[np.floating]


class MLP:
    """A sequential stack of layers trained with backpropagation."""

    def __init__(self, layers: list[Layer] | None = None) -> None:
        """Creates an MLP.

        Args:
            layers: Optional initial list of layers (defaults to an empty network).
        """
        ...

    @classmethod
    def from_dimensions(
        cls,
        layer_dims: list[int],
        hidden_activation: type[Layer],
        output_activation: type[Layer] | None = None,
    ) -> "MLP":
        """Builds an MLP from a list of neuron counts.

        Example:
            ``MLP.from_dimensions([784, 128, 10], ReLU(), Softmax())``.

        Args:
            layer_dims: Neuron counts per layer, e.g. ``[784, 128, 64, 10]``.
            hidden_activation: Activation instance inserted after each hidden dense layer.
            output_activation: Optional activation instance appended after the output layer.

        Returns:
            A configured :class:`MLP` instance.
        """
        ...

    def add(self, layer: Layer) -> None:
        """Appends a layer to the network.

        Args:
            layer: The layer to append.
        """
        ...

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Passes inputs sequentially through all layers.

        Args:
            inputs: Input array of shape ``(batch_size, num_features)``.
            training: Whether the network runs in training mode.

        Returns:
            The network output.
        """
        ...

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Propagates the loss gradient backward through all layers in reverse order.

        Args:
            grad_output: Gradient ``dL/dy_pred`` produced by the loss.

        Returns:
            The gradient with respect to the network inputs.
        """
        ...

    def train_step(
        self,
        x_batch: FloatArray,
        y_batch: FloatArray,
        loss_fn: Loss,
        optimizer: Optimizer,
    ) -> float:
        """Runs a single forward pass, loss computation, backward pass and optimizer step.

        Args:
            x_batch: Batch of input features.
            y_batch: Batch of targets.
            loss_fn: Loss function used to compute the objective and its gradient.
            optimizer: Optimizer used to update the parameters.

        Returns:
            The loss value for this batch.
        """
        ...

    def fit(
        self,
        x_train: FloatArray,
        y_train: FloatArray,
        loss_fn: Loss,
        optimizer: Optimizer,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple[FloatArray, FloatArray] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Trains the network over the given number of epochs and mini-batches.

        Args:
            x_train: Training features.
            y_train: Training targets.
            loss_fn: Loss function to optimize.
            optimizer: Optimizer used to update the parameters.
            epochs: Number of passes over the training data.
            batch_size: Number of samples per mini-batch.
            validation_data: Optional ``(x_val, y_val)`` evaluated each epoch.
            verbose: Whether to print progress.

        Returns:
            A history dictionary, e.g. ``{"loss": [...], "val_loss": [...]}``.
        """
        ...

    def predict(self, inputs: FloatArray) -> FloatArray:
        """Runs a forward pass in inference mode (``training=False``).

        Args:
            inputs: Input array of shape ``(num_samples, num_features)``.

        Returns:
            The network output.
        """
        ...

    def evaluate(
        self,
        x_test: FloatArray,
        y_test: FloatArray,
        loss_fn: Loss,
    ) -> float:
        """Computes the loss of the network on a dataset.

        Args:
            x_test: Test features.
            y_test: Test targets.
            loss_fn: Loss function used for evaluation.

        Returns:
            The mean loss over the dataset.
        """
        ...
