"""Multi-Layer Perceptron model container and training loop."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .activations import Activation
from .layers import Dense, Layer
from .losses import Loss
from .optimizers import Optimizer
from .utils import batch_iterator

FloatArray = NDArray[np.floating]


class MLP:
    """A sequential stack of layers trained with backpropagation."""

    def __init__(self, layers: list[Layer] | None = None) -> None:
        """Creates an MLP.

        Args:
            layers: Optional initial list of layers (defaults to an empty network).
        """
        self.layers: list[Layer] = list(layers) if layers is not None else []

    @classmethod
    def from_dimensions(
        cls,
        layer_dims: list[int],
        hidden_activation: type[Activation],
        output_activation: type[Activation] | None = None,
    ) -> MLP:
        """Builds an MLP from a list of neuron counts.

        Example:
            ``MLP.from_dimensions([784, 128, 10], ReLU, Softmax)``.

        Args:
            layer_dims: Neuron counts per layer, e.g. ``[784, 128, 64, 10]``.
            hidden_activation: Activation type inserted after each hidden dense layer.
            output_activation: Optional activation type appended after the output layer.

        Returns:
            A configured :class:`MLP` instance.
        """
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must contain at least 2 dimensions (input and output).")

        layers: list[Layer] = []
        num_layers = len(layer_dims) - 1
        for i in range(num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            layers.append(Dense(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(hidden_activation())
            elif output_activation is not None:
                layers.append(output_activation())

        return cls(layers)

    def add(self, layer: Layer) -> None:
        """Appends a layer to the network.

        Args:
            layer: The layer to append.
        """
        self.layers.append(layer)

    def forward(self, inputs: FloatArray, training: bool = True) -> FloatArray:
        """Passes inputs sequentially through all layers.

        Args:
            inputs: Input array of shape ``(batch_size, num_features)``.
            training: Whether the network runs in training mode.

        Returns:
            The network output.
        """
        out = inputs
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Propagates the loss gradient backward through all layers in reverse order.

        Args:
            grad_output: Gradient ``dL/dy_pred`` produced by the loss.

        Returns:
            The gradient with respect to the network inputs.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

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
        optimizer.zero_grad(self.layers)
        y_pred = self.forward(x_batch, training=True)
        loss = loss_fn.forward(y_pred, y_batch)
        grad_output = loss_fn.backward(y_pred, y_batch)
        self.backward(grad_output)
        optimizer.step(self.layers)
        return float(loss)

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
        history: dict[str, list[float]] = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        for epoch in range(1, epochs + 1):
            epoch_loss_sum = 0.0
            total_samples = 0
            for x_batch, y_batch in batch_iterator(
                x_train, y_train, batch_size=batch_size, shuffle=True
            ):
                batch_loss = self.train_step(x_batch, y_batch, loss_fn, optimizer)
                batch_len = len(x_batch)
                epoch_loss_sum += batch_loss * batch_len
                total_samples += batch_len

            epoch_loss = float(epoch_loss_sum / total_samples) if total_samples > 0 else 0.0
            history["loss"].append(epoch_loss)

            if validation_data is not None:
                x_val, y_val = validation_data
                val_loss = self.evaluate(x_val, y_val, loss_fn)
                history["val_loss"].append(val_loss)
                if verbose:
                    print(
                        f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}"
                    )
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

        return history

    def predict(self, inputs: FloatArray) -> FloatArray:
        """Runs a forward pass in inference mode (``training=False``).

        Args:
            inputs: Input array of shape ``(num_samples, num_features)``.

        Returns:
            The network output.
        """
        return self.forward(inputs, training=False)

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
        y_pred = self.predict(x_test)
        return float(loss_fn.forward(y_pred, y_test))
