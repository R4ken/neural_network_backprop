from __future__ import annotations

from .activations import Activation, ReLU, Sigmoid, Softmax, Tanh
from .layers import Dense, Layer
from .losses import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss, Loss, MSELoss
from .mlp import MLP
from .optimizers import SGD, Optimizer
from .utils import (
    batch_iterator,
    check_matching_shapes,
    compute_multiclass_accuracy,
    compute_multilabel_accuracy,
    get_rng,
    initialize_weights,
    set_seed,
)

__all__ = [
    "Activation",
    "BinaryCrossEntropyLoss",
    "CategoricalCrossEntropyLoss",
    "Dense",
    "Layer",
    "Loss",
    "MLP",
    "MSELoss",
    "Optimizer",
    "ReLU",
    "SGD",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "batch_iterator",
    "check_matching_shapes",
    "compute_multiclass_accuracy",
    "compute_multilabel_accuracy",
    "get_rng",
    "initialize_weights",
    "set_seed",
]


def main() -> None:
    """CLI entry point for the ``py-nn1`` command."""
    print("Hello from py-nn1!")
