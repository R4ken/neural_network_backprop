"""Utility helpers: global random seed, weight initialization and data batching.

A module-level NumPy :class:`~numpy.random.Generator` is shared across the
library so that calling :func:`set_seed` makes weight initialization and batch
shuffling deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def set_seed(seed: int | None = None) -> None:
    """Sets the global random seed for the whole library.

    Controls weight initialization and batch shuffling so runs are reproducible.

    Args:
        seed: The seed to use. ``None`` re-seeds from OS entropy.
    """
    ...


def get_rng() -> np.random.Generator:
    """Returns the shared global NumPy random generator."""
    ...


def initialize_weights(shape: tuple[int, ...], method: str = "he") -> FloatArray:
    """Initializes a weight array using the global random generator.

    Args:
        shape: Shape of the weight array to create.
        method: Initialization scheme, e.g. ``"he"``, ``"xavier"``, ``"normal"``, ``"zeros"``.

    Returns:
        A new array of the requested shape.
    """
    ...


def batch_iterator(
    x: FloatArray,
    y: FloatArray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Iterator[tuple[FloatArray, FloatArray]]:
    """Yields mini-batches of ``(x_batch, y_batch)``.

    Args:
        x: Feature array of shape ``(num_samples, num_features)``.
        y: Target array of shape ``(num_samples, ...)``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the samples (using the global RNG) each epoch.

    Yields:
        Tuples ``(x_batch, y_batch)`` for each mini-batch.
    """
    ...
