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

# Global random number generator
_rng: np.random.Generator = np.random.default_rng()


def set_seed(seed: int | None = None) -> None:
    """Sets the global random seed for the whole library.

    Controls weight initialization and batch shuffling so runs are reproducible.

    Args:
        seed: The seed to use. ``None`` re-seeds from OS entropy.
    """
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Returns the shared global NumPy random generator."""
    return _rng


def initialize_weights(shape: tuple[int, ...], method: str = "he") -> FloatArray:
    """Initializes a weight array using the global random generator.

    Args:
        shape: Shape of the weight array to create.
        method: Initialization scheme, e.g. ``"he"``, ``"xavier"``, ``"normal"``, ``"zeros"``.

    Source for ``"he"`` and ``"xavier"``:
        https://medium.com/thedeephub/he-and-xavier-weight-initialization-functions-acedc5322ce5

    Returns:
        A new array of the requested shape.
    """
    fan_in = shape[0]

    if method == "he":
        scale = np.sqrt(2.0 / fan_in)
        return _rng.standard_normal(shape) * scale
    elif method == "xavier":
        scale = np.sqrt(1.0 / fan_in)
        return _rng.standard_normal(shape) * scale
    elif method == "normal":
        return _rng.standard_normal(shape)
    elif method == "zeros":
        return np.zeros(shape)
    else:
        raise ValueError(f"Unknown initialization method: {method!r}")


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
    n_samples = x.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        _rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]

    if shuffle:
        _rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]

