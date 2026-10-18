"""Data loading utilities for MLP regression and classification experiments.

This module provides dataset loaders for the benchmarks defined in data/p1/:
- Regression: 'data.activation' and 'data.cube'
- Classification: 'data.simple' and 'data.three_gauss'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

def to_one_hot(y: NDArray[np.integer] | FloatArray, num_classes: int | None = None) -> FloatArray:
    """Converts 0-indexed integer labels into a one-hot encoded binary matrix.

    Args:
        y: Integer class labels of shape (N,) or (N, 1).
        num_classes: Number of distinct classes. If None, inferred as max(y) + 1.

    Returns:
        One-hot matrix of shape (N, num_classes) with float values.
    """
    labels = np.asarray(y, dtype=int).ravel()
    if labels.size == 0:
        n_classes = 0 if num_classes is None else num_classes
        return np.empty((0, n_classes), dtype=float)

    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    return np.eye(num_classes, dtype=float)[labels]


@dataclass
class RegressionDataset:
    """Container for regression dataset splits."""

    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    name: str = ""
    n_samples: int = 1000

    def __iter__(self) -> Iterator[FloatArray]:
        """Supports 4-tuple unpacking: x_tr, y_tr, x_te, y_te = dataset."""
        return iter((self.x_train, self.y_train, self.x_test, self.y_test))


@dataclass
class ClassificationDataset:
    """Container for classification dataset splits and discrete integer class labels."""

    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    train_labels: NDArray[np.int64]
    test_labels: NDArray[np.int64]
    num_classes: int = 2
    name: str = ""
    n_samples: int = 1000

    def __iter__(self) -> Iterator[FloatArray]:
        """Supports 4-tuple unpacking: x_tr, y_tr, x_te, y_te = dataset."""
        return iter((self.x_train, self.y_train, self.x_test, self.y_test))


def _resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    """Locates the 'data/p1' root directory."""
    if data_dir is not None:
        path = Path(data_dir)
        if path.is_dir():
            return path
        raise FileNotFoundError(f"Provided data directory does not exist: {data_dir}")

    candidate_locations = [
        Path(__file__).resolve().parents[2] / "data" / "p1",
        Path.cwd() / "data" / "p1",
        Path.cwd().parent / "data" / "p1",
    ]

    for candidate in candidate_locations:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find 'data/p1' directory. Please specify data_dir explicitly."
    )


def load_regression_data(
    name: str,
    n_samples: int = 1000,
    data_dir: Path | str | None = None,
) -> RegressionDataset:
    """Loads a regression dataset benchmark from data/p1/regression.

    Args:
        name: Name of the dataset, either 'activation' or 'cube' (or 'data.activation', etc.).
        n_samples: Dataset size variant: 100, 500, 1000, or 10000.
        data_dir: Optional explicit path to the 'data/p1' folder.

    Returns:
        A RegressionDataset dataclass instance containing raw training and test splits.
    """
    clean_name = name.removeprefix("data.")
    if clean_name not in {"activation", "cube"}:
        raise ValueError(
            f"Unknown regression dataset '{name}'. Expected 'activation' or 'cube'."
        )

    valid_sizes = {100, 500, 1000, 10000}
    if n_samples not in valid_sizes:
        raise ValueError(f"Invalid n_samples {n_samples}. Expected one of {sorted(valid_sizes)}.")

    root_dir = _resolve_data_dir(data_dir) / "regression"
    train_file = root_dir / f"data.{clean_name}.train.{n_samples}.csv"
    test_file = root_dir / f"data.{clean_name}.test.{n_samples}.csv"

    if not train_file.is_file():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if not test_file.is_file():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    train_raw = np.loadtxt(train_file, delimiter=",", skiprows=1, dtype=float)
    test_raw = np.loadtxt(test_file, delimiter=",", skiprows=1, dtype=float)

    # 1D input x and 1D target y formatted as (N, 1) column vectors
    x_train = train_raw[:, 0:1].copy()
    y_train = train_raw[:, 1:2].copy()
    x_test = test_raw[:, 0:1].copy()
    y_test = test_raw[:, 1:2].copy()

    return RegressionDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        name=clean_name,
        n_samples=n_samples,
    )


def load_classification_data(
    name: str,
    n_samples: int = 1000,
    target_encoding: Literal["auto", "binary", "one_hot", "indices"] = "auto",
    data_dir: Path | str | None = None,
) -> ClassificationDataset:
    """Loads a classification dataset benchmark from data/p1/classification.

    Args:
        name: Name of the dataset, either 'simple' or 'three_gauss' (or 'data.simple', etc.).
        n_samples: Dataset size variant: 100, 500, 1000, or 10000.
        target_encoding: Format for y_train and y_test:
            - 'auto': (N, 1) binary {0.0, 1.0} for 'simple'; (N, 3) one-hot for 'three_gauss'.
            - 'binary': (N, 1) binary float targets (only valid for 2-class problems).
            - 'one_hot': (N, C) one-hot encoded matrix.
            - 'indices': (N, 1) integer class index column vector.
        data_dir: Optional explicit path to the 'data/p1' folder.

    Returns:
        A ClassificationDataset dataclass instance containing splits and discrete class labels.
    """
    clean_name = name.removeprefix("data.")
    if clean_name not in {"simple", "three_gauss"}:
        raise ValueError(
            f"Unknown classification dataset '{name}'. Expected 'simple' or 'three_gauss'."
        )

    valid_sizes = {100, 500, 1000, 10000}
    if n_samples not in valid_sizes:
        raise ValueError(f"Invalid n_samples {n_samples}. Expected one of {sorted(valid_sizes)}.")

    root_dir = _resolve_data_dir(data_dir) / "classification"
    train_file = root_dir / f"data.{clean_name}.train.{n_samples}.csv"
    test_file = root_dir / f"data.{clean_name}.test.{n_samples}.csv"

    if not train_file.is_file():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    if not test_file.is_file():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    train_raw = np.loadtxt(train_file, delimiter=",", skiprows=1, dtype=float)
    test_raw = np.loadtxt(test_file, delimiter=",", skiprows=1, dtype=float)

    # 2D features
    x_train = train_raw[:, :2].copy()
    x_test = test_raw[:, :2].copy()

    # Raw class column (1-indexed in source CSV files: {1, 2} or {1, 2, 3})
    train_raw_cls = train_raw[:, 2].astype(int)
    test_raw_cls = test_raw[:, 2].astype(int)

    # Convert to 0-indexed discrete integer class labels {0, 1} or {0, 1, 2}
    train_labels = (train_raw_cls - 1).astype(np.int64)
    test_labels = (test_raw_cls - 1).astype(np.int64)

    num_classes = 2 if clean_name == "simple" else 3

    # Format output targets according to target_encoding
    if target_encoding == "auto":
        if clean_name == "simple":
            y_train = train_labels.reshape(-1, 1).astype(float)
            y_test = test_labels.reshape(-1, 1).astype(float)
        else:
            y_train = to_one_hot(train_labels, num_classes=num_classes)
            y_test = to_one_hot(test_labels, num_classes=num_classes)
    elif target_encoding == "binary":
        if num_classes != 2:
            raise ValueError(
                f"target_encoding='binary' is only valid for 2-class problems, but '{name}' has {num_classes} classes."
            )
        y_train = train_labels.reshape(-1, 1).astype(float)
        y_test = test_labels.reshape(-1, 1).astype(float)
    elif target_encoding == "one_hot":
        y_train = to_one_hot(train_labels, num_classes=num_classes)
        y_test = to_one_hot(test_labels, num_classes=num_classes)
    elif target_encoding == "indices":
        y_train = train_labels.reshape(-1, 1)
        y_test = test_labels.reshape(-1, 1)
    else:
        raise ValueError(
            f"Invalid target_encoding '{target_encoding}'. Expected 'auto', 'binary', 'one_hot', or 'indices'."
        )

    return ClassificationDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_labels=train_labels,
        test_labels=test_labels,
        num_classes=num_classes,
        name=clean_name,
        n_samples=n_samples,
    )
