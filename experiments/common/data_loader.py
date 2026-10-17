"""Data loading and preprocessing utilities for MLP regression and classification experiments.

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


class StandardScaler:
    """Z-score feature/target standardizer: (x - mean) / std.

    Calculates mean and standard deviation on the training set and uses those
    statistics to transform training, validation, and test sets, avoiding data leakage.
    Provides `inverse_transform` to convert predictions back to the original domain.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initializes the scaler.

        Args:
            eps: Epsilon added or used as lower bound to prevent division by zero.
        """
        self.eps: float = eps
        self.mean_: FloatArray | None = None
        self.std_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def fit(self, x: FloatArray) -> StandardScaler:
        """Computes the mean and standard deviation from the data.

        Args:
            x: Array of shape (num_samples, num_features) or (num_samples,).

        Returns:
            The fitted scaler instance.
        """
        self.mean_ = np.mean(x, axis=0, keepdims=True)
        self.std_ = np.std(x, axis=0, keepdims=True)
        self.scale_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: FloatArray) -> FloatArray:
        """Standardizes input data using the fitted statistics.

        Args:
            x: Array to transform.

        Returns:
            Standardized array.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: FloatArray) -> FloatArray:
        """Fits the scaler to data, then returns the standardized data.

        Args:
            x: Array to fit and transform.

        Returns:
            Standardized array.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x: FloatArray) -> FloatArray:
        """Scales the standardized data back to the original distribution.

        Args:
            x: Standardized array.

        Returns:
            Data transformed back to original scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling inverse_transform().")
        return x * self.scale_ + self.mean_


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
    """Container for regression dataset splits and corresponding scalers."""

    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    scaler_x: StandardScaler | None = None
    scaler_y: StandardScaler | None = None
    name: str = ""
    n_samples: int = 1000

    def __iter__(self) -> Iterator[FloatArray]:
        """Supports 4-tuple unpacking: x_tr, y_tr, x_te, y_te = dataset."""
        return iter((self.x_train, self.y_train, self.x_test, self.y_test))


@dataclass
class ClassificationDataset:
    """Container for classification dataset splits, labels and optional scalers."""

    x_train: FloatArray
    y_train: FloatArray
    x_test: FloatArray
    y_test: FloatArray
    y_train_labels: NDArray[np.int64]
    y_test_labels: NDArray[np.int64]
    num_classes: int = 2
    scaler_x: StandardScaler | None = None
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

    # Standard relative locations
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
    standardize: bool = True,
    data_dir: Path | str | None = None,
) -> RegressionDataset:
    """Loads a regression dataset benchmark from data/p1/regression.

    Args:
        name: Name of the dataset, either 'activation' or 'cube' (or 'data.activation', etc.).
        n_samples: Dataset size variant: 100, 500, 1000, or 10000.
        standardize: If True, fits StandardScaler on train set (both x and y) and applies it to test.
        data_dir: Optional explicit path to the 'data/p1' folder.

    Returns:
        A RegressionDataset dataclass instance containing training/test splits and scalers.
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

    scaler_x: StandardScaler | None = None
    scaler_y: StandardScaler | None = None

    if standardize:
        scaler_x = StandardScaler().fit(x_train)
        x_train = scaler_x.transform(x_train)
        x_test = scaler_x.transform(x_test)

        scaler_y = StandardScaler().fit(y_train)
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)

    return RegressionDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        name=clean_name,
        n_samples=n_samples,
    )


def load_classification_data(
    name: str,
    n_samples: int = 1000,
    target_encoding: Literal["auto", "binary", "one_hot", "indices"] = "auto",
    standardize: bool = False,
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
        standardize: If True, standardizes features x using statistics fitted on train set.
        data_dir: Optional explicit path to the 'data/p1' folder.

    Returns:
        A ClassificationDataset dataclass instance containing splits, discrete labels, and scalers.
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
    y_train_labels = (train_raw_cls - 1).astype(np.int64)
    y_test_labels = (test_raw_cls - 1).astype(np.int64)

    num_classes = 2 if clean_name == "simple" else 3

    # Format output targets according to target_encoding
    if target_encoding == "auto":
        if clean_name == "simple":
            y_train = y_train_labels.reshape(-1, 1).astype(float)
            y_test = y_test_labels.reshape(-1, 1).astype(float)
        else:
            y_train = to_one_hot(y_train_labels, num_classes=num_classes)
            y_test = to_one_hot(y_test_labels, num_classes=num_classes)
    elif target_encoding == "binary":
        if num_classes != 2:
            raise ValueError(
                f"target_encoding='binary' is only valid for 2-class problems, but '{name}' has {num_classes} classes."
            )
        y_train = y_train_labels.reshape(-1, 1).astype(float)
        y_test = y_test_labels.reshape(-1, 1).astype(float)
    elif target_encoding == "one_hot":
        y_train = to_one_hot(y_train_labels, num_classes=num_classes)
        y_test = to_one_hot(y_test_labels, num_classes=num_classes)
    elif target_encoding == "indices":
        y_train = y_train_labels.reshape(-1, 1)
        y_test = y_test_labels.reshape(-1, 1)
    else:
        raise ValueError(
            f"Invalid target_encoding '{target_encoding}'. Expected 'auto', 'binary', 'one_hot', or 'indices'."
        )

    scaler_x: StandardScaler | None = None
    if standardize:
        scaler_x = StandardScaler().fit(x_train)
        x_train = scaler_x.transform(x_train)
        x_test = scaler_x.transform(x_test)

    return ClassificationDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
        num_classes=num_classes,
        scaler_x=scaler_x,
        name=clean_name,
        n_samples=n_samples,
    )

