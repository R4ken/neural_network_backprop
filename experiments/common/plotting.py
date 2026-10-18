"""Visualization utilities for MLP classification and regression experiments.

This module provides plotting functions adhering to project requirements:
- Code and public APIs in English.
- Generated plot artifacts (titles, axis labels, legends) in Polish.
- Clean and readable presentation of datasets and model evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

# Support running directly as a script as well as an imported module
if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.common.data_loader import ClassificationDataset, load_classification_data
    from src.py_nn1 import NeuronTrajectory
else:
    from .data_loader import ClassificationDataset, load_classification_data
    from src.py_nn1 import NeuronTrajectory


def _resolve_classification_results_dir() -> Path:
    """Locates or creates the default results directory for classification."""
    results_dir = Path(__file__).resolve().parents[1] / "results" / "classification"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_classification_dataset(
    dataset: ClassificationDataset,
    save_path: Path | str | None = None,
    alpha: float = 0.7,
    point_size: float = 25.0,
) -> Path:
    """Plots a 2D classification dataset in a side-by-side (1x2) train and test layout.

    Args:
        dataset: ClassificationDataset instance containing splits and class labels.
        save_path: Destination path for saving the figure. If None, defaults to
            'experiments/results/classification/dataset_{name}_{n_samples}.png'.
        alpha: Transparency level of points for visualizing density.
        point_size: Scatter point size (s).

    Returns:
        Path pointing to the saved figure file.
    """
    if save_path is None:
        filename = f"dataset_{dataset.name}_{dataset.n_samples}.png"
        target_path = _resolve_classification_results_dir() / filename
    else:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Use standard high-contrast palette
    colors = plt.cm.tab10.colors

    splits = [
        (ax_train, dataset.x_train, dataset.train_labels, f"Zbiór uczący (N = {len(dataset.x_train)})"),
        (ax_test, dataset.x_test, dataset.test_labels, f"Zbiór testowy (N = {len(dataset.x_test)})"),
    ]

    for ax, x, labels, split_title in splits:
        for cls_idx in range(dataset.num_classes):
            mask = labels == cls_idx
            ax.scatter(
                x[mask, 0],
                x[mask, 1],
                color=colors[cls_idx % len(colors)],
                label=f"Klasa {cls_idx}",
                alpha=alpha,
                s=point_size,
                edgecolors="none",
            )

        ax.set_title(split_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Cecha $x_1$", fontsize=11)
        ax.set_ylabel("Cecha $x_2$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Klasy", loc="best")

    fig.suptitle(
        f"Zbiór danych klasyfikacji: {dataset.name} (N = {dataset.n_samples})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return target_path


def plot_decision_boundaries_grid(
    models: dict[str, Any],
    dataset: ClassificationDataset,
    save_path: Path | str,
    is_binary: bool,
    point_size: float = 20.0,
    alpha: float = 0.8,
) -> Path:
    """Plots decision boundaries for multiple models side-by-side on a 2D grid.

    Args:
        models: Dictionary mapping activation name to a trained model instance.
        dataset: ClassificationDataset instance.
        save_path: Destination path for the saved image.
        is_binary: True if binary classification, False for multi-class.
        point_size: Scatter point size for test samples.
        alpha: Alpha transparency for points.

    Returns:
        Path to the saved figure.
    """
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    x_all = np.vstack([dataset.x_train, dataset.x_test])
    pad1 = (x_all[:, 0].max() - x_all[:, 0].min()) * 0.08
    pad2 = (x_all[:, 1].max() - x_all[:, 1].min()) * 0.08
    x1_min, x1_max = x_all[:, 0].min() - pad1, x_all[:, 0].max() + pad1
    x2_min, x2_max = x_all[:, 1].min() - pad2, x_all[:, 1].max() + pad2

    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 250), np.linspace(x2_min, x2_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    class_colors = plt.cm.tab10.colors

    for ax, (act_name, model) in zip(axes, models.items()):
        preds = model.predict(grid)

        if is_binary:
            probs = preds.reshape(xx.shape)
            ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.55)
            cs = ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.8, linestyles="--")
            ax.clabel(cs, inline=True, fontsize=9, fmt="p=0.5")
        else:
            class_map = np.argmax(preds, axis=1).reshape(xx.shape)
            ax.contourf(
                xx,
                yy,
                class_map,
                levels=np.arange(-0.5, dataset.num_classes + 0.5, 1),
                cmap="Pastel1",
                alpha=0.6,
            )

        for cls_idx in range(dataset.num_classes):
            mask = dataset.test_labels == cls_idx
            ax.scatter(
                dataset.x_test[mask, 0],
                dataset.x_test[mask, 1],
                color=class_colors[cls_idx % len(class_colors)],
                label=f"Klasa {cls_idx}",
                edgecolors="black",
                linewidths=0.5,
                s=point_size,
                alpha=alpha,
            )

        ax.set_title(f"Aktywacja: {act_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cecha $x_1$", fontsize=11)
        ax.set_ylabel("Cecha $x_2$", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(title="Klasy", loc="lower right", fontsize=9)

    fig.suptitle(
        f"Granice decyzyjne w przestrzeni cech 2D: {dataset.name} (Zbiór testowy, N = {len(dataset.x_test)})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target_path


def plot_learning_curves_with_std(
    epochs: list[int] | np.ndarray,
    results_by_act: dict[str, dict[str, list[list[float]]]],
    metric_label: str,
    title: str,
    save_path: Path | str,
) -> Path:
    """Plots training and test learning curves side-by-side with std dev bands across seeds.

    Args:
        epochs: Sequence of epoch numbers (1 to E).
        results_by_act: Dict of {act_name: {"train": [runs], "test": [runs]}}.
        metric_label: Label for the metric on the Y-axis (in Polish).
        title: Main title of the figure (in Polish).
        save_path: Destination path for the saved image.

    Returns:
        Path to the saved figure.
    """
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    palette = {"Sigmoid": "#1f77b4", "Tanh": "#2ca02c", "ReLU": "#d62728"}
    splits = [
        (ax_train, "train", "Zbiór uczący"),
        (ax_test, "test", "Zbiór testowy"),
    ]

    for ax, split_key, split_title in splits:
        for act_name, data in results_by_act.items():
            runs = np.array(data[split_key], dtype=float)
            mean_vals = np.mean(runs, axis=0)
            std_vals = np.std(runs, axis=0)
            color = palette.get(act_name, None)

            ax.plot(epochs, mean_vals, label=act_name, color=color, linewidth=2)
            ax.fill_between(
                epochs,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.18,
            )

        ax.set_title(split_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoka", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Funkcja aktywacji", loc="best", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target_path


def plot_weight_norms_with_std(
    epochs: list[int] | np.ndarray,
    norms_by_act: dict[str, list[list[list[float]]]],
    layer_names: list[str],
    title: str,
    save_path: Path | str,
) -> Path:
    """Plots weight norms for each layer across epochs with std dev bands across seeds.

    Args:
        epochs: Sequence of epoch numbers (1 to E).
        norms_by_act: Dict of {act_name: [seed_runs]}, where each seed_run is
            list of epochs, and each epoch is list of norms per Dense layer.
        layer_names: Descriptive Polish names for each Dense layer.
        title: Main title of the figure (in Polish).
        save_path: Destination path for the saved image.

    Returns:
        Path to the saved figure.
    """
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    n_layers = len(layer_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(5.5 * n_layers, 5), sharex=True)
    if n_layers == 1:
        axes = [axes]

    palette = {"Sigmoid": "#1f77b4", "Tanh": "#2ca02c", "ReLU": "#d62728"}

    for layer_idx, (ax, l_name) in enumerate(zip(axes, layer_names)):
        for act_name, runs in norms_by_act.items():
            arr = np.array(runs, dtype=float)
            layer_arr = arr[:, :, layer_idx]
            mean_vals = np.mean(layer_arr, axis=0)
            std_vals = np.std(layer_arr, axis=0)
            color = palette.get(act_name, None)

            ax.plot(epochs, mean_vals, label=act_name, color=color, linewidth=2)
            ax.fill_between(
                epochs,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.18,
            )

        ax.set_title(l_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoka", fontsize=11)
        ax.set_ylabel(r"Norma Frobeniusa $\|W\|_F$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Funkcja aktywacji", loc="best", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
def plot_neuron_weights_trajectory(
    epochs: list[int] | np.ndarray,
    trajectories_by_act: dict[str, NeuronTrajectory],
    layer_names: list[str],
    title: str,
    save_path: Path | str,
) -> Path:
    """Plots weights and biases of the first neuron for each layer across epochs for a single representative seed.

    Args:
        epochs: Sequence of epoch numbers (1 to E).
        trajectories_by_act: Dict mapping activation name to a single seed run.
            The run contains epochs, each epoch contains list of layer dicts {"w": ..., "b": ...}.
        layer_names: Descriptive Polish names for each layer.
        title: Main title of the figure (in Polish).
        save_path: Destination path for saving the figure.

    Returns:
        Path to the saved figure file.
    """
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    n_layers = len(layer_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(6.0 * n_layers, 5.2), sharex=True)
    if n_layers == 1:
        axes = [axes]

    palette = {"Sigmoid": "#1f77b4", "Tanh": "#2ca02c", "ReLU": "#d62728"}

    for layer_idx, (ax, l_name) in enumerate(zip(axes, layer_names)):
        for act_name, epoch_list in trajectories_by_act.items():
            w_vals = [epoch_layers[layer_idx].get("w", 0.0) for epoch_layers in epoch_list]
            b_vals = [epoch_layers[layer_idx].get("b", 0.0) for epoch_layers in epoch_list]
            color = palette.get(act_name, None)

            # Weight trajectory (solid line)
            ax.plot(
                epochs,
                w_vals,
                label=f"{act_name} (waga $w_{{0,0}}$)",
                color=color,
                linestyle="-",
                linewidth=2.0,
            )

            # Bias trajectory (dashed line)
            ax.plot(
                epochs,
                b_vals,
                label=f"{act_name} (bias $b_0$)",
                color=color,
                linestyle="--",
                linewidth=1.8,
            )

        ax.set_title(l_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoka", fontsize=11)
        ax.set_ylabel("Wartość parametru (Waga / Bias)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target_path


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Generowanie przykladowych wykresow zbiorow danych klasyfikacji...")
    for dataset_name in ("simple", "three_gauss"):
        for n in (100, 1000):
            loaded_ds = load_classification_data(name=dataset_name, n_samples=n)
            output_file = plot_classification_dataset(loaded_ds)
            print(f"-> Zapisano wykres dla '{dataset_name}': {output_file}")

