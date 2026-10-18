"""Visualization utilities for MLP classification and regression experiments.

This module provides plotting functions adhering to project requirements:
- Code and public APIs in English.
- Generated plot artifacts (titles, axis labels, legends) in Polish.
- Clean and readable presentation of datasets and model evaluation.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Support running directly as a script as well as an imported module
if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.common.data_loader import ClassificationDataset, load_classification_data
else:
    from .data_loader import ClassificationDataset, load_classification_data


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

