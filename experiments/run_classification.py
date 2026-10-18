"""Classification experiments runner for MLP benchmarks.

Executes experiments defined in docs/experiments.md:
- K1: Impact of hidden layer activation functions (Sigmoid vs Tanh vs ReLU)
- K2: Impact of network depth and layer widths (0 to 4 hidden layers)
- K3: Impact of loss function (Cross-Entropy vs MSE)

Usage:
    python experiments/run_classification.py 1
    python experiments/run_classification.py 2
    python experiments/run_classification.py 3
    python experiments/run_classification.py all
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.common.data_loader import ClassificationDataset, load_classification_data
from experiments.common.plotting import (
    plot_decision_boundaries_grid,
    plot_learning_curves_with_std,
    plot_weight_norms_with_std,
)
from src.py_nn1 import (
    MLP,
    SGD,
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    compute_multiclass_accuracy,
    compute_multilabel_accuracy,
    set_seed,
)


def _get_results_dir() -> Path:
    """Returns and ensures the results directory for classification experiments."""
    results_dir = PROJECT_ROOT / "experiments" / "results" / "classification"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_experiment_k1(seeds: list[int] | None = None) -> None:
    """Runs Experiment K1: Impact of activation functions in hidden layers.

    Evaluates Sigmoid, Tanh, and ReLU on data.simple (binary) and data.three_gauss
    (multiclass), repeating across seeds to report mean and standard deviation.

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
    """
    if seeds is None:
        seeds = [42, 123, 999]

    n_samples = 100

    results_dir = _get_results_dir()
    print("=" * 70)
    print("EKSPERYMENT K1: Wplyw funkcji aktywacji w warstwach ukrytych")
    print(f"Badane aktywacje: Sigmoid, Tanh, ReLU | Ziarna losowe: {seeds}")
    print("=" * 70)

    dataset_configs = [
        {
            "name": "simple",
            "is_binary": True,
            "output_dim": 1,
            "output_act": Sigmoid,
            "loss_fn_cls": BinaryCrossEntropyLoss,
            "acc_metric": compute_multilabel_accuracy,
        },
        {
            "name": "three_gauss",
            "is_binary": False,
            "output_dim": 3,
            "output_act": Softmax,
            "loss_fn_cls": CategoricalCrossEntropyLoss,
            "acc_metric": compute_multiclass_accuracy,
        },
    ]

    activation_configs = [
        ("Sigmoid", Sigmoid, "xavier"),
        ("Tanh", Tanh, "xavier"),
        ("ReLU", ReLU, "he"),
    ]

    summary_rows: list[dict[str, Any]] = []

    for ds_conf in dataset_configs:
        ds_name = ds_conf["name"]
        is_binary = ds_conf["is_binary"]
        output_dim = ds_conf["output_dim"]
        output_act = ds_conf["output_act"]
        loss_fn_cls = ds_conf["loss_fn_cls"]
        acc_metric = ds_conf["acc_metric"]

        print(f"\n--> Przetwarzanie zbioru: {ds_name} (N = {n_samples})...")
        dataset: ClassificationDataset = load_classification_data(ds_name, n_samples=n_samples)

        layer_dims = [2, 32, 16, output_dim]
        layer_names = [
            "Warstwa 1 (2 -> 32)",
            "Warstwa 2 (32 -> 16)",
            f"Warstwa wyjsciowa (16 -> {output_dim})",
        ]

        # Containers for aggregated curves across seeds
        loss_curves: dict[str, dict[str, list[list[float]]]] = {}
        acc_curves: dict[str, dict[str, list[list[float]]]] = {}
        norm_curves: dict[str, list[list[list[float]]]] = {}
        rep_models: dict[str, MLP] = {}

        for act_name, act_cls, init_method in activation_configs:
            loss_curves[act_name] = {"train": [], "test": []}
            acc_curves[act_name] = {"train": [], "test": []}
            norm_curves[act_name] = []

            final_train_accs: list[float] = []
            final_test_accs: list[float] = []
            final_train_losses: list[float] = []
            final_test_losses: list[float] = []

            for seed in seeds:
                set_seed(seed)
                model = MLP.from_dimensions(
                    layer_dims=layer_dims,
                    hidden_activation=act_cls,
                    output_activation=output_act,
                    use_bias=True,
                    weight_init=init_method,
                )
                optimizer = SGD(learning_rate=0.05)
                loss_fn = loss_fn_cls()

                history = model.fit(
                    x_train=dataset.x_train,
                    y_train=dataset.y_train,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=150,
                    batch_size=32,
                    validation_data=(dataset.x_test, dataset.y_test),
                    metrics={"acc": acc_metric},
                    verbose=False,
                )

                loss_curves[act_name]["train"].append(history["loss"])
                loss_curves[act_name]["test"].append(history["val_loss"])
                acc_curves[act_name]["train"].append(history["acc"])
                acc_curves[act_name]["test"].append(history["val_acc"])
                norm_curves[act_name].append(history["weight_norms"])

                final_train_losses.append(history["loss"][-1])
                final_test_losses.append(history["val_loss"][-1])
                final_train_accs.append(history["acc"][-1])
                final_test_accs.append(history["val_acc"][-1])

                # Store the model trained on the primary seed (42) for boundary visualization
                if seed == seeds[0]:
                    rep_models[act_name] = model

            summary_rows.append({
                "dataset": ds_name,
                "activation": act_name,
                "test_acc_mean": float(np.mean(final_test_accs)),
                "test_acc_std": float(np.std(final_test_accs)),
                "train_acc_mean": float(np.mean(final_train_accs)),
                "train_acc_std": float(np.std(final_train_accs)),
                "test_loss_mean": float(np.mean(final_test_losses)),
                "test_loss_std": float(np.std(final_test_losses)),
                "train_loss_mean": float(np.mean(final_train_losses)),
                "train_loss_std": float(np.std(final_train_losses)),
            })

            print(
                f"   * [{act_name:<7}] Dokladnosc testowa: "
                f"{np.mean(final_test_accs) * 100:6.2f}% (+/- {np.std(final_test_accs) * 100:4.2f}%) | "
                f"Strata testowa: {np.mean(final_test_losses):.4f}"
            )

        # Plot 1: Decision boundaries grid for the representative seed
        boundary_path = results_dir / f"k1_{ds_name}_decision_boundaries.png"
        plot_decision_boundaries_grid(
            models=rep_models,
            dataset=dataset,
            save_path=boundary_path,
            is_binary=is_binary,
        )
        print(f"   [Wykres] Zapisano granice decyzyjne: {boundary_path.name}")

        # Plot 2: Loss curves with std dev band
        epochs_range = np.arange(1, 151)
        loss_plot_path = results_dir / f"k1_{ds_name}_loss_curves.png"
        plot_learning_curves_with_std(
            epochs=epochs_range,
            results_by_act=loss_curves,
            metric_label="Wartość funkcji straty (Loss)",
            title=f"Krzywe straty w funkcji epok: {ds_name} (Średnia z {len(seeds)} ziaren)",
            save_path=loss_plot_path,
        )
        print(f"   [Wykres] Zapisano krzywe straty: {loss_plot_path.name}")

        # Plot 3: Accuracy curves with std dev band
        acc_plot_path = results_dir / f"k1_{ds_name}_accuracy_curves.png"
        plot_learning_curves_with_std(
            epochs=epochs_range,
            results_by_act=acc_curves,
            metric_label="Dokładność klasyfikacji (Accuracy)",
            title=f"Krzywe dokładności w funkcji epok: {ds_name} (Średnia z {len(seeds)} ziaren)",
            save_path=acc_plot_path,
        )
        print(f"   [Wykres] Zapisano krzywe dokładności: {acc_plot_path.name}")

        # Plot 4: Weight norms evolution across layers
        norm_plot_path = results_dir / f"k1_{ds_name}_weight_norms.png"
        plot_weight_norms_with_std(
            epochs=epochs_range,
            norms_by_act=norm_curves,
            layer_names=layer_names,
            title=f"Ewolucja norm Frobeniusa wag sieci: {ds_name} (Średnia z {len(seeds)} ziaren)",
            save_path=norm_plot_path,
        )
        print(f"   [Wykres] Zapisano ewolucję norm wag: {norm_plot_path.name}")

    # Save summary metrics to CSV
    csv_path = results_dir / "k1_summary_metrics.csv"
    fieldnames = [
        "dataset",
        "activation",
        "test_acc_mean",
        "test_acc_std",
        "train_acc_mean",
        "train_acc_std",
        "test_loss_mean",
        "test_loss_std",
        "train_loss_mean",
        "train_loss_std",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[Podsumowanie] Zapisano tabelę metryk: {csv_path.name}")
    print("=" * 70)


def run_experiment_k2() -> None:
    """Placeholder for Experiment K2: Impact of network depth and layer widths."""
    print("Eksperyment K2 nie jest jeszcze zaimplementowany.")


def run_experiment_k3() -> None:
    """Placeholder for Experiment K3: Impact of loss functions (Cross-Entropy vs MSE)."""
    print("Eksperyment K3 nie jest jeszcze zaimplementowany.")


def main() -> None:
    """CLI entry point parsing experiment target."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Runner eksperymentów klasyfikacji MLP (K1, K2, K3)",
    )
    parser.add_argument(
        "experiment",
        choices=["1", "2", "3", "all"],
        nargs="?",
        default="1",
        help="Numer eksperymentu do uruchomienia: '1' (K1), '2' (K2), '3' (K3) lub 'all' (domyślnie: '1')",
    )

    args = parser.parse_args()

    if args.experiment == "1":
        run_experiment_k1()
    elif args.experiment == "2":
        run_experiment_k2()
    elif args.experiment == "3":
        run_experiment_k3()
    elif args.experiment == "all":
        run_experiment_k1()
        run_experiment_k2()
        run_experiment_k3()


if __name__ == "__main__":
    main()

