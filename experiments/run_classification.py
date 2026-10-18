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
    plot_architecture_accuracy_bars,
    plot_decision_boundaries_grid,
    plot_deep_model_weight_norms,
    plot_learning_curves_with_std,
    plot_multimodel_learning_curves_with_std,
    plot_neuron_weights_comparison_1x3,
    plot_neuron_weights_trajectory,
    plot_weight_norms_with_std,
)
from src.py_nn1 import (
    MLP,
    SGD,
    BinaryCrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    NeuronTrajectory,
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


def run_experiment_k1(seeds: list[int] | None = None, n_samples: int = 100) -> None:
    """Runs Experiment K1: Impact of activation functions in hidden layers.

    Evaluates Sigmoid, Tanh, and ReLU on data.simple (binary) and data.three_gauss
    (multiclass), repeating across seeds to report mean and standard deviation.

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
        n_samples: Number of samples in dataset. Defaults to 100 so that results are intentionally worse.
    """
    if seeds is None:
        seeds = [42, 123, 999]

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
        rep_neuron_trajectories: dict[str, NeuronTrajectory] = {}
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

                # Store the model and neuron trajectories for the primary seed (42)
                if seed == seeds[0]:
                    rep_models[act_name] = model
                    rep_neuron_trajectories[act_name] = history["tracked_neuron_weights"]

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

        # Plot 5: First neuron weights evolution across layers (representative seed)
        neuron_plot_path = results_dir / f"k1_{ds_name}_neuron_weights.png"
        plot_neuron_weights_trajectory(
            epochs=epochs_range,
            trajectories_by_act=rep_neuron_trajectories,
            layer_names=layer_names,
            title=f"Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw (Sigmoid vs ReLU vs Tanh): {ds_name} (Ziarno: {seeds[0]})",
            save_path=neuron_plot_path,
        )
        print(f"   [Wykres] Zapisano ewolucję wag pierwszego neuronu: {neuron_plot_path.name}")

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


def run_experiment_k2(
    seeds: list[int] | None = None,
    n_samples: int = 100,
    epochs: int = 250,
    lr: float = 0.01,
    activation_cls: type = ReLU,
    init_method: str = "he",
) -> None:
    """Runs Experiment K2: Impact of network depth and layer widths.

    Evaluates 8 architectures (from 0 to 4 hidden layers) using ReLU on
    data.simple (binary) and data.three_gauss (multiclass).

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
        n_samples: Number of samples in dataset. Defaults to 100.
        epochs: Number of training epochs. Defaults to 250.
        lr: Learning rate for SGD optimizer. Defaults to 0.01.
        activation_cls: Activation class for hidden layers. Defaults to ReLU.
        init_method: Weight initialization strategy. Defaults to "he".
    """
    if seeds is None:
        seeds = [42, 123, 999]

    results_dir = _get_results_dir()
    act_name = activation_cls.__name__

    print("=" * 70)
    print("EKSPERYMENT K2: Wplyw liczby warstw ukrytych oraz ich licznosci")
    print(f"Aktywacja ukryta: {act_name} ({init_method}) | Epoki: {epochs} | LR: {lr} | Ziarna: {seeds}")
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

    arch_configs = [
        {
            "key": "0_hidden",
            "name": "0 ukrytych: [2]",
            "num_hidden": 0,
            "hidden_dims": [],
        },
        {
            "key": "1_hidden_narrow",
            "name": "1 ukryta (4): [2, 4]",
            "num_hidden": 1,
            "hidden_dims": [4],
        },
        {
            "key": "1_hidden_medium",
            "name": "1 ukryta (16): [2, 16]",
            "num_hidden": 1,
            "hidden_dims": [16],
        },
        {
            "key": "1_hidden_wide",
            "name": "1 ukryta (64): [2, 64]",
            "num_hidden": 1,
            "hidden_dims": [64],
        },
        {
            "key": "2_hidden_a",
            "name": "2 ukryte (A): [2, 16, 8]",
            "num_hidden": 2,
            "hidden_dims": [16, 8],
        },
        {
            "key": "2_hidden_b",
            "name": "2 ukryte (B): [2, 32, 16]",
            "num_hidden": 2,
            "hidden_dims": [32, 16],
        },
        {
            "key": "3_hidden",
            "name": "3 ukryte: [2, 32, 16, 8]",
            "num_hidden": 3,
            "hidden_dims": [32, 16, 8],
        },
        {
            "key": "4_hidden",
            "name": "4 ukryte: [2, 64, 32, 16, 8]",
            "num_hidden": 4,
            "hidden_dims": [64, 32, 16, 8],
        },
    ]

    # Target architectures for the 1x3 neuron comparison plot
    neuron_comp_keys = ["1_hidden_medium", "2_hidden_b", "4_hidden"]

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

        loss_curves: dict[str, dict[str, list[list[float]]]] = {}
        acc_curves: dict[str, dict[str, list[list[float]]]] = {}
        rep_models: dict[str, MLP] = {}
        rep_neuron_trajectories: dict[str, NeuronTrajectory] = {}

        deep_norm_runs: list[list[list[float]]] = []
        deep_layer_names = [
            "Warstwa 1 (2 -> 64)",
            "Warstwa 2 (64 -> 32)",
            "Warstwa 3 (32 -> 16)",
            "Warstwa 4 (16 -> 8)",
            f"Warstwa wyjściowa (8 -> {output_dim})",
        ]

        train_acc_means: list[float] = []
        train_acc_stds: list[float] = []
        test_acc_means: list[float] = []
        test_acc_stds: list[float] = []
        arch_labels: list[str] = []

        for arch in arch_configs:
            arch_key = arch["key"]
            arch_name = arch["name"]
            num_hidden = arch["num_hidden"]
            hidden_dims = arch["hidden_dims"]
            layer_dims = [2] + hidden_dims + [output_dim]
            arch_labels.append(arch_name)

            loss_curves[arch_name] = {"train": [], "test": []}
            acc_curves[arch_name] = {"train": [], "test": []}

            final_train_accs: list[float] = []
            final_test_accs: list[float] = []
            final_train_losses: list[float] = []
            final_test_losses: list[float] = []

            for seed in seeds:
                set_seed(seed)
                model = MLP.from_dimensions(
                    layer_dims=layer_dims,
                    hidden_activation=activation_cls,
                    output_activation=output_act,
                    use_bias=True,
                    weight_init=init_method,
                )
                optimizer = SGD(learning_rate=lr)
                loss_fn = loss_fn_cls()

                history = model.fit(
                    x_train=dataset.x_train,
                    y_train=dataset.y_train,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(dataset.x_test, dataset.y_test),
                    metrics={"acc": acc_metric},
                    verbose=False,
                )

                loss_curves[arch_name]["train"].append(history["loss"])
                loss_curves[arch_name]["test"].append(history["val_loss"])
                acc_curves[arch_name]["train"].append(history["acc"])
                acc_curves[arch_name]["test"].append(history["val_acc"])

                if arch_key == "4_hidden":
                    deep_norm_runs.append(history["weight_norms"])

                final_train_losses.append(history["loss"][-1])
                final_test_losses.append(history["val_loss"][-1])
                final_train_accs.append(history["acc"][-1])
                final_test_accs.append(history["val_acc"][-1])

                if seed == seeds[0]:
                    rep_models[arch_name] = model
                    if arch_key in neuron_comp_keys:
                        rep_neuron_trajectories[arch_name] = history["tracked_neuron_weights"]

            mean_tr_acc = float(np.mean(final_train_accs))
            std_tr_acc = float(np.std(final_train_accs))
            mean_te_acc = float(np.mean(final_test_accs))
            std_te_acc = float(np.std(final_test_accs))
            mean_tr_loss = float(np.mean(final_train_losses))
            std_tr_loss = float(np.std(final_train_losses))
            mean_te_loss = float(np.mean(final_test_losses))
            std_te_loss = float(np.std(final_test_losses))

            train_acc_means.append(mean_tr_acc)
            train_acc_stds.append(std_tr_acc)
            test_acc_means.append(mean_te_acc)
            test_acc_stds.append(std_te_acc)

            summary_rows.append({
                "dataset": ds_name,
                "architecture": arch_name,
                "num_hidden_layers": num_hidden,
                "test_acc_mean": mean_te_acc,
                "test_acc_std": std_te_acc,
                "train_acc_mean": mean_tr_acc,
                "train_acc_std": std_tr_acc,
                "test_loss_mean": mean_te_loss,
                "test_loss_std": std_te_loss,
                "train_loss_mean": mean_tr_loss,
                "train_loss_std": std_tr_loss,
            })

            print(
                f"   * [{arch_name:<26}] Dokladnosc testowa: "
                f"{mean_te_acc * 100:6.2f}% (+/- {std_te_acc * 100:4.2f}%) | "
                f"Strata testowa: {mean_te_loss:.4f}"
            )

        epochs_range = np.arange(1, epochs + 1)

        # Plot 1: Decision boundaries grid 2x4 for the representative seed
        boundary_path = results_dir / f"k2_{ds_name}_decision_boundaries.png"
        plot_decision_boundaries_grid(
            models=rep_models,
            dataset=dataset,
            save_path=boundary_path,
            is_binary=is_binary,
            nrows=2,
            ncols=4,
            title=f"Granice decyzyjne w zależności od architektury: {ds_name} (Zbiór testowy, N = {len(dataset.x_test)})",
        )
        print(f"   [Wykres 1/6] Zapisano granice decyzyjne: {boundary_path.name}")

        # Plot 2: Grouped accuracy bar chart (Train vs Test) with error bars
        acc_bar_path = results_dir / f"k2_{ds_name}_accuracy_bars.png"
        plot_architecture_accuracy_bars(
            arch_labels=arch_labels,
            train_accs=train_acc_means,
            train_stds=train_acc_stds,
            test_accs=test_acc_means,
            test_stds=test_acc_stds,
            title=f"Dokładność klasyfikacji na zbiorze uczącym i testowym w zależności od architektury: {ds_name}",
            save_path=acc_bar_path,
        )
        print(f"   [Wykres 2/6] Zapisano słupki dokładności: {acc_bar_path.name}")

        # Plot 3: Loss curves side-by-side (Train vs Test) with std dev bands
        loss_plot_path = results_dir / f"k2_{ds_name}_loss_curves.png"
        plot_multimodel_learning_curves_with_std(
            epochs=epochs_range,
            results_by_model=loss_curves,
            metric_label="Wartość funkcji straty (Loss)",
            title=f"Krzywe straty w funkcji epok dla badanych architektur: {ds_name} (Średnia z {len(seeds)} ziaren)",
            save_path=loss_plot_path,
        )
        print(f"   [Wykres 3/6] Zapisano krzywe straty: {loss_plot_path.name}")

        # Plot 4: Accuracy curves side-by-side (Train vs Test) with std dev bands
        acc_plot_path = results_dir / f"k2_{ds_name}_accuracy_curves.png"
        plot_multimodel_learning_curves_with_std(
            epochs=epochs_range,
            results_by_model=acc_curves,
            metric_label="Dokładność klasyfikacji (Accuracy) [%]",
            title=f"Dokładność klasyfikacji w kolejnych epokach: {ds_name} (Średnia z {len(seeds)} ziaren)",
            save_path=acc_plot_path,
            as_percentage=True,
        )
        print(f"   [Wykres 4/6] Zapisano krzywe dokładności: {acc_plot_path.name}")

        # Plot 5: Weight Frobenius norms evolution across all Dense layers in the 4-layer network
        norm_plot_path = results_dir / f"k2_{ds_name}_weight_norms_deep.png"
        plot_deep_model_weight_norms(
            epochs=epochs_range,
            seed_runs=deep_norm_runs,
            layer_names=deep_layer_names,
            title=f"Ewolucja norm wag poszczególnych warstw w sieci 4-warstwowej: {ds_name} (Średnia z {len(seeds)} ziaren)",
            save_path=norm_plot_path,
        )
        print(f"   [Wykres 5/6] Zapisano normy wag sieci 4-warstwowej: {norm_plot_path.name}")

        # Plot 6: First neuron comparison (1x3) across shallow, medium, deep architectures
        neuron_plot_path = results_dir / f"k2_{ds_name}_neuron_weights.png"
        plot_neuron_weights_comparison_1x3(
            epochs=epochs_range,
            trajectories_by_model=rep_neuron_trajectories,
            title=f"Ewolucja wagi i biasu pierwszego neuronu w pierwszej warstwie ukrytej: {ds_name} (Ziarno: {seeds[0]})",
            save_path=neuron_plot_path,
        )
        print(f"   [Wykres 6/6] Zapisano porównanie neuronu w warstwie 1 (1x3): {neuron_plot_path.name}")

    # Save summary metrics to CSV
    csv_path = results_dir / "k2_summary_metrics.csv"
    fieldnames = [
        "dataset",
        "architecture",
        "num_hidden_layers",
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

    print(f"\n[Podsumowanie K2] Zapisano tabelę metryk: {csv_path.name}")
    print("=" * 70)


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
        help="Numer eksperymentu do uruchomienia: '1' (K1), '2' (K2), '3' (K3) lub 'all'",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Liczba próbek w zbiorze danych (domyślnie: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Liczba epok treningu dla K2 (domyślnie: 250)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Współczynnik uczenia learning rate dla K2 (domyślnie: 0.01)",
    )

    args = parser.parse_args()

    if args.experiment == "1":
        run_experiment_k1(n_samples=args.n_samples)
    elif args.experiment == "2":
        run_experiment_k2(n_samples=args.n_samples, epochs=args.epochs, lr=args.lr)
    elif args.experiment == "3":
        run_experiment_k3()
    elif args.experiment == "all":
        run_experiment_k1(n_samples=args.n_samples)
        run_experiment_k2(n_samples=args.n_samples, epochs=args.epochs, lr=args.lr)
        run_experiment_k3()


if __name__ == "__main__":
    main()

