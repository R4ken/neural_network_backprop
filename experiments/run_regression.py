"""Regression experiments runner for MLP benchmarks.

Executes experiments defined in docs/experiments.md:
- R1: Impact of hidden layer activation functions (Sigmoid vs Tanh vs ReLU)
- R2: Impact of network depth and layer widths (0 to 4 hidden layers)
- R3: Impact of the loss function (MSE vs MAE)
- R4: Impact of the training set size (N = 100, 500, 1000, 10000)

Usage:
    python experiments/run_regression.py 1
    python experiments/run_regression.py 2
    python experiments/run_regression.py 3
    python experiments/run_regression.py 4
    python experiments/run_regression.py all
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

from experiments.common.data_loader import RegressionDataset, load_regression_data
from experiments.common.plotting import (
    plot_learning_curves_with_std,
    plot_neuron_weights_trajectory,
    plot_regression_approximations_grid,
    plot_weight_norms_with_std,
)
from src.py_nn1 import (
    MLP,
    SGD,
    MSELoss,
    NeuronTrajectory,
    ReLU,
    Sigmoid,
    Tanh,
    compute_mae,
    compute_r2,
    set_seed,
)


def _get_results_dir() -> Path:
    """Returns and ensures the results directory for regression experiments."""
    results_dir = PROJECT_ROOT / "experiments" / "results" / "regression"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_experiment_r1(seeds: list[int] | None = None, n_samples: int = 1000) -> None:
    """Runs Experiment R1: Impact of activation functions in hidden layers.

    Evaluates Sigmoid, Tanh, and ReLU on data.activation and data.cube, repeating
    across seeds to report mean and standard deviation. Inputs and targets are
    used in their raw (unstandardized) form, which makes the test split of
    data.cube a genuine extrapolation benchmark.

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
        n_samples: Number of samples in dataset. Defaults to 1000.
    """
    if seeds is None:
        seeds = [42, 123, 999]

    results_dir = _get_results_dir()
    print("=" * 70)
    print("EKSPERYMENT R1: Wplyw funkcji aktywacji w warstwach ukrytych")
    print(f"Badane aktywacje: Sigmoid, Tanh, ReLU | Ziarna losowe: {seeds}")
    print("=" * 70)

    dataset_names = ["activation", "cube"]

    activation_configs = [
        ("Sigmoid", Sigmoid, "xavier"),
        ("Tanh", Tanh, "xavier"),
        ("ReLU", ReLU, "he"),
    ]

    summary_rows: list[dict[str, Any]] = []

    for ds_name in dataset_names:
        print(f"\n--> Przetwarzanie zbioru: {ds_name} (N = {n_samples})...")
        dataset: RegressionDataset = load_regression_data(ds_name, n_samples=n_samples)

        layer_dims = [1, 32, 16, 1]
        layer_names = [
            "Warstwa 1 (1 -> 32)",
            "Warstwa 2 (32 -> 16)",
            "Warstwa wyjsciowa (16 -> 1)",
        ]

        # Containers for aggregated curves across seeds
        loss_curves: dict[str, dict[str, list[list[float]]]] = {}
        mae_curves: dict[str, dict[str, list[list[float]]]] = {}
        norm_curves: dict[str, list[list[list[float]]]] = {}
        rep_neuron_trajectories: dict[str, NeuronTrajectory] = {}
        rep_models: dict[str, MLP] = {}

        for act_name, act_cls, init_method in activation_configs:
            loss_curves[act_name] = {"train": [], "test": []}
            mae_curves[act_name] = {"train": [], "test": []}
            norm_curves[act_name] = []

            final_train_losses: list[float] = []
            final_test_losses: list[float] = []
            final_train_maes: list[float] = []
            final_test_maes: list[float] = []
            final_train_r2s: list[float] = []
            final_test_r2s: list[float] = []

            for seed in seeds:
                set_seed(seed)
                model = MLP.from_dimensions(
                    layer_dims=layer_dims,
                    hidden_activation=act_cls,
                    output_activation=None,
                    use_bias=True,
                    weight_init=init_method,
                )
                optimizer = SGD(learning_rate=0.001)
                loss_fn = MSELoss()

                history = model.fit(
                    x_train=dataset.x_train,
                    y_train=dataset.y_train,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=150,
                    batch_size=32,
                    validation_data=(dataset.x_test, dataset.y_test),
                    metrics={"mae": compute_mae, "r2": compute_r2},
                    verbose=False,
                )

                loss_curves[act_name]["train"].append(history["loss"])
                loss_curves[act_name]["test"].append(history["val_loss"])
                mae_curves[act_name]["train"].append(history["mae"])
                mae_curves[act_name]["test"].append(history["val_mae"])
                norm_curves[act_name].append(history["weight_norms"])

                # Exact final metrics computed over the whole split
                y_pred_train = model.predict(dataset.x_train)
                y_pred_test = model.predict(dataset.x_test)

                final_train_losses.append(
                    model.evaluate(dataset.x_train, dataset.y_train, loss_fn)
                )
                final_test_losses.append(
                    model.evaluate(dataset.x_test, dataset.y_test, loss_fn)
                )
                final_train_maes.append(compute_mae(y_pred_train, dataset.y_train))
                final_test_maes.append(compute_mae(y_pred_test, dataset.y_test))
                final_train_r2s.append(compute_r2(y_pred_train, dataset.y_train))
                final_test_r2s.append(compute_r2(y_pred_test, dataset.y_test))

                # Store the model and neuron trajectories for the primary seed (42)
                if seed == seeds[0]:
                    rep_models[act_name] = model
                    rep_neuron_trajectories[act_name] = history["tracked_neuron_weights"]

            summary_rows.append({
                "dataset": ds_name,
                "activation": act_name,
                "test_mse_mean": float(np.mean(final_test_losses)),
                "test_mse_std": float(np.std(final_test_losses)),
                "train_mse_mean": float(np.mean(final_train_losses)),
                "train_mse_std": float(np.std(final_train_losses)),
                "test_mae_mean": float(np.mean(final_test_maes)),
                "test_mae_std": float(np.std(final_test_maes)),
                "train_mae_mean": float(np.mean(final_train_maes)),
                "train_mae_std": float(np.std(final_train_maes)),
                "test_r2_mean": float(np.mean(final_test_r2s)),
                "test_r2_std": float(np.std(final_test_r2s)),
                "train_r2_mean": float(np.mean(final_train_r2s)),
                "train_r2_std": float(np.std(final_train_r2s)),
            })

            print(
                f"   * [{act_name:<7}] MSE testowe: "
                f"{np.mean(final_test_losses):10.4f} (+/- {np.std(final_test_losses):9.4f}) | "
                f"MAE testowe: {np.mean(final_test_maes):9.4f} | "
                f"R2 testowe: {np.mean(final_test_r2s):8.4f}"
            )

        # Plot 1: Approximation curves on the training split (representative seed)
        train_plot_path = results_dir / f"r1_{ds_name}_approximation_train.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=train_plot_path,
            split="train",
            seed=seeds[0],
        )
        print(f"   [Wykres] Zapisano aproksymacje (zbior uczacy): {train_plot_path.name}")

        # Plot 2: Approximation curves on the test split (representative seed)
        test_plot_path = results_dir / f"r1_{ds_name}_approximation_test.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=test_plot_path,
            split="test",
            seed=seeds[0],
        )
        print(f"   [Wykres] Zapisano aproksymacje (zbior testowy): {test_plot_path.name}")

        # Plot 3: MSE loss curves with std dev band
        epochs_range = np.arange(1, 151)
        loss_plot_path = results_dir / f"r1_{ds_name}_loss_curves.png"
        plot_learning_curves_with_std(
            epochs=epochs_range,
            results_by_act=loss_curves,
            metric_label="Wartość funkcji straty (MSE)",
            title=(
                f"Przebieg funkcji straty (MSE) w kolejnych epokach: {ds_name} "
                f"(Średnia z {len(seeds)} ziaren)"
            ),
            save_path=loss_plot_path,
        )
        print(f"   [Wykres] Zapisano krzywe straty: {loss_plot_path.name}")

        # Plot 4: MAE curves with std dev band
        mae_plot_path = results_dir / f"r1_{ds_name}_mae_curves.png"
        plot_learning_curves_with_std(
            epochs=epochs_range,
            results_by_act=mae_curves,
            metric_label="Średni błąd bezwzględny (MAE)",
            title=(
                f"Przebieg średniego błędu bezwzględnego (MAE) w kolejnych epokach: {ds_name} "
                f"(Średnia z {len(seeds)} ziaren)"
            ),
            save_path=mae_plot_path,
        )
        print(f"   [Wykres] Zapisano krzywe MAE: {mae_plot_path.name}")

        # Plot 5: Weight norms evolution across layers
        norm_plot_path = results_dir / f"r1_{ds_name}_weight_norms.png"
        plot_weight_norms_with_std(
            epochs=epochs_range,
            norms_by_act=norm_curves,
            layer_names=layer_names,
            title=(
                f"Ewolucja norm Frobeniusa wag sieci: {ds_name} "
                f"(Średnia z {len(seeds)} ziaren)"
            ),
            save_path=norm_plot_path,
        )
        print(f"   [Wykres] Zapisano ewolucję norm wag: {norm_plot_path.name}")

        # Plot 6: First neuron weights evolution across layers (representative seed)
        neuron_plot_path = results_dir / f"r1_{ds_name}_neuron_weights.png"
        plot_neuron_weights_trajectory(
            epochs=epochs_range,
            trajectories_by_act=rep_neuron_trajectories,
            layer_names=layer_names,
            title=(
                "Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw "
                f"(Sigmoid vs ReLU vs Tanh): {ds_name} (Ziarno: {seeds[0]})"
            ),
            save_path=neuron_plot_path,
        )
        print(f"   [Wykres] Zapisano ewolucję wag pierwszego neuronu: {neuron_plot_path.name}")

    # Save summary metrics to CSV
    csv_path = results_dir / "r1_summary_metrics.csv"
    fieldnames = [
        "dataset",
        "activation",
        "test_mse_mean",
        "test_mse_std",
        "train_mse_mean",
        "train_mse_std",
        "test_mae_mean",
        "test_mae_std",
        "train_mae_mean",
        "train_mae_std",
        "test_r2_mean",
        "test_r2_std",
        "train_r2_mean",
        "train_r2_std",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[Podsumowanie] Zapisano tabelę metryk: {csv_path.name}")
    print("=" * 70)


def run_experiment_r2() -> None:
    """Placeholder for Experiment R2: Impact of network depth and layer widths."""
    print("Eksperyment R2 nie jest jeszcze zaimplementowany.")


def run_experiment_r3() -> None:
    """Placeholder for Experiment R3: Impact of the loss function (MSE vs MAE)."""
    print("Eksperyment R3 nie jest jeszcze zaimplementowany.")


def run_experiment_r4() -> None:
    """Placeholder for Experiment R4: Impact of the training set size."""
    print("Eksperyment R4 nie jest jeszcze zaimplementowany.")


def main() -> None:
    """CLI entry point parsing experiment target."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Runner eksperymentów regresji MLP (R1, R2, R3, R4)",
    )
    parser.add_argument(
        "experiment",
        choices=["1", "2", "3", "4", "all"],
        nargs="?",
        default="1",
        help=(
            "Numer eksperymentu do uruchomienia: '1' (R1), '2' (R2), '3' (R3), "
            "'4' (R4) lub 'all' (domyślnie: '1')"
        ),
    )

    args = parser.parse_args()

    if args.experiment == "1":
        run_experiment_r1()
    elif args.experiment == "2":
        run_experiment_r2()
    elif args.experiment == "3":
        run_experiment_r3()
    elif args.experiment == "4":
        run_experiment_r4()
    elif args.experiment == "all":
        run_experiment_r1()
        run_experiment_r2()
        run_experiment_r3()
        run_experiment_r4()


if __name__ == "__main__":
    main()
