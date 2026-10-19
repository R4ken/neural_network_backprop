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
    plot_architecture_regression_bars,
    plot_deep_model_weight_norms,
    plot_k3_neuron_weights_grid_2x3,
    plot_k3_weight_update_dynamics,
    plot_learning_curves_with_std,
    plot_multimodel_learning_curves_with_std,
    plot_neuron_weights_comparison_1x3,
    plot_neuron_weights_trajectory,
    plot_r3_loss_curves_split,
    plot_r3_metric_curves,
    plot_regression_approximations_grid,
    plot_sample_size_generalization_curve,
    plot_weight_norms_with_std,
)
from src.py_nn1 import (
    MAELoss,
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


def _compute_final_regression_metrics(
    model: MLP,
    dataset: RegressionDataset,
    loss_fn: Any,
) -> dict[str, float]:
    """Computes final MSE, MAE and R^2 on both full regression splits.

    Args:
        model: Trained model.
        dataset: RegressionDataset with train and test splits.
        loss_fn: Loss used for the ``mse`` entries.

    Returns:
        Dictionary with keys ``train_mse``, ``test_mse``, ``train_mae``,
        ``test_mae``, ``train_r2`` and ``test_r2``.
    """
    y_pred_train = model.predict(dataset.x_train)
    y_pred_test = model.predict(dataset.x_test)
    return {
        "train_mse": model.evaluate(dataset.x_train, dataset.y_train, loss_fn),
        "test_mse": model.evaluate(dataset.x_test, dataset.y_test, loss_fn),
        "train_mae": compute_mae(y_pred_train, dataset.y_train),
        "test_mae": compute_mae(y_pred_test, dataset.y_test),
        "train_r2": compute_r2(y_pred_train, dataset.y_train),
        "test_r2": compute_r2(y_pred_test, dataset.y_test),
    }


def _build_summary_row(
    base_fields: dict[str, Any],
    metric_runs: dict[str, list[float]],
) -> dict[str, Any]:
    """Builds a CSV summary row with ``_mean`` and ``_std`` suffixes.

    Args:
        base_fields: Identifying fields such as dataset and architecture.
        metric_runs: Mapping from metric name to final values across seeds.

    Returns:
        A flat dictionary ready to be written by :class:`csv.DictWriter`.
    """
    row = dict(base_fields)
    for metric_name, values in metric_runs.items():
        row[f"{metric_name}_mean"] = float(np.mean(values))
        row[f"{metric_name}_std"] = float(np.std(values))
    return row


def _write_summary_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Writes summary metrics to a CSV file."""
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def run_experiment_r2(
    seeds: list[int] | None = None,
    n_samples: int = 1000,
    epochs: int = 250,
    lr: float = 0.0005,
    activation_cls: type = Tanh,
    init_method: str = "xavier",
) -> None:
    """Runs Experiment R2: Impact of network depth and layer widths.

    Evaluates 8 architectures (from 0 to 4 hidden layers) on data.activation and
    data.cube. Tanh with Xavier initialization is used by default because it
    remains numerically stable for the deepest raw-data configuration.

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
        n_samples: Number of samples in dataset. Defaults to 1000.
        epochs: Number of training epochs. Defaults to 250.
        lr: Learning rate for SGD optimizer. Defaults to 0.0005.
        activation_cls: Activation class for hidden layers. Defaults to Tanh.
        init_method: Weight initialization strategy. Defaults to "xavier".
    """
    if seeds is None:
        seeds = [42, 123, 999]

    results_dir = _get_results_dir()
    act_name = activation_cls.__name__

    print("=" * 70)
    print("EKSPERYMENT R2: Wplyw liczby warstw ukrytych oraz ich licznosci")
    print(
        f"Aktywacja ukryta: {act_name} ({init_method}) | Epoki: {epochs} | "
        f"LR: {lr} | Ziarna: {seeds}"
    )
    print("=" * 70)

    arch_configs = [
        {
            "key": "0_hidden",
            "name": "0 ukrytych: [1, 1]",
            "short_name": "0 ukrytych",
            "num_hidden": 0,
            "hidden_dims": [],
        },
        {
            "key": "1_hidden_narrow",
            "name": "1 ukryta (4): [1, 4, 1]",
            "short_name": "1 ukryta (4)",
            "num_hidden": 1,
            "hidden_dims": [4],
        },
        {
            "key": "1_hidden_medium",
            "name": "1 ukryta (16): [1, 16, 1]",
            "short_name": "1 ukryta (16)",
            "num_hidden": 1,
            "hidden_dims": [16],
        },
        {
            "key": "1_hidden_wide",
            "name": "1 ukryta (64): [1, 64, 1]",
            "short_name": "1 ukryta (64)",
            "num_hidden": 1,
            "hidden_dims": [64],
        },
        {
            "key": "2_hidden_a",
            "name": "2 ukryte (A): [1, 16, 8, 1]",
            "short_name": "2 ukryte (16-8)",
            "num_hidden": 2,
            "hidden_dims": [16, 8],
        },
        {
            "key": "2_hidden_b",
            "name": "2 ukryte (B): [1, 32, 16, 1]",
            "short_name": "2 ukryte (32-16)",
            "num_hidden": 2,
            "hidden_dims": [32, 16],
        },
        {
            "key": "3_hidden",
            "name": "3 ukryte: [1, 32, 16, 8, 1]",
            "short_name": "3 ukryte",
            "num_hidden": 3,
            "hidden_dims": [32, 16, 8],
        },
        {
            "key": "4_hidden",
            "name": "4 ukryte: [1, 64, 32, 16, 8, 1]",
            "short_name": "4 ukryte",
            "num_hidden": 4,
            "hidden_dims": [64, 32, 16, 8],
        },
    ]

    neuron_comp_keys = ["1_hidden_medium", "2_hidden_b", "4_hidden"]
    summary_rows: list[dict[str, Any]] = []

    for ds_name in ("activation", "cube"):
        print(f"\n--> Przetwarzanie zbioru: {ds_name} (N = {n_samples})...")
        dataset = load_regression_data(ds_name, n_samples=n_samples)

        loss_curves: dict[str, dict[str, list[list[float]]]] = {}
        mae_curves: dict[str, dict[str, list[list[float]]]] = {}
        rep_models: dict[str, MLP] = {}
        rep_neuron_trajectories: dict[str, NeuronTrajectory] = {}
        deep_norm_runs: list[list[list[float]]] = []

        deep_layer_names = [
            "Warstwa 1 (1 -> 64)",
            "Warstwa 2 (64 -> 32)",
            "Warstwa 3 (32 -> 16)",
            "Warstwa 4 (16 -> 8)",
            "Warstwa wyjsciowa (8 -> 1)",
        ]

        arch_labels: list[str] = []
        train_mse_means: list[float] = []
        train_mse_stds: list[float] = []
        test_mse_means: list[float] = []
        test_mse_stds: list[float] = []

        for arch in arch_configs:
            arch_key = arch["key"]
            arch_name = arch["name"]
            num_hidden = arch["num_hidden"]
            hidden_dims = arch["hidden_dims"]
            layer_dims = [1] + hidden_dims + [1]
            arch_labels.append(arch["short_name"])

            loss_curves[arch_name] = {"train": [], "test": []}
            mae_curves[arch_name] = {"train": [], "test": []}

            metric_runs: dict[str, list[float]] = {
                "test_mse": [],
                "train_mse": [],
                "test_mae": [],
                "train_mae": [],
                "test_r2": [],
                "train_r2": [],
            }

            for seed in seeds:
                set_seed(seed)
                model = MLP.from_dimensions(
                    layer_dims=layer_dims,
                    hidden_activation=activation_cls,
                    output_activation=None,
                    use_bias=True,
                    weight_init=init_method,
                )
                optimizer = SGD(learning_rate=lr)
                loss_fn = MSELoss()

                history = model.fit(
                    x_train=dataset.x_train,
                    y_train=dataset.y_train,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(dataset.x_test, dataset.y_test),
                    metrics={"mae": compute_mae},
                    verbose=False,
                )

                loss_curves[arch_name]["train"].append(history["loss"])
                loss_curves[arch_name]["test"].append(history["val_loss"])
                mae_curves[arch_name]["train"].append(history["mae"])
                mae_curves[arch_name]["test"].append(history["val_mae"])

                if arch_key == "4_hidden":
                    deep_norm_runs.append(history["weight_norms"])

                final_metrics = _compute_final_regression_metrics(model, dataset, loss_fn)
                for metric_name, values in metric_runs.items():
                    values.append(final_metrics[metric_name])

                if seed == seeds[0]:
                    rep_models[arch_name] = model
                    if arch_key in neuron_comp_keys:
                        rep_neuron_trajectories[arch_name] = history["tracked_neuron_weights"]

            summary_rows.append(
                _build_summary_row(
                    {
                        "dataset": ds_name,
                        "architecture": arch_name,
                        "num_hidden_layers": num_hidden,
                    },
                    metric_runs,
                )
            )

            mean_tr_mse = float(np.mean(metric_runs["train_mse"]))
            std_tr_mse = float(np.std(metric_runs["train_mse"]))
            mean_te_mse = float(np.mean(metric_runs["test_mse"]))
            std_te_mse = float(np.std(metric_runs["test_mse"]))
            mean_te_mae = float(np.mean(metric_runs["test_mae"]))
            mean_te_r2 = float(np.mean(metric_runs["test_r2"]))

            train_mse_means.append(mean_tr_mse)
            train_mse_stds.append(std_tr_mse)
            test_mse_means.append(mean_te_mse)
            test_mse_stds.append(std_te_mse)

            print(
                f"   * [{arch_name:<28}] MSE testowe: "
                f"{mean_te_mse:10.4f} (+/- {std_te_mse:9.4f}) | "
                f"MAE testowe: {mean_te_mae:9.4f} | "
                f"R2 testowe: {mean_te_r2:8.4f}"
            )

        epochs_range = np.arange(1, epochs + 1)

        # Plot 1: Approximations on the training split (representative seed)
        approx_train_path = results_dir / f"r2_{ds_name}_approximation_train.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=approx_train_path,
            split="train",
            seed=seeds[0],
            nrows=2,
            ncols=4,
            title=(
                f"Aproksymacja funkcji jednoargumentowej: {ds_name} – zbiór uczący "
                f"(architektury 0–4 warstw, Ziarno: {seeds[0]})"
            ),
        )
        print(f"   [Wykres 1/7] Zapisano aproksymacje (zbior uczacy): {approx_train_path.name}")

        # Plot 2: Approximations on the test split with training boundaries
        approx_test_path = results_dir / f"r2_{ds_name}_approximation_test.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=approx_test_path,
            split="test",
            seed=seeds[0],
            nrows=2,
            ncols=4,
            title=(
                f"Aproksymacja funkcji jednoargumentowej: {ds_name} – zbiór testowy "
                f"(ekstrapolacja, architektury 0–4 warstw, Ziarno: {seeds[0]})"
            ),
        )
        print(f"   [Wykres 2/7] Zapisano aproksymacje (zbior testowy): {approx_test_path.name}")

        # Plot 3: Final MSE bars (train vs test)
        bars_path = results_dir / f"r2_{ds_name}_mse_bars.png"
        plot_architecture_regression_bars(
            arch_labels=arch_labels,
            train_values=train_mse_means,
            train_stds=train_mse_stds,
            test_values=test_mse_means,
            test_stds=test_mse_stds,
            title=(
                f"Błąd MSE na zbiorze uczącym i testowym w zależności od architektury: "
                f"{ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=bars_path,
        )
        print(f"   [Wykres 3/7] Zapisano słupki MSE: {bars_path.name}")

        # Plot 4: MSE learning curves
        loss_plot_path = results_dir / f"r2_{ds_name}_loss_curves.png"
        plot_multimodel_learning_curves_with_std(
            epochs=epochs_range,
            results_by_model=loss_curves,
            metric_label="Wartość funkcji straty (MSE)",
            title=(
                f"Przebieg funkcji straty (MSE) dla architektur o różnej głębokości: "
                f"{ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=loss_plot_path,
            legend_title="Architektura",
            sharey=False,
        )
        print(f"   [Wykres 4/7] Zapisano krzywe straty: {loss_plot_path.name}")

        # Plot 5: MAE learning curves
        mae_plot_path = results_dir / f"r2_{ds_name}_mae_curves.png"
        plot_multimodel_learning_curves_with_std(
            epochs=epochs_range,
            results_by_model=mae_curves,
            metric_label="Średni błąd bezwzględny (MAE)",
            title=(
                f"Przebieg średniego błędu bezwzględnego (MAE) dla architektur "
                f"o różnej głębokości: {ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=mae_plot_path,
            legend_title="Architektura",
            sharey=False,
        )
        print(f"   [Wykres 5/7] Zapisano krzywe MAE: {mae_plot_path.name}")

        # Plot 6: Weight norms of all Dense layers in the 4-hidden-layer network
        norm_plot_path = results_dir / f"r2_{ds_name}_weight_norms_deep.png"
        plot_deep_model_weight_norms(
            epochs=epochs_range,
            seed_runs=deep_norm_runs,
            layer_names=deep_layer_names,
            title=(
                f"Ewolucja norm wag poszczególnych warstw w sieci 4-warstwowej: "
                f"{ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=norm_plot_path,
        )
        print(f"   [Wykres 6/7] Zapisano normy wag sieci 4-warstwowej: {norm_plot_path.name}")

        # Plot 7: First-neuron comparison across shallow, medium and deep models
        neuron_plot_path = results_dir / f"r2_{ds_name}_neuron_weights.png"
        plot_neuron_weights_comparison_1x3(
            epochs=epochs_range,
            trajectories_by_model=rep_neuron_trajectories,
            title=(
                "Ewolucja wagi i biasu pierwszego neuronu w pierwszej warstwie ukrytej "
                f"(1, 2 i 4 warstwy ukryte): {ds_name} (Ziarno: {seeds[0]})"
            ),
            save_path=neuron_plot_path,
        )
        print(f"   [Wykres 7/7] Zapisano porównanie neuronu (1x3): {neuron_plot_path.name}")

    csv_path = results_dir / "r2_summary_metrics.csv"
    fieldnames = [
        "dataset",
        "architecture",
        "num_hidden_layers",
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
    _write_summary_csv(csv_path, summary_rows, fieldnames)

    print(f"\n[Podsumowanie R2] Zapisano tabelę metryk: {csv_path.name}")
    print("=" * 70)


def run_experiment_r3(
    seeds: list[int] | None = None,
    n_samples: int = 1000,
    epochs: int = 150,
    learning_rates: list[float] | None = None,
    activation_cls: type = Tanh,
    init_method: str = "xavier",
) -> None:
    """Runs Experiment R3: Impact of the loss function (MSE vs MAE).

    Compares MSELoss and MAELoss across several learning rates on data.activation
    and data.cube. A constant MAE gradient requires a different step size than
    the residual-dependent MSE gradient, hence the learning-rate sweep.

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
        n_samples: Number of samples in dataset. Defaults to 1000.
        epochs: Number of training epochs. Defaults to 150.
        learning_rates: Learning rates to evaluate. Defaults to [0.001, 0.005, 0.01].
        activation_cls: Activation class for hidden layers. Defaults to Tanh.
        init_method: Weight initialization strategy. Defaults to "xavier".
    """
    if seeds is None:
        seeds = [42, 123, 999]
    if learning_rates is None:
        learning_rates = [0.001, 0.005, 0.01]

    results_dir = _get_results_dir()
    act_name = activation_cls.__name__

    print("=" * 70)
    print("EKSPERYMENT R3: Wplyw miary bledu na wyjsciu sieci (MSE vs MAE)")
    print(
        f"Aktywacja ukryta: {act_name} ({init_method}) | Epoki: {epochs} | "
        f"LR: {learning_rates} | Ziarna: {seeds}"
    )
    print("=" * 70)

    loss_configs = [
        ("MSE", MSELoss),
        ("MAE", MAELoss),
    ]
    summary_rows: list[dict[str, Any]] = []

    for ds_name in ("activation", "cube"):
        print(f"\n--> Przetwarzanie zbioru: {ds_name} (N = {n_samples})...")
        dataset = load_regression_data(ds_name, n_samples=n_samples)

        layer_dims = [1, 32, 16, 1]
        layer_names = [
            "Warstwa 1 (1 -> 32)",
            "Warstwa 2 (32 -> 16)",
            "Warstwa wyjsciowa (16 -> 1)",
        ]

        mae_by_lr: dict[float, dict[str, dict[str, list[list[float]]]]] = {
            lr: {
                loss_name: {"train": [], "test": []}
                for loss_name, _ in loss_configs
            }
            for lr in learning_rates
        }
        loss_by_fn: dict[str, dict[float, dict[str, list[list[float]]]]] = {
            loss_name: {
                lr: {"train": [], "test": []}
                for lr in learning_rates
            }
            for loss_name, _ in loss_configs
        }
        updates_by_lr: dict[float, dict[str, list[list[float]]]] = {
            lr: {loss_name: [] for loss_name, _ in loss_configs}
            for lr in learning_rates
        }
        rep_neuron_trajectories_grid: dict[str, dict[float, NeuronTrajectory]] = {
            loss_name: {} for loss_name, _ in loss_configs
        }
        rep_models: dict[str, MLP] = {}

        for loss_name, loss_cls in loss_configs:
            for lr in learning_rates:
                metric_runs: dict[str, list[float]] = {
                    "test_mse": [],
                    "train_mse": [],
                    "test_mae": [],
                    "train_mae": [],
                    "test_r2": [],
                    "train_r2": [],
                }

                for seed in seeds:
                    set_seed(seed)
                    model = MLP.from_dimensions(
                        layer_dims=layer_dims,
                        hidden_activation=activation_cls,
                        output_activation=None,
                        use_bias=True,
                        weight_init=init_method,
                    )
                    optimizer = SGD(learning_rate=lr)
                    loss_fn = loss_cls()

                    history = model.fit(
                        x_train=dataset.x_train,
                        y_train=dataset.y_train,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(dataset.x_test, dataset.y_test),
                        metrics={"mae": compute_mae},
                        verbose=False,
                    )

                    mae_by_lr[lr][loss_name]["train"].append(history["mae"])
                    mae_by_lr[lr][loss_name]["test"].append(history["val_mae"])
                    loss_by_fn[loss_name][lr]["train"].append(history["loss"])
                    loss_by_fn[loss_name][lr]["test"].append(history["val_loss"])

                    total_updates = [
                        float(np.sqrt(sum(u**2 for u in epoch_updates)))
                        for epoch_updates in history["weight_update_norms"]
                    ]
                    updates_by_lr[lr][loss_name].append(total_updates)

                    # MSE/MAE/R^2 are always reported with the same definitions,
                    # independently of the optimized loss function.
                    final_metrics = _compute_final_regression_metrics(
                        model, dataset, MSELoss()
                    )
                    for metric_name, values in metric_runs.items():
                        values.append(final_metrics[metric_name])

                    if seed == seeds[0]:
                        model_label = f"{loss_name} (lr = {lr})"
                        rep_models[model_label] = model
                        rep_neuron_trajectories_grid[loss_name][lr] = (
                            history["tracked_neuron_weights"]
                        )

                summary_rows.append(
                    _build_summary_row(
                        {
                            "dataset": ds_name,
                            "loss_function": loss_name,
                            "learning_rate": lr,
                        },
                        metric_runs,
                    )
                )

                mean_te_mse = float(np.mean(metric_runs["test_mse"]))
                std_te_mse = float(np.std(metric_runs["test_mse"]))
                mean_te_mae = float(np.mean(metric_runs["test_mae"]))
                mean_te_r2 = float(np.mean(metric_runs["test_r2"]))

                print(
                    f"   * [{loss_name:<3} | lr={lr:<6}] MSE testowe: "
                    f"{mean_te_mse:10.4f} (+/- {std_te_mse:9.4f}) | "
                    f"MAE testowe: {mean_te_mae:9.4f} | "
                    f"R2 testowe: {mean_te_r2:8.4f}"
                )

        epochs_range = np.arange(1, epochs + 1)

        # Plot 1: Approximations on the training split (representative seed)
        approx_train_path = results_dir / f"r3_{ds_name}_approximation_train.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=approx_train_path,
            split="train",
            seed=seeds[0],
            nrows=2,
            ncols=3,
            title=(
                f"Aproksymacja funkcji jednoargumentowej: {ds_name} – zbiór uczący "
                f"(MSE vs MAE, Ziarno: {seeds[0]})"
            ),
        )
        print(f"   [Wykres 1/6] Zapisano aproksymacje (zbior uczacy): {approx_train_path.name}")

        # Plot 2: Approximations on the test split with training boundaries
        approx_test_path = results_dir / f"r3_{ds_name}_approximation_test.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=approx_test_path,
            split="test",
            seed=seeds[0],
            nrows=2,
            ncols=3,
            title=(
                f"Aproksymacja funkcji jednoargumentowej: {ds_name} – zbiór testowy "
                f"(ekstrapolacja, MSE vs MAE, Ziarno: {seeds[0]})"
            ),
        )
        print(f"   [Wykres 2/6] Zapisano aproksymacje (zbior testowy): {approx_test_path.name}")

        # Plot 3: Common MAE metric across learning rates (train/test rows)
        mae_plot_path = results_dir / f"r3_{ds_name}_mae_curves.png"
        plot_r3_metric_curves(
            epochs=epochs_range,
            metric_by_lr=mae_by_lr,
            metric_label="Średni błąd bezwzględny (MAE)",
            title=(
                f"Przebieg średniego błędu bezwzględnego (MAE): MSE vs MAE "
                f"w zależności od lr: {ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=mae_plot_path,
        )
        print(f"   [Wykres 3/6] Zapisano krzywe MAE: {mae_plot_path.name}")

        # Plot 4: Native loss curves in split scales
        loss_plot_path = results_dir / f"r3_{ds_name}_loss_curves.png"
        plot_r3_loss_curves_split(
            epochs=epochs_range,
            loss_by_fn=loss_by_fn,
            title=(
                f"Przebieg funkcji straty: MSE vs MAE dla badanych lr: {ds_name} "
                f"(Średnia z {len(seeds)} ziaren)"
            ),
            save_path=loss_plot_path,
        )
        print(f"   [Wykres 4/6] Zapisano krzywe strat (rozdzielone skale): {loss_plot_path.name}")

        # Plot 5: Weight update dynamics
        dynamics_path = results_dir / f"r3_{ds_name}_weight_update_dynamics.png"
        plot_k3_weight_update_dynamics(
            epochs=epochs_range,
            updates_by_lr=updates_by_lr,
            title=(
                f"Dynamika aktualizacji wag w kolejnych epokach (MSE vs MAE): {ds_name}"
            ),
            save_path=dynamics_path,
        )
        print(f"   [Wykres 5/6] Zapisano dynamikę aktualizacji wag: {dynamics_path.name}")

        # Plot 6: First-neuron trajectories on a 2x3 grid (loss x learning rate)
        neuron_plot_path = results_dir / f"r3_{ds_name}_neuron_weights_grid.png"
        plot_k3_neuron_weights_grid_2x3(
            epochs=epochs_range,
            trajectories_grid=rep_neuron_trajectories_grid,
            title=(
                f"Ewolucja wagi i biasu pierwszego neuronu w warstwie 1 "
                f"(MSE vs MAE): {ds_name} (Ziarno: {seeds[0]})"
            ),
            save_path=neuron_plot_path,
            loss_names=["MSE", "MAE"],
        )
        print(f"   [Wykres 6/6] Zapisano ewolucję neuronu (2x3): {neuron_plot_path.name}")

    csv_path = results_dir / "r3_summary_metrics.csv"
    fieldnames = [
        "dataset",
        "loss_function",
        "learning_rate",
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
    _write_summary_csv(csv_path, summary_rows, fieldnames)

    print(f"\n[Podsumowanie R3] Zapisano tabelę metryk: {csv_path.name}")
    print("=" * 70)


def run_experiment_r4(
    seeds: list[int] | None = None,
    sample_sizes: list[int] | None = None,
    epochs: int = 150,
    lr: float = 0.001,
    activation_cls: type = Tanh,
    init_method: str = "xavier",
) -> None:
    """Runs Experiment R4: Impact of the training set size.

    Trains the same architecture on data.activation and data.cube for
    N in {100, 500, 1000, 10000}, comparing the generalization gap and the
    shape of the fitted curve.

    Args:
        seeds: Random seeds for statistical evaluation. Defaults to [42, 123, 999].
        sample_sizes: Training-set sizes to evaluate. Defaults to [100, 500, 1000, 10000].
        epochs: Number of training epochs. Defaults to 150.
        lr: Learning rate for SGD optimizer. Defaults to 0.001.
        activation_cls: Activation class for hidden layers. Defaults to Tanh.
        init_method: Weight initialization strategy. Defaults to "xavier".
    """
    if seeds is None:
        seeds = [42, 123, 999]
    if sample_sizes is None:
        sample_sizes = [100, 500, 1000, 10000]

    results_dir = _get_results_dir()
    act_name = activation_cls.__name__

    print("=" * 70)
    print("EKSPERYMENT R4: Wplyw rozmiaru zbioru danych uczacych")
    print(
        f"Aktywacja ukryta: {act_name} ({init_method}) | Epoki: {epochs} | "
        f"LR: {lr} | N: {sample_sizes} | Ziarna: {seeds}"
    )
    print("=" * 70)

    summary_rows: list[dict[str, Any]] = []
    layer_dims = [1, 32, 16, 1]
    layer_names = [
        "Warstwa 1 (1 -> 32)",
        "Warstwa 2 (32 -> 16)",
        "Warstwa wyjsciowa (16 -> 1)",
    ]

    for ds_name in ("activation", "cube"):
        print(f"\n--> Przetwarzanie zbioru: {ds_name}...")

        loss_curves: dict[str, dict[str, list[list[float]]]] = {}
        norm_curves: dict[str, list[list[list[float]]]] = {}
        rep_models: dict[str, MLP] = {}
        rep_neuron_trajectories: dict[str, NeuronTrajectory] = {}

        train_mse_means: list[float] = []
        train_mse_stds: list[float] = []
        test_mse_means: list[float] = []
        test_mse_stds: list[float] = []

        for n in sample_sizes:
            dataset = load_regression_data(ds_name, n_samples=n)
            label = f"N = {n}"
            print(f"   -> N = {n} (zbior uczacy: {len(dataset.x_train)} probek)")

            loss_curves[label] = {"train": [], "test": []}
            norm_curves[label] = []

            metric_runs: dict[str, list[float]] = {
                "test_mse": [],
                "train_mse": [],
                "test_mae": [],
                "train_mae": [],
                "test_r2": [],
                "train_r2": [],
            }

            for seed in seeds:
                set_seed(seed)
                model = MLP.from_dimensions(
                    layer_dims=layer_dims,
                    hidden_activation=activation_cls,
                    output_activation=None,
                    use_bias=True,
                    weight_init=init_method,
                )
                optimizer = SGD(learning_rate=lr)
                loss_fn = MSELoss()

                history = model.fit(
                    x_train=dataset.x_train,
                    y_train=dataset.y_train,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(dataset.x_test, dataset.y_test),
                    metrics={"mae": compute_mae},
                    verbose=False,
                )

                loss_curves[label]["train"].append(history["loss"])
                loss_curves[label]["test"].append(history["val_loss"])
                norm_curves[label].append(history["weight_norms"])

                final_metrics = _compute_final_regression_metrics(model, dataset, loss_fn)
                for metric_name, values in metric_runs.items():
                    values.append(final_metrics[metric_name])

                if seed == seeds[0]:
                    rep_models[label] = model
                    rep_neuron_trajectories[label] = history["tracked_neuron_weights"]

            summary_rows.append(
                _build_summary_row(
                    {"dataset": ds_name, "n_samples": n},
                    metric_runs,
                )
            )

            mean_tr_mse = float(np.mean(metric_runs["train_mse"]))
            std_tr_mse = float(np.std(metric_runs["train_mse"]))
            mean_te_mse = float(np.mean(metric_runs["test_mse"]))
            std_te_mse = float(np.std(metric_runs["test_mse"]))
            mean_te_mae = float(np.mean(metric_runs["test_mae"]))
            mean_te_r2 = float(np.mean(metric_runs["test_r2"]))

            train_mse_means.append(mean_tr_mse)
            train_mse_stds.append(std_tr_mse)
            test_mse_means.append(mean_te_mse)
            test_mse_stds.append(std_te_mse)

            print(
                f"      [{label:<9}] MSE testowe: "
                f"{mean_te_mse:10.4f} (+/- {std_te_mse:9.4f}) | "
                f"MAE testowe: {mean_te_mae:9.4f} | "
                f"R2 testowe: {mean_te_r2:8.4f}"
            )

        epochs_range = np.arange(1, epochs + 1)

        # Plot 1: Approximations on the training split (representative seed)
        approx_train_path = results_dir / f"r4_{ds_name}_approximation_train.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=approx_train_path,
            split="train",
            seed=seeds[0],
            nrows=1,
            ncols=len(sample_sizes),
            point_size=7.0,
            alpha=0.3,
            title=(
                f"Aproksymacja funkcji jednoargumentowej: {ds_name} – zbiór uczący "
                f"w zależności od N (Ziarno: {seeds[0]})"
            ),
        )
        print(f"   [Wykres 1/6] Zapisano aproksymacje (zbior uczacy): {approx_train_path.name}")

        # Plot 2: Approximations on the test split with training boundaries
        approx_test_path = results_dir / f"r4_{ds_name}_approximation_test.png"
        plot_regression_approximations_grid(
            models=rep_models,
            dataset=dataset,
            save_path=approx_test_path,
            split="test",
            seed=seeds[0],
            nrows=1,
            ncols=len(sample_sizes),
            point_size=7.0,
            alpha=0.3,
            title=(
                f"Aproksymacja funkcji jednoargumentowej: {ds_name} – zbiór testowy "
                f"(ekstrapolacja) w zależności od N (Ziarno: {seeds[0]})"
            ),
        )
        print(f"   [Wykres 2/6] Zapisano aproksymacje (zbior testowy): {approx_test_path.name}")

        # Plot 3: Generalization curve (MSE vs N, train/test panels)
        generalization_path = results_dir / f"r4_{ds_name}_generalization_curve.png"
        plot_sample_size_generalization_curve(
            sample_sizes=sample_sizes,
            train_means=train_mse_means,
            train_stds=train_mse_stds,
            test_means=test_mse_means,
            test_stds=test_mse_stds,
            title=(
                f"Błąd MSE na zbiorze uczącym i testowym w funkcji wielkości próby N: "
                f"{ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=generalization_path,
        )
        print(f"   [Wykres 3/6] Zapisano krzywą generalizacji: {generalization_path.name}")

        # Plot 4: MSE learning curves for each N
        loss_plot_path = results_dir / f"r4_{ds_name}_loss_curves.png"
        plot_multimodel_learning_curves_with_std(
            epochs=epochs_range,
            results_by_model=loss_curves,
            metric_label="Wartość funkcji straty (MSE)",
            title=(
                f"Przebieg funkcji straty (MSE) w zależności od rozmiaru zbioru N: "
                f"{ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=loss_plot_path,
            legend_title="Rozmiar zbioru N",
            sharey=False,
        )
        print(f"   [Wykres 4/6] Zapisano krzywe straty: {loss_plot_path.name}")

        # Plot 5: Weight norms across layers for each N
        norm_plot_path = results_dir / f"r4_{ds_name}_weight_norms.png"
        plot_weight_norms_with_std(
            epochs=epochs_range,
            norms_by_act=norm_curves,
            layer_names=layer_names,
            title=(
                f"Ewolucja norm Frobeniusa wag sieci w zależności od rozmiaru zbioru N: "
                f"{ds_name} (Średnia z {len(seeds)} ziaren)"
            ),
            save_path=norm_plot_path,
            legend_title="Rozmiar zbioru N",
        )
        print(f"   [Wykres 5/6] Zapisano ewolucję norm wag: {norm_plot_path.name}")

        # Plot 6: First-neuron trajectories for each N
        neuron_plot_path = results_dir / f"r4_{ds_name}_neuron_weights.png"
        plot_neuron_weights_trajectory(
            epochs=epochs_range,
            trajectories_by_act=rep_neuron_trajectories,
            layer_names=layer_names,
            title=(
                f"Ewolucja wagi i biasu pierwszego neuronu poszczególnych warstw "
                f"w zależności od rozmiaru zbioru N: {ds_name} (Ziarno: {seeds[0]})"
            ),
            save_path=neuron_plot_path,
        )
        print(f"   [Wykres 6/6] Zapisano ewolucję wag pierwszego neuronu: {neuron_plot_path.name}")

    csv_path = results_dir / "r4_summary_metrics.csv"
    fieldnames = [
        "dataset",
        "n_samples",
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
    _write_summary_csv(csv_path, summary_rows, fieldnames)

    print(f"\n[Podsumowanie R4] Zapisano tabelę metryk: {csv_path.name}")
    print("=" * 70)


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
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Liczba próbek w zbiorze danych dla R1-R3 (domyślnie: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Liczba epok treningu dla R1, R3 i R4 (domyślnie: 150)",
    )
    parser.add_argument(
        "--r2-epochs",
        type=int,
        default=250,
        help="Liczba epok treningu dla R2 (domyślnie: 250)",
    )
    parser.add_argument(
        "--r2-lr",
        type=float,
        default=0.0005,
        help="Współczynnik uczenia dla R2 (domyślnie: 0.0005)",
    )

    args = parser.parse_args()

    if args.experiment == "1":
        run_experiment_r1(n_samples=args.n_samples)
    elif args.experiment == "2":
        run_experiment_r2(
            n_samples=args.n_samples,
            epochs=args.r2_epochs,
            lr=args.r2_lr,
        )
    elif args.experiment == "3":
        run_experiment_r3(n_samples=args.n_samples, epochs=args.epochs)
    elif args.experiment == "4":
        run_experiment_r4(epochs=args.epochs)
    elif args.experiment == "all":
        run_experiment_r1(n_samples=args.n_samples)
        run_experiment_r2(
            n_samples=args.n_samples,
            epochs=args.r2_epochs,
            lr=args.r2_lr,
        )
        run_experiment_r3(n_samples=args.n_samples, epochs=args.epochs)
        run_experiment_r4(epochs=args.epochs)


if __name__ == "__main__":
    main()
