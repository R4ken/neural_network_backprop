import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Support running as a standalone script or importing from project root
_repo_root = Path(__file__).resolve().parents[1]
_experiments_dir = Path(__file__).resolve().parent
for _p in [str(_repo_root), str(_experiments_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data_loader import load_mnist_data
from common.plotting import (
    compute_confusion_matrix,
    compute_confusion_matrix_metrics,
    plot_confusion_matrix,
    plot_training_and_validation_curves,
    save_metrics_report,
)
from py_nn1 import (
    MLP,
    ReLU,
    Softmax,
    CategoricalCrossEntropyLoss,
    SGD,
    compute_multiclass_accuracy,
    set_seed
)


DEFAULT_SEEDS = list(range(54217, 54227))
CURVE_KEYS = ("loss", "val_loss", "accuracy", "val_accuracy")


def aggregate_histories(histories: list[dict]) -> tuple[dict, dict]:
    """Return per-epoch means and population standard deviations (ddof=0)."""
    if not histories:
        raise ValueError("At least one training history is required.")
    means, stds = {}, {}
    epoch_count = len(histories[0]["loss"])
    if epoch_count == 0:
        raise ValueError("Training histories must contain at least one epoch.")
    for key in CURVE_KEYS:
        runs = np.asarray([history[key] for history in histories], dtype=float)
        if runs.shape != (len(histories), epoch_count) or not np.all(np.isfinite(runs)):
            raise ValueError(f"Invalid or inconsistent histories for {key}.")
        means[key] = runs.mean(axis=0).tolist()
        stds[key] = runs.std(axis=0, ddof=0).tolist()
    return means, stds


def plot_misclassified_samples(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_pairs: list[tuple[int, int]] | None = None,
    save_path: Path | str | None = None,
) -> Path:
    """Finds and plots misclassified test samples side-by-side in a 1x3 panel.

    Args:
        x_test: Test features of shape (N, 784).
        y_true: True labels (one-hot matrix or 1D array of class indices).
        y_pred: Predicted class probabilities (N, 10) or 1D array of predicted class indices.
        target_pairs: List of (true_class, pred_class) tuples to prioritize displaying.
            Defaults to [(4, 9), (5, 3), (7, 2)].
        save_path: Destination path for saving the figure.

    Returns:
        Path to the saved figure file.
    """
    if target_pairs is None:
        target_pairs = [(4, 9), (5, 3), (7, 2)]

    true_labels = np.argmax(y_true, axis=1) if y_true.ndim == 2 else np.asarray(y_true)
    pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else np.asarray(y_pred)

    chosen_indices: list[int] = []
    chosen_pairs: list[tuple[int, int]] = []

    # 1. Search for specified target pairs
    for t_cls, p_cls in target_pairs:
        matches = np.where((true_labels == t_cls) & (pred_labels == p_cls))[0]
        if len(matches) > 0:
            chosen_indices.append(int(matches[0]))
            chosen_pairs.append((t_cls, p_cls))

    # 2. If fewer than 3 pairs found, fill with other misclassified samples
    if len(chosen_indices) < 3:
        all_misclassified = np.where(true_labels != pred_labels)[0]
        for idx in all_misclassified:
            if idx not in chosen_indices:
                chosen_indices.append(int(idx))
                chosen_pairs.append((int(true_labels[idx]), int(pred_labels[idx])))
                if len(chosen_indices) == 3:
                    break

    fig, axes = plt.subplots(1, len(chosen_indices), figsize=(9, 3.2))
    if len(chosen_indices) == 1:
        axes = [axes]

    for ax, idx, (t_cls, p_cls) in zip(axes, chosen_indices, chosen_pairs):
        img = x_test[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(
            f"Prawdziwa: {t_cls}  |  Predykcja: {p_cls}",
            fontsize=11,
            fontweight="bold",
            pad=8
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path is None:
        target_path = Path(__file__).resolve().parent / "results" / "mnist" / "mnist_misclassified_examples.png"
    else:
        target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target_path


def evaluate_and_plot_mnist(
    model: MLP,
    history: dict,
    x_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path | str | None = None,
) -> dict:
    """Prepares and saves plots and numerical metrics for the MNIST experiment.

    Generates:
        1. Training and validation loss curves for each epoch.
        2. Training and validation accuracy curves for each epoch.
        3. A 10x10 confusion matrix heatmap on the test set.
        4. Numerical metrics derived from the confusion matrix (TPR, TNR, PPV, FPR, F1, Support)
           saved to both JSON and CSV files.
        5. Visual examples of misclassified test images (1x3 panel).

    Args:
        model: Trained MLP instance.
        history: Dictionary returned by model.fit containing loss and metric histories.
        x_test: Test features.
        y_test: Test labels (one-hot or 1D indices).
        results_dir: Output directory (defaults to experiments/results/mnist).

    Returns:
        Dictionary containing the confusion matrix, computed metrics, and output paths.
    """
    if results_dir is None:
        target_dir = Path(__file__).resolve().parent / "results" / "mnist"
    else:
        target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1 & 2. Plot learning curves (loss and accuracy across epochs)
    curves_path = target_dir / "mnist_learning_curves.png"
    plot_training_and_validation_curves(history, save_path=curves_path)
    print(f"Zapisano wykresy krzywych uczenia: {curves_path}")

    # 3. Compute 10x10 confusion matrix on the test dataset
    probabilities = model.predict(x_test)
    cm = compute_confusion_matrix(y_true=y_test, y_pred=probabilities, num_classes=10)

    cm_path = target_dir / "mnist_confusion_matrix.png"
    plot_confusion_matrix(cm, save_path=cm_path)
    print(f"Zapisano macierz pomylek (Confusion Matrix): {cm_path}")

    # 4. Compute numerical evaluation scores from the confusion matrix
    metrics = compute_confusion_matrix_metrics(cm)
    csv_path = target_dir / "mnist_confusion_matrix_metrics.csv"
    save_metrics_report(metrics, csv_path=csv_path)
    print(f"Zapisano tabele metryk w formacie CSV: {csv_path}")

    # 5. Plot misclassified sample examples
    misclassified_path = target_dir / "mnist_misclassified_examples.png"
    plot_misclassified_samples(
        x_test=x_test,
        y_true=y_test,
        y_pred=probabilities,
        save_path=misclassified_path,
    )
    print(f"Zapisano przykłady błędnie sklasyfikowanych próbek: {misclassified_path}")

    # Print formatted table to stdout
    print("\n" + "=" * 80)
    print("MNIST TEST DATASET - WYNIKI NUMERYCZNE DLA MACIERZY POMYLEK (10x10)")
    print("=" * 80)
    print(f"{'Klasa':<7} | {'Support':<8} | {'TPR/Recall':<11} | {'PPV/Precision':<14} | {'TNR/Spec.':<10} | {'F1-Score':<9}")
    print("-" * 80)
    for c_name, vals in metrics["per_class"].items():
        print(
            f"{c_name:<7} | {vals['support']:<8} | {vals['tpr']:<11.4f} | "
            f"{vals['ppv']:<14.4f} | {vals['tnr']:<10.4f} | {vals['f1_score']:<9.4f}"
        )
    print("-" * 80)
    macro = metrics["macro_avg"]
    print(
        f"{'Macro':<7} | {metrics['total_samples']:<8} | {macro['tpr']:<11.4f} | "
        f"{macro['ppv']:<14.4f} | {macro['tnr']:<10.4f} | {macro['f1_score']:<9.4f}"
    )
    weighted = metrics["weighted_avg"]
    print(
        f"{'Weighted':<7} | {metrics['total_samples']:<8} | {weighted['tpr']:<11.4f} | "
        f"{weighted['ppv']:<14.4f} | {weighted['tnr']:<10.4f} | {weighted['f1_score']:<9.4f}"
    )
    print(f"Dokladnosc calkowita (Overall Accuracy): {metrics['overall_accuracy'] * 100:.2f}%")
    print("=" * 80 + "\n")

    return {
        "confusion_matrix": cm,
        "metrics": metrics,
        "curves_path": curves_path,
        "cm_path": cm_path,
        "csv_path": csv_path,
        "misclassified_path": misclassified_path,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Trenowanie i ewaluacja modelu MLP na zbiorze MNIST.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Liczba epok treningowych (domyslnie: 30)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rozmiar mini-batcha (domyslnie: 32)."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Wspolczynnik uczenia SGD (domyslnie: 0.05)."
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Liczba probek uczacych (domyslnie: 55000)."
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5000,
        help="Liczba probek walidacyjnych (domyslnie: 5000)."
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Liczba probek testowych (domyslnie: wszystkie 10000)."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tryb szybki (1 epoka, 1000 probek) do testow technicznych."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Ziarna treningu (domyslnie: 54217--54226, 10 uruchomien).")
    parser.add_argument("--split-seed", type=int, default=54217,
                        help="Stale ziarno podzialu trening/walidacja (domyslnie: 54217).")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Katalog wynikow; tryb quick domyslnie zapisuje w mnist/quick.")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "learning_rate", "eval_samples", "train_samples", "test_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} musi byc dodatnie.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Ziarna treningu musza byc rozne.")
    if any(seed < 0 or seed >= 2**32 for seed in [*args.seeds, args.split_seed]):
        parser.error("Ziarna musza nalezec do przedzialu [0, 2**32).")
    return args


def main():
    args = parse_args()

    data = load_mnist_data()
    rng = np.random.default_rng(args.split_seed)
    indices = rng.permutation(len(data.x_train))

    eval_count = 200 if args.quick else args.eval_samples
    if eval_count >= len(indices):
        raise ValueError("Validation split must leave at least one training sample.")
    eval_indices = indices[:eval_count]
    remaining_indices = indices[eval_count:]

    if args.quick:
        train_count = min(1000, len(remaining_indices))
        epochs = 1
    else:
        train_count = args.train_samples if args.train_samples is not None else len(remaining_indices)
        epochs = args.epochs

    if train_count > len(remaining_indices):
        raise ValueError("Requested training sample count exceeds the available split.")

    train_indices = remaining_indices[:train_count]

    x_eval = data.x_train[eval_indices]
    y_eval = data.y_train[eval_indices]

    x_train = data.x_train[train_indices]
    y_train = data.y_train[train_indices]

    if args.quick:
        test_count = 500
        x_test = data.x_test[:test_count]
        y_test = data.y_test[:test_count]
    elif args.test_samples is not None:
        if args.test_samples > len(data.x_test):
            raise ValueError("Requested test sample count exceeds the available test split.")
        x_test = data.x_test[:args.test_samples]
        y_test = data.y_test[:args.test_samples]
    else:
        x_test = data.x_test
        y_test = data.y_test

    target_dir = args.results_dir or Path(__file__).resolve().parent / "results" / "mnist"
    if args.quick and args.results_dir is None:
        target_dir /= "quick"
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Liczba probek: trening={len(x_train)}, walidacja={len(x_eval)}, test={len(x_test)}")
    summary = {
        "seeds": args.seeds, "split_seed": args.split_seed,
        "train_samples": len(x_train), "eval_samples": len(x_eval), "test_samples": len(x_test),
        "epochs": epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "layer_dims": [784, 150, 50, 10], "std_ddof": 0, "quick": args.quick,
        "representative_seed": args.seeds[0], "runs": [],
    }
    histories = []
    for seed in args.seeds:
        print(f"\nZiarno {seed} ({len(histories) + 1}/{len(args.seeds)})", flush=True)
        set_seed(seed)
        model = MLP.from_dimensions(
            layer_dims=[784, 150, 50, 10], hidden_activation=ReLU,
            output_activation=Softmax, use_bias=True, weight_init="he",
        )
        loss_fn = CategoricalCrossEntropyLoss()
        history = model.fit(
            x_train=x_train, y_train=y_train, loss_fn=loss_fn,
            optimizer=SGD(learning_rate=args.learning_rate), epochs=epochs,
            batch_size=args.batch_size, validation_data=(x_eval, y_eval),
            metrics={"accuracy": compute_multiclass_accuracy}, verbose=True,
        )
        curves = {key: history[key] for key in CURVE_KEYS}
        histories.append(curves)
        summary["runs"].append({
            "seed": seed, "history": curves,
            "test_loss": float(model.evaluate(x_test, y_test, loss_fn)),
            "test_accuracy": float(compute_multiclass_accuracy(model.predict(x_test), y_test)),
        })
        # Keep the original seed's test figures separate from the aggregate curves.
        if seed == args.seeds[0]:
            evaluate_and_plot_mnist(model, history, x_test, y_test, results_dir=target_dir)
        # Save progress after each completed seed, including raw curves for reproducibility.
        (target_dir / "mnist_training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )

    summary["mean"], summary["std"] = aggregate_histories(histories)
    (target_dir / "mnist_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    curves_path = target_dir / "mnist_learning_curves_mean_std.png"
    plot_training_and_validation_curves(
        summary["mean"], history_std=summary["std"], save_path=curves_path,
        title=f"MNIST: średnia z {len(histories)} ziaren ± odchylenie standardowe",
    )
    print(f"Zapisano usrednione krzywe uczenia: {curves_path}")


if __name__ == "__main__":
    main()
