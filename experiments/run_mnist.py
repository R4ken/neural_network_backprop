import sys
from pathlib import Path

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
import numpy as np


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
        "csv_path": csv_path
    }


def main():
    set_seed(54217)

    data = load_mnist_data()
    rng = np.random.default_rng(54217)
    indices = rng.permutation(len(data.x_train))

    eval_indices = indices[:5000]
    train_indices = indices[5000:]

    x_eval = data.x_train[eval_indices]
    y_eval = data.y_train[eval_indices]

    x_train = data.x_train[train_indices]
    y_train = data.y_train[train_indices]

    model = MLP.from_dimensions (
        layer_dims=[784, 150, 50, 10],
        hidden_activation=ReLU,
        output_activation=Softmax,
        use_bias=True,
        weight_init="he"
    )

    loss_fn = CategoricalCrossEntropyLoss()
    optimizer = SGD(learning_rate=0.05)

    history = model.fit(
        x_train=x_train,
        y_train=y_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=30,
        batch_size=32,
        validation_data=(x_eval, y_eval),
        metrics={"accuracy": compute_multiclass_accuracy},
        verbose=True
    )

    test_loss = model.evaluate(data.x_test, data.y_test, loss_fn)
    print(f"Loss: {test_loss:.5f}")

    probabilities = model.predict(data.x_test)
    test_accuracy = compute_multiclass_accuracy(
        probabilities,
        data.y_test
    )

    print(f"Test dataset accuracy: {test_accuracy:.3f}")

    evaluate_and_plot_mnist(
        model=model,
        history=history,
        x_test=data.x_test,
        y_test=data.y_test,
    )


if __name__ == "__main__":
    main()