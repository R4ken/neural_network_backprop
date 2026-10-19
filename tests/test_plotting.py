"""Unit tests for visualization utilities in experiments.common.plotting."""

from pathlib import Path
import tempfile
import unittest

from experiments.common.data_loader import load_classification_data, load_regression_data
from experiments.common.plotting import (
    plot_architecture_regression_bars,
    plot_classification_dataset,
    plot_r3_loss_curves_split,
    plot_r3_metric_curves,
    plot_regression_approximations_grid,
    plot_regression_dataset,
    plot_sample_size_generalization_curve,
)


class _DummyRegressionModel:
    """Minimal model stub exposing the predict() interface used by plotting."""

    def predict(self, inputs):
        return 2.0 * inputs + 1.0


class TestPlotting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_classification_dataset_custom_path(self):
        ds = load_classification_data("simple", n_samples=100)
        target_path = self.output_dir / "custom_simple.png"

        saved_path = plot_classification_dataset(ds, save_path=target_path)

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_classification_dataset_default_path(self):
        ds = load_classification_data("three_gauss", n_samples=100)

        saved_path = plot_classification_dataset(ds)

        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)
        self.assertIn("three_gauss", saved_path.name)

    def test_plot_regression_dataset_train_and_test(self):
        ds = load_regression_data("activation", n_samples=100)
        target_path = self.output_dir / "regression_dataset.png"

        saved_path = plot_regression_dataset(ds, save_path=target_path)

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_regression_approximations_grid(self):
        ds = load_regression_data("cube", n_samples=100)
        models = {
            "Sigmoid": _DummyRegressionModel(),
            "Tanh": _DummyRegressionModel(),
            "ReLU": _DummyRegressionModel(),
        }

        for split in ("train", "test"):
            target_path = self.output_dir / f"approximation_{split}.png"
            saved_path = plot_regression_approximations_grid(
                models=models,
                dataset=ds,
                save_path=target_path,
                split=split,
                seed=42,
            )

            self.assertEqual(saved_path, target_path)
            self.assertTrue(saved_path.is_file())
            self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_regression_approximations_grid_invalid_split(self):
        ds = load_regression_data("activation", n_samples=100)
        with self.assertRaises(ValueError):
            plot_regression_approximations_grid(
                models={"Tanh": _DummyRegressionModel()},
                dataset=ds,
                save_path=self.output_dir / "invalid.png",
                split="validation",
            )

    def test_plot_regression_approximations_grid_custom_layout(self):
        ds = load_regression_data("activation", n_samples=100)
        models = {f"Model {i}": _DummyRegressionModel() for i in range(4)}
        target_path = self.output_dir / "approximation_grid_2x2.png"

        saved_path = plot_regression_approximations_grid(
            models=models,
            dataset=ds,
            save_path=target_path,
            split="test",
            nrows=2,
            ncols=2,
            title="Testowa siatka aproksymacji",
        )

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_architecture_regression_bars(self):
        target_path = self.output_dir / "regression_bars.png"

        saved_path = plot_architecture_regression_bars(
            arch_labels=["A", "B"],
            train_values=[1.0, 2.0],
            train_stds=[0.1, 0.2],
            test_values=[10.0, 20.0],
            test_stds=[1.0, 2.0],
            title="Testowe słupki",
            save_path=target_path,
        )

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_r3_metric_curves(self):
        epochs = [1, 2, 3]
        runs = [[1.0, 0.8, 0.6], [1.1, 0.9, 0.7]]
        metric_by_lr = {
            lr: {
                "MSE": {"train": runs, "test": runs},
                "MAE": {"train": runs, "test": runs},
            }
            for lr in (0.001, 0.005, 0.01)
        }
        target_path = self.output_dir / "r3_metric_curves.png"

        saved_path = plot_r3_metric_curves(
            epochs=epochs,
            metric_by_lr=metric_by_lr,
            metric_label="MAE",
            title="Testowe krzywe metryki",
            save_path=target_path,
        )

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_r3_loss_curves_split(self):
        epochs = [1, 2, 3]
        runs = [[1.0, 0.8, 0.6], [1.1, 0.9, 0.7]]
        loss_by_fn = {
            fn_name: {
                lr: {"train": runs, "test": runs}
                for lr in (0.001, 0.005, 0.01)
            }
            for fn_name in ("MSE", "MAE")
        }
        target_path = self.output_dir / "r3_loss_curves.png"

        saved_path = plot_r3_loss_curves_split(
            epochs=epochs,
            loss_by_fn=loss_by_fn,
            title="Testowe krzywe strat",
            save_path=target_path,
        )

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

    def test_plot_sample_size_generalization_curve(self):
        target_path = self.output_dir / "generalization_curve.png"

        saved_path = plot_sample_size_generalization_curve(
            sample_sizes=[100, 500, 1000],
            train_means=[10.0, 5.0, 2.0],
            train_stds=[1.0, 0.5, 0.2],
            test_means=[1000.0, 500.0, 200.0],
            test_stds=[100.0, 50.0, 20.0],
            title="Testowa krzywa generalizacji",
            save_path=target_path,
        )

        self.assertEqual(saved_path, target_path)
        self.assertTrue(saved_path.is_file())
        self.assertGreater(saved_path.stat().st_size, 0)

