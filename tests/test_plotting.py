"""Unit tests for visualization utilities in experiments.common.plotting."""

from pathlib import Path
import tempfile
import unittest

from experiments.common.data_loader import load_classification_data, load_regression_data
from experiments.common.plotting import (
    plot_classification_dataset,
    plot_regression_approximations_grid,
    plot_regression_dataset,
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

