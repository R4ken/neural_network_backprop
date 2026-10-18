"""Unit tests for visualization utilities in experiments.common.plotting."""

from pathlib import Path
import tempfile
import unittest

from experiments.common.data_loader import load_classification_data
from experiments.common.plotting import plot_classification_dataset


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

