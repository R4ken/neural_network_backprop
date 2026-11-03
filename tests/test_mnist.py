"""Tests for multi-seed MNIST learning curves and experiment configuration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_mnist import CURVE_KEYS, aggregate_histories, parse_args
from experiments.common.plotting import plot_training_and_validation_curves


class TestMnistAggregation(unittest.TestCase):
    def test_mean_and_population_std_across_ten_seeds(self):
        histories = [{key: [float(i), float(i + 1)] for key in CURVE_KEYS}
                     for i in range(10)]
        mean, std = aggregate_histories(histories)
        for key in CURVE_KEYS:
            np.testing.assert_allclose(mean[key], [4.5, 5.5])
            np.testing.assert_allclose(std[key], [np.sqrt(8.25)] * 2)

    def test_single_seed_has_zero_std(self):
        history = {key: [0.5, 0.75] for key in CURVE_KEYS}
        mean, std = aggregate_histories([history])
        self.assertEqual(mean, history)
        self.assertTrue(all(values == [0.0, 0.0] for values in std.values()))

    def test_invalid_histories(self):
        for histories in ([], [{key: [] for key in CURVE_KEYS}],
                          [{key: [float("nan")] for key in CURVE_KEYS}]):
            with self.assertRaises(ValueError):
                aggregate_histories(histories)

    def test_different_epoch_counts(self):
        with self.assertRaises(ValueError):
            aggregate_histories([{key: [1.0] for key in CURVE_KEYS},
                                 {key: [1.0, 2.0] for key in CURVE_KEYS}])

    def test_default_seeds_and_full_test_split(self):
        args = parse_args([])
        self.assertEqual(args.seeds, list(range(54217, 54227)))
        self.assertEqual(args.split_seed, 54217)
        self.assertIsNone(args.test_samples)

    def test_invalid_cli_values(self):
        for argv in (["--seeds", "1", "1"], ["--epochs", "0"],
                     ["--test-samples", "-1"], ["--seeds", "-1"]):
            with self.subTest(argv=argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_plot_bands_and_accuracy_percentage_scaling(self):
        mean = {key: [0.5, 0.75] for key in CURVE_KEYS}
        std = {key: [0.1, 0.05] for key in CURVE_KEYS}
        with tempfile.TemporaryDirectory() as directory, patch.object(plt, "close"):
            path = plot_training_and_validation_curves(
                mean, history_std=std, save_path=Path(directory) / "curves.png",
            )
            self.assertTrue(path.is_file())
            figure = plt.gcf()
            for ax in figure.axes:
                self.assertEqual(len(ax.collections), 2)
            np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [50, 75])
            vertices = figure.axes[1].collections[0].get_paths()[0].vertices
            self.assertAlmostEqual(vertices[:, 1].min(), 40.0)
            self.assertAlmostEqual(vertices[:, 1].max(), 80.0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
