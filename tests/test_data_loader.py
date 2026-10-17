"""Unit tests for experiments.common.data_loader."""

import unittest
import numpy as np

from experiments.common.data_loader import (
    StandardScaler,
    to_one_hot,
    load_regression_data,
    load_classification_data,
    RegressionDataset,
    ClassificationDataset,
)


class TestStandardScaler(unittest.TestCase):
    def test_fit_transform_inverse_transform(self):
        data = np.array([[10.0, 200.0], [20.0, 400.0], [30.0, 600.0]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # Check zero mean and unit variance
        np.testing.assert_allclose(np.mean(scaled, axis=0), [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(np.std(scaled, axis=0), [1.0, 1.0], atol=1e-7)

        # Check inverse transform recovers original
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, data, atol=1e-7)

    def test_constant_feature(self):
        data = np.array([[5.0], [5.0], [5.0]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        # Difference from mean is 0, std is 0 -> scale_ fallback 1.0 -> 0.0
        np.testing.assert_allclose(scaled, [[0.0], [0.0], [0.0]], atol=1e-7)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, data, atol=1e-7)


class TestToOneHot(unittest.TestCase):
    def test_to_one_hot_basic(self):
        labels = np.array([0, 1, 2, 0])
        one_hot = to_one_hot(labels, num_classes=3)
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(one_hot, expected)

    def test_to_one_hot_2d_input(self):
        labels = np.array([[1], [0], [1]])
        one_hot = to_one_hot(labels)
        expected = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        np.testing.assert_allclose(one_hot, expected)


class TestRegressionDataLoader(unittest.TestCase):
    def test_load_activation_with_standardization(self):
        ds = load_regression_data("activation", n_samples=100, standardize=True)
        self.assertIsInstance(ds, RegressionDataset)
        self.assertEqual(ds.x_train.shape, (100, 1))
        self.assertEqual(ds.y_train.shape, (100, 1))
        self.assertEqual(ds.x_test.shape[1], 1)
        self.assertEqual(ds.y_test.shape[1], 1)

        # Check standardized stats on train
        np.testing.assert_allclose(np.mean(ds.x_train), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.std(ds.x_train), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.mean(ds.y_train), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.std(ds.y_train), 1.0, atol=1e-6)

        # Test unpacking
        x_tr, y_tr, x_te, y_te = ds
        self.assertEqual(x_tr.shape, (100, 1))
        self.assertEqual(y_tr.shape, (100, 1))

        # Test inverse scaling on y
        self.assertIsNotNone(ds.scaler_y)
        unscaled_y_tr = ds.scaler_y.inverse_transform(y_tr)
        # Activation y range is [-265, -2]
        self.assertTrue(np.all(unscaled_y_tr < 0.0))

    def test_load_cube_without_standardization(self):
        ds = load_regression_data("cube", n_samples=500, standardize=False)
        self.assertEqual(ds.x_train.shape, (500, 1))
        self.assertEqual(ds.y_train.shape, (500, 1))
        self.assertIsNone(ds.scaler_x)
        self.assertIsNone(ds.scaler_y)


class TestClassificationDataLoader(unittest.TestCase):
    def test_load_simple_auto_binary(self):
        ds = load_classification_data("simple", n_samples=100, target_encoding="auto")
        self.assertIsInstance(ds, ClassificationDataset)
        self.assertEqual(ds.x_train.shape, (100, 2))
        self.assertEqual(ds.y_train.shape, (100, 1))
        self.assertEqual(ds.num_classes, 2)

        # Labels are 0 and 1
        unique_labels = np.unique(ds.y_train_labels)
        np.testing.assert_array_equal(unique_labels, [0, 1])

        # Targets are in {0.0, 1.0}
        self.assertTrue(set(np.unique(ds.y_train)).issubset({0.0, 1.0}))

        # Unpacking test
        x_tr, y_tr, x_te, y_te = ds
        self.assertEqual(x_tr.shape, (100, 2))

    def test_load_simple_one_hot(self):
        ds = load_classification_data("simple", n_samples=100, target_encoding="one_hot")
        self.assertEqual(ds.y_train.shape, (100, 2))
        np.testing.assert_allclose(np.sum(ds.y_train, axis=1), np.ones(100))

    def test_load_three_gauss_auto_multiclass(self):
        ds = load_classification_data("three_gauss", n_samples=500, target_encoding="auto")
        # In three_gauss, N=500 means 500 samples per class -> 1500 total
        self.assertEqual(ds.x_train.shape, (1500, 2))
        self.assertEqual(ds.y_train.shape, (1500, 3))
        self.assertEqual(ds.num_classes, 3)

        # Labels 0, 1, 2
        unique_labels = np.unique(ds.y_train_labels)
        np.testing.assert_array_equal(unique_labels, [0, 1, 2])

        # One hot rows sum to 1
        np.testing.assert_allclose(np.sum(ds.y_train, axis=1), np.ones(1500))

    def test_load_three_gauss_indices(self):
        ds = load_classification_data("three_gauss", n_samples=100, target_encoding="indices")
        # 3 classes * 100 = 300 rows
        self.assertEqual(ds.y_train.shape, (300, 1))
        self.assertTrue(set(np.unique(ds.y_train)).issubset({0, 1, 2}))

    def test_invalid_parameters_raise_value_errors(self):
        with self.assertRaises(ValueError):
            load_regression_data("unknown_dataset", n_samples=100)
        with self.assertRaises(ValueError):
            load_regression_data("activation", n_samples=999)
        with self.assertRaises(ValueError):
            load_classification_data("unknown_dataset", n_samples=100)
        with self.assertRaises(ValueError):
            load_classification_data("simple", n_samples=100, target_encoding="invalid")
        with self.assertRaises(ValueError):
            load_classification_data("three_gauss", n_samples=100, target_encoding="binary")


if __name__ == "__main__":
    unittest.main()
