"""Numerical checks against independent definitions and sklearn."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.manifold import trustworthiness

from isdr.evaluation import metrics
from isdr.methods import core


class MetricTests(unittest.TestCase):
    def test_exact_rank_metrics_match_sklearn(self):
        rng = np.random.default_rng(19)
        X, Z = rng.normal(size=(97, 8)), rng.normal(size=(97, 2))
        self.assertAlmostEqual(metrics.trustworthiness(Z, X, 7), trustworthiness(X, Z, n_neighbors=7))
        self.assertAlmostEqual(metrics.continuity(Z, X, 7), trustworthiness(Z, X, n_neighbors=7))

    def test_tied_ranks_match_sklearn(self):
        rng = np.random.default_rng(17)
        X = rng.integers(0, 3, size=(80, 4)).astype(np.float64)
        Z = rng.normal(size=(80, 2))
        self.assertAlmostEqual(metrics.trustworthiness(Z, X, 7), trustworthiness(X, Z, n_neighbors=7))

    def test_neighbors_exclude_self_at_coincident_points(self):
        Z = np.array([[0, 0], [0, 0], [1, 1], [2, 2]], dtype=np.float32)
        for indices in (metrics._neighbors(Z, 99), core._build_knn_graph(Z, 99, n_jobs=1)):
            self.assertEqual(indices.shape, (4, 3))
            for i, row in enumerate(indices):
                self.assertNotIn(i, row)
                self.assertEqual(set(row), set(range(4)) - {i})

    def test_categorical_purity_is_invariant_to_label_encoding(self):
        rng = np.random.default_rng(7)
        Z = rng.normal(size=(40, 2))
        labels = np.arange(40) % 4
        encoded = np.array([100, -5, 11, 0])[labels]
        a = metrics.compute_all_metrics(Z, Z, labels, score_type="discrete")
        b = metrics.compute_all_metrics(Z, Z, encoded, score_type="discrete")
        self.assertEqual(a["label_neighbor_purity"], b["label_neighbor_purity"])

    def test_nonfinite_and_collapsed_layouts_fail(self):
        for Z in (np.zeros((12, 2)), np.full((12, 2), np.nan)):
            with self.assertRaises(ValueError):
                metrics.compute_all_metrics(Z, np.ones((12, 4)), np.arange(12))

    def test_column_signals_match_vector_metrics(self):
        rng = np.random.default_rng(12)
        Z = rng.normal(size=(30, 2)).astype(np.float32)
        X = rng.normal(size=(30, 5)).astype(np.float32)
        scores = rng.normal(size=30).astype(np.float32)
        vector = metrics.compute_all_metrics(Z, X, scores)
        column = metrics.compute_all_metrics(Z, X, scores[:, None])
        for name in vector:
            np.testing.assert_equal(vector[name], column[name], err_msg=name)
        self.assertEqual(
            metrics.local_regression_mae(Z, scores), metrics.local_regression_mae(Z, scores[:, None])
        )
        self.assertEqual(
            metrics.local_regression_rmse(Z, scores), metrics.local_regression_rmse(Z, scores[:, None])
        )

    def test_invalid_feature_or_signal_rows_fail(self):
        Z = np.arange(24).reshape(12, 2)
        X = np.ones((12, 4))
        for features, scores in [
            (X, np.arange(11)),
            (X, np.full(12, np.nan)),
            (X * np.inf, np.arange(12)),
            (X, np.arange(12).reshape(3, 4)),
        ]:
            with self.subTest(shape=np.shape(scores)), self.assertRaises(ValueError):
                metrics.compute_all_metrics(Z, features, scores)


class GradientTests(unittest.TestCase):
    def test_student_t_gradient_by_finite_differences(self):
        rng = np.random.default_rng(10)
        Z, scores = rng.normal(size=(9, 2)), rng.normal(size=9)
        neighbors = core._build_knn_graph(Z, 5, n_jobs=1)
        with patch.object(core, "nb", None):
            loss, gradient = core.student_t_regloss_and_grad(Z, scores, neighbors)
            numerical = np.zeros_like(Z)
            for row, dim in np.ndindex(Z.shape):
                plus, minus = Z.copy(), Z.copy()
                plus[row, dim] += 1e-5
                minus[row, dim] -= 1e-5
                numerical[row, dim] = (
                    core.student_t_regloss_and_grad(plus, scores, neighbors)[0]
                    - core.student_t_regloss_and_grad(minus, scores, neighbors)[0]
                ) / 2e-5
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-7)

    @unittest.skipIf(core.nb is None, "Numba extra not installed")
    def test_accelerated_gradient_matches_numpy(self):
        rng = np.random.default_rng(8)
        Z, scores = rng.normal(size=(40, 2)), rng.normal(size=40)
        neighbors = core._build_knn_graph(Z, 7, n_jobs=1)
        loss, grad = core.student_t_regloss_and_grad(Z, scores, neighbors)
        with patch.object(core, "nb", None):
            reference_loss, reference_grad = core.student_t_regloss_and_grad(Z, scores, neighbors)
        self.assertAlmostEqual(loss, reference_loss)
        np.testing.assert_allclose(grad, reference_grad, rtol=1e-5, atol=1e-7)
