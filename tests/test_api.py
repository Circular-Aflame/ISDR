"""The public array API must use the benchmark optimizer without writing artifacts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from isdr import fit_transform
from isdr.methods import core


class ArrayAPITests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(36, 6)).astype(np.float32)
        self.scores = rng.normal(size=36)
        self.params = dict(
            init="random",
            pca_dim=0,
            perplexity=8,
            regloss_k=8,
            n_iter=15,
            lr=5,
            n_jobs=1,
            num_threads=1,
        )

    def test_matches_benchmark_path_without_files_or_input_changes(self):
        X_before, scores_before = self.X.copy(), self.scores.copy()
        with tempfile.TemporaryDirectory() as tmp, patch.object(core, "nb", None):
            expected = core.isdr_method(
                self.X,
                self.scores,
                None,
                None,
                {**self.params, "alpha": 0.4},
                tmp,
                42,
            )["Z_train"]
            self.assertTrue((Path(tmp) / "Y_init.npy").is_file())
            self.assertTrue((Path(tmp) / "profile.json").is_file())
            with patch.object(core.np, "save", wraps=np.save) as save:
                actual = fit_transform(self.X, self.scores, alpha=0.4, **self.params)
            save.assert_not_called()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.X, X_before)
        np.testing.assert_array_equal(self.scores, scores_before)
        self.assertEqual(actual.shape, (len(self.X), 2))
        self.assertEqual(actual.dtype, np.float32)

    def test_unknown_options_fail_before_initialization(self):
        for option in ("n_itter", "output_dir", "profile_path", "embed_test_separately"):
            with (
                self.subTest(option=option),
                patch.object(core, "initial_layout") as initialize,
                self.assertRaisesRegex(TypeError, option),
            ):
                fit_transform(self.X, self.scores, **{option: 1})
            initialize.assert_not_called()

    def test_invalid_arrays_and_weights_fail_before_initialization(self):
        for X, scores, alpha in [
            (self.X[:2], self.scores[:2], 0.4),
            (self.X, self.scores[:-1], 0.4),
            (self.X * np.nan, self.scores, 0.4),
            (self.X, np.full_like(self.scores, np.inf), 0.4),
            (self.X, self.scores, -0.1),
            (self.X, self.scores, 1.1),
        ]:
            with (
                self.subTest(shape=X.shape, alpha=alpha),
                patch.object(core, "initial_layout") as initialize,
                self.assertRaises(ValueError),
            ):
                fit_transform(X, scores, alpha=alpha)
            initialize.assert_not_called()
