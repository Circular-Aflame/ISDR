"""Schedule controls and compatibility; run with unittest discovery."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from isdr.methods import core


class RefinementScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X = rng.normal(size=(48, 6)).astype(np.float32)
        cls.scores = rng.normal(size=48)
        cls.Y = rng.normal(scale=0.1, size=(48, 2)).astype(np.float32)
        cls.P = core.compute_tsne_affinity_matrix(cls.X, perplexity=8, n_jobs=1)

    def refine(self, **kwargs):
        params = dict(
            alpha=0.4,
            perplexity=8,
            regloss_k=8,
            n_iter=20,
            lr=5,
            warmup_steps=5,
            knn_rebuild_interval=10,
            P=self.P,
            n_jobs=1,
            num_threads=1,
        )
        params.update(kwargs)
        # Exercise the NumPy fallback without requiring the optional JIT package.
        with patch.object(core, "nb", None):
            return core.run_isdr_refinement(self.X, self.scores, self.Y, **params)

    def test_standard_defaults_match_explicit_paper_schedule(self):
        implicit = self.refine(n_iter=750, warmup_steps=50)
        explicit = self.refine(
            n_iter=750,
            warmup_steps=50,
            early_exaggeration=12,
            early_exaggeration_iters=250,
            initial_momentum=0.5,
            final_momentum=0.8,
            momentum_switch_iter=250,
        )
        np.testing.assert_array_equal(implicit, explicit)

    def test_short_run_keeps_original_duration(self):
        np.testing.assert_array_equal(
            self.refine(),
            self.refine(early_exaggeration_iters=10, momentum_switch_iter=10),
        )

    def test_disabling_exaggeration_can_preserve_momentum_timing(self):
        disabled = self.refine(early_exaggeration_iters=0, momentum_switch_iter=10)
        neutral = self.refine(early_exaggeration=1, early_exaggeration_iters=10)
        np.testing.assert_array_equal(disabled, neutral)
        immediate_switch = self.refine(early_exaggeration_iters=0)
        self.assertFalse(np.array_equal(disabled, immediate_switch))

    def test_fixed_weight_at_alpha_one_skips_geometry_force(self):
        with patch("sklearn.manifold._barnes_hut_tsne.gradient") as gradient:
            result = self.refine(alpha=1, warmup_steps=0, early_exaggeration_iters=0)
        gradient.assert_not_called()
        self.assertTrue(np.isfinite(result).all())

    def test_profile_records_resolved_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            self.refine(profile_path=str(path))
            profile = json.loads(path.read_text())
        expected = dict(
            warmup_steps=5,
            early_exaggeration=12,
            early_exaggeration_iters=10,
            initial_momentum=0.5,
            final_momentum=0.8,
            momentum_switch_iter=10,
            knn_rebuild_interval=10,
            num_threads=1,
            lr=5,
        )
        for name, value in expected.items():
            self.assertEqual(profile[name], value, name)

    def test_invalid_schedules_are_rejected(self):
        cases = [
            ("early_exaggeration", 0),
            ("early_exaggeration", float("nan")),
            ("early_exaggeration_iters", -1),
            ("early_exaggeration_iters", 1.5),
            ("warmup_steps", -1),
            ("momentum_switch_iter", -1),
            ("initial_momentum", 1),
            ("final_momentum", float("inf")),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                self.refine(**{name: value})

    def test_method_forwards_controls_to_both_splits(self):
        params = dict(
            pca_dim=0,
            warmup_steps=0,
            early_exaggeration=1,
            early_exaggeration_iters=0,
            initial_momentum=0.3,
            final_momentum=0.7,
            momentum_switch_iter=12,
            num_threads=2,
            embed_test_separately=True,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("sklearn.manifold.TSNE") as tsne,
            patch.object(core, "run_isdr_refinement") as refine,
        ):
            tsne.return_value.fit_transform.side_effect = lambda X: np.zeros((len(X), 2))
            refine.side_effect = lambda X, scores, Y, **kw: Y
            result = core.isdr_method(
                self.X,
                self.scores,
                self.X[:12],
                self.scores[:12],
                params,
                tmp,
                42,
            )
        self.assertEqual(refine.call_count, 2)
        for call in refine.call_args_list:
            for name, value in params.items():
                if name not in {"pca_dim", "embed_test_separately"}:
                    self.assertEqual(call.kwargs[name], value, name)
        self.assertEqual(result["Z_train"].shape, (48, 2))
        self.assertEqual(result["Z_test"].shape, (12, 2))

    def test_test_labels_are_unused_by_default(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(core, "initial_layout", return_value=self.Y) as initialize,
            patch.object(core, "run_isdr_refinement", return_value=self.Y) as refine,
        ):
            result = core.isdr_method(self.X, self.scores, self.X[:12], self.scores[:12], {}, tmp, 42)
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(refine.call_count, 1)
        self.assertIsNone(result["Z_test"])


if __name__ == "__main__":
    unittest.main()
