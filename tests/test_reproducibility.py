"""Check fixed data sampling and cache invalidation across protocol changes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from isdr.data.loaders import DatasetBundle, load_dataset, split_data
from isdr.evaluation.sweep import run_sweep


class ReproducibilityTests(unittest.TestCase):
    def test_method_seed_does_not_change_sample(self):
        config = {"name": "digits", "n_samples": 40, "data_seed": 42}
        first, second = load_dataset(config, 42), load_dataset(config, 123)
        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.scores, second.scores)
        changed = load_dataset({**config, "data_seed": 7}, 42)
        self.assertFalse(np.array_equal(first.X, changed.X))

    def test_scaling_uses_training_statistics(self):
        X = np.arange(60, dtype=np.float32).reshape(20, 3) ** 2
        split = split_data(X, np.arange(20), 0.2, 42)
        train = X[split["train_idx"]]
        expected = (X[split["test_idx"]] - train.mean(axis=0)) / (train.std(axis=0) + 1e-6)
        np.testing.assert_allclose(split["X_test"], expected, rtol=1e-6)

    def test_resume_requires_matching_data_seed_params_and_metrics(self):
        rng = np.random.default_rng(42)
        bundle = DatasetBundle(rng.normal(size=(30, 4)), rng.normal(size=30), "fixture", "continuous", {})
        calls = []

        def method(X, scores, *args, **kwargs):
            calls.append(kwargs["seed"])
            return {"Z_train": X[:, :2], "Z_test": None}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("isdr.evaluation.sweep.get_method", return_value=method),
            patch("isdr.evaluation.sweep.load_dataset", return_value=bundle),
        ):
            path = Path(tmp) / "config.yaml"
            config = {
                "seed": 42,
                "output_dir": str(Path(tmp) / "run"),
                "execution": {"mode": "in_process"},
                "dataset": {},
                "methods": [{"name": "pca"}],
            }

            def run():
                path.write_text(yaml.safe_dump(config))
                return run_sweep(path)

            first = run()
            self.assertEqual(first, run())
            self.assertEqual(len(calls), 1)
            config["seed"] = 123
            run()
            config["methods"][0]["base_params"] = {"new_param": 1}
            run()
            config["evaluation"] = {"knn_k": 3}
            run()
            bundle.X[0, 0] += 10
            run()
            self.assertEqual(len(calls), 5)
            config["methods"] = []
            self.assertEqual(run(), {})
            self.assertEqual(json.loads((Path(tmp) / "run/sweep_results.json").read_text()), {})

    def test_failed_rerun_cannot_reuse_previous_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "output_dir": str(Path(tmp) / "run"),
                "dataset": {"name": "digits", "n_samples": 30},
                "methods": [{"name": "pca"}],
                "execution": {"mode": "in_process"},
            }
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            run_sweep(path)
            config["execution"]["resume"] = False
            path.write_text(yaml.safe_dump(config))
            with patch("isdr.evaluation.sweep.get_method", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    run_sweep(path)
            error = Path(tmp) / "run/pca/default/error.json"
            self.assertIn("failed", error.read_text())
            self.assertEqual(json.loads((Path(tmp) / "run/sweep_results.json").read_text()), {})
