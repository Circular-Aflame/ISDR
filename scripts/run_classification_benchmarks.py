#!/usr/bin/env python3
"""Run scalar-label visualization diagnostics on the four public image datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from isdr.evaluation.regression import SEEDS, aggregate_dataset, method_grid
from isdr.evaluation.sweep import run_sweep


DATASETS = ["mnist", "fashion_mnist", "organcmnist", "pathmnist"]


def classification_config(dataset, seed, output_dir, data_root="data", n_samples=10000, quick=False):
    methods = method_grid(30, quick=quick)
    for method in methods:
        params = method["base_params"]
        params["pca_dim"] = 0 if method["name"] in {"rfphate", "rfae"} else 50
        if method["name"] in {"rfphate", "rfae"}:
            params["prediction_type"] = "classification"
        if method["name"] == "tpse":
            params["n_classes"] = 11 if dataset == "organcmnist" else 10
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "device": "cpu",
        "execution": {"mode": "subprocess", "resume": True, "timeout_seconds": 7200},
        "dataset": {
            "name": dataset,
            "data_root": data_root,
            "n_samples": n_samples,
            "data_seed": 42,
            "split_seed": 42,
            "test_ratio": 0.2,
        },
        "evaluation": {"knn_k": 10, "x_metric": "trustworthiness", "y_metric": "score_dirichlet_energy"},
        "methods": methods,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="results/classification_benchmarks")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for dataset in args.datasets:
        dataset_dir = Path(args.output_root).resolve() / dataset
        for seed in args.seeds:
            seed_dir = dataset_dir / f"seed_{seed}"
            config = classification_config(dataset, seed, seed_dir, args.data_root, args.n_samples, args.quick)
            if args.methods:
                available = {m["name"] for m in config["methods"]}
                if set(args.methods) - available:
                    parser.error(f"Unknown methods: {set(args.methods) - available}")
                config["methods"] = [m for m in config["methods"] if m["name"] in args.methods]
            config["device"] = args.device
            config["execution"]["resume"] = not args.force
            seed_dir.mkdir(parents=True, exist_ok=True)
            config_path = seed_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            run_sweep(config_path)
        aggregate_dataset(dataset_dir, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
