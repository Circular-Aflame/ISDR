#!/usr/bin/env python3
"""Run and aggregate public tabular regression benchmark sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from isdr.data import build_regression_benchmark, regression_dataset_keys
from isdr.data.regression import SPECS
from isdr.evaluation.regression import SEEDS, aggregate_dataset, benchmark_config, dataset_order
from isdr.evaluation.sweep import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", *regression_dataset_keys()])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-root", default="results/regression_benchmarks")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true", help="Run a short PCA/t-SNE/ISDR sanity grid")
    parser.add_argument("--methods", nargs="+", default=None, help="Optional method-name filter, e.g. --methods isdr")
    parser.add_argument(
        "--min-samples", type=int, default=0, help="Skip datasets with fewer samples than this threshold"
    )
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--execution-mode",
        default="subprocess",
        choices=["subprocess", "in-process"],
        help="How to execute each method point",
    )
    parser.add_argument(
        "--point-timeout-seconds", type=int, default=7200, help="Per-method-point timeout; use 0 to disable"
    )
    return parser.parse_args()


def filter_methods(methods: list[dict], selected: list[str] | None) -> list[dict]:
    if not selected:
        return methods
    allowed = {name.lower() for name in selected}
    unknown = allowed - {method["name"] for method in methods}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return [method for method in methods if str(method.get("name", "")).lower() in allowed]


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    datasets = dataset_order(args.datasets)
    if args.min_samples > 0:
        skipped = [dataset for dataset in datasets if SPECS[dataset].n_samples < args.min_samples]
        datasets = [dataset for dataset in datasets if SPECS[dataset].n_samples >= args.min_samples]
        for dataset in skipped:
            print(f"[skip-dataset] {dataset}: n={SPECS[dataset].n_samples} < min_samples={args.min_samples}")
    if not datasets:
        raise ValueError("No datasets remain after filtering")
    if args.build_data:
        for dataset in datasets:
            build_regression_benchmark(dataset, data_root=args.data_root, force=False)

    failures = []
    for dataset in datasets:
        dataset_dir = root / args.output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            seed_dir = dataset_dir / f"seed_{seed}"
            config = benchmark_config(
                dataset,
                seed,
                output_dir=str(Path(args.output_root) / dataset / f"seed_{seed}"),
                data_root=args.data_root,
                quick=args.quick,
                device=args.device,
            )
            config["methods"] = filter_methods(config["methods"], args.methods)
            config["execution"] = {
                "mode": args.execution_mode.replace("-", "_"),
                "timeout_seconds": int(args.point_timeout_seconds),
                "resume": not args.force,
            }
            if not config["methods"]:
                raise ValueError(f"No methods remain after filtering with --methods {args.methods}")
            seed_dir.mkdir(parents=True, exist_ok=True)
            config_path = seed_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"[run] {dataset} seed={seed} quick={args.quick}")
            try:
                run_sweep(config_path)
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": seed, "error": repr(exc)})
                print(f"[error] {dataset} seed={seed}: {exc!r}")
                if not args.continue_on_error:
                    raise
        try:
            aggregate_dataset(dataset_dir, list(args.seeds))
        except FileNotFoundError as exc:
            failures.append({"dataset": dataset, "error": repr(exc)})

    if failures:
        failure_path = root / args.output_root / "failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
