#!/usr/bin/env python3
"""Plot regression or classification sweeps with the paper's renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isdr.data.regression import SPECS
from isdr.plotting.pareto import GEOMETRY_CONFIG, METHOD_STYLES, plot_panels


DATASET_LABELS = {key: spec.label for key, spec in SPECS.items()} | {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "organcmnist": "OrganCMNIST",
    "pathmnist": "PathMNIST",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/regression_benchmarks")
    parser.add_argument("--output-dir", default=None, help="Defaults to <results-root>/plots")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_STYLES))
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument("--geometry-metric", choices=list(GEOMETRY_CONFIG), default="trustworthiness")
    parser.add_argument("--preset", choices=["main", "all"], default="main")
    parser.add_argument("--de-scale", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf", "svg"])
    args = parser.parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir) if args.output_dir else root / "plots"
    datasets = args.datasets or sorted(p.name for p in root.glob("*") if (p / "aggregated_results.json").is_file())
    panels = []
    for dataset in datasets:
        folder = root / dataset
        if args.min_samples:
            info_path = next(folder.glob("seed_*/dataset_info.json"), None)
            if info_path:
                info = json.loads(info_path.read_text())
                count = info["n_train"] + info["n_test"]
            else:
                count = SPECS[dataset].n_samples if dataset in SPECS else 0
            if count < args.min_samples:
                continue
        results = json.loads((folder / "aggregated_results.json").read_text())
        if args.methods:
            results = {key: value for key, value in results.items() if key in args.methods}
        if not results:
            continue
        panels.append((DATASET_LABELS.get(dataset, dataset), results))
        plot_panels(
            [panels[-1]],
            output / dataset,
            args.geometry_metric,
            preset=args.preset,
            de_scale=args.de_scale,
            dpi=args.dpi,
            formats=args.formats,
        )
    if not panels:
        parser.error(f"No matching benchmark results under {root}")
    if len(panels) > 1:
        plot_panels(
            panels,
            output / "overview",
            args.geometry_metric,
            preset=args.preset,
            de_scale=args.de_scale,
            dpi=args.dpi,
            formats=args.formats,
        )
    print(f"Saved benchmark plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
