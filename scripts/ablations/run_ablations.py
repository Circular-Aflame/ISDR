#!/usr/bin/env python3
"""Run initialization, optimizer, and support ablations on any public config."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from isdr.evaluation.regression import SEEDS, aggregate_seed_results
from isdr.evaluation.sweep import run_sweep
from isdr.plotting import plot_pareto


GROUPS = ["initializer", "raw-initializer", "schedule", "support", "common-start"]


def variants(group, alpha):
    fixed = {"warmup_steps": 0, "early_exaggeration": 1.0, "early_exaggeration_iters": 0, "momentum_switch_iter": 0}
    if group == "initializer":
        return [(name, "isdr", {"init": name}) for name in ("tsne", "pca", "random", "umap")]
    if group == "raw-initializer":
        return [
            (
                f"{family}_{stage}",
                "isdr",
                {**fixed, "init": "tsne" if stage == "matched" else family, "tsne_init": family},
            )
            for family in ("pca", "random", "umap")
            for stage in ("matched", "bypass")
        ]
    if group == "schedule":
        return [
            ("standard", "isdr", {}),
            ("no_weight_warmup", "isdr", {"warmup_steps": 0}),
            ("fixed_objective", "isdr", fixed),
            ("geometry_only", "isdr", {"alpha": 0.0}),
            ("readout_only", "isdr", {**fixed, "alpha": 1.0}),
        ]
    if group == "support":
        return [(f"k{k}", "isdr", {"regloss_k": k}) for k in (30, 60, 90, 150)] + [
            (f"R{r}", "isdr", {"knn_rebuild_interval": r}) for r in (25, 50, 100, 250)
        ]
    if group == "common-start":
        return [
            ("isdr", "isdr", {}),
            (
                "sumap",
                "sumap",
                {"init": "tsne", "target_metric": "l2", "target_weight": alpha, "n_neighbors": 15, "min_dist": 0.1},
            ),
        ]
    raise ValueError(group)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablations/digits.yaml")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--output-root", default="results/ablations")
    parser.add_argument("--quick", action="store_true", help="350 initializer and 80 refinement steps")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    template = next((m for m in config["methods"] if m["name"] == "isdr"), {})
    base = {
        "alpha": args.alpha,
        "perplexity": 30,
        "tsne_iter": 1000,
        "n_iter": 750,
        "lr": 100,
        "warmup_steps": 50,
        "early_exaggeration_iters": 250,
        "momentum_switch_iter": 250,
        "knn_rebuild_interval": 50,
        **template.get("base_params", {}),
    }
    if args.quick:
        base.update(tsne_iter=350, n_iter=80, warmup_steps=20, early_exaggeration_iters=40, momentum_switch_iter=40)
    base["alpha"] = args.alpha
    output = Path(args.output_root).resolve()
    for group in args.groups:
        seed_results = []
        for seed in args.seeds:
            combined = {}
            for variant, method, overrides in variants(group, args.alpha):
                point_config = deepcopy(config)
                params = {**base, **overrides}
                if method == "sumap":
                    params = {
                        k: v
                        for k, v in params.items()
                        if k
                        in {
                            "init",
                            "tsne_init",
                            "tsne_iter",
                            "perplexity",
                            "pca_dim",
                            "n_jobs",
                            "target_metric",
                            "target_weight",
                            "n_neighbors",
                            "min_dist",
                        }
                    }
                point_config.update(seed=seed, output_dir=str(output / group / f"seed_{seed}" / variant))
                point_config["execution"] = {**point_config.get("execution", {}), "resume": not args.force}
                # Both methods consume the same standardized rows and t-SNE parameters.
                point_config["methods"] = [{"name": method, "base_params": params}]
                folder = Path(point_config["output_dir"])
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "config.yaml"
                path.write_text(yaml.safe_dump(point_config, sort_keys=False))
                result = run_sweep(path)
                combined[variant] = result[method]
            seed_results.append(combined)
        summary = aggregate_seed_results(seed_results)
        (output / group / "aggregated_results.json").write_text(json.dumps(summary, indent=2) + "\n")
        means = {
            name: {
                key: {**entry, "train_metrics": {k: v["mean"] for k, v in entry["train_metrics"].items()}}
                for key, entry in points.items()
            }
            for name, points in summary.items()
        }
        plot_pareto(means, output / group / "comparison.png", "trustworthiness", "score_dirichlet_energy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
