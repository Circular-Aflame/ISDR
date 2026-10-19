from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isdr.data.regression import SPECS, regression_dataset_keys


METHOD_LABELS = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "pacmap": "PaCMAP",
    "sumap": "S-UMAP",
    "cebra": "CEBRA",
    "rfphate": "RF-PHATE",
    "rfae": "RF-AE",
    "akrmap": "AKRMap",
    "tpse": "TP-SE",
    "isdr": r"\textbf{\sys}",
}

METHOD_ORDER = ["pca", "tsne", "umap", "pacmap", "sumap", "cebra", "tpse", "rfphate", "rfae", "akrmap", "isdr"]
ALPHA_METHODS = {"sumap", "akrmap", "tpse", "isdr"}
SEEDS = [42, 123, 456, 789, 2026]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def method_grid(perplexity: int, quick: bool = False) -> list[dict]:
    if quick:
        return [
            {"name": "pca", "param_name": "_dummy", "values": [0], "base_params": {}},
            {
                "name": "tsne",
                "param_name": "_dummy",
                "values": [0],
                "base_params": {"perplexity": perplexity, "n_iter": 350, "pca_dim": 0},
            },
            {
                "name": "isdr",
                "param_name": "alpha",
                "values": [0.0, 0.4],
                "base_params": {
                    "perplexity": perplexity,
                    "tsne_iter": 350,
                    "n_iter": 20,
                    "warmup_steps": 5,
                    "knn_rebuild_interval": 10,
                    "regloss_k": 3 * perplexity,
                    "pca_dim": 0,
                },
            },
        ]
    return [
        {"name": "pca", "param_name": "_dummy", "values": [0], "base_params": {}},
        {
            "name": "tsne",
            "param_name": "_dummy",
            "values": [0],
            "base_params": {"perplexity": perplexity, "n_iter": 1000, "pca_dim": 0},
        },
        {
            "name": "umap",
            "param_name": "_dummy",
            "values": [0],
            "base_params": {"n_neighbors": 15, "min_dist": 0.1, "pca_dim": 0},
        },
        {"name": "pacmap", "param_name": "_dummy", "values": [0], "base_params": {"n_neighbors": 10, "pca_dim": 0}},
        {
            "name": "sumap",
            "param_name": "target_weight",
            "values": ALPHAS,
            "base_params": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "target_metric": "l2",
                "pca_dim": 0,
            },
        },
        {
            "name": "rfphate",
            "param_name": "_dummy",
            "values": [0],
            "base_params": {
                "prediction_type": "regression",
                "n_estimators": 500,
                "n_landmark": 2000,
                "n_pca": 100,
            },
        },
        {
            "name": "rfae",
            "param_name": "_dummy",
            "values": [0],
            "base_params": {
                "prediction_type": "regression",
                "epochs": 50,
                "batch_size": 256,
                "lr": 0.001,
                "n_pca": 100,
                "lam": 1.0e-2,
            },
        },
        {
            "name": "akrmap",
            "param_name": "alpha",
            "values": ALPHAS,
            "base_params": {
                "n_epochs": 20,
                "batch_size": 1000,
                "lr": 0.002,
                "pca_dim": 0,
            },
        },
        {
            "name": "tpse",
            "param_name": "alpha",
            "values": ALPHAS,
            "base_params": {
                "n_epochs": 40,
                "batch_size": 512,
                "pca_dim": 0,
                "n_classes": 10,
            },
        },
        {
            "name": "isdr",
            "param_name": "alpha",
            "values": ALPHAS,
            "base_params": {
                "perplexity": perplexity,
                "tsne_iter": 1000,
                "n_iter": 750,
                "warmup_steps": 50,
                "knn_rebuild_interval": 50,
                "regloss_k": 3 * perplexity,
                "pca_dim": 0,
            },
        },
        {
            "name": "cebra",
            "param_name": "_dummy",
            "values": [0],
            "base_params": {
                "temperature": 1.0,
                "max_iterations": 2000,
                "batch_size": 512,
                "learning_rate": 0.0003,
                "pca_dim": 0,
            },
        },
    ]


def benchmark_config(
    dataset: str, seed: int, output_dir: str, data_root: str = "data", quick: bool = False, device: str = "cpu"
) -> dict:
    spec = SPECS[dataset]
    return {
        "seed": int(seed),
        "output_dir": output_dir,
        "device": device,
        "execution": {"mode": "subprocess", "timeout_seconds": 7200, "resume": True},
        "dataset": {"name": dataset, "data_root": data_root, "test_ratio": 0.0, "data_seed": 42, "split_seed": 42},
        "evaluation": {"knn_k": 10, "x_metric": "trustworthiness", "y_metric": "score_dirichlet_energy"},
        "methods": method_grid(spec.perplexity, quick=quick),
    }


def _summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0), "n": int(arr.size)}


def aggregate_seed_results(seed_results: list[dict]) -> dict:
    grouped: dict[str, dict] = {}
    for result in seed_results:
        for method, points in result.items():
            grouped.setdefault(method, {})
            for key, entry in points.items():
                grouped[method].setdefault(key, [])
                grouped[method][key].append(entry)

    out: dict[str, dict] = {}
    for method, points in grouped.items():
        out[method] = {}
        for key, entries in points.items():
            metric_names = sorted({name for entry in entries for name in entry.get("train_metrics", {})})
            metric_summary = {
                name: _summary([entry.get("train_metrics", {}).get(name, np.nan) for entry in entries])
                for name in metric_names
            }
            out[method][key] = {
                "method": method,
                "param_name": entries[0].get("param_name", "_dummy"),
                "param_value": entries[0].get("param_value", 0),
                "params": entries[0].get("params", {}),
                "time_seconds": _summary([entry.get("time_seconds", np.nan) for entry in entries]),
                "train_metrics": metric_summary,
                "seeds": [entry.get("seed") for entry in entries if entry.get("seed") is not None],
            }
    return out


def load_seed_result(path: Path, seed: int) -> dict:
    with path.open() as handle:
        result = json.load(handle)
    for points in result.values():
        for entry in points.values():
            entry["seed"] = int(seed)
    return result


def aggregate_dataset(dataset_dir: str | Path, seeds: list[int]) -> dict:
    dataset_dir = Path(dataset_dir)
    seed_results = []
    for seed in seeds:
        path = dataset_dir / f"seed_{seed}" / "sweep_results.json"
        if path.exists():
            seed_results.append(load_seed_result(path, seed))
    if not seed_results:
        raise FileNotFoundError(f"No seed results found under {dataset_dir}")
    aggregated = aggregate_seed_results(seed_results)
    with (dataset_dir / "aggregated_results.json").open("w") as handle:
        json.dump(aggregated, handle, indent=2)
    return aggregated


def completed_operating_points(aggregated: dict) -> int:
    return sum(
        1
        for points in aggregated.values()
        for entry in points.values()
        if entry.get("time_seconds", {}).get("n", 0) > 0
    )


def total_worker_seconds(aggregated: dict) -> float:
    total = 0.0
    for points in aggregated.values():
        for entry in points.values():
            t = entry.get("time_seconds", {})
            total += float(t.get("mean", 0.0)) * int(t.get("n", 0))
    return total


def dataset_order(selected: list[str] | None = None) -> list[str]:
    if not selected or selected == ["all"]:
        return regression_dataset_keys()
    return selected
