#!/usr/bin/env python3
"""Plot benchmark embeddings at representative supervision weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from isdr.data.regression import SPECS
from isdr.plotting.selection import pareto_knee_index


DATASET_LABELS = {key: spec.label for key, spec in SPECS.items()} | {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "organcmnist": "OrganCMNIST",
    "pathmnist": "PathMNIST",
}
SUPERVISED = {"sumap", "akrmap", "tpse", "isdr"}
METHOD_LABELS = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "pacmap": "PaCMAP",
    "sumap": "S-UMAP",
    "cebra": "CEBRA",
    "tpse": "TP-SE",
    "rfphate": "RF-PHATE",
    "rfae": "RF-AE",
    "akrmap": "AKRMap",
    "isdr": "ISDR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render regression or classification benchmark embeddings.")
    parser.add_argument("--results-root", default="results/regression_benchmarks")
    parser.add_argument("--seed-dir", default="seed_42")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_LABELS), default=["pca", "tsne", "isdr"])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--point-size", type=float, default=1.25)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to <results-root>/plots/embeddings.svg",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def metric_value(metrics: Dict, metric_name: str) -> float:
    value = metrics[metric_name]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def parse_method_dirs(method_dir: Path) -> List[Tuple[float, Path]]:
    pairs: List[Tuple[float, Path]] = []
    if not method_dir.exists():
        return pairs
    for child in method_dir.iterdir():
        if not child.is_dir() or "=" not in child.name:
            continue
        try:
            value = float(child.name.split("=", 1)[1])
        except ValueError:
            continue
        pairs.append((value, child))
    return sorted(pairs, key=lambda item: item[0])


def resolve_point_dir(seed_dir: Path, method: str, param_name: str, param_value: float) -> Path:
    method_dir = seed_dir / method
    if param_name == "_dummy":
        default_dir = method_dir / "default"
        if default_dir.exists():
            return default_dir
    exact_dir = method_dir / f"{param_name}={param_value:g}"
    if exact_dir.exists():
        return exact_dir
    default_dir = method_dir / "default"
    if default_dir.exists() and math.isclose(param_value, 0.0, abs_tol=1e-12):
        return default_dir
    pairs = parse_method_dirs(method_dir)
    if not pairs:
        raise FileNotFoundError(f"No point directories found for {method} in {method_dir}")
    best_value, best_dir = min(pairs, key=lambda item: abs(item[0] - param_value))
    if not math.isclose(best_value, param_value, rel_tol=0.0, abs_tol=5e-4):
        raise FileNotFoundError(f"Could not match {method} {param_name}={param_value} in {method_dir}")
    return best_dir


def load_embedding(point_dir: Path, split: str) -> np.ndarray:
    path = point_dir / f"Z_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding: {path}")
    return np.load(path)


def load_scores(seed_dir: Path, split: str) -> np.ndarray:
    path = seed_dir / "split_data.npz"
    if not path.exists():
        path = seed_dir / "_data_cache" / "split_data.npz"
    with np.load(path) as data:
        return np.asarray(data[f"S_{split}"], dtype=np.float64)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    z = np.asarray(embedding, dtype=np.float64)
    p_low = z.min(axis=0)
    p_high = z.max(axis=0)
    center = (p_high + p_low) / 2.0
    span = (p_high - p_low) / 2.0
    scale = max(float(np.max(span)), 1e-12)
    return ((z - center) / scale) * 0.85


def select_value(dataset_root: Path, seed_dir: Path, method: str, split: str) -> float:
    if method not in SUPERVISED:
        return 0.0
    aggregate_path = dataset_root / "aggregated_results.json"
    source = load_json(aggregate_path if aggregate_path.exists() else seed_dir / "sweep_results.json")
    candidates: List[Tuple[float, float, float]] = []
    for info in source.get(method, {}).values():
        metrics = info.get(f"{split}_metrics")
        if not metrics:
            continue
        trust = metric_value(metrics, "trustworthiness")
        de = metric_value(metrics, "score_dirichlet_energy")
        candidates.append((trust, de, float(info.get("param_value", 0.0))))
    if not candidates:
        raise KeyError(f"No selectable candidates for {dataset_root.name}/{method}")
    return candidates[pareto_knee_index([c[0] for c in candidates], [c[1] for c in candidates])][2]


def panel_for(dataset_root: Path, seed_dir: Path, method: str, value: float, split: str) -> Dict:
    results = load_json(seed_dir / "sweep_results.json")
    for info in results.get(method, {}).values():
        if abs(float(info.get("param_value", 0.0)) - value) < 1e-9:
            return {
                "method": method,
                "value": value,
                "metrics": info[f"{split}_metrics"],
                "point_dir": resolve_point_dir(seed_dir, method, info["param_name"], value),
            }
    raise KeyError(f"Missing seed embedding for {dataset_root.name}/{method}={value}")


def draw_panel(axis, panel: Dict, scores: np.ndarray, split: str, point_size: float, color_norm, text_box) -> object:
    embedding = normalize_embedding(load_embedding(panel["point_dir"], split))
    if len(embedding) != len(scores):
        raise ValueError(f"Embedding-score length mismatch for {panel['point_dir']}")
    permutation = np.random.RandomState(42).permutation(len(embedding))
    embedding = embedding[permutation]
    point_scores = scores[permutation]
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=point_scores,
        cmap="viridis",
        norm=color_norm,
        s=point_size,
        linewidths=0,
        alpha=0.80,
        rasterized=True,
        zorder=2,
    )
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#777777")
        spine.set_zorder(20)
    metrics = panel["metrics"]
    axis.text(
        0.03,
        0.97,
        f"100 × DE: {metrics['score_dirichlet_energy'] * 100:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color="#000000",
        bbox=text_box,
        zorder=30,
    )
    axis.text(
        0.97,
        0.97,
        f"Tr: {metrics['trustworthiness']:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#000000",
        bbox=text_box,
        zorder=30,
    )
    return scatter


def selected_datasets(results_root: Path, requested: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    keys = requested or sorted(p.name for p in results_root.glob("*") if (p / "aggregated_results.json").is_file())
    return [(key, DATASET_LABELS.get(key, key)) for key in keys]


def save_outputs(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix not in {".svg", ".pdf"}:
        raise ValueError("--output must end in .svg or .pdf")
    figure.savefig(output_path, format=suffix.lstrip("."), bbox_inches="tight")
    companion = output_path.with_suffix(".pdf" if suffix == ".svg" else ".svg")
    figure.savefig(companion, format=companion.suffix.lstrip("."), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")
    print(f"Saved {companion}")


def render(
    results_root: Path,
    output_path: Path,
    dataset_items: List[Tuple[str, str]],
    seed_name: str,
    split: str,
    point_size: float,
    dpi: int,
    methods: Sequence[str],
) -> None:
    row_count, col_count = len(dataset_items), len(methods)
    figure, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(12.5, 2.6 * row_count + 0.9),
        dpi=dpi,
        constrained_layout=False,
        squeeze=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    figure.subplots_adjust(left=0.11, right=0.93, top=0.89, bottom=0.18)
    color_norm = Normalize(vmin=0.0, vmax=1.0)
    text_box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    scatter = None

    for row_idx, (dataset_key, dataset_label) in enumerate(dataset_items):
        dataset_root = results_root / dataset_key
        seed_dir = dataset_root / seed_name
        scores = load_scores(seed_dir, split)
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        scores = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(scores)
        for col_idx, method in enumerate(methods):
            axis = axes[row_idx, col_idx]
            value = select_value(dataset_root, seed_dir, method, split)
            panel = panel_for(dataset_root, seed_dir, method, value, split)
            scatter = draw_panel(axis, panel, scores, split, point_size, color_norm, text_box)
            if row_idx == 0:
                axis.set_title(METHOD_LABELS.get(method, method), fontsize=11, fontweight="bold", pad=6)
            if method in SUPERVISED:
                axis.text(
                    0.5,
                    0.04,
                    rf"$\lambda$={value:.1f}",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    bbox=text_box,
                    zorder=30,
                )
            if col_idx == 0:
                axis.text(
                    -0.13,
                    0.5,
                    dataset_label,
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

    if scatter is not None:
        color_axis = figure.add_axes([0.76, 0.055, 0.17, 0.024])
        colorbar = figure.colorbar(scatter, cax=color_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=8)
        colorbar.outline.set_linewidth(0.8)
        colorbar.outline.set_edgecolor("#555555")
        figure.text(0.845, 0.091, "Signal Value", ha="center", va="center", fontsize=9.2)
    figure.text(
        0.11,
        0.082,
        r"Pareto-knee points   |   $\mathbf{D}$irichlet $\mathbf{E}$nergy $\downarrow$   |   $\mathbf{Tr}$ustworthiness $\uparrow$",
        ha="left",
        va="center",
        fontsize=10.0,
        color="#333333",
    )
    save_outputs(figure, output_path)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    dataset_items = selected_datasets(results_root, args.datasets)
    if not dataset_items:
        raise SystemExit(f"No matching benchmark datasets found under {results_root}")
    render(
        results_root,
        Path(args.output).resolve() if args.output else results_root / "plots/embeddings.svg",
        dataset_items,
        args.seed_dir,
        args.split,
        args.point_size,
        args.dpi,
        args.methods,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
