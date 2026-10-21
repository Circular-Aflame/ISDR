"""Neighborhood geometry, scalar-signal variation, and contraction diagnostics."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors


def _neighbors(data: np.ndarray, k: int) -> np.ndarray:
    n = data.shape[0]
    k_eff = max(0, min(int(k), n - 1))
    if k_eff == 0:
        return np.empty((n, 0), dtype=np.int64)
    # X=None excludes each query's own index, including coincident points.
    return NearestNeighbors(n_neighbors=k_eff).fit(data).kneighbors(return_distance=False)


def trustworthiness(Z: np.ndarray, X_high: np.ndarray, k: int = 10) -> float:
    """Exact neighborhood ranks, computed in chunks to bound peak memory."""
    n = len(Z)
    k = min(int(k), (n - 1) // 2)
    if k < 1:
        raise ValueError("Neighborhood metrics need at least three samples and k >= 1")
    neighbors = _neighbors(Z, k)
    penalty = 0.0
    start = 0
    for distances in pairwise_distances_chunked(X_high, working_memory=64):
        rows = np.arange(len(distances))
        distances[rows, start + rows] = np.inf
        targets = neighbors[start : start + len(rows)]
        selected = np.empty_like(targets)
        tied = np.zeros(len(rows), dtype=bool)
        # Only k ranks are needed; avoid sorting every distance in every row.
        for col in range(k):
            threshold = distances[rows, targets[:, col], None]
            selected[:, col] = (distances < threshold).sum(axis=1) + 1
            tied |= (distances == threshold).sum(axis=1) > 1
        if np.any(tied):
            ranks = np.argsort(np.argsort(distances[tied], axis=1), axis=1)
            selected[tied] = np.take_along_axis(ranks, targets[tied], axis=1) + 1
        penalty += np.maximum(selected - k, 0).sum(dtype=np.float64)
        start += len(rows)
    return float(1 - 2 * penalty / (n * k * (2 * n - 3 * k - 1)))


def continuity(Z: np.ndarray, X_high: np.ndarray, k: int = 10) -> float:
    return trustworthiness(X_high, Z, k)


def knn_overlap(Z: np.ndarray, X_high: np.ndarray, k: int = 10) -> float:
    k_eff = min(k, X_high.shape[0] - 1)
    high = np.sort(_neighbors(X_high, k_eff), axis=1)
    low = np.sort(_neighbors(Z, k_eff), axis=1)
    return float((high[:, :, None] == low[:, None, :]).any(axis=2).sum(axis=1).mean() / k_eff)


def score_dirichlet_energy(Z: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Mean squared signal difference on directed layout kNN edges, before plot scaling."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    denom = scores.max() - scores.min()
    if denom <= 1e-12:
        normalized = np.zeros_like(scores)
    else:
        normalized = (scores - scores.min()) / denom
    idx = _neighbors(Z, k)
    if idx.size == 0:
        return math.nan
    diff = normalized[:, None] - normalized[idx]
    return float(np.mean(diff * diff))


def local_regression_mae(Z: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    scores = np.asarray(scores).reshape(-1)
    idx = _neighbors(Z, k)
    if idx.size == 0:
        return math.nan
    pred = scores[idx].mean(axis=1)
    return float(np.mean(np.abs(scores - pred)))


def local_regression_rmse(Z: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    scores = np.asarray(scores).reshape(-1)
    idx = _neighbors(Z, k)
    if idx.size == 0:
        return math.nan
    pred = scores[idx].mean(axis=1)
    err = scores - pred
    return float(np.sqrt(np.mean(err * err)))


def distance_corr_spearman(Z: np.ndarray, X_high: np.ndarray, n_sample: int = 2000, seed: int = 42) -> float:
    try:
        n = min(n_sample, Z.shape[0])
        rng = np.random.default_rng(seed)
        idx = rng.choice(Z.shape[0], size=n, replace=False) if n < Z.shape[0] else np.arange(n)
        dh = pairwise_distances(X_high[idx])
        dl = pairwise_distances(Z[idx])
        tri = np.triu_indices(n, k=1)
        return float(spearmanr(dh[tri], dl[tri]).statistic)
    except Exception:
        return math.nan


def distance_corr_pearson(Z: np.ndarray, X_high: np.ndarray, n_sample: int = 2000, seed: int = 42) -> float:
    try:
        n = min(n_sample, Z.shape[0])
        rng = np.random.default_rng(seed)
        idx = rng.choice(Z.shape[0], size=n, replace=False) if n < Z.shape[0] else np.arange(n)
        dh = pairwise_distances(X_high[idx])
        dl = pairwise_distances(Z[idx])
        tri = np.triu_indices(n, k=1)
        return float(pearsonr(dh[tri], dl[tri]).statistic)
    except Exception:
        return math.nan


def sample_unordered_pairs(n_items: int, n_pairs: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n_items, size=n_pairs * 2)
    second = rng.integers(0, n_items, size=n_pairs * 2)
    mask = first != second
    pairs = np.column_stack([first[mask], second[mask]])[:n_pairs]
    pairs.sort(axis=1)
    return pairs.astype(np.int64)


def pair_distances(points: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    diff = points[pairs[:, 0]] - points[pairs[:, 1]]
    return np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64))


def theory_proxy_metrics(
    Z: np.ndarray,
    X_high: np.ndarray,
    scores: np.ndarray,
    n_pairs: int = 100000,
    seed: int = 42,
    score_type: str = "continuous",
) -> dict:
    if Z.shape[0] < 2:
        return {}
    pairs = sample_unordered_pairs(Z.shape[0], min(n_pairs, max(1, Z.shape[0] * (Z.shape[0] - 1) // 2)), seed)
    ref = pair_distances(X_high, pairs)
    low = pair_distances(Z, pairs)
    valid = ref > 1e-12
    if not np.any(valid):
        return {}
    expansion = low[valid] / ref[valid]
    mean_expansion = expansion.mean()
    if mean_expansion <= 1e-12:
        return {}
    normalized = expansion / mean_expansion
    sigma = float(np.mean((normalized - 1.0) ** 2))
    score_diff = np.abs(np.asarray(scores)[pairs[valid, 0]] - np.asarray(scores)[pairs[valid, 1]])
    eps = 0.01 * float(np.std(scores) + 1e-12)
    band = score_diff == 0 if score_type == "discrete" else score_diff <= eps
    contraction = float(np.mean(normalized[band])) if np.any(band) else math.nan
    return {"scale_invariant_sigma_distortion": sigma, "contraction_proxy": contraction}


def compute_all_metrics(
    Z: np.ndarray,
    X_high: np.ndarray,
    scores: np.ndarray,
    knn_k: int = 10,
    include_theory: bool = True,
    score_type: str = "continuous",
) -> dict:
    """Evaluate one layout against matching feature rows and scalar signals.

    Signals may be a vector or a single-column array. Returned DE is unscaled;
    the plotting module multiplies it by 100 for display.
    """
    Z = np.asarray(Z)
    X_high = np.asarray(X_high)
    scores = np.asarray(scores)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if X_high.ndim != 2 or X_high.shape[1] == 0 or scores.ndim != 1 or len(scores) != len(X_high):
        raise ValueError("Expected feature rows and one scalar signal per sample")
    if not np.isfinite(X_high).all() or not np.isfinite(scores).all():
        raise ValueError("Features and signals must be finite")
    if Z.shape != (len(X_high), 2) or not np.isfinite(Z).all():
        raise ValueError("Expected a finite (n_samples, 2) embedding")
    if np.any(np.ptp(Z, axis=0) <= 1e-12):
        raise ValueError("Embedding collapsed along a coordinate")
    out = {
        "trustworthiness": trustworthiness(Z, X_high, knn_k),
        "continuity": continuity(Z, X_high, knn_k),
        "knn_overlap": knn_overlap(Z, X_high, knn_k),
        "score_dirichlet_energy": score_dirichlet_energy(Z, scores, knn_k),
        "local_regression_mae": local_regression_mae(Z, scores, knn_k),
        "local_regression_rmse": local_regression_rmse(Z, scores, knn_k),
        "distance_corr_spearman": distance_corr_spearman(Z, X_high),
        "distance_corr_pearson": distance_corr_pearson(Z, X_high),
    }
    if include_theory:
        out.update(theory_proxy_metrics(Z, X_high, scores, seed=42, score_type=score_type))
    if score_type == "discrete":
        labels, encoded = np.unique(scores, return_inverse=True)
        neighbors = encoded[_neighbors(Z, knn_k)]
        counts = (neighbors[:, :, None] == np.arange(len(labels))).sum(axis=1)
        out["knn_label_accuracy"] = float(np.mean(counts.argmax(axis=1) == encoded))
        out["label_neighbor_purity"] = float(np.mean(neighbors == encoded[:, None]))
    return out
