from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

from .common import initial_layout, pca_reduce, safe_perplexity, set_seed, transform_pca
from .registry import register

try:
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None


if nb is not None:

    @nb.njit(parallel=True)
    def _student_t_regloss_grad_numba(Y, scores, knn_indices):
        n_samples = Y.shape[0]
        n_neighbors = knn_indices.shape[1]
        n_threads = nb.get_num_threads()
        grad = np.zeros((n_samples, 2), dtype=np.float64)
        grad_key = np.zeros((n_threads, n_samples, 2), dtype=np.float64)
        loss_sum = 0.0
        for i in nb.prange(n_samples):
            y0 = Y[i, 0]
            y1 = Y[i, 1]
            ksum = 0.0
            weighted = 0.0
            for p in range(n_neighbors):
                j = knn_indices[i, p]
                d0 = y0 - Y[j, 0]
                d1 = y1 - Y[j, 1]
                kval = 1.0 / (1.0 + d0 * d0 + d1 * d1)
                ksum += kval
                weighted += kval * scores[j]
            if ksum < 1e-12:
                ksum = 1e-12
            pred = weighted / ksum
            resid = scores[i] - pred
            loss_sum += resid * resid
            scale = (2.0 / n_samples) * resid / ksum
            thread_idx = nb.get_thread_id()
            for p in range(n_neighbors):
                j = knn_indices[i, p]
                d0 = y0 - Y[j, 0]
                d1 = y1 - Y[j, 1]
                dist_sq = d0 * d0 + d1 * d1
                kval = 1.0 / (1.0 + dist_sq)
                common = kval * kval * (scores[j] - pred)
                g0 = scale * 2.0 * d0 * common
                g1 = scale * 2.0 * d1 * common
                grad[i, 0] += g0
                grad[i, 1] += g1
                grad_key[thread_idx, j, 0] -= g0
                grad_key[thread_idx, j, 1] -= g1
        for t in range(n_threads):
            for i in range(n_samples):
                grad[i, 0] += grad_key[t, i, 0]
                grad[i, 1] += grad_key[t, i, 1]
        return loss_sum / n_samples, grad


def _resolve_threads(num_threads=None) -> int:
    if num_threads is None or int(num_threads) == -1:
        try:
            return int(_openmp_effective_n_threads())
        except Exception:
            return max(1, os.cpu_count() or 1)
    return max(1, int(num_threads))


def compute_tsne_affinity_matrix(X_high: np.ndarray, perplexity: float = 30.0, n_jobs: int = -1):
    from sklearn.manifold._t_sne import _joint_probabilities_nn

    n_samples = X_high.shape[0]
    perplexity = safe_perplexity(n_samples, perplexity)
    n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X_high)
    distances = nn.kneighbors_graph(mode="distance")
    distances.data **= 2
    return _joint_probabilities_nn(distances, perplexity, verbose=0)


def _build_knn_graph(Y: np.ndarray, k: int, n_jobs: int = -1) -> np.ndarray:
    k = max(1, min(int(k), Y.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=n_jobs)
    nn.fit(Y)
    return nn.kneighbors(return_distance=False)


def student_t_regloss_and_grad(Y: np.ndarray, scores: np.ndarray, knn_indices: np.ndarray):
    n_samples, dim = Y.shape
    Y = np.ascontiguousarray(Y.astype(np.float64, copy=False))
    scores = np.ascontiguousarray(scores.astype(np.float64, copy=False).reshape(-1))
    knn_indices = np.ascontiguousarray(knn_indices.astype(np.int64, copy=False))
    if nb is not None and dim == 2:
        loss, grad = _student_t_regloss_grad_numba(Y, scores, knn_indices)
        return float(loss), grad.astype(np.float32)

    Y_nb = Y[knn_indices]
    s_nb = scores[knn_indices]
    diff = Y[:, None, :] - Y_nb
    dist_sq = np.sum(diff * diff, axis=2)
    kernel = 1.0 / (1.0 + dist_sq)
    denom = np.maximum(kernel.sum(axis=1, keepdims=True), 1e-12)
    pred = (kernel * s_nb).sum(axis=1, keepdims=True) / denom
    resid = scores[:, None] - pred
    loss = float(np.mean(resid * resid))

    common = 2.0 * diff * (kernel * kernel)[:, :, None]
    score_diff = s_nb - pred
    weighted = common * score_diff[:, :, None]
    grad_query = (2.0 / n_samples) * resid * weighted.sum(axis=1) / denom
    contrib = (-2.0 / n_samples) * resid[:, :, None] * weighted / denom[:, :, None]
    targets = knn_indices.ravel()
    flat = contrib.reshape(-1, dim)
    grad_key = np.column_stack([np.bincount(targets, weights=flat[:, d], minlength=n_samples) for d in range(dim)])
    return loss, (grad_query + grad_key).astype(np.float32)


def normalize_scores_for_regloss(scores: np.ndarray, mode: str = "zscore") -> tuple[np.ndarray, dict]:
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    mode = str(mode or "zscore").lower().replace("-", "_")
    stats = {
        "mode": mode,
        "raw_mean": float(raw.mean()) if raw.size else 0.0,
        "raw_std": float(raw.std()) if raw.size else 0.0,
        "raw_min": float(raw.min()) if raw.size else 0.0,
        "raw_max": float(raw.max()) if raw.size else 0.0,
    }
    if mode in {"none", "raw", "identity", "false"}:
        return raw, stats
    if mode in {"zscore", "standard", "standardize"}:
        scale = max(float(raw.std()), 1e-12)
        stats["scale"] = scale
        return (raw - float(raw.mean())) / scale, stats
    if mode in {"minmax", "min_max"}:
        scale = max(float(raw.max() - raw.min()), 1e-12)
        stats["scale"] = scale
        return (raw - float(raw.min())) / scale, stats
    raise ValueError("score_normalization must be one of: zscore, minmax, none")


def run_isdr_refinement(
    X_high: np.ndarray,
    scores: np.ndarray,
    Y_init: np.ndarray,
    alpha: float = 0.5,
    perplexity: float = 30.0,
    regloss_k: int | None = None,
    n_iter: int = 750,
    lr: float = 100.0,
    warmup_steps: int = 50,
    knn_rebuild_interval: int = 50,
    P=None,
    n_jobs: int = -1,
    num_threads=None,
    score_normalization: str = "zscore",
    profile_path: str | None = None,
    verbose: bool = False,
    early_exaggeration: float = 12.0,
    early_exaggeration_iters: int | None = None,
    initial_momentum: float = 0.5,
    final_momentum: float = 0.8,
    momentum_switch_iter: int | None = None,
) -> np.ndarray:
    """Refine a layout with configurable, zero-based optimizer schedules.

    ``early_exaggeration_iters=None`` preserves the original duration:
    ``min(250, n_iter // 2)``. Momentum switches at that duration unless an
    independent ``momentum_switch_iter`` is supplied. Set ``warmup_steps=0``
    and ``early_exaggeration_iters=0`` for fixed-weight, unexaggerated forces.
    These settings affect refinement only, not the t-SNE initializer.
    """
    from sklearn.manifold import _barnes_hut_tsne

    def nonnegative_steps(name, value):
        if not np.isfinite(value) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    n_iter = nonnegative_steps("n_iter", n_iter)
    warmup_steps = nonnegative_steps("warmup_steps", warmup_steps)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if X_high.ndim != 2 or len(X_high) < 3 or np.shape(Y_init) != (len(X_high), 2):
        raise ValueError("Expected features and an (n_samples, 2) initializer for at least three samples")
    if np.size(scores) != len(X_high) or not all(np.isfinite(a).all() for a in (X_high, scores, Y_init)):
        raise ValueError("Features, scores, and initializer must be finite and have matching rows")
    knn_rebuild_interval = nonnegative_steps("knn_rebuild_interval", knn_rebuild_interval)
    if knn_rebuild_interval == 0:
        raise ValueError("knn_rebuild_interval must be positive")
    early_exag = float(early_exaggeration)
    if not np.isfinite(early_exag) or early_exag <= 0:
        raise ValueError("early_exaggeration must be finite and positive")
    ee_iters = (
        min(250, n_iter // 2)
        if early_exaggeration_iters is None
        else nonnegative_steps("early_exaggeration_iters", early_exaggeration_iters)
    )
    momentum_iter = (
        ee_iters if momentum_switch_iter is None else nonnegative_steps("momentum_switch_iter", momentum_switch_iter)
    )
    initial_momentum = float(initial_momentum)
    final_momentum = float(final_momentum)
    for name, value in [("initial_momentum", initial_momentum), ("final_momentum", final_momentum)]:
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{name} must be finite and in [0, 1)")

    n_samples = X_high.shape[0]
    perplexity = safe_perplexity(n_samples, perplexity)
    regloss_k = int(3 * perplexity) if regloss_k is None else nonnegative_steps("regloss_k", regloss_k)
    if regloss_k == 0:
        raise ValueError("regloss_k must be positive")
    regloss_k = max(1, min(regloss_k, n_samples - 1))
    scores, score_stats = normalize_scores_for_regloss(scores, score_normalization)
    Y = np.asarray(Y_init, dtype=np.float32).copy()
    threads = _resolve_threads(num_threads)
    profile = {
        "n_samples": int(n_samples),
        "alpha": float(alpha),
        "perplexity": float(perplexity),
        "regloss_k": int(regloss_k),
        "n_iter": int(n_iter),
        "warmup_steps": warmup_steps,
        "early_exaggeration": early_exag,
        "early_exaggeration_iters": ee_iters,
        "initial_momentum": initial_momentum,
        "final_momentum": final_momentum,
        "momentum_switch_iter": momentum_iter,
        "knn_rebuild_interval": int(knn_rebuild_interval),
        "num_threads": threads,
        "score_normalization": score_stats,
        "seconds": {"p_matrix": 0.0, "initial_support": 0.0, "refinement": 0.0},
    }

    t0 = time.perf_counter()
    P = compute_tsne_affinity_matrix(X_high, perplexity, n_jobs=n_jobs) if P is None else P.copy()
    profile["seconds"]["p_matrix"] = time.perf_counter() - t0
    t_support = time.perf_counter()
    knn_indices = _build_knn_graph(Y, regloss_k, n_jobs=n_jobs) if alpha > 1e-8 else None
    profile["seconds"]["initial_support"] = time.perf_counter() - t_support

    n_components = 2
    dof = max(n_components - 1, 1)
    update = np.zeros_like(Y).ravel()
    gains = np.ones_like(Y).ravel()
    momentum = initial_momentum
    min_gain = 0.01
    if lr == "auto" or lr is None:
        lr = float(max(n_samples / early_exag / 4.0, 50.0))
    if not np.isfinite(lr) or float(lr) <= 0:
        raise ValueError("lr must be finite and positive, or 'auto'")
    profile["lr"] = float(lr)

    p_data = P.data.astype(np.float32, copy=True)
    p_data_ee = p_data * early_exag
    neighbors = P.indices.astype(np.int64)
    indptr = P.indptr.astype(np.int64)
    t_ref = time.perf_counter()
    for step in range(int(n_iter)):
        alpha_t = alpha * step / max(1, warmup_steps) if step < warmup_steps else alpha
        if step == momentum_iter:
            momentum = final_momentum
        if alpha > 1e-8 and step > 0 and step % int(knn_rebuild_interval) == 0:
            knn_indices = _build_knn_graph(Y, regloss_k, n_jobs=n_jobs)

        if 1.0 - alpha_t > 1e-8:
            grad_kl = np.zeros(Y.shape, dtype=np.float32)
            _barnes_hut_tsne.gradient(
                p_data_ee if step < ee_iters else p_data,
                np.ascontiguousarray(Y, dtype=np.float32),
                neighbors,
                indptr,
                grad_kl,
                0.5,
                n_components,
                False,
                dof=dof,
                compute_error=False,
                num_threads=threads,
            )
            grad_kl *= 2.0 * (dof + 1.0) / dof
        else:
            grad_kl = np.zeros(Y.shape, dtype=np.float32)

        if alpha_t > 1e-8:
            if knn_indices is None:
                knn_indices = _build_knn_graph(Y, regloss_k, n_jobs=n_jobs)
            _, grad_reg = student_t_regloss_and_grad(Y, scores, knn_indices)
        else:
            grad_reg = np.zeros_like(Y)

        grad = (1.0 - alpha_t) * grad_kl.ravel() + alpha_t * grad_reg.ravel()
        inc = update * grad < 0.0
        gains[inc] += 0.2
        gains[~inc] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        update = momentum * update - float(lr) * gains * grad
        Y = (Y.ravel() + update).reshape(n_samples, n_components)

    profile["seconds"]["refinement"] = time.perf_counter() - t_ref
    if not np.isfinite(Y).all():
        raise FloatingPointError("ISDR refinement produced non-finite coordinates")
    if profile_path:
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2) + "\n")
    return Y.astype(np.float32)


@register("isdr")
def isdr_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    set_seed(seed)
    p = dict(params or {})
    alpha = float(p.get("alpha", 0.5))
    perplexity = safe_perplexity(X_train.shape[0], float(p.get("perplexity", 30.0)))
    n_iter = int(p.get("n_iter", 750))
    warmup = int(p.get("warmup_steps", 50))
    rebuild = int(p.get("knn_rebuild_interval", 50))
    regloss_k = p.get("regloss_k", None)
    pca_dim = int(p.get("pca_dim", 50))
    refinement_schedule = {
        "early_exaggeration": p.get("early_exaggeration", 12.0),
        "early_exaggeration_iters": p.get("early_exaggeration_iters"),
        "initial_momentum": p.get("initial_momentum", 0.5),
        "final_momentum": p.get("final_momentum", 0.8),
        "momentum_switch_iter": p.get("momentum_switch_iter"),
        "num_threads": p.get("num_threads"),
    }

    t_pca = time.perf_counter()
    X_pca, pca = pca_reduce(X_train, pca_dim, seed)
    pca_seconds = time.perf_counter() - t_pca
    t_init = time.perf_counter()
    Y_init = initial_layout(X_pca, seed, p)
    init_seconds = time.perf_counter() - t_init
    np.save(Path(output_dir) / "Y_init.npy", Y_init)
    profile_path = Path(output_dir) / "profile.json"
    Z_train = run_isdr_refinement(
        X_pca,
        scores_train,
        Y_init,
        alpha=alpha,
        perplexity=perplexity,
        regloss_k=regloss_k,
        n_iter=n_iter,
        lr=p.get("lr", 100.0),
        warmup_steps=warmup,
        knn_rebuild_interval=rebuild,
        n_jobs=int(p.get("n_jobs", -1)),
        score_normalization=p.get("score_normalization", "zscore"),
        profile_path=str(profile_path),
        verbose=False,
        **refinement_schedule,
    )
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        profile["init"] = p.get("init", "tsne")
        profile["tsne_init"] = p.get("tsne_init", "pca")
        profile["seconds"].update(pca=pca_seconds, initialization=init_seconds)
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")

    Z_test = None
    if p.get("embed_test_separately", False) and X_test is not None and scores_test is not None:
        X_test_pca = transform_pca(pca, X_test)
        test_perp = safe_perplexity(X_test_pca.shape[0], perplexity)
        Y_test = initial_layout(X_test_pca, seed, {**p, "perplexity": test_perp})
        Z_test = run_isdr_refinement(
            X_test_pca,
            scores_test,
            Y_test,
            alpha=alpha,
            perplexity=test_perp,
            regloss_k=regloss_k,
            n_iter=n_iter,
            lr=p.get("lr", 100.0),
            warmup_steps=warmup,
            knn_rebuild_interval=rebuild,
            n_jobs=int(p.get("n_jobs", -1)),
            score_normalization=p.get("score_normalization", "zscore"),
            verbose=False,
            **refinement_schedule,
        )
    return {"Z_train": Z_train, "Z_test": Z_test}
