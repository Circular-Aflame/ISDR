from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.decomposition import PCA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pca_reduce(X: np.ndarray, n_components: int = 50, seed: int = 42):
    n_components = int(min(n_components, X.shape[0], X.shape[1]))
    if n_components <= 0 or n_components >= X.shape[1]:
        return np.asarray(X, dtype=np.float32), None
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(X).astype(np.float32), pca


def transform_pca(pca, X: np.ndarray) -> np.ndarray:
    if pca is None:
        return np.asarray(X, dtype=np.float32)
    return pca.transform(X).astype(np.float32)


def project_torch_model(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def score_bins(scores: np.ndarray, n_bins: int = 10) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.unique(values).size <= n_bins and np.allclose(values, np.round(values)):
        labels = values.astype(int)
        _, labels = np.unique(labels, return_inverse=True)
        return labels.astype(np.int64)
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-8
    return np.clip(np.digitize(values, edges[1:]), 0, n_bins - 1).astype(np.int64)


def safe_perplexity(n_samples: int, requested: float) -> float:
    if n_samples <= 3:
        return 1.0
    return float(min(float(requested), max(1.0, (n_samples - 1) / 3.0)))


def tsne_kwargs(seed: int, perplexity: float, n_iter: int, init="pca", n_jobs: int = -1) -> dict:
    from sklearn.manifold import TSNE

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        learning_rate="auto",
        method="barnes_hut",
        init=init,
        verbose=0,
    )
    if "max_iter" in TSNE.__init__.__code__.co_varnames:
        kwargs["max_iter"] = int(n_iter)
    else:
        kwargs["n_iter"] = int(n_iter)
    if "n_jobs" in TSNE.__init__.__code__.co_varnames:
        kwargs["n_jobs"] = n_jobs
    return kwargs


def initial_layout(X: np.ndarray, seed: int, params: dict) -> np.ndarray:
    """Build fresh coordinates in the current row order for initialization studies."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    name = params.get("init", "tsne")
    if name == "tsne":
        raw = params.get("tsne_init", "pca")
        if raw == "umap":
            raw = initial_layout(X, seed, {"init": "umap"})
            raw = (raw - raw.mean(axis=0)) / max(float(raw[:, 0].std()), 1e-12) * 1e-4
        return (
            TSNE(
                **tsne_kwargs(
                    seed,
                    safe_perplexity(len(X), float(params.get("perplexity", 30))),
                    int(params.get("tsne_iter", params.get("tsne_n_iter", 1000))),
                    init=raw,
                    n_jobs=int(params.get("n_jobs", -1)),
                )
            )
            .fit_transform(X)
            .astype(np.float32)
        )
    if name == "random":
        return np.random.RandomState(seed).normal(0, 1e-4, (len(X), 2)).astype(np.float32)
    if name == "pca":
        Z = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit_transform(X)
        return (Z / max(float(Z[:, 0].std()), 1e-12) * 1e-4).astype(np.float32)
    if name == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP initialization requires `pip install umap-learn`.") from exc
        return umap.UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(X).astype(np.float32)
    raise ValueError("init must be tsne, pca, random, or umap")
