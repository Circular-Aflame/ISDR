"""Unsupervised baseline adapters for the common benchmark interface."""

from __future__ import annotations

import numpy as np

from .common import pca_reduce, safe_perplexity, set_seed, transform_pca, tsne_kwargs
from .registry import register


@register("pca")
def pca_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    from sklearn.decomposition import PCA

    set_seed(seed)
    model = PCA(n_components=2, random_state=seed)
    Z_train = model.fit_transform(X_train).astype(np.float32)
    Z_test = model.transform(X_test).astype(np.float32) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}


@register("tsne")
def tsne_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    from sklearn.manifold import TSNE

    set_seed(seed)
    p = params or {}
    perplexity = safe_perplexity(X_train.shape[0], float(p.get("perplexity", 30.0)))
    n_iter = int(p.get("n_iter", p.get("max_iter", 1000)))
    X_pca, _ = pca_reduce(X_train, int(p.get("pca_dim", 50)), seed)
    Z_train = TSNE(**tsne_kwargs(seed, perplexity, n_iter)).fit_transform(X_pca).astype(np.float32)
    return {"Z_train": Z_train, "Z_test": None}


@register("umap")
def umap_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP requires `pip install umap-learn`.") from exc

    set_seed(seed)
    p = params or {}
    X_pca, pca = pca_reduce(X_train, int(p.get("pca_dim", 50)), seed)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(p.get("n_neighbors", 15)),
        min_dist=float(p.get("min_dist", 0.1)),
        random_state=seed,
        n_jobs=int(p.get("n_jobs", -1)),
    )
    Z_train = reducer.fit_transform(X_pca).astype(np.float32)
    Z_test = reducer.transform(transform_pca(pca, X_test)).astype(np.float32) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}


@register("pacmap")
def pacmap_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    try:
        import pacmap
    except ImportError as exc:
        raise ImportError("PaCMAP requires `pip install pacmap`.") from exc

    set_seed(seed)
    p = params or {}
    X_pca, _ = pca_reduce(X_train, int(p.get("pca_dim", 50)), seed)
    mapper = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=int(p.get("n_neighbors", 10)),
        random_state=seed,
    )
    return {"Z_train": mapper.fit_transform(X_pca).astype(np.float32), "Z_test": None}
