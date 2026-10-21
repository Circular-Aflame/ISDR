"""Supervised affinity-graph and contrastive baseline adapters."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .common import initial_layout, pca_reduce, score_bins, set_seed, transform_pca
from .registry import register


@register("sumap")
@register("supervised_umap")
def supervised_umap_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("S-UMAP requires `pip install umap-learn`.") from exc

    set_seed(seed)
    params = params or {}
    X_pca, pca = pca_reduce(X_train, int(params.get("pca_dim", 50)), seed)
    target_metric = str(params.get("target_metric", "l2"))
    y = score_bins(scores_train, int(params.get("n_bins", 10))) if target_metric == "categorical" else scores_train
    init = initial_layout(X_pca, seed, params) if params.get("init") == "tsne" else params.get("init", "spectral")
    if isinstance(init, np.ndarray):
        np.save(Path(output_dir) / "Y_init.npy", init)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(params.get("n_neighbors", 15)),
        min_dist=float(params.get("min_dist", 0.1)),
        target_metric=target_metric,
        target_weight=float(params.get("target_weight", params.get("alpha", 0.5))),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", -1)),
        init=init,
    )
    Z_train = reducer.fit_transform(X_pca, y=y).astype(np.float32)
    Z_test = reducer.transform(transform_pca(pca, X_test)).astype(np.float32) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}


@register("cebra")
def cebra_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    try:
        import cebra
    except ImportError as exc:
        raise ImportError("CEBRA requires `pip install cebra`.") from exc

    set_seed(seed)
    params = params or {}
    X_pca, pca = pca_reduce(X_train, int(params.get("pca_dim", 100)), seed)
    conditional = params.get("conditional", "delta")
    delta = params.get("delta", 0.1 if conditional == "delta" else None)
    model = cebra.CEBRA(
        model_architecture=params.get("model_architecture", "offset1-model-mse"),
        batch_size=int(params.get("batch_size", 512)),
        learning_rate=float(params.get("learning_rate", 3e-4)),
        max_iterations=int(params.get("max_iterations", 2000)),
        output_dimension=2,
        temperature=float(params.get("temperature", 1.0)),
        distance=params.get("distance", "euclidean"),
        conditional=conditional,
        delta=delta,
        device=params.get("cebra_device", device),
        verbose=bool(params.get("verbose", False)),
    )
    y = np.asarray(scores_train, dtype=np.float32).reshape(-1, 1)
    y = (y - y.mean(axis=0, keepdims=True)) / (y.std(axis=0, keepdims=True) + 1e-8)
    try:
        model.fit(X_pca.astype(np.float32), y)
    except TypeError:
        model.fit(X_pca.astype(np.float32), continuous_index=y)
    Z_train = model.transform(X_pca.astype(np.float32)).astype(np.float32)
    Z_test = model.transform(transform_pca(pca, X_test)).astype(np.float32) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}


@register("rfphate")
@register("rf_phate")
@register("rf-phate")
def rfphate_method(X_train, scores_train, X_test, scores_test, params, output_dir, seed, device="cpu"):
    try:
        from rfphate import RFPHATE
    except ImportError as exc:
        raise ImportError("RF-PHATE requires `pip install rfphate rfgap`.") from exc

    set_seed(seed)
    params = params or {}
    if int(params.get("pca_dim", 0) or 0) > 0:
        raise ValueError("RF-PHATE uses raw standardized input; set n_pca for RF-PHATE's internal PHATE PCA.")
    X_fit = np.asarray(X_train, dtype=np.float32)
    rfphate_n_pca = max(1, int(params.get("rfphate_n_pca", params.get("n_pca", 100))))
    n_landmark = max(2, min(int(params.get("n_landmark", 2000)), X_fit.shape[0] - 1))
    scores = np.asarray(scores_train).reshape(-1)
    prediction_type = params.get("prediction_type")
    if prediction_type is None and np.unique(scores).size <= 20 and np.allclose(scores, np.round(scores)):
        prediction_type = "classification"
        y_fit = scores.astype(int)
    else:
        prediction_type = str(prediction_type or "regression")
        y_fit = scores.astype(int) if prediction_type == "classification" else scores
    model = RFPHATE(
        prediction_type=prediction_type,
        n_components=2,
        prox_method=params.get("prox_method", "rfgap"),
        n_estimators=int(params.get("n_estimators", 500)),
        n_landmark=n_landmark,
        t=params.get("t", "auto"),
        n_pca=rfphate_n_pca,
        mds_solver=params.get("mds_solver", "sgd"),
        beta=float(params.get("beta", 0.9)),
        random_state=seed,
        n_jobs=int(params.get("n_jobs", -1)),
        verbose=0,
    )
    Z_train = model.fit_transform(X_fit, y_fit).astype(np.float32)
    Z_test = model.transform(np.asarray(X_test, dtype=np.float32)).astype(np.float32) if X_test is not None else None
    return {"Z_train": Z_train, "Z_test": Z_test}
