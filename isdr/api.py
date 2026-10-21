"""Array-based entry point for embedding data without the experiment runner."""

from __future__ import annotations

from inspect import signature

import numpy as np
from sklearn.utils.validation import check_X_y


def fit_transform(X, scores, *, alpha: float = 0.5, random_state: int = 42, **params) -> np.ndarray:
    """Return a supervised 2D layout in the input row order, without writing files.

    ``X`` is a finite dense matrix with at least three rows and two features;
    ``scores`` contains one scalar per row. Features are used as supplied:
    standardize them beforehand if their units differ. Scores are z-scored
    inside the regularizer by default.

    ``alpha`` lies in [0, 1] and controls supervision. Additional keyword
    arguments use the configuration names in docs/METHODS.md, including
    ``pca_dim``, ``init``, ``tsne_init``, ``tsne_iter``, ``perplexity``,
    ``n_iter``, ``lr``, and the refinement schedule controls. This builds a
    nonparametric layout; it does not train an out-of-sample transform.
    """
    from .methods.core import isdr_method, run_isdr_refinement

    # Share the optimizer's option names, excluding its internal data and I/O.
    refinement_options = set(signature(run_isdr_refinement).parameters) - {
        "X_high",
        "scores",
        "Y_init",
        "P",
        "profile_path",
        "verbose",
        "alpha",
    }
    initialization_options = {"pca_dim", "init", "tsne_init", "tsne_iter", "tsne_n_iter"}
    unknown = set(params) - refinement_options - initialization_options
    if unknown:
        raise TypeError(f"Unknown ISDR options: {', '.join(sorted(unknown))}")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    X, scores = check_X_y(
        X,
        scores,
        dtype=np.float32,
        ensure_min_samples=3,
        ensure_min_features=2,
        y_numeric=True,
    )
    result = isdr_method(X, scores, None, None, {**params, "alpha": alpha}, None, random_state)
    return result["Z_train"]
