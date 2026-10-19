from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_embedding(
    Z: np.ndarray, scores: np.ndarray, output_path: str | Path, title: str = "", score_type: str = "continuous"
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=scores, cmap="viridis", s=7, alpha=0.9, linewidths=0)
    colorbar = fig.colorbar(sc, ax=ax, shrink=0.78)
    colorbar.set_label("Encoded class label" if score_type == "discrete" else "Observed target")
    if score_type == "discrete":
        colorbar.set_ticks(np.unique(scores))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
