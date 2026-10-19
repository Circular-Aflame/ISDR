"""Geometry–signal plots with the paper's method styles and sweep rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
}
_VIRIDIS = matplotlib.colormaps["viridis"]


def _vc(t: float) -> str:
    """Hex color sampled from viridis at position t ∈ [0, 1]."""
    r, g, b, _ = _VIRIDIS(float(t))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


GEOMETRY_CONFIG = {
    "trustworthiness": {"xlabel": "$\\mathbf{Tr}$ustworthiness $\\uparrow$", "cap_at_one": True},
    "continuity": {"xlabel": "$\\mathbf{Cont}$inuity $\\uparrow$", "cap_at_one": True},
    "knn_overlap": {"xlabel": "$\\mathbf{kNN}$ Overlap $\\uparrow$", "cap_at_one": False},
}


@dataclass(frozen=True)
class MethodStyle:
    label: str
    viridis_t: float
    marker: str
    supervised: bool
    linewidth: float
    markersize: float
    zorder: int
    label_mode: str = "geom"
    label_dx: float = 0.0
    label_dy: float = 0.0
    label_ha: str = "left"
    label_va: str = "center"
    label_fs: float = 9.5
    highlight: bool = False


METHOD_STYLES: Dict[str, MethodStyle] = {
    "isdr": MethodStyle("ISDR (Ours)", 0.23, "^", True, 0.82, 3.3, 10, "middle", -18, -6, "right", "top", 9.6, True),
    "akrmap": MethodStyle("AKRMap", 0.36, "o", True, 0.64, 2.65, 8, "smooth", -10, 8, "right", "bottom", 9.6),
    "sumap": MethodStyle("S-UMAP", 0.68, "o", True, 0.64, 2.65, 7, "smooth", 8, 14, "left", "bottom", 9.6),
    "cebra": MethodStyle("CEBRA", 0.03, "o", True, 0.54, 2.45, 6, "smooth", -8, -8, "right", "top", 9.1),
    "tpse": MethodStyle("TP-SE", 0.53, "o", True, 0.58, 2.55, 6, "smooth", -8, -10, "right", "top", 9.1),
    "rfae": MethodStyle("RF-AE", 0.8, "o", True, 0.0, 2.35, 5, "smooth", -6, -8, "right", "top", 8.8),
    "rfphate": MethodStyle("RF-PHATE", 0.18, "o", True, 0.0, 2.35, 4, "geom", 8, 8, "left", "bottom", 8.8),
    "umap": MethodStyle("UMAP", 0.0, "X", False, 0.0, 3.0, 30, "geom", -12, 14, "right", "bottom", 8.4),
    "tsne": MethodStyle("t-SNE", 0.0, "P", False, 0.0, 3.0, 31, "geom", -12, -8, "right", "top", 8.9),
    "pacmap": MethodStyle("PaCMAP", 0.0, "D", False, 0.0, 2.9, 29, "geom", 20, 6, "left", "bottom", 8.4),
    "pca": MethodStyle("PCA", 0.0, "s", False, 0.0, 2.95, 28, "geom", -12, -10, "right", "top", 8.2),
}


_UNSUP_COLORS = {"umap": "#7c7c7c", "tsne": "#4d4d4d", "pacmap": "#969696", "pca": "#b0b0b0"}


_SUP_COLORS = {
    "isdr": "#3f559b",
    "akrmap": "#2b7b9f",
    "sumap": "#36b86f",
    "tpse": "#1f9e89",
    "cebra": "#7a1fa2",
    "rfae": "#d95f02",
    "rfphate": "#c43b8e",
}


METHOD_ORDER = ["isdr", "akrmap", "sumap", "cebra", "tpse", "rfae", "rfphate", "umap", "tsne", "pacmap", "pca"]


UNSUPERVISED_METHODS = ["pca", "tsne", "umap", "pacmap"]


SUPERVISED_METHODS = ["isdr", "akrmap", "sumap", "tpse", "cebra", "rfae", "rfphate"]


PRIMARY_SUPERVISED = {"isdr", "akrmap", "sumap"}


UNSUPERVISED_DIRECT_LABELS = {"umap", "tsne", "pacmap", "pca"}


@dataclass
class MethodSeries:
    name: str
    style: MethodStyle
    color: str
    params: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_err: np.ndarray
    y_err: np.ndarray


@dataclass(frozen=True)
class YAxisCompression:
    low: float
    high: float
    compression: float
    ticks: Tuple[float, ...]


def _method_color(name: str, style: MethodStyle) -> str:
    if not style.supervised:
        return _UNSUP_COLORS.get(name, "#999999")
    return _SUP_COLORS.get(name, _vc(style.viridis_t))


def _parameter_order(s: MethodSeries) -> np.ndarray:
    if len(s.params) == len(s.x):
        return np.argsort(s.params, kind="mergesort")
    return np.lexsort((s.y, s.x))


def _bounded_errors(
    values: np.ndarray, errors: np.ndarray, lower: Optional[float] = None, upper: Optional[float] = None
) -> Optional[np.ndarray]:
    errors = np.maximum(np.asarray(errors, dtype=float), 0.0)
    if not np.any(errors > 0):
        return None
    values = np.asarray(values, dtype=float)
    lower_errors = errors.copy()
    upper_errors = errors.copy()
    if lower is not None:
        lower_errors = np.minimum(lower_errors, np.maximum(values - lower, 0.0))
    if upper is not None:
        upper_errors = np.minimum(upper_errors, np.maximum(upper - values, 0.0))
    return np.vstack([lower_errors, upper_errors])


def _draw_sweep_series(ax: plt.Axes, s: MethodSeries, geom: str, preset: str) -> None:
    order = _parameter_order(s)
    x = s.x[order]
    y = s.y[order]
    xerr = _bounded_errors(x, s.x_err[order], lower=0.0, upper=1.0 if GEOMETRY_CONFIG[geom]["cap_at_one"] else None)
    yerr = _bounded_errors(y, s.y_err[order], lower=0.0, upper=None)
    is_primary = s.name in PRIMARY_SUPERVISED
    line_alpha = 0.96 if s.style.highlight else 0.88 if is_primary or preset == "main" else 0.76
    point_alpha = 0.98 if s.style.highlight else 0.92 if is_primary or preset == "main" else 0.86
    error_alpha = 0.7 if s.style.highlight else 0.58 if is_primary or preset == "main" else 0.48
    error_linewidth = 0.42 if s.style.highlight else 0.34
    error_capsize = 1.0 if s.style.highlight else 0.78
    error_capthick = 0.38 if s.style.highlight else 0.3
    error_zorder = s.style.zorder + 0.05
    error_color = s.color
    marker_area = (s.style.markersize * 1.35) ** 2
    if preset == "all" and (not (is_primary or not s.style.supervised)):
        marker_area *= 0.86
    if preset == "main" and s.name == "sumap" and (len(x) >= 10):
        error_alpha = min(error_alpha, 0.34)
        error_linewidth = 0.3
        error_capsize = 0.68
        error_capthick = 0.26
        error_color = s.color
        error_zorder = max(0.8, s.style.zorder - 0.55)
    path_fx = [
        pe.Stroke(linewidth=s.style.linewidth + (0.24 if s.style.highlight else 0.18), foreground="white", alpha=0.76),
        pe.Normal(),
    ]
    if len(x) > 1 and s.style.linewidth > 0:
        ax.plot(
            x,
            y,
            color=s.color,
            linewidth=s.style.linewidth,
            alpha=line_alpha,
            zorder=s.style.zorder,
            solid_capstyle="round",
            solid_joinstyle="round",
            path_effects=path_fx,
            gid=f"line_{s.name}_{geom}",
        )
    if xerr is not None or yerr is not None:
        err_lines = ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=error_color,
            elinewidth=error_linewidth,
            capsize=error_capsize,
            capthick=error_capthick,
            alpha=error_alpha,
            zorder=error_zorder,
        )
        for el in err_lines[1] + err_lines[2]:
            el.set_gid(f"err_{s.name}_{geom}")
    sc = ax.scatter(
        x,
        y,
        s=marker_area,
        c=s.color,
        marker=s.style.marker,
        edgecolors="white",
        linewidths=0.45 if s.style.highlight else 0.35,
        alpha=point_alpha,
        zorder=s.style.zorder + 0.12,
        path_effects=[pe.withStroke(linewidth=0.55 if s.style.highlight else 0.45, foreground="white")],
    )
    sc.set_gid(f"scatter_{s.name}_{geom}")


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the non-dominated mask for max-x / min-y Pareto optimality."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i]))
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep


def draw_all_series(ax: plt.Axes, series_list: Sequence[MethodSeries], geom: str, preset: str) -> None:
    for s in sorted(series_list, key=lambda s: s.style.zorder):
        _draw_sweep_series(ax, s, geom, preset)


def style_axes(ax: plt.Axes, geom: str) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color("#767676")
    ax.grid(True, color="#e8e8e8", linewidth=0.42)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.7, width=0.55, length=2.5, colors="#333333")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def _piecewise_y_functions(cfg: YAxisCompression):
    compressed_gap = (cfg.high - cfg.low) * cfg.compression

    def forward(values):
        arr = np.asarray(values, dtype=float)
        return np.where(
            arr <= cfg.low,
            arr,
            np.where(
                arr <= cfg.high,
                cfg.low + (arr - cfg.low) * cfg.compression,
                cfg.low + compressed_gap + (arr - cfg.high),
            ),
        )

    def inverse(values):
        arr = np.asarray(values, dtype=float)
        high_mapped = cfg.low + compressed_gap
        return np.where(
            arr <= cfg.low,
            arr,
            np.where(arr <= high_mapped, cfg.low + (arr - cfg.low) / cfg.compression, cfg.high + (arr - high_mapped)),
        )

    return (forward, inverse)


def _maybe_y_compression(
    series_list: Sequence[MethodSeries], preset: str, title: Optional[str]
) -> Optional[YAxisCompression]:
    pca_series = next((s for s in series_list if s.name == "pca"), None)
    if pca_series is None or len(pca_series.y) == 0:
        return None
    pca_y = float(np.min(pca_series.y))
    other_values = [s.y for s in series_list if s.name != "pca" and len(s.y) > 0]
    if not other_values:
        return None
    other_y = np.concatenate(other_values)
    body_ref = float(np.percentile(other_y, 75))
    if pca_y < max(body_ref * 2.2, body_ref + 1.0):
        return None
    bulk_cutoff = max(body_ref * 1.8, body_ref + 0.75)
    bulk_y = other_y[other_y <= bulk_cutoff]
    if bulk_y.size == 0:
        bulk_y = other_y
    body_top = float(np.max(bulk_y))
    low = min(pca_y - 0.25, max(1.18, body_top + 0.06))
    y_max = float(max(np.max(other_y), np.max(pca_series.y)))
    high = max(low + 0.18, y_max - 0.025)
    upper_tick = round(y_max * 2.0) / 2.0
    return YAxisCompression(low=low, high=high, compression=0.006, ticks=(0.0, 0.5, 1.0, 1.5, upper_tick))


def _apply_y_compression(ax: plt.Axes, cfg: YAxisCompression) -> None:
    ax.set_yscale("function", functions=_piecewise_y_functions(cfg))
    ax.yaxis.set_major_locator(FixedLocator(cfg.ticks))
    ax.yaxis.set_minor_locator(NullLocator())


def _add_y_compression_marks(ax: plt.Axes, cfg: Optional[YAxisCompression]) -> None:
    if cfg is None:
        return
    dy = max(0.035, (cfg.high - cfg.low) * 0.01)
    for y_value in (cfg.low, cfg.high):
        for x_center in (0.0, 1.0):
            ax.plot(
                [x_center - 0.01, x_center + 0.01],
                [y_value - dy, y_value + dy],
                transform=ax.get_yaxis_transform(),
                color="#767676",
                linewidth=0.95,
                solid_capstyle="round",
                clip_on=False,
                zorder=60,
            )


def set_axis_limits(
    ax: plt.Axes,
    series_list: Sequence[MethodSeries],
    geom: str,
    preset: str,
    y_compression: Optional[YAxisCompression] = None,
) -> None:
    all_x = np.concatenate([s.x for s in series_list])
    all_y = np.concatenate([s.y for s in series_list])
    xmin, xmax = (float(np.min(all_x)), float(np.max(all_x)))
    ymax = float(np.max(all_y))
    xspan = max(0.02, xmax - xmin)
    if GEOMETRY_CONFIG[geom]["cap_at_one"]:
        xleft = max(0.0, xmin - max(0.015, xspan * 0.05))
        xright = 1.0
    else:
        xleft = max(0.0, xmin - xspan * 0.1)
        xright = xmax + xspan * 0.12
    ax.set_xlim(xleft, xright)
    top_pad = max(0.25, ymax * 0.06)
    if preset == "all":
        top_pad = max(0.3, ymax * 0.065)
    if y_compression is not None:
        _apply_y_compression(ax, y_compression)
        top_pad = max(0.04, ymax * 0.006)
    ax.set_ylim(0.0, ymax + top_pad)


def add_optimal_region(ax: plt.Axes, preset: str) -> None:
    if preset != "all":
        return
    region = Polygon(
        [(0.9, 0.0), (1.0, 0.0), (1.0, 0.125), (0.95, 0.075)],
        closed=True,
        transform=ax.transAxes,
        facecolor="#bfe7d7",
        edgecolor="none",
        alpha=0.12,
        zorder=0.1,
    )
    ax.add_patch(region)
    ax.plot(
        [0.95, 1.0],
        [0.075, 0.0],
        transform=ax.transAxes,
        color="#2b8a6e",
        linewidth=0.95,
        linestyle=(0, (3, 3)),
        alpha=0.3,
        zorder=0.2,
    )


def _select_label_anchor(s: MethodSeries) -> Tuple[float, float]:
    if s.style.supervised:
        fmask = pareto_mask(s.x, s.y)
        fidx = np.where(fmask)[0][np.argsort(s.x[fmask])]
        fx, fy = (s.x[fidx], s.y[fidx])
        if len(fx) == 0:
            return (float(s.x[0]), float(s.y[0]))
        mode = s.style.label_mode
        if mode == "smooth":
            idx = int(np.argmin(fy))
        elif mode == "middle":
            idx = len(fx) // 2
        else:
            idx = int(np.argmax(fx))
        return (float(fx[idx]), float(fy[idx]))
    return (float(s.x[np.argmax(s.x)]), float(s.y[np.argmax(s.x)]))


def add_direct_labels(ax: plt.Axes, series_list: Sequence[MethodSeries], preset: str) -> None:
    if preset == "all":
        for target, ref in [("umap", "sumap"), ("tsne", "isdr")]:
            target_series = next((s for s in series_list if s.name == target), None)
            ref_series = next((s for s in series_list if s.name == ref), None)
            if target_series and ref_series and (len(ref_series.x) > 0):
                idx = np.argmax(ref_series.y)
                lx, ly = (ref_series.x[idx], ref_series.y[idx])
                txt = ax.annotate(
                    target_series.style.label,
                    xy=(lx, ly),
                    xytext=(target_series.style.label_dx, target_series.style.label_dy),
                    textcoords="offset points",
                    ha=target_series.style.label_ha,
                    va=target_series.style.label_va,
                    fontsize=target_series.style.label_fs,
                    color=target_series.color,
                    fontweight="normal",
                    alpha=1.0,
                    annotation_clip=True,
                )
                txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
                txt.set_gid(f"label_{target}_start")
    for s in sorted(series_list, key=lambda item: item.style.zorder):
        if preset == "main" and s.name not in UNSUPERVISED_DIRECT_LABELS:
            continue
        if preset == "all" and s.name not in set(UNSUPERVISED_DIRECT_LABELS) - {"umap", "tsne"}:
            continue
        lx, ly = _select_label_anchor(s)
        is_primary_label = s.name in PRIMARY_SUPERVISED or not s.style.supervised
        txt = ax.annotate(
            s.style.label,
            xy=(lx, ly),
            xytext=(s.style.label_dx, s.style.label_dy),
            textcoords="offset points",
            ha=s.style.label_ha,
            va=s.style.label_va,
            fontsize=s.style.label_fs,
            color=s.color,
            fontweight="bold" if s.name == "isdr" else "normal",
            alpha=1.0 if is_primary_label else 0.84,
            annotation_clip=True,
        )
        txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
        txt.set_gid(f"label_{s.name}_geom")


def _method_order_key(s: MethodSeries) -> int:
    try:
        return METHOD_ORDER.index(s.name)
    except ValueError:
        return len(METHOD_ORDER)


def _group_order_key(s: MethodSeries, group: str) -> int:
    order = UNSUPERVISED_METHODS if group == "unsupervised" else SUPERVISED_METHODS
    try:
        return order.index(s.name)
    except ValueError:
        return len(order) + _method_order_key(s)


def _merged_legend_series(series_groups: Sequence[Sequence[MethodSeries]]) -> List[MethodSeries]:
    by_name: Dict[str, MethodSeries] = {}
    for group in series_groups:
        for series in group:
            by_name.setdefault(series.name, series)
    return sorted(by_name.values(), key=_method_order_key)


def method_legend_handles(
    series_list: Sequence[MethodSeries],
    preset: str = "main",
    group: Optional[str] = None,
    include_group_label: bool = False,
) -> Tuple[List[Line2D], List[str]]:
    handles: List[Line2D] = []
    labels: List[str] = []
    if include_group_label and group:
        handles.append(Line2D([], [], linestyle="None", marker=None, alpha=0.0))
        labels.append("Unsupervised:" if group == "unsupervised" else "Supervised:")
    filtered = list(series_list)
    if group == "supervised":
        filtered = [s for s in filtered if s.style.supervised]
    elif group == "unsupervised":
        filtered = [s for s in filtered if not s.style.supervised]
    for s in sorted(filtered, key=lambda item: _group_order_key(item, group or "supervised")):
        has_path = len(s.x) > 1 and s.style.linewidth > 0
        line_color = s.color if has_path else "none"
        if group == "unsupervised":
            line_color = "none"
            has_path = False
        if group == "supervised" and (not s.style.supervised):
            continue
        handles.append(
            Line2D(
                [],
                [],
                color=line_color,
                marker=s.style.marker,
                linestyle="-" if has_path else "None",
                linewidth=max(1.2, s.style.linewidth * 0.72) if has_path else 0.0,
                markersize=max(5.8, s.style.markersize - 0.6),
                markerfacecolor=s.color,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )
        )
        labels.append(s.style.label)
    return (handles, labels)


def add_figure_method_legend(
    fig: plt.Figure,
    series_groups: Sequence[Sequence[MethodSeries]],
    preset: str,
    seed_counts: Optional[Sequence[int]] = None,
) -> None:
    legend_series = _merged_legend_series(series_groups)
    if preset == "all":
        sup_handles, sup_labels = method_legend_handles(
            legend_series, preset=preset, group="supervised", include_group_label=True
        )
        if sup_handles:
            legend = fig.legend(
                sup_handles,
                sup_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=len(sup_labels),
                frameon=False,
                fontsize=8.0,
                handlelength=1.1,
                handletextpad=0.34,
                columnspacing=0.6,
            )
            if legend.get_texts():
                legend.get_texts()[0].set_fontweight("bold")
                legend.get_texts()[0].set_color("#333333")
        return
    handles, labels = method_legend_handles(legend_series, preset=preset, group="supervised")
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.535, 0.985),
            ncol=min(len(labels), 8),
            frameon=False,
            fontsize=9.0,
            handlelength=1.55,
            handletextpad=0.45,
            columnspacing=1.0,
        )


def add_main_legend(ax: plt.Axes, series_list: Sequence[MethodSeries], preset: str) -> None:
    if preset != "main":
        return
    handles, labels = method_legend_handles(series_list, preset="main")
    leg = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.905),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d8d8d8",
        fontsize=8.4,
        borderpad=0.34,
        handlelength=1.45,
        handletextpad=0.48,
        labelspacing=0.34,
    )
    leg.get_frame().set_linewidth(0.75)


def _color_optimal_ticks(ax: plt.Axes) -> None:
    green_tick = "#1a845e"
    for tick_label in ax.get_xticklabels():
        try:
            if abs(float(tick_label.get_text().replace("−", "-")) - 1.0) < 0.005:
                tick_label.set_color(green_tick)
                tick_label.set_fontweight("bold")
        except ValueError:
            pass
    for tick_label in ax.get_yticklabels():
        try:
            if abs(float(tick_label.get_text().replace("−", "-"))) < 0.005:
                tick_label.set_color(green_tick)
                tick_label.set_fontweight("bold")
        except ValueError:
            pass


def collect_series(results: dict, x_metric: str, y_metric: str, de_scale: float = 100.0) -> list[MethodSeries]:
    """Read either per-run metrics or seed aggregates without dropping sweep points."""
    series_list = []
    scale = de_scale if y_metric == "score_dirichlet_energy" else 1.0
    for method, points in results.items():
        rows = []
        for entry in points.values():
            metrics = entry.get("train_metrics", {})
            if x_metric not in metrics or y_metric not in metrics:
                continue
            x, y = metrics[x_metric], metrics[y_metric]
            x_mean = float(x["mean"] if isinstance(x, dict) else x)
            y_mean = float(y["mean"] if isinstance(y, dict) else y)
            x_std = max(0.0, float(x.get("std", 0.0))) if isinstance(x, dict) else 0.0
            y_std = max(0.0, float(y.get("std", 0.0))) if isinstance(y, dict) else 0.0
            if not np.isfinite([x_mean, y_mean, x_std, y_std]).all():
                continue
            rows.append((float(entry.get("param_value", 0)), x_mean, y_mean * scale, x_std, y_std * scale))
        if not rows:
            continue
        values = np.asarray(sorted(rows, key=lambda row: row[0]))
        style = METHOD_STYLES.get(method)
        if style is None:
            # Ablation variants use their variant names as method keys.
            style = replace(METHOD_STYLES["isdr"], label=method, marker="o", highlight=False)
            color = _vc(0.15 + 0.7 * len(series_list) / max(1, len(results) - 1))
        else:
            color = _method_color(method, style)
        series_list.append(MethodSeries(method, style, color, *values.T))
    return series_list


def render_axis(axis, series_list, x_metric, y_metric, preset="main", title="", de_scale=100.0):
    """Draw the same parameter-ordered paths and bounded SD bars as the paper."""
    paper_axes = x_metric in GEOMETRY_CONFIG and y_metric == "score_dirichlet_energy"
    if series_list and paper_axes:
        compression = _maybe_y_compression(series_list, preset, title)
        draw_all_series(axis, series_list, x_metric, preset)
        style_axes(axis, x_metric)
        set_axis_limits(axis, series_list, x_metric, preset, compression)
        add_optimal_region(axis, preset)
        _add_y_compression_marks(axis, compression)
        add_direct_labels(axis, series_list, preset)
    else:
        for series in series_list:
            axis.errorbar(
                series.x,
                series.y,
                xerr=series.x_err,
                yerr=series.y_err,
                color=series.color,
                marker=series.style.marker,
                linewidth=series.style.linewidth,
                markersize=series.style.markersize,
                linestyle="-" if len(series.x) > 1 else "none",
            )
        style_axes(axis, x_metric)
    axis.set_title(title, loc="left", fontsize=9.8, fontweight="bold", color="#222222", pad=5)
    size = 10.8 if preset == "main" else 9.2
    axis.set_xlabel(GEOMETRY_CONFIG.get(x_metric, {}).get("xlabel", x_metric), fontsize=size, labelpad=5)
    if y_metric == "score_dirichlet_energy":
        ylabel = r"$\mathbf{DE}$ ($\times10^{-2}$) $\downarrow$" if de_scale == 100 else f"{de_scale:g} × DE ↓"
    else:
        ylabel = y_metric
    axis.set_ylabel(ylabel, fontsize=size, labelpad=5)


def plot_panels(
    panels,
    output_path,
    x_metric="trustworthiness",
    y_metric="score_dirichlet_energy",
    *,
    preset="main",
    de_scale=100.0,
    dpi=220,
    formats=None,
):
    """Plot (title, results) panels; output formats follow the requested suffix by default."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formats = formats or [output_path.suffix.lstrip(".") or "png"]
    columns = min(2, max(1, len(panels)))
    rows = max(1, (len(panels) + columns - 1) // columns)
    figsize = (8.0, 4.85) if len(panels) == 1 else (5.3 * columns, 4.55 * rows)
    with matplotlib.rc_context(PAPER_RC):
        figure, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
        all_series = []
        for index, (title, results) in enumerate(panels):
            axis = axes.flat[index]
            series = collect_series(results, x_metric, y_metric, de_scale)
            all_series.append(series)
            prefix = f"({chr(97 + index)}) " if len(panels) > 1 else ""
            render_axis(axis, series, x_metric, y_metric, preset, prefix + title, de_scale)
        for axis in list(axes.flat)[len(panels) :]:
            axis.set_visible(False)
        if len(panels) == 1 and preset == "main":
            if all_series[0]:
                add_main_legend(axes[0, 0], all_series[0], preset)
            figure.subplots_adjust(bottom=0.17, left=0.11, right=0.985, top=0.93)
        else:
            add_figure_method_legend(figure, all_series, preset)
            figure.subplots_adjust(
                bottom=0.18 / rows, left=0.075, right=0.995, top=1 - 0.175 / rows, wspace=0.22, hspace=0.28
            )
        figure.canvas.draw()
        if x_metric in GEOMETRY_CONFIG and y_metric == "score_dirichlet_energy":
            for axis in axes.flat:
                _color_optimal_ticks(axis)
        try:
            for extension in formats:
                figure.savefig(output_path.with_suffix("." + extension), dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(figure)


def plot_pareto(results: dict, output_path: str | Path, x_metric: str, y_metric: str, de_scale: float = 100.0) -> None:
    plot_panels([("", results)], output_path, x_metric, y_metric, de_scale=de_scale)
