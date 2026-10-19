# Methods

| Group | Methods | Implementation |
|---|---|---|
| Unsupervised | PCA, t-SNE, UMAP, PaCMAP | Public package APIs |
| Supervised | S-UMAP, CEBRA, RF-PHATE, RF-AE | Public package APIs |
| Local adaptations | AKRMap, TP-SE | Tabular neural models; differences from the references are listed below |
| ISDR | ISDR | t-SNE refinement with a leave-one-out Student-t regression loss |

Installation commands are in the [README](../README.md#benchmarks). The RF installer pins compatible RF-GAP, RF-PHATE, and RF-AE commits. The optional baseline extra requires scikit-learn `<1.9` for RF-GAP's forest API.

## Preprocessing

Set `pca_dim: 0` to use standardized features directly. PCA is also skipped when the requested dimension is at least the input dimension. The image benchmarks use PCA-50 before t-SNE, ISDR, UMAP, PaCMAP, and S-UMAP.

CEBRA and TP-SE have input reducers for neural training; AKRMap supports a larger PCA reducer. RF-PHATE and RF-AE receive the features without external PCA; their `n_pca` parameter controls PCA inside RF-PHATE/PHATE.

ISDR z-scores scalar targets inside the regression loss by default, so `alpha` has a comparable scale across datasets. Evaluation uses the original target values.

## Refinement schedule

Set these parameters in the method's `base_params`. They apply after the geometry-only t-SNE initializer, which has its own schedule. Benchmarks use 1,000 initialization iterations followed by 750 refinement steps at `lr: 100`.

| Parameter | Default | Meaning |
|---|---|---|
| `warmup_steps` | `50` | Linearly increase the regularization weight from zero to `alpha`; `0` uses `alpha` from the first step. |
| `early_exaggeration` | `12.0` | Multiplier on the attractive t-SNE affinities during refinement's early phase. |
| `early_exaggeration_iters` | `None` | Resolve to `min(250, n_iter // 2)`. Set `0` to disable refinement exaggeration. |
| `initial_momentum` | `0.5` | Momentum before the switch. |
| `final_momentum` | `0.8` | Momentum from the switch onward. |
| `momentum_switch_iter` | `None` | Follow the resolved exaggeration duration, or supply an independent zero-based switch iteration. |
| `num_threads` | `None` | Use the effective OpenMP thread count for the refinement Barnes–Hut force, or supply an explicit count. |

For the fixed-weight refinement used in the raw-initializer study, set:

```yaml
warmup_steps: 0
early_exaggeration: 1.0
early_exaggeration_iters: 0
momentum_switch_iter: 0
final_momentum: 0.8
lr: 100
```

To disable exaggeration while keeping the standard momentum timing, set `early_exaggeration_iters: 0` and `momentum_switch_iter: 250`. Schedule indices start at zero. Resolved settings are saved in `profile.json`. `num_threads` controls refinement only; initializer, BLAS, and Numba threads are configured separately.

## Initialization and test layouts

`init` selects the refinement start: `tsne` (default), `pca`, `random`, or `umap`. `tsne_init` selects the raw start for the t-SNE stage: `pca`, `random`, or `umap`. PCA/random starts have scale 1e-4; UMAP is rescaled only when used before the t-SNE stage.

ISDR returns no test embedding by default. `embed_test_separately: true` builds a second supervised layout using test labels. Its coordinates have a separate reference frame, so it cannot be used as a prediction in the training map.

## Baseline scope

The `akrmap` adapter combines parametric t-SNE and adaptive kernel regression in an MLP. It omits AKRMap's original training schedule and interactive components. The `tpse` adapter jointly trains a tabular MLP for reconstruction and classification, with continuous targets placed in quantile bins. It replaces TP-SE's convolutional, two-phase MNIST training. Report both as adaptations when comparing results.

Upstream sources and license notices are listed in the [README](../README.md#license-and-credits).
