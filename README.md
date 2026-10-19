# ISDR

ISDR makes 2D visualizations guided by a scalar target. It refines a t-SNE layout with a leave-one-out Student-t regression loss. The weight `alpha` controls the balance between local geometry and target smoothness.

Paper: [Beyond Contraction: Geometry-Faithful Supervised Dimensionality Reduction for Data Visualization](https://nips.cc/virtual/2026/loc/sydney/poster/155065).
Xu Zhu, Yilin Ye, and Shixia Liu. NeurIPS 2026.

## Quick start

Use Python 3.10+ and run commands from the repository root:

```bash
python -m pip install -e '.[public-data,regression-data,accelerate]'
python -m isdr.cli run --config configs/smoke_digits.yaml
```

Datasets are cached in `data/`. Coordinates, metrics, plots, and run configurations are saved in `results/`.

## Benchmarks

Run PCA, t-SNE, and ISDR on six regression datasets and four image classification datasets:

```bash
python -m isdr.cli prepare-regression-benchmarks --dataset all
python scripts/run_regression_benchmarks.py --methods pca tsne isdr
python scripts/run_classification_benchmarks.py --methods pca tsne isdr
```

Both runners use five seeds: 42, 123, 456, 789, and 2026. Add `--quick --seeds 42` for a short execution check, or `--datasets <name>` to select a dataset.

To include all baselines, install their dependencies and omit `--methods` from the commands above:

```bash
python -m pip install -e '.[baselines]'
python scripts/install_rf_baselines.py
```

With the baselines installed, run the initialization, schedule, neighborhood, and common-start ablations:

```bash
python scripts/ablations/run_ablations.py --config configs/ablations/digits.yaml
```

See [dataset and experiment details](docs/EXPERIMENTS.md) for plotting and [method settings](docs/METHODS.md) for configuration. Classification benchmarks measure labeled layouts; their Dirichlet energy depends on label ordering and does not measure held-out prediction accuracy.

## License and credits

ISDR is released under the [MIT license](LICENSE). The AKRMap and TP-SE baselines are tabular adaptations of [yilinye/AKRMap](https://github.com/yilinye/AKRMap) ([MIT notice](licenses/AKRMap-MIT.txt)) and [JohnNellas/TPSE-PyTorch](https://github.com/JohnNellas/TPSE-PyTorch) ([BSD 3-Clause notice](licenses/TPSE-BSD-3-Clause.txt)).

Other baselines use upstream packages under their own licenses. Datasets are downloaded separately and remain subject to their providers' terms.
