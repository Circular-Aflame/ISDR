# Experiments

## Datasets

| Regression dataset | Samples | Features | Target | Source |
|---|---:|---:|---|---|
| California Housing | 20,640 | 8 | Median house value | [sklearn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) |
| Superconductivity | 21,263 | 81 | Critical temperature | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |
| Wine Quality | 6,497 | 12 | Quality score | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Bike Sharing | 17,379 | 12 | log1p hourly count | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| Energy Efficiency | 768 | 8 | Heating load | [UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| Concrete | 1,030 | 8 | Compressive strength | [UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |

Regression caches contain raw features, observed targets, and a source manifest. Features are standardized after splitting. Bike Sharing excludes `casual` and `registered`, which sum to the target; Wine Quality includes a red/white indicator.

| Classification diagnostic | Sampled population | Samples | Features | Source |
|---|---|---:|---:|---|
| MNIST | Official training set | 10,000 | 784 | [torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) |
| Fashion-MNIST | Official training set | 10,000 | 784 | [Zalando](https://github.com/zalandoresearch/fashion-mnist) |
| OrganCMNIST | Official train + test | 10,000 | 784 | [MedMNIST](https://medmnist.com/) |
| PathMNIST | Official train + test | 10,000 | 2,352 | [MedMNIST](https://medmnist.com/) |

Classification datasets are sampled with `RandomState(42)` and split 80/20 with seed 42. Features use training-only standardization; metrics describe the 8,000 training points. The pooled MedMNIST data are used for visualization, not predictive test accuracy. Dirichlet energy uses integer labels and depends on their ordering; `knn_label_accuracy` and `label_neighbor_purity` provide categorical diagnostics.

Data and split seeds stay fixed across runs. Method seeds default to 42, 123, 456, 789, and 2026. Each seed's `config.yaml` records the full grid. Benchmark commands are in the [README](../README.md#benchmarks).

## Ablations

```bash
python scripts/ablations/run_ablations.py --config configs/ablations/digits.yaml
```

Install `.[baselines]` for UMAP/S-UMAP. Select studies with `--groups` or seeds with `--seeds`. Normal runs use the supplied config; `--quick` reduces the budget to 350 t-SNE iterations and 80 refinement steps for an execution check.

| Group | Comparison |
|---|---|
| `initializer` | Direct t-SNE, PCA, random, and UMAP starts with otherwise identical refinement |
| `raw-initializer` | PCA/random/UMAP coordinates with or without the geometry-only t-SNE stage; fixed supervision and no refinement exaggeration |
| `schedule` | Standard schedule, no weight warmup, fixed objective, geometry-only, and readout-only |
| `support` | Readout neighborhoods of 30/60/90/150; support rebuild intervals of 25/50/100/250 |
| `common-start` | ISDR and S-UMAP both initialized from freshly computed, identical t-SNE coordinates |

The raw-initializer study rescales UMAP before the t-SNE stage; its bypass condition uses native UMAP coordinates. PCA/random bypass starts use the t-SNE raw scale. All initializers are computed from the current feature rows. Common-start S-UMAP uses the same scalar target as ISDR (`target_metric: l2`).

Results go to `results/ablations/<group>/seed_<seed>/<variant>/`, with seed aggregates and a comparison plot per group. `Y_init.npy` saves the refinement start; `profile.json` saves the resolved schedule.

## Run files and timing

Each sweep saves its configuration, dataset information, `split_data.npz`, `run_context.json`, and `sweep_results.json`. Each method point has coordinates, metrics, and a plot. Points run in isolated subprocesses with a two-hour timeout. Interrupted runs reuse matching results; changes to data, seed, parameters, evaluation, source, or dependency versions trigger recomputation. Failed points leave `error.json`.

`time_seconds` covers the complete method call: seeding, preprocessing, initialization, refinement, and any test transform. It excludes data loading, metrics, and plotting. GPU work is synchronized at both timer boundaries; Numba compilation is included. `metric_seconds` records evaluation time separately.

## Plots

Every sweep produces per-point embeddings and a trustworthiness/Dirichlet-energy plot. Dirichlet energy is shown as `100 × DE`; lower is better. Embedding colors show observed target values, or integer labels for classification diagnostics.

The benchmark plotter uses the same method styles, parameter ordering, and mean ± SD rendering as the paper. It accepts either regression or classification results:

```bash
python scripts/plot_benchmarks.py --results-root results/regression_benchmarks
python scripts/plot_benchmarks.py --results-root results/classification_benchmarks
python scripts/plot_benchmark_embeddings.py --results-root results/regression_benchmarks
```

Plots go under `<results-root>/plots/`. Use `--datasets` or `--methods` to select panels. Embedding panels select supervised weights with the [Pareto-knee rule](../isdr/plotting/selection.py), show all samples, and share a target color range within each dataset.

## Tabula Muris FACS

Place the public FACS archive and dissociation gene list under `data/tabula_muris/raw/`:

```bash
python -m isdr.cli prepare-tabula-muris \
  --archive data/tabula_muris/raw/tabula_muris_facs_v7.bin \
  --dissociation-genes data/tabula_muris/raw/genes_affected_by_dissociation.csv \
  --output data/tabula_muris/facs_dissociation
python -m isdr.cli run --config configs/tabula_muris_facs.yaml
```

The importer stores 50 biological principal components, cell metadata, and a dissociation score. This PCA is separate from method-level preprocessing; the archive and derived data are downloaded separately.
