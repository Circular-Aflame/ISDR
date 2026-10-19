# Local data

Downloads and generated caches are ignored by Git.

```bash
python -m isdr.cli prepare-regression-benchmarks --dataset all
```

Regression caches contain raw feature arrays, observed scalar targets, and source manifests under `regression_benchmarks_cache/`. Features are standardized after splitting. `--force` rebuilds existing caches, including older standardized caches.

Image dataset loaders download into `torchvision/` and `medmnist/`. Dataset sources and sampling conventions are documented in [experiment protocols](../docs/EXPERIMENTS.md).
