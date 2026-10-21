"""Public dataset loaders and fixed-seed splits with training-only scaling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .regression import SPECS as REGRESSION_SPECS
from .regression import load_regression_cache


@dataclass
class DatasetBundle:
    """Feature rows, scalar signals, and provenance before splitting or scaling."""
    X: np.ndarray
    scores: np.ndarray
    name: str
    score_type: str
    metadata: dict


def _subsample(X: np.ndarray, scores: np.ndarray, n_samples, seed: int):
    if n_samples is None or int(n_samples) <= 0 or int(n_samples) >= X.shape[0]:
        return X, scores
    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])[: int(n_samples)]
    return X[idx], scores[idx]


def load_digits_dataset(n_samples=None, seed: int = 42, **kwargs) -> DatasetBundle:
    from sklearn.datasets import load_digits

    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    scores = data.target.astype(np.float32)
    X, scores = _subsample(X, scores, n_samples, seed)
    return DatasetBundle(X, scores, "sklearn digits", "discrete", {"source": "sklearn.datasets.load_digits"})


def load_swiss_roll(
    n_samples: int = 10000, seed: int = 42, noise: float = 0.5, n_noise_dims: int = 97, **kwargs
) -> DatasetBundle:
    from sklearn.datasets import make_swiss_roll

    X3, scores = make_swiss_roll(n_samples=int(n_samples), noise=float(noise), random_state=seed)
    rng = np.random.RandomState(seed)
    noise_block = rng.normal(0.0, 0.5, size=(X3.shape[0], int(n_noise_dims)))
    X = np.hstack([X3, noise_block]).astype(np.float32)
    return DatasetBundle(
        X, scores.astype(np.float32), "Swiss Roll", "continuous", {"source": "sklearn make_swiss_roll"}
    )


def load_s_curve(
    n_samples: int = 10000, seed: int = 42, noise: float = 0.1, n_noise_dims: int = 97, **kwargs
) -> DatasetBundle:
    from sklearn.datasets import make_s_curve

    X3, scores = make_s_curve(n_samples=int(n_samples), noise=float(noise), random_state=seed)
    rng = np.random.RandomState(seed)
    noise_block = rng.normal(0.0, 0.5, size=(X3.shape[0], int(n_noise_dims)))
    X = np.hstack([X3, noise_block]).astype(np.float32)
    return DatasetBundle(X, scores.astype(np.float32), "S-Curve", "continuous", {"source": "sklearn make_s_curve"})


def load_mnist(n_samples=10000, seed: int = 42, data_root: str | Path = "data", **kwargs) -> DatasetBundle:
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError("MNIST loading requires `pip install torchvision`.") from exc

    dataset = MNIST(root=str(Path(data_root) / "torchvision"), train=True, download=True)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dataset))[: int(n_samples or len(dataset))]
    X = np.empty((len(idx), 784), dtype=np.float32)
    scores = np.empty(len(idx), dtype=np.float32)
    for row, item_idx in enumerate(idx):
        image, label = dataset[int(item_idx)]
        X[row] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        scores[row] = float(label)
    return DatasetBundle(X, scores, "MNIST", "discrete", {"source": "torchvision MNIST"})


def load_fashion_mnist(n_samples=10000, seed: int = 42, data_root: str | Path = "data", **kwargs) -> DatasetBundle:
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        raise ImportError("Fashion-MNIST loading requires `pip install torchvision`.") from exc

    dataset = FashionMNIST(root=str(Path(data_root) / "torchvision"), train=True, download=True)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dataset))[: int(n_samples or len(dataset))]
    X = np.empty((len(idx), 784), dtype=np.float32)
    scores = np.empty(len(idx), dtype=np.float32)
    for row, item_idx in enumerate(idx):
        image, label = dataset[int(item_idx)]
        X[row] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
        scores[row] = float(label)
    return DatasetBundle(X, scores, "Fashion-MNIST", "discrete", {"source": "torchvision FashionMNIST"})


def load_medmnist(
    name: str, n_samples=10000, seed: int = 42, data_root: str | Path = "data", **kwargs
) -> DatasetBundle:
    try:
        import medmnist
    except ImportError as exc:
        raise ImportError("MedMNIST datasets require `pip install 'isdr[public-data]'`.") from exc
    root = Path(data_root) / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    info = medmnist.INFO[name]
    cls = getattr(medmnist, info["python_class"])
    datasets = [cls(split=split, root=str(root), download=True) for split in ("train", "test")]
    X = np.concatenate([ds.imgs.reshape(len(ds), -1) for ds in datasets]).astype(np.float32) / 255.0
    scores = np.concatenate([ds.labels.reshape(-1) for ds in datasets]).astype(np.float32)
    X, scores = _subsample(X, scores, n_samples, seed)
    return DatasetBundle(
        X,
        scores,
        name,
        "discrete",
        {"source": info["url"], "labels": info["label"], "population": "official train + test, visualization only"},
    )


def load_tabula_muris_facs(
    processed_dir: str | Path = "data/tabula_muris/facs_dissociation", n_samples=None, seed: int = 42, **kwargs
) -> DatasetBundle:
    data_path = Path(processed_dir) / "full_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing processed Tabula Muris dataset: {data_path}. Run `isdr prepare-tabula-muris` first."
        )
    with np.load(data_path, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        if "S" in data.files:
            scores = data["S"].astype(np.float32)
        elif "scores" in data.files:
            scores = data["scores"].astype(np.float32)
        else:
            raise KeyError(f"No score array found in {data_path}")
    X, scores = _subsample(X, scores, n_samples, seed)
    return DatasetBundle(X, scores, "Tabula Muris FACS", "continuous", {"source": str(data_path)})


def load_regression_benchmark(
    name: str, data_root: str | Path = "data", n_samples=None, seed: int = 42, **kwargs
) -> DatasetBundle:
    X, scores, spec = load_regression_cache(name, data_root=data_root, n_samples=n_samples, seed=seed)
    return DatasetBundle(
        X,
        scores,
        spec.label,
        "continuous",
        {
            "source": spec.source_url,
            "dataset_key": spec.key,
            "domain": spec.domain,
            "target": spec.target,
            "citation_key": spec.citation_key,
            "default_perplexity": spec.perplexity,
        },
    )


LOADERS = {
    "digits": load_digits_dataset,
    "sklearn_digits": load_digits_dataset,
    "swiss_roll": load_swiss_roll,
    "s_curve": load_s_curve,
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "organcmnist": partial(load_medmnist, "organcmnist"),
    "pathmnist": partial(load_medmnist, "pathmnist"),
    "tabula_muris_facs": load_tabula_muris_facs,
}

for _regression_key in REGRESSION_SPECS:
    LOADERS[_regression_key] = partial(load_regression_benchmark, _regression_key)


def load_dataset(config: dict, seed: int) -> DatasetBundle:
    """Load fixed rows using ``data_seed``; ``seed`` is the method seed, kept separate."""
    name = str(config.get("name", "digits")).lower()
    if name not in LOADERS:
        available = ", ".join(sorted(LOADERS))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    kwargs = dict(config)
    kwargs.pop("name", None)
    data_seed = int(kwargs.pop("data_seed", 42))
    return LOADERS[name](seed=data_seed, **kwargs)


def split_data(X: np.ndarray, scores: np.ndarray, test_ratio: float, seed: int) -> dict:
    """Split rows reproducibly and standardize both splits with training statistics."""
    if X.ndim != 2 or len(X) != len(scores) or len(X) < 3:
        raise ValueError("Expected matching features and scalar scores for at least three samples")
    if not np.isfinite(X).all() or not np.isfinite(scores).all():
        raise ValueError("Features and scores must be finite")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1)")
    n = X.shape[0]
    n_test = int(n * float(test_ratio))
    if n - n_test < 3:
        raise ValueError("The training split must contain at least three samples")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    train_idx = idx[:-n_test] if n_test > 0 else idx
    test_idx = idx[-n_test:] if n_test > 0 else np.empty((0,), dtype=np.int64)
    X_train = X[train_idx].astype(np.float32)
    X_test = X[test_idx].astype(np.float32) if test_idx.size else None
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = ((X_train - mean) / std).astype(np.float32)
    if X_test is not None:
        X_test = ((X_test - mean) / std).astype(np.float32)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "S_train": scores[train_idx].astype(np.float32),
        "S_test": scores[test_idx].astype(np.float32) if test_idx.size else None,
        "train_idx": train_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
    }
