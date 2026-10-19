from __future__ import annotations

import shutil
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class RegressionDatasetSpec:
    key: str
    label: str
    n_samples: int
    n_features: int
    target: str
    domain: str
    source_url: str
    citation_key: str
    perplexity: int


SPECS = {
    "california_housing": RegressionDatasetSpec(
        "california_housing",
        "California Housing",
        20640,
        8,
        "median house value",
        "geographic/economic",
        "https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset",
        "pace1997california",
        30,
    ),
    "superconductivity": RegressionDatasetSpec(
        "superconductivity",
        "Superconductivity",
        21263,
        81,
        "critical temperature",
        "materials science",
        "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "hamidieh2018superconductor",
        30,
    ),
    "wine_quality": RegressionDatasetSpec(
        "wine_quality",
        "Wine Quality",
        6497,
        12,
        "sensory quality score",
        "chemistry/sensory",
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "cortez2009wine",
        30,
    ),
    "bike_sharing": RegressionDatasetSpec(
        "bike_sharing",
        "Bike Sharing",
        17379,
        12,
        "log1p hourly rental count",
        "demand forecasting",
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "fanaee2014event",
        30,
    ),
    "energy_efficiency": RegressionDatasetSpec(
        "energy_efficiency",
        "Energy Efficiency",
        768,
        8,
        "heating load",
        "building physics",
        "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
        "tsanas2012energy",
        15,
    ),
    "concrete": RegressionDatasetSpec(
        "concrete",
        "Concrete",
        1030,
        8,
        "compressive strength",
        "civil engineering",
        "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
        "yeh1998concrete",
        15,
    ),
}


def regression_dataset_keys() -> list[str]:
    return list(SPECS)


def cache_paths(dataset: str, data_root: str | Path = "data") -> tuple[Path, Path]:
    cache_dir = Path(data_root) / "regression_benchmarks_cache"
    return cache_dir / f"{dataset}_X.npy", cache_dir / f"{dataset}_s.npy"


def _feature_array(values) -> np.ndarray:
    # Scaling is fitted on the training split by split_data, never in a cache.
    return np.asarray(values, dtype=np.float32)


def _save(dataset: str, X: np.ndarray, s: np.ndarray, data_root: str | Path) -> tuple[Path, Path]:
    X_path, s_path = cache_paths(dataset, data_root)
    X_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_path, np.asarray(X, dtype=np.float32))
    np.save(s_path, np.asarray(s, dtype=np.float32).reshape(-1))
    manifest = {
        "dataset": dataset,
        "source": SPECS[dataset].source_url,
        "target": SPECS[dataset].target,
        "shape": list(X.shape),
        "features": "raw; standardized after splitting",
        "format_version": 2,
    }
    X_path.with_name(f"{dataset}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return X_path, s_path


def _download(url: str, output: Path, force: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output
    tmp = output.with_suffix(output.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(output)
    return output


def _zip_member(zip_path: Path, basename: str) -> str:
    with ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
    if not matches:
        raise FileNotFoundError(f"Could not find {basename} inside {zip_path}")
    return matches[0]


def _extract(zip_path: Path, basename: str, output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / basename
    if target.exists() and not force:
        return target
    member = _zip_member(zip_path, basename)
    with ZipFile(zip_path) as archive, archive.open(member) as source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return target


def _raw_zip(dataset: str, data_root: str | Path, force: bool = False) -> Path:
    spec = SPECS[dataset]
    raw_dir = Path(data_root) / "regression_benchmarks" / "raw"
    filename = f"{dataset}.zip"
    return _download(spec.source_url, raw_dir / filename, force=force)


def build_california_housing(data_root: str | Path = "data", force: bool = False) -> tuple[Path, Path]:
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=False, data_home=Path(data_root) / "regression_benchmarks" / "raw")
    X = _feature_array(data.data)
    s = data.target.astype(np.float32)
    print(f"California Housing: X={X.shape}, s range [{s.min():.3f}, {s.max():.3f}]")
    return _save("california_housing", X, s, data_root)


def build_superconductivity(data_root: str | Path = "data", force: bool = False) -> tuple[Path, Path]:
    import pandas as pd

    zip_path = _raw_zip("superconductivity", data_root, force=force)
    csv_path = _extract(
        zip_path, "train.csv", Path(data_root) / "regression_benchmarks" / "superconductivity", force=force
    )
    df = pd.read_csv(csv_path)
    X = _feature_array(df.iloc[:, :-1].values)
    s = df["critical_temp"].values.astype(np.float32)
    print(f"Superconductivity: X={X.shape}, s range [{s.min():.3f}, {s.max():.3f}]")
    return _save("superconductivity", X, s, data_root)


def build_wine_quality(data_root: str | Path = "data", force: bool = False) -> tuple[Path, Path]:
    import pandas as pd

    zip_path = _raw_zip("wine_quality", data_root, force=force)
    raw_dir = Path(data_root) / "regression_benchmarks" / "wine_quality"
    red_path = _extract(zip_path, "winequality-red.csv", raw_dir, force=force)
    white_path = _extract(zip_path, "winequality-white.csv", raw_dir, force=force)
    red = pd.read_csv(red_path, sep=";")
    red["is_red"] = 1
    white = pd.read_csv(white_path, sep=";")
    white["is_red"] = 0
    df = pd.concat([red, white], ignore_index=True)
    target = "quality"
    features = [column for column in df.columns if column != target]
    X = _feature_array(df[features].values)
    s = df[target].values.astype(np.float32)
    print(f"Wine Quality: X={X.shape}, s range [{s.min():.0f}, {s.max():.0f}]")
    return _save("wine_quality", X, s, data_root)


def build_bike_sharing(data_root: str | Path = "data", force: bool = False) -> tuple[Path, Path]:
    import pandas as pd

    zip_path = _raw_zip("bike_sharing", data_root, force=force)
    csv_path = _extract(zip_path, "hour.csv", Path(data_root) / "regression_benchmarks" / "bike_sharing", force=force)
    df = pd.read_csv(csv_path)
    drop = ["instant", "dteday", "casual", "registered"]
    features = [column for column in df.columns if column not in drop + ["cnt"]]
    X = _feature_array(df[features].values)
    s = np.log1p(df["cnt"].values).astype(np.float32)
    print(f"Bike Sharing: X={X.shape}, s range [{s.min():.3f}, {s.max():.3f}] (log1p cnt)")
    return _save("bike_sharing", X, s, data_root)


def build_energy_efficiency(data_root: str | Path = "data", force: bool = False) -> tuple[Path, Path]:
    import pandas as pd

    zip_path = _raw_zip("energy_efficiency", data_root, force=force)
    xlsx_path = _extract(
        zip_path, "ENB2012_data.xlsx", Path(data_root) / "regression_benchmarks" / "energy_efficiency", force=force
    )
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    features = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    X = _feature_array(df[features].values)
    s = df["Y1"].values.astype(np.float32)
    print(f"Energy Efficiency: X={X.shape}, s range [{s.min():.3f}, {s.max():.3f}] (Y1 heating)")
    return _save("energy_efficiency", X, s, data_root)


def build_concrete(data_root: str | Path = "data", force: bool = False) -> tuple[Path, Path]:
    import pandas as pd

    zip_path = _raw_zip("concrete", data_root, force=force)
    xls_path = _extract(
        zip_path, "Concrete_Data.xls", Path(data_root) / "regression_benchmarks" / "concrete", force=force
    )
    df = pd.read_excel(xls_path)
    X = _feature_array(df.iloc[:, :-1].values)
    s = df.iloc[:, -1].values.astype(np.float32)
    print(f"Concrete: X={X.shape}, s range [{s.min():.3f}, {s.max():.3f}]")
    return _save("concrete", X, s, data_root)


BUILDERS = {
    "california_housing": build_california_housing,
    "superconductivity": build_superconductivity,
    "wine_quality": build_wine_quality,
    "bike_sharing": build_bike_sharing,
    "energy_efficiency": build_energy_efficiency,
    "concrete": build_concrete,
}


def build_regression_benchmark(
    dataset: str, data_root: str | Path = "data", force: bool = False
) -> list[tuple[Path, Path]]:
    dataset = dataset.lower()
    keys = regression_dataset_keys() if dataset == "all" else [dataset]
    outputs = []
    for key in keys:
        if key not in BUILDERS:
            available = ", ".join(["all", *regression_dataset_keys()])
            raise ValueError(f"Unknown regression benchmark '{key}'. Available: {available}")
        paths = cache_paths(key, data_root)
        if all(path.exists() for path in paths) and not force:
            outputs.append(paths)
            continue
        outputs.append(BUILDERS[key](data_root=data_root, force=force))
    return outputs


def load_regression_cache(dataset: str, data_root: str | Path = "data", n_samples=None, seed: int = 42):
    dataset = dataset.lower()
    if dataset not in SPECS:
        raise ValueError(f"Unknown regression benchmark '{dataset}'")
    X_path, s_path = cache_paths(dataset, data_root)
    if not X_path.exists() or not s_path.exists():
        raise FileNotFoundError(
            f"Missing regression cache for {dataset}. Run `python -m isdr.cli prepare-regression-benchmarks --dataset {dataset}` first."
        )
    X = np.load(X_path).astype(np.float32)
    s = np.load(s_path).astype(np.float32)
    if n_samples is not None and int(n_samples) > 0 and int(n_samples) < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(X.shape[0])[: int(n_samples)]
        X, s = X[idx], s[idx]
    return X, s, SPECS[dataset]
