from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


def _read_tables(archive: Path):
    import pandas as pd

    outer = ZipFile(archive)
    metadata = pd.read_csv(outer.open("metadata_FACS.csv"))
    annotations = pd.read_csv(outer.open("annotations_FACS.csv"))
    inner = ZipFile(io.BytesIO(outer.read("FACS.zip")))
    return metadata, annotations, inner


def _sample_rows(rows, sample_size: int | None, seed: int):
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= len(rows):
        return rows.copy().sort_values(["tissue", "cell"]).reset_index(drop=True)
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=int(sample_size), random_state=seed)
        selected, _ = next(splitter.split(np.zeros(len(rows)), rows["tissue"]))
        return rows.iloc[selected].copy().sort_values(["tissue", "cell"]).reset_index(drop=True)
    except Exception:
        return rows.sample(n=int(sample_size), random_state=seed).sort_values(["tissue", "cell"]).reset_index(drop=True)


def _load_counts(inner: ZipFile, sampled):
    import pandas as pd

    members = sorted(name for name in inner.namelist() if name.endswith("-counts.csv"))
    if not members:
        raise RuntimeError("No FACS count matrices found in archive")
    gene_names = pd.read_csv(inner.open(members[0]), usecols=[0]).iloc[:, 0].astype(str).to_numpy()
    blocks = []
    for tissue, tissue_rows in sampled.groupby("tissue", sort=True):
        member = f"FACS/{tissue}-counts.csv"
        if member not in inner.namelist():
            raise KeyError(f"Missing tissue count file: {member}")
        header = pd.read_csv(inner.open(member), nrows=0)
        selected_cells = tissue_rows["cell"].tolist()
        usecols = [header.columns[0]] + [c for c in header.columns if c in selected_cells]
        matrix = pd.read_csv(inner.open(member), index_col=0, usecols=usecols)
        matrix = matrix.loc[:, selected_cells]
        if not np.array_equal(matrix.index.astype(str).to_numpy(), gene_names):
            raise RuntimeError(f"Gene axis mismatch in {member}")
        blocks.append(sparse.csr_matrix(matrix.to_numpy(dtype=np.float32).T))
    counts = sparse.vstack(blocks, format="csr")
    if counts.shape[0] != len(sampled):
        raise RuntimeError("Loaded count rows do not align with metadata rows")
    return counts, gene_names


def _log_normalize(counts, totals, scale: float):
    safe = np.maximum(totals.astype(np.float32), 1.0)
    out = counts.multiply((float(scale) / safe).reshape(-1, 1)).tocsr()
    out.data = np.log1p(out.data)
    return out


def _zscore(values):
    values = np.asarray(values, dtype=np.float32)
    return ((values - values.mean()) / (values.std() + 1e-6)).astype(np.float32)


def build_tabula_muris_facs(
    archive: str | Path,
    output: str | Path,
    dissociation_genes: str | Path,
    sample_size: int | None = None,
    seed: int = 42,
    n_hvg: int = 2000,
    pca_dim: int = 50,
) -> Path:
    import pandas as pd

    archive = Path(archive)
    output = Path(output)
    dissociation_genes = Path(dissociation_genes)
    if not archive.exists():
        raise FileNotFoundError(archive)
    if not dissociation_genes.exists():
        raise FileNotFoundError(dissociation_genes)

    metadata, annotations, inner = _read_tables(archive)
    rows = (
        annotations.merge(
            metadata,
            on=["plate.barcode", "mouse.id", "mouse.sex", "tissue"],
            how="left",
            suffixes=("", "_meta"),
        )
        .drop_duplicates(subset=["cell"])
        .reset_index(drop=True)
    )
    sampled = _sample_rows(rows, sample_size, seed)
    counts, gene_names = _load_counts(inner, sampled)

    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    log_norm = _log_normalize(counts, totals, scale=1e4)
    mean = np.asarray(log_norm.mean(axis=0)).ravel().astype(np.float32)
    mean_sq = np.asarray(log_norm.power(2).mean(axis=0)).ravel().astype(np.float32)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    hvg_idx = np.sort(np.argsort(variance)[-min(int(n_hvg), log_norm.shape[1]) :])
    X_hvg = log_norm[:, hvg_idx].toarray().astype(np.float32)
    pca = PCA(
        n_components=min(int(pca_dim), X_hvg.shape[0], X_hvg.shape[1]), svd_solver="randomized", random_state=seed
    )
    X = pca.fit_transform(X_hvg).astype(np.float32)

    gene_list = pd.read_csv(dissociation_genes).iloc[:, 0].dropna().astype(str).tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names.tolist())}
    matched = [gene_to_idx[g] for g in gene_list if g in gene_to_idx]
    if len(matched) < max(20, int(0.5 * len(gene_list))):
        raise RuntimeError(f"Too few dissociation genes matched: {len(matched)} / {len(gene_list)}")
    dissociation = _log_normalize(counts[:, np.asarray(matched)].tocsr(), totals, scale=1e6)
    scores = _zscore(np.asarray(dissociation.mean(axis=1)).ravel())

    output.mkdir(parents=True, exist_ok=True)
    row_ids = sampled["cell"].astype(str).to_numpy()
    np.savez_compressed(output / "full_data.npz", X=X, S=scores, row_ids=row_ids)
    table = sampled.copy()
    table.insert(0, "id", row_ids)
    table["dissociation_score"] = scores
    table.to_parquet(output / "rows.parquet", index=False)
    manifest = {
        "dataset": "Tabula Muris FACS",
        "n_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "score": "z-scored mean log1p(CPM) over matched dissociation genes",
        "dissociation_genes_used": int(len(matched)),
        "artifacts": {"full_data": "full_data.npz", "rows": "rows.parquet"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
