"""Configured sweeps with isolated workers, resumable outputs, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from isdr.data import load_dataset, split_data
from isdr.evaluation.metrics import compute_all_metrics
from isdr.methods import get_method
from isdr.plotting import plot_embedding, plot_pareto


DEFAULT_POINT_TIMEOUT_SECONDS = 7200


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=_jsonable)


def _load_config(config_path: str | Path) -> tuple[Path, Path, dict]:
    config_path = Path(config_path).resolve()
    root = Path.cwd()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return config_path, root, config


def _run_context(split: dict, config: dict) -> dict:
    digest = hashlib.sha256()
    for key in ("X_train", "S_train", "X_test", "S_test", "train_idx", "test_idx"):
        array = split.get(key)
        if array is not None:
            array = np.ascontiguousarray(array)
            digest.update(str((key, array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parents[1].rglob("*.py")):
        source.update(path.relative_to(Path(__file__).parents[1]).as_posix().encode())
        source.update(path.read_bytes())
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "numba",
        "umap-learn",
        "pacmap",
        "cebra",
        "rfphate",
        "rfae",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "seed": int(config.get("seed", 42)),
        "device": config.get("device", "cpu"),
        "evaluation": config.get("evaluation", {}),
        "data_sha256": digest.hexdigest(),
        "source_sha256": source.hexdigest(),
        "versions": versions,
        "score_type": split["score_type"],
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
    }


def _build_params(base_params: dict | None, param_name: str | None, value):
    params = deepcopy(base_params or {})
    if not param_name or param_name == "_dummy":
        return params
    pieces = str(param_name).split(".")
    cursor = params
    for piece in pieces[:-1]:
        cursor[piece] = dict(cursor.get(piece, {}))
        cursor = cursor[piece]
    cursor[pieces[-1]] = value
    return params


def _value_key(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".15g")
    return str(value)


def _point_dir(output_dir: Path, method_name: str, param_name: str | None, key: str) -> Path:
    return output_dir / method_name / ("default" if param_name == "_dummy" else f"{param_name}={key}")


def _point_label(method_name: str, param_name: str | None, value) -> str:
    if param_name == "_dummy":
        return f"{method_name}/default"
    return f"{method_name}/{param_name}={_value_key(value)}"


def _execution_options(config: dict) -> tuple[str, float | None, bool]:
    execution = dict(config.get("execution") or {})
    if "isolate_methods" in execution and "mode" not in execution:
        mode = "subprocess" if bool(execution["isolate_methods"]) else "in_process"
    else:
        mode = str(execution.get("mode", "subprocess"))
    mode = mode.lower().replace("-", "_")
    if mode in {"inline", "inprocess"}:
        mode = "in_process"
    if mode in {"isolated", "process"}:
        mode = "subprocess"
    if mode not in {"subprocess", "in_process"}:
        raise ValueError("execution.mode must be 'subprocess' or 'in_process'")

    timeout_raw = execution.get("timeout_seconds", DEFAULT_POINT_TIMEOUT_SECONDS)
    if timeout_raw in {None, 0, "0", "none", "None"}:
        timeout = None
    else:
        timeout = float(timeout_raw)
        if timeout <= 0:
            timeout = None
    resume = bool(execution.get("resume", True))
    return mode, timeout, resume


def _prepare_dataset(root: Path, config: dict, seed: int, output_dir: Path, write_shared: bool) -> dict:
    dataset_cfg = dict(config.get("dataset", {}))
    for key in ["data_root", "processed_dir"]:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            dataset_cfg[key] = str(_resolve_path(root, dataset_cfg[key]))
    bundle = load_dataset(dataset_cfg, seed=seed)
    split = split_data(
        bundle.X, bundle.scores, float(dataset_cfg.get("test_ratio", 0.0)), int(dataset_cfg.get("split_seed", 42))
    )
    split["score_type"] = bundle.score_type
    split["context"] = _run_context(split, config)
    if write_shared:
        np.savez_compressed(
            output_dir / "split_data.npz", **{k: v for k, v in split.items() if isinstance(v, np.ndarray)}
        )
        _write_json(
            output_dir / "dataset_info.json",
            {
                "name": bundle.name,
                "score_type": bundle.score_type,
                "metadata": bundle.metadata,
                "n_train": split["n_train"],
                "n_test": split["n_test"],
            },
        )
        _write_json(output_dir / "run_context.json", split["context"])
    return split


def _load_completed_point(
    point_dir: Path, method_name: str, param_name: str | None, value, params: dict, context: dict
) -> dict | None:
    metrics_path = point_dir / "metrics.json"
    if not metrics_path.exists() or not (point_dir / "Z_train.npy").exists():
        return None
    try:
        with metrics_path.open() as handle:
            entry = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if entry.get("method") != method_name:
        return None
    if entry.get("param_name") != param_name:
        return None
    if _value_key(entry.get("param_value")) != _value_key(value):
        return None
    if entry.get("params", {}) != params:
        return None
    if entry.get("context") != context or (point_dir / "error.json").exists():
        return None
    try:
        embedding = np.load(point_dir / "Z_train.npy", mmap_mode="r")
        if embedding.ndim != 2 or embedding.shape[1] != 2 or not np.isfinite(embedding).all():
            return None
    except (OSError, ValueError):
        return None
    return entry


def _synchronize_cuda(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def _run_point(split: dict, output_dir: Path, method_cfg: dict, value, seed: int, device: str, knn_k: int) -> dict:
    method_name = str(method_cfg["name"])
    param_name = method_cfg.get("param_name", "_dummy")
    base_params = method_cfg.get("base_params", {})
    key = _value_key(value)
    point_dir = _point_dir(output_dir, method_name, param_name, key)
    point_dir.mkdir(parents=True, exist_ok=True)
    params = _build_params(base_params, param_name, value)
    method_fn = get_method(method_name)
    _synchronize_cuda(device)
    t0 = time.perf_counter()
    result = method_fn(
        split["X_train"],
        split["S_train"],
        split["X_test"],
        split["S_test"],
        params=params,
        output_dir=str(point_dir),
        seed=seed,
        device=device,
    )
    _synchronize_cuda(device)
    elapsed = time.perf_counter() - t0
    Z_train = np.asarray(result["Z_train"], dtype=np.float32)
    np.save(point_dir / "Z_train.npy", Z_train)
    if result.get("Z_test") is not None:
        np.save(point_dir / "Z_test.npy", np.asarray(result["Z_test"], dtype=np.float32))
    else:
        (point_dir / "Z_test.npy").unlink(missing_ok=True)
    t_metrics = time.perf_counter()
    train_metrics = compute_all_metrics(
        Z_train, split["X_train"], split["S_train"], knn_k=knn_k, score_type=split["score_type"]
    )
    entry = {
        "method": method_name,
        "param_name": param_name,
        "param_value": value,
        "time_seconds": float(elapsed),
        "metric_seconds": time.perf_counter() - t_metrics,
        "seed": seed,
        "context": split["context"],
        "train_metrics": train_metrics,
        "params": params,
    }
    plot_embedding(
        Z_train,
        split["S_train"],
        point_dir / "embedding_train.png",
        f"{method_name} {param_name}={value}",
        score_type=split["score_type"],
    )
    _write_json(point_dir / "metrics.json", entry)
    error_path = point_dir / "error.json"
    if error_path.exists():
        error_path.unlink()
    return entry


def run_point_from_config(config_path: str | Path, method_index: int, value_index: int) -> dict:
    config_path, root, config = _load_config(config_path)
    seed = int(config.get("seed", 42))
    output_dir = _resolve_path(root, config.get("output_dir", "results/run"))
    eval_cfg = config.get("evaluation", {})
    knn_k = int(eval_cfg.get("knn_k", 10))
    device = str(config.get("device", "cpu"))
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    split = _prepare_dataset(root, config, seed, output_dir, write_shared=False)
    method_cfg = config.get("methods", [])[method_index]
    values = method_cfg.get("values", [0])
    return _run_point(split, output_dir, method_cfg, values[value_index], seed, device, knn_k)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


def _wait_subprocess(cmd: list[str], root: Path, timeout: float | None) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    popen_kwargs: dict[str, Any] = {"cwd": str(root), "env": env}
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        raise
    except BaseException:
        _terminate_process_group(proc)
        raise


def _run_point_subprocess(
    config_path: Path,
    root: Path,
    output_dir: Path,
    method_index: int,
    value_index: int,
    method_cfg: dict,
    value,
    timeout: float | None,
    context: dict,
) -> dict:
    method_name = str(method_cfg["name"])
    param_name = method_cfg.get("param_name", "_dummy")
    params = _build_params(method_cfg.get("base_params", {}), param_name, value)
    key = _value_key(value)
    point_dir = _point_dir(output_dir, method_name, param_name, key)
    point_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "isdr.evaluation.sweep",
        "--run-point",
        "--config",
        str(config_path),
        "--method-index",
        str(method_index),
        "--value-index",
        str(value_index),
    ]
    try:
        return_code = _wait_subprocess(cmd, root, timeout)
    except subprocess.TimeoutExpired as exc:
        _write_json(
            point_dir / "error.json",
            {
                "method": method_name,
                "param_name": param_name,
                "param_value": value,
                "error": "timeout",
                "timeout_seconds": timeout,
            },
        )
        raise TimeoutError(f"{_point_label(method_name, param_name, value)} timed out after {timeout} seconds") from exc
    if return_code != 0:
        _write_json(
            point_dir / "error.json",
            {
                "method": method_name,
                "param_name": param_name,
                "param_value": value,
                "error": "subprocess_failed",
                "return_code": return_code,
            },
        )
        raise RuntimeError(f"{_point_label(method_name, param_name, value)} failed with exit code {return_code}")
    entry = _load_completed_point(point_dir, method_name, param_name, value, params, context)
    if entry is None:
        _write_json(
            point_dir / "error.json",
            {
                "method": method_name,
                "param_name": param_name,
                "param_value": value,
                "error": "missing_metrics",
            },
        )
        raise RuntimeError(
            f"{_point_label(method_name, param_name, value)} has missing artifacts or a mismatched run context; check whether code or data changed during execution"
        )
    return entry


def run_sweep(config_path: str | Path) -> dict:
    """Run a YAML grid and save per-point results; relative paths use the current directory."""
    config_path, root, config = _load_config(config_path)

    seed = int(config.get("seed", 42))
    output_dir = _resolve_path(root, config.get("output_dir", "results/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    mode, timeout, resume = _execution_options(config)
    split = _prepare_dataset(root, config, seed, output_dir, write_shared=True)

    eval_cfg = config.get("evaluation", {})
    knn_k = int(eval_cfg.get("knn_k", 10))
    device = str(config.get("device", "cpu"))
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild the index from the requested grid, so removed points cannot leak in.
    results: dict[str, dict] = {}
    _write_json(output_dir / "sweep_results.json", results)
    for method_index, method_cfg in enumerate(config.get("methods", [])):
        method_name = str(method_cfg["name"])
        param_name = method_cfg.get("param_name", "_dummy")
        values = method_cfg.get("values", [0])
        base_params = method_cfg.get("base_params", {})
        results.setdefault(method_name, {})
        for value_index, value in enumerate(values):
            key = _value_key(value)
            params = _build_params(base_params, param_name, value)
            point_dir = _point_dir(output_dir, method_name, param_name, key)
            entry = (
                _load_completed_point(point_dir, method_name, param_name, value, params, split["context"])
                if resume
                else None
            )
            if entry is not None:
                print(f"[skip-point] {_point_label(method_name, param_name, value)}", flush=True)
            else:
                print(f"[point] {_point_label(method_name, param_name, value)} mode={mode}", flush=True)
                _write_json(point_dir / "error.json", {"error": "incomplete", "method": method_name})
                try:
                    if mode == "subprocess":
                        entry = _run_point_subprocess(
                            config_path,
                            root,
                            output_dir,
                            method_index,
                            value_index,
                            method_cfg,
                            value,
                            timeout,
                            split["context"],
                        )
                    else:
                        entry = _run_point(split, output_dir, method_cfg, value, seed, device, knn_k)
                except Exception as exc:
                    _write_json(point_dir / "error.json", {"method": method_name, "params": params, "error": repr(exc)})
                    raise
            results[method_name][key] = entry
            _write_json(output_dir / "sweep_results.json", results)

    plot_pareto(
        results,
        output_dir / "plots" / "pareto_train.png",
        str(eval_cfg.get("x_metric", "trustworthiness")),
        str(eval_cfg.get("y_metric", "score_dirichlet_energy")),
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-point", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method-index", type=int, required=True)
    parser.add_argument("--value-index", type=int, required=True)
    args = parser.parse_args(argv)
    if not args.run_point:
        raise ValueError("This module is intended for internal --run-point execution.")
    run_point_from_config(args.config, args.method_index, args.value_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
