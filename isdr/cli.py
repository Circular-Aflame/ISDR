from __future__ import annotations

import argparse

from isdr.data import build_regression_benchmark, build_tabula_muris_facs, regression_dataset_keys
from isdr.evaluation import run_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="isdr")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a configured benchmark sweep")
    run_p.add_argument("--config", required=True)

    tabula_p = sub.add_parser("prepare-tabula-muris", help="Build the public Tabula Muris FACS dataset")
    tabula_p.add_argument("--archive", required=True)
    tabula_p.add_argument("--dissociation-genes", required=True)
    tabula_p.add_argument("--output", required=True)
    tabula_p.add_argument("--sample-size", type=int, default=None)
    tabula_p.add_argument("--seed", type=int, default=42)
    tabula_p.add_argument("--n-hvg", type=int, default=2000)
    tabula_p.add_argument("--pca-dim", type=int, default=50)

    reg_p = sub.add_parser("prepare-regression-benchmarks", help="Build public tabular regression benchmark caches")
    reg_p.add_argument("--dataset", default="all", choices=["all", *regression_dataset_keys()])
    reg_p.add_argument("--data-root", default="data")
    reg_p.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_sweep(args.config)
        return 0
    if args.command == "prepare-tabula-muris":
        out = build_tabula_muris_facs(
            archive=args.archive,
            output=args.output,
            dissociation_genes=args.dissociation_genes,
            sample_size=args.sample_size,
            seed=args.seed,
            n_hvg=args.n_hvg,
            pca_dim=args.pca_dim,
        )
        print(f"Wrote {out}")
        return 0
    if args.command == "prepare-regression-benchmarks":
        outputs = build_regression_benchmark(args.dataset, data_root=args.data_root, force=args.force)
        for X_path, s_path in outputs:
            print(f"Wrote {X_path} and {s_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
