#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize mean/std of scaled metrics across multiple seed runs."
    )
    parser.add_argument(
        "--base-name",
        required=True,
        help="Base run name without seed suffix, e.g. anchor_refine_adv4_5_la0_25_k10_st20_sim0_06_m0_012",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed list, e.g. 0 1 2 3 4",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("experiments/BMMC_codes"),
        help="Root directory containing metrics_<run_name> folders.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for summary table.",
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        run_name = f"{args.base_name}_s{seed}"
        csv_path = args.metrics_root / f"metrics_{run_name}" / "scaled_metrics_local.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        if run_name not in df.index:
            raise KeyError(f"Row '{run_name}' not found in {csv_path}")
        row = pd.to_numeric(df.loc[run_name], errors="coerce")
        row.name = seed
        rows.append(row)

    seed_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "mean": seed_df.mean(axis=0),
            "std": seed_df.std(axis=0, ddof=1),
            "n": len(args.seeds),
        }
    )

    print(f"Base name: {args.base_name}")
    print(f"Seeds: {args.seeds}")
    print("")
    print(summary.round(6).to_string())

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv)
        print(f"\nSaved summary: {args.out_csv}")


if __name__ == "__main__":
    main()
