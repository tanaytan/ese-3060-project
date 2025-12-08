#!/usr/bin/env python3
"""
analyze_results.py

Usage:
    python analyze_results.py --csv cifar10/logs/part1_cifar10.csv

Requires:
    pip install pandas
"""

import argparse
import pandas as pd
from pathlib import Path


def summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by config (experiment, git_hash, tta_level, train_epochs) and
    compute stats over runs.
    """
    group_cols = ["experiment", "git_hash", "tta_level", "train_epochs"]

    agg = df.groupby(group_cols).agg(
        n_runs=("final_tta_acc", "size"),
        mean_time=("total_time_seconds", "mean"),
        std_time=("total_time_seconds", "std"),
        mean_acc=("final_tta_acc", "mean"),
        std_acc=("final_tta_acc", "std"),
    )

    # Sort by experiment then time
    agg = agg.reset_index().sort_values(
        by=["experiment", "train_epochs", "tta_level"]
    )

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to part1_cifar10.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write summary CSV (default: cifar10/logs/summary_by_config.csv)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Make sure the important columns exist
    required_cols = ["experiment", "git_hash",
                     "total_time_seconds", "final_tta_acc"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    # Summarize
    summary = summarize_by_config(df)

    # Pretty print to console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print("\n=== Summary by (experiment, git_hash, tta_level, train_epochs) ===\n")
    print(summary.to_string(index=False))

    # Optionally write to CSV
    if args.out is None:
        out_path = csv_path.parent / "summary_by_config.csv"
    else:
        out_path = Path(args.out)

    summary.to_csv(out_path, index=False)
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
