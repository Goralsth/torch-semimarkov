#!/usr/bin/env python3
"""Plot training convergence curves comparing K=1 and K=30 from Lightning CSV logs.

Usage:
    python3 benchmarks/practical_demonstration/timit/plot_convergence.py \
        --k1-csv lightning_logs/version_0/metrics.csv \
        --k30-csv lightning_logs/version_1/metrics.csv \
        --output results/timit_seqbounds_f64/convergence.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must follow matplotlib.use()
import pandas as pd  # noqa: E402


def load_val_metrics(csv_path: str | Path) -> pd.DataFrame:
    """Load a Lightning metrics CSV and return one row per epoch with val metrics."""
    df = pd.read_csv(csv_path)
    # Val metrics are logged once per epoch; drop rows without val/per
    val = df.dropna(subset=["val/per"]).copy()
    val = val.reset_index(drop=True)
    val["epoch_num"] = val["epoch"].astype(int)
    return val


def plot_convergence(
    k1_csv: str | Path,
    k30_csv: str | Path,
    out_path: str | Path,
):
    k1 = load_val_metrics(k1_csv)
    k30 = load_val_metrics(k30_csv)

    panels = [
        ("val/per", "Phone Error Rate (PER)", True),
        ("val/boundary_f1", "Boundary F1", False),
        ("val/segment_f1", "Segment F1", False),
        ("val/entropy_mean", "Boundary Entropy (mean)", False),
        ("val/position_entropy_mean", "Position Entropy (mean)", False),
        ("val/loss", "Validation Loss", True),
    ]

    # Drop panels not present in both CSVs
    panels = [
        (col, title, lower)
        for col, title, lower in panels
        if col in k1.columns and col in k30.columns
    ]

    nrows = 2
    ncols = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout="constrained")
    axes = axes.flatten()

    for i, (col, title, lower_better) in enumerate(panels):
        ax = axes[i]
        ax.plot(k1["epoch_num"], k1[col], color="orange", lw=1.5, label="K=1 (Linear CRF)")
        ax.plot(k30["epoch_num"], k30[col], color="steelblue", lw=1.5, label="K=30 (Semi-CRF)")

        # Mark best epoch
        if lower_better:
            k1_best = k1.loc[k1[col].idxmin()]
            k30_best = k30.loc[k30[col].idxmin()]
        else:
            k1_best = k1.loc[k1[col].idxmax()]
            k30_best = k30.loc[k30[col].idxmax()]

        ax.axvline(k1_best["epoch_num"], color="orange", ls="--", alpha=0.4, lw=0.8)
        ax.axvline(k30_best["epoch_num"], color="steelblue", ls="--", alpha=0.4, lw=0.8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.tick_params(labelsize=8)

        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Training Convergence: K=1 Linear CRF vs K=30 Semi-CRF", fontsize=12)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot K=1 vs K=30 training convergence")
    parser.add_argument("--k1-csv", type=Path, required=True, help="K=1 Lightning metrics CSV")
    parser.add_argument("--k30-csv", type=Path, required=True, help="K=30 Lightning metrics CSV")
    parser.add_argument(
        "--output", type=Path, default=Path("convergence.png"), help="Output PNG path"
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_convergence(args.k1_csv, args.k30_csv, args.output)


if __name__ == "__main__":
    main()
