#!/usr/bin/env python3

import os
os.environ["MPLCONFIGDIR"] = "/varidata/research/projects/jang/TommyGoralski/tmp/matplotlib"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


TRITON_COLOR = "#ffb000"


def plot_k_sweep(json_path: Path, output_dir: Path):
    # ---------------- Load ----------------
    with open(json_path) as f:
        data = json.load(f)

    T = data.get("config", {}).get("T", "unknown")
    C = data.get("config", {}).get("C", "unknown")

    results = pd.DataFrame(data["results"])
    results = results[results["status"] == "success"].copy()
    results = results.sort_values("K")

    # ---------------- Metrics ----------------
    results["time_min"] = results["time_ms_median"] / 1000 / 60
    results["time_per_K_sec"] = (results["time_ms_median"] / 1000) / results["K"]

    # Choose best available memory column
    if "peak_reserved_gb" in results.columns:
        results["memory_gb"] = results["peak_reserved_gb"]
    else:
        results["memory_gb"] = results["peak_allocated_gb"]

    # ---------------- Figure ----------------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ============================================================
    # 1) Runtime vs K
    # ============================================================
    ax = axes[0]
    ax.plot(
        results["K"], results["time_min"],
        marker="o", color=TRITON_COLOR,
        linewidth=2, markersize=6,
        label="triton_streaming"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(results["K"])
    ax.set_xticklabels(results["K"])

    ax.yaxis.set_major_formatter(mticker.LogFormatter(labelOnlyBase=False))

    ax.set_xlabel("Max segment length $K$")
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Runtime vs $K$")

    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # slope annotation
    logK = np.log10(results["K"])
    logT = np.log10(results["time_min"])
    slope = np.polyfit(logK, logT, 1)[0]

    ax.text(
        0.05, 0.92, f"Slope ≈ {slope:.2f}",
        transform=ax.transAxes,
        fontsize=10, verticalalignment="top"
    )

    # ============================================================
    # 2) Time per K
    # ============================================================
    ax = axes[1]
    ax.plot(
        results["K"], results["time_per_K_sec"],
        marker="o", color=TRITON_COLOR,
        linewidth=2, markersize=6,
        label="triton_streaming"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(results["K"])
    ax.set_xticklabels(results["K"])

    ax.yaxis.set_major_formatter(mticker.LogFormatter(labelOnlyBase=False))

    ax.set_xlabel("Max segment length $K$")
    ax.set_ylabel("Time / $K$ (seconds)")
    ax.set_title("Normalized Runtime")

    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # ============================================================
    # 3) Memory vs K  (NEW)
    # ============================================================
    ax = axes[2]
    ax.plot(
        results["K"], results["memory_gb"],
        marker="o", color=TRITON_COLOR,
        linewidth=2, markersize=6,
        label="triton_streaming"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(results["K"])
    ax.set_xticklabels(results["K"])

    ax.yaxis.set_major_formatter(mticker.LogFormatter(labelOnlyBase=False))

    ax.set_xlabel("Max segment length $K$")
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("Memory vs $K$")

    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # ============================================================
    # Global title
    # ============================================================
    fig.suptitle(
        f"Runtime and Memory Scaling\n($T$={T:,}, $C$={C})",
        fontsize=13
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # ---------------- Save ----------------
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "k_sweep_full.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "k_sweep_full.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plot_k_sweep(args.input, args.output_dir)


if __name__ == "__main__":
    main()
