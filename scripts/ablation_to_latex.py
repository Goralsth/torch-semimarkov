#!/usr/bin/env python3
"""Convert ablate_zero_centering.py JSON output to LaTeX tables for the manuscript.

Reads the JSON exported by ``ablate_zero_centering.py --json`` and emits
publication-ready LaTeX tables (booktabs format).

Usage:
    # Run ablation and export JSON
    python3 scripts/ablate_zero_centering.py --json results/ablation.json

    # Convert to LaTeX (stdout)
    python3 scripts/ablation_to_latex.py results/ablation.json

    # Write individual .tex files
    python3 scripts/ablation_to_latex.py results/ablation.json --outdir docs/manuscript/tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ======================================================================
# LaTeX formatters
# ======================================================================


def fmt(val, decimals=2):
    """Format a float for LaTeX."""
    if abs(val) < 0.01 and val != 0:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def mode_label(mode):
    """LaTeX-formatted mode name."""
    return {
        "mean": r"\textsc{mean}",
        "position": r"\textsc{position}",
        "none": r"\textsc{none}",
    }[mode]


CENTERING_MODES = ["mean", "position", "none"]


# ======================================================================
# Table generators
# ======================================================================


def stability_table(data):
    """§1: Numerical stability table."""
    config = data["config"]
    rows = data["rows"]
    T_values = sorted({r["T"] for r in rows})

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cumulative score magnitude scaling under three centering strategies.")
    lines.append(r"  \textsc{mean} centering achieves $O(\sqrt{T})$ growth, while")
    lines.append(r"  \textsc{position} and \textsc{none} grow $O(T)$.")
    lines.append(
        r"  Config: $B{=}" + str(config["batch"]) + r"$, "
        r"$C{=}" + str(config["C"]) + r"$, "
        r"$K{=}" + str(config["K"]) + r"$.}"
    )
    lines.append(r"\label{tab:stability}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$T$ & Mode & Endpoint & Peak & Peak/$\sqrt{T}$ & Peak/$T$ \\")
    lines.append(r"\midrule")

    for i, T in enumerate(T_values):
        if i > 0:
            lines.append(r"\addlinespace")
        for mode in CENTERING_MODES:
            r = next(x for x in rows if x["T"] == T and x["mode"] == mode)
            lines.append(
                f"  {T:,} & {mode_label(mode)} & "
                f"{fmt(r['endpoint_mag'])} & {fmt(r['peak_mag'])} & "
                f"{fmt(r['peak_sqrt_t'])} & {fmt(r['peak_over_t'], 4)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def combined_prior_table(scenario_data):
    """Combined table: implicit duration prior + segment redistribution effect.

    Main text table showing cause (penalty) and effect (segment counts) together.
    """
    penalty_rows = scenario_data["penalty"]
    viterbi_rows = scenario_data["viterbi"]

    # Index viterbi data by (label, mode)
    viterbi_idx = {}
    for r in viterbi_rows:
        viterbi_idx[(r["label"], r["mode"])] = r

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Implicit duration prior from \textsc{mean} centering under label imbalance."
    )
    lines.append(
        r"  The per-label temporal mean $\nu_{b,c}$ creates a duration penalty $-\nu_{b,c} \cdot k$"
    )
    lines.append(
        r"  that penalizes long segments of prevalent labels and redistributes probability mass"
    )
    lines.append(
        r"  to rare labels. Right columns show downstream effect on decoded segmentation.}"
    )
    lines.append(r"\label{tab:implicit-prior}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{3}{c}{Duration penalty $-\nu \cdot k$}")
    lines.append(r" & \multicolumn{2}{c}{Segments} \\")
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-7}")
    lines.append(r"Label & $\nu_{b,c}$ & $k{=}10$ & $k{=}25$ & $k{=}50$")
    lines.append(r" & \textsc{mean} & \textsc{none} \\")
    lines.append(r"\midrule")

    for r in penalty_rows:
        label = r["label"]
        mean_segs = viterbi_idx.get((label, "mean"), {})
        none_segs = viterbi_idx.get((label, "none"), {})
        n_mean = mean_segs.get("n_segments", "--")
        n_none = none_segs.get("n_segments", "--")
        lines.append(
            f"  {label} & {fmt(r['nu'], 3)} & "
            f"{fmt(r['k10'])} & {fmt(r['k25'])} & {fmt(r['k50'])} & "
            f"{n_mean} & {n_none} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def penalty_table(penalty_rows, scenario):
    """§3 Part A: Duration penalty table."""
    scenario_label = "imbalanced" if scenario == "imbalanced" else "balanced"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Implicit duration penalty $-\nu_{b,c} \cdot k$ under \textsc{mean} centering"
        f" ({scenario_label} labels)."
    )
    lines.append(r"  Globally-prevalent labels (e.g., intron at 70\%) incur a large per-step cost,")
    lines.append(r"  while rare labels receive a duration bonus.}")
    lines.append(r"\label{tab:penalty-" + scenario + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Label & $\nu_{b,c}$ & $k{=}1$ & $k{=}5$ & $k{=}10$ & $k{=}25$ & $k{=}50$ \\")
    lines.append(r"\midrule")

    for r in penalty_rows:
        lines.append(
            f"  {r['label']} & {fmt(r['nu'], 3)} & "
            f"{fmt(r['k1'])} & {fmt(r['k5'])} & {fmt(r['k10'])} & "
            f"{fmt(r['k25'])} & {fmt(r['k50'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def partition_table(partition_rows, scenario):
    """§3 Part B: Partition function and divergence table."""
    scenario_label = "imbalanced" if scenario == "imbalanced" else "balanced"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Partition function and marginal divergence under"
        f" {scenario_label} label proportions."
    )
    lines.append(
        r"  TV distance is measured against \textsc{position} (path-invariant reference).}"
    )
    lines.append(r"\label{tab:partition-" + scenario + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l r r r}")
    lines.append(r"\toprule")
    lines.append(r"Mode & $\log Z$ & TV vs \textsc{position} & Marginal std \\")
    lines.append(r"\midrule")

    for r in partition_rows:
        lines.append(
            f"  {mode_label(r['mode'])} & {fmt(r['log_z'], 1)} & "
            f"{fmt(r['tv'], 4)} & {fmt(r['marginal_std'], 4)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def viterbi_table(viterbi_rows, scenario):
    """§3 Part C: Viterbi segment lengths table."""
    scenario_label = "imbalanced" if scenario == "imbalanced" else "balanced"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Viterbi segment lengths by label under" f" {scenario_label} label proportions."
    )
    lines.append(
        r"  \textsc{mean} centering redistributes segments from prevalent"
        r" to rare labels under imbalance.}"
    )
    lines.append(r"\label{tab:viterbi-" + scenario + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Label & Mode & Segments & Mean len & Min & Max \\")
    lines.append(r"\midrule")

    labels_seen = []
    for r in viterbi_rows:
        if r["label"] not in labels_seen:
            if labels_seen:
                lines.append(r"\addlinespace")
            labels_seen.append(r["label"])
        lines.append(
            f"  {r['label']} & {mode_label(r['mode'])} & "
            f"{r['n_segments']} & {fmt(r['mean_len'], 1)} & "
            f"{r['min_len']} & {r['max_len']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert zero-centering ablation JSON to LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="JSON file from ablate_zero_centering.py --json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Write .tex files to this directory (default: print to stdout)",
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    tables = {}

    # §1: Stability table
    if "stability" in data:
        tables["stability"] = stability_table(data["stability"])

    # §3: Prior tables
    if "prior" in data:
        scenarios = data["prior"]["scenarios"]

        # Combined main-text table (imbalanced only)
        if "imbalanced" in scenarios:
            tables["implicit_prior"] = combined_prior_table(scenarios["imbalanced"])

        # Individual supplement tables
        for scenario in ["imbalanced", "balanced"]:
            if scenario not in scenarios:
                continue
            d = scenarios[scenario]
            tables[f"penalty_{scenario}"] = penalty_table(d["penalty"], scenario)
            tables[f"partition_{scenario}"] = partition_table(d["partition"], scenario)
            tables[f"viterbi_{scenario}"] = viterbi_table(d["viterbi"], scenario)

    if not tables:
        print("No data found in JSON. Run ablation with --section all.", file=sys.stderr)
        return 1

    # Output
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        for name, tex in tables.items():
            path = args.outdir / f"tab_{name}.tex"
            path.write_text(tex + "\n")
            print(f"  Wrote {path}", file=sys.stderr)
    else:
        for name, tex in tables.items():
            print(f"% === {name} ===")
            print(tex)
            print()

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
