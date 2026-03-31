#!/usr/bin/env python3
"""T-sweep benchmark: wall time and peak memory vs sequence length.

Holds K and C fixed, sweeps T across orders of magnitude. Benchmarks all
requested backends on the same T values for direct comparison. Uses the
shared ``lib`` runner infrastructure for consistent backend dispatch.

Generates publication figures:
  - Fig A: Log-log wall time vs T (one line per backend)
  - Fig B: Peak GPU memory vs T (with theoretical O(TKC²) reference)
  - Fig C: Speedup vs T (each backend relative to slowest)

Place this script in the ``benchmarks/`` directory alongside
``benchmark_memory_analysis.py`` so that ``from lib import ...`` resolves.

Usage:
    # Default: Triton + Streaming scan, K=50, C=24 (custom sweep)
    python benchmarks/benchmark_t_sweep.py

    # All backends (exact ones will OOM at large T — that's the point)
    python benchmarks/benchmark_t_sweep.py \\
        --backends triton_streaming,linear_scan_streaming,linear_scan_vectorized,binary_tree_sharded

    # Genomics scale: K=1000, C=32
    python benchmarks/benchmark_t_sweep.py --K 1000 --C 32 --B 1

    # Quick mode
    python benchmarks/benchmark_t_sweep.py --quick

    # Custom T values
    python benchmarks/benchmark_t_sweep.py --T 100,500,1000,5000,10000,50000

    # Forward-only timing (isolate O(KC) memory claim)
    python benchmarks/benchmark_t_sweep.py --phase forward

    # Skip plotting (just collect data)
    python benchmarks/benchmark_t_sweep.py --no-plot
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from lib import (
    BenchmarkResult,
    run_single_benchmark,
    should_skip_config,
)
from lib.runner import (
    LOG_MAX_BACKENDS,
    LOG_SEMIRING_ONLY_BACKENDS,
)
from lib.sampling import parse_int_list

try:
    from flash_semicrf.streaming import HAS_TRITON
except ImportError:
    HAS_TRITON = False


# ======================================================================
# Default configurations
# ======================================================================

DEFAULT_T_VALUES = [100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
QUICK_T_VALUES = [100, 500, 2000, 10000]

DEFAULT_BACKENDS = ["triton_streaming", "linear_scan_streaming"]

ALL_BACKENDS = [
    "triton_streaming",
    "linear_scan_streaming",
    "linear_scan_vectorized",
    "linear_scan",
    "binary_tree",
    "binary_tree_sharded",
]

REGIMES = {
    "ner": {
        "description": "NER-scale (CoNLL-2003 BIO tagging, C=9, KC=144)",
        "K": 16,
        "C": 9,
        "B": 4,
        "T": [64, 128, 256, 512, 1024, 2048, 4096],
        "backends": ALL_BACKENDS,
    },
    "speech": {
        "description": "Speech segmentation (TIMIT phone boundaries, C=39, KC=1170)",
        "K": 30,
        "C": 39,
        "B": 4,
        "T": [200, 300, 500, 1000, 2000, 5000, 10000, 20000],
        "backends": ["triton_streaming", "linear_scan_streaming"],
    },
    "streaming": {
        "description": "Genomics-scale streaming-only (chromosome-length)",
        "K": 200,
        "C": 6,
        "B": 2,
        "T": [10000, 25000, 50000, 100000, 250000, 500000, 1000000],
        "backends": ["triton_streaming", "linear_scan_streaming"],
    },
}

# Plotting colors per backend
BACKEND_STYLE = {
    "triton_streaming": {"color": "#2563eb", "marker": "o", "ls": "-", "label": "Triton streaming"},
    "linear_scan_streaming": {
        "color": "#dc2626",
        "marker": "s",
        "ls": "--",
        "label": "Streaming scan",
    },
    "linear_scan_vectorized": {
        "color": "#16a34a",
        "marker": "^",
        "ls": "-.",
        "label": "Vectorized scan",
    },
    "binary_tree_sharded": {
        "color": "#9333ea",
        "marker": "D",
        "ls": ":",
        "label": "Binary tree (sharded)",
    },
    "linear_scan": {"color": "#ea580c", "marker": "v", "ls": "--", "label": "Linear scan"},
    "binary_tree": {"color": "#64748b", "marker": "p", "ls": ":", "label": "Binary tree"},
    "block_triangular": {
        "color": "#0d9488",
        "marker": "h",
        "ls": "-.",
        "label": "Block triangular",
    },
    "banded": {"color": "#ca8a04", "marker": "*", "ls": "--", "label": "Banded"},
}
THEORY_COLOR = "#9ca3af"


def _format_throughput(tp: float) -> str:
    """Format throughput (positions/second) with appropriate unit.

    Handles range from < 1 pos/s to > 1G pos/s.
    """
    if tp <= 0:
        return "---"
    elif tp >= 1e9:
        return f"{tp/1e9:.1f}G pos/s"
    elif tp >= 1e6:
        return f"{tp/1e6:.1f}M pos/s"
    elif tp >= 1e3:
        return f"{tp/1e3:.1f}K pos/s"
    elif tp >= 1:
        return f"{tp:.1f} pos/s"
    else:
        return f"{tp:.2f} pos/s"


# ======================================================================
# Sweep runner
# ======================================================================


def run_sweep(
    T_values: list[int],
    K: int,
    C: int,
    B: int,
    backends: list[str],
    device: torch.device,
    repeats: int,
    phase: str,
    semiring: str,
    max_memory_gb: float,
) -> list[BenchmarkResult]:
    """Run all (backend x T) combinations, skipping after OOM."""

    results: list[BenchmarkResult] = []

    # Track OOM per backend so we stop early
    oom_history: dict[str, list[tuple[int, int, int]]] = {b: [] for b in backends}

    for backend in backends:
        # Check semiring compatibility up front
        if backend in LOG_SEMIRING_ONLY_BACKENDS and semiring != "Log":
            print(f"\n  SKIP {backend}: only supports Log semiring (got {semiring})")
            for T in T_values:
                results.append(
                    _skip_result(
                        T,
                        K,
                        C,
                        B,
                        backend,
                        semiring,
                        phase,
                        f"{backend} only supports Log semiring",
                    )
                )
            continue
        if backend in LOG_MAX_BACKENDS and semiring not in {"Log", "Max"}:
            print(f"\n  SKIP {backend}: only supports Log/Max semirings (got {semiring})")
            for T in T_values:
                results.append(
                    _skip_result(
                        T,
                        K,
                        C,
                        B,
                        backend,
                        semiring,
                        phase,
                        f"{backend} only supports Log/Max semirings",
                    )
                )
            continue

        style = BACKEND_STYLE.get(backend, {})
        label = style.get("label", backend)
        print(f"\n  {label.upper()}")
        print("  " + "-" * 88)
        print(
            f"  {'T':>8} | {'Time':>10} | {'Std':>8} | {'Mem Alloc':>10} | {'Mem Rsrv':>10} | {'Throughput':>14} | Status"
        )
        print("  " + "-" * 88)

        backend_oom = False

        for T in T_values:
            # Skip if K > T
            if K > T:
                r = _skip_result(T, K, C, B, backend, semiring, phase, f"K={K} > T={T}")
                results.append(r)
                print(
                    f"  {T:>8} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | skip (K > T)"
                )
                continue

            # Skip after OOM
            if backend_oom:
                r = _skip_result(T, K, C, B, backend, semiring, phase, "Previous T OOM'd")
                results.append(r)
                print(
                    f"  {T:>8} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | skip (prev OOM)"
                )
                continue

            # Check predicted OOM
            skip, reason = should_skip_config(
                T,
                K,
                C,
                backend,
                {backend: oom_history[backend]},
                max_memory_gb,
                B=B,
            )
            if skip:
                r = _skip_result(T, K, C, B, backend, semiring, phase, reason)
                results.append(r)
                print(
                    f"  {T:>8} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | skip ({reason})"
                )
                continue

            gc.collect()
            torch.cuda.empty_cache()

            r = run_single_benchmark(
                T,
                K,
                C,
                B,
                backend,
                device,
                repeats,
                semiring_name=semiring,
                phase=phase,
            )
            results.append(r)

            if r.status == "success":
                tp = (B * T) / (r.time_ms_median / 1000) if r.time_ms_median > 0 else 0
                tp_str = _format_throughput(tp)
                iqr = r.time_ms_iqr_high - r.time_ms_iqr_low
                std_str = f"{iqr:.1f}ms" if iqr < 1000 else f"{iqr/1000:.2f}s"
                print(
                    f"  {T:>8} | {r.time_ms_median:>8.1f}ms | {std_str:>8} | {r.peak_allocated_gb:>8.3f}GB | "
                    f"{r.peak_reserved_gb:>8.3f}GB | {tp_str:>14} | ok"
                )
            elif r.status == "oom":
                backend_oom = True
                oom_history[backend].append((T, K, C))
                print(f"  {T:>8} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | OOM")
            else:
                print(
                    f"  {T:>8} | {'':>10} | {'':>8} | {'':>10} | {'':>10} | {'':>14} | {r.status}"
                )

    return results


def _skip_result(T, K, C, B, backend, semiring, phase, reason) -> BenchmarkResult:
    """Create a skip/not_tested BenchmarkResult."""
    return BenchmarkResult(
        T=T,
        K=K,
        C=C,
        B=B,
        KC=K * C,
        backend=backend,
        semiring=semiring,
        phase=phase,
        time_ms_median=float("nan"),
        time_ms_iqr_low=float("nan"),
        time_ms_iqr_high=float("nan"),
        time_per_position_ms=float("nan"),
        peak_allocated_gb=float("nan"),
        peak_reserved_gb=float("nan"),
        status="not_tested",
        error_msg=reason,
    )


# ======================================================================
# Summary
# ======================================================================


def print_sweep_summary(
    results: list[BenchmarkResult], backends: list[str], K: int, C: int, B: int
):
    """Print comparison summary."""

    ok = {b: [r for r in results if r.backend == b and r.status == "success"] for b in backends}

    # Memory scaling per backend
    print(f"\n  MEMORY SCALING (K={K}, C={C}, B={B})")
    print("  " + "-" * 60)
    for b in backends:
        pts = ok.get(b, [])
        if len(pts) < 2:
            continue
        label = BACKEND_STYLE.get(b, {}).get("label", b)
        min_r = min(pts, key=lambda r: r.T)
        max_r = max(pts, key=lambda r: r.T)
        T_ratio = max_r.T / min_r.T
        mem_ratio = max_r.peak_allocated_gb / max(min_r.peak_allocated_gb, 1e-6)
        print(
            f"  {label:30s}: T grew {T_ratio:>6.0f}x, memory grew {mem_ratio:>5.1f}x "
            f"({min_r.peak_allocated_gb:.2f} -> {max_r.peak_allocated_gb:.2f} GB)"
        )

    # OOM boundaries
    oom = {b: [r for r in results if r.backend == b and r.status == "oom"] for b in backends}
    oom_backends = {b: v for b, v in oom.items() if v}
    if oom_backends:
        print("\n  OOM BOUNDARIES")
        print("  " + "-" * 60)
        for b, oom_pts in oom_backends.items():
            label = BACKEND_STYLE.get(b, {}).get("label", b)
            first_oom_T = min(r.T for r in oom_pts)
            print(f"  {label:30s}: OOM at T={first_oom_T:,}")

    # Speedup table (relative to slowest backend at each T)
    active_ok = {b: pts for b, pts in ok.items() if pts}
    if len(active_ok) >= 2:
        T_sets = [{r.T for r in pts} for pts in active_ok.values()]
        shared_Ts = sorted(set.intersection(*T_sets))
    else:
        shared_Ts = []

    if shared_Ts:
        active_names = list(active_ok.keys())
        print("\n  SPEEDUP TABLE (relative to slowest at each T)")
        col_w = 14
        print("  " + "-" * (12 + col_w * len(active_names)))
        header = f"  {'T':>8} |"
        for b in active_names:
            lbl = BACKEND_STYLE.get(b, {}).get("label", b)[: col_w - 2]
            header += f" {lbl:>{col_w - 2}} |"
        print(header)
        print("  " + "-" * (12 + col_w * len(active_names)))

        by_b_T = {b: {r.T: r for r in pts} for b, pts in active_ok.items()}
        for T in shared_Ts:
            times = {b: by_b_T[b][T].time_ms_median for b in active_names if T in by_b_T[b]}
            if len(times) < 2:
                continue
            slowest = max(times.values())
            row = f"  {T:>8} |"
            for b in active_names:
                if b in times:
                    speedup = slowest / times[b]
                    row += f" {speedup:>{col_w - 2}.1f}x |"
                else:
                    row += f" {'---':>{col_w - 2}} |"
            print(row)


# ======================================================================
# Plotting
# ======================================================================


def generate_figures(
    results: list[BenchmarkResult], outdir: Path, backends: list[str], K: int, C: int, B: int
):
    """Generate publication figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not available -- skipping figures")
        return

    ok = {
        b: sorted(
            [r for r in results if r.backend == b and r.status == "success"], key=lambda r: r.T
        )
        for b in backends
    }
    oom = {b: [r for r in results if r.backend == b and r.status == "oom"] for b in backends}

    active = [b for b in backends if ok.get(b)]
    if not active:
        print("  No successful results -- skipping figures")
        return

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    def fmt_T(x, _):
        if x >= 1000:
            return f"{int(x/1000)}K"
        return f"{int(x)}"

    # -- Fig 1: Wall time vs T -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Ts = [r.T for r in ok[b]]
        times = [r.time_ms_median for r in ok[b]]
        lo = [r.time_ms_iqr_low for r in ok[b]]
        hi = [r.time_ms_iqr_high for r in ok[b]]
        ax.plot(
            Ts,
            times,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        ax.fill_between(Ts, lo, hi, color=s["color"], alpha=0.12)

    # Mark OOM boundaries
    for b in active:
        if oom.get(b):
            s = BACKEND_STYLE.get(b, {"color": "gray"})
            oom_T = min(r.T for r in oom[b])
            ax.axvline(oom_T, color=s["color"], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Wall Time vs Sequence Length  (K={K}, C={C}, B={B})")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_T))

    fig.tight_layout()
    fig.savefig(outdir / "fig_time_vs_T.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_time_vs_T.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_time_vs_T.pdf/png")

    # -- Fig 2: Memory vs T (MB units for visibility) --------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    mem_floor_mb = 0.1  # floor for log scale (sub-MB values clamp here)

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Ts = [r.T for r in ok[b]]
        mems_mb = [max(r.peak_allocated_gb * 1024, mem_floor_mb) for r in ok[b]]
        ax.plot(
            Ts,
            mems_mb,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        # Annotate backends with very low memory (< 10 MB at max T)
        if mems_mb and mems_mb[-1] < 10:
            raw_mb = ok[b][-1].peak_allocated_gb * 1024
            label_text = f"{raw_mb:.1f} MB" if raw_mb >= 1 else f"{raw_mb * 1024:.0f} KB"
            ax.annotate(
                label_text,
                (Ts[-1], mems_mb[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=8,
                color=s["color"],
                fontweight="bold",
            )

    # Theoretical O(TKC^2) exact backend reference
    all_Ts = sorted({r.T for r in results if r.status == "success"})
    if all_Ts:
        exact_mb = [B * T * K * C * C * 4 / (1024**2) for T in all_Ts]
        ax.plot(
            all_Ts,
            exact_mb,
            "--",
            color=THEORY_COLOR,
            label=r"O(TKC$^2$) edge tensor",
            linewidth=1.5,
        )

    # Mark OOM
    for b in active:
        if oom.get(b):
            s = BACKEND_STYLE.get(b, {"color": "gray"})
            oom_T = min(r.T for r in oom[b])
            ax.axvline(oom_T, color=s["color"], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Peak allocated GPU memory (MB)")
    ax.set_title(f"Memory Scaling  (K={K}, C={C}, B={B})")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_T))

    fig.tight_layout()
    fig.savefig(outdir / "fig_memory_vs_T.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_memory_vs_T.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_memory_vs_T.pdf/png")

    # -- Fig 3: Speedup vs T (relative to slowest) --------------------
    if len(active) >= 2:
        T_sets = [{r.T for r in ok[b]} for b in active]
        shared_Ts = sorted(set.intersection(*T_sets))

        if len(shared_Ts) >= 2:
            by_b_T = {b: {r.T: r.time_ms_median for r in ok[b]} for b in active}

            # Reference = slowest backend at largest shared T
            max_shared_T = max(shared_Ts)
            ref_backend = max(active, key=lambda b: by_b_T[b].get(max_shared_T, 0))

            fig, ax = plt.subplots(figsize=(8, 5))

            for b in active:
                if b == ref_backend:
                    continue
                s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
                Ts_b = [T for T in shared_Ts if T in by_b_T[b] and T in by_b_T[ref_backend]]
                speedups = [by_b_T[ref_backend][T] / by_b_T[b][T] for T in Ts_b]
                ref_label = BACKEND_STYLE.get(ref_backend, {}).get("label", ref_backend)
                ax.plot(
                    Ts_b,
                    speedups,
                    marker=s["marker"],
                    linestyle=s["ls"],
                    color=s["color"],
                    label=f'{s["label"]} vs {ref_label}',
                    markersize=6,
                    linewidth=1.5,
                )

                # Annotate last point
                if Ts_b and speedups:
                    ax.annotate(
                        f"{speedups[-1]:.1f}x",
                        (Ts_b[-1], speedups[-1]),
                        textcoords="offset points",
                        xytext=(8, 0),
                        fontsize=9,
                        color=s["color"],
                    )

            ax.axhline(1.0, color=THEORY_COLOR, linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("Sequence length T")
            ax.set_ylabel("Speedup (x)")
            ref_label = BACKEND_STYLE.get(ref_backend, {}).get("label", ref_backend)
            ax.set_title(f"Speedup vs {ref_label}  (K={K}, C={C}, B={B})")
            ax.legend(loc="upper left")
            ax.grid(True, which="both", alpha=0.25)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_T))

            fig.tight_layout()
            fig.savefig(outdir / "fig_speedup_vs_T.pdf", bbox_inches="tight")
            fig.savefig(outdir / "fig_speedup_vs_T.png", bbox_inches="tight")
            plt.close(fig)
            print("  Saved fig_speedup_vs_T.pdf/png")

    # -- Fig 4: Throughput vs T ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for b in active:
        s = BACKEND_STYLE.get(b, {"color": "gray", "marker": "o", "ls": "-", "label": b})
        Ts = [r.T for r in ok[b]]
        # positions per second = T / (time_ms / 1000) = T * 1000 / time_ms
        throughput = [r.T * 1000 / r.time_ms_median for r in ok[b]]
        # IQR on throughput: faster time -> higher throughput (bounds invert)
        tp_lo = [r.T * 1000 / r.time_ms_iqr_high for r in ok[b]]
        tp_hi = [r.T * 1000 / r.time_ms_iqr_low for r in ok[b]]
        ax.plot(
            Ts,
            throughput,
            marker=s["marker"],
            linestyle=s["ls"],
            color=s["color"],
            label=s["label"],
            markersize=5,
            linewidth=1.5,
        )
        ax.fill_between(Ts, tp_lo, tp_hi, color=s["color"], alpha=0.12)
        # Annotate last point with throughput value
        if throughput:
            val = throughput[-1]
            if val >= 1e6:
                label_text = f"{val / 1e6:.1f}M"
            elif val >= 1e3:
                label_text = f"{val / 1e3:.1f}K"
            else:
                label_text = f"{val:.0f}"
            ax.annotate(
                f"{label_text} pos/s",
                (Ts[-1], val),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=8,
                color=s["color"],
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Throughput (positions / second)")
    ax.set_title(f"Throughput vs Sequence Length  (K={K}, C={C}, B={B})")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_T))

    fig.tight_layout()
    fig.savefig(outdir / "fig_throughput_vs_T.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig_throughput_vs_T.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_throughput_vs_T.pdf/png")


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="T-sweep: wall time and memory vs sequence length across backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--K", type=int, default=50, help="Max duration (fixed). Default: 50")
    parser.add_argument("--C", type=int, default=24, help="Num classes (fixed). Default: 24")
    parser.add_argument("--B", type=int, default=2, help="Batch size. Default: 2")
    parser.add_argument("--repeats", type=int, default=5, help="Timed iterations. Default: 5")
    parser.add_argument(
        "--T",
        type=str,
        default=None,
        help="Comma-separated T values. Default: log-spaced 100-100K",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help=f"Comma-separated backends. Available: {', '.join(ALL_BACKENDS)}. "
        f"Default: {','.join(DEFAULT_BACKENDS)}",
    )
    parser.add_argument(
        "--all-backends",
        action="store_true",
        help=f"Run all backends: {', '.join(ALL_BACKENDS)}",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["forward", "backward", "both"],
        help="Which phase to time. Default: both",
    )
    parser.add_argument(
        "--semiring",
        type=str,
        default="Log",
        help="Semiring to use. Default: Log",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=40.0,
        help="Skip configs predicted to exceed this memory. Default: 40",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=list(REGIMES.keys()),
        help="Benchmark regime preset. 'ner': CoNLL-2003 BIO scale, all backends "
        "(K=16,C=9,T=64-4K). 'speech': TIMIT phone segmentation, streaming only "
        "(K=30,C=39,T=500-20K). 'streaming': genomics-scale, streaming only "
        "(K=200,C=6,T=10K-1M). Explicit --K/--C/--T/--backends flags override.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer T values, fewer repeats"
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--outdir", type=Path, default=Path("results/t_sweep"), help="Output directory"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1

    device = torch.device("cuda")

    # Apply regime preset (explicit CLI flags override)
    regime_T = None
    regime_backends = None
    if args.regime:
        regime = REGIMES[args.regime]
        if "--K" not in sys.argv:
            args.K = regime["K"]
        if "--C" not in sys.argv:
            args.C = regime["C"]
        if "--B" not in sys.argv:
            args.B = regime["B"]
        regime_T = regime["T"]
        regime_backends = regime["backends"]
        # Put regime results in a subdirectory
        args.outdir = args.outdir / args.regime

    # Parse T values (priority: --T > --quick > regime > default)
    if args.T:
        T_values = parse_int_list(args.T)
    elif args.quick:
        T_values = QUICK_T_VALUES
        args.repeats = 3
    elif regime_T:
        T_values = regime_T
    else:
        T_values = DEFAULT_T_VALUES

    # Parse backends (priority: --all-backends > --backends > regime > default)
    if args.all_backends:
        backends = ALL_BACKENDS
    elif "--backends" in sys.argv:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    elif regime_backends:
        backends = regime_backends
    else:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    K, C, B = args.K, args.C, args.B
    args.outdir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 72)
    print("  T-SWEEP BENCHMARK: Speed & Memory vs Sequence Length")
    print("=" * 72)
    if args.regime:
        print(f"  Regime: {args.regime} \u2014 {REGIMES[args.regime]['description']}")
    print(f"  GPU: {gpu_name} ({total_gb:.0f} GB)")
    print(f"  Fixed: K={K}, C={C}, B={B}")
    print(f"  T values: {T_values}")
    print(f"  Backends: {backends}")
    print(f"  Phase: {args.phase}, Semiring: {args.semiring}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Triton available: {HAS_TRITON}")
    print("=" * 72)

    # -- Run sweep -----------------------------------------------------
    results = run_sweep(
        T_values=T_values,
        K=K,
        C=C,
        B=B,
        backends=backends,
        device=device,
        repeats=args.repeats,
        phase=args.phase,
        semiring=args.semiring,
        max_memory_gb=args.max_memory_gb,
    )

    # -- Summary -------------------------------------------------------
    print_sweep_summary(results, backends, K, C, B)

    # -- Save JSON -----------------------------------------------------
    json_path = args.outdir / "t_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "regime": args.regime,
                    "K": K,
                    "C": C,
                    "B": B,
                    "phase": args.phase,
                    "semiring": args.semiring,
                    "gpu": gpu_name,
                    "repeats": args.repeats,
                },
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Results saved to {json_path}")

    # -- Figures -------------------------------------------------------
    if not args.no_plot:
        print("\n  Generating figures...")
        generate_figures(results, args.outdir, backends, K, C, B)

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
