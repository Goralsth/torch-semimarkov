#!/usr/bin/env python3
"""K-sweep benchmark: wall time and peak memory vs max duration K.

Sweeps K while holding T, C fixed. Runs only Triton streaming backend.

Config:
  - K: [200, 500, 1000, 2000, 8000]
  - T: 50000
  - C: 6
  - B: configurable (32–64 recommended)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from lib import BenchmarkResult, run_single_benchmark, should_skip_config

try:
    from flash_semicrf.streaming import HAS_TRITON
except ImportError:
    HAS_TRITON = False


# ======================================================================
# Config
# ======================================================================

DEFAULT_K_VALUES = [200, 500, 1000, 2000, 8000]
QUICK_K_VALUES = [200, 500, 1000]

BACKEND = "triton_streaming"


# ======================================================================
# Sweep
# ======================================================================


def run_sweep(
    K_values: list[int],
    T: int,
    C: int,
    B: int,
    device: torch.device,
    repeats: int,
    phase: str,
    semiring: str,
    max_memory_gb: float,
) -> list[BenchmarkResult]:

    results: list[BenchmarkResult] = []
    oom_history: list[tuple[int, int, int]] = []
    backend_oom = False

    print(f"\n  TRITON STREAMING")
    print("  " + "-" * 88)
    print(
        f"  {'K':>8} | {'Time':>10} | {'Mem Alloc':>10} | {'Mem Rsrv':>10} | Status"
    )
    print("  " + "-" * 88)

    for K in K_values:

        if backend_oom:
            r = _skip_result(T, K, C, B, semiring, phase, "Previous K OOM'd")
            results.append(r)
            print(f"  {K:>8} | {'':>10} | {'':>10} | {'':>10} | skip (prev OOM)")
            continue

        skip, reason = should_skip_config(
            T,
            K,
            C,
            BACKEND,
            {BACKEND: oom_history},
            max_memory_gb,
            B=B,
        )
        if skip:
            r = _skip_result(T, K, C, B, semiring, phase, reason)
            results.append(r)
            print(f"  {K:>8} | {'':>10} | {'':>10} | {'':>10} | skip ({reason})")
            continue

        gc.collect()
        torch.cuda.empty_cache()

        r = run_single_benchmark(
            T,
            K,
            C,
            B,
            BACKEND,
            device,
            repeats,
            semiring_name=semiring,
            phase=phase,
        )
        results.append(r)

        if r.status == "success":
            print(
                f"  {K:>8} | {r.time_ms_median:>8.1f}ms | "
                f"{r.peak_allocated_gb:>8.3f}GB | {r.peak_reserved_gb:>8.3f}GB | ok"
            )
        elif r.status == "oom":
            backend_oom = True
            oom_history.append((T, K, C))
            print(f"  {K:>8} | {'':>10} | {'':>10} | {'':>10} | OOM")
        else:
            print(f"  {K:>8} | {'':>10} | {'':>10} | {'':>10} | {r.status}")

    return results


def _skip_result(T, K, C, B, semiring, phase, reason) -> BenchmarkResult:
    return BenchmarkResult(
        T=T,
        K=K,
        C=C,
        B=B,
        KC=K * C,
        backend=BACKEND,
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
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="K-sweep benchmark (T fixed)")
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--C", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--phase", type=str, default="both", choices=["forward", "backward", "both"])
    parser.add_argument("--semiring", type=str, default="Log")
    parser.add_argument("--max-memory-gb", type=float, default=40.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("results/k_sweep"))

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return 1

    device = torch.device("cuda")

    T = 50000
    C = args.C
    B = args.B

    if args.quick:
        K_values = QUICK_K_VALUES
        args.repeats = 3
    else:
        K_values = DEFAULT_K_VALUES

    args.outdir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print("=" * 72)
    print("  K-SWEEP BENCHMARK")
    print("=" * 72)
    print(f"  GPU: {gpu_name} ({total_gb:.0f} GB)")
    print(f"  Fixed: T={T}, C={C}, B={B}")
    print(f"  K values: {K_values}")
    print(f"  Backend: {BACKEND}")
    print(f"  Triton available: {HAS_TRITON}")
    print("=" * 72)

    results = run_sweep(
        K_values=K_values,
        T=T,
        C=C,
        B=B,
        device=device,
        repeats=args.repeats,
        phase=args.phase,
        semiring=args.semiring,
        max_memory_gb=args.max_memory_gb,
    )

    json_path = args.outdir / "k_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "T": T,
                    "C": C,
                    "B": B,
                    "gpu": gpu_name,
                },
                "results": [asdict(r) for r in results],
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
