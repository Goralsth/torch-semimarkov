#!/usr/bin/env python3
"""
Memory analysis benchmark for Semi-Markov CRF backends.

Generates data for:
1. OOM feasibility heatmaps (with consistent GB units)
2. Time vs state-space size plots (with median/IQR)
3. Memory breakdown by allocation category

Supports:
- Multiple backends including Triton-accelerated scan
- Forward-only, backward-only, or combined timing
- Different semirings (Log, Max, Entropy, etc.)

Example:
    python benchmarks/benchmark_memory_analysis.py \
        --device cuda:0 \
        --T 128,256,512,1024 \
        --K 4,8,12,16,20,24 \
        --C 3,6,9,12 \
        --B 4 \
        --repeats 5 \
        --phases both \
        --semirings Log,Max \
        --output-dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lib import (
    BenchmarkResult,
    print_summary,
    run_single_benchmark,
    sample_configurations,
    save_results,
    should_skip_config,
)
from lib.runner import (
    LOG_MAX_BACKENDS,
    LOG_SEMIRING_ONLY_BACKENDS,
    SEMIRING_MAP,
)
from lib.sampling import parse_int_list

try:
    from flash_semicrf.streaming import HAS_TRITON
except ImportError:
    HAS_TRITON = False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--T", type=str, default="128,256,512,1024")
    parser.add_argument("--K", type=str, default="4,8,12,16,20,24")
    parser.add_argument("--C", type=str, default="3,6,9,12")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--sample-configs",
        type=int,
        default=0,
        help=(
            "Sample this many (T, K*C) configs from the full grid, anchored at "
            "min/max BTKC. Use 0 to run the full grid. Ignored if --compile-friendly is set."
        ),
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="linear_scan_vectorized,linear_scan_streaming,binary_tree_sharded,triton_streaming",
        help=(
            "Comma-separated list of backends. Options: "
            "binary_tree, binary_tree_sharded, banded, block_triangular, "
            "linear_scan, linear_scan_vectorized, linear_scan_streaming, "
            "triton_streaming. "
            "Note: binary_tree/block_triangular have O((KC)^2) memory. "
            "binary_tree_sharded uses CheckpointShardSemiring for reduced peak memory. "
            "banded is prototype (Log semiring only). "
            "triton_streaming supports Log/Max only."
        ),
    )
    parser.add_argument(
        "--semirings",
        type=str,
        default="Log",
        help=(
            "Comma-separated list of semirings. Options: Log, Max, Entropy. "
            "Note: triton_streaming and banded have semiring restrictions."
        ),
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="both",
        help=(
            "Comma-separated list of phases to time. Options: "
            "forward (forward pass only), backward (backward pass only), "
            "both (forward + backward together). Default: both"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=40.0,
        help="Skip configs predicted to exceed this memory",
    )
    parser.add_argument(
        "--skip-adjacent-oom",
        action="store_true",
        default=True,
        help="Skip configs if smaller adjacent config OOM'd",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    T_list = parse_int_list(args.T)
    K_list = parse_int_list(args.K)
    C_list = parse_int_list(args.C)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    semirings = [s.strip() for s in args.semirings.split(",") if s.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Validate semirings
    for s in semirings:
        if s not in SEMIRING_MAP:
            print(f"WARNING: Unknown semiring '{s}', available: {list(SEMIRING_MAP.keys())}")

    # Validate phases
    valid_phases = {"forward", "backward", "both"}
    for p in phases:
        if p not in valid_phases:
            print(f"WARNING: Unknown phase '{p}', available: {valid_phases}")

    # Check Triton availability if needed
    if "triton_streaming" in backends:
        if not HAS_TRITON:
            print("WARNING: Triton not available, triton_streaming will use PyTorch fallback")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    results: list[BenchmarkResult] = []
    # OOM history is tracked per (backend, semiring, phase) combination
    oom_history: dict[str, list[tuple[int, int, int]]] = {}
    for b in backends:
        for s in semirings:
            for p in phases:
                oom_history[f"{b}_{s}_{p}"] = []

    full_config_count = len(T_list) * len(K_list) * len(C_list)

    # Choose sampling strategy
    if args.sample_configs > 0:
        configs = sample_configurations(T_list, K_list, C_list, args.B, args.sample_configs)
    else:
        # Full grid
        configs = [(T, K, C) for T in T_list for K in K_list for C in C_list]

    total_configs = len(configs) * len(backends) * len(semirings) * len(phases)
    completed = 0

    print(f"Running {total_configs} configurations...")
    print(f"Device: {device}")
    print(f"T: {T_list}, K: {K_list}, C: {C_list}, B: {args.B}")
    print(f"Backends: {backends}")
    print(f"Semirings: {semirings}")
    print(f"Phases: {phases}")
    print(f"Repeats: {args.repeats}")

    if args.sample_configs > 0 and len(configs) != full_config_count:
        print(f"Sampling {len(configs)} of {full_config_count} T/K/C configs by T and K*C range")

    print("-" * 80)

    for backend in backends:
        for semiring_name in semirings:
            for phase in phases:
                for T, K, C in configs:
                    KC = K * C
                    completed += 1
                    oom_key = f"{backend}_{semiring_name}_{phase}"

                    # Check backend/semiring compatibility
                    if backend in LOG_SEMIRING_ONLY_BACKENDS and semiring_name != "Log":
                        print(
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                            f"{backend}/{semiring_name}/{phase}: "
                            f"{backend} only supports Log semiring"
                        )
                        results.append(
                            BenchmarkResult(
                                T=T,
                                K=K,
                                C=C,
                                B=args.B,
                                KC=KC,
                                backend=backend,
                                semiring=semiring_name,
                                phase=phase,
                                time_ms_median=float("nan"),
                                time_ms_iqr_low=float("nan"),
                                time_ms_iqr_high=float("nan"),
                                time_per_position_ms=float("nan"),
                                peak_allocated_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                status="not_supported",
                                error_msg=f"{backend} only supports Log semiring",
                            )
                        )
                        continue

                    # Check streaming backend semiring compatibility (Log, Max only)
                    if backend in LOG_MAX_BACKENDS and semiring_name not in {"Log", "Max"}:
                        print(
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                            f"{backend}/{semiring_name}/{phase}: "
                            f"{backend} only supports Log/Max semirings"
                        )
                        results.append(
                            BenchmarkResult(
                                T=T,
                                K=K,
                                C=C,
                                B=args.B,
                                KC=KC,
                                backend=backend,
                                semiring=semiring_name,
                                phase=phase,
                                time_ms_median=float("nan"),
                                time_ms_iqr_low=float("nan"),
                                time_ms_iqr_high=float("nan"),
                                time_per_position_ms=float("nan"),
                                peak_allocated_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                status="not_supported",
                                error_msg=f"{backend} only supports Log/Max semirings",
                            )
                        )
                        continue

                    # Skip configs where K > T (max duration exceeds sequence length)
                    # These waste memory on unreachable states
                    if K > T:
                        print(
                            f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                            f"{backend}/{semiring_name}/{phase}: K > T (unreachable durations)"
                        )
                        results.append(
                            BenchmarkResult(
                                T=T,
                                K=K,
                                C=C,
                                B=args.B,
                                KC=KC,
                                backend=backend,
                                semiring=semiring_name,
                                phase=phase,
                                time_ms_median=float("nan"),
                                time_ms_iqr_low=float("nan"),
                                time_ms_iqr_high=float("nan"),
                                time_per_position_ms=float("nan"),
                                peak_allocated_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                status="skipped",
                                error_msg=f"K={K} > T={T}: unreachable durations",
                            )
                        )
                        continue

                    # Check if we should skip based on OOM history
                    if args.skip_adjacent_oom:
                        # Use backend-only key for memory estimation
                        skip, reason = should_skip_config(
                            T,
                            K,
                            C,
                            backend,
                            {backend: oom_history.get(oom_key, [])},
                            args.max_memory_gb,
                            B=args.B,
                        )
                        if skip:
                            print(
                                f"[{completed}/{total_configs}] SKIP T={T}, K={K}, C={C}, "
                                f"{backend}/{semiring_name}/{phase}: {reason}"
                            )
                            results.append(
                                BenchmarkResult(
                                    T=T,
                                    K=K,
                                    C=C,
                                    B=args.B,
                                    KC=KC,
                                    backend=backend,
                                    semiring=semiring_name,
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
                            )
                            continue

                    print(
                        f"[{completed}/{total_configs}] T={T}, K={K}, C={C}, KC={KC}, "
                        f"{backend}/{semiring_name}/{phase}...",
                        end=" ",
                        flush=True,
                    )

                    result = run_single_benchmark(
                        T,
                        K,
                        C,
                        args.B,
                        backend,
                        device,
                        args.repeats,
                        semiring_name=semiring_name,
                        phase=phase,
                    )
                    results.append(result)

                    if result.status == "success":
                        print(
                            f"OK: {result.time_ms_median:.1f}ms, "
                            f"{result.peak_allocated_gb:.3f}GB allocated, "
                            f"{result.peak_reserved_gb:.3f}GB reserved"
                        )
                    elif result.status == "oom":
                        print("OOM")
                        oom_history[oom_key].append((T, K, C))
                    else:
                        print(f"{result.status}: {result.error_msg}")

    # Save results and print summary
    save_results(results, args.output_dir, backends, semirings, phases)
    print_summary(results, backends, semirings, phases)


if __name__ == "__main__":
    main()
