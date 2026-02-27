#!/usr/bin/env python3
"""Ablation study: effect of emission zero-centering on numerical stability and modeling.

Zero-centering (scores - scores.mean(dim=1, keepdim=True)) before cumsum has two effects:

1. **Numerical**: Cumsum magnitude grows O(√T) instead of O(T), preventing float32
   catastrophic cancellation at genome scale.

2. **Modeling**: The subtracted per-label mean ν_{b,c} creates an implicit duration prior
   -ν_{b,c} * k that penalizes long segments of globally-prevalent labels. For a sequence
   that's 85% intron with μ_intron ≈ 2.0, every intron segment of length k pays an extra
   cost of 2.0*k, while rare labels get a boost.

This script quantifies both effects by comparing three centering strategies:

- **mean** (current default): Per-label temporal mean. Numerical stability + duration prior.
- **position**: Per-position max across labels. Path-invariant — no modeling effect.
- **none**: Raw emissions. No centering at all.

Usage:
    python ablate_zero_centering.py                      # §1 + §3 (default)
    python ablate_zero_centering.py --section stability   # §1: Numerical stability only
    python ablate_zero_centering.py --section precision    # §2: Float32 precision loss (opt-in)
    python ablate_zero_centering.py --section prior        # §3: Duration prior effect only
    python ablate_zero_centering.py --scale quick          # Smaller T values

Note: §2 (precision) is excluded from --section all because the Triton kernel uses
float64 internally, so the test only measures cumsum quantization — not DP precision.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import torch

# We deliberately pass non-centered cum_scores to test precision degradation.
# Suppress the "non-zero-centered input" warning from the streaming autograd.
warnings.filterwarnings("ignore", message="cum_scores endpoint magnitude")

from flash_semicrf import SemiMarkovCRFHead  # noqa: E402
from flash_semicrf.streaming import semi_crf_streaming_forward  # noqa: E402

# ======================================================================
# Utilities
# ======================================================================


def section_header(title):
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def result_line(label, passed, detail=""):
    status = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
    if detail:
        print(f"  {status} {label}: {detail}")
    else:
        print(f"  {status} {label}")
    return passed


CENTERING_MODES = ["mean", "position", "none"]
IMBALANCED_PROPORTIONS = [0.70, 0.15, 0.10, 0.05]


def build_cum_scores(scores, mode, dtype=torch.float64):
    """Build cumulative scores under a given centering mode.

    Args:
        scores: Raw emission scores (batch, T, C).
        mode: One of "mean", "position", "none".
        dtype: Output dtype.

    Returns:
        cum_scores: (batch, T+1, C) cumulative scores.
        shift_info: Dict with centering diagnostics (per-label means, etc.)
    """
    batch, T, C = scores.shape
    scores_float = scores.to(dtype)

    shift_info = {}

    if T > 1:
        if mode == "mean":
            shift = scores_float.mean(dim=1, keepdim=True)  # (B, 1, C)
            scores_float = scores_float - shift
            shift_info["per_label_mean"] = shift.squeeze(1)  # (B, C)
        elif mode == "position":
            shift = scores_float.max(dim=-1, keepdim=True).values  # (B, T, 1)
            scores_float = scores_float - shift
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown centering mode: {mode}")

    cum_scores = torch.zeros(batch, T + 1, C, dtype=dtype, device=scores.device)
    cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

    return cum_scores, shift_info


def make_balanced_emissions(batch, T, C, scale=0.1, seed=42, device="cpu"):
    """Create balanced random emissions (no label imbalance)."""
    torch.manual_seed(seed)
    return torch.randn(batch, T, C, device=device) * scale


def make_imbalanced_emissions(
    batch, T, C, proportions, signal=2.0, noise=0.3, seed=42, device="cpu"
):
    """Create label-imbalanced emissions mimicking genomic segmentation.

    Each position is assigned a true label from the imbalanced distribution.
    The encoder output has high scores for the true label and low scores elsewhere.

    Args:
        proportions: List of C floats summing to 1.0 (label frequencies).
        signal: Mean emission for the true label at each position.
        noise: Gaussian noise added to all scores.
    """
    torch.manual_seed(seed)
    assert len(proportions) == C
    assert abs(sum(proportions) - 1.0) < 1e-6

    scores = torch.randn(batch, T, C, device=device) * noise

    # Assign true labels according to proportions
    boundaries = torch.tensor([0.0] + list(torch.cumsum(torch.tensor(proportions), 0)))
    uniform = torch.rand(batch, T, device=device)

    for c in range(C):
        mask = (uniform >= boundaries[c]) & (uniform < boundaries[c + 1])  # (batch, T)
        # True label gets +signal, others get -signal/(C-1)
        scores[:, :, c] += mask.float() * (signal + signal / (C - 1))
        scores[:, :, c] -= (~mask).float() * (signal / (C - 1))

    return scores


# ======================================================================
# §1 Numerical Stability (cumsum magnitude)
# ======================================================================


def _get_proportions(C):
    """Get label proportions for C classes."""
    proportions = IMBALANCED_PROPORTIONS[:C] if C >= 4 else [1.0 / C] * C
    if C > 4:
        extra = C - 4
        proportions = [p * 0.9 for p in IMBALANCED_PROPORTIONS] + [0.1 / extra] * extra
        total = sum(proportions)
        proportions = [p / total for p in proportions]
    return proportions


def test_numerical_stability(T_values, C, K, device):
    section_header("§1  NUMERICAL STABILITY (Cumsum Magnitude)")

    batch = 2
    proportions = _get_proportions(C)

    print(f"  Config: batch={batch}, C={C}, K={K}")
    print("  Using imbalanced emissions (trained-encoder-like, nonzero per-label mean)")
    print()

    # Header
    print(
        f"  {'T':>7}  {'Mode':>10}  {'Endpoint Mag':>14}  "
        f"{'Peak Mag':>12}  {'Peak/√T':>10}  {'Peak/T':>10}"
    )
    print(f"  {'-'*7}  {'-'*10}  {'-'*14}  " f"{'-'*12}  {'-'*10}  {'-'*10}")

    import math

    rows = []
    for T in T_values:
        scores = make_imbalanced_emissions(
            batch, T, C, proportions, signal=2.0, noise=0.3, device=device
        )

        for mode in CENTERING_MODES:
            cum_scores, _ = build_cum_scores(scores, mode)
            endpoint_mag = cum_scores[:, -1, :].abs().max().item()
            peak_mag = cum_scores.abs().max().item()
            peak_over_sqrt_T = peak_mag / math.sqrt(T)
            peak_over_T = peak_mag / T

            rows.append(
                {
                    "T": T,
                    "mode": mode,
                    "endpoint_mag": endpoint_mag,
                    "peak_mag": peak_mag,
                    "peak_sqrt_t": peak_over_sqrt_T,
                    "peak_over_t": peak_over_T,
                }
            )

            print(
                f"  {T:>7}  {mode:>10}  {endpoint_mag:>14.2f}  "
                f"{peak_mag:>12.2f}  {peak_over_sqrt_T:>10.2f}  {peak_over_T:>10.4f}"
            )

        if T != T_values[-1]:
            print()

    print()
    print("  Expected scaling:")
    print("    mean:     Peak/√T ≈ constant (O(√T) random walk after centering)")
    print("    position: Peak/T  ≈ constant (O(T) drift, reduced coefficient)")
    print("    none:     Peak/T  ≈ constant (O(T) drift from nonzero label means)")

    return {"config": {"batch": batch, "C": C, "K": K}, "rows": rows}


# ======================================================================
# §2 Float32 Precision Loss
# ======================================================================


def test_precision_loss(T_values, C, K, device, n_seeds=5):
    section_header("§2  FLOAT32 PRECISION LOSS")

    batch = 2
    use_triton = device == "cuda"
    proportions = IMBALANCED_PROPORTIONS[:C] if C >= 4 else [1.0 / C] * C
    if C > 4:
        extra = C - 4
        proportions = [p * 0.9 for p in IMBALANCED_PROPORTIONS] + [0.1 / extra] * extra
        total = sum(proportions)
        proportions = [p / total for p in proportions]

    print(f"  Config: batch={batch}, C={C}, K={K}, n_seeds={n_seeds}, use_triton={use_triton}")
    print("  Using imbalanced emissions (trained-encoder-like)")
    print()

    torch.manual_seed(0)
    transition = torch.randn(C, C, device=device) * 0.1
    duration_bias = torch.randn(K, C, device=device) * 0.1

    print(
        f"  {'T':>7}  {'Mode':>10}  "
        f"{'Mean Rel Err':>14}  {'Std Rel Err':>14}  {'Max Rel Err':>14}"
    )
    print(f"  {'-'*7}  {'-'*10}  " f"{'-'*14}  {'-'*14}  {'-'*14}")

    for T in T_values:
        for mode in CENTERING_MODES:
            rel_errors = []
            for seed in range(n_seeds):
                scores = make_imbalanced_emissions(
                    batch,
                    T,
                    C,
                    proportions,
                    signal=2.0,
                    noise=0.3,
                    seed=seed,
                    device=device,
                )
                lengths = torch.full((batch,), T, device=device, dtype=torch.long)

                # Float64 (ground truth)
                cum_f64, _ = build_cum_scores(scores, mode, dtype=torch.float64)
                Z_f64 = semi_crf_streaming_forward(
                    cum_f64,
                    transition.double(),
                    duration_bias.double(),
                    lengths,
                    K,
                    use_triton=use_triton,
                )

                # Float32
                cum_f32, _ = build_cum_scores(scores, mode, dtype=torch.float32)
                Z_f32 = semi_crf_streaming_forward(
                    cum_f32,
                    transition.float(),
                    duration_bias.float(),
                    lengths,
                    K,
                    use_triton=use_triton,
                )

                rel_err = (
                    ((Z_f64.double() - Z_f32.double()).abs() / (Z_f64.double().abs() + 1e-30))
                    .max()
                    .item()
                )
                rel_errors.append(rel_err)

            mean_err = sum(rel_errors) / len(rel_errors)
            std_err = (sum((e - mean_err) ** 2 for e in rel_errors) / len(rel_errors)) ** 0.5
            max_err = max(rel_errors)

            print(
                f"  {T:>7}  {mode:>10}  " f"{mean_err:>14.2e}  {std_err:>14.2e}  {max_err:>14.2e}"
            )

        if T != T_values[-1]:
            print()


# ======================================================================
# §3 Implicit Duration Prior Under Label Imbalance
# ======================================================================


def test_duration_prior(device):
    section_header("§3  IMPLICIT DURATION PRIOR (Label Imbalance)")

    T = 2000
    C = 4
    K = 50
    batch = 1
    label_names = ["intron", "exon", "UTR", "intergenic"]
    proportions_imbalanced = IMBALANCED_PROPORTIONS
    proportions_balanced = [1.0 / C] * C

    use_triton = device == "cuda"

    print(f"  Config: T={T}, C={C}, K={K}, batch={batch}")
    print(f"  Labels: {', '.join(label_names)}")
    print()

    json_scenarios = {}

    for scenario_name, proportions in [
        ("IMBALANCED", proportions_imbalanced),
        ("BALANCED", proportions_balanced),
    ]:
        print(f"  --- Scenario: {scenario_name} ---")
        print(f"  Label proportions: {proportions}")
        print()

        scores = make_imbalanced_emissions(
            batch, T, C, proportions, signal=2.0, noise=0.3, device=device
        )

        # ── Part A: Effective duration prior ──
        # Compute per-label means (the ν_{b,c} values)
        _, shift_info = build_cum_scores(scores, "mean")
        per_label_mean = shift_info["per_label_mean"][0]  # (C,)

        print("  Part A: Per-label emission means and implicit duration penalty")
        print(f"    {'Label':>12}  {'ν_{b,c}':>8}  ", end="")
        k_vals = [1, 5, 10, 25, 50]
        for k in k_vals:
            print(f"{'k='+str(k):>8}", end="  ")
        print()
        print(f"    {'-'*12}  {'-'*8}  ", end="")
        for _ in k_vals:
            print(f"{'-'*8}", end="  ")
        print()

        penalty_rows = []
        for c in range(C):
            nu = per_label_mean[c].item()
            row = {
                "label": label_names[c],
                "nu": nu,
            }
            for k in k_vals:
                row[f"k{k}"] = -nu * k
            penalty_rows.append(row)
            print(f"    {label_names[c]:>12}  {nu:>8.3f}  ", end="")
            for k in k_vals:
                penalty = -nu * k
                print(f"{penalty:>8.2f}", end="  ")
            print()

        print()

        # ── Part B: Marginal comparison ──
        # Compute partition and marginals under each centering mode
        torch.manual_seed(99)  # Same parameters for fair comparison
        transition = torch.randn(C, C, device=device, dtype=torch.float64) * 0.3
        # Duration bias that peaks around k=10-15, encouraging moderate segments
        k_range = torch.arange(K, dtype=torch.float64, device=device)
        mu_k = 12.0  # Peak duration
        duration_bias = -0.05 * (k_range.unsqueeze(1) - mu_k).pow(2).expand(K, C)
        lengths = torch.full((batch,), T, device=device, dtype=torch.long)

        print("  Part B: Partition function and marginal divergence")

        marginals_by_mode = {}
        logZ_by_mode = {}

        for mode in CENTERING_MODES:
            cum_scores, _ = build_cum_scores(scores, mode, dtype=torch.float64)
            cum_scores_ag = cum_scores.detach().requires_grad_(True)

            log_Z = semi_crf_streaming_forward(
                cum_scores_ag,
                transition,
                duration_bias,
                lengths,
                K,
                use_triton=use_triton,
            )
            log_Z.sum().backward()

            # Recover per-position emission marginals from cumulative-score gradients.
            # ∂log Z/∂cum_scores[t,c] = Σ_{segments ending after t} p(seg) - Σ_{ending at t} p(seg),
            # i.e., the difference in total posterior mass crossing position t for label c.
            # A reverse cumsum converts these "net flow" gradients into per-position
            # marginals p(label_t = c): each position is covered by exactly one segment
            # in any valid path, so the reverse cumsum unravels the prefix-sum structure.
            grad_cs = cum_scores_ag.grad[:, 1:, :]  # (batch, T, C)
            emission_marginals = torch.flip(
                torch.cumsum(torch.flip(grad_cs, [1]), dim=1), [1]
            )  # (batch, T, C)

            marginals_by_mode[mode] = emission_marginals.detach()
            logZ_by_mode[mode] = log_Z[0].item()

        # Compute TV distance vs position (path-invariant = canonical distribution)
        ref_marginals = marginals_by_mode["position"]

        print(f"    {'Mode':>10}  {'log Z':>14}  " f"{'TV vs position':>16}  {'Marginal std':>14}")
        print(f"    {'-'*10}  {'-'*14}  " f"{'-'*16}  {'-'*14}")

        partition_rows = []
        for mode in CENTERING_MODES:
            m = marginals_by_mode[mode]
            marginal_std = m.std().item()
            # TV distance: 0.5 * Σ_c |p(t,c) - q(t,c)| averaged over positions
            tv = 0.5 * (m - ref_marginals).abs().sum(dim=2).mean().item()

            partition_rows.append(
                {
                    "mode": mode,
                    "log_z": logZ_by_mode[mode],
                    "tv": tv,
                    "marginal_std": marginal_std,
                }
            )

            print(
                f"    {mode:>10}  {logZ_by_mode[mode]:>14.4f}  "
                f"{tv:>16.6f}  {marginal_std:>14.6f}"
            )

        print()

        # ── Part C: Viterbi segment length comparison ──
        print("  Part C: Viterbi segment lengths by label")

        seg_stats_by_mode = {}

        for mode in CENTERING_MODES:
            cum_scores_mode, _ = build_cum_scores(scores, mode, dtype=torch.float64)

            # Build a CRF head and inject our parameters + cum_scores
            crf = SemiMarkovCRFHead(num_classes=C, max_duration=K, init_scale=0.1).to(device)

            # Override parameters
            with torch.no_grad():
                crf.transition.copy_(transition.float())
                crf.duration_dist.duration_bias.copy_(duration_bias.float())

            # Use decode_with_traceback with streaming backend
            # We need to use the streaming forward directly with our custom cum_scores
            # So we'll do Viterbi via the max semiring streaming forward
            from flash_semicrf.streaming import (
                semi_crf_streaming_viterbi_with_backpointers,
            )

            max_scores, bp_k, bp_c, final_labels = semi_crf_streaming_viterbi_with_backpointers(
                cum_scores_mode,
                transition,
                duration_bias,
                lengths,
                K,
            )

            # Traceback
            segments_by_label = defaultdict(list)
            for b in range(batch):
                t = int(lengths[b].item())
                c = int(final_labels[b].item())
                while t > 0:
                    k = int(bp_k[b, t - 1, c].item())
                    c_prev = int(bp_c[b, t - 1, c].item())
                    if k == 0:
                        break
                    segments_by_label[c].append(k)
                    t = t - k
                    c = c_prev

            seg_stats = {}
            for c in range(C):
                segs = segments_by_label.get(c, [])
                if segs:
                    seg_stats[c] = {
                        "count": len(segs),
                        "mean": sum(segs) / len(segs),
                        "min": min(segs),
                        "max": max(segs),
                    }
                else:
                    seg_stats[c] = {"count": 0, "mean": 0, "min": 0, "max": 0}

            seg_stats_by_mode[mode] = seg_stats

        # Print comparison table
        print(
            f"    {'Label':>12}  {'Mode':>10}  {'Segments':>10}  "
            f"{'Mean Len':>10}  {'Min':>6}  {'Max':>6}"
        )
        print(f"    {'-'*12}  {'-'*10}  {'-'*10}  " f"{'-'*10}  {'-'*6}  {'-'*6}")

        viterbi_rows = []
        for c in range(C):
            for mode in CENTERING_MODES:
                s = seg_stats_by_mode[mode][c]
                viterbi_rows.append(
                    {
                        "label": label_names[c],
                        "mode": mode,
                        "n_segments": s["count"],
                        "mean_len": s["mean"],
                        "min_len": s["min"],
                        "max_len": s["max"],
                    }
                )
                print(
                    f"    {label_names[c]:>12}  {mode:>10}  {s['count']:>10}  "
                    f"{s['mean']:>10.1f}  {s['min']:>6}  {s['max']:>6}"
                )
            if c < C - 1:
                print()

        print()

        json_scenarios[scenario_name.lower()] = {
            "proportions": proportions,
            "penalty": penalty_rows,
            "partition": partition_rows,
            "viterbi": viterbi_rows,
        }

    return {
        "config": {"T": T, "C": C, "K": K, "batch": batch, "label_names": label_names},
        "scenarios": json_scenarios,
    }


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: emission zero-centering effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--section",
        choices=["stability", "precision", "prior", "all"],
        default="all",
        help="Which section to run (default: all = stability + prior; precision is opt-in)",
    )
    parser.add_argument(
        "--scale",
        choices=["quick", "full"],
        default="full",
        help="Scale: quick (smaller T) or full (up to T=50K)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export results to JSON file for downstream processing (e.g., LaTeX tables)",
    )
    args = parser.parse_args()

    C = 24
    K = 50

    if args.scale == "quick":
        T_values = [100, 500, 2_000]
        n_seeds = 5
    else:
        T_values = [100, 500, 2_000, 10_000, 50_000]
        n_seeds = 10

    print("=" * 72)
    print("  EMISSION ZERO-CENTERING ABLATION STUDY")
    print("=" * 72)
    print(f"  Device: {args.device}")
    print(f"  Scale: {args.scale}")
    print(f"  T values: {T_values}")
    print(f"  C={C}, K={K}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    sections = ["stability", "prior"] if args.section == "all" else [args.section]

    json_data = {}
    for section in sections:
        if section == "stability":
            json_data["stability"] = test_numerical_stability(T_values, C, K, args.device)
        elif section == "precision":
            test_precision_loss(T_values, C, K, args.device, n_seeds=n_seeds)
        elif section == "prior":
            json_data["prior"] = test_duration_prior(args.device)

    # Export JSON if requested
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  Results exported to {args.json}")

    section_header("SUMMARY")
    print("  Centering mode comparison:")
    print()
    print("  mean (current default):")
    print("    + O(√T) cumsum growth — stable at genome scale")
    print("    + Adaptive duration prior: penalizes long segments of prevalent labels")
    print("    + Well-suited for genomic segmentation with label imbalance")
    print()
    print("  position (path-invariant):")
    print("    + Moderate cumsum stabilization")
    print("    + No modeling effect — preserves canonical semi-CRF distribution")
    print("    + Appropriate when marginals serve as downstream features")
    print()
    print("  none (raw):")
    print("    - Cumsum grows O(T) — catastrophic at T > 10K in float32")
    print("    - No implicit duration prior")
    print("    - Only safe for short sequences or zero-mean encoders")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
