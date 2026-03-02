#!/usr/bin/env python3
"""Ground-truth correctness validation for streaming Semi-CRF Triton kernels.

Validates the Triton implementation using implementation-independent tests
that DO NOT compare against the PyTorch reference (which uses different
reduction order and thus different floating-point rounding).

Three validation strategies:

1. **Self-consistency** (§1): Do boundary marginals sum to the expected value?
   - Sum of boundary marginals over all positions ≈ T (each position covered)
   - All marginals in [0, 1]
   - Marginals at positions beyond sequence length = 0

2. **Finite differences** (§2): Does autograd match numerical gradient?
   - Perturb each parameter by ±ε, compute (f(x+ε) - f(x-ε)) / 2ε
   - Compare to autograd gradient from backward pass
   - This is the gold standard: if it passes, the implementation is correct

3. **Training convergence** (§3): Do all backends converge to the same loss?
   - Train identical models with three backends: streaming PyTorch, streaming
     Triton, pytorch-struct dp_standard
   - All should reach similar final NLL
   - Validates end-to-end correctness against established SOTA implementations

Usage:
    # Run all three benchmarks (default: moderate scale)
    python validate_correctness.py

    # Run individual benchmarks
    python validate_correctness.py --test self-consistency
    python validate_correctness.py --test finite-diff
    python validate_correctness.py --test convergence

    # Scale options
    python validate_correctness.py --scale small    # Quick smoke test
    python validate_correctness.py --scale medium   # Default
    python validate_correctness.py --scale large    # Thorough (slow)
    python validate_correctness.py --genome         # Genome-scale self-consistency (T=50K, K=100, C=24)

    # Finite-diff on specific parameters only
    python validate_correctness.py --test finite-diff --params cum_scores transition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# ──────────────────────────────────────────────────────────────────────
# Imports from flash_semicrf
# ──────────────────────────────────────────────────────────────────────
from flash_semicrf.streaming import semi_crf_streaming_forward
from flash_semicrf.streaming.triton_forward import (
    HAS_TRITON,
    launch_streaming_triton_kernel,
)

if HAS_TRITON:
    from flash_semicrf.streaming.triton_backward import (
        launch_streaming_triton_marginals,
    )


# ======================================================================
# Shared utilities
# ======================================================================


@dataclass
class ScaleConfig:
    """Test scale parameters."""

    name: str
    T: int  # sequence length
    C: int  # classes
    K: int  # max duration
    batch: int
    # finite-diff specific
    fd_T: int  # smaller T for finite diff (expensive)
    fd_C: int
    fd_K: int
    fd_eps: float
    # convergence specific
    conv_T: int
    conv_C: int
    conv_K: int
    conv_epochs: int
    # checkpoint control (None = auto, K = aggressive for stability)
    checkpoint_interval: int = None


SCALES = {
    "small": ScaleConfig(
        name="small",
        T=100,
        C=8,
        K=10,
        batch=2,
        fd_T=10,
        fd_C=3,
        fd_K=4,
        fd_eps=1e-3,
        conv_T=50,
        conv_C=4,
        conv_K=5,
        conv_epochs=50,
    ),
    "medium": ScaleConfig(
        name="medium",
        T=500,
        C=16,
        K=25,
        batch=4,
        fd_T=20,
        fd_C=4,
        fd_K=6,
        fd_eps=1e-3,
        conv_T=100,
        conv_C=8,
        conv_K=10,
        conv_epochs=100,
    ),
    "large": ScaleConfig(
        name="large",
        T=2000,
        C=32,
        K=50,
        batch=4,
        fd_T=100,
        fd_C=16,
        fd_K=25,
        fd_eps=1e-3,
        conv_T=500,
        conv_C=32,
        conv_K=50,
        conv_epochs=200,
    ),
    "genome": ScaleConfig(
        name="genome",
        T=50000,
        C=24,
        K=100,
        batch=2,
        # Finite-diff impractical at this scale (placeholder values)
        fd_T=50,
        fd_C=8,
        fd_K=20,
        fd_eps=1e-3,
        # Convergence also expensive (placeholder values)
        conv_T=500,
        conv_C=16,
        conv_K=50,
        conv_epochs=50,
        # Default checkpoint_interval now caps at max(K, 64), giving 500 ckpts
        # (previously required manual override to checkpoint_interval=K)
    ),
}


def make_inputs(batch, T, C, K, device="cuda", seed=42):
    """Create synthetic inputs with deterministic seeding."""
    set_all_seeds(seed)

    scores = torch.randn(batch, T, C, device=device) * 0.1
    scores = scores - scores.mean(dim=1, keepdim=True)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=torch.float32)
    cum_scores[:, 1:] = torch.cumsum(scores, dim=1)

    transition = torch.randn(C, C, device=device, dtype=torch.float32) * 0.1
    duration_bias = torch.randn(K, C, device=device, dtype=torch.float32) * 0.1
    lengths = torch.full((batch,), T, device=device, dtype=torch.long)

    return cum_scores, transition, duration_bias, lengths


def set_all_seeds(seed: int):
    """Set CPU and CUDA RNG seeds for reproducible results."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def prob_range_stats(values: torch.Tensor, tol: float = 5e-3):
    """Return [0,1]-range diagnostics with tolerance for FP drift."""
    min_val = values.min().item()
    max_val = values.max().item()
    oob_mask = (values < -tol) | (values > (1.0 + tol))
    frac_oob = oob_mask.float().mean().item()
    return min_val, max_val, frac_oob


# ======================================================================
# §1  Self-consistency: marginal sum checks
# ======================================================================


def test_self_consistency(cfg: ScaleConfig, device: str, seed: int = 42) -> tuple[bool, dict]:
    """Validate that boundary marginals satisfy known invariants.

    For a Semi-CRF with boundary marginals p(t) = probability a segment
    starts at position t:
      - Each p(t) ∈ [0, 1] (allow small FP tolerance)
      - Σ_t p(t) should be a reasonable number of segments ∈ [1, T]
      - Positions beyond the sequence length should have marginal = 0

    Additionally, we validate emission marginals via the autograd chain:
      - ∂log Z / ∂emission[t, c] = P(y_t = c), the marginal class probability
      - Recovered from ∂log Z / ∂cum_scores via reverse cumsum (suffix sum)
      - These must sum to 1 over classes for each position and lie in [0, 1]
    """
    section_header("§1  SELF-CONSISTENCY (Marginal Invariants)")

    all_passed = True
    json_configs = []
    configs = [
        (cfg.batch, cfg.T, cfg.C, cfg.K, "main"),
    ]
    # Also test a small config for quick sanity
    if cfg.T > 100:
        configs.append((2, 50, 8, 10, "small"))

    # Use checkpoint_interval from config (None for auto, K for aggressive)
    ckpt_interval = cfg.checkpoint_interval

    for batch, T, C, K, label in configs:
        print(f"\n  Config [{label}]: batch={batch}, T={T}, C={C}, K={K}", end="")
        if ckpt_interval is not None:
            print(f", checkpoint_interval={ckpt_interval}")
        else:
            print()

        cum_scores, transition, duration_bias, lengths = make_inputs(
            batch, T, C, K, device=device, seed=seed
        )

        # --- Triton forward ---
        log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            checkpoint_interval=ckpt_interval,
        )

        # --- Boundary marginals ---
        marginals = launch_streaming_triton_marginals(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            interval,
        )  # (batch, T)

        cfg_data = {"label": label, "batch": batch, "T": T, "C": C, "K": K, "checks": []}

        # Check 1: Boundary marginals should lie in [0, 1] up to FP tolerance.
        # We allow tiny drift but fail if out-of-range mass becomes non-trivial.
        # Also gate hard on gross violations to catch immediate failures.
        min_val, max_val, frac_oob = prob_range_stats(marginals, tol=5e-3)
        boundary_range_ok = frac_oob <= 5e-4 and min_val >= -0.05 and max_val <= 1.05
        all_passed &= result_line(
            "Boundary marginals in [0, 1]",
            boundary_range_ok,
            f"min={min_val:.6f}, max={max_val:.6f}, frac_oob={frac_oob:.4%}",
        )
        cfg_data["checks"].append(
            {
                "name": "boundary_range_check",
                "passed": bool(boundary_range_ok),
                "min": min_val,
                "max": max_val,
                "frac_oob": frac_oob,
            }
        )

        # Check 2: Marginal sum ≈ expected number of segments
        # Should be between 1 (one big segment) and T (all singletons)
        marginal_sums = marginals.sum(dim=1)  # (batch,)
        mean_sum = marginal_sums.mean().item()
        sum_reasonable = (marginal_sums >= 0.5).all().item() and (
            marginal_sums <= T + 0.5
        ).all().item()
        all_passed &= result_line(
            "Marginal sum in [1, T]",
            sum_reasonable,
            f"mean={mean_sum:.2f}, range=[{marginal_sums.min().item():.2f}, {marginal_sums.max().item():.2f}]",
        )
        cfg_data["checks"].append(
            {
                "name": "marginal_sum_in_range",
                "passed": bool(sum_reasonable),
                "mean": mean_sum,
                "range_min": marginal_sums.min().item(),
                "range_max": marginal_sums.max().item(),
            }
        )

        # Check 3: Emission marginals via autograd (reverse cumsum)
        #
        # Math: cum_scores[:, s, c] = Σ_{t=0}^{s-1} emission[t, c]
        # So ∂log_Z/∂emission[t, c] = Σ_{s=t+1}^{T} ∂log_Z/∂cum_scores[s, c]
        # This suffix sum recovers P(y_t = c), the marginal class probability.
        # Invariant: Σ_c P(y_t = c) = 1 for each position t.
        cum_scores_ag = cum_scores.clone().detach().requires_grad_(True)
        log_Z_ag = semi_crf_streaming_forward(
            cum_scores_ag,
            transition.detach(),
            duration_bias.detach(),
            lengths,
            K,
            use_triton=True,
            checkpoint_interval=ckpt_interval,
        )
        log_Z_ag.sum().backward()
        grad_cs = cum_scores_ag.grad  # (batch, T+1, C)

        # Reverse cumsum (suffix sum) to recover emission marginals
        # emission_marginals[b, t, c] = Σ_{s=t+1}^{T} grad_cs[b, s, c]
        grad_cs_slice = grad_cs[:, 1:, :]  # (batch, T, C), indices for s=1..T
        emission_marginals = torch.flip(
            torch.cumsum(torch.flip(grad_cs_slice, [1]), dim=1), [1]
        )  # (batch, T, C)

        # Each position's class marginals should sum to ~1
        class_sums = emission_marginals.sum(dim=2)  # (batch, T)
        max_deviation = (class_sums - 1.0).abs().max().item()
        all_passed &= result_line(
            "Emission marginals Σ_c P(c|t) = 1",
            max_deviation < 0.02,
            f"max deviation = {max_deviation:.6f}",
        )
        cfg_data["checks"].append(
            {
                "name": "emission_marginals_sum_to_1",
                "passed": max_deviation < 0.02,
                "max_deviation": max_deviation,
            }
        )

        # Total across all positions and classes should be T
        # Threshold scales with T: accumulated FP error is ~0.00004 per position
        # At T=50K, expect ~2.0 deviation; at T=100, expect ~0.004
        position_total = class_sums.sum(dim=1)  # (batch,)
        total_deviation = (position_total - T).abs().max().item()
        total_threshold = max(1.0, T * 0.0001)  # 0.01% of T
        all_passed &= result_line(
            f"Total emission marginals ≈ T={T}",
            total_deviation < total_threshold,
            f"max |total - T| = {total_deviation:.4f}",
        )
        cfg_data["checks"].append(
            {
                "name": "total_emission_marginals",
                "passed": total_deviation < total_threshold,
                "max_deviation": total_deviation,
                "T": T,
            }
        )

        # Check 4: Emission marginals should also lie in [0, 1]
        em_min, em_max, em_frac_oob = prob_range_stats(emission_marginals, tol=5e-3)
        emission_range_ok = em_frac_oob <= 5e-4 and em_min >= -0.05 and em_max <= 1.05
        all_passed &= result_line(
            "Emission marginals in [0, 1]",
            emission_range_ok,
            f"min={em_min:.6f}, max={em_max:.6f}, frac_oob={em_frac_oob:.4%}",
        )
        cfg_data["checks"].append(
            {
                "name": "emission_marginals_non_negative",
                "passed": em_min >= -0.01,
                "min": em_min,
            }
        )

        json_configs.append(cfg_data)

    # Check 5: Variable-length masking
    print("\n  Config [variable-length]: Testing masking")
    batch_vl, T_vl, C_vl, K_vl = 4, 100, 8, 10
    cum_scores, transition, duration_bias, _ = make_inputs(
        batch_vl, T_vl, C_vl, K_vl, device=device, seed=seed
    )
    # Variable lengths: some shorter than T
    lengths_vl = torch.tensor([100, 80, 50, 30], device=device, dtype=torch.long)

    log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
        cum_scores, transition, duration_bias, lengths_vl, K_vl
    )
    marginals = launch_streaming_triton_marginals(
        cum_scores,
        transition,
        duration_bias,
        lengths_vl,
        log_Z,
        ring_ckpts,
        log_norm_ckpts,
        interval,
    )

    # Marginals beyond each sequence's length should be 0
    padding_ok = True
    for b in range(batch_vl):
        L = lengths_vl[b].item()
        if L < T_vl:
            padding_vals = marginals[b, L:].abs().max().item()
            if padding_vals > 1e-5:
                padding_ok = False
                print(f"    WARNING: batch {b}, L={L}: max padding marginal = {padding_vals:.6e}")
    all_passed &= result_line("Padding positions have zero marginals", padding_ok)

    json_data = {
        "configs": json_configs,
        "variable_length_passed": bool(padding_ok),
        "all_passed": bool(all_passed),
    }
    return all_passed, json_data


# ======================================================================
# §2  Finite differences (gold standard)
# ======================================================================


def test_finite_differences(
    cfg: ScaleConfig,
    device: str,
    params: list[str] | None = None,
    seed: int = 42,
) -> tuple[bool, dict]:
    """Compare autograd to numerical gradient via central differences.

    This is the gold-standard correctness test. If autograd matches finite
    differences, the backward pass is mathematically correct — regardless of
    what the PyTorch reference implementation says.

    Pass criteria (per parameter):
      1. Cosine similarity > 0.999 (directional agreement)
      2. Normalized max error < 5% (magnitude agreement)
         where normalized = max|ag - fd| / max(max|ag|, max|fd|, eps)

    The element-wise allclose fraction is reported for diagnostics but NOT
    used as a pass criterion. At eps=1e-3, central differences have O(eps²)
    truncation error and O(machine_eps/eps) rounding error, which dominates
    at near-zero gradient elements. Cosine similarity and normalized max
    error are robust to this and together fully characterize correctness.
    """
    section_header("§2  FINITE DIFFERENCES (Gold Standard)")

    T, C, K = cfg.fd_T, cfg.fd_C, cfg.fd_K
    batch = 1  # Keep batch=1 for tractable finite diff
    eps = cfg.fd_eps
    all_params = params or ["cum_scores", "transition", "duration_bias"]

    print(f"  Config: batch={batch}, T={T}, C={C}, K={K}, eps={eps:.0e}")
    print(f"  Parameters: {', '.join(all_params)}")
    print()

    cum_scores, transition, duration_bias, lengths = make_inputs(
        batch, T, C, K, device=device, seed=seed
    )

    all_passed = True
    json_params = []

    def forward_fn(cs, tr, db):
        """Compute log Z given current parameters."""
        return semi_crf_streaming_forward(
            cs,
            tr,
            db,
            lengths,
            K,
            use_triton=True,
        ).sum()

    # Get autograd gradients
    cs_ag = cum_scores.clone().detach().requires_grad_(True)
    tr_ag = transition.clone().detach().requires_grad_(True)
    db_ag = duration_bias.clone().detach().requires_grad_(True)

    log_Z = forward_fn(cs_ag, tr_ag, db_ag)
    log_Z.backward()

    autograd_grads = {
        "cum_scores": cs_ag.grad.clone(),
        "transition": tr_ag.grad.clone(),
        "duration_bias": db_ag.grad.clone(),
    }

    for param_name in all_params:
        if param_name == "cum_scores":
            param = cum_scores
        elif param_name == "transition":
            param = transition
        elif param_name == "duration_bias":
            param = duration_bias
        else:
            print(f"  Unknown parameter: {param_name}")
            continue

        autograd = autograd_grads[param_name]

        t0 = time.perf_counter()

        # For cum_scores, skip the t=0 slice (always 0, no gradient)
        if param_name == "cum_scores":
            flat_param = param[:, 1:, :].reshape(-1)
            flat_numerical = torch.zeros_like(flat_param)
            flat_autograd = autograd[:, 1:, :].reshape(-1)
        else:
            flat_param = param.view(-1)
            flat_numerical = torch.zeros_like(flat_param)
            flat_autograd = autograd.view(-1)

        for i in range(flat_param.numel()):
            orig = flat_param[i].item()

            flat_param[i] = orig + eps
            f_plus = forward_fn(cum_scores, transition, duration_bias).item()

            flat_param[i] = orig - eps
            f_minus = forward_fn(cum_scores, transition, duration_bias).item()

            flat_param[i] = orig  # restore

            flat_numerical[i] = (f_plus - f_minus) / (2 * eps)

        elapsed = time.perf_counter() - t0

        # ── Metrics ──

        abs_diff = (flat_autograd - flat_numerical).abs()

        # 1. Cosine similarity (directional agreement — insensitive to scale)
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_autograd.unsqueeze(0), flat_numerical.unsqueeze(0)
        ).item()

        # 2. Normalized max error (magnitude agreement — insensitive to near-zero)
        # Uses the larger of the two gradient magnitudes as the scale
        grad_scale = max(flat_autograd.abs().max().item(), flat_numerical.abs().max().item(), 1e-8)
        normalized_max_err = abs_diff.max().item() / grad_scale

        # 3. Element-wise allclose fraction
        # |ag - fd| < atol + rtol * |fd|  with atol=1e-3, rtol=0.05
        close_mask = abs_diff < (1e-3 + 0.05 * flat_numerical.abs())
        frac_close = close_mask.float().mean().item()

        # 4. Informational: mean relative error (NOT used for pass/fail)
        denom = flat_numerical.abs().clamp(min=1e-8)
        mean_rel = (abs_diff / denom).mean().item()

        # ── Pass criteria ──
        cos_pass = cos_sim > 0.999
        norm_pass = normalized_max_err < 0.05
        passed = cos_pass and norm_pass

        print(f"  {param_name} ({flat_param.numel()} elements, {elapsed:.1f}s):")
        print(f"    Cosine similarity:   {cos_sim:.8f}  {'✓' if cos_pass else '✗'} (> 0.999)")
        print(
            f"    Normalized max err:  {normalized_max_err:.4f}  {'✓' if norm_pass else '✗'} (< 0.05)"
        )
        print(f"    Fraction allclose:   {frac_close:.4f}  (info only)")
        print(f"    (info) Mean rel err: {mean_rel:.2e}")
        print(f"    (info) Max abs diff: {abs_diff.max().item():.2e}")

        all_passed &= result_line(
            f"{param_name} finite-diff",
            passed,
            f"cos={cos_sim:.6f}, norm_max={normalized_max_err:.4f}, frac_close={frac_close:.2%}",
        )
        print()

        json_params.append(
            {
                "name": param_name,
                "elements": flat_param.numel(),
                "elapsed_s": round(elapsed, 1),
                "cosine_similarity": cos_sim,
                "normalized_max_err": normalized_max_err,
                "fraction_allclose": frac_close,
                "mean_rel_err": mean_rel,
                "max_abs_diff": abs_diff.max().item(),
                "passed": bool(passed),
            }
        )

    json_data = {
        "config": {"batch": batch, "T": T, "C": C, "K": K, "eps": eps},
        "parameters": json_params,
        "all_passed": bool(all_passed),
    }
    return all_passed, json_data


# ======================================================================
# §3  Training convergence
# ======================================================================


def test_training_convergence(cfg: ScaleConfig, device: str, seed: int = 42) -> tuple[bool, dict]:
    """Train with multiple backends, compare final loss.

    All backends compute valid gradients (just with different FP rounding
    from reduction order). If all converge to similar final loss, the
    gradients are functionally equivalent for optimization.

    Backends tested:
      - pytorch: Streaming PyTorch reference (on-the-fly edge computation)
      - triton: Streaming Triton kernel (on-the-fly edge computation)
      - dp_standard: pytorch-struct _dp_standard (materialized edge + list comprehension)
    """
    section_header("§3  TRAINING CONVERGENCE")

    T, C, K = cfg.conv_T, cfg.conv_C, cfg.conv_K
    batch = 4
    n_epochs = cfg.conv_epochs
    hidden_dim = 32
    lr = 1e-3

    print(f"  Config: batch={batch}, T={T}, C={C}, K={K}")
    print(f"  Training: {n_epochs} epochs, lr={lr}, hidden_dim={hidden_dim}")
    print()

    # Generate fixed synthetic data (same for all backends)
    set_all_seeds(seed)

    # Synthetic encoder outputs and labels
    hidden_states = torch.randn(batch, T, hidden_dim, device=device)
    lengths = torch.full((batch,), T, device=device, dtype=torch.long)

    # Create structured labels: segments of varying length
    labels = torch.zeros(batch, T, device=device, dtype=torch.long)
    for b in range(batch):
        t = 0
        while t < T:
            c = torch.randint(0, C, (1,)).item()
            seg_len = torch.randint(1, min(K, T - t) + 1, (1,)).item()
            labels[b, t : t + seg_len] = c
            t += seg_len

    all_passed = True
    results = {}

    # Backend configs: (display_name, use_triton, backend_str)
    # Streaming backends compute edges on-the-fly; exact backends materialize
    # the full O(TKC²) edge tensor then run DP on it.
    backends = [
        ("pytorch", False, "streaming"),
        ("triton", True, "streaming"),
        ("dp_standard", False, "dp_standard"),
    ]

    for backend_name, use_triton, backend_str in backends:
        # Skip Triton if not available
        if use_triton and not HAS_TRITON:
            print(f"  Skipping {backend_name}: Triton not available")
            continue
        if use_triton and device != "cuda":
            print(f"  Skipping {backend_name}: requires CUDA")
            continue

        # Fresh model with identical initialization
        set_all_seeds(seed + 1)
        from flash_semicrf import SemiMarkovCRFHead

        model = SemiMarkovCRFHead(
            num_classes=C,
            max_duration=K,
            hidden_dim=hidden_dim,
            init_scale=0.1,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []
        t0 = time.perf_counter()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = model.compute_loss(
                hidden_states,
                lengths,
                labels,
                use_triton=use_triton,
                backend=backend_str,
            )
            loss.backward()

            # Check for NaN gradients
            has_nan = False
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    print(f"    WARNING: NaN gradient in {name} at epoch {epoch}")
                    break

            if has_nan:
                print(f"    Stopping {backend_name} early due to NaN gradients")
                break

            optimizer.step()
            losses.append(loss.item())

        elapsed = time.perf_counter() - t0
        final_loss = losses[-1] if losses else float("nan")
        results[backend_name] = {
            "losses": losses,
            "final_loss": final_loss,
            "elapsed": elapsed,
        }

        print(
            f"  {backend_name:>12}: final_loss={final_loss:.4f}, "
            f"time={elapsed:.1f}s, epochs={len(losses)}"
        )

    # Compare all backends against the reference (first available)
    ref_name = next((name for name in ["pytorch", "dp_standard"] if name in results), None)
    if ref_name is None or len(results) < 2:
        print("  Cannot compare: need at least two backends")
        return False, {"all_passed": False}

    ref_loss = results[ref_name]["final_loss"]
    ref_losses = results[ref_name]["losses"]

    print()
    print(f"  Final loss comparison (reference: {ref_name}):")
    print(f"    {'Backend':<14} {'Final Loss':>12} {'Abs Diff':>12} {'Rel Diff':>10}")
    print(f"    {'-'*14} {'-'*12} {'-'*12} {'-'*10}")

    for name, res in results.items():
        loss_diff = abs(res["final_loss"] - ref_loss)
        loss_rel = loss_diff / (abs(ref_loss) + 1e-8)
        marker = "(ref)" if name == ref_name else ""
        print(
            f"    {name:<14} {res['final_loss']:>12.6f} {loss_diff:>12.6f} {loss_rel:>9.4%} {marker}"
        )

    # Build JSON comparisons while checking
    json_comparisons = []

    # Check all backends agree within 5% relative to reference
    for name, res in results.items():
        if name == ref_name:
            continue
        loss_diff = abs(res["final_loss"] - ref_loss)
        loss_rel = loss_diff / (abs(ref_loss) + 1e-8)
        all_passed &= result_line(
            f"{name} vs {ref_name} agreement", loss_rel < 0.05, f"rel_diff={loss_rel:.4%}"
        )
        json_comparisons.append(
            {
                "name": name,
                "abs_diff": loss_diff,
                "rel_diff": loss_rel,
            }
        )

    # Check all backends converge (loss decreases)
    for name, res in results.items():
        if res["losses"]:
            converged = res["losses"][-1] < res["losses"][0]
            all_passed &= result_line(f"{name} converges (loss decreases)", converged)

    # Check loss curves are correlated (all pairs vs reference)
    ref_tensor = torch.tensor(ref_losses)
    for name, res in results.items():
        if name == ref_name:
            continue
        other_tensor = torch.tensor(res["losses"])
        min_len = min(len(ref_tensor), len(other_tensor))
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_tensor[:min_len].unsqueeze(0),
            other_tensor[:min_len].unsqueeze(0),
        ).item()
        all_passed &= result_line(
            f"{name} vs {ref_name} loss curve", cos_sim > 0.99, f"cosine_sim={cos_sim:.6f}"
        )
        # Attach curve cosine to the existing comparison entry
        for comp in json_comparisons:
            if comp["name"] == name:
                comp["loss_curve_cosine"] = cos_sim
                break

    json_backends = []
    for name, res in results.items():
        json_backends.append(
            {
                "name": name,
                "final_loss": res["final_loss"],
                "elapsed_s": round(res["elapsed"], 1),
                "epochs": len(res["losses"]),
                "is_reference": name == ref_name,
            }
        )

    json_data = {
        "config": {
            "batch": batch,
            "T": T,
            "C": C,
            "K": K,
            "epochs": n_epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
        },
        "reference": ref_name,
        "backends": json_backends,
        "comparisons": json_comparisons,
        "all_passed": bool(all_passed),
    }
    return all_passed, json_data


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Ground-truth correctness validation for Semi-CRF Triton kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test",
        "-t",
        choices=["self-consistency", "finite-diff", "convergence", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--scale",
        "-s",
        choices=list(SCALES.keys()),
        default="medium",
        help="Test scale (default: medium)",
    )
    parser.add_argument(
        "--genome",
        action="store_true",
        help="Shortcut for --scale genome --test self-consistency (genome-scale sanity check)",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        choices=["cum_scores", "transition", "duration_bias"],
        default=None,
        help="Parameters for finite-diff check (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export results to JSON file for downstream processing (e.g., LaTeX tables)",
    )
    args = parser.parse_args()

    # Handle --genome shortcut
    if args.genome:
        args.scale = "genome"
        args.test = "self-consistency"

    cfg = SCALES[args.scale]

    print("=" * 72)
    print("  SEMI-CRF TRITON KERNEL: GROUND-TRUTH CORRECTNESS VALIDATION")
    print("=" * 72)
    print(f"  Scale: {cfg.name}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Triton available: {HAS_TRITON}")
    print(f"  PyTorch: {torch.__version__}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n  ERROR: CUDA requested but not available.")
        return 1
    if not HAS_TRITON:
        print("\n  ERROR: Triton not available. Install triton to run these tests.")
        return 1

    tests_to_run = (
        ["self-consistency", "finite-diff", "convergence"] if args.test == "all" else [args.test]
    )

    overall_passed = True
    results_summary = []
    json_data = {
        "config": {
            "scale": cfg.name,
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if args.device == "cuda" and torch.cuda.is_available()
                else None
            ),
            "pytorch": torch.__version__,
        },
    }

    for test_name in tests_to_run:
        if test_name == "self-consistency":
            passed, data = test_self_consistency(cfg, args.device, args.seed)
            json_data["self_consistency"] = data
        elif test_name == "finite-diff":
            passed, data = test_finite_differences(
                cfg, args.device, params=args.params, seed=args.seed
            )
            json_data["finite_diff"] = data
        elif test_name == "convergence":
            passed, data = test_training_convergence(cfg, args.device, args.seed)
            json_data["convergence"] = data
        else:
            continue

        results_summary.append((test_name, passed))
        overall_passed &= passed

    # Export JSON if requested
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n  Results exported to {args.json}")

    # Final summary
    section_header("SUMMARY")
    for name, passed in results_summary:
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {status}  {name}")

    print()
    if overall_passed:
        print("  ✓ ALL TESTS PASSED")
        print()
        print("  The Triton Semi-CRF implementation is mathematically correct")
        print("  for the validation stages executed in this run:")
        for name, _ in results_summary:
            if name == "self-consistency":
                print("  - Marginals satisfy probabilistic invariants")
            elif name == "finite-diff":
                print("  - Autograd matches finite differences")
            elif name == "convergence":
                print("  - Training converges equivalently across all backends")
                print("    (streaming PyTorch, Triton, pytorch-struct dp_standard)")
    else:
        print("  ✗ SOME TESTS FAILED")
        print("  Review the detailed output above for diagnostics.")

    print("=" * 72)
    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
