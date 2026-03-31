#!/usr/bin/env python3
"""TIMIT model comparison, duration analysis visualization, and uncertainty analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .timit_data import MAX_HEATMAP_LABELS, NUM_PHONES, PHONE_COLORS, PHONES_39, TIMITDataset
from .timit_lightning import HAS_LIGHTNING, timit_collate_for_lightning
from .timit_metrics import (
    compute_phone_error_rate,
    labels_to_segments,
    load_corpus_duration_stats,
)
from .timit_models import HAS_TORCHCRF, BiLSTMEncoder
from .timit_training import train_model

if HAS_LIGHTNING:
    from .timit_lightning import TIMITLightningModule

logger = logging.getLogger(__name__)


# =============================================================================
# Model Comparison
# =============================================================================


def _print_duration_analysis(results: dict, has_pytorch_crf: bool = False):
    """Print duration distribution analysis comparing models."""
    print("\n" + "=" * 60)
    print("DURATION ANALYSIS (frames @ 10ms)")
    print("=" * 60)
    print("\nThis shows how well each model captures phoneme duration patterns.")
    print("Lower MAE = better duration modeling. Semi-CRF should excel here.")
    print("Semi DP-Std and Semi Triton should have similar MAE (validates Triton).\n")

    # Get reference durations from semi_crf_triton
    ref_stats = results["semi_crf_triton"].duration_stats["per_phone"]

    # Select phones to display (most frequent + phonetically interesting)
    interesting_phones = ["aa", "iy", "eh", "ah", "p", "t", "k", "s", "sh", "n", "l", "sil"]
    display_phones = [p for p in interesting_phones if ref_stats[p]["ref_count"] > 50]

    # Get stats from available models
    l_stats = results["linear_crf_triton"].duration_stats["per_phone"]
    tr_stats = results["semi_crf_triton"].duration_stats["per_phone"]
    std_model = results.get("semi_crf_dp_standard")
    py_stats = std_model.duration_stats["per_phone"] if std_model else None
    p_stats = results["pytorch_crf"].duration_stats["per_phone"] if has_pytorch_crf else None

    has_dp_standard_ref = py_stats is not None

    if has_pytorch_crf and has_dp_standard_ref:
        # 4-column: p-crf | K=1 | Std | Tr
        print(
            f"{'Phone':<6} {'Ref':>6} {'p-crf':>6} {'K=1':>6} {'Std':>6} {'Tr':>6} │ "
            f"{'MAE p':>6} {'MAE K1':>7} {'MAE Std':>8} {'MAE Tr':>7}"
        )
        print("-" * 90)

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            p_mean = p_stats[phone]["pred_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            py_mean = py_stats[phone]["pred_mean"]
            tr_mean = tr_stats[phone]["pred_mean"]
            p_mae = p_stats[phone]["mae"]
            l_mae = l_stats[phone]["mae"]
            py_mae = py_stats[phone]["mae"]
            tr_mae = tr_stats[phone]["mae"]

            best_semi_mae = min(py_mae, tr_mae)
            s_marker = "*" if best_semi_mae < p_mae and best_semi_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {p_mean:>6.1f} {l_mean:>6.1f} "
                f"{py_mean:>6.1f} {tr_mean:>6.1f} │ "
                f"{p_mae:>6.2f} {l_mae:>7.2f} {py_mae:>7.2f} {tr_mae:>6.2f}{s_marker}"
            )

        print("-" * 90)
        p_overall = results["pytorch_crf"].duration_stats["overall"]
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        std_overall = std_model.duration_stats["overall"]
        tr_overall = results["semi_crf_triton"].duration_stats["overall"]

        print(
            f"{'MAE':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['mean_absolute_error']:>6.2f} "
            f"{l_overall['mean_absolute_error']:>7.2f} "
            f"{std_overall['mean_absolute_error']:>8.2f} "
            f"{tr_overall['mean_absolute_error']:>6.2f}"
        )
        print(
            f"{'Corr':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['duration_correlation']:>6.3f} "
            f"{l_overall['duration_correlation']:>7.3f} "
            f"{std_overall['duration_correlation']:>8.3f} "
            f"{tr_overall['duration_correlation']:>6.3f}"
        )

    elif has_pytorch_crf:
        # 3-column: p-crf | K=1 | Tr
        print(
            f"{'Phone':<6} {'Ref':>6} {'p-crf':>6} {'K=1':>6} {'Tr':>6} │ "
            f"{'MAE p':>6} {'MAE K1':>7} {'MAE Tr':>7}"
        )
        print("-" * 75)

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            p_mean = p_stats[phone]["pred_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            tr_mean = tr_stats[phone]["pred_mean"]
            p_mae = p_stats[phone]["mae"]
            l_mae = l_stats[phone]["mae"]
            tr_mae = tr_stats[phone]["mae"]

            s_marker = "*" if tr_mae < p_mae and tr_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {p_mean:>6.1f} {l_mean:>6.1f} "
                f"{tr_mean:>6.1f} │ "
                f"{p_mae:>6.2f} {l_mae:>7.2f} {tr_mae:>6.2f}{s_marker}"
            )

        print("-" * 75)
        p_overall = results["pytorch_crf"].duration_stats["overall"]
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        tr_overall = results["semi_crf_triton"].duration_stats["overall"]

        print(
            f"{'MAE':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['mean_absolute_error']:>6.2f} "
            f"{l_overall['mean_absolute_error']:>7.2f} "
            f"{tr_overall['mean_absolute_error']:>6.2f}"
        )
        print(
            f"{'Corr':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{p_overall['duration_correlation']:>6.3f} "
            f"{l_overall['duration_correlation']:>7.3f} "
            f"{tr_overall['duration_correlation']:>6.3f}"
        )

    elif has_dp_standard_ref:
        # 3-column: K=1 | Std | Tr
        print(
            f"{'Phone':<6} {'Ref':>6} {'K=1':>6} {'Std':>6} {'Tr':>6} │ "
            f"{'MAE K1':>7} {'MAE Std':>8} {'MAE Tr':>7}"
        )
        print("-" * 72)

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            py_mean = py_stats[phone]["pred_mean"]
            tr_mean = tr_stats[phone]["pred_mean"]
            l_mae = l_stats[phone]["mae"]
            py_mae = py_stats[phone]["mae"]
            tr_mae = tr_stats[phone]["mae"]

            best_semi_mae = min(py_mae, tr_mae)
            s_marker = "*" if best_semi_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {l_mean:>6.1f} "
                f"{py_mean:>6.1f} {tr_mean:>6.1f} │ "
                f"{l_mae:>7.2f} {py_mae:>8.2f} {tr_mae:>6.2f}{s_marker}"
            )

        print("-" * 72)
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        std_overall = std_model.duration_stats["overall"]
        tr_overall = results["semi_crf_triton"].duration_stats["overall"]

        print(
            f"{'MAE':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['mean_absolute_error']:>7.2f} "
            f"{std_overall['mean_absolute_error']:>8.2f} "
            f"{tr_overall['mean_absolute_error']:>6.2f}"
        )
        print(
            f"{'Corr':<6} {'-':>6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['duration_correlation']:>7.3f} "
            f"{std_overall['duration_correlation']:>8.3f} "
            f"{tr_overall['duration_correlation']:>6.3f}"
        )

    else:
        # 2-column: K=1 | Tr (default paper run)
        print(f"{'Phone':<6} {'Ref':>6} {'K=1':>6} {'Tr':>6} │ " f"{'MAE K1':>7} {'MAE Tr':>7}")
        print("-" * 55)

        for phone in display_phones:
            ref_mean = ref_stats[phone]["ref_mean"]
            l_mean = l_stats[phone]["pred_mean"]
            tr_mean = tr_stats[phone]["pred_mean"]
            l_mae = l_stats[phone]["mae"]
            tr_mae = tr_stats[phone]["mae"]

            s_marker = "*" if tr_mae < l_mae else " "

            print(
                f"{phone:<6} {ref_mean:>6.1f} {l_mean:>6.1f} "
                f"{tr_mean:>6.1f} │ "
                f"{l_mae:>7.2f} {tr_mae:>6.2f}{s_marker}"
            )

        print("-" * 55)
        l_overall = results["linear_crf_triton"].duration_stats["overall"]
        tr_overall = results["semi_crf_triton"].duration_stats["overall"]

        print(
            f"{'MAE':<6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['mean_absolute_error']:>7.2f} "
            f"{tr_overall['mean_absolute_error']:>6.2f}"
        )
        print(
            f"{'Corr':<6} {'-':>6} {'-':>6} {'-':>6} │ "
            f"{l_overall['duration_correlation']:>7.3f} "
            f"{tr_overall['duration_correlation']:>6.3f}"
        )

    print("\n* = Semi-CRF has lowest MAE for this phone")
    print("Std = Semi-CRF DP-standard (torch-struct baseline), Tr = Semi-CRF Triton (optimized)")
    print("Corr = correlation between predicted and reference mean durations")


def compare_models(
    data_dir: Path,
    max_duration: int = 30,
    include_dp_standard_ref: bool = False,
    uncertainty_analysis: Path | None = None,
    uncertainty_num_utterances: int = 8,
    uncertainty_selection: str = "confident-error",
    **kwargs,
):
    """
    Compare CRF models for the paper 3-way table (or 4-way for algorithm comparison).

    Models compared:
    1. pytorch-crf (optional): External linear CRF baseline
    2. K=1 Triton: Linear CRF via flash-semicrf streaming kernel
    3. Semi-CRF DP-standard: K>1 with torch-struct linear scan (algorithmic baseline)
    4. Semi-CRF Triton: K>1 with Triton streaming kernel (paper result)

    Args:
        include_dp_standard_ref: If True, also train the K>1 DP-standard reference
            (torch-struct linear scan) to compare against the Triton kernel.
            For paper runs, leave False — Triton correctness is covered by
            scripts/validate_correctness.py.
        uncertainty_analysis: If not None, run uncertainty analysis after comparison
            and write outputs to this directory.  Uses the trained K=1 (linear) and
            K>1 (semi-CRF Triton) models in-memory — no checkpoints required.
        uncertainty_num_utterances: Number of utterances to visualize (default 8).
        uncertainty_selection: Utterance selection strategy for uncertainty plots.
    """
    results = {}
    models = {}

    # 1. pytorch-crf baseline (optional - skip if not installed)
    if HAS_TORCHCRF:
        logger.info("=" * 60)
        logger.info("Training PYTORCH-CRF (external library baseline)")
        logger.info("=" * 60)
        pytorch_crf_model, pytorch_crf_metrics = train_model(
            data_dir, model_type="pytorch-crf", **kwargs
        )
        results["pytorch_crf"] = pytorch_crf_metrics
        models["pytorch_crf"] = pytorch_crf_model
    else:
        logger.warning("=" * 60)
        logger.warning("pytorch-crf not installed, skipping baseline comparison")
        logger.warning("Install with: pip install pytorch-crf")
        logger.warning("=" * 60)

    # 2. flash-semicrf K=1 (linear CRF via Triton streaming)
    logger.info("=" * 60)
    logger.info("Training LINEAR CRF (flash-semicrf K=1, Triton)")
    logger.info("=" * 60)
    linear_model, linear_metrics = train_model(
        data_dir, model_type="linear", backend="streaming", use_triton=True, **kwargs
    )
    results["linear_crf_triton"] = linear_metrics
    models["linear_crf_triton"] = linear_model

    # 3. Semi-CRF with DP-standard (torch-struct linear scan — optional algorithmic baseline)
    if include_dp_standard_ref:
        logger.info("=" * 60)
        logger.info(f"Training SEMI-CRF DP-STANDARD (K={max_duration}, torch-struct linear scan)")
        logger.info("=" * 60)
        dp_standard_model, dp_standard_metrics = train_model(
            data_dir,
            model_type="semicrf",
            max_duration=max_duration,
            backend="dp_standard",
            **kwargs,
        )
        results["semi_crf_dp_standard"] = dp_standard_metrics
        models["semi_crf_dp_standard"] = dp_standard_model

    # 4. Semi-CRF with Triton streaming (paper result)
    logger.info("=" * 60)
    logger.info(f"Training SEMI-CRF TRITON (K={max_duration}, streaming kernel)")
    logger.info("=" * 60)
    triton_model, triton_metrics = train_model(
        data_dir,
        model_type="semicrf",
        max_duration=max_duration,
        backend="streaming",
        use_triton=True,
        **kwargs,
    )
    results["semi_crf_triton"] = triton_metrics
    models["semi_crf_triton"] = triton_model

    # Load corpus duration statistics for comparison with raw TIMIT
    corpus_stats = load_corpus_duration_stats(data_dir)

    # Print comparison
    has_dp_standard_ref = "semi_crf_dp_standard" in results
    logger.info("\n" + "=" * 60)
    if has_dp_standard_ref and HAS_TORCHCRF:
        logger.info("4-WAY COMPARISON: pytorch-crf | K=1 Triton | Semi DP-Std | Semi Triton")
    elif has_dp_standard_ref:
        logger.info("3-WAY COMPARISON: K=1 Triton | Semi DP-Std | Semi Triton")
    elif HAS_TORCHCRF:
        logger.info("3-WAY COMPARISON: pytorch-crf | K=1 Triton | Semi Triton")
    else:
        logger.info("2-WAY COMPARISON: K=1 Triton | Semi Triton")
    logger.info("=" * 60)

    # Print comparison table
    _print_four_way_comparison(results, has_pytorch_crf=HAS_TORCHCRF)

    # Duration analysis (with raw TIMIT stats)
    _print_duration_analysis(results, has_pytorch_crf=HAS_TORCHCRF)

    # Print corpus comparison if stats available
    if corpus_stats:
        _print_corpus_comparison(results, corpus_stats)

    # Uncertainty analysis (uses in-memory models — no checkpoint round-trip)
    if uncertainty_analysis is not None:
        logger.info("\n" + "=" * 60)
        logger.info("UNCERTAINTY ANALYSIS (semi-CRF vs linear CRF)")
        logger.info("=" * 60)

        # Load eval dataset with same normalization as training
        train_dataset = TIMITDataset(data_dir / "train.jsonl", normalize=True)
        eval_dataset = TIMITDataset(
            data_dir / "test.jsonl",
            normalize=True,
            mean=train_dataset.mean,
            std=train_dataset.std,
        )

        device = next(models["semi_crf_triton"].parameters()).device
        _run_uncertainty_analysis(
            models["semi_crf_triton"],
            models["linear_crf_triton"],
            eval_dataset,
            device,
            uncertainty_analysis,
            batch_size=kwargs.get("batch_size", 32),
            num_utterances=uncertainty_num_utterances,
            selection=uncertainty_selection,
        )

    return results


def _print_corpus_comparison(results: dict, corpus_stats: dict):
    """Print comparison between model predictions and raw TIMIT corpus statistics."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH RAW TIMIT CORPUS")
    print("=" * 60)
    print("\nThis compares model predictions directly against corpus statistics")
    print("from train_segment_stats.json (computed during preprocessing).\n")

    # Select interesting phonemes
    interesting_phones = ["aa", "iy", "eh", "ah", "p", "t", "s", "sil"]
    display_phones = [p for p in interesting_phones if p in corpus_stats]

    print(f"{'Phone':<6} {'Corpus':>12} {'Semi Triton':>12} {'Diff':>10} {'Semi vs Lin':>12}")
    print(f"{'':6} {'mean±std':>12} {'mean±std':>12} {'(frames)':>10} {'improvement':>12}")
    print("-" * 65)

    semi_metrics = results["semi_crf_triton"]
    linear_metrics = results["linear_crf_triton"]

    for phone in display_phones:
        corpus = corpus_stats.get(phone, {})
        corpus_mean = corpus.get("mean", 0)
        corpus_std = corpus.get("std", 0)

        semi_stats = semi_metrics.duration_stats["per_phone"].get(phone, {})
        linear_stats = linear_metrics.duration_stats["per_phone"].get(phone, {})

        semi_mean = semi_stats.get("pred_mean", 0)
        semi_std = semi_stats.get("pred_std", 0)
        semi_mae = semi_stats.get("mae", 0)
        linear_mae = linear_stats.get("mae", 0)

        # Improvement: how much better semi-CRF is vs linear
        improvement = linear_mae - semi_mae

        print(
            f"{phone:<6} {corpus_mean:>5.1f}±{corpus_std:<4.1f} "
            f"{semi_mean:>5.1f}±{semi_std:<4.1f} "
            f"{semi_mean - corpus_mean:>+10.1f} {improvement:>+12.2f}"
        )

    print("-" * 65)
    print("Positive 'Semi vs Lin improvement' = Semi-CRF captures duration better")


def _print_four_way_comparison(results: dict, has_pytorch_crf: bool = False):
    """Print 4-way comparison table."""
    l_metrics = results["linear_crf_triton"]
    tr_metrics = results["semi_crf_triton"]
    py_metrics = results.get("semi_crf_dp_standard")  # optional algorithmic baseline
    p_metrics = results.get("pytorch_crf")

    has_dp_standard_ref = py_metrics is not None

    # Header — column layout depends on which models are present
    if has_pytorch_crf and has_dp_standard_ref:
        print(
            f"\n{'Metric':<20} {'pytorch-crf':>12} {'K=1 Triton':>12} "
            f"{'Semi DP-Std':>12} {'Semi Triton':>12} {'Δ Linear':>10} {'Δ Semi':>10}"
        )
        print("-" * 100)
    elif has_pytorch_crf:
        print(
            f"\n{'Metric':<20} {'pytorch-crf':>12} {'K=1 Triton':>12} "
            f"{'Semi Triton':>12} {'Δ Linear':>10} {'Δ Semi vs K=1':>14}"
        )
        print("-" * 90)
    elif has_dp_standard_ref:
        print(
            f"\n{'Metric':<20} {'K=1 Triton':>12} "
            f"{'Semi DP-Std':>12} {'Semi Triton':>12} {'Δ Semi':>10}"
        )
        print("-" * 80)
    else:
        print(f"\n{'Metric':<20} {'K=1 Triton':>12} {'Semi Triton':>12} {'Δ Semi vs K=1':>14}")
        print("-" * 65)

    # Phone Error Rate (lower is better)
    for metric_name, display_name in [
        ("phone_error_rate", "Phone Error Rate"),
        ("boundary_f1", "Boundary F1"),
        ("segment_f1", "Segment F1"),
    ]:
        l_val = getattr(l_metrics, metric_name)
        tr_val = getattr(tr_metrics, metric_name)
        py_val = getattr(py_metrics, metric_name) if py_metrics else None
        p_val = getattr(p_metrics, metric_name) if p_metrics else None

        if has_pytorch_crf and has_dp_standard_ref:
            delta_linear = l_val - p_val
            delta_semi = tr_val - py_val
            print(
                f"{display_name:<20} {p_val:>12.4f} {l_val:>12.4f} "
                f"{py_val:>12.4f} {tr_val:>12.4f} {delta_linear:>+10.4f} {delta_semi:>+10.4f}"
            )
        elif has_pytorch_crf:
            delta_linear = l_val - p_val
            delta_semi_vs_k1 = tr_val - l_val
            print(
                f"{display_name:<20} {p_val:>12.4f} {l_val:>12.4f} "
                f"{tr_val:>12.4f} {delta_linear:>+10.4f} {delta_semi_vs_k1:>+14.4f}"
            )
        elif has_dp_standard_ref:
            delta_semi = tr_val - py_val
            print(
                f"{display_name:<20} {l_val:>12.4f} "
                f"{py_val:>12.4f} {tr_val:>12.4f} {delta_semi:>+10.4f}"
            )
        else:
            delta_semi_vs_k1 = tr_val - l_val
            print(f"{display_name:<20} {l_val:>12.4f} {tr_val:>12.4f} {delta_semi_vs_k1:>+14.4f}")

    # Boundary F1 tolerances
    print("\nBoundary F1 at different tolerances:")
    for tol in [0, 1, 2]:
        l_val = l_metrics.boundary_f1_tolerances.get(tol, 0)
        tr_val = tr_metrics.boundary_f1_tolerances.get(tol, 0)
        py_val = py_metrics.boundary_f1_tolerances.get(tol, 0) if py_metrics else None
        p_val = p_metrics.boundary_f1_tolerances.get(tol, 0) if p_metrics else None

        if has_pytorch_crf and has_dp_standard_ref:
            print(
                f"  tol={tol:<2} {p_val:>12.4f} {l_val:>12.4f} " f"{py_val:>12.4f} {tr_val:>12.4f}"
            )
        elif has_pytorch_crf:
            print(f"  tol={tol:<2} {p_val:>12.4f} {l_val:>12.4f} {tr_val:>12.4f}")
        elif has_dp_standard_ref:
            print(f"  tol={tol:<2} {l_val:>12.4f} {py_val:>12.4f} {tr_val:>12.4f}")
        else:
            print(f"  tol={tol:<2} {l_val:>12.4f} {tr_val:>12.4f}")

    # Timing comparison
    print("\nTiming (lower is better):")
    l_time = l_metrics.training_time_per_epoch
    tr_time = tr_metrics.training_time_per_epoch

    if has_pytorch_crf and has_dp_standard_ref:
        p_time = p_metrics.training_time_per_epoch
        py_time = py_metrics.training_time_per_epoch
        speedup_linear = p_time / l_time if l_time > 0 else 0
        speedup_semi = py_time / tr_time if tr_time > 0 else 0
        print(
            f"{'Train (s/epoch)':<20} {p_time:>12.2f} {l_time:>12.2f} "
            f"{py_time:>12.2f} {tr_time:>12.2f} {speedup_linear:>9.2f}x {speedup_semi:>9.2f}x"
        )
    elif has_pytorch_crf:
        p_time = p_metrics.training_time_per_epoch
        speedup_linear = p_time / l_time if l_time > 0 else 0
        print(
            f"{'Train (s/epoch)':<20} {p_time:>12.2f} {l_time:>12.2f} "
            f"{tr_time:>12.2f} {speedup_linear:>9.2f}x"
        )
    elif has_dp_standard_ref:
        py_time = py_metrics.training_time_per_epoch
        speedup_semi = py_time / tr_time if tr_time > 0 else 0
        print(
            f"{'Train (s/epoch)':<20} {l_time:>12.2f} "
            f"{py_time:>12.2f} {tr_time:>12.2f} {speedup_semi:>9.2f}x"
        )
    else:
        print(f"{'Train (s/epoch)':<20} {l_time:>12.2f} {tr_time:>12.2f}")

    l_infer = l_metrics.inference_time
    tr_infer = tr_metrics.inference_time

    if has_pytorch_crf and has_dp_standard_ref:
        p_infer = p_metrics.inference_time
        py_infer = py_metrics.inference_time
        speedup_linear = p_infer / l_infer if l_infer > 0 else 0
        speedup_semi = py_infer / tr_infer if tr_infer > 0 else 0
        print(
            f"{'Inference (s)':<20} {p_infer:>12.2f} {l_infer:>12.2f} "
            f"{py_infer:>12.2f} {tr_infer:>12.2f} {speedup_linear:>9.2f}x {speedup_semi:>9.2f}x"
        )
    elif has_pytorch_crf:
        p_infer = p_metrics.inference_time
        speedup_linear = p_infer / l_infer if l_infer > 0 else 0
        print(
            f"{'Inference (s)':<20} {p_infer:>12.2f} {l_infer:>12.2f} "
            f"{tr_infer:>12.2f} {speedup_linear:>9.2f}x"
        )
    elif has_dp_standard_ref:
        py_infer = py_metrics.inference_time
        speedup_semi = py_infer / tr_infer if tr_infer > 0 else 0
        print(
            f"{'Inference (s)':<20} {l_infer:>12.2f} "
            f"{py_infer:>12.2f} {tr_infer:>12.2f} {speedup_semi:>9.2f}x"
        )
    else:
        print(f"{'Inference (s)':<20} {l_infer:>12.2f} {tr_infer:>12.2f}")

    # Validation notes
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if has_pytorch_crf:
        print("Linear CRF: K=1 Triton should match pytorch-crf accuracy (Δ Linear ≈ 0)")
    if has_dp_standard_ref:
        print("Semi-CRF: Triton should match DP-standard accuracy (Δ Semi ≈ 0)")
    print(
        "Timing: speedup shown as multiplier vs the slower reference (e.g., 2.0x = twice as fast)"
    )


# =============================================================================
# Uncertainty Analysis (requires flash-semicrf[lightning] + matplotlib + numpy)
# =============================================================================


def _filter_heatmap_labels(
    semi_marginals,
    linear_marginals,
    ref_labels,
    pred_labels_semi,
    pred_labels_linear,
    max_heatmap_labels,
):
    """Return (indices, names) of phone classes to show in the posterior heatmap.

    Includes the union of reference + predicted labels from both models, then
    pads with highest-mean-marginal remaining classes up to max_heatmap_labels.
    """
    active = sorted(set(ref_labels) | set(pred_labels_semi) | set(pred_labels_linear))

    if len(active) >= max_heatmap_labels:
        keep = active[:max_heatmap_labels]
    else:
        mean_marg = (semi_marginals.mean(0) + linear_marginals.mean(0)) / 2.0
        active_set = set(active)
        remaining = sorted(
            [c for c in range(semi_marginals.shape[1]) if c not in active_set],
            key=lambda c: -mean_marg[c],
        )
        n_extra = max_heatmap_labels - len(active)
        keep = active + remaining[:n_extra]

    names = [PHONES_39[i] if i < len(PHONES_39) else str(i) for i in keep]
    return keep, names


def run_uncertainty_inference(model, dataset, device, batch_size=32, indices=None):
    """Run forward passes on a dataset and collect per-utterance uncertainty results.

    Parameters
    ----------
    model : TIMITLightningModule
    dataset : TIMITDataset
    device : torch.device
    batch_size : int
    indices : list[int] or None
        Run on a subset of the dataset. None runs the full dataset.

    Returns
    -------
    list[dict] — one dict per utterance with keys:
        utterance_id, length, reference_labels, pred_labels, viterbi_segments,
        boundary_marginals (T,), position_marginals (T, C), entropy, per
    """
    from torch.utils.data import DataLoader, Subset

    if indices is not None:
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=timit_collate_for_lightning,
    )

    model = model.to(device)
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"]
            utterance_ids = batch["utterance_ids"]

            hidden = model._encode(inputs)

            boundary_margs = model.crf.compute_boundary_marginals(hidden, lengths)
            # compute_position_marginals uses torch.enable_grad() internally
            position_margs = model.crf.compute_position_marginals(hidden, lengths)
            entropy = model.crf.compute_entropy_streaming(hidden, lengths)
            viterbi_result = model.crf.decode_with_traceback(hidden, lengths)

            for i, seg_list in enumerate(viterbi_result.segments):
                seq_len = int(lengths[i].item())

                # Convert Segment list (inclusive end) to frame-level labels
                pred_labels = [0] * seq_len
                for seg in seg_list:
                    for j in range(seg.start, min(seg.end + 1, seq_len)):
                        pred_labels[j] = seg.label

                ref_labels = labels[i, :seq_len].cpu().tolist()
                per = compute_phone_error_rate([pred_labels], [ref_labels])

                results.append(
                    {
                        "utterance_id": utterance_ids[i],
                        "length": seq_len,
                        "reference_labels": ref_labels,
                        "pred_labels": pred_labels,
                        "viterbi_segments": seg_list,
                        "boundary_marginals": boundary_margs[i, :seq_len].cpu().float().numpy(),
                        "position_marginals": position_margs[i, :seq_len].cpu().float().numpy(),
                        "entropy": float(entropy[i].item()),
                        "per": per,
                    }
                )

    return results


def select_utterances(semi_results, linear_results, strategy, n, seed=42):
    """Return N indices into the result lists according to the given selection strategy.

    Parameters
    ----------
    semi_results, linear_results : list[dict]
        Must be the same length.
    strategy : str
        One of "confident-error", "entropy", "semi-advantage", "random".
    n : int
        Number of utterances to select.
    seed : int
        RNG seed for "random" strategy.

    Returns
    -------
    list[int]
    """
    import random as _random

    assert len(semi_results) == len(linear_results), "Result lists must be same length."
    m = len(semi_results)

    if strategy == "confident-error":
        linear_pers = np.array([r["per"] for r in linear_results])
        semi_entropies = np.array([r["entropy"] for r in semi_results])
        # Rank both signals descending; lower rank-product → higher priority
        rank_per = np.argsort(np.argsort(-linear_pers))
        rank_ent = np.argsort(np.argsort(-semi_entropies))
        order = np.argsort(rank_per * rank_ent)
    elif strategy == "entropy":
        semi_entropies = np.array([r["entropy"] for r in semi_results])
        order = np.argsort(-semi_entropies)
    elif strategy == "semi-advantage":
        linear_pers = np.array([r["per"] for r in linear_results])
        semi_pers = np.array([r["per"] for r in semi_results])
        order = np.argsort(-(linear_pers - semi_pers))
    elif strategy == "random":
        order = list(range(m))
        _random.seed(seed)
        _random.shuffle(order)
        order = np.array(order)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return order[:n].tolist()


def compute_confidence_at_errors(position_marginals, pred_labels, ref_labels):
    """Compute model confidence at correct vs error frames for one utterance.

    Parameters
    ----------
    position_marginals : np.ndarray, shape (T, C)
    pred_labels : list[int], length T
    ref_labels : list[int], length T

    Returns
    -------
    dict with keys:
        mean_max_posterior_at_errors, mean_max_posterior_at_correct,
        confidence_gap, n_error_frames, n_correct_frames
    """

    max_posterior = position_marginals.max(axis=1)  # (T,)
    pred = np.array(pred_labels)
    ref = np.array(ref_labels)

    correct_mask = pred == ref
    error_mask = ~correct_mask

    n_correct = int(correct_mask.sum())
    n_error = int(error_mask.sum())
    mean_correct = float(max_posterior[correct_mask].mean()) if n_correct > 0 else 0.0
    mean_error = float(max_posterior[error_mask].mean()) if n_error > 0 else 0.0

    return {
        "mean_max_posterior_at_errors": mean_error,
        "mean_max_posterior_at_correct": mean_correct,
        "confidence_gap": mean_correct - mean_error,
        "n_error_frames": n_error,
        "n_correct_frames": n_correct,
    }


def _draw_phone_blocks(ax, segs, T, phone_colors, phone_map, ref_segs_for_errors=None):
    """Draw colored phoneme blocks on ax (helper for supplement figure).

    segs : list[SegmentAnnotation]  (exclusive end)
    ref_segs_for_errors : list[SegmentAnnotation] or None — if given, outlines
        segments that don't exactly match a reference segment in red.
    """
    import matplotlib.pyplot as plt

    for seg in segs:
        color = phone_colors.get(seg.label, (0.5, 0.5, 0.5, 1.0))
        rect = plt.Rectangle(
            (seg.start, 0),
            seg.end - seg.start,
            1,
            facecolor=color,
            alpha=0.9,
            edgecolor="none",
        )
        ax.add_patch(rect)
        if ref_segs_for_errors is not None:
            matches = any(
                r.start == seg.start and r.end == seg.end and r.label == seg.label
                for r in ref_segs_for_errors
            )
            if not matches:
                rect.set_edgecolor("red")
                rect.set_linewidth(1.5)
        span = seg.end - seg.start
        if span >= 3:
            label_str = phone_map[seg.label] if seg.label < len(phone_map) else str(seg.label)
            ax.text(
                seg.start + span / 2,
                0.5,
                label_str,
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_yticks([])


def plot_per_utterance_uncertainty(
    semi_result,
    linear_result,
    phone_map,
    out_path,
    max_heatmap_labels=MAX_HEATMAP_LABELS,
):
    """4-panel uncertainty figure for a single utterance (main paper figure).

    Panels (height ratios [0.6, 1.4, 1.4, 0.8]):
        1. Ground truth phoneme color blocks
        2. Semi-CRF label posteriors p(c|t) — heatmap
        3. Linear CRF label posteriors p(c|t) — heatmap, same colorscale
        4. Boundary marginals overlay (semi=steelblue, linear=orange, GT=dashed red)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for uncertainty plots. pip install matplotlib")
        return

    ref_labels = semi_result["reference_labels"]
    T = len(ref_labels)
    frames = np.arange(T)

    keep, keep_names = _filter_heatmap_labels(
        semi_result["position_marginals"],
        linear_result["position_marginals"],
        ref_labels,
        semi_result["pred_labels"],
        linear_result["pred_labels"],
        max_heatmap_labels,
    )
    semi_pm = semi_result["position_marginals"][:, keep]
    linear_pm = linear_result["position_marginals"][:, keep]

    # Data-driven colorscale: use the max across both models so they share the same range
    vmax = max(semi_pm.max(), linear_pm.max(), 0.01)

    phone_colors = PHONE_COLORS
    ref_segs = labels_to_segments(ref_labels)  # SegmentAnnotation (exclusive end)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [0.4, 2.0, 2.0, 0.6]},
        sharex=True,
        layout="constrained",
    )

    # --- Panel 1: Ground truth ---
    ax = axes[0]
    for seg in ref_segs:
        color = phone_colors.get(seg.label, (0.5, 0.5, 0.5, 1.0))
        ax.axvspan(seg.start, seg.end, color=color, alpha=0.9)
        span = seg.end - seg.start
        if span >= 3:
            label_str = phone_map[seg.label] if seg.label < len(phone_map) else str(seg.label)
            ax.text(
                seg.start + span / 2,
                0.5,
                label_str,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_yticks([])
    ax.set_ylabel("GT", fontsize=9)
    ax.set_xlim(0, T)

    # --- Panel 2: Semi-CRF label posterior heatmap ---
    ax = axes[1]
    im = ax.imshow(
        semi_pm.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, T, -0.5, len(keep) - 0.5],
        vmin=0,
        vmax=vmax,
    )
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(keep_names, fontsize=5)
    ax.set_ylabel("Semi-CRF p(c|t)", fontsize=9)

    # --- Panel 3: Linear CRF label posterior heatmap (same colorscale) ---
    ax = axes[2]
    ax.imshow(
        linear_pm.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, T, -0.5, len(keep) - 0.5],
        vmin=0,
        vmax=vmax,
    )
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(keep_names, fontsize=5)
    ax.set_ylabel("Linear CRF p(c|t)", fontsize=9)

    # --- Panel 4: Boundary marginals overlay ---
    ax = axes[3]
    ax.plot(frames, semi_result["boundary_marginals"], color="steelblue", lw=1.2, label="Semi-CRF")
    ax.plot(
        frames,
        linear_result["boundary_marginals"],
        color="orange",
        lw=1.0,
        alpha=0.8,
        label="Linear CRF",
    )
    for seg in ref_segs:
        if seg.start > 0:
            ax.axvline(seg.start, color="red", lw=0.8, linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("p(boundary)", fontsize=9)
    ax.set_xlabel("Frame", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    uid = semi_result["utterance_id"]
    s_ent = semi_result["entropy"]
    s_per = semi_result["per"]
    l_per = linear_result["per"]
    fig.suptitle(
        f"{uid}  |  Semi entropy={s_ent:.2f}  |  Semi PER={s_per:.3f}  |  Linear PER={l_per:.3f}",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes[1:3], shrink=0.6, label="p(c|t)")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_utterance_uncertainty_supplement(
    semi_result,
    linear_result,
    phone_map,
    out_path,
    max_heatmap_labels=MAX_HEATMAP_LABELS,
):
    """7-panel supplement figure: adds Viterbi blocks (red outlines on errors) and
    splits boundary marginals into separate panels.

    Panels (height ratios [0.5, 0.5, 1.4, 0.5, 1.4, 0.7, 0.7]):
        1. Ground truth blocks
        2. Semi-CRF Viterbi blocks (red outline on incorrect segments)
        3. Semi-CRF posterior heatmap
        4. Linear CRF Viterbi blocks (red outline on incorrect segments)
        5. Linear CRF posterior heatmap
        6. Semi-CRF boundary marginals
        7. Linear CRF boundary marginals
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    ref_labels = semi_result["reference_labels"]
    T = len(ref_labels)
    frames = np.arange(T)

    keep, keep_names = _filter_heatmap_labels(
        semi_result["position_marginals"],
        linear_result["position_marginals"],
        ref_labels,
        semi_result["pred_labels"],
        linear_result["pred_labels"],
        max_heatmap_labels,
    )
    semi_pm = semi_result["position_marginals"][:, keep]
    linear_pm = linear_result["position_marginals"][:, keep]

    # Data-driven colorscale
    vmax = max(semi_pm.max(), linear_pm.max(), 0.01)

    phone_colors = PHONE_COLORS
    ref_segs = labels_to_segments(ref_labels)

    height_ratios = [0.4, 0.4, 2.0, 0.4, 2.0, 0.6, 0.6]
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14, 20),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
        layout="constrained",
    )

    _draw_phone_blocks(axes[0], ref_segs, T, phone_colors, phone_map)
    axes[0].set_ylabel("GT", fontsize=9)

    semi_vit_segs = labels_to_segments(semi_result["pred_labels"])
    _draw_phone_blocks(
        axes[1], semi_vit_segs, T, phone_colors, phone_map, ref_segs_for_errors=ref_segs
    )
    axes[1].set_ylabel("Semi Viterbi", fontsize=9)

    im = axes[2].imshow(
        semi_pm.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, T, -0.5, len(keep) - 0.5],
        vmin=0,
        vmax=vmax,
    )
    axes[2].set_yticks(range(len(keep)))
    axes[2].set_yticklabels(keep_names, fontsize=5)
    axes[2].set_ylabel("Semi-CRF p(c|t)", fontsize=9)

    linear_vit_segs = labels_to_segments(linear_result["pred_labels"])
    _draw_phone_blocks(
        axes[3], linear_vit_segs, T, phone_colors, phone_map, ref_segs_for_errors=ref_segs
    )
    axes[3].set_ylabel("Linear Viterbi", fontsize=9)

    axes[4].imshow(
        linear_pm.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, T, -0.5, len(keep) - 0.5],
        vmin=0,
        vmax=vmax,
    )
    axes[4].set_yticks(range(len(keep)))
    axes[4].set_yticklabels(keep_names, fontsize=5)
    axes[4].set_ylabel("Linear CRF p(c|t)", fontsize=9)

    axes[5].plot(frames, semi_result["boundary_marginals"], color="steelblue", lw=1.2)
    for seg in ref_segs:
        if seg.start > 0:
            axes[5].axvline(seg.start, color="red", lw=0.8, linestyle="--", alpha=0.7)
    axes[5].set_ylim(0, 1.05)
    axes[5].set_ylabel("Semi p(bdy)", fontsize=9)

    axes[6].plot(frames, linear_result["boundary_marginals"], color="orange", lw=1.2)
    for seg in ref_segs:
        if seg.start > 0:
            axes[6].axvline(seg.start, color="red", lw=0.8, linestyle="--", alpha=0.7)
    axes[6].set_ylim(0, 1.05)
    axes[6].set_ylabel("Linear p(bdy)", fontsize=9)
    axes[6].set_xlabel("Frame", fontsize=9)

    uid = semi_result["utterance_id"]
    s_ent = semi_result["entropy"]
    s_per = semi_result["per"]
    l_per = linear_result["per"]
    fig.suptitle(
        f"{uid}  |  Semi entropy={s_ent:.2f}  |  Semi PER={s_per:.3f}  |  Linear PER={l_per:.3f}",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes[2:5], shrink=0.5, label="p(c|t)")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_vs_per_scatter(semi_results, linear_results, out_path):
    """Scatter: semi-CRF sequence entropy (x) vs PER for both models (y).

    Annotates Pearson r for each model. Claim: high-entropy utterances have
    higher error rates — semi-CRF "knows what it doesn't know".
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required. pip install matplotlib")
        return

    def _pearsonr(x, y):
        xm = x - x.mean()
        ym = y - y.mean()
        denom = float(np.sqrt((xm**2).sum() * (ym**2).sum()))
        return float(np.dot(xm, ym) / denom) if denom > 0 else 0.0

    semi_ents = np.array([r["entropy"] for r in semi_results])
    semi_pers = np.array([r["per"] for r in semi_results])
    linear_pers = np.array([r["per"] for r in linear_results])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(semi_ents, semi_pers, c="steelblue", marker="o", s=18, alpha=0.55, label="Semi-CRF")
    ax.scatter(semi_ents, linear_pers, c="orange", marker="s", s=18, alpha=0.55, label="Linear CRF")

    r_semi = _pearsonr(semi_ents, semi_pers)
    r_linear = _pearsonr(semi_ents, linear_pers)
    ax.text(
        0.05,
        0.95,
        f"Semi-CRF r = {r_semi:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        color="steelblue",
        va="top",
    )
    ax.text(
        0.05,
        0.89,
        f"Linear CRF r = {r_linear:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        color="darkorange",
        va="top",
    )

    ax.set_xlabel("Semi-CRF Sequence Entropy", fontsize=11)
    ax.set_ylabel("Phone Error Rate (PER)", fontsize=11)
    ax.set_title("Entropy vs. PER: Semi-CRF signals its own uncertainty", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_summary(
    semi_results, linear_results, semi_conf_agg, linear_conf_agg, out_path
):
    """Two-panel calibration summary figure.

    Panel A — Reliability diagram: mean PER per semi-CRF entropy quintile for both models.
    Panel B — Confidence-at-errors bar chart: mean max posterior at correct vs error frames.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    semi_ents = np.array([r["entropy"] for r in semi_results])
    semi_pers = np.array([r["per"] for r in semi_results])
    linear_pers = np.array([r["per"] for r in linear_results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Reliability diagram — PER by entropy quintile
    edges = np.percentile(semi_ents, [0, 20, 40, 60, 80, 100])
    bin_labels, semi_bins, linear_bins = [], [], []
    for qi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (semi_ents >= lo) & (semi_ents <= hi)
        if mask.sum() > 0:
            bin_labels.append(f"Q{qi + 1}")
            semi_bins.append(float(semi_pers[mask].mean()))
            linear_bins.append(float(linear_pers[mask].mean()))

    x = np.arange(len(bin_labels))
    width = 0.35
    ax1.bar(x - width / 2, semi_bins, width, label="Semi-CRF", color="steelblue", alpha=0.85)
    ax1.bar(x + width / 2, linear_bins, width, label="Linear CRF", color="orange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=9)
    ax1.set_xlabel("Entropy Quintile (Semi-CRF)", fontsize=10)
    ax1.set_ylabel("Mean PER", fontsize=10)
    ax1.set_title("A. Reliability: PER by Entropy Quintile", fontsize=10)
    ax1.legend(fontsize=9)

    # Panel B: Confidence at errors vs correct frames
    bar_labels = ["Semi\nErrors", "Semi\nCorrect", "Linear\nErrors", "Linear\nCorrect"]
    values = [
        semi_conf_agg["mean_max_posterior_at_errors"],
        semi_conf_agg["mean_max_posterior_at_correct"],
        linear_conf_agg["mean_max_posterior_at_errors"],
        linear_conf_agg["mean_max_posterior_at_correct"],
    ]
    colors = ["steelblue", "steelblue", "orange", "orange"]
    alphas = [0.55, 0.90, 0.55, 0.90]
    bars = ax2.bar(bar_labels, values, color=colors)
    for bar, alpha, val in zip(bars, alphas, values, strict=True):
        bar.set_alpha(alpha)
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    semi_gap = semi_conf_agg["confidence_gap"]
    linear_gap = linear_conf_agg["confidence_gap"]
    ax2.text(
        0.5,
        0.97,
        f"Semi gap: {semi_gap:+.3f}",
        transform=ax2.transAxes,
        fontsize=9,
        ha="center",
        color="steelblue",
        va="top",
    )
    ax2.text(
        0.5,
        0.91,
        f"Linear gap: {linear_gap:+.3f}",
        transform=ax2.transAxes,
        fontsize=9,
        ha="center",
        color="darkorange",
        va="top",
    )
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Mean Max Posterior p(c*|t)", fontsize=10)
    ax2.set_title("B. Confidence: Correct vs. Error Frames", fontsize=10)
    ax2.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_ckpt_model(ckpt_path, max_duration, device, hidden_dim=256, num_layers=3):
    """Load a TIMITLightningModule from a Lightning checkpoint.

    Infers input_dim from encoder.input_proj.weight in the state dict.
    Does not use Lightning's load_from_checkpoint (encoder/crf are not hparams).
    """
    from flash_semicrf import UncertaintySemiMarkovCRFHead

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Infer input_dim from encoder projection weight (shape: hidden_dim × input_dim)
    proj_key = "encoder.input_proj.weight"
    if proj_key not in state_dict:
        raise KeyError(
            f"Cannot infer input_dim: key '{proj_key}' not found in checkpoint. "
            "Pass --input-dim explicitly."
        )
    input_dim = state_dict[proj_key].shape[1]

    # Infer use_sequence_boundaries from checkpoint keys
    use_seq_bounds = "crf.pi_start" in state_dict

    encoder = BiLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    crf = UncertaintySemiMarkovCRFHead(
        num_classes=NUM_PHONES,
        max_duration=max_duration,
        hidden_dim=hidden_dim,
        use_sequence_boundaries=use_seq_bounds,
    )
    model = TIMITLightningModule(encoder=encoder, crf=crf, scheduler="none")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _run_uncertainty_analysis(
    semi_model,
    linear_model,
    eval_dataset,
    device: torch.device,
    out_dir: Path,
    *,
    batch_size: int = 32,
    num_utterances: int = 8,
    selection: str = "confident-error",
    max_heatmap_labels: int = MAX_HEATMAP_LABELS,
):
    """Core uncertainty analysis: inference, metrics, plots, and summary.

    Shared by ``run_analyze_uncertainty`` (checkpoint path) and
    ``compare_models`` (in-memory models).  Both ``semi_model`` and
    ``linear_model`` must expose ``._encode(inputs)`` and ``.crf`` with
    ``compute_boundary_marginals``, ``compute_position_marginals``,
    ``compute_entropy_streaming``, and ``decode_with_traceback``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run inference on both models ----
    print(f"Running semi-CRF uncertainty inference on {len(eval_dataset)} utterances...")
    semi_results = run_uncertainty_inference(
        semi_model, eval_dataset, device, batch_size=batch_size
    )

    print(f"Running linear CRF uncertainty inference on {len(eval_dataset)} utterances...")
    linear_results = run_uncertainty_inference(
        linear_model, eval_dataset, device, batch_size=batch_size
    )

    # ---- Aggregate confidence-at-errors ----
    semi_n_correct_total = semi_n_error_total = 0
    semi_sum_correct = semi_sum_error = 0.0
    linear_n_correct_total = linear_n_error_total = 0
    linear_sum_correct = linear_sum_error = 0.0

    for sr, lr in zip(semi_results, linear_results, strict=True):
        sc = compute_confidence_at_errors(
            sr["position_marginals"], sr["pred_labels"], sr["reference_labels"]
        )
        lc = compute_confidence_at_errors(
            lr["position_marginals"], lr["pred_labels"], lr["reference_labels"]
        )
        # Store per-utterance gap in result dict for optional downstream use
        sr["confidence_gap"] = sc["confidence_gap"]
        lr["confidence_gap"] = lc["confidence_gap"]

        # Frame-count-weighted accumulation
        semi_sum_correct += sc["mean_max_posterior_at_correct"] * sc["n_correct_frames"]
        semi_sum_error += sc["mean_max_posterior_at_errors"] * sc["n_error_frames"]
        semi_n_correct_total += sc["n_correct_frames"]
        semi_n_error_total += sc["n_error_frames"]

        linear_sum_correct += lc["mean_max_posterior_at_correct"] * lc["n_correct_frames"]
        linear_sum_error += lc["mean_max_posterior_at_errors"] * lc["n_error_frames"]
        linear_n_correct_total += lc["n_correct_frames"]
        linear_n_error_total += lc["n_error_frames"]

    semi_conf_agg = {
        "mean_max_posterior_at_correct": semi_sum_correct / max(semi_n_correct_total, 1),
        "mean_max_posterior_at_errors": semi_sum_error / max(semi_n_error_total, 1),
        "confidence_gap": (semi_sum_correct / max(semi_n_correct_total, 1))
        - (semi_sum_error / max(semi_n_error_total, 1)),
    }
    linear_conf_agg = {
        "mean_max_posterior_at_correct": linear_sum_correct / max(linear_n_correct_total, 1),
        "mean_max_posterior_at_errors": linear_sum_error / max(linear_n_error_total, 1),
        "confidence_gap": (linear_sum_correct / max(linear_n_correct_total, 1))
        - (linear_sum_error / max(linear_n_error_total, 1)),
    }

    # ---- Select utterances ----
    indices = select_utterances(semi_results, linear_results, selection, num_utterances)
    print(f"Selected {len(indices)} utterances via '{selection}' strategy: " f"indices {indices}")

    # ---- Per-utterance figures ----
    for rank, idx in enumerate(indices):
        sr = semi_results[idx]
        lr = linear_results[idx]
        uid_safe = sr["utterance_id"].replace("/", "_").replace("\\", "_")
        stem = f"uncertainty_{rank:02d}_{uid_safe}"

        main_path = out_dir / f"{stem}.png"
        sup_path = out_dir / f"{stem}_supplement.png"

        print(f"  [{rank + 1}/{len(indices)}] {sr['utterance_id']} → {stem}.png")
        plot_per_utterance_uncertainty(
            sr, lr, PHONES_39, main_path, max_heatmap_labels=max_heatmap_labels
        )
        plot_per_utterance_uncertainty_supplement(
            sr, lr, PHONES_39, sup_path, max_heatmap_labels=max_heatmap_labels
        )

    # ---- Summary figures ----
    scatter_path = out_dir / "entropy_vs_per_scatter.png"
    print(f"Saving scatter plot → {scatter_path}")
    plot_entropy_vs_per_scatter(semi_results, linear_results, scatter_path)

    cal_path = out_dir / "calibration_summary.png"
    print(f"Saving calibration summary → {cal_path}")
    plot_calibration_summary(semi_results, linear_results, semi_conf_agg, linear_conf_agg, cal_path)

    # ---- Print summary ----
    semi_mean_per = float(np.mean([r["per"] for r in semi_results]))
    linear_mean_per = float(np.mean([r["per"] for r in linear_results]))
    semi_mean_ent = float(np.mean([r["entropy"] for r in semi_results]))

    print("\n" + "=" * 60)
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Utterances evaluated:   {len(semi_results)}")
    print(f"  Semi-CRF  mean PER:     {semi_mean_per:.4f}")
    print(f"  Linear CRF mean PER:    {linear_mean_per:.4f}")
    print(f"  Semi-CRF  mean entropy: {semi_mean_ent:.4f}")
    print("-" * 60)
    print(f"  Semi-CRF  confidence gap (correct−error): {semi_conf_agg['confidence_gap']:+.4f}")
    print(f"  Linear CRF confidence gap (correct−error): {linear_conf_agg['confidence_gap']:+.4f}")
    print(
        f"  Semi-CRF  p(c|t) at correct frames: {semi_conf_agg['mean_max_posterior_at_correct']:.4f}"
    )
    print(
        f"  Semi-CRF  p(c|t) at error frames:   {semi_conf_agg['mean_max_posterior_at_errors']:.4f}"
    )
    print(
        f"  Linear CRF p(c|t) at correct frames: {linear_conf_agg['mean_max_posterior_at_correct']:.4f}"
    )
    print(
        f"  Linear CRF p(c|t) at error frames:   {linear_conf_agg['mean_max_posterior_at_errors']:.4f}"
    )
    print("=" * 60)
    print(f"Output written to: {out_dir}/")


def run_analyze_uncertainty(args):
    """Orchestration for the analyze-uncertainty subcommand (checkpoint path)."""
    if not HAS_LIGHTNING:
        print(
            "PyTorch Lightning is required for this command. "
            "Install with: pip install flash-semicrf[lightning]"
        )
        return

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)

    # ---- Load datasets (use train stats for normalization) ----
    print("Loading datasets...")
    train_dataset = TIMITDataset(
        args.data_dir / "train.jsonl",
        normalize=True,
    )
    jsonl_file = "test.jsonl"  # both val and test map to TIMIT test split
    eval_dataset = TIMITDataset(
        args.data_dir / jsonl_file,
        normalize=True,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    # ---- Load models ----
    print(f"Loading semi-CRF checkpoint: {args.semi_crf_checkpoint}")
    semi_model = _load_ckpt_model(
        args.semi_crf_checkpoint,
        max_duration=args.semi_max_duration,
        device=device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    print(f"Loading linear CRF checkpoint: {args.linear_crf_checkpoint}")
    linear_model = _load_ckpt_model(
        args.linear_crf_checkpoint,
        max_duration=1,
        device=device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    _run_uncertainty_analysis(
        semi_model,
        linear_model,
        eval_dataset,
        device,
        out_dir,
        batch_size=args.batch_size,
        num_utterances=args.num_utterances,
        selection=args.selection,
        max_heatmap_labels=args.max_heatmap_labels,
    )
