#!/usr/bin/env python3
"""TIMIT evaluation metrics, duration analysis, and TIMITMetrics dataclass."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .timit_data import NUM_PHONES, PHONES_39, SegmentAnnotation

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Metrics
# =============================================================================


def levenshtein_distance(s1: list, s2: list) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def compute_phone_error_rate(
    predictions: list[list[int]],
    references: list[list[int]],
) -> float:
    """
    Compute Phone Error Rate (PER).

    PER = (substitutions + insertions + deletions) / reference_length
    """
    total_distance = 0
    total_ref_length = 0

    for pred, ref in zip(predictions, references, strict=False):
        # Convert frame-level to segment-level (collapse consecutive)
        pred_segments = collapse_to_segments(pred)
        ref_segments = collapse_to_segments(ref)

        total_distance += levenshtein_distance(pred_segments, ref_segments)
        total_ref_length += len(ref_segments)

    return total_distance / total_ref_length if total_ref_length > 0 else 0


def collapse_to_segments(labels: list[int]) -> list[int]:
    """Collapse frame-level labels to segment-level (remove consecutive duplicates)."""
    if not labels:
        return []

    segments = [labels[0]]
    for label in labels[1:]:
        if label != segments[-1]:
            segments.append(label)

    return segments


def extract_boundaries(labels: list[int]) -> set[int]:
    """Extract boundary positions from label sequence."""
    boundaries = set()
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            boundaries.add(i)
    return boundaries


def compute_boundary_metrics(
    predictions: list[list[int]],
    references: list[list[int]],
    tolerances: list[int] | None = None,
) -> dict[str, float]:
    """Compute boundary detection metrics."""
    if tolerances is None:
        tolerances = [0, 1, 2]
    results = {f"tol_{t}": {"tp": 0, "fp": 0, "fn": 0} for t in tolerances}

    for pred, ref in zip(predictions, references, strict=False):
        pred_bounds = extract_boundaries(pred)
        ref_bounds = extract_boundaries(ref)

        for tol in tolerances:
            key = f"tol_{tol}"
            matched_ref = set()

            for pb in pred_bounds:
                for rb in ref_bounds:
                    if abs(pb - rb) <= tol and rb not in matched_ref:
                        results[key]["tp"] += 1
                        matched_ref.add(rb)
                        break
                else:
                    results[key]["fp"] += 1

            results[key]["fn"] += len(ref_bounds) - len(matched_ref)

    metrics = {}
    for tol in tolerances:
        key = f"tol_{tol}"
        tp = results[key]["tp"]
        fp = results[key]["fp"]
        fn = results[key]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"boundary_f1_tol{tol}"] = f1
        if tol == 0:
            metrics["boundary_precision"] = precision
            metrics["boundary_recall"] = recall

    return metrics


def compute_segment_metrics(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
) -> dict[str, float]:
    """Compute segment-level metrics."""
    tp = 0
    total_pred = 0
    total_true = 0

    for pred_segs, true_segs in zip(pred_segments, true_segments, strict=False):
        pred_set = {(s.start, s.end, s.label) for s in pred_segs}
        true_set = {(s.start, s.end, s.label) for s in true_segs}

        tp += len(pred_set & true_set)
        total_pred += len(pred_set)
        total_true += len(true_set)

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
    }


def compute_duration_stats(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
) -> dict[str, dict[str, float]]:
    """
    Compute per-phoneme duration statistics for predictions vs references.

    Returns dict with:
        - per_phone: {phone_name: {pred_mean, pred_std, ref_mean, ref_std, mae, count}}
        - overall: {mean_absolute_error, correlation}
    """
    pred_durations = defaultdict(list)
    ref_durations = defaultdict(list)

    for pred_segs, ref_segs in zip(pred_segments, true_segments, strict=False):
        for seg in pred_segs:
            pred_durations[seg.label].append(seg.end - seg.start)
        for seg in ref_segs:
            ref_durations[seg.label].append(seg.end - seg.start)

    per_phone = {}
    all_pred_means = []
    all_ref_means = []

    for label in range(NUM_PHONES):
        phone_name = PHONES_39[label]
        pred_d = np.array(pred_durations[label]) if pred_durations[label] else np.array([0])
        ref_d = np.array(ref_durations[label]) if ref_durations[label] else np.array([0])

        pred_mean = float(np.mean(pred_d))
        ref_mean = float(np.mean(ref_d))

        per_phone[phone_name] = {
            "pred_mean": pred_mean,
            "pred_std": float(np.std(pred_d)),
            "ref_mean": ref_mean,
            "ref_std": float(np.std(ref_d)),
            "mae": abs(pred_mean - ref_mean),
            "pred_count": len(pred_durations[label]),
            "ref_count": len(ref_durations[label]),
        }

        if ref_durations[label]:  # Only include phones that appear in reference
            all_pred_means.append(pred_mean)
            all_ref_means.append(ref_mean)

    # Overall correlation between predicted and reference mean durations
    if len(all_pred_means) > 1:
        correlation = float(np.corrcoef(all_pred_means, all_ref_means)[0, 1])
    else:
        correlation = 0.0

    overall_mae = float(
        np.mean([abs(p - r) for p, r in zip(all_pred_means, all_ref_means, strict=True)])
    )

    return {
        "per_phone": per_phone,
        "overall": {
            "mean_absolute_error": overall_mae,
            "duration_correlation": correlation,
        },
    }


def labels_to_segments(labels: list[int]) -> list[SegmentAnnotation]:
    """Convert label sequence to segments."""
    if not labels:
        return []

    segments = []
    current_label = labels[0]
    current_start = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append(SegmentAnnotation(current_start, i, current_label))
            current_label = labels[i]
            current_start = i

    segments.append(SegmentAnnotation(current_start, len(labels), current_label))
    return segments


# =============================================================================
# Enhanced Duration Analysis Functions
# =============================================================================


def load_corpus_duration_stats(data_dir: Path) -> dict:
    """Load raw TIMIT duration statistics from preprocessing.

    Args:
        data_dir: Path to preprocessed TIMIT data directory

    Returns:
        Dictionary with per-phoneme statistics from train_segment_stats.json
    """
    stats_file = data_dir / "train_segment_stats.json"
    if not stats_file.exists():
        logger.warning(f"Corpus stats file not found: {stats_file}")
        return {}

    with open(stats_file) as f:
        return json.load(f)


def compute_kl_divergence(
    pred_durations: list[int], ref_durations: list[int], max_dur: int = 50
) -> float:
    """Compute KL divergence between predicted and reference duration distributions.

    Args:
        pred_durations: List of predicted segment durations
        ref_durations: List of reference segment durations
        max_dur: Maximum duration for histogram binning

    Returns:
        KL divergence D_KL(pred || ref), or 0 if insufficient data
    """
    if len(pred_durations) < 5 or len(ref_durations) < 5:
        return 0.0

    # Bin durations into histogram
    bins = range(1, max_dur + 2)
    pred_hist, _ = np.histogram(pred_durations, bins=bins, density=True)
    ref_hist, _ = np.histogram(ref_durations, bins=bins, density=True)

    # Add smoothing to avoid log(0)
    eps = 1e-10
    pred_hist = pred_hist + eps
    ref_hist = ref_hist + eps

    # Normalize
    pred_hist = pred_hist / pred_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # KL divergence: D_KL(pred || ref) = sum(pred * log(pred / ref))
    return float(np.sum(pred_hist * np.log(pred_hist / ref_hist)))


def compute_js_divergence(
    pred_durations: list[int], ref_durations: list[int], max_dur: int = 50
) -> float:
    """Compute Jensen-Shannon divergence (symmetric alternative to KL).

    Args:
        pred_durations: List of predicted segment durations
        ref_durations: List of reference segment durations
        max_dur: Maximum duration for histogram binning

    Returns:
        JS divergence, or 0 if insufficient data
    """
    if len(pred_durations) < 5 or len(ref_durations) < 5:
        return 0.0

    # Bin durations into histogram
    bins = range(1, max_dur + 2)
    pred_hist, _ = np.histogram(pred_durations, bins=bins, density=True)
    ref_hist, _ = np.histogram(ref_durations, bins=bins, density=True)

    # Add smoothing
    eps = 1e-10
    pred_hist = pred_hist + eps
    ref_hist = ref_hist + eps

    # Normalize
    pred_hist = pred_hist / pred_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # Midpoint distribution
    m = (pred_hist + ref_hist) / 2

    # JS divergence = (KL(pred || m) + KL(ref || m)) / 2
    kl_pm = np.sum(pred_hist * np.log(pred_hist / m))
    kl_rm = np.sum(ref_hist * np.log(ref_hist / m))

    return float((kl_pm + kl_rm) / 2)


def compute_enhanced_duration_stats(
    pred_segments: list[list[SegmentAnnotation]],
    true_segments: list[list[SegmentAnnotation]],
    max_dur: int = 50,
) -> dict:
    """Compute enhanced duration statistics including KL divergence.

    Returns dict with:
        - per_phone: {phone_name: {..., kl_div, js_div}}
        - overall: {..., weighted_kl, weighted_js}
    """
    pred_durations = defaultdict(list)
    ref_durations = defaultdict(list)

    for pred_segs, ref_segs in zip(pred_segments, true_segments, strict=False):
        for seg in pred_segs:
            pred_durations[seg.label].append(seg.end - seg.start)
        for seg in ref_segs:
            ref_durations[seg.label].append(seg.end - seg.start)

    per_phone = {}
    all_pred_means = []
    all_ref_means = []
    weighted_kl = 0.0
    weighted_js = 0.0
    total_ref_count = 0

    for label in range(NUM_PHONES):
        phone_name = PHONES_39[label]
        pred_d = list(pred_durations[label]) if pred_durations[label] else [0]
        ref_d = list(ref_durations[label]) if ref_durations[label] else [0]

        pred_mean = float(np.mean(pred_d))
        ref_mean = float(np.mean(ref_d))

        # Compute KL and JS divergence
        kl_div = compute_kl_divergence(pred_d, ref_d, max_dur)
        js_div = compute_js_divergence(pred_d, ref_d, max_dur)

        per_phone[phone_name] = {
            "pred_mean": pred_mean,
            "pred_std": float(np.std(pred_d)),
            "ref_mean": ref_mean,
            "ref_std": float(np.std(ref_d)),
            "mae": abs(pred_mean - ref_mean),
            "pred_count": len(pred_durations[label]),
            "ref_count": len(ref_durations[label]),
            "kl_divergence": kl_div,
            "js_divergence": js_div,
        }

        if ref_durations[label]:
            all_pred_means.append(pred_mean)
            all_ref_means.append(ref_mean)
            # Weighted by reference count
            weighted_kl += kl_div * len(ref_durations[label])
            weighted_js += js_div * len(ref_durations[label])
            total_ref_count += len(ref_durations[label])

    # Overall correlation
    if len(all_pred_means) > 1:
        correlation = float(np.corrcoef(all_pred_means, all_ref_means)[0, 1])
    else:
        correlation = 0.0

    overall_mae = float(
        np.mean([abs(p - r) for p, r in zip(all_pred_means, all_ref_means, strict=True)])
    )

    return {
        "per_phone": per_phone,
        "overall": {
            "mean_absolute_error": overall_mae,
            "duration_correlation": correlation,
            "weighted_kl_divergence": weighted_kl / total_ref_count if total_ref_count > 0 else 0,
            "weighted_js_divergence": weighted_js / total_ref_count if total_ref_count > 0 else 0,
        },
    }


def export_duration_analysis(results: dict, corpus_stats: dict, output_path: Path):
    """Export detailed duration analysis to JSON and CSV.

    Args:
        results: Dictionary of model results (each with TIMITMetrics)
        corpus_stats: Raw TIMIT corpus statistics from preprocessing
        output_path: Base path for output files (will create .json and .csv)
    """
    output = {
        "corpus_stats": corpus_stats,
        "model_comparisons": {},
    }

    for model_name, metrics in results.items():
        if hasattr(metrics, "duration_stats") and metrics.duration_stats:
            output["model_comparisons"][model_name] = {
                "per_phone": metrics.duration_stats["per_phone"],
                "overall": metrics.duration_stats["overall"],
            }

    # JSON export
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Duration analysis JSON saved to {json_path}")

    # CSV export (flattened per-phoneme table)
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phone",
                "corpus_mean",
                "corpus_std",
                "corpus_p95",
                "model",
                "pred_mean",
                "pred_std",
                "mae",
                "kl_div",
            ]
        )

        for phone in PHONES_39:
            corpus = corpus_stats.get(phone, {})
            corpus_mean = corpus.get("mean", 0)
            corpus_std = corpus.get("std", 0)
            corpus_p95 = corpus.get("p95", 0)

            for model_name, metrics in results.items():
                if hasattr(metrics, "duration_stats") and metrics.duration_stats:
                    phone_stats = metrics.duration_stats["per_phone"].get(phone, {})
                    writer.writerow(
                        [
                            phone,
                            f"{corpus_mean:.2f}",
                            f"{corpus_std:.2f}",
                            f"{corpus_p95:.2f}",
                            model_name,
                            f"{phone_stats.get('pred_mean', 0):.2f}",
                            f"{phone_stats.get('pred_std', 0):.2f}",
                            f"{phone_stats.get('mae', 0):.2f}",
                            f"{phone_stats.get('kl_divergence', 0):.4f}",
                        ]
                    )

    logger.info(f"Duration analysis CSV saved to {csv_path}")


def plot_duration_distributions(
    results: dict, corpus_stats: dict, output_dir: Path, max_dur: int = 50
):
    """Generate per-phoneme duration distribution plots.

    Args:
        results: Dictionary of model results
        corpus_stats: Raw TIMIT corpus statistics
        output_dir: Directory for output plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping duration plots")
        return

    # Select interesting phonemes (vowels, stops, fricatives)
    phones_to_plot = ["aa", "iy", "p", "t", "s", "sil"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, phone in zip(axes.flat, phones_to_plot, strict=False):
        # Plot corpus reference if available
        corpus = corpus_stats.get(phone, {})
        if corpus:
            ax.axvline(
                corpus.get("mean", 0),
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"TIMIT (μ={corpus.get('mean', 0):.1f})",
            )

        # Plot each model's predicted distribution
        colors = ["blue", "green", "red", "orange"]
        for (model_name, metrics), color in zip(results.items(), colors, strict=False):
            if hasattr(metrics, "duration_stats") and metrics.duration_stats:
                phone_stats = metrics.duration_stats["per_phone"].get(phone, {})
                pred_mean = phone_stats.get("pred_mean", 0)
                pred_std = phone_stats.get("pred_std", 1)

                # Draw normal approximation
                x = np.linspace(max(0, pred_mean - 3 * pred_std), pred_mean + 3 * pred_std, 100)
                y = np.exp(-0.5 * ((x - pred_mean) / pred_std) ** 2) / (
                    pred_std * np.sqrt(2 * np.pi)
                )
                ax.plot(x, y, color=color, label=f"{model_name} (μ={pred_mean:.1f})")

        ax.set_title(f"/{phone}/")
        ax.set_xlabel("Duration (frames)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, max_dur)

    plt.suptitle("Duration Distributions: TIMIT Corpus vs Model Predictions")
    plt.tight_layout()

    output_path = output_dir / "duration_distributions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Duration plot saved to {output_path}")


# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class TIMITMetrics:
    """Metrics for TIMIT evaluation."""

    phone_error_rate: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_f1_tolerances: dict[int, float]
    segment_precision: float
    segment_recall: float
    segment_f1: float
    # Duration analysis
    duration_stats: dict | None = None
    # Timing metrics (optional, set during training)
    training_time_per_epoch: float = 0.0  # seconds
    total_training_time: float = 0.0  # seconds
    inference_time: float = 0.0  # seconds for full test set
    # Throughput metrics (for scaling analysis)
    throughput_utterances_per_sec: float = 0.0  # avg utterances/sec per epoch
    throughput_utterances_per_sec_std: float = 0.0  # std dev across epochs
    throughput_frames_per_sec: float = 0.0  # avg frames/sec per epoch
    throughput_frames_per_sec_std: float = 0.0  # std dev across epochs
    num_train_utterances: int = 0  # dataset size for context
    batch_size: int = 0  # batch size for context

    def to_dict(self) -> dict:
        """Convert metrics to JSON-serializable dict."""
        result = {
            "phone_error_rate": self.phone_error_rate,
            "boundary_precision": self.boundary_precision,
            "boundary_recall": self.boundary_recall,
            "boundary_f1": self.boundary_f1,
            "boundary_f1_tolerances": {str(k): v for k, v in self.boundary_f1_tolerances.items()},
            "segment_precision": self.segment_precision,
            "segment_recall": self.segment_recall,
            "segment_f1": self.segment_f1,
            "training_time_per_epoch": self.training_time_per_epoch,
            "total_training_time": self.total_training_time,
            "inference_time": self.inference_time,
            "throughput_utterances_per_sec": self.throughput_utterances_per_sec,
            "throughput_utterances_per_sec_std": self.throughput_utterances_per_sec_std,
            "throughput_frames_per_sec": self.throughput_frames_per_sec,
            "throughput_frames_per_sec_std": self.throughput_frames_per_sec_std,
            "num_train_utterances": self.num_train_utterances,
            "batch_size": self.batch_size,
        }
        if self.duration_stats:
            result["duration_stats"] = self.duration_stats
        return result
