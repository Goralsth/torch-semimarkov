#!/usr/bin/env python3
"""TIMIT training loop: train_epoch, evaluate, train_model."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .timit_data import TIMITDataset, make_collate_fn
from .timit_metrics import (
    TIMITMetrics,
    compute_boundary_metrics,
    compute_duration_stats,
    compute_phone_error_rate,
    compute_segment_metrics,
    labels_to_segments,
)
from .timit_models import (
    HAS_TORCHCRF,
    BiLSTMEncoder,
    TIMITModel,
    TIMITModelPytorchCRF,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Training Loop
# =============================================================================


def train_epoch(
    model: TIMITModel | TIMITModelPytorchCRF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    backend: str = "streaming",
    use_triton: bool = True,
    crf_reg: float = 0.0,
    profile: bool = False,
) -> tuple[float, float, int, int, dict[str, float] | None]:
    """Train for one epoch.

    Args:
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only).
            Helps prevent gradient explosion from unbounded transition/duration_bias.
        profile: If True, insert torch.cuda.synchronize() barriers to measure
            per-phase wall-clock time. This prevents async overlap and will slow
            training, but provides accurate timing breakdown.

    Returns:
        Tuple of (average_loss, elapsed_time_seconds, num_utterances, num_frames, phase_times).
        phase_times is None when profile=False, otherwise a dict mapping phase names
        to cumulative seconds across all batches.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    total_utterances = 0
    total_frames = 0

    phase_times = None
    if profile:
        phase_times = defaultdict(float)

    start_time = time.perf_counter()

    for batch in dataloader:
        if profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # Track throughput metrics
        total_utterances += features.shape[0]
        total_frames += lengths.sum().item()

        if profile:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            phase_times["data_transfer"] += t1 - t0

        optimizer.zero_grad()

        if profile and isinstance(model, TIMITModel):
            # Profile each phase of compute_loss separately
            hidden = model.encoder(features)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            phase_times["encoder"] += t2 - t1

            result = model.crf(hidden, lengths, use_triton=use_triton, backend=backend)
            partition = result["partition"]
            cum_scores = result["cum_scores"]
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            phase_times["crf_forward"] += t3 - t2

            gold_score = model.crf._score_gold(
                cum_scores,
                labels,
                lengths,
                result.get("proj_start"),
                result.get("proj_end"),
            )
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            phase_times["gold_scoring"] += t4 - t3

            loss = (partition - gold_score).mean()
            if crf_reg > 0:
                loss = loss + crf_reg * model.crf.parameter_penalty()

            loss.backward()
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            phase_times["backward"] += t5 - t4

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            phase_times["optimizer"] += t6 - t5
        else:
            # Normal (non-profiled) path
            loss = model.compute_loss(
                features, lengths, labels, backend=backend, use_triton=use_triton
            )

            if crf_reg > 0 and isinstance(model, TIMITModel):
                loss = loss + crf_reg * model.crf.parameter_penalty()

            loss.backward()

            if profile:
                # For pytorch-crf models, lump everything into one phase
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                phase_times["forward_backward"] += t2 - t1

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if profile:
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                phase_times["optimizer"] += t3 - t2

        total_loss += loss.item()
        num_batches += 1

    elapsed = time.perf_counter() - start_time
    return total_loss / num_batches, elapsed, total_utterances, total_frames, phase_times


@torch.no_grad()
def evaluate(
    model: TIMITModel | TIMITModelPytorchCRF,
    dataloader: DataLoader,
    device: torch.device,
    backend: str = "streaming",
) -> tuple[TIMITMetrics, float]:
    """Evaluate model.

    Args:
        backend: Backend for decode (only used for TIMITModel, ignored for pytorch-crf).

    Returns:
        Tuple of (metrics, elapsed_time_seconds).
    """
    model.eval()

    all_predictions = []
    all_references = []
    all_pred_segments = []
    all_true_segments = []

    # Check if this is a pytorch-crf model (returns list) vs TIMITModel (returns ViterbiResult)
    is_pytorch_crf = isinstance(model, TIMITModelPytorchCRF)

    start_time = time.perf_counter()

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        # pytorch-crf doesn't support backend parameter
        if is_pytorch_crf:
            result = model.decode(features, lengths)
        else:
            result = model.decode(features, lengths, backend=backend)

        for i in range(len(lengths)):
            seq_len = lengths[i].item()

            if is_pytorch_crf:
                # pytorch-crf returns list of label sequences directly
                pred_labels = result[i][:seq_len]
            else:
                # TIMITModel returns ViterbiResult with segments
                # NOTE: flash_semicrf.Segment uses INCLUSIVE end (end=5 means position 5 included)
                # Convert to exclusive for iteration: range(start, end+1)
                pred_labels = [0] * seq_len
                for seg in result.segments[i]:
                    for j in range(seg.start, min(seg.end + 1, seq_len)):
                        pred_labels[j] = seg.label

            ref_labels = labels[i, :seq_len].cpu().tolist()

            all_predictions.append(pred_labels)
            all_references.append(ref_labels)

            # Both paths use labels_to_segments for consistent segment merging
            # This ensures consecutive frames with the same label are merged into single segments
            # (critical for K=1 semimarkov which returns per-frame segments from Viterbi)
            pred_segs = labels_to_segments(pred_labels)
            true_segs = labels_to_segments(ref_labels)

            all_pred_segments.append(pred_segs)
            all_true_segments.append(true_segs)

    elapsed = time.perf_counter() - start_time

    per = compute_phone_error_rate(all_predictions, all_references)
    boundary_metrics = compute_boundary_metrics(all_predictions, all_references)
    segment_metrics = compute_segment_metrics(all_pred_segments, all_true_segments)
    duration_stats = compute_duration_stats(all_pred_segments, all_true_segments)

    metrics = TIMITMetrics(
        phone_error_rate=per,
        boundary_precision=boundary_metrics["boundary_precision"],
        boundary_recall=boundary_metrics["boundary_recall"],
        boundary_f1=boundary_metrics.get("boundary_f1_tol0", 0),
        boundary_f1_tolerances={
            int(k.split("tol")[1]): v
            for k, v in boundary_metrics.items()
            if k.startswith("boundary_f1_tol")
        },
        segment_precision=segment_metrics["segment_precision"],
        segment_recall=segment_metrics["segment_recall"],
        segment_f1=segment_metrics["segment_f1"],
        duration_stats=duration_stats,
    )
    return metrics, elapsed


def train_model(
    data_dir: Path,
    model_type: Literal["pytorch-crf", "linear", "semicrf"] = "semicrf",
    max_duration: int = 30,
    hidden_dim: int = 256,
    num_layers: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    backend: str = "streaming",
    use_triton: bool = True,
    log_every: int = 1,
    crf_reg: float = 0.0,
    fixed_length: int | None = None,
    profile: bool = False,
    precision: str = "float32",
    use_sequence_boundaries: bool = False,
) -> tuple[TIMITModel | TIMITModelPytorchCRF, TIMITMetrics]:
    """Train a model and return it with metrics.

    Args:
        crf_reg: L2 regularization coefficient for CRF parameters (Semi-Markov only).
        fixed_length: If provided, force all sequences to this length (for debugging).
        profile: If True, log per-phase GPU timing breakdown each epoch.
        use_sequence_boundaries: If True, add scalar start/end transition vectors
            (pytorch-crf equivalent). Only applies to flash-semicrf models, not pytorch-crf.
    """
    device = torch.device(device)

    # Load data with normalization
    # Training dataset computes normalization stats from its own data
    train_dataset = TIMITDataset(data_dir / "train.jsonl", normalize=True)

    # Test dataset uses training stats to ensure consistent normalization
    test_dataset = TIMITDataset(
        data_dir / "test.jsonl",
        normalize=True,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    # Determine feature dimension from first sample
    sample = train_dataset[0]
    input_dim = sample["features"].shape[-1]

    # Create collate function (with optional fixed length for debugging)
    collate_fn = make_collate_fn(fixed_length=fixed_length)
    if fixed_length is not None:
        logger.info(f"Using fixed sequence length: {fixed_length}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Build model
    encoder = BiLSTMEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    if model_type == "pytorch-crf":
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model type. " "Install with: pip install pytorch-crf"
            )
        model = TIMITModelPytorchCRF(
            encoder=encoder,
            hidden_dim=hidden_dim,
        ).to(device)
        k = 1  # For logging purposes
    elif model_type == "linear":
        k = 1
        model = TIMITModel(
            encoder=encoder,
            max_duration=k,
            hidden_dim=hidden_dim,
            precision=precision,
            use_sequence_boundaries=use_sequence_boundaries,
        ).to(device)
    else:  # semicrf
        k = max_duration
        model = TIMITModel(
            encoder=encoder,
            max_duration=k,
            hidden_dim=hidden_dim,
            precision=precision,
            use_sequence_boundaries=use_sequence_boundaries,
        ).to(device)

    seq_bounds_str = ", seq_boundaries=True" if use_sequence_boundaries else ""
    logger.info(
        f"Model: {model_type}, K={k}{seq_bounds_str}, params={sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_per = float("inf")
    best_metrics = None
    total_train_time = 0.0
    epoch_times = []
    epoch_utterances = []
    epoch_frames = []
    epoch_utt_rates = []  # utterances/sec per epoch
    epoch_frame_rates = []  # frames/sec per epoch

    # Accumulate per-phase timing across non-warmup epochs (skip epoch 0 for torch.compile)
    agg_phase_times: dict[str, float] = defaultdict(float)
    agg_phase_epochs = 0

    for epoch in range(epochs):
        train_loss, epoch_time, num_utterances, num_frames, phase_times = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            backend=backend,
            use_triton=use_triton,
            crf_reg=crf_reg,
            profile=profile,
        )
        total_train_time += epoch_time
        epoch_times.append(epoch_time)
        epoch_utterances.append(num_utterances)
        epoch_frames.append(num_frames)
        # Track per-epoch throughput rates
        epoch_utt_rates.append(num_utterances / epoch_time if epoch_time > 0 else 0)
        epoch_frame_rates.append(num_frames / epoch_time if epoch_time > 0 else 0)
        scheduler.step()

        # Log per-phase timing breakdown when profiling
        if profile and phase_times:
            num_batches = len(train_loader)
            total_phase = sum(phase_times.values())

            # Skip epoch 0 in aggregate (torch.compile + Triton JIT warmup)
            if epoch > 0:
                agg_phase_epochs += 1
                for phase, secs in phase_times.items():
                    agg_phase_times[phase] += secs

            logger.info(
                f"Epoch {epoch+1} phase breakdown ({num_batches} batches, {total_phase:.1f}s total):"
            )
            for phase, secs in phase_times.items():
                pct = 100.0 * secs / total_phase if total_phase > 0 else 0
                per_batch_ms = 1000.0 * secs / num_batches
                logger.info(
                    f"  {phase:<20s} {secs:>7.1f}s ({pct:>5.1f}%)  {per_batch_ms:>7.1f}ms/batch"
                )

        # Log CRF parameter magnitudes for debugging gradient explosion
        if isinstance(model, TIMITModel):
            trans_max = model.crf.transition.abs().max().item()
            dur_max = model.crf.duration_bias.abs().max().item()
            logger.debug(
                f"Epoch {epoch+1} CRF params: "
                f"transition_max={trans_max:.4f}, duration_bias_max={dur_max:.4f}"
            )
            # Warn if parameters are drifting to extreme values
            if trans_max > 20 or dur_max > 20:
                logger.warning(
                    f"Epoch {epoch+1}: CRF parameters drifting high! "
                    f"trans_max={trans_max:.2f}, dur_max={dur_max:.2f}. "
                    f"Consider increasing --crf-reg."
                )

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            test_metrics, inference_time = evaluate(model, test_loader, device, backend=backend)

            # Calculate throughput for this epoch
            utt_per_sec = num_utterances / epoch_time if epoch_time > 0 else 0
            frames_per_sec = num_frames / epoch_time if epoch_time > 0 else 0

            logger.info(
                f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                f"PER: {test_metrics.phone_error_rate:.4f} | "
                f"Boundary F1: {test_metrics.boundary_f1:.4f} | "
                f"Segment F1: {test_metrics.segment_f1:.4f} | "
                f"Train: {epoch_time:.1f}s ({utt_per_sec:.1f} utt/s, {frames_per_sec/1000:.1f}k fr/s) | "
                f"Infer: {inference_time:.1f}s"
            )

            if test_metrics.phone_error_rate < best_per:
                best_per = test_metrics.phone_error_rate
                # Update metrics with timing info
                test_metrics.total_training_time = total_train_time
                test_metrics.training_time_per_epoch = sum(epoch_times) / len(epoch_times)
                test_metrics.inference_time = inference_time
                # Update metrics with throughput info (mean and std across epochs)
                utt_rates = np.array(epoch_utt_rates)
                frame_rates = np.array(epoch_frame_rates)
                test_metrics.throughput_utterances_per_sec = float(np.mean(utt_rates))
                test_metrics.throughput_utterances_per_sec_std = float(np.std(utt_rates))
                test_metrics.throughput_frames_per_sec = float(np.mean(frame_rates))
                test_metrics.throughput_frames_per_sec_std = float(np.std(frame_rates))
                test_metrics.num_train_utterances = len(train_dataset)
                test_metrics.batch_size = batch_size
                best_metrics = test_metrics

    # Print aggregate phase breakdown (excluding warmup epoch 0)
    if profile and agg_phase_epochs > 0:
        total_agg = sum(agg_phase_times.values())
        num_batches = len(train_loader)
        logger.info("")
        logger.info(f"AGGREGATE PHASE BREAKDOWN (epochs 2-{epochs}, {agg_phase_epochs} epochs):")
        for phase, secs in agg_phase_times.items():
            avg = secs / agg_phase_epochs
            pct = 100.0 * secs / total_agg if total_agg > 0 else 0
            per_batch_ms = 1000.0 * avg / num_batches
            logger.info(
                f"  {phase:<20s} {avg:>7.1f}s/ep ({pct:>5.1f}%)  {per_batch_ms:>7.1f}ms/batch"
            )

    return model, best_metrics
