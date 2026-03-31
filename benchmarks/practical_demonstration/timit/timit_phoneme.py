#!/usr/bin/env python3
"""
TIMIT Phoneme Segmentation Benchmark

This is the classic benchmark for demonstrating Semi-CRF advantages over linear CRFs.
TIMIT has been used since the original Semi-CRF paper (Sarawagi & Cohen, 2004) and
provides a well-studied setting with published baselines.

Three-Way Model Comparison:
    This benchmark supports comparing three CRF implementations:

    1. **pytorch-crf** (optional): External linear CRF library baseline
    2. **flash-semicrf K=1**: Linear CRF via Triton streaming kernel
    3. **flash-semicrf K>1**: Full semi-CRF with duration modeling

    The comparison validates that:
    - K=1 Triton matches pytorch-crf accuracy (correctness)
    - K=1 Triton is faster than pytorch-crf (performance)
    - Semi-CRF improves on linear CRF (duration modeling value)

Why Semi-CRFs help on TIMIT:
    - Phonemes have characteristic durations (vowels longer than stops)
    - Duration is linguistically meaningful and predictable
    - A linear CRF cannot encode "this phoneme typically lasts 50-100ms"
    - A semi-CRF learns duration priors per phoneme class

Dataset:
    - 630 speakers (462 train, 168 test)
    - ~6300 utterances total
    - 61 phoneme classes (typically collapsed to 39)
    - Standard train/test split defined by NIST
    - Requires LDC license (widely available in practice)

Features:
    - 13 MFCCs + delta + delta-delta = 39 features
    - 10ms frame shift (100 Hz)
    - Alternative: 80-dim log mel filterbanks

Metrics:
    - Phone Error Rate (PER): Levenshtein distance / reference phones
    - Boundary F1: Exact match and within-tolerance
    - Segment F1: Full segment match (start, end, label)
    - Training/inference timing

Historical context:
    - Sarawagi & Cohen (2004): Semi-CRF improved ~1-2% over linear CRF
    - Modern encoders (BiLSTM, Transformer) have pushed overall PER down
    - But the relative advantage of duration modeling should persist

Requirements:
    pip install torchaudio librosa soundfile

    Optional (for three-way comparison with external baseline):
    pip install pytorch-crf

Note on data access:
    TIMIT requires a license from LDC (Linguistic Data Consortium).
    This code assumes the standard TIMIT directory structure:

    TIMIT/
    ├── TRAIN/
    │   ├── DR1/
    │   │   ├── FCJF0/
    │   │   │   ├── SA1.WAV
    │   │   │   ├── SA1.PHN
    │   │   │   ├── SA1.TXT
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── TEST/
        └── ...

Usage:
    # Preprocess TIMIT
    python timit_phoneme.py preprocess \
        --timit-dir /path/to/TIMIT \
        --output-dir data/timit_benchmark/

    # Train a specific model type
    python timit_phoneme.py train \
        --data-dir data/timit_benchmark/ \
        --model semicrf \
        --max-duration 30

    # Model types: pytorch-crf, linear (K=1 Triton), semicrf (K>1)
    python timit_phoneme.py train --model pytorch-crf ...
    python timit_phoneme.py train --model linear ...
    python timit_phoneme.py train --model semicrf ...

    # Three-way comparison (or two-way if pytorch-crf not installed)
    python timit_phoneme.py compare \
        --data-dir data/timit_benchmark/ \
        --output-json results/timit_comparison.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support `python3 timit_phoneme.py` invocation in addition to
# `python3 -m timit.timit_phoneme` from the parent directory.
_HERE = Path(__file__).resolve().parent
_PARENT = str(_HERE.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

from timit.timit_analysis import compare_models, run_analyze_uncertainty  # noqa: E402
from timit.timit_data import MAX_HEATMAP_LABELS, NUM_PHONES, preprocess_timit  # noqa: E402
from timit.timit_lightning import HAS_LIGHTNING  # noqa: E402
from timit.timit_metrics import (  # noqa: E402
    export_duration_analysis,
    load_corpus_duration_stats,
    plot_duration_distributions,
)
from timit.timit_models import BiLSTMEncoder  # noqa: E402
from timit.timit_training import train_model  # noqa: E402

if HAS_LIGHTNING:
    from timit.timit_lightning import TIMITDataModule, TIMITLightningModule

    try:
        import lightning.pytorch as L
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:
        import pytorch_lightning as L  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess TIMIT")
    preprocess_parser.add_argument(
        "--timit-dir", type=Path, required=True, help="TIMIT root directory"
    )
    preprocess_parser.add_argument("--output-dir", type=Path, required=True)
    preprocess_parser.add_argument("--feature-type", choices=["mfcc", "mel"], default="mfcc")
    preprocess_parser.add_argument("--n-mfcc", type=int, default=13)
    preprocess_parser.add_argument("--n-mels", type=int, default=80)

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument(
        "--model",
        choices=["pytorch-crf", "linear", "semicrf"],
        default="semicrf",
        help="Model type: pytorch-crf (external lib), linear (K=1 Triton), semicrf (K>1)",
    )
    train_parser.add_argument("--max-duration", type=int, default=30)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--num-layers", type=int, default=3)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument(
        "--crf-reg",
        type=float,
        default=0.0,
        help="L2 regularization coefficient for CRF parameters (transition, duration_bias). "
        "Helps prevent gradient explosion in Semi-Markov CRF training. Typical values: 0.001-0.1",
    )
    train_parser.add_argument(
        "--backend",
        choices=["streaming", "binary_tree_sharded", "exact", "auto"],
        default="streaming",
        help="CRF backend: streaming (Triton), binary_tree_sharded (sharded matmuls), "
        "exact (edge tensor), auto (heuristic)",
    )
    train_parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Disable Triton kernels (use PyTorch reference implementation)",
    )
    train_parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log metrics every N epochs (default: 1)",
    )
    train_parser.add_argument(
        "--fixed-length",
        type=int,
        default=None,
        help="Force all sequences to this fixed length (for debugging boundary handling). "
        "Sequences shorter are padded, longer are truncated.",
    )
    train_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-phase GPU timing breakdown. Inserts cuda.synchronize() barriers "
        "between phases (data transfer, encoder, CRF forward, gold scoring, backward, "
        "optimizer) to measure wall-clock time accurately. This prevents async overlap "
        "and will slow training — use for diagnostics only.",
    )
    train_parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float32",
        help="Kernel compute precision. float32 is ~2-4x faster on L40S with negligible "
        "accuracy loss at TIMIT scale (T<800). Use float64 for T>10K or exact marginals.",
    )
    train_parser.add_argument(
        "--sequence-boundaries",
        action="store_true",
        help="Add learnable start/end transition vectors (2C parameters). "
        "Equivalent to pytorch-crf's start_transitions/end_transitions. "
        "Only applies to flash-semicrf models (linear, semicrf), not pytorch-crf.",
    )

    # Compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare CRF models (pytorch-crf if installed, K=1 Triton, Semi-CRF)",
    )
    compare_parser.add_argument("--data-dir", type=Path, required=True)
    compare_parser.add_argument("--max-duration", type=int, default=30)
    compare_parser.add_argument("--hidden-dim", type=int, default=256)
    compare_parser.add_argument("--num-layers", type=int, default=3)
    compare_parser.add_argument("--epochs", type=int, default=50)
    compare_parser.add_argument("--batch-size", type=int, default=32)
    compare_parser.add_argument(
        "--output-json", type=Path, default=None, help="Save results to JSON file"
    )
    compare_parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log metrics every N epochs (default: 1)",
    )
    compare_parser.add_argument(
        "--export-duration",
        type=Path,
        default=None,
        help="Export duration analysis to this path (creates .json and .csv files)",
    )
    compare_parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for duration distribution plots (requires matplotlib)",
    )
    compare_parser.add_argument(
        "--dp-standard-ref",
        action="store_true",
        default=False,
        help="Also train the K>1 DP-standard reference (torch-struct linear scan) "
        "to compare algorithms and validate Triton correctness.",
    )
    compare_parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float32",
        help="Kernel compute precision. float32 is ~2-4x faster on L40S with negligible "
        "accuracy loss at TIMIT scale (T<800). Use float64 for T>10K or exact marginals.",
    )
    compare_parser.add_argument(
        "--sequence-boundaries",
        action="store_true",
        help="Add learnable start/end transition vectors (2C parameters) to "
        "flash-semicrf models. Equivalent to pytorch-crf's start_transitions/"
        "end_transitions. Enables controlled comparison of the parameterization gap.",
    )
    compare_parser.add_argument(
        "--uncertainty-analysis",
        type=Path,
        default=None,
        help="Run uncertainty analysis (semi-CRF vs linear CRF calibration) after "
        "comparison and write output PNGs to this directory. Uses trained models "
        "in-memory — no Lightning checkpoints required.",
    )
    compare_parser.add_argument(
        "--uncertainty-num-utterances",
        type=int,
        default=8,
        help="Number of utterances to visualize in uncertainty analysis (default: 8).",
    )
    compare_parser.add_argument(
        "--uncertainty-selection",
        choices=["confident-error", "entropy", "semi-advantage", "random"],
        default="confident-error",
        help="Utterance selection strategy for uncertainty analysis: "
        "confident-error (rank-product of linear PER and semi entropy), "
        "entropy (highest semi entropy), "
        "semi-advantage (largest linear_per − semi_per), "
        "random (seeded shuffle).",
    )

    # Train with Lightning (optional — requires flash-semicrf[lightning])
    if HAS_LIGHTNING:
        lightning_parser = subparsers.add_parser(
            "train-lightning",
            help="Train with PyTorch Lightning (supports multi-GPU DDP). "
            "Requires: pip install flash-semicrf[lightning]",
        )
        lightning_parser.add_argument("--data-dir", type=Path, required=True)
        lightning_parser.add_argument(
            "--model",
            choices=["linear", "semicrf"],
            default="semicrf",
            help="Model type: linear (K=1 semi-CRF), semicrf (K>1 semi-CRF)",
        )
        lightning_parser.add_argument("--max-duration", type=int, default=30)
        lightning_parser.add_argument("--hidden-dim", type=int, default=256)
        lightning_parser.add_argument("--num-layers", type=int, default=3)
        lightning_parser.add_argument("--epochs", type=int, default=50)
        lightning_parser.add_argument("--batch-size", type=int, default=32)
        lightning_parser.add_argument("--lr", type=float, default=1e-3)
        lightning_parser.add_argument(
            "--crf-lr-scale",
            type=float,
            default=0.1,
            help="LR multiplier for CRF structural params (transition, duration). "
            "Projection layers use the full --lr.",
        )
        lightning_parser.add_argument(
            "--crf-reg",
            type=float,
            default=0.0,
            help="L2 regularization weight for CRF structural params (penalty_weight). "
            "Typical values: 0.001–0.1.",
        )
        lightning_parser.add_argument(
            "--devices",
            default="-1",
            help="GPUs to use: -1 for all visible (default), count (e.g. 2), "
            "or specific indices (e.g. 2,3). "
            "Set CUDA_VISIBLE_DEVICES=2,3 to restrict which GPUs are visible.",
        )
        lightning_parser.add_argument(
            "--accelerator", type=str, default="auto", help="gpu, cpu, or auto"
        )
        lightning_parser.add_argument(
            "--strategy",
            type=str,
            default="auto",
            help="Training strategy: ddp, auto. "
            "Use ddp with gradient_checkpointing=True and find_unused_parameters=False.",
        )
        lightning_parser.add_argument(
            "--num-workers", type=int, default=0, help="DataLoader worker processes"
        )
        lightning_parser.add_argument(
            "--num-nodes", type=int, default=1, help="Number of nodes for multi-node DDP."
        )
        lightning_parser.add_argument(
            "--max-length",
            type=int,
            default=None,
            help="Truncate sequences longer than this many frames.",
        )
        lightning_parser.add_argument(
            "--precision",
            choices=["float32", "float64"],
            default="float32",
            help="Kernel compute precision. float32 is ~2-4x faster on L40S with negligible "
            "accuracy loss at TIMIT scale (T<800). Use float64 for T>10K or exact marginals.",
        )
        lightning_parser.add_argument(
            "--sequence-boundaries",
            action="store_true",
            help="Add learnable start/end transition vectors (2C parameters). "
            "Equivalent to pytorch-crf's start_transitions/end_transitions.",
        )

        # analyze-uncertainty subcommand
        ua_parser = subparsers.add_parser(
            "analyze-uncertainty",
            help="Compare semi-CRF vs linear CRF calibration / uncertainty. "
            "Requires two trained Lightning checkpoints.",
        )
        ua_parser.add_argument(
            "--semi-crf-checkpoint",
            type=Path,
            required=True,
            help="Path to trained semi-CRF Lightning .ckpt (max_duration > 1).",
        )
        ua_parser.add_argument(
            "--linear-crf-checkpoint",
            type=Path,
            required=True,
            help="Path to trained linear CRF Lightning .ckpt (max_duration=1).",
        )
        ua_parser.add_argument("--data-dir", type=Path, required=True)
        ua_parser.add_argument(
            "--split",
            choices=["val", "test"],
            default="val",
            help="Dataset split to evaluate (both map to test.jsonl in TIMIT).",
        )
        ua_parser.add_argument(
            "--num-utterances",
            type=int,
            default=8,
            help="Number of utterances to visualize.",
        )
        ua_parser.add_argument(
            "--selection",
            choices=["confident-error", "entropy", "semi-advantage", "random"],
            default="confident-error",
            help="Utterance selection strategy: "
            "confident-error (rank-product of linear PER and semi entropy), "
            "entropy (highest semi entropy), "
            "semi-advantage (largest linear_per − semi_per), "
            "random (seeded shuffle).",
        )
        ua_parser.add_argument(
            "--max-heatmap-labels",
            type=int,
            default=MAX_HEATMAP_LABELS,
            help=f"Max phone classes shown in posterior heatmap (default: {MAX_HEATMAP_LABELS}).",
        )
        ua_parser.add_argument(
            "--semi-max-duration",
            type=int,
            default=30,
            help="max_duration used when training the semi-CRF (must match checkpoint).",
        )
        ua_parser.add_argument(
            "--hidden-dim",
            type=int,
            default=256,
            help="Encoder hidden dim (must match checkpoints).",
        )
        ua_parser.add_argument(
            "--num-layers",
            type=int,
            default=3,
            help="Encoder LSTM layers (must match checkpoints).",
        )
        ua_parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Inference batch size.",
        )
        ua_parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("./uncertainty_analysis"),
            help="Directory for output PNGs.",
        )
        ua_parser.add_argument(
            "--device",
            type=str,
            default="cuda" if __import__("torch").cuda.is_available() else "cpu",
            help="Inference device: cuda or cpu.",
        )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_timit(
            args.timit_dir,
            args.output_dir,
            feature_type=args.feature_type,
            n_mfcc=args.n_mfcc,
            n_mels=args.n_mels,
        )
    elif args.command == "train":
        _model, metrics = train_model(
            args.data_dir,
            model_type=args.model,
            max_duration=args.max_duration,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            backend=args.backend,
            use_triton=not args.no_triton,
            log_every=args.log_every,
            crf_reg=args.crf_reg,
            fixed_length=args.fixed_length,
            profile=args.profile,
            precision=args.precision,
            use_sequence_boundaries=args.sequence_boundaries,
        )
        # Print training summary with throughput
        k = 1 if args.model in ("pytorch-crf", "linear") else args.max_duration
        triton_str = "Triton" if not args.no_triton else "PyTorch"
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        seq_bounds_str = " +seq_boundaries" if args.sequence_boundaries else ""
        print(f"  Model: {args.model} (K={k}, {triton_str}{seq_bounds_str})")
        print(f"  Backend: {args.backend}")
        print(f"  Batch size: {metrics.batch_size}")
        print(f"  Dataset: {metrics.num_train_utterances} utterances")
        print(f"  Epochs: {args.epochs}")
        print(f"  Total training time: {metrics.total_training_time:.1f}s")
        print(f"  Avg time per epoch: {metrics.training_time_per_epoch:.2f}s")
        print(
            f"  Throughput: {metrics.throughput_utterances_per_sec:.1f} ± "
            f"{metrics.throughput_utterances_per_sec_std:.1f} utt/s"
        )
        print(
            f"             {metrics.throughput_frames_per_sec/1000:.1f} ± "
            f"{metrics.throughput_frames_per_sec_std/1000:.1f}k frames/s"
        )
        print("-" * 60)
        print(f"  Best PER: {metrics.phone_error_rate:.4f}")
        print(f"  Boundary F1: {metrics.boundary_f1:.4f}")
        print(f"  Segment F1: {metrics.segment_f1:.4f}")
        print("=" * 60)
    elif args.command == "compare":
        results = compare_models(
            args.data_dir,
            max_duration=args.max_duration,
            include_dp_standard_ref=args.dp_standard_ref,
            uncertainty_analysis=args.uncertainty_analysis,
            uncertainty_num_utterances=args.uncertainty_num_utterances,
            uncertainty_selection=args.uncertainty_selection,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_every=args.log_every,
            precision=args.precision,
            use_sequence_boundaries=args.sequence_boundaries,
        )
        if args.output_json:
            from datetime import datetime

            output = {
                "task": "timit",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_duration": args.max_duration,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                },
                "linear_crf_triton": results["linear_crf_triton"].to_dict(),
                "semi_crf_triton": results["semi_crf_triton"].to_dict(),
            }
            if "semi_crf_dp_standard" in results:
                output["semi_crf_dp_standard"] = results["semi_crf_dp_standard"].to_dict()
            if "pytorch_crf" in results:
                output["pytorch_crf"] = results["pytorch_crf"].to_dict()
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {args.output_json}")

        # Export duration analysis if requested
        if args.export_duration:
            corpus_stats = load_corpus_duration_stats(args.data_dir)
            args.export_duration.parent.mkdir(parents=True, exist_ok=True)
            export_duration_analysis(results, corpus_stats, args.export_duration)

        # Generate duration plots if requested
        if args.plot_dir:
            corpus_stats = load_corpus_duration_stats(args.data_dir)
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            plot_duration_distributions(results, corpus_stats, args.plot_dir)

    elif args.command == "train-lightning":
        if not HAS_LIGHTNING:
            print(
                "PyTorch Lightning is required for this command. "
                "Install with: pip install flash-semicrf[lightning]"
            )
            return

        from flash_semicrf import UncertaintySemiMarkovCRFHead

        # Parse --devices: "1" → 1 (int), "2,3" → [2, 3] (list), "-1" → -1 (all)
        raw_devices = str(args.devices)
        if "," in raw_devices:
            devices = [int(d) for d in raw_devices.split(",")]
        else:
            devices = int(raw_devices)

        # Build DataModule — setup() computes train normalization stats and
        # passes them to the test dataset for consistent z-score normalization.
        datamodule = TIMITDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
        )
        datamodule.setup()
        input_dim = datamodule.input_dim

        k = 1 if args.model == "linear" else args.max_duration

        encoder = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
        crf = UncertaintySemiMarkovCRFHead(
            num_classes=NUM_PHONES,
            max_duration=k,
            hidden_dim=args.hidden_dim,
            precision=args.precision,
            use_sequence_boundaries=args.sequence_boundaries,
        )
        model = TIMITLightningModule(
            encoder=encoder,
            crf=crf,
            lr=args.lr,
            crf_lr_scale=args.crf_lr_scale,
            penalty_weight=args.crf_reg,
            scheduler="cosine",
            max_epochs=args.epochs,
            log_uncertainty_stats=True,
        )

        # Save best checkpoint by val/per so analyze-uncertainty gets the best model
        ckpt_callback = ModelCheckpoint(
            monitor="val/per",
            mode="min",
            save_top_k=1,
            filename=f"best-{args.model}-K{k}",
            save_last=True,
        )

        # Use Lightning precision=32 (standard float32 tensors throughout).
        # The SemiCRF kernel precision is controlled separately via args.precision.
        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            devices=devices,
            num_nodes=args.num_nodes,
            strategy=args.strategy,
            precision=32,
            enable_progress_bar=True,
            log_every_n_steps=10,
            callbacks=[ckpt_callback],
        )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Training TIMIT with Lightning: model={args.model} K={k}, "
            f"nodes={args.num_nodes}, devices={devices}, strategy={args.strategy}, "
            f"params={num_params:,}"
        )
        trainer.fit(model, datamodule)

        final_per = trainer.callback_metrics.get("val/per", "N/A")
        final_loss = trainer.callback_metrics.get("val/loss", "N/A")
        final_f1 = trainer.callback_metrics.get("val/boundary_f1", "N/A")
        best_ckpt = ckpt_callback.best_model_path
        print("\n" + "=" * 60)
        print("LIGHTNING TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Model:         {args.model} (K={k})")
        print(f"  Devices:       {devices} x {args.accelerator}")
        print(f"  Strategy:      {args.strategy}")
        print(f"  Epochs:        {args.epochs}")
        print("-" * 60)
        print(f"  Final val loss:        {final_loss}")
        print(f"  Final val PER:         {final_per}")
        print(f"  Final val Boundary F1: {final_f1}")
        print("-" * 60)
        print(f"  Best checkpoint:       {best_ckpt}")
        print("=" * 60)

    elif args.command == "analyze-uncertainty":
        if not HAS_LIGHTNING:
            print(
                "PyTorch Lightning is required for this command. "
                "Install with: pip install flash-semicrf[lightning]"
            )
            return
        run_analyze_uncertainty(args)


if __name__ == "__main__":
    main()
