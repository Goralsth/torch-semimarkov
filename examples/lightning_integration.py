#!/usr/bin/env python3
"""Example: Using SemiCRFLightningModule with PyTorch Lightning.

This example demonstrates how to integrate flash-semicrf with PyTorch Lightning
for distributed training of sequence segmentation models.

Key points:
1. SemiCRFLightningModule wraps any encoder + SemiMarkovCRFHead (or Uncertainty head)
2. DDP works automatically — no special handling needed
3. Must use precision=32 for numerical stability (the streaming algorithm requires float64)
4. pad_and_collate handles variable-length sequences in DataLoaders

Usage:
    # Single GPU
    python lightning_integration.py --devices 1

    # Multi-GPU DDP
    python lightning_integration.py --devices 4 --strategy ddp

    # CPU (for testing)
    python lightning_integration.py --accelerator cpu
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Lightning import (supports lightning>=2.0 and legacy pytorch-lightning)
# ---------------------------------------------------------------------------
try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import LearningRateMonitor

    HAS_LIGHTNING = True
except ImportError:
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import LearningRateMonitor  # type: ignore[no-redef]

        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False
        print(
            "PyTorch Lightning not installed. " "Install with: pip install flash-semicrf[lightning]"
        )

from flash_semicrf import SemiCRFLightningModule, SemiMarkovCRFHead, pad_and_collate
from flash_semicrf.uncertainty import UncertaintySemiMarkovCRFHead

# ---------------------------------------------------------------------------
# Encoder (replace with your actual encoder: Mamba, Transformer, CNN, etc.)
# ---------------------------------------------------------------------------


class SimpleEncoder(nn.Module):
    """Simple encoder for demonstration purposes."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: Input token IDs of shape (batch, T)

        Returns:
            Hidden states of shape (batch, T, hidden_dim)
        """
        return self.layers(self.embedding(x))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DummyDataset(Dataset):
    """Synthetic dataset with random sequences and segment labels."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 100,
        vocab_size: int = 5,
        num_classes: int = 10,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = torch.zeros(self.seq_length, dtype=torch.long)
        pos = 0
        while pos < self.seq_length:
            seg_len = min(torch.randint(1, 21, (1,)).item(), self.seq_length - pos)
            label = torch.randint(0, self.num_classes, (1,)).item()
            labels[pos : pos + seg_len] = label
            pos += seg_len
        return {
            "inputs": inputs,
            "labels": labels,
            "lengths": torch.tensor(self.seq_length),
        }


# ---------------------------------------------------------------------------
# DataModule (concrete Lightning DataModule using pad_and_collate)
# ---------------------------------------------------------------------------

if HAS_LIGHTNING:

    class DummyDataModule(L.LightningDataModule):
        """Example DataModule with variable-length sequence support.

        Uses :func:`~flash_semicrf.pad_and_collate` as the collate function.
        Lightning automatically handles DistributedSampler in DDP — do NOT
        construct it manually.
        """

        def __init__(
            self,
            num_samples: int = 1000,
            seq_length: int = 100,
            vocab_size: int = 5,
            num_classes: int = 10,
            batch_size: int = 32,
            num_workers: int = 0,
        ):
            super().__init__()
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size
            self.num_classes = num_classes
            self.batch_size = batch_size
            self.num_workers = num_workers

            self.train_dataset: DummyDataset | None = None
            self.val_dataset: DummyDataset | None = None

        def setup(self, stage=None):
            self.train_dataset = DummyDataset(
                num_samples=self.num_samples,
                seq_length=self.seq_length,
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
            )
            self.val_dataset = DummyDataset(
                num_samples=max(self.num_samples // 10, 10),
                seq_length=self.seq_length,
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
            )

        def train_dataloader(self) -> DataLoader:
            # shuffle=True is converted to DistributedSampler(shuffle=True) by
            # Lightning in DDP — do NOT construct DistributedSampler manually.
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=pad_and_collate,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=pad_and_collate,
            )

        def predict_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=pad_and_collate,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Lightning + SemiCRFLightningModule Example")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (ddp, etc)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (gpu, cpu)")
    parser.add_argument("--max_epochs", type=int, default=5, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--max_duration", type=int, default=50, help="Max segment duration")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument(
        "--uncertainty", action="store_true", help="Use UncertaintySemiMarkovCRFHead"
    )
    args = parser.parse_args()

    if not HAS_LIGHTNING:
        print("PyTorch Lightning is required. Install with: pip install flash-semicrf[lightning]")
        return

    # Build encoder
    encoder = SimpleEncoder(vocab_size=5, hidden_dim=args.hidden_dim)

    # Build CRF head
    crf_cls = UncertaintySemiMarkovCRFHead if args.uncertainty else SemiMarkovCRFHead
    crf = crf_cls(
        num_classes=args.num_classes,
        max_duration=args.max_duration,
        hidden_dim=args.hidden_dim,
    )

    # Build Lightning module
    model = SemiCRFLightningModule(
        encoder=encoder,
        crf=crf,
        lr=1e-3,
        crf_lr_scale=0.1,  # transition/duration params trained at lower LR
        scheduler="plateau",
        log_uncertainty_stats=args.uncertainty,
    )

    # DataModule
    datamodule = DummyDataModule(
        num_samples=1000,
        seq_length=args.seq_length,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
    )

    # Callbacks
    callbacks = [LearningRateMonitor(logging_interval="step")]

    # Trainer
    # CRITICAL: precision=32 is required — the streaming semi-CRF algorithm requires
    # float64 for the partition function. The head casts internally, but mixed-precision
    # training produces NaN gradients at the float64↔fp16 boundary.
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=32,  # REQUIRED — see module docstring
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    head_name = "UncertaintySemiMarkovCRFHead" if args.uncertainty else "SemiMarkovCRFHead"
    print(f"\nTraining with {args.devices} device(s), strategy={args.strategy}")
    print(f"CRF head: {head_name}, {args.num_classes} classes, max_duration={args.max_duration}")
    print(f"Sequence length: {args.seq_length}, batch size: {args.batch_size}\n")

    trainer.fit(model, datamodule)

    print("\nTraining complete!")
    print(f"Final train loss: {trainer.callback_metrics.get('train/loss', 'N/A'):.4f}")
    print(f"Final val loss:   {trainer.callback_metrics.get('val/loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
