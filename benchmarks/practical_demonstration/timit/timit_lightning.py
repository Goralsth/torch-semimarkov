#!/usr/bin/env python3
"""TIMIT Lightning integration: TIMITDataModule and TIMITLightningModule."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as L

    HAS_LIGHTNING = True
except ImportError:
    try:
        import pytorch_lightning as L  # type: ignore[no-redef]

        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False

from .timit_data import TIMITDataset, collate_timit
from .timit_metrics import (
    compute_boundary_metrics,
    compute_phone_error_rate,
    compute_segment_metrics,
    labels_to_segments,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Lightning Integration (optional)
# =============================================================================


def timit_collate_for_lightning(batch: list[dict]) -> dict:
    """Collate a TIMIT batch for :class:`SemiCRFLightningModule`.

    Wraps :func:`collate_timit` and renames ``"features"`` → ``"inputs"`` so the
    batch is compatible with :class:`~flash_semicrf.SemiCRFLightningModule`, which
    reads ``batch["inputs"]`` in every step method.

    The ``"lengths"`` key (plural) is already produced by :func:`collate_timit`.
    ``"utterance_ids"`` is kept as a list and passed through unchanged.
    """
    out = collate_timit(batch)
    out["inputs"] = out.pop("features")
    return out


if HAS_LIGHTNING:
    from flash_semicrf import SemiCRFLightningModule

    class TIMITDataModule(L.LightningDataModule):
        """LightningDataModule for TIMIT phoneme segmentation.

        Computes normalization statistics from the training set and applies them to
        the test set, ensuring consistent z-score normalization across splits.
        Lightning automatically wraps the DataLoader with :class:`DistributedSampler`
        in DDP — do NOT construct it manually.
        """

        def __init__(
            self,
            data_dir: Path,
            batch_size: int = 32,
            num_workers: int = 0,
            max_length: int | None = None,
        ):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.max_length = max_length

            self.train_dataset: TIMITDataset | None = None
            self.val_dataset: TIMITDataset | None = None

        @property
        def input_dim(self) -> int:
            """Feature dimension — available after :meth:`setup` is called."""
            if self.train_dataset is None:
                raise RuntimeError("Call setup() before accessing input_dim.")
            return int(self.train_dataset[0]["features"].shape[-1])

        def setup(self, stage=None):
            # Train dataset computes normalization stats from its own data
            self.train_dataset = TIMITDataset(
                self.data_dir / "train.jsonl",
                max_length=self.max_length,
                normalize=True,
            )
            # Test dataset inherits train stats for consistent normalization
            self.val_dataset = TIMITDataset(
                self.data_dir / "test.jsonl",
                max_length=self.max_length,
                normalize=True,
                mean=self.train_dataset.mean,
                std=self.train_dataset.std,
            )

        def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=timit_collate_for_lightning,
                drop_last=True,  # ensures equal batch sizes across ranks for correct DDP loss averaging
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=timit_collate_for_lightning,
            )

        def predict_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=timit_collate_for_lightning,
            )

    class TIMITLightningModule(SemiCRFLightningModule):
        """SemiCRFLightningModule subclass with TIMIT-specific validation metrics.

        Extends the base Lightning module with per-epoch Phone Error Rate (PER)
        and boundary F1 computation during validation.  Viterbi decode runs
        under :func:`torch.no_grad` (already guaranteed by Lightning's val loop)
        and reuses the hidden states computed for the NLL loss.

        Additional logged metrics:
            - ``val/per`` — Phone Error Rate (Levenshtein / reference phones)
            - ``val/boundary_f1`` — Boundary F1 at exact match (tolerance = 0)
            - ``val/boundary_f1_tol2`` — Boundary F1 at ±2 frame tolerance

        DDP note: ``val/per`` is per-rank (each rank computes PER from its own
        shard).  For exact cross-rank PER, gather predictions before computing.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._val_predictions: list[list[int]] = []
            self._val_references: list[list[int]] = []

        def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
            """NLL loss + Viterbi decode for PER/F1 accumulation (single encoder pass)."""
            # Single encoder pass shared by loss computation and Viterbi decode.
            hidden = self._encode(batch["inputs"])
            bsz = hidden.shape[0]
            loss = self.crf.compute_loss(hidden, batch["lengths"], batch["labels"])
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=bsz,
            )
            if self._has_uncertainty and self.hparams.log_uncertainty_stats:
                entropy = self.crf.compute_entropy_streaming(hidden, batch["lengths"])
                self.log(
                    "val/entropy_mean",
                    entropy.mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bsz,
                )
                self.log(
                    "val/entropy_max",
                    entropy.max(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bsz,
                )
                # Per-position label entropy: informative for K=1 where boundary
                # entropy is trivially uniform (every frame is a boundary).
                pos_margs = self.crf.compute_position_marginals(hidden, batch["lengths"])
                # H(t) = -sum_c p(c|t) log p(c|t), averaged over valid frames
                pos_log = torch.log(pos_margs.clamp(min=1e-10))
                pos_ent = -(pos_margs * pos_log).sum(dim=-1)  # (batch, T)
                mask = torch.arange(pos_ent.shape[1], device=pos_ent.device).unsqueeze(0) < batch[
                    "lengths"
                ].unsqueeze(1)
                pos_ent_mean = (pos_ent * mask).sum() / mask.sum()
                self.log(
                    "val/position_entropy_mean",
                    pos_ent_mean,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bsz,
                )

            result = self.crf.decode_with_traceback(hidden, batch["lengths"])

            for i, seg_list in enumerate(result.segments):
                seq_len = int(batch["lengths"][i].item())
                pred_labels = [0] * seq_len
                for seg in seg_list:
                    # flash_semicrf.Segment uses inclusive end; convert to exclusive range
                    for j in range(seg.start, min(seg.end + 1, seq_len)):
                        pred_labels[j] = seg.label
                self._val_predictions.append(pred_labels)
                self._val_references.append(batch["labels"][i, :seq_len].cpu().tolist())

            return loss

        def on_validation_epoch_end(self) -> None:
            """Compute and log PER, boundary, and segment metrics over the full validation epoch."""
            if not self._val_predictions:
                return
            per = compute_phone_error_rate(self._val_predictions, self._val_references)
            boundary = compute_boundary_metrics(self._val_predictions, self._val_references)
            pred_segs = [labels_to_segments(p) for p in self._val_predictions]
            ref_segs = [labels_to_segments(r) for r in self._val_references]
            segment = compute_segment_metrics(pred_segs, ref_segs)

            self.log("val/per", per, prog_bar=True, sync_dist=True)
            self.log("val/boundary_f1", boundary["boundary_f1_tol0"], prog_bar=True, sync_dist=True)
            self.log("val/boundary_f1_tol1", boundary.get("boundary_f1_tol1", 0.0), sync_dist=True)
            self.log("val/boundary_f1_tol2", boundary.get("boundary_f1_tol2", 0.0), sync_dist=True)
            self.log("val/boundary_precision", boundary["boundary_precision"], sync_dist=True)
            self.log("val/boundary_recall", boundary["boundary_recall"], sync_dist=True)
            self.log("val/segment_f1", segment["segment_f1"], prog_bar=True, sync_dist=True)
            self.log("val/segment_precision", segment["segment_precision"], sync_dist=True)
            self.log("val/segment_recall", segment["segment_recall"], sync_dist=True)

            self._val_predictions.clear()
            self._val_references.clear()
