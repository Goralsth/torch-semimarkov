"""PyTorch Lightning integration for flash-semicrf.

This module provides :class:`SemiCRFLightningModule` for training and inference
with PyTorch Lightning, including full DDP support.

**Precision requirement**

The streaming forward-backward algorithm requires float64 for numerical stability
of the partition function and log-space cumulative sums over long sequences.
The CRF head casts to float64 internally, but mixed-precision training (bf16/fp16)
produces NaN gradients at the float64↔fp16 boundary during backpropagation.
Standalone inference is numerically correct regardless of trainer precision
(the internal cast protects the forward pass), but ``precision=32`` is recommended
uniformly to avoid confusion::

    trainer = L.Trainer(precision=32, ...)   # Required for training

**DDP notes**

- ``sync_dist=True`` is used on all epoch-level metrics. Per-step ``train/loss``
  is logged with ``sync_dist=False`` to avoid an all-reduce on every step.
- ``gradient_checkpointing=True`` requires
  ``Trainer(strategy=DDPStrategy(find_unused_parameters=False), ...)``.
- ``predict_step`` returns a serializable dict; ``segments`` stay local to each rank
  and are NOT gathered across ranks. For full-dataset predictions, use a
  ``BasePredictionWriter`` callback or run inference on a single GPU (Viterbi
  traceback is sequential; DDP confers no benefit).
- Do NOT construct ``DistributedSampler`` manually — Lightning handles it automatically
  when ``shuffle=True`` is set on the train DataLoader.

**Example**::

    from flash_semicrf import SemiMarkovCRFHead, SemiCRFLightningModule, pad_and_collate

    encoder = MyEncoder(vocab_size=5, hidden_dim=256)
    crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=256)

    model = SemiCRFLightningModule(
        encoder=encoder,
        crf=crf,
        lr=1e-3,
        scheduler="plateau",
    )

    trainer = L.Trainer(
        max_epochs=50,
        precision=32,          # required
        accelerator="gpu",
        devices=4,
        strategy="ddp",
    )
    trainer.fit(model, train_loader, val_loader)
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import lightning.pytorch as L  # lightning>=2.0 (new package name)
except ImportError:
    try:
        import pytorch_lightning as L  # legacy fallback
    except ImportError:
        raise ImportError(
            "flash-semicrf Lightning integration requires pytorch-lightning>=2.0 or "
            "lightning>=2.0. Install with: pip install flash-semicrf[lightning]"
        ) from None

from .nn import SemiMarkovCRFHead, ViterbiResult
from .uncertainty import UncertaintySemiMarkovCRFHead

__all__ = ["pad_and_collate", "SemiCRFLightningModule"]


def pad_and_collate(batch: list[dict]) -> dict:
    """Collate variable-length sequences for SemiCRF training.

    Pads any tensor whose leading dimension equals ``item["lengths"]`` (sequence
    tensors). Stacks fixed-shape tensors as-is. Non-tensor values are collected
    as Python lists. All batch items must have identical keys.

    Args:
        batch: List of dicts from a Dataset. Each dict must contain at minimum:

            - ``"inputs"``: Tensor of shape ``(T,)`` or ``(T, D)``
            - ``"labels"``: Tensor of shape ``(T,)`` — per-position integer labels
            - ``"lengths"``: Scalar tensor or int — actual sequence length

            Additional keys are handled automatically. Tensors whose leading dim
            equals ``item["lengths"]`` are padded to ``T_max``; all others are stacked.

    Returns:
        Dict with the same keys as the input items. ``"lengths"`` is a
        ``(batch,)`` int64 tensor. Sequence tensors are padded with zeros.

    Raises:
        ValueError: If batch items have different keys.

    Note:
        Labels are padded with zeros. This is safe because ``lengths`` masks out
        padded positions inside ``compute_loss`` / ``score_gold_vectorized``.
    """
    all_keys = set(batch[0].keys())
    if not all(set(item.keys()) == all_keys for item in batch):
        raise ValueError("All batch items must have identical keys. " f"Expected {all_keys}.")

    lengths = torch.tensor([item["lengths"] for item in batch], dtype=torch.long)
    T_max = int(lengths.max())
    result: dict = {"lengths": lengths}

    for key in batch[0]:
        if key == "lengths":
            continue
        values = [item[key] for item in batch]
        if not isinstance(values[0], Tensor):
            # Non-tensor fields (string IDs, Python scalars, etc.): collect as list
            result[key] = values
            continue
        # Pad only if leading dim matches per-item sequence length (sequence tensor)
        if values[0].shape[0] == int(batch[0]["lengths"]):
            if values[0].dim() == 1:
                result[key] = torch.stack([F.pad(t, (0, T_max - t.shape[0])) for t in values])
            elif values[0].dim() == 2:
                result[key] = torch.stack([F.pad(t, (0, 0, 0, T_max - t.shape[0])) for t in values])
            else:
                result[key] = torch.stack(
                    [F.pad(t, (0, 0) * (t.dim() - 1) + (0, T_max - t.shape[0])) for t in values]
                )
        else:
            # Fixed-shape tensor (feature vectors, class embeddings, etc.): stack as-is
            result[key] = torch.stack(values)

    return result


class SemiCRFLightningModule(L.LightningModule):
    """PyTorch Lightning module for Semi-Markov CRF sequence segmentation.

    Wraps a user-supplied encoder and a :class:`~flash_semicrf.SemiMarkovCRFHead`
    (or :class:`~flash_semicrf.UncertaintySemiMarkovCRFHead`) into a standard
    Lightning training loop with full DDP support.

    Args:
        encoder: Encoder module. Called as ``encoder(batch["inputs"])`` and must
            return hidden states of shape ``(batch, T, hidden_dim)``. For integer
            token ID inputs, the embedding layer creates the float boundary needed
            for gradient checkpointing.
        crf: CRF head. Accepts :class:`~flash_semicrf.SemiMarkovCRFHead` or
            :class:`~flash_semicrf.UncertaintySemiMarkovCRFHead`. Uncertainty
            entropy stats are logged during validation/test when the latter is used.
        lr: Base learning rate for the encoder and CRF projection layers.
        crf_lr_scale: Multiplier applied to ``lr`` for CRF structural parameters
            (``transition``, ``duration_dist``). Defaults to 0.1 — structural params
            converge faster and should be trained at a lower rate.
        weight_decay: AdamW weight decay applied to all param groups.
        penalty_weight: Weight of the Lp regularization penalty on CRF parameters.
            0.0 disables regularization.
        penalty_p: Exponent for :meth:`~flash_semicrf.SemiMarkovCRFHead.parameter_penalty`.
        scheduler: LR scheduler. One of ``"plateau"`` (ReduceLROnPlateau monitoring
            ``val/loss``), ``"cosine"`` (CosineAnnealingLR, requires ``max_epochs``),
            or ``"none"``.
        max_epochs: Required when ``scheduler="cosine"``.
        plateau_patience: Patience for ReduceLROnPlateau.
        plateau_factor: Factor for ReduceLROnPlateau.
        gradient_checkpointing: If ``True``, wraps encoder forward in
            :func:`torch.utils.checkpoint.checkpoint` during training to reduce
            activation memory. Requires
            ``Trainer(strategy=DDPStrategy(find_unused_parameters=False))``.
        log_uncertainty_stats: If ``True`` and ``crf`` is an
            :class:`~flash_semicrf.UncertaintySemiMarkovCRFHead`, logs
            ``{prefix}/entropy_mean`` and ``{prefix}/entropy_max`` during
            validation and test.

    Example::

        encoder = SimpleEncoder(vocab_size=5, hidden_dim=64)
        crf = SemiMarkovCRFHead(num_classes=10, max_duration=50, hidden_dim=64)
        model = SemiCRFLightningModule(encoder, crf, lr=1e-3)

        trainer = L.Trainer(max_epochs=10, precision=32, accelerator="gpu")
        trainer.fit(model, train_loader, val_loader)
    """

    def __init__(
        self,
        encoder: nn.Module,
        crf: Union[SemiMarkovCRFHead, UncertaintySemiMarkovCRFHead],
        lr: float = 1e-3,
        crf_lr_scale: float = 0.1,
        weight_decay: float = 0.01,
        penalty_weight: float = 0.0,
        penalty_p: float = 2.0,
        scheduler: Literal["plateau", "cosine", "none"] = "plateau",
        max_epochs: Optional[int] = None,
        plateau_patience: int = 10,
        plateau_factor: float = 0.5,
        gradient_checkpointing: bool = False,
        log_uncertainty_stats: bool = True,
    ) -> None:
        super().__init__()
        # Save scalar hparams; exclude nn.Module args (not yaml-serializable)
        self.save_hyperparameters(ignore=["encoder", "crf"])

        self.encoder = encoder
        self.crf = crf
        self._has_uncertainty = isinstance(crf, UncertaintySemiMarkovCRFHead)

        if scheduler == "cosine" and max_epochs is None:
            raise ValueError("scheduler='cosine' requires max_epochs to be set in the constructor.")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Warn if trainer precision will cause NaN gradients during training."""
        prec = str(self.trainer.precision)
        if prec not in {"32", "32-true"}:
            warnings.warn(
                f"SemiMarkovCRFHead requires precision=32 (got {prec!r}). "
                "The streaming forward-backward algorithm requires float64 for numerical "
                "stability of the partition function over long sequences. The CRF head "
                "casts to float64 internally, but mixed-precision training (bf16/fp16) "
                "produces NaN gradients at the float64-fp16 boundary during backpropagation. "
                "Set Trainer(precision=32).",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, inputs: Tensor) -> Tensor:
        """Encoder forward with optional gradient checkpointing during training.

        For integer token ID inputs, the embedding layer creates the float boundary;
        checkpointing wraps the full encoder including the embedding.
        ``use_reentrant=False`` is required for PyTorch >= 2.0 to avoid conflicts
        with Lightning's gradient scaler and DDP hooks.
        """
        if self.hparams.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.encoder, inputs, use_reentrant=False)
        return self.encoder(inputs)

    def _eval_step(self, batch: dict, prefix: str) -> Tensor:
        """Shared logic for validation_step and test_step."""
        hidden = self._encode(batch["inputs"])
        loss = self.crf.compute_loss(hidden, batch["lengths"], batch["labels"])
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        if self._has_uncertainty and self.hparams.log_uncertainty_stats:
            # Lightning's val/test loops already disable gradients; no no_grad() needed.
            entropy = self.crf.compute_entropy_streaming(hidden, batch["lengths"])
            self.log(
                f"{prefix}/entropy_mean",
                entropy.mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/entropy_max",
                entropy.max(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: Tensor, lengths: Tensor) -> dict:
        """Forward pass returning the partition function dict.

        Args:
            inputs: Shape ``(batch, T, hidden_dim)`` or ``(batch, T, C)``.
            lengths: Shape ``(batch,)`` — actual sequence lengths.

        Returns:
            Dict with keys ``partition``, ``cum_scores`` (and optionally
            ``proj_start``, ``proj_end`` if boundary projections are enabled).
        """
        hidden = self._encode(inputs)
        return self.crf(hidden, lengths)

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Training step: NLL loss + optional parameter penalty.

        Logs:
            - ``train/loss``: per-step, no all-reduce (cheap for DDP).
            - ``train/loss_epoch``: epoch-level with cross-rank averaging.
        """
        hidden = self._encode(batch["inputs"])
        loss = self.crf.compute_loss(hidden, batch["lengths"], batch["labels"])

        if self.hparams.penalty_weight > 0:
            loss = loss + self.hparams.penalty_weight * self.crf.parameter_penalty(
                self.hparams.penalty_p
            )

        # Per-step: sync_dist=False — all-reduce on every step is expensive.
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,
        )
        # Epoch-level: sync_dist=True for correct cross-rank averaging.
        self.log(
            "train/loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._eval_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._eval_step(batch, "test")

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Viterbi decoding step.

        Lightning's predict loop runs under ``torch.no_grad()`` automatically.

        Returns a serializable dict rather than :class:`~flash_semicrf.ViterbiResult`
        because ``ViterbiResult.segments`` (``List[List[Segment]]``) cannot survive
        DDP all_gather. In DDP, every rank runs ``predict_step`` on its own shard;
        results are collected on rank 0. Segments stay local to each rank and are
        NOT gathered across ranks.

        For full-dataset predictions across all ranks, use a
        ``BasePredictionWriter`` callback or run inference on a single GPU
        (Viterbi traceback is sequential; DDP confers no throughput benefit).

        Returns:
            Dict with keys:

            - ``"scores"``: Tensor of shape ``(batch,)`` — max segmentation scores.
            - ``"segments"``: ``List[List[Segment]]`` — decoded segments (local rank).
            - ``"lengths"``: Tensor of shape ``(batch,)`` — sequence lengths.
        """
        hidden = self._encode(batch["inputs"])
        result: ViterbiResult = self.crf.decode_with_traceback(hidden, batch["lengths"])
        return {
            "scores": result.scores,
            "segments": result.segments,
            "lengths": batch["lengths"],
        }

    # ------------------------------------------------------------------
    # Optimizers and schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW with separate LR for encoder+projections vs CRF structural params.

        CRF structural parameters (``transition``, ``duration_dist``) converge faster
        than the emission projections and are trained at ``lr * crf_lr_scale``.
        Projection layers (``projection``, ``proj_start_layer``, ``proj_end_layer``)
        are encoder-adjacent and use the full ``lr``.

        When ``hidden_dim=None``, the CRF has no projection layers; all CRF params
        fall into the structural group (empty projection list is accepted by AdamW).
        """
        crf_projection_params = [
            p for n, p in self.crf.named_parameters() if "projection" in n or "proj_" in n
        ]
        crf_structural_params = [
            p for n, p in self.crf.named_parameters() if "projection" not in n and "proj_" not in n
        ]

        # Sanity check: every CRF param must appear in exactly one group
        all_crf_ids = {id(p) for p in self.crf.parameters()}
        grouped_ids = {id(p) for p in crf_projection_params + crf_structural_params}
        if all_crf_ids != grouped_ids:
            raise RuntimeError(
                "Param group split missed or double-counted CRF parameters. "
                "Check projection name filters if crf attributes were renamed."
            )

        param_groups = [
            {
                "params": list(self.encoder.parameters()) + crf_projection_params,
                "lr": self.hparams.lr,
            },
            {
                "params": crf_structural_params,
                "lr": self.hparams.lr * self.hparams.crf_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "none":
            return optimizer

        if self.hparams.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.plateau_factor,
                patience=self.hparams.plateau_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if self.hparams.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }

        raise ValueError(
            f"Unknown scheduler {self.hparams.scheduler!r}. "
            "Expected 'plateau', 'cosine', or 'none'."
        )
