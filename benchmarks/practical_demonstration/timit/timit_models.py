#!/usr/bin/env python3
"""TIMIT model definitions: BiLSTMEncoder, TIMITModel, TIMITModelPytorchCRF."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchcrf import CRF as TorchCRF

    HAS_TORCHCRF = True
except ImportError:
    HAS_TORCHCRF = False
    TorchCRF = None

from .timit_data import NUM_PHONES

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for acoustic features."""

    def __init__(
        self,
        input_dim: int = 39,  # 13 MFCC * 3
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            hidden: (batch, T, hidden_dim)
        """
        x = self.input_proj(x)
        x = self.dropout(x)
        output, _ = self.lstm(x)
        return output


class TIMITModel(nn.Module):
    """
    Combined encoder + CRF head for TIMIT phoneme segmentation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_PHONES,
        max_duration: int = 1,
        hidden_dim: int = 256,
        duration_distribution: str = "learned",
        precision: str = "float32",
        use_sequence_boundaries: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.max_duration = max_duration

        # Use UncertaintySemiMarkovCRFHead for calibration/uncertainty estimation
        from flash_semicrf import UncertaintySemiMarkovCRFHead

        self.crf = UncertaintySemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=hidden_dim,
            duration_distribution=duration_distribution,
            precision=precision,
            use_sequence_boundaries=use_sequence_boundaries,
        )

    def _encode(self, inputs: Tensor) -> Tensor:
        """Encode inputs to hidden states (compatible with run_uncertainty_inference)."""
        return self.encoder(inputs)

    def forward(self, features: Tensor, lengths: Tensor) -> dict:
        hidden = self.encoder(features)
        return self.crf(hidden, lengths)

    def compute_loss(
        self,
        features: Tensor,
        lengths: Tensor,
        labels: Tensor,
        backend: str = "exact",
        use_triton: bool = False,
    ) -> Tensor:
        """
        Compute NLL loss for phoneme segmentation.

        Args:
            features: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            labels: Per-position labels (batch, T)
            backend: "exact", "streaming", or "auto"
            use_triton: Whether to use Triton kernels (streaming only)
        """
        hidden = self.encoder(features)
        return self.crf.compute_loss(
            hidden, lengths, labels, backend=backend, use_triton=use_triton
        )

    def decode(self, features: Tensor, lengths: Tensor, backend: str = "streaming"):
        hidden = self.encoder(features)
        # dp_standard is only valid for the partition function; map to exact for Viterbi
        decode_backend = "exact" if backend == "dp_standard" else backend
        return self.crf.decode_with_traceback(hidden, lengths, backend=decode_backend)


class TIMITModelPytorchCRF(nn.Module):
    """
    TIMIT model using pytorch-crf for baseline comparison.

    Uses the same BiLSTMEncoder but replaces SemiMarkovCRFHead with torchcrf.CRF.
    This enables fair comparison between pytorch-crf and flash-semicrf K=1.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = NUM_PHONES,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if not HAS_TORCHCRF:
            raise ImportError(
                "pytorch-crf required for this model. Install with: pip install pytorch-crf"
            )

        self.encoder = encoder
        self.emission_proj = nn.Linear(hidden_dim, num_classes)
        self.crf = TorchCRF(num_classes, batch_first=True)

    def forward(self, features: Tensor, _lengths: Tensor) -> Tensor:
        """Forward pass returning emission scores."""
        hidden = self.encoder(features)
        return self.emission_proj(hidden)

    def compute_loss(
        self,
        features: Tensor,
        lengths: Tensor,
        labels: Tensor,
        **_kwargs,
    ) -> Tensor:
        """
        Compute NLL loss using pytorch-crf.

        Args:
            features: Input features (batch, T, input_dim)
            lengths: Sequence lengths (batch,)
            labels: Per-position labels (batch, T)
            **_kwargs: Ignored (for API compatibility with TIMITModel)
        """
        hidden = self.encoder(features)
        emissions = self.emission_proj(hidden)

        # pytorch-crf expects mask of shape (batch, seq_len)
        _, seq_len = features.shape[:2]
        mask = torch.arange(seq_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)

        # pytorch-crf.forward() returns log-likelihood, we want NLL
        log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
        return -log_likelihood

    def decode(self, features: Tensor, lengths: Tensor) -> list[list[int]]:
        """
        Viterbi decode to get best label sequences.

        Returns:
            List of label sequences (one per batch element).
        """
        hidden = self.encoder(features)
        emissions = self.emission_proj(hidden)

        _, seq_len = features.shape[:2]
        mask = torch.arange(seq_len, device=features.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Returns list of lists (batch_size x variable_length)
        return self.crf.decode(emissions, mask=mask)
