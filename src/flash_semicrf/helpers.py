import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .semirings import LogSemiring
from .validation import validate_cum_scores, validate_labels, validate_lengths


@dataclass
class Segment:
    """A segment in a semi-Markov segmentation: [start, end] with label."""

    start: int
    end: int
    label: int
    score: float = 0.0

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


@dataclass
class ViterbiResult:
    """Viterbi result: scores (batch,) and segments (per-batch list of Segment)."""

    scores: Tensor
    segments: list[list[Segment]]


class _Struct:
    """Base class for structured prediction with semiring DP algorithms.

    Provides common infrastructure for semiring-parameterized dynamic programming:
    chart allocation, scoring, marginal computation, and binary tree utilities.

    Args:
        semiring: Semiring class defining zero, one, sum, and prod operations.
            Defaults to LogSemiring for partition function computation.
    """

    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def score(self, potentials, parts, batch_dims=None):
        """Score a structure given potentials and binary parts indicator.

        Args:
            potentials: Log-potentials tensor.
            parts: Binary indicator tensor (same shape as potentials).
            batch_dims: List of batch dimension indices. Defaults to [0].

        Returns:
            Semiring product of selected potentials, shape determined by batch_dims.
        """
        if batch_dims is None:
            batch_dims = [0]
        score = torch.mul(potentials, parts)
        batch = tuple(score.shape[b] for b in batch_dims)
        return self.semiring.prod(score.view(batch + (-1,)))

    def _bin_length(self, length: int) -> tuple[int, int]:
        """Compute binary tree parameters for given sequence length.

        Args:
            length: Original sequence length.

        Returns:
            Tuple of (log_N, bin_N) where log_N is the tree depth and
            bin_N is the padded power-of-2 length.
        """
        log_N = int(math.ceil(math.log(length, 2)))
        bin_N = int(math.pow(2, log_N))
        return log_N, bin_N

    def _get_dimension(self, edge):
        """Extract dimensions from edge potentials and enable gradients.

        Args:
            edge: Edge potentials tensor or list of tensors.

        Returns:
            Shape tuple from edge (or first element if list).
        """
        if isinstance(edge, list):
            for t in edge:
                t.requires_grad_(True)
            return edge[0].shape
        else:
            edge.requires_grad_(True)
            return edge.shape

    def _chart(self, size: tuple, potentials: Tensor, force_grad: bool) -> Tensor:
        """Allocate a single DP chart tensor.

        Args:
            size: Shape tuple for the chart (excluding semiring dimension).
            potentials: Reference tensor for dtype and device.
            force_grad: Whether to enable gradients on the chart.

        Returns:
            Chart tensor of shape (semiring.size(), *size) initialized to semiring.zero.
        """
        return self._make_chart(1, size, potentials, force_grad)[0]

    def _make_chart(
        self, N: int, size: tuple, potentials: Tensor, force_grad: bool = False
    ) -> list:
        """Allocate N DP chart tensors initialized to semiring.zero.

        Args:
            N: Number of chart tensors to allocate.
            size: Shape tuple for each chart (excluding semiring dimension).
            potentials: Reference tensor for dtype and device.
            force_grad: Whether to enable gradients (only if potentials doesn't require grad).

        Returns:
            List of N chart tensors, each of shape (semiring.size(), *size).
        """
        chart = []
        for _ in range(N):
            c = torch.zeros(
                *((self.semiring.size(),) + size), dtype=potentials.dtype, device=potentials.device
            )
            c[:] = self.semiring.zero.view((self.semiring.size(),) + len(size) * (1,))
            c.requires_grad_(force_grad and not potentials.requires_grad)
            chart.append(c)
        return chart

    def sum(self, logpotentials, lengths=None, _raw=False, **kwargs):
        """Semiring sum over structures. LogSemiring: log Z. MaxSemiring: Viterbi."""
        v = self.logpartition(logpotentials, lengths, **kwargs)[0]
        if _raw:
            return v
        return self.semiring.unconvert(v)

    def marginals(self, logpotentials, lengths=None, _raw=False, **kwargs):
        """Posterior marginals via d(log Z)/d(potentials)."""
        v, edges, _ = self.logpartition(logpotentials, lengths=lengths, force_grad=True, **kwargs)
        if _raw:
            all_m = []
            for k in range(v.shape[0]):
                obj = v[k].sum(dim=0)
                marg = torch.autograd.grad(
                    obj, edges, create_graph=True, only_inputs=True, allow_unused=False
                )
                all_m.append(self.semiring.unconvert(self._arrange_marginals(marg)))
            return torch.stack(all_m, dim=0)
        else:
            obj = self.semiring.unconvert(v).sum(dim=0)
            marg = torch.autograd.grad(
                obj, edges, create_graph=True, only_inputs=True, allow_unused=False
            )
            return self.semiring.unconvert(self._arrange_marginals(marg))

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        """Base impl: return spans unchanged. Subclasses override."""
        return spans

    @staticmethod
    def from_parts(spans):
        """Base impl: return (spans, None). Subclasses override."""
        return spans, None

    def _arrange_marginals(self, marg):
        """Arrange marginal gradients into output format."""
        return marg[0]


def score_gold_vectorized(
    cum_scores: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    max_duration: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Score gold label sequences for Semi-CRF loss computation.

    Computes :math:`\text{score}(y^*) = \sum_{\text{segments}} (\text{content} + \text{duration} + \text{transition} + \text{boundary})`

    Args:
        cum_scores (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`.
        labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)`.
        duration_bias (Tensor): Duration bias of shape :math:`(K, C)`.
        max_duration (int): Maximum segment duration (K).
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``

    Returns:
        Tensor: Gold sequence scores of shape :math:`(\text{batch},)`.
    """
    # Input validation
    validate_cum_scores(cum_scores)
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    validate_labels(labels, C, batch_size=batch, seq_length=T)
    validate_lengths(lengths, T, batch_size=batch)

    if proj_start is not None:
        if proj_start.shape != torch.Size([batch, T, C]):
            raise ValueError(
                f"proj_start must have shape (batch={batch}, T={T}, C={C}), "
                f"got {tuple(proj_start.shape)}"
            )
    if proj_end is not None:
        if proj_end.shape != torch.Size([batch, T, C]):
            raise ValueError(
                f"proj_end must have shape (batch={batch}, T={T}, C={C}), "
                f"got {tuple(proj_end.shape)}"
            )

    device = cum_scores.device
    dtype = cum_scores.dtype

    if T == 0:
        return torch.zeros(batch, device=device, dtype=dtype)

    # Handle single-position sequences
    if T == 1:
        # Single segment per batch: content + duration_bias[dur_idx, label]
        # Duration index convention: min(duration, K-1) where K = max_duration
        # For K=1: min(1, 0) = 0. For K>1: min(1, K-1) = 1
        label_0 = labels[:, 0]  # (batch,)
        content = cum_scores[:, 1, :].gather(1, label_0.unsqueeze(1)).squeeze(1)
        content -= cum_scores[:, 0, :].gather(1, label_0.unsqueeze(1)).squeeze(1)
        dur_idx = min(1, max_duration - 1)
        dur_bias = duration_bias[dur_idx, label_0]
        scores = content + dur_bias
        if proj_start is not None:
            scores = scores + proj_start[:, 0, :].gather(1, label_0.unsqueeze(1)).squeeze(1)
        if proj_end is not None:
            scores = scores + proj_end[:, 0, :].gather(1, label_0.unsqueeze(1)).squeeze(1)
        # Zero out scores for zero-length sequences
        scores = scores * (lengths > 0).to(dtype)
        return scores

    # Handle K=1 (linear CRF) specially: each frame is its own segment
    # This ensures gold scoring matches the partition function which assumes
    # duration-1 segments for K=1
    if max_duration == 1:
        # Content: sum of scores[t, label[t]] for all positions
        # = sum over t of (cum_scores[t+1, label[t]] - cum_scores[t, label[t]])
        labels_exp = labels.unsqueeze(-1)  # (batch, T, 1)
        cum_at_pos = cum_scores[:, :-1, :].gather(2, labels_exp).squeeze(-1)  # (batch, T)
        cum_at_next = cum_scores[:, 1:, :].gather(2, labels_exp).squeeze(-1)  # (batch, T)
        content_per_pos = cum_at_next - cum_at_pos  # (batch, T)

        # Duration: duration_bias[0, label[t]] for each position
        dur_per_pos = duration_bias[0, labels]  # (batch, T)

        # Transition: transition[label[t-1], label[t]] for t >= 1
        # First position has no incoming transition
        prev_labels = torch.zeros_like(labels)
        prev_labels[:, 1:] = labels[:, :-1]
        trans_flat = transition.view(-1)  # (C * C,)
        trans_indices = prev_labels * C + labels  # (batch, T)
        trans_per_pos = trans_flat[trans_indices]  # (batch, T)
        # Zero out first position's transition
        trans_per_pos[:, 0] = 0

        # Mask for valid positions
        pos_indices = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        valid_mask = pos_indices < lengths.unsqueeze(1)  # (batch, T)

        # For K=1 each position t is its own segment [t, t], so start == end == t
        total_per_pos = content_per_pos + dur_per_pos + trans_per_pos
        if proj_start is not None:
            total_per_pos = total_per_pos + proj_start.gather(2, labels_exp).squeeze(-1)
        if proj_end is not None:
            total_per_pos = total_per_pos + proj_end.gather(2, labels_exp).squeeze(-1)

        # Sum all components with masking
        total_per_pos = total_per_pos * valid_mask.to(dtype)
        return total_per_pos.sum(dim=1)  # (batch,)

    # Step 1: Detect segment boundaries (where label changes)
    # changes[b, t] = True if labels[b, t] != labels[b, t+1]
    changes = labels[:, :-1] != labels[:, 1:]  # (batch, T-1)

    # Create position indices
    positions = torch.arange(T - 1, device=device)  # (T-1,)

    # Step 2: Build segment representation
    # For each batch, segments end at change positions and at seq_len-1
    # We'll create padded tensors for seg_starts, seg_ends, seg_labels

    # Count segments per batch: num_changes + 1
    num_changes = changes.sum(dim=1)  # (batch,)

    # Create mask for valid positions within each sequence
    pos_mask = positions.unsqueeze(0) < (lengths.unsqueeze(1) - 1)  # (batch, T-1)
    changes_masked = changes & pos_mask  # Only count changes within valid positions

    # Recount with mask
    num_changes = changes_masked.sum(dim=1)  # (batch,)
    num_segments = num_changes + 1  # (batch,)
    max_segments = num_segments.max().item()

    if max_segments == 0:
        return torch.zeros(batch, device=device, dtype=dtype)

    # Create padded segment tensors
    seg_starts = torch.zeros(batch, max_segments, device=device, dtype=torch.long)
    seg_ends = torch.zeros(batch, max_segments, device=device, dtype=torch.long)
    seg_labels = torch.zeros(batch, max_segments, device=device, dtype=torch.long)
    seg_mask = torch.zeros(batch, max_segments, device=device, dtype=torch.bool)

    # Fill segment tensors - this loop is over batch, not segments
    # For very large batches, this could be further optimized, but batch is typically small
    for b in range(batch):
        seq_len = lengths[b].item()
        if seq_len == 0:
            continue

        # Find change positions for this batch element
        if seq_len == 1:
            # Single position = single segment
            seg_starts[b, 0] = 0
            seg_ends[b, 0] = 0
            seg_labels[b, 0] = labels[b, 0]
            seg_mask[b, 0] = True
        else:
            change_positions = torch.where(changes_masked[b])[0]
            n_segs = change_positions.shape[0] + 1

            # Segment ends: change positions and seq_len-1
            ends = torch.cat([change_positions, torch.tensor([seq_len - 1], device=device)])
            # Segment starts: 0 and change positions + 1
            starts = torch.cat([torch.tensor([0], device=device), change_positions + 1])

            seg_starts[b, :n_segs] = starts
            seg_ends[b, :n_segs] = ends
            seg_labels[b, :n_segs] = labels[b, starts]
            seg_mask[b, :n_segs] = True

    # Step 3: Vectorized score computation

    # Content scores: cum_scores[b, end+1, label] - cum_scores[b, start, label]

    # Gather end positions (end + 1 for cumulative indexing)
    end_positions = seg_ends + 1  # (batch, max_segments)
    start_positions = seg_starts  # (batch, max_segments)

    # Create indices for gathering: need (batch, max_segments) indices into (batch, T+1, C)
    # We'll gather all C values then select the correct label

    # cum_scores shape: (batch, T+1, C)
    # We want cum_scores[b, end+1, :] and cum_scores[b, start, :]

    # Expand indices for gathering
    end_idx = end_positions.unsqueeze(-1).expand(-1, -1, C)  # (batch, max_segments, C)
    start_idx = start_positions.unsqueeze(-1).expand(-1, -1, C)  # (batch, max_segments, C)

    # Gather cumulative scores at segment boundaries
    cum_at_end = cum_scores.gather(1, end_idx)  # (batch, max_segments, C)
    cum_at_start = cum_scores.gather(1, start_idx)  # (batch, max_segments, C)

    # Compute content scores for all labels, then select correct label
    content_all = cum_at_end - cum_at_start  # (batch, max_segments, C)

    # Select content score for the segment's label
    seg_labels_expanded = seg_labels.unsqueeze(-1)  # (batch, max_segments, 1)
    content_scores = content_all.gather(2, seg_labels_expanded).squeeze(-1)  # (batch, max_segments)

    # Duration scores
    # Duration index convention: duration d uses index min(d, K-1) where K = max_duration
    # For K=1: all durations map to index 0
    # For K>1: duration d maps to min(d, K-1)
    durations = seg_ends - seg_starts + 1  # (batch, max_segments), always >= 1
    dur_indices = durations.clamp(max=max_duration - 1)

    # Gather duration bias: duration_bias[dur_idx, label]
    # duration_bias shape: (K, C), we need (batch, max_segments) values
    dur_bias_flat = duration_bias.view(-1)  # (K * C,)
    flat_indices = dur_indices * C + seg_labels  # (batch, max_segments)
    dur_scores = dur_bias_flat[flat_indices]  # (batch, max_segments)

    # Transition scores
    # Need transition[prev_label, curr_label] for segments 1, 2, ...
    # First segment has no transition
    prev_labels = torch.zeros_like(seg_labels)
    prev_labels[:, 1:] = seg_labels[:, :-1]  # Shift labels right

    # Gather transitions: transition[prev, curr]
    trans_flat = transition.view(-1)  # (C * C,)
    trans_indices = prev_labels * C + seg_labels  # (batch, max_segments)
    trans_scores = trans_flat[trans_indices]  # (batch, max_segments)

    # Zero out first segment's transition
    # Zero out first segment's transition (no predecessor)
    # Use torch.where to avoid inf * 0 = NaN
    first_seg_mask = torch.zeros_like(seg_mask)
    first_seg_mask[:, 0] = True
    trans_scores = torch.where(first_seg_mask, torch.zeros_like(trans_scores), trans_scores)

    # Step 4: Boundary scores (optional)
    # Safe indices: padding slots have seg_mask=False and will be zeroed out,
    # but their index values must still be in-bounds for the gather to succeed.
    if proj_start is not None or proj_end is not None:
        safe_starts = seg_starts * seg_mask.long()  # 0 for padding (always valid)
        safe_ends = seg_ends * seg_mask.long()
        b_idx = torch.arange(batch, device=device).unsqueeze(1)  # (batch, 1)

    total_per_segment = content_scores + dur_scores + trans_scores  # (batch, max_segments)

    if proj_start is not None:
        start_scores = proj_start[b_idx, safe_starts, seg_labels]  # (batch, max_seg)
        total_per_segment = total_per_segment + start_scores

    if proj_end is not None:
        end_scores = proj_end[b_idx, safe_ends, seg_labels]  # (batch, max_seg)
        total_per_segment = total_per_segment + end_scores

    # Step 5: Sum with masking
    # Use torch.where instead of multiplication to avoid inf * 0 = NaN
    # This can happen if parameters drift to extreme values during training
    total_per_segment = torch.where(
        seg_mask, total_per_segment, torch.zeros_like(total_per_segment)
    )

    scores = total_per_segment.sum(dim=1)  # (batch,)

    return scores
