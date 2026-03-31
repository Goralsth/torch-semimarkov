r"""PyTorch reference implementation for streaming Semi-CRF.

Pure PyTorch implementations for CPU fallback and testing Triton kernels.
"""

import math
import warnings
from typing import Optional

import torch

from .constants import NEG_INF


def _compute_checkpoint_interval(T: int, K: int, compute_dtype: str = "float64") -> int:
    """Checkpoint interval balancing memory and numerical stability.

    Uses sqrt(T*K) for optimal memory/compute tradeoff, capped to bound
    alpha drift between normalization points in the backward pass.

    For float64 compute:
        When K >= 64, the interval equals K (normalize at every segment boundary).
        When K < 64, the interval is min(sqrt(T*K), 64), floored at K.

    For float32 compute:
        Tighter bound to prevent alpha drift exceeding float32 precision.
        When K >= 32, the interval equals K.
        When K < 32, the interval is min(sqrt(T*K), 32), floored at K.

    Args:
        T: Maximum sequence length.
        K: Maximum segment duration.
        compute_dtype: Internal compute precision, ``"float64"`` or ``"float32"``.
    """
    optimal = int(math.sqrt(T * K))
    if compute_dtype == "float32":
        max_interval = max(K, 32)
    else:
        max_interval = max(K, 64)
    return max(K, min(optimal, max_interval))


# =============================================================================
# K=1 Linear CRF Fast Path
# =============================================================================


def linear_crf_forward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    lengths: torch.Tensor,
    duration_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Optimized K=1 (linear CRF) forward pass.

    For K=1, the semi-CRF reduces to a standard linear-chain CRF. This
    implementation eliminates all K-related overhead (ring buffers, duration
    loops, checkpointing) for maximum performance.

    Complexity: O(T * C²) time, O(batch * C) memory.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        lengths: Sequence lengths of shape (batch,).
        duration_bias: Optional duration bias of shape (K, C). Only index 0 is used.

    Returns:
        Tensor: Log partition function of shape (batch,).
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Initialize alpha (all labels equally likely at start)
    alpha = torch.zeros(batch, C, device=device, dtype=dtype)

    # Track final alpha for variable lengths
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    # Optional duration bias (only index 0 for K=1)
    dur_bias = duration_bias[0] if duration_bias is not None else 0.0

    for t in range(1, T + 1):
        # Emission score: cumsum difference for this timestep
        emission = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + dur_bias  # (batch, C)

        # Linear CRF recurrence:
        # alpha[t, c_dst] = logsumexp_{c_src}(alpha[t-1, c_src] + trans[c_src, c_dst]) + emission[c_dst]
        # alpha: (batch, C_src) -> unsqueeze to (batch, C_src, 1)
        # transition: (C_src, C_dst)
        # broadcast: (batch, C_src, C_dst) -> logsumexp over C_src -> (batch, C_dst)
        alpha_new = torch.logsumexp(alpha.unsqueeze(-1) + transition, dim=-2) + emission

        # Update alpha only for active sequences
        active_mask = (t <= lengths).view(batch, 1)
        alpha = torch.where(active_mask, alpha_new, alpha)

        # Capture final alpha at sequence endpoints
        final_mask = (t == lengths).view(batch, 1)
        final_alpha = torch.where(final_mask, alpha_new, final_alpha)

    # Partition function: logsumexp over final labels
    return torch.logsumexp(final_alpha, dim=-1)


def linear_crf_backward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    lengths: torch.Tensor,
    log_Z: torch.Tensor,
    duration_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""Optimized K=1 (linear CRF) backward pass via forward-backward algorithm.

    Computes gradients for cum_scores, transition, and duration_bias using
    the forward-backward algorithm. No checkpointing needed for K=1.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        lengths: Sequence lengths of shape (batch,).
        log_Z: Log partition function of shape (batch,).
        duration_bias: Optional duration bias of shape (K, C).

    Returns:
        Tuple of (grad_cum_scores, grad_transition, grad_duration_bias).
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Optional duration bias
    dur_bias = duration_bias[0] if duration_bias is not None else 0.0

    # Forward pass: compute all alpha values
    alpha_all = torch.full((batch, T + 1, C), NEG_INF, device=device, dtype=dtype)
    alpha_all[:, 0, :] = 0.0  # Initial state

    for t in range(1, T + 1):
        emission = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + dur_bias
        alpha_new = (
            torch.logsumexp(alpha_all[:, t - 1, :].unsqueeze(-1) + transition, dim=-2) + emission
        )
        active_mask = (t <= lengths).view(batch, 1)
        alpha_all[:, t, :] = torch.where(active_mask, alpha_new, alpha_all[:, t - 1, :])

    # Backward pass: compute all beta values
    beta_all = torch.full((batch, T + 1, C), NEG_INF, device=device, dtype=dtype)

    # Initialize beta at final positions
    for b in range(batch):
        beta_all[b, lengths[b].item(), :] = 0.0

    for t in range(T - 1, -1, -1):
        # For sequences where t < length, compute beta[t] from beta[t+1]
        emission_next = cum_scores[:, t + 1, :] - cum_scores[:, t, :] + dur_bias

        # beta[t, c_src] = logsumexp_{c_dst}(trans[c_src, c_dst] + emission[c_dst] + beta[t+1, c_dst])
        # transition: (C_src, C_dst)
        # emission_next + beta[t+1]: (batch, C_dst) -> unsqueeze to (batch, 1, C_dst)
        # broadcast: (batch, C_src, C_dst) -> logsumexp over C_dst -> (batch, C_src)
        beta_new = torch.logsumexp(
            transition.unsqueeze(0) + (emission_next + beta_all[:, t + 1, :]).unsqueeze(-2),
            dim=-1,
        )

        active_mask = (t < lengths).view(batch, 1)
        beta_all[:, t, :] = torch.where(active_mask, beta_new, beta_all[:, t, :])

    # Compute gradients via marginals
    grad_cum_scores = torch.zeros_like(cum_scores)
    grad_transition = torch.zeros(batch, C, C, device=device, dtype=dtype)
    grad_duration_bias = (
        torch.zeros(batch, C, device=device, dtype=dtype) if duration_bias is not None else None
    )

    for t in range(1, T + 1):
        active_mask = (t <= lengths).view(batch, 1, 1)
        emission = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + dur_bias

        # Edge marginal: P(c_src -> c_dst at time t)
        # log_marginal[b, c_src, c_dst] = alpha[t-1, c_src] + trans[c_src, c_dst] + emission[c_dst] + beta[t, c_dst] - log_Z
        log_marginal = (
            alpha_all[:, t - 1, :].unsqueeze(-1)  # (batch, C_src, 1)
            + transition.unsqueeze(0)  # (1, C_src, C_dst)
            + (emission + beta_all[:, t, :]).unsqueeze(-2)  # (batch, 1, C_dst)
            - log_Z.view(batch, 1, 1)
        )
        log_marginal = torch.clamp(log_marginal, min=-80.0, max=80.0)
        marginal = torch.exp(log_marginal)
        marginal = torch.where(active_mask, marginal, torch.zeros_like(marginal))

        # Gradient contributions
        marginal_sum_dst = marginal.sum(dim=-2)  # (batch, C_dst)

        # grad_cum_scores: +1 at position t, -1 at position t-1
        grad_cum_scores[:, t, :] += marginal_sum_dst
        grad_cum_scores[:, t - 1, :] -= marginal_sum_dst

        # grad_transition: sum over batch of marginals
        grad_transition += marginal

        # grad_duration_bias: same as marginal_sum_dst
        if grad_duration_bias is not None:
            grad_duration_bias += marginal_sum_dst

    # grad_duration_bias needs to be (batch, K, C) format with K=1
    if duration_bias is not None:
        grad_duration_bias = grad_duration_bias.unsqueeze(1)  # (batch, 1, C)

    return grad_cum_scores, grad_transition, grad_duration_bias


def linear_crf_viterbi_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    lengths: torch.Tensor,
    duration_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Optimized K=1 (linear CRF) Viterbi decoding.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        lengths: Sequence lengths of shape (batch,).
        duration_bias: Optional duration bias of shape (K, C).

    Returns:
        Tuple of (viterbi_scores, best_paths) where best_paths has shape (batch, T).
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Optional duration bias
    dur_bias = duration_bias[0] if duration_bias is not None else 0.0

    # Viterbi forward pass with backpointers
    alpha = torch.zeros(batch, C, device=device, dtype=dtype)
    backpointers = torch.zeros(batch, T, C, dtype=torch.long, device=device)

    for t in range(1, T + 1):
        emission = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + dur_bias

        # Viterbi: max instead of logsumexp
        # scores[c_src, c_dst] = alpha[c_src] + trans[c_src, c_dst]
        scores = alpha.unsqueeze(-1) + transition  # (batch, C_src, C_dst)
        alpha_new, bp = torch.max(scores, dim=-2)  # (batch, C_dst)
        alpha_new = alpha_new + emission

        active_mask = (t <= lengths).view(batch, 1)
        alpha = torch.where(active_mask, alpha_new, alpha)
        backpointers[:, t - 1, :] = bp

    # alpha[b] already holds the correct terminal state for sequence b.
    # The active_mask in the loop above freezes each row once t > lengths[b],
    # so alpha[b] = alpha at timestep lengths[b] for every batch element.
    viterbi_scores, best_final_labels = torch.max(alpha, dim=-1)

    # Traceback
    best_paths = torch.zeros(batch, T, dtype=torch.long, device=device)
    for b in range(batch):
        L = lengths[b].item()
        current_label = best_final_labels[b].item()
        for t in range(L - 1, -1, -1):
            best_paths[b, t] = current_label
            if t > 0:
                current_label = backpointers[b, t, current_label].item()

    return viterbi_scores, best_paths


# =============================================================================
# K=2 Specialized Path
# =============================================================================


def semi_crf_k2_forward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    r"""Optimized K=2 semi-CRF forward pass.

    For K=2, segments can have duration 1 or 2. This implementation uses
    explicit 2-step history instead of a ring buffer, eliminating modular
    arithmetic and checkpoint overhead.

    Complexity: O(T * C²) time, O(batch * C) memory.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        duration_bias: Duration bias of shape (K, C) where K=2.
        lengths: Sequence lengths of shape (batch,).

    Returns:
        Tensor: Log partition function of shape (batch,).
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Initialize alpha history (explicit 2-step, no ring buffer)
    alpha_prev1 = torch.zeros(batch, C, device=device, dtype=dtype)  # alpha[t-1]
    alpha_prev2 = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)  # alpha[t-2]

    # Track final alpha for variable lengths
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    for t in range(1, T + 1):
        # Duration k=1: segment from t-1 to t
        emission_k1 = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + duration_bias[0]
        score_k1 = torch.logsumexp(alpha_prev1.unsqueeze(-1) + transition, dim=-2) + emission_k1

        # Duration k=2: segment from t-2 to t (if t >= 2)
        if t >= 2:
            emission_k2 = cum_scores[:, t, :] - cum_scores[:, t - 2, :] + duration_bias[1]
            score_k2 = torch.logsumexp(alpha_prev2.unsqueeze(-1) + transition, dim=-2) + emission_k2
            # Combine scores from both durations
            alpha_new = torch.logsumexp(torch.stack([score_k1, score_k2], dim=-1), dim=-1)
        else:
            alpha_new = score_k1

        # Update history (shift)
        alpha_prev2 = alpha_prev1.clone()

        # Update alpha_prev1 only for active sequences
        active_mask = (t <= lengths).view(batch, 1)
        alpha_prev1 = torch.where(active_mask, alpha_new, alpha_prev1)

        # Capture final alpha at sequence endpoints
        final_mask = (t == lengths).view(batch, 1)
        final_alpha = torch.where(final_mask, alpha_new, final_alpha)

    return torch.logsumexp(final_alpha, dim=-1)


def semi_crf_k2_backward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    log_Z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Optimized K=2 semi-CRF backward pass via forward-backward algorithm.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        duration_bias: Duration bias of shape (K, C) where K=2.
        lengths: Sequence lengths of shape (batch,).
        log_Z: Log partition function of shape (batch,).

    Returns:
        Tuple of (grad_cum_scores, grad_transition, grad_duration_bias).
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Forward pass: store all alpha values for backward
    alpha_all = torch.full((batch, T + 1, C), NEG_INF, device=device, dtype=dtype)
    alpha_all[:, 0, :] = 0.0

    for t in range(1, T + 1):
        alpha_prev1 = alpha_all[:, t - 1, :]
        alpha_prev2 = (
            alpha_all[:, t - 2, :]
            if t >= 2
            else torch.full((batch, C), NEG_INF, device=device, dtype=dtype)
        )

        emission_k1 = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + duration_bias[0]
        score_k1 = torch.logsumexp(alpha_prev1.unsqueeze(-1) + transition, dim=-2) + emission_k1

        if t >= 2:
            emission_k2 = cum_scores[:, t, :] - cum_scores[:, t - 2, :] + duration_bias[1]
            score_k2 = torch.logsumexp(alpha_prev2.unsqueeze(-1) + transition, dim=-2) + emission_k2
            alpha_new = torch.logsumexp(torch.stack([score_k1, score_k2], dim=-1), dim=-1)
        else:
            alpha_new = score_k1

        active_mask = (t <= lengths).view(batch, 1)
        alpha_all[:, t, :] = torch.where(active_mask, alpha_new, alpha_all[:, t - 1, :])

    # Backward pass: compute beta values
    beta_all = torch.full((batch, T + 1, C), NEG_INF, device=device, dtype=dtype)

    # Initialize beta at final positions
    for b in range(batch):
        beta_all[b, lengths[b].item(), :] = 0.0

    for t in range(T - 1, -1, -1):
        beta_new = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

        # Contribution from k=1 segments ending at t+1
        if t + 1 <= T:
            emission_k1 = cum_scores[:, t + 1, :] - cum_scores[:, t, :] + duration_bias[0]
            contrib_k1 = torch.logsumexp(
                transition.unsqueeze(0) + (emission_k1 + beta_all[:, t + 1, :]).unsqueeze(-2),
                dim=-1,
            )
            beta_new = torch.logsumexp(torch.stack([beta_new, contrib_k1], dim=-1), dim=-1)

        # Contribution from k=2 segments ending at t+2
        if t + 2 <= T:
            emission_k2 = cum_scores[:, t + 2, :] - cum_scores[:, t, :] + duration_bias[1]
            contrib_k2 = torch.logsumexp(
                transition.unsqueeze(0) + (emission_k2 + beta_all[:, t + 2, :]).unsqueeze(-2),
                dim=-1,
            )
            beta_new = torch.logsumexp(torch.stack([beta_new, contrib_k2], dim=-1), dim=-1)

        active_mask = (t < lengths).view(batch, 1)
        beta_all[:, t, :] = torch.where(active_mask, beta_new, beta_all[:, t, :])

    # Compute gradients via marginals
    grad_cum_scores = torch.zeros_like(cum_scores)
    grad_transition = torch.zeros(batch, C, C, device=device, dtype=dtype)
    grad_duration_bias = torch.zeros(batch, 2, C, device=device, dtype=dtype)

    for t in range(1, T + 1):
        # k=1 edges: segments from t-1 to t
        active_mask_k1 = (t <= lengths).view(batch, 1, 1)
        emission_k1 = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + duration_bias[0]

        log_marginal_k1 = (
            alpha_all[:, t - 1, :].unsqueeze(-1)
            + transition.unsqueeze(0)
            + (emission_k1 + beta_all[:, t, :]).unsqueeze(-2)
            - log_Z.view(batch, 1, 1)
        )
        log_marginal_k1 = torch.clamp(log_marginal_k1, min=-80.0, max=80.0)
        marginal_k1 = torch.exp(log_marginal_k1)
        marginal_k1 = torch.where(active_mask_k1, marginal_k1, torch.zeros_like(marginal_k1))

        marginal_k1_sum = marginal_k1.sum(dim=-2)  # (batch, C_dst)
        grad_cum_scores[:, t, :] += marginal_k1_sum
        grad_cum_scores[:, t - 1, :] -= marginal_k1_sum
        grad_transition += marginal_k1
        grad_duration_bias[:, 0, :] += marginal_k1_sum

        # k=2 edges: segments from t-2 to t (if t >= 2)
        if t >= 2:
            active_mask_k2 = (t <= lengths).view(batch, 1, 1)
            emission_k2 = cum_scores[:, t, :] - cum_scores[:, t - 2, :] + duration_bias[1]

            log_marginal_k2 = (
                alpha_all[:, t - 2, :].unsqueeze(-1)
                + transition.unsqueeze(0)
                + (emission_k2 + beta_all[:, t, :]).unsqueeze(-2)
                - log_Z.view(batch, 1, 1)
            )
            log_marginal_k2 = torch.clamp(log_marginal_k2, min=-80.0, max=80.0)
            marginal_k2 = torch.exp(log_marginal_k2)
            marginal_k2 = torch.where(active_mask_k2, marginal_k2, torch.zeros_like(marginal_k2))

            marginal_k2_sum = marginal_k2.sum(dim=-2)
            grad_cum_scores[:, t, :] += marginal_k2_sum
            grad_cum_scores[:, t - 2, :] -= marginal_k2_sum
            grad_transition += marginal_k2
            grad_duration_bias[:, 1, :] += marginal_k2_sum

    return grad_cum_scores, grad_transition, grad_duration_bias


def semi_crf_k2_viterbi_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Optimized K=2 semi-CRF Viterbi decoding.

    Args:
        cum_scores: Cumulative projected scores of shape (batch, T+1, C).
        transition: Label transition scores of shape (C, C).
        duration_bias: Duration bias of shape (K, C) where K=2.
        lengths: Sequence lengths of shape (batch,).

    Returns:
        Tuple of (viterbi_scores, best_paths, best_durations).
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Viterbi forward with backpointers
    alpha_prev1 = torch.zeros(batch, C, device=device, dtype=dtype)
    alpha_prev2 = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    # Backpointers: (batch, T, C) for best (k, c_src)
    bp_k = torch.zeros(batch, T, C, dtype=torch.long, device=device)
    bp_c = torch.zeros(batch, T, C, dtype=torch.long, device=device)

    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    for t in range(1, T + 1):
        # k=1 scores
        emission_k1 = cum_scores[:, t, :] - cum_scores[:, t - 1, :] + duration_bias[0]
        scores_k1 = alpha_prev1.unsqueeze(-1) + transition  # (batch, C_src, C_dst)
        max_k1, argmax_k1 = torch.max(scores_k1, dim=-2)
        max_k1 = max_k1 + emission_k1

        if t >= 2:
            # k=2 scores
            emission_k2 = cum_scores[:, t, :] - cum_scores[:, t - 2, :] + duration_bias[1]
            scores_k2 = alpha_prev2.unsqueeze(-1) + transition
            max_k2, argmax_k2 = torch.max(scores_k2, dim=-2)
            max_k2 = max_k2 + emission_k2

            # Compare k=1 vs k=2
            k1_better = max_k1 >= max_k2
            alpha_new = torch.where(k1_better, max_k1, max_k2)
            best_k = torch.where(
                k1_better, torch.ones_like(argmax_k1), torch.full_like(argmax_k1, 2)
            )
            best_c = torch.where(k1_better, argmax_k1, argmax_k2)
        else:
            alpha_new = max_k1
            best_k = torch.ones(batch, C, dtype=torch.long, device=device)
            best_c = argmax_k1

        bp_k[:, t - 1, :] = best_k
        bp_c[:, t - 1, :] = best_c

        # Update history
        alpha_prev2 = alpha_prev1.clone()
        active_mask = (t <= lengths).view(batch, 1)
        alpha_prev1 = torch.where(active_mask, alpha_new, alpha_prev1)

        final_mask = (t == lengths).view(batch, 1)
        final_alpha = torch.where(final_mask, alpha_new, final_alpha)

    # Get best final scores and labels
    viterbi_scores, best_final_labels = torch.max(final_alpha, dim=-1)

    # Traceback
    best_paths = torch.zeros(batch, T, dtype=torch.long, device=device)
    best_durations = torch.zeros(batch, T, dtype=torch.long, device=device)

    for b in range(batch):
        L = lengths[b].item()
        t = L - 1
        current_label = best_final_labels[b].item()

        while t >= 0:
            best_paths[b, t] = current_label
            k = bp_k[b, t, current_label].item()
            best_durations[b, t] = k
            prev_label = bp_c[b, t, current_label].item()

            # Move back by duration k
            t = t - k
            current_label = prev_label

    return viterbi_scores, best_paths, best_durations


def compute_edge_block_streaming(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    t: int,
    k: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""compute_edge_block_streaming(cum_scores, transition, duration_bias, t, k, proj_start=None, proj_end=None) -> Tensor

    Compute the edge block for segment ``(t, t+k)`` via prefix-sum on the fly.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)` or
            :math:`(K, C, C)` for duration-dependent transitions.
        duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
        t (int): Segment start position (0-indexed).
        k (int): Segment duration (1-indexed).
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``

    Returns:
        Tensor: Edge block of shape :math:`(\text{batch}, C_\text{dst}, C_\text{src})`.
    """
    # Content score via cumsum difference: (batch, C)
    content_score = cum_scores[:, t + k, :] - cum_scores[:, t, :]

    # Add duration bias: duration k uses index k-1
    dur_idx = k - 1
    segment_score = content_score + duration_bias[dur_idx]

    # Add boundary scores if provided
    if proj_start is not None:
        segment_score = segment_score + proj_start[:, t, :]
    if proj_end is not None:
        segment_score = segment_score + proj_end[:, t + k - 1, :]

    # Build edge block: segment_score[c_dest] + transition[c_src, c_dest]
    # segment_score: (batch, C) -> unsqueeze to (batch, C, 1) for c_dest
    # transition: (C_src, C_dest) -> transpose to (C_dest, C_src) -> (1, C_dest, C_src)
    # Result: (batch, C_dest, C_src)
    if transition.ndim == 2:
        # Static transitions: (C, C)
        trans_k = transition
    else:
        # Duration-dependent transitions: (K, C, C) - duration k uses index k-1
        trans_k = transition[k - 1]

    edge_block = segment_score.unsqueeze(-1) + trans_k.T.unsqueeze(0)

    return edge_block


def _recompute_alpha_segment(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    checkpoint: torch.Tensor,
    seg_start: int,
    seg_end: int,
    K: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Recompute alpha values for one checkpoint segment from a saved ring checkpoint.

    .. note::

        **Not yet implemented — stub for planned refactoring.**

        ``semi_crf_streaming_backward_pytorch`` (segment loop ~lines 1046-1105)
        and ``semi_crf_streaming_marginals_pytorch`` (segment loop ~lines 1340-1390)
        both implement this identical operation: restore the ring buffer from
        ``checkpoint``, then replay the forward recurrence from ``seg_start+1``
        to ``seg_end-1``, writing each recomputed alpha into a local
        ``alpha_segment`` buffer.  The two copies differ only in their outer
        context (backward accumulates gradients; marginals accumulates boundary
        probabilities).

        Extracting this helper eliminates the duplicated ~45-line recompute
        block. Before implementing, run ``scripts/validate_correctness.py``
        before and after to confirm numerical identity.

    Args:
        cum_scores (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)` or
            :math:`(K, C, C)`.
        duration_bias (Tensor): Duration bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        checkpoint (Tensor): Ring buffer state at ``seg_start`` of shape
            :math:`(\text{batch}, K, C)`, i.e. ``ring_checkpoints[:, ckpt_idx]``.
        seg_start (int): First position of the segment.
        seg_end (int): One past the last position of the segment.
        K (int): Maximum segment duration.
        proj_start (Tensor, optional): Start boundary scores. Default: ``None``
        proj_end (Tensor, optional): End boundary scores. Default: ``None``

    Returns:
        Tensor: ``alpha_segment`` of shape
        :math:`(\text{batch}, \text{seg\_end} - \text{seg\_start}, C)`,
        where ``alpha_segment[:, 0, :]`` is ``checkpoint[:, seg_start % K, :]``
        and ``alpha_segment[:, i, :]`` is the recomputed alpha at
        ``seg_start + i``.
    """
    raise NotImplementedError("_recompute_alpha_segment is a planned extraction — see docstring.")


def semi_crf_streaming_forward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
    checkpoint_interval: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    r"""semi_crf_streaming_forward_pytorch(cum_scores, transition, duration_bias, lengths, K, semiring="log", proj_start=None, proj_end=None, checkpoint_interval=None) -> tuple[Tensor, Tensor, int, Tensor]

    Streaming Semi-CRF forward pass with :math:`O(KC)` ring buffer memory.

    At each checkpoint boundary, alpha values are normalized by subtracting
    their max, and the cumulative shift is stored in ``log_norm_checkpoints``
    for use in the backward pass. This prevents alpha values from growing
    unbounded at extreme sequence lengths (T>100K).

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)` or
            :math:`(K, C, C)` for duration-dependent transitions.
        duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        semiring (str, optional): ``"log"`` or ``"max"``. Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        checkpoint_interval (int, optional): Ring buffer save interval.
            If ``None``, computed from :func:`_compute_checkpoint_interval`.
            Default: ``None``

    Returns:
        tuple[Tensor, Tensor, int, Tensor]:
            - **partition** (Tensor): Log partition values of shape :math:`(\text{batch},)`.
            - **ring_checkpoints** (Tensor): Saved ring states of shape
              :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
            - **checkpoint_interval** (int): Interval used.
            - **log_norm_checkpoints** (Tensor): Cumulative normalization factors
              of shape :math:`(\text{batch}, \text{num\_ckpts})`.
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Warn if cum_scores appears non-zero-centered (only during eager execution)
    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        endpoint_magnitude = cum_scores[:, -1, :].abs().mean()
        if endpoint_magnitude > 1000:
            warnings.warn(
                f"cum_scores endpoint magnitude {endpoint_magnitude.item():.0f} suggests "
                "non-zero-centered input. This may cause precision loss at T>100K. "
                "Zero-center before cumsum: projected = projected - projected.mean(dim=1, keepdim=True)",
                stacklevel=2,
            )

    # Determine checkpoint interval
    if checkpoint_interval is None:
        checkpoint_interval = _compute_checkpoint_interval(T, K)
    else:
        checkpoint_interval = max(checkpoint_interval, K)

    num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

    # Allocate ring checkpoint storage: (batch, num_checkpoints, K, C)
    ring_checkpoints = torch.full(
        (batch, num_checkpoints, K, C), NEG_INF, device=device, dtype=dtype
    )

    # Log normalization checkpoints for numerical stability at extreme T
    # Stores cumulative log normalization factor at each checkpoint boundary
    log_norm_checkpoints = torch.zeros((batch, num_checkpoints), device=device, dtype=dtype)

    # Track cumulative normalization factor per batch element
    accum_log_norm = torch.zeros(batch, device=device, dtype=dtype)

    # Ring buffer for alpha values: (batch, K, C)
    alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
    alpha_ring[:, 0, :] = 0.0  # Initial: all labels equally likely

    # Store initial ring buffer state (checkpoint 0)
    ring_checkpoints[:, 0, :, :] = alpha_ring

    # Track final alpha for variable lengths
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    # Main forward loop
    for t in range(1, T + 1):
        # Include t == lengths to compute alpha at final position
        active_mask = t <= lengths

        # Number of valid durations at this position: k = 1, 2, ..., min(K, t)
        k_eff = min(K, t)

        scores_all = []
        for k in range(1, k_eff + 1):
            start = t - k

            # Get alpha[start] from ring buffer
            ring_idx = start % K
            alpha_prev = alpha_ring[:, ring_idx, :]  # (batch, C_src)

            # Compute edge block on-the-fly (streaming)
            edge_block = compute_edge_block_streaming(
                cum_scores, transition, duration_bias, start, k, proj_start, proj_end
            )  # (batch, C_dest, C_src)

            # scores[c_dest, c_src] = alpha_prev[c_src] + edge[c_dest, c_src]
            scores = alpha_prev.unsqueeze(-2) + edge_block  # (batch, C_dest, C_src)
            scores_all.append(scores)

        # Stack: (batch, k_eff, C_dest, C_src)
        scores_stacked = torch.stack(scores_all, dim=1)

        if semiring == "log":
            # logsumexp over (k, c_src) -> (batch, C_dest)
            scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
            alpha_t = torch.logsumexp(scores_over_src, dim=1)
        else:  # max
            scores_over_src = torch.max(scores_stacked, dim=-1)[0]
            alpha_t = torch.max(scores_over_src, dim=1)[0]

        # Update ring buffer (only for active sequences)
        ring_idx_t = t % K
        alpha_ring[:, ring_idx_t, :] = torch.where(
            active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
        )

        # Save checkpoint and apply normalization at interval boundaries
        # Flash Attention-style: normalize to prevent unbounded growth at extreme T
        if t % checkpoint_interval == 0:
            ckpt_idx = t // checkpoint_interval
            if ckpt_idx < num_checkpoints:
                # ===== NORMALIZATION STEP (Flash Attention pattern) =====
                # 1. Find max alpha value for normalization (over all classes)
                #    Use alpha_t (just computed) since it represents current state
                alpha_for_norm = torch.where(
                    active_mask.view(batch, 1),
                    alpha_t,
                    torch.full_like(alpha_t, NEG_INF),
                )
                shift = alpha_for_norm.max(dim=-1, keepdim=True)[0]  # (batch, 1)

                # Guard against all-NEG_INF case (shouldn't happen, but be safe)
                shift = torch.where(shift < NEG_INF + 1.0, torch.zeros_like(shift), shift)

                # 2. Update cumulative normalization factor (only for active sequences)
                accum_log_norm = torch.where(
                    active_mask, accum_log_norm + shift.squeeze(-1), accum_log_norm
                )

                # 3. Normalize alpha_t register
                #    CRITICAL: Must update alpha_t before final_alpha capture
                #    This ensures consistency if seq_len falls on checkpoint boundary
                alpha_t = torch.where(active_mask.view(batch, 1), alpha_t - shift, alpha_t)

                # 4. Normalize ALL K slots in ring buffer
                #    The ring buffer is used for K more iterations after checkpoint,
                #    so all slots must be shifted to maintain consistency
                for k_slot in range(K):
                    alpha_ring[:, k_slot, :] = torch.where(
                        active_mask.view(batch, 1),
                        alpha_ring[:, k_slot, :] - shift,
                        alpha_ring[:, k_slot, :],
                    )

                # 5. Re-update alpha_ring at current slot with normalized alpha_t
                #    (since we normalized alpha_t after storing it to ring buffer)
                alpha_ring[:, ring_idx_t, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
                )

                # 6. Save normalized ring buffer to checkpoint
                for k_slot in range(K):
                    ring_checkpoints[:, ckpt_idx, k_slot, :] = torch.where(
                        active_mask.view(batch, 1),
                        alpha_ring[:, k_slot, :],
                        ring_checkpoints[:, ckpt_idx, k_slot, :],
                    )

                # 7. Save cumulative log normalization factor
                log_norm_checkpoints[:, ckpt_idx] = torch.where(
                    active_mask, accum_log_norm, log_norm_checkpoints[:, ckpt_idx]
                )

        # Track final alpha for sequences at their final position (t == lengths)
        # At iteration t, alpha_t represents segments ending at position t-1
        # For sequence of length L, we need alpha at t=L (segments ending at L-1)
        is_final = t == lengths
        if is_final.any():
            final_alpha = torch.where(is_final.view(batch, 1), alpha_t, final_alpha)

    # Compute partition function
    # Add back cumulative normalization to get true partition function
    if semiring == "log":
        raw_partition = torch.logsumexp(final_alpha, dim=-1)
        partition = raw_partition + accum_log_norm
    else:  # max
        # For max semiring, normalization doesn't affect the argmax result
        # but we still need to add back the shift for correct score magnitude
        partition = torch.max(final_alpha, dim=-1)[0] + accum_log_norm

    return partition, ring_checkpoints, checkpoint_interval, log_norm_checkpoints


def semi_crf_streaming_viterbi_with_backpointers(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Viterbi forward pass with backpointer tracking for traceback.

    Computes the maximum-scoring path using the max semiring and tracks
    backpointers for efficient O(T) traceback (instead of O(T*K) recomputation).

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Label transition scores of shape :math:`(C, C)` or
            :math:`(K, C, C)` for duration-dependent transitions.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        proj_start (Tensor, optional): Start boundary scores.
        proj_end (Tensor, optional): End boundary scores.

    Returns:
        tuple: (viterbi_scores, bp_k, bp_c, final_labels) where:
            - viterbi_scores: Best scores of shape (batch,)
            - bp_k: Backpointer durations of shape (batch, T, C)
            - bp_c: Backpointer source labels of shape (batch, T, C)
            - final_labels: Best final label for each batch of shape (batch,)
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Ring buffer for alpha values: (batch, K, C)
    alpha_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)
    alpha_ring[:, 0, :] = 0.0  # Initial: all labels equally likely

    # Backpointer storage: (batch, T, C) - stores best (k, c_src) for each (t, c_dest)
    bp_k = torch.zeros((batch, T, C), dtype=torch.long, device=device)
    bp_c = torch.zeros((batch, T, C), dtype=torch.long, device=device)

    # Track final alpha for variable lengths
    final_alpha = torch.full((batch, C), NEG_INF, device=device, dtype=dtype)

    # Main forward loop
    for t in range(1, T + 1):
        active_mask = t <= lengths
        # Number of valid durations at this position: k = 1, 2, ..., min(K, t)
        k_eff = min(K, t)

        # Collect scores for all valid durations
        scores_list = []
        k_indices = []

        for k in range(1, k_eff + 1):
            start = t - k
            ring_idx = start % K
            alpha_prev = alpha_ring[:, ring_idx, :]  # (batch, C_src)

            # Compute edge block on-the-fly
            edge_block = compute_edge_block_streaming(
                cum_scores, transition, duration_bias, start, k, proj_start, proj_end
            )  # (batch, C_dest, C_src)

            # scores[c_dest, c_src] = alpha_prev[c_src] + edge[c_dest, c_src]
            scores = alpha_prev.unsqueeze(-2) + edge_block  # (batch, C_dest, C_src)
            scores_list.append(scores)
            k_indices.append(k)

        # Stack: (batch, num_k, C_dest, C_src)
        scores_stacked = torch.stack(scores_list, dim=1)

        # First: max over c_src -> (batch, num_k, C_dest) + argmax
        scores_over_src, best_c_src_per_k = torch.max(scores_stacked, dim=-1)

        # Second: max over k -> (batch, C_dest) + argmax
        alpha_t, best_k_idx = torch.max(scores_over_src, dim=1)

        # Convert k_idx to actual k value and get corresponding c_src
        # best_k_idx is index into k_indices list (0 to num_k-1)
        # We need to gather the c_src from best_c_src_per_k using best_k_idx
        k_values = torch.tensor(k_indices, device=device, dtype=torch.long)
        best_k = k_values[best_k_idx]  # (batch, C_dest)

        # Gather best c_src: need to index best_c_src_per_k[b, best_k_idx[b, c], c]
        # best_c_src_per_k: (batch, num_k, C)
        # best_k_idx: (batch, C)
        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, C)
        c_idx = torch.arange(C, device=device).unsqueeze(0).expand(batch, -1)
        best_c_src = best_c_src_per_k[batch_idx, best_k_idx, c_idx]  # (batch, C)

        # Store backpointers. bp arrays are 0-indexed (positions 0..T-1), so
        # loop variable t maps to bp index t-1.
        bp_k[:, t - 1, :] = torch.where(active_mask.view(batch, 1), best_k, bp_k[:, t - 1, :])
        bp_c[:, t - 1, :] = torch.where(active_mask.view(batch, 1), best_c_src, bp_c[:, t - 1, :])

        # Update ring buffer
        ring_idx_t = t % K
        alpha_ring[:, ring_idx_t, :] = torch.where(
            active_mask.view(batch, 1), alpha_t, alpha_ring[:, ring_idx_t, :]
        )

        # Track final alpha
        is_final = t == lengths
        if is_final.any():
            final_alpha = torch.where(is_final.view(batch, 1), alpha_t, final_alpha)

    # Compute Viterbi scores and best final labels
    viterbi_scores, final_labels = torch.max(final_alpha, dim=-1)

    return viterbi_scores, bp_k, bp_c, final_labels


def semi_crf_streaming_backward_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    log_Z: torch.Tensor,
    ring_checkpoints: torch.Tensor,
    log_norm_checkpoints: torch.Tensor,
    checkpoint_interval: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
]:
    r"""semi_crf_streaming_backward_pytorch(cum_scores, transition, duration_bias, lengths, K, log_Z, ring_checkpoints, log_norm_checkpoints, checkpoint_interval, semiring="log", proj_start=None, proj_end=None) -> tuple

    Streaming Semi-CRF backward pass via the forward-backward algorithm with
    checkpointing. Alpha values are recomputed from ring buffer checkpoints
    rather than stored, keeping memory at :math:`O(KC)`.

    Marginals use relative log-marginals with global scaling to prevent overflow
    when ``alpha + beta >> log_Z`` at large T.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)` or
            :math:`(K, C, C)`.
        duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        log_Z (Tensor): Partition values from forward of shape :math:`(\text{batch},)`.
        ring_checkpoints (Tensor): Saved ring states from forward of shape
            :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
        log_norm_checkpoints (Tensor): Cumulative normalization factors of shape
            :math:`(\text{batch}, \text{num\_ckpts})`.
        checkpoint_interval (int): Interval used during forward pass.
        semiring (str, optional): ``"log"`` only; ``"max"`` is not supported.
            Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores. Default: ``None``
        proj_end (Tensor, optional): End boundary scores. Default: ``None``

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor or None, Tensor or None]:
            ``(grad_cum_scores, grad_transition, grad_duration_bias,
            grad_proj_start, grad_proj_end)``
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    effective_interval = max(checkpoint_interval, K)

    # Initialize gradient accumulators
    # Note: grad_transition and grad_duration_bias are per-batch to allow proper
    # weighting by grad_output in the autograd wrapper. The caller (autograd.py)
    # will apply einsum to compute: grad = Σ_b[grad_output[b] * grad_per_batch[b]]
    grad_cum_scores = torch.zeros_like(cum_scores)
    if transition.ndim == 2:
        grad_transition = torch.zeros(batch, C, C, device=device, dtype=dtype)
    else:
        grad_transition = torch.zeros(batch, K, C, C, device=device, dtype=dtype)
    grad_duration_bias = torch.zeros(batch, K, C, device=device, dtype=dtype)
    grad_proj_start = torch.zeros_like(proj_start) if proj_start is not None else None
    grad_proj_end = torch.zeros_like(proj_end) if proj_end is not None else None

    # Segment buffer for alpha values
    segment_size = effective_interval + K
    alpha_segment = torch.full((batch, segment_size, C), NEG_INF, device=device, dtype=dtype)

    # Beta ring buffer
    beta_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)

    # Initialize beta at final positions
    # After the forward fix, final alpha is captured at t=lengths, so beta
    # should be initialized at position lengths (not lengths-1)
    for b in range(batch):
        final_ring_idx = lengths[b].item() % K
        beta_ring[b, final_ring_idx, :] = 0.0

    num_checkpoints = ring_checkpoints.shape[1]

    # Process segments in reverse order
    for ckpt_idx in range(num_checkpoints - 1, -1, -1):
        seg_start = ckpt_idx * checkpoint_interval
        seg_end = min((ckpt_idx + 1) * checkpoint_interval, T)

        # Load cumulative log normalization factor for this checkpoint
        # This is used to restore proper scale when computing marginals
        log_norm_at_ckpt = log_norm_checkpoints[:, ckpt_idx]  # (batch,)

        alpha_segment.fill_(NEG_INF)
        alpha_ring = ring_checkpoints[:, ckpt_idx, :, :].clone()

        # Store alpha[seg_start] at local position 0
        alpha_segment[:, 0, :] = alpha_ring[:, seg_start % K, :]

        # Recompute alpha for positions seg_start+1 to seg_end-1
        for t in range(seg_start + 1, seg_end):
            active_mask = t < lengths

            # Number of valid durations at this position: k = 1, 2, ..., min(K, t)
            k_eff = min(K, t)
            scores_all = []

            for k in range(1, k_eff + 1):
                start = t - k
                ring_idx = start % K
                alpha_prev = alpha_ring[:, ring_idx, :]

                # Compute edge on-the-fly
                edge_block = compute_edge_block_streaming(
                    cum_scores, transition, duration_bias, start, k, proj_start, proj_end
                )

                scores = alpha_prev.unsqueeze(-2) + edge_block
                scores_all.append(scores)

            if scores_all:
                scores_stacked = torch.stack(scores_all, dim=1)
                if semiring == "log":
                    scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
                    alpha_t = torch.logsumexp(scores_over_src, dim=1)
                else:
                    scores_over_src = torch.max(scores_stacked, dim=-1)[0]
                    alpha_t = torch.max(scores_over_src, dim=1)[0]

                alpha_ring[:, t % K, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_ring[:, t % K, :]
                )

                # Store in segment buffer
                local_t = t - seg_start
                alpha_segment[:, local_t, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_segment[:, local_t, :]
                )

        # Compute beta backward and gradients
        for t in range(seg_end - 1, seg_start - 1, -1):
            local_t = t - seg_start
            alpha_t = alpha_segment[:, local_t, :]

            # Active if we can start a segment at position t
            # (need at least one position for segment to cover)
            active_mask = t < lengths
            if not active_mask.any():
                continue

            # Maximum duration: segments can end at position lengths (using cum_scores[:, lengths, :])
            # k = 1, 2, ..., min(K, T - t)
            max_k = min(K, T - t)
            new_beta_scores = []

            for k in range(1, max_k + 1):
                end_pos = t + k
                # Include segments where end_pos == lengths (covering positions t to lengths-1)
                valid_mask = (end_pos <= lengths) & active_mask

                if not valid_mask.any():
                    continue

                ring_k_idx = end_pos % K
                beta_next = beta_ring[:, ring_k_idx, :]

                # Compute edge on-the-fly
                edge_block = compute_edge_block_streaming(
                    cum_scores, transition, duration_bias, t, k, proj_start, proj_end
                )

                # === Gradient computation ===
                # Use relative log-marginal computation for numerical stability at
                # extreme sequence lengths. Following Flash Attention pattern:
                #   1. Compute log_joint without normalizing by log_Z
                #   2. Subtract local_ref (max over tile) to keep exp bounded
                #   3. Apply scale factor exp(local_ref + log_norm_at_ckpt - log_Z) separately
                # This prevents overflow when alpha + beta >> log_Z at scale.
                #
                # At checkpoint boundaries, alpha values were normalized (max ~10-50),
                # but log_Z is the true partition (~250k at T=100k). log_norm_at_ckpt
                # contains the cumulative shift that bridges this gap.

                # Step 1: Clamp inputs for safety (values should be bounded due to normalization)
                alpha_t_safe = torch.clamp(alpha_t, min=-1e6, max=1e6)
                beta_next_safe = torch.clamp(beta_next, min=-1e6, max=1e6)
                edge_block_safe = torch.clamp(edge_block, min=-1e6, max=1e6)

                # Step 2: Compute log_joint (without log_Z subtraction)
                log_joint = (
                    alpha_t_safe.unsqueeze(-2)  # (batch, 1, C_src)
                    + edge_block_safe  # (batch, C_dest, C_src)
                    + beta_next_safe.unsqueeze(-1)  # (batch, C_dest, 1)
                )

                # Step 3: Find local reference (max over valid entries)
                log_joint_masked = torch.where(
                    valid_mask.view(batch, 1, 1),
                    log_joint,
                    torch.full_like(log_joint, NEG_INF),
                )
                # Max over (C_dest, C_src) dimensions -> (batch,)
                local_ref = log_joint_masked.amax(dim=(-2, -1))

                # Guard against all-NEG_INF case (shouldn't happen, but be safe)
                is_local_ref_neginf = local_ref < NEG_INF + 1.0
                local_ref_safe = torch.where(
                    is_local_ref_neginf, torch.zeros_like(local_ref), local_ref
                )

                # Step 4: Compute relative log-marginal (bounded in (-inf, 0])
                log_marginal_rel = log_joint - local_ref_safe.view(batch, 1, 1)

                # Step 5: Compute unnormalized marginal (bounded in (0, 1])
                marginal_unnorm = torch.exp(log_marginal_rel)

                # Step 6: CRITICAL - Compute scale factor using log_norm_at_ckpt
                # This bridges the gap between normalized alpha and full log_Z
                #
                # Example at t=50,000 (midpoint of T=100k):
                #   normalized_alpha ≈ 0 to 50    (after shift)
                #   log_norm_at_ckpt ≈ 125,000    (accumulated shift from positions 0-50k)
                #   beta ≈ 125,000                (backward mass from future positions)
                #   log_Z ≈ 250,000               (true partition)
                #
                #   local_ref ≈ 0 + 0 + 125k = 125k
                #   log_scale = 125k + 125k - 250k ~ 0  [OK] (scale ~ 1)
                log_scale = local_ref_safe + log_norm_at_ckpt - log_Z

                # Step 7: Defensive clamping (scale should be <= 1 for valid marginals)
                log_scale_clamped = torch.clamp(log_scale, min=-700.0, max=0.0)
                scale = torch.exp(log_scale_clamped).view(batch, 1, 1)

                # Step 8: Final marginal = unnorm * scale
                marginal = marginal_unnorm * scale
                marginal = torch.where(
                    valid_mask.view(batch, 1, 1), marginal, torch.zeros_like(marginal)
                )

                # === Accumulate gradients ===

                # grad_cum_scores: contribution from segments
                marginal_sum_dest = marginal.sum(dim=-1)  # (batch, C_dest)

                # grad_cum_scores[end_pos, c_dest] += sum over c_src of marginal[c_dest, c_src]
                grad_cum_scores[:, end_pos, :] += marginal_sum_dest
                # grad_cum_scores[t, c_dest] -= sum over c_src of marginal[c_dest, c_src]
                grad_cum_scores[:, t, :] -= marginal_sum_dest

                # grad_transition: per-batch accumulation (don't sum over batch)
                # marginal is (batch, C_dest, C_src), transpose to (batch, C_src, C_dest)
                # For duration-dependent transitions (K, C, C), index by dur_idx = k - 1
                # (forward pass uses transition[k - 1] for duration k)
                dur_idx = k - 1
                if transition.ndim == 2:
                    grad_transition += marginal.transpose(-1, -2)  # (batch, C_src, C_dest)
                else:
                    grad_transition[:, dur_idx] += marginal.transpose(
                        -1, -2
                    )  # (batch, C_src, C_dest) at index dur_idx

                # grad_duration_bias: per-batch accumulation
                # marginal.sum(dim=-1) sums over C_src -> (batch, C_dest)
                # Duration k uses index dur_idx = k - 1
                grad_duration_bias[:, dur_idx, :] += marginal.sum(dim=-1)  # (batch, C_dest)

                # grad_proj_start, grad_proj_end
                if grad_proj_start is not None:
                    grad_proj_start[:, t, :] += marginal_sum_dest
                if grad_proj_end is not None:
                    grad_proj_end[:, end_pos - 1, :] += marginal_sum_dest

                # === Beta contribution ===
                scores_for_beta = edge_block + beta_next.unsqueeze(-1)
                scores_for_beta = torch.where(
                    valid_mask.view(batch, 1, 1),
                    scores_for_beta,
                    torch.full_like(scores_for_beta, NEG_INF),
                )
                new_beta_scores.append(scores_for_beta)

            # Update beta ring buffer
            if new_beta_scores:
                stacked = torch.stack(new_beta_scores, dim=1)
                if semiring == "log":
                    over_dest = torch.logsumexp(stacked, dim=-2)
                    new_beta = torch.logsumexp(over_dest, dim=1)
                else:
                    over_dest = torch.max(stacked, dim=-2)[0]
                    new_beta = torch.max(over_dest, dim=1)[0]

                ring_t_idx = t % K
                beta_ring[:, ring_t_idx, :] = torch.where(
                    active_mask.view(batch, 1), new_beta, beta_ring[:, ring_t_idx, :]
                )

    return grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end


def semi_crf_streaming_marginals_pytorch(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
    checkpoint_interval: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""semi_crf_streaming_marginals_pytorch(cum_scores, transition, duration_bias, lengths, K, proj_start=None, proj_end=None, checkpoint_interval=None) -> tuple[Tensor, Tensor]

    Compute segment boundary marginals via the forward-backward algorithm.

    Runs a full forward pass to obtain the partition function and checkpoints,
    then a backward-like pass to accumulate boundary marginal probabilities.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`.
        transition (Tensor): Transition scores of shape :math:`(C, C)` or
            :math:`(K, C, C)`.
        duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        proj_start (Tensor, optional): Start boundary scores. Default: ``None``
        proj_end (Tensor, optional): End boundary scores. Default: ``None``
        checkpoint_interval (int, optional): Forwarded to the forward pass.
            Default: ``None``

    Returns:
        tuple[Tensor, Tensor]:
            - **marginals** (Tensor): Boundary marginal probabilities of shape
              :math:`(\text{batch}, T)`.
            - **log_Z** (Tensor): Log partition values of shape
              :math:`(\text{batch},)`.
    """
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    device = cum_scores.device
    dtype = cum_scores.dtype

    # Forward pass
    log_Z, ring_checkpoints, effective_interval, log_norm_checkpoints = (
        semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="log",
            proj_start=proj_start,
            proj_end=proj_end,
            checkpoint_interval=checkpoint_interval,
        )
    )

    # Backward pass to compute marginals
    effective_interval = max(effective_interval, K)

    # Initialize boundary marginals accumulator
    boundary_marginals = torch.zeros(batch, T, device=device, dtype=dtype)

    # Segment buffer for alpha values
    segment_size = effective_interval + K
    alpha_segment = torch.full((batch, segment_size, C), NEG_INF, device=device, dtype=dtype)

    # Beta ring buffer
    beta_ring = torch.full((batch, K, C), NEG_INF, device=device, dtype=dtype)

    # Initialize beta at final positions
    for b in range(batch):
        final_ring_idx = lengths[b].item() % K
        beta_ring[b, final_ring_idx, :] = 0.0

    num_checkpoints = ring_checkpoints.shape[1]

    # Process segments in reverse order
    for ckpt_idx in range(num_checkpoints - 1, -1, -1):
        seg_start = ckpt_idx * effective_interval
        seg_end = min((ckpt_idx + 1) * effective_interval, T)

        # Load cumulative log normalization factor for this checkpoint
        log_norm_at_ckpt = log_norm_checkpoints[:, ckpt_idx]  # (batch,)

        # Clear segment buffer
        alpha_segment.fill_(NEG_INF)

        # Recompute alpha from checkpoint
        alpha_ring = ring_checkpoints[:, ckpt_idx, :, :].clone()

        # Store alpha[seg_start] at local position 0
        alpha_segment[:, 0, :] = alpha_ring[:, seg_start % K, :]

        # Recompute alpha for positions seg_start+1 to seg_end-1
        for t in range(seg_start + 1, seg_end):
            active_mask = t < lengths

            # Number of valid durations at this position: k = 1, 2, ..., min(K, t)
            k_eff = min(K, t)
            scores_all = []

            for k in range(1, k_eff + 1):
                start = t - k
                ring_idx = start % K
                alpha_prev = alpha_ring[:, ring_idx, :]

                edge_block = compute_edge_block_streaming(
                    cum_scores, transition, duration_bias, start, k, proj_start, proj_end
                )

                scores = alpha_prev.unsqueeze(-2) + edge_block
                scores_all.append(scores)

            if scores_all:
                scores_stacked = torch.stack(scores_all, dim=1)
                scores_over_src = torch.logsumexp(scores_stacked, dim=-1)
                alpha_t = torch.logsumexp(scores_over_src, dim=1)

                alpha_ring[:, t % K, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_ring[:, t % K, :]
                )

                local_t = t - seg_start
                alpha_segment[:, local_t, :] = torch.where(
                    active_mask.view(batch, 1), alpha_t, alpha_segment[:, local_t, :]
                )

        # Compute beta backward and accumulate marginals
        for t in range(seg_end - 1, seg_start - 1, -1):
            local_t = t - seg_start
            alpha_t = alpha_segment[:, local_t, :]

            active_mask = t < lengths
            if not active_mask.any():
                continue

            # Maximum duration: k = 1, 2, ..., min(K, T - t)
            max_k = min(K, T - t)
            new_beta_scores = []

            for k in range(1, max_k + 1):
                end_pos = t + k
                valid_mask = (end_pos <= lengths) & active_mask

                if not valid_mask.any():
                    continue

                ring_k_idx = end_pos % K
                beta_next = beta_ring[:, ring_k_idx, :]

                edge_block = compute_edge_block_streaming(
                    cum_scores, transition, duration_bias, t, k, proj_start, proj_end
                )

                # Compute marginal probability for this segment
                # Use relative log-marginal computation for numerical stability
                # (same pattern as backward pass - see detailed comments there)

                # Step 1: Clamp inputs for safety
                alpha_t_safe = torch.clamp(alpha_t, min=-1e6, max=1e6)
                beta_next_safe = torch.clamp(beta_next, min=-1e6, max=1e6)
                edge_block_safe = torch.clamp(edge_block, min=-1e6, max=1e6)

                # Step 2: Compute log_joint (without log_Z subtraction)
                log_joint = (
                    alpha_t_safe.unsqueeze(-2)  # (batch, 1, C_src)
                    + edge_block_safe  # (batch, C_dest, C_src)
                    + beta_next_safe.unsqueeze(-1)  # (batch, C_dest, 1)
                )

                # Step 3: Find local reference (max over valid entries)
                log_joint_masked = torch.where(
                    valid_mask.view(batch, 1, 1),
                    log_joint,
                    torch.full_like(log_joint, NEG_INF),
                )
                local_ref = log_joint_masked.amax(dim=(-2, -1))

                # Guard against all-NEG_INF case
                is_local_ref_neginf = local_ref < NEG_INF + 1.0
                local_ref_safe = torch.where(
                    is_local_ref_neginf, torch.zeros_like(local_ref), local_ref
                )

                # Step 4: Compute relative log-marginal
                log_marginal_rel = log_joint - local_ref_safe.view(batch, 1, 1)

                # Step 5: Compute unnormalized marginal
                marginal_unnorm = torch.exp(log_marginal_rel)

                # Step 6: Compute scale factor using log_norm_at_ckpt
                log_scale = local_ref_safe + log_norm_at_ckpt - log_Z

                # Step 7: Defensive clamping
                log_scale_clamped = torch.clamp(log_scale, min=-700.0, max=0.0)
                scale = torch.exp(log_scale_clamped).view(batch, 1, 1)

                # Step 8: Final marginal
                marginal = marginal_unnorm * scale
                marginal = torch.where(
                    valid_mask.view(batch, 1, 1), marginal, torch.zeros_like(marginal)
                )

                # Accumulate boundary marginal: sum over k, c_dest, c_src
                # All segments starting at position t contribute to boundary_marginals[:, t]
                boundary_marginals[:, t] += marginal.sum(dim=(1, 2))

                # === Beta contribution (same as backward pass) ===
                scores_for_beta = edge_block + beta_next.unsqueeze(-1)
                scores_for_beta = torch.where(
                    valid_mask.view(batch, 1, 1),
                    scores_for_beta,
                    torch.full_like(scores_for_beta, NEG_INF),
                )
                new_beta_scores.append(scores_for_beta)

            # Update beta ring buffer
            if new_beta_scores:
                stacked = torch.stack(new_beta_scores, dim=1)
                over_dest = torch.logsumexp(stacked, dim=-2)
                new_beta = torch.logsumexp(over_dest, dim=1)

                ring_t_idx = t % K
                beta_ring[:, ring_t_idx, :] = torch.where(
                    active_mask.view(batch, 1), new_beta, beta_ring[:, ring_t_idx, :]
                )

    return boundary_marginals, log_Z
