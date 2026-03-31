r"""Autograd functions and public API for streaming Semi-CRF."""

import warnings
from typing import Optional

import torch

from ..validation import validate_cum_scores, validate_lengths, validate_streaming_shapes
from .pytorch_reference import (
    linear_crf_backward_pytorch,
    linear_crf_forward_pytorch,
    linear_crf_viterbi_pytorch,
    semi_crf_k2_backward_pytorch,
    semi_crf_k2_forward_pytorch,
    semi_crf_k2_viterbi_pytorch,
    semi_crf_streaming_backward_pytorch,
    semi_crf_streaming_forward_pytorch,
)

# Triton imports are conditional
try:
    from .triton_backward import launch_streaming_triton_backward
    from .triton_forward import HAS_TRITON, launch_streaming_triton_kernel
except ImportError:
    HAS_TRITON = False
    launch_streaming_triton_kernel = None
    launch_streaming_triton_backward = None


# =============================================================================
# Shared validation helpers (used by all backward methods)
# =============================================================================


def _check_partition(partition: torch.Tensor, context: str) -> None:
    """Raise RuntimeError if partition contains NaN or Inf values.

    Called at the start of every backward pass. A non-finite partition from
    the forward pass means all gradients will be garbage; catching it here
    gives a cleaner error message than letting NaN propagate into parameters.

    Args:
        partition (Tensor): Partition function values of shape ``(batch,)``.
        context (str): Short label identifying the caller, e.g. ``"Triton"``.
    """
    if not torch.isfinite(partition).all():
        nan_count = torch.isnan(partition).sum().item()
        inf_count = torch.isinf(partition).sum().item()
        raise RuntimeError(
            f"Non-finite partition from forward pass ({context}): "
            f"{nan_count} NaN, {inf_count} Inf. "
            "Check forward pass numerical stability."
        )


def _check_tensor_finite(tensor: torch.Tensor, name: str, context: str) -> None:
    """Raise RuntimeError if a gradient tensor contains NaN or Inf values.

    Called after each backward computation to catch numerical instability
    before corrupted gradients reach model parameters.

    Args:
        tensor (Tensor): Gradient tensor to validate.
        name (str): Tensor name for the error message, e.g. ``"grad_cum_scores"``.
        context (str): Short label identifying the caller, e.g. ``"Triton"``.
    """
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise RuntimeError(
            f"Non-finite values in CRF backward ({context}): "
            f"{name} has {nan_count} NaN, {inf_count} Inf"
        )


class SemiCRFStreaming(torch.autograd.Function):
    r"""Pure PyTorch autograd function for streaming Semi-CRF. O(KC) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        proj_start: Optional[torch.Tensor] = None,
        proj_end: Optional[torch.Tensor] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> torch.Tensor:
        if semiring == "max":
            raise ValueError(
                "semiring='max' does not support autograd backward. "
                "Use semi_crf_streaming_forward() with torch.no_grad() instead."
            )

        # Detach inputs for forward computation
        partition, ring_checkpoints, actual_checkpoint_interval, log_norm_checkpoints = (
            semi_crf_streaming_forward_pytorch(
                cum_scores.detach(),
                transition.detach(),
                duration_bias.detach(),
                lengths,
                K,
                semiring,
                proj_start.detach() if proj_start is not None else None,
                proj_end.detach() if proj_end is not None else None,
                checkpoint_interval=checkpoint_interval,
            )
        )

        # Save for backward
        ctx.save_for_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            log_norm_checkpoints,
            partition,
            proj_start,
            proj_end,
        )
        ctx.K = K
        ctx.semiring = semiring
        ctx.checkpoint_interval = actual_checkpoint_interval

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            log_norm_checkpoints,
            partition,
            proj_start,
            proj_end,
        ) = ctx.saved_tensors

        _check_partition(partition, "PyTorch")

        grads = semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ctx.K,
            partition,
            ring_checkpoints,
            log_norm_checkpoints,
            ctx.checkpoint_interval,
            ctx.semiring,
            proj_start,
            proj_end,
        )

        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end = grads

        _check_tensor_finite(grad_cum_scores, "grad_cum_scores", "PyTorch")

        # Scale by upstream gradient
        #
        # Per-batch parameters (cum_scores, proj_start, proj_end):
        #   Scale each batch element by its grad_output[b]
        #
        # Shared parameters (transition, duration_bias):
        #   These now come as per-batch tensors: (batch, C, C) or (batch, K, C, C)
        #   We apply weighted sum: grad = Σ_b[grad_output[b] * grad_per_batch[b]]
        #   Using einsum for memory efficiency (avoids large intermediate tensor)
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale

        # Shared parameters: weighted sum via einsum (memory-efficient)
        # Notation: b=batch, k=duration, i=src_state, j=dst_state, c=state
        if grad_transition.ndim == 3:  # (batch, C, C) - static transitions
            grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
        else:  # (batch, K, C, C) - duration-dependent
            grad_transition = torch.einsum("bkij, b -> kij", grad_transition, grad_output)

        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)

        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start * scale
        if grad_proj_end is not None:
            grad_proj_end = grad_proj_end * scale

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # K
            None,  # semiring
            grad_proj_start,
            grad_proj_end,
            None,  # checkpoint_interval
        )


class SemiCRFStreamingTriton(torch.autograd.Function):
    r"""Triton-accelerated autograd function for streaming Semi-CRF. O(KC) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        proj_start: Optional[torch.Tensor] = None,
        proj_end: Optional[torch.Tensor] = None,
        num_warps: int = 4,
        checkpoint_interval: Optional[int] = None,
        precision: str = "float64",
    ) -> torch.Tensor:
        if semiring == "max":
            raise ValueError(
                "semiring='max' does not support autograd backward. "
                "Use semi_crf_streaming_forward() with torch.no_grad() instead."
            )

        partition, ring_checkpoints, actual_checkpoint_interval, log_norm_checkpoints = (
            launch_streaming_triton_kernel(
                cum_scores.detach(),
                transition.detach(),
                duration_bias.detach(),
                lengths,
                K,
                semiring,
                checkpoint_interval=checkpoint_interval,
                proj_start=proj_start.detach() if proj_start is not None else None,
                proj_end=proj_end.detach() if proj_end is not None else None,
                num_warps=num_warps,
                precision=precision,
            )
        )

        # The Triton kernel always returns float64 partition. Save that for the
        # backward (log_Z precision matters), but return in the caller's input dtype.
        ctx.save_for_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            log_norm_checkpoints,
            partition,  # float64 — used directly by launch_streaming_triton_backward
            proj_start,
            proj_end,
        )
        ctx.K = K
        ctx.semiring = semiring
        ctx.checkpoint_interval = actual_checkpoint_interval
        ctx.num_warps = num_warps
        ctx.precision = precision

        return partition.to(cum_scores.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            cum_scores,
            transition,
            duration_bias,
            lengths,
            ring_checkpoints,
            log_norm_checkpoints,
            partition,
            proj_start,
            proj_end,
        ) = ctx.saved_tensors

        _check_partition(partition, "Triton")

        # launch_streaming_triton_backward scales gradients by grad_output internally.
        # It returns 6 values; the 6th (boundary_marginals) is only populated when
        # return_boundary_marginals=True, which autograd never requests.
        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end, _ = (
            launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                log_norm_checkpoints,
                ctx.checkpoint_interval,
                grad_output,
                proj_start=proj_start,
                proj_end=proj_end,
                num_warps=ctx.num_warps,
                precision=ctx.precision,
            )
        )

        for name, tensor in [
            ("grad_cum_scores", grad_cum_scores),
            ("grad_transition", grad_transition),
            ("grad_duration_bias", grad_duration_bias),
        ]:
            _check_tensor_finite(tensor, name, "Triton")

        if grad_proj_start is not None:
            _check_tensor_finite(grad_proj_start, "grad_proj_start", "Triton")
            _check_tensor_finite(grad_proj_end, "grad_proj_end", "Triton")

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # K
            None,  # semiring
            grad_proj_start,
            grad_proj_end,
            None,  # num_warps
            None,  # checkpoint_interval
            None,  # precision
        )


# =============================================================================
# K=1 Linear CRF Fast Path
# =============================================================================


class LinearCRFStreaming(torch.autograd.Function):
    r"""Optimized K=1 (linear CRF) autograd function. O(batch*C) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        semiring: str = "log",
    ) -> torch.Tensor:
        if semiring == "max":
            raise ValueError(
                "semiring='max' does not support autograd backward. "
                "Use semi_crf_streaming_forward() with torch.no_grad() instead."
            )

        # Forward pass
        partition = linear_crf_forward_pytorch(
            cum_scores.detach(),
            transition.detach(),
            lengths,
            duration_bias.detach(),
        )

        # Save for backward
        ctx.save_for_backward(cum_scores, transition, duration_bias, lengths, partition)
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cum_scores, transition, duration_bias, lengths, partition = ctx.saved_tensors

        _check_partition(partition, "K=1")

        grad_cum_scores, grad_transition, grad_duration_bias = linear_crf_backward_pytorch(
            cum_scores,
            transition,
            lengths,
            partition,
            duration_bias,
        )

        _check_tensor_finite(grad_cum_scores, "grad_cum_scores", "K=1")

        # Scale by upstream gradient
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale

        # Shared parameters: weighted sum via einsum
        grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)

        if grad_duration_bias is not None:
            # grad_duration_bias is (batch, 1, C) -> reduce to (1, C)
            grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # semiring
        )


# =============================================================================
# K=2 Specialized Path
# =============================================================================


class SemiCRFK2Streaming(torch.autograd.Function):
    r"""Optimized K=2 semi-CRF autograd function. O(batch*C) memory."""

    @staticmethod
    def forward(
        ctx,
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        semiring: str = "log",
    ) -> torch.Tensor:
        if semiring == "max":
            raise ValueError(
                "semiring='max' does not support autograd backward. "
                "Use semi_crf_streaming_forward() with torch.no_grad() instead."
            )

        # Forward pass
        partition = semi_crf_k2_forward_pytorch(
            cum_scores.detach(),
            transition.detach(),
            duration_bias.detach(),
            lengths,
        )

        # Save for backward
        ctx.save_for_backward(cum_scores, transition, duration_bias, lengths, partition)
        ctx.semiring = semiring

        return partition

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cum_scores, transition, duration_bias, lengths, partition = ctx.saved_tensors

        _check_partition(partition, "K=2")

        grad_cum_scores, grad_transition, grad_duration_bias = semi_crf_k2_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            partition,
        )

        _check_tensor_finite(grad_cum_scores, "grad_cum_scores", "K=2")

        # Scale by upstream gradient
        batch = grad_output.shape[0]
        scale = grad_output.view(batch, 1, 1)
        grad_cum_scores = grad_cum_scores * scale

        # Shared parameters: weighted sum via einsum
        grad_transition = torch.einsum("bij, b -> ij", grad_transition, grad_output)
        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_duration_bias, grad_output)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            None,  # lengths
            None,  # semiring
        )


def semi_crf_streaming_forward(
    cum_scores: torch.Tensor,
    transition: torch.Tensor,
    duration_bias: torch.Tensor,
    lengths: torch.Tensor,
    K: int,
    semiring: str = "log",
    proj_start: Optional[torch.Tensor] = None,
    proj_end: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    num_warps: int = 4,
    checkpoint_interval: Optional[int] = None,
    precision: str = "float64",
) -> torch.Tensor:
    r"""Compute Semi-CRF partition function with streaming edge computation.

    O(KC) memory (ring buffer), O(T*K*C²) compute. Automatically dispatches to
    optimized implementations based on K:

    - **K=1**: Linear CRF fast path. O(batch*C) memory, no ring buffer.
    - **K=2**: Specialized 2-step path. O(batch*C) memory, explicit history.
    - **K≥3**: Triton streaming kernel on GPU, PyTorch fallback on CPU.

    .. warning::
        ``cum_scores`` **MUST** be float32 or float64 for numerical stability at T > 100K.
        Float64 is recommended as all Triton kernels compute internally in float64.
        Zero-centering before cumsum is critical to prevent precision loss.

    .. note::
        The Triton kernel requires K≥3 for correct operation due to ring buffer
        architecture constraints. K=1 and K=2 use specialized PyTorch paths.

    Args:
        cum_scores (Tensor): Cumulative projected scores of shape
            :math:`(\text{batch}, T+1, C)`. Must be float32 or float64 (float64
            recommended) and zero-centered before cumsum for numerical stability.
        transition (Tensor): Label transition scores of shape :math:`(C, C)` for
            static transitions, or :math:`(K, C, C)` for duration-dependent
            transitions. ``transition[c_src, c_dest]`` is the score for
            c_src -> c_dest.
        duration_bias (Tensor): Duration-specific label bias of shape :math:`(K, C)`.
            Required to compensate for sum-pooling length bias.
        lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
        K (int): Maximum segment duration.
        semiring (str, optional): ``"log"`` (logsumexp for partition) or ``"max"``
            (Viterbi). Default: ``"log"``
        proj_start (Tensor, optional): Start boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        proj_end (Tensor, optional): End boundary scores of shape
            :math:`(\text{batch}, T, C)`. Default: ``None``
        use_triton (bool, optional): If ``True``, use Triton kernels when available.
            Default: ``True``
        num_warps (int, optional): Number of warps per block for Triton kernels.
            Higher values increase parallelism but also register pressure.
            Recommended range: 2-8. Default: ``4``

    Returns:
        Tensor: Log partition function (or max score) of shape :math:`(\text{batch},)`.

    Examples::

        >>> import torch
        >>> from flash_semicrf.streaming import semi_crf_streaming_forward
        >>>
        >>> # Encoder output
        >>> batch, T, hidden_dim, C, K = 2, 100, 64, 4, 8
        >>> h = torch.randn(batch, T, hidden_dim)  # encoder output
        >>> W_content = torch.randn(hidden_dim, C)
        >>>
        >>> # Pre-project to label space (loop-invariant: outside kernel)
        >>> projected = h @ W_content  # (batch, T, C)
        >>>
        >>> # CRITICAL: Zero-center before cumsum
        >>> projected = projected - projected.mean(dim=1, keepdim=True)
        >>>
        >>> # Cumsum in float64
        >>> cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float64)
        >>> cum_scores[:, 1:, :] = torch.cumsum(projected.double(), dim=1)
        >>>
        >>> # Model parameters
        >>> transition = torch.randn(C, C) * 0.1
        >>> duration_bias = torch.randn(K, C) * 0.1
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Streaming forward (uses Triton on GPU)
        >>> partition = semi_crf_streaming_forward(
        ...     cum_scores, transition, duration_bias, lengths, K
        ... )
        >>> partition.shape
        torch.Size([2])
    """
    if semiring not in ("log", "max"):
        raise ValueError(f"semiring must be 'log' or 'max', got {semiring!r}")

    # Input validation
    validate_cum_scores(cum_scores)
    batch, T_plus_1, C = cum_scores.shape
    T = T_plus_1 - 1
    validate_lengths(lengths, T, batch_size=batch)
    validate_streaming_shapes(K, C, batch, T, transition, duration_bias, proj_start, proj_end)

    # Determine if gradients are needed
    # Gate on torch.is_grad_enabled() so torch.no_grad() with trainable params
    # (e.g. model.decode() during training) correctly takes the inference path.
    needs_grad = torch.is_grad_enabled() and (
        cum_scores.requires_grad
        or transition.requires_grad
        or duration_bias.requires_grad
        or (proj_start is not None and proj_start.requires_grad)
        or (proj_end is not None and proj_end.requires_grad)
    )

    # Guard: max semiring is inference-only (Viterbi scores are not differentiable)
    if semiring == "max" and needs_grad:
        raise ValueError(
            "semiring='max' (Viterbi) does not support gradients. "
            "The max semiring computes the highest-scoring path, which is not "
            "differentiable (requires Viterbi subgradients, not implemented). "
            "Use torch.no_grad() for decoding, or semiring='log' for training."
        )

    # =========================================================================
    # K=1 Fast Path: Linear CRF (no ring buffer, no duration loop)
    # =========================================================================
    if K == 1:
        if proj_start is not None or proj_end is not None:
            # Boundary projections are not yet supported for the K=1 fast path.
            # Skip to the generic K>=3 implementation below.
            pass
        elif needs_grad:
            return LinearCRFStreaming.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                semiring,
            )
        else:
            # Inference path
            if semiring == "max":
                scores, _ = linear_crf_viterbi_pytorch(
                    cum_scores, transition, lengths, duration_bias
                )
                return scores
            else:
                return linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias)

    # =========================================================================
    # K=2 Fast Path: Explicit 2-step history (no ring buffer)
    # =========================================================================
    if K == 2:
        if proj_start is not None or proj_end is not None:
            # Boundary projections are not yet supported for the K=2 fast path.
            # Skip to the generic K>=3 implementation below.
            pass
        elif needs_grad:
            return SemiCRFK2Streaming.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                semiring,
            )
        else:
            # Inference path
            if semiring == "max":
                scores, _, _ = semi_crf_k2_viterbi_pytorch(
                    cum_scores, transition, duration_bias, lengths
                )
                return scores
            else:
                return semi_crf_k2_forward_pytorch(cum_scores, transition, duration_bias, lengths)

    # =========================================================================
    # Generic path: Triton (K>=3 only) or PyTorch fallback
    # =========================================================================
    can_use_triton = HAS_TRITON and use_triton and cum_scores.is_cuda and K >= 3

    # Single-boundary: Triton requires both proj_start and proj_end.
    # Fall back to PyTorch path which handles each independently.
    _single_boundary = (proj_start is None) != (proj_end is None)
    if _single_boundary and can_use_triton:
        warnings.warn(
            f"Only one boundary tensor provided (K={K}, device={cum_scores.device}); "
            "falling back to PyTorch path. Pass both proj_start and proj_end "
            "for Triton acceleration.",
            UserWarning,
            stacklevel=2,
        )
        can_use_triton = False

    if needs_grad:
        # Training path
        if can_use_triton:
            # Use Triton forward + Triton backward kernels
            return SemiCRFStreamingTriton.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start,
                proj_end,
                num_warps,
                checkpoint_interval,
                precision,
            )
        else:
            # Pure PyTorch path
            return SemiCRFStreaming.apply(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start,
                proj_end,
                checkpoint_interval,
            )
    else:
        # Inference path (no gradients)
        if can_use_triton:
            # Use fast custom Triton kernel
            partition, _, _, _ = launch_streaming_triton_kernel(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                checkpoint_interval=checkpoint_interval,
                proj_start=proj_start,
                proj_end=proj_end,
                num_warps=num_warps,
                precision=precision,
            )
            return partition
        else:
            # CPU fallback
            partition, _, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring,
                proj_start,
                proj_end,
                checkpoint_interval=checkpoint_interval,
            )
            return partition
