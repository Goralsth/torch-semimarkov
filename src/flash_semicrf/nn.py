r"""Neural network modules for Semi-Markov CRF.

Provides :class:`torch.nn.Module` wrappers around streaming Semi-CRF kernels.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .duration import DurationDistribution, LearnedDuration, create_duration_distribution
from .helpers import Segment, ViterbiResult, score_gold_vectorized
from .streaming import (
    HAS_TRITON,
    semi_crf_streaming_forward,
    semi_crf_streaming_viterbi_with_backpointers,
)

# Conditionally import Triton viterbi for GPU acceleration
if HAS_TRITON:
    from .streaming import semi_crf_streaming_viterbi_triton
from .streaming.constants import NEG_INF
from .streaming.pytorch_reference import compute_edge_block_streaming
from .validation import (
    validate_device_consistency,
    validate_hidden_states,
    validate_labels,
    validate_lengths,
)

__all__ = [
    "SemiMarkovCRFHead",
    "Segment",
    "ViterbiResult",
]


class SemiMarkovCRFHead(nn.Module):
    r"""CRF head for Semi-Markov sequence labeling.

    Wraps Triton streaming kernels in a simple :class:`torch.nn.Module` with learnable
    transition and duration parameters. Compatible with DDP - gradients sync
    automatically via standard PyTorch mechanisms.

    The module computes:

    - Partition function :math:`Z` via streaming forward algorithm
    - Gold sequence score for NLL loss computation
    - Memory: :math:`O(KC)` independent of sequence length :math:`T`

    Args:
        num_classes (int): Number of label classes (C).
        max_duration (int): Maximum segment duration (K).
        hidden_dim (int, optional): If provided, adds a projection layer from
            ``hidden_dim`` to ``num_classes``. Default: ``None``
        init_scale (float, optional): Scale for parameter initialization.
            Default: ``0.1``
        duration_distribution (str, DurationDistribution, optional): Duration
            distribution to use. Can be:

            - ``None`` or ``"learned"``: Fully learned bias (default, current behavior)
            - ``"geometric"``: Geometric distribution with learnable rate
            - ``"negative_binomial"`` or ``"negbin"``: Negative binomial distribution
            - ``"poisson"``: Poisson-like distribution
            - ``"uniform"``: Uniform (no duration preference)
            - A :class:`~flash_semicrf.duration.DurationDistribution` instance

            Default: ``None`` (uses learned duration bias)
        edge_memory_threshold (float, optional): Memory threshold in bytes for
            the edge tensor. Used only for non-log/max semirings to determine
            whether the exact backend can fit in memory. Log/max semirings
            always use the streaming backend regardless of this threshold.
            Default: ``8e9`` (8GB)
        num_warps (int, optional): Number of warps per block for Triton kernels.
            Higher values increase parallelism but also register pressure.
            Recommended range: 2-8. Default: ``4``
        use_boundary_projections (bool, optional): If ``True``, creates per-position
            boundary projection layers that project encoder hidden states to
            segment-start and segment-end scores at every position. Requires
            ``hidden_dim`` to be set. Default: ``False``
        use_sequence_boundaries (bool, optional): If ``True``, adds learnable scalar
            start/end transition vectors :attr:`pi_start` and :attr:`pi_end` of shape
            :math:`(C,)`. These are added at sequence position 0 and the final position
            (per-sequence ``lengths[b] - 1``) respectively. Equivalent to pytorch-crf's
            ``start_transitions`` and ``end_transitions``. Does NOT require ``hidden_dim``.
            Orthogonal to ``use_boundary_projections`` — both can be enabled simultaneously.
            Default: ``False``

    Attributes:
        transition (Parameter): Label transition scores of shape :math:`(C, C)`.
            Uses ``(source, destination)`` indexing: ``transition[i, j]`` is the
            score for transitioning FROM label ``i`` TO label ``j``.
        duration_dist (DurationDistribution): Duration distribution module.
        projection (Linear or None): Optional projection from encoder hidden dim.
        pi_start (Parameter or None): Scalar start transition vector of shape
            :math:`(C,)`, or ``None`` if ``use_sequence_boundaries=False``.
        pi_end (Parameter or None): Scalar end transition vector of shape
            :math:`(C,)`, or ``None`` if ``use_sequence_boundaries=False``.

    Properties:
        duration_bias: Returns the current duration bias tensor of shape :math:`(K, C)`.

    Transition Matrix Convention:
        ``transition[i, j]`` = score for transitioning FROM label ``i`` TO label ``j``.

    Examples::

        >>> import torch
        >>> from flash_semicrf import SemiMarkovCRFHead
        >>>
        >>> # Create CRF head with default learned duration
        >>> crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)
        >>>
        >>> # Or with geometric duration distribution
        >>> crf = SemiMarkovCRFHead(
        ...     num_classes=24, max_duration=100, hidden_dim=512,
        ...     duration_distribution="geometric"
        ... )
        >>>
        >>> # Encoder output
        >>> batch, T = 4, 1000
        >>> hidden_states = torch.randn(batch, T, 512)
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Forward pass
        >>> result = crf(hidden_states, lengths)
        >>> print(result['partition'].shape)  # (batch,)
        >>>
        >>> # Training with labels
        >>> labels = torch.randint(0, 24, (batch, T))
        >>> loss = crf.compute_loss(hidden_states, lengths, labels)
        >>> loss.backward()

    .. note::
        For T > 100K, use ``precision="float64"`` (the default) for numerical stability.
        For shorter sequences or on GPUs with poor FP64 throughput (e.g. L40S),
        ``precision="float32"`` can give 2-4x speedup with negligible accuracy loss.
    """

    def __init__(
        self,
        num_classes: int,
        max_duration: int,
        hidden_dim: Optional[int] = None,
        init_scale: float = 0.1,
        duration_distribution: Optional[Union[str, DurationDistribution]] = None,
        edge_memory_threshold: float = 8e9,
        num_warps: int = 4,
        use_boundary_projections: bool = False,
        use_sequence_boundaries: bool = False,
        precision: str = "float64",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_duration = max_duration
        self.edge_memory_threshold = edge_memory_threshold
        self.num_warps = num_warps
        self.precision = precision

        # CRF parameters
        self.transition = nn.Parameter(torch.randn(num_classes, num_classes) * init_scale)

        # Duration distribution (supports multiple parameterizations)
        if duration_distribution is None or duration_distribution == "learned":
            # Default: fully learned, use init_scale for consistency
            self.duration_dist = LearnedDuration(max_duration, num_classes, init_std=init_scale)
        else:
            self.duration_dist = create_duration_distribution(
                duration_distribution, max_duration, num_classes
            )

        # Optional projection from encoder hidden dim
        if hidden_dim is not None:
            self.projection = nn.Linear(hidden_dim, num_classes)
        else:
            self.projection = None

        # Optional boundary projections: learn per-position segment-start / segment-end scores
        # Requires hidden_dim so we have encoder representations to project from.
        if use_boundary_projections:
            if hidden_dim is None:
                raise ValueError("use_boundary_projections=True requires hidden_dim to be provided")
            self.proj_start_layer = nn.Linear(hidden_dim, num_classes, bias=False)
            self.proj_end_layer = nn.Linear(hidden_dim, num_classes, bias=False)
            nn.init.normal_(self.proj_start_layer.weight, std=init_scale)
            nn.init.normal_(self.proj_end_layer.weight, std=init_scale)
        else:
            self.proj_start_layer = None
            self.proj_end_layer = None

        # Optional scalar sequence-boundary transitions: learned vectors pi_start (C,)
        # and pi_end (C,) added at positions 0 and T-1 respectively. Equivalent to
        # pytorch-crf's start_transitions/end_transitions. Does NOT require hidden_dim.
        if use_sequence_boundaries:
            self.pi_start = nn.Parameter(torch.zeros(num_classes))
            self.pi_end = nn.Parameter(torch.zeros(num_classes))
        else:
            self.pi_start = None
            self.pi_end = None

    @property
    def duration_bias(self) -> Tensor:
        """Duration bias tensor of shape :math:`(K, C)`."""
        return self.duration_dist()

    def _should_use_streaming(self, T: int) -> bool:
        """Return True if edge tensor would exceed memory threshold."""
        K = self.max_duration
        C = self.num_classes

        # Edge tensor size in bytes: T * K * C * C * 8 (float64)
        edge_tensor_bytes = T * K * C * C * 8

        return edge_tensor_bytes > self.edge_memory_threshold

    def _select_backend(self, T: int, semiring: str, use_triton: bool) -> tuple[str, bool]:
        """Select backend based on semiring requirements.

        For log/max semirings, always returns streaming. Triton vs PyTorch
        reference is decided downstream in autograd.py based on device,
        HAS_TRITON, and K value.

        For other semirings, returns exact (required for full semiring support).
        Raises ValueError if the edge tensor would exceed memory threshold.
        """
        # Semirings beyond log/max require exact backend
        if semiring not in ("log", "max"):
            if self._should_use_streaming(T):
                K, C = self.max_duration, self.num_classes
                raise ValueError(
                    f"Semiring '{semiring}' requires exact backend, but T={T}, K={K}, C={C} "
                    f"would require ~{T * K * C * C * 8 / 1e9:.1f}GB edge tensor. "
                    f"Use 'log' or 'max' semiring for streaming, or reduce T/K/C."
                )
            return "exact", False

        # log/max: always streaming. Triton vs PyTorch-reference decided
        # downstream (autograd.py checks is_cuda, HAS_TRITON, K>=3).
        return "streaming", use_triton

    def _validate_backend_for_boundaries(self, backend: str) -> None:
        """Raise if boundary features are enabled with a non-streaming backend.

        Non-streaming backends (exact, binary_tree_sharded, dp_standard) compute
        the partition function without boundary terms. If _score_gold includes
        boundaries but the partition does not, NLL = log Z - score(y*) is
        incorrectly normalized.
        """
        has_boundaries = self.proj_start_layer is not None or self.pi_start is not None
        if has_boundaries and backend not in ("auto", "streaming"):
            raise ValueError(
                f"Boundary projections require streaming backend, got backend={backend!r}. "
                f"Non-streaming backends do not include boundary terms in the partition "
                f"function, which would produce incorrect NLL."
            )

    def _compute_boundary_projections(self, hidden_states: Tensor):
        """Project encoder hidden states to boundary scores.

        Uses float64 internally for numerical stability at genomic-scale
        sequence lengths (T > 100K), consistent with cum_scores precision.

        Returns:
            Tuple of (proj_start, proj_end), each (batch, T, C) in float64,
            or (None, None) if boundary projections are disabled.
        """
        if self.proj_start_layer is None:
            return None, None
        return (
            self.proj_start_layer(hidden_states).double(),  # (batch, T, C)
            self.proj_end_layer(hidden_states).double(),  # (batch, T, C)
        )

    @staticmethod
    def _center_scores(scores: Tensor, lengths: Tensor) -> Tensor:
        """Subtract per-sequence masked mean from scores before cumsum.

        Computes the mean over valid positions only (u < lengths[b]),
        so padding tokens do not influence the centering baseline.

        Args:
            scores: Raw projected scores ``(batch, T, C)`` in any dtype.
            lengths: Sequence lengths ``(batch,)``.

        Returns:
            Centered scores in float64, same shape as input.
        """
        scores_float = scores.double()
        T = scores_float.shape[1]
        if T <= 1:
            return scores_float
        # Mask: (batch, T, 1) — True for valid positions
        mask = torch.arange(T, device=scores.device)[None, :] < lengths[:, None]
        masked_scores = scores_float * mask.unsqueeze(-1)  # zero out padding
        baseline = masked_scores.sum(dim=1, keepdim=True) / lengths[:, None, None].double()
        return scores_float - baseline

    def _apply_sequence_boundaries(
        self,
        proj_start: Optional[Tensor],
        proj_end: Optional[Tensor],
        cum_scores: Tensor,
        lengths: Tensor,
        device: torch.device,
    ) -> tuple[Optional[Tensor], Optional[Tensor], Tensor]:
        """Fold scalar sequence-boundary vectors into the model.

        When full boundary projections are active (``proj_start is not None``),
        adds :attr:`pi_start` and :attr:`pi_end` to positions 0 and ``lengths[b]-1``
        of the projection tensors. This triggers ``HAS_BOUNDARIES`` in the kernel
        (needed anyway for the full projections).

        When only scalar boundaries are active (``proj_start is None``), folds
        :attr:`pi_start` and :attr:`pi_end` directly into ``cum_scores`` via
        prefix-sum arithmetic, avoiding the boundary code path in the kernel
        entirely (zero overhead)::

            cum_scores[:, 0, :] -= pi_start   # content at t=0 gains +pi_start
            cum_scores[b, L, :] += pi_end     # content at t+k=L gains +pi_end

        Args:
            proj_start: Existing start projections ``(batch, T, C)`` or ``None``.
            proj_end: Existing end projections ``(batch, T, C)`` or ``None``.
            cum_scores: Cumulative scores ``(batch, T+1, C)``.
            lengths: Sequence lengths ``(batch,)``.
            device: Device for tensor allocation.

        Returns:
            Tuple of ``(proj_start, proj_end, cum_scores)`` — potentially modified.
            When scalar-only, proj_start/proj_end remain ``None`` and cum_scores
            is modified instead.
        """
        if self.pi_start is None:
            return proj_start, proj_end, cum_scores

        pi_start = self.pi_start.double()
        pi_end = self.pi_end.double()

        if proj_start is not None:
            # Both modes: add to existing projection tensors.
            # Clone to avoid in-place modification of autograd graph tensors.
            proj_start = proj_start.clone()
            proj_end = proj_end.clone()
            proj_start[:, 0, :] = proj_start[:, 0, :] + pi_start
            b_idx = torch.arange(lengths.shape[0], device=device)
            end_positions = (lengths - 1).long()
            proj_end[b_idx, end_positions, :] = proj_end[b_idx, end_positions, :] + pi_end
        else:
            # Scalar-only: fold into cum_scores to avoid HAS_BOUNDARIES overhead.
            # Subtracting pi_start from cum_scores[:, 0, :] makes content_score
            # (= cum[end] - cum[start]) include +pi_start for segments starting at 0.
            # Adding pi_end to cum_scores[b, L, :] adds it for segments ending at L-1.
            cum_scores = cum_scores.clone()
            cum_scores[:, 0, :] = cum_scores[:, 0, :] - pi_start
            b_idx = torch.arange(lengths.shape[0], device=device)
            cum_scores[b_idx, lengths.long(), :] = cum_scores[b_idx, lengths.long(), :] + pi_end

        return proj_start, proj_end, cum_scores

    def _build_edge_tensor(self, scores: Tensor, lengths: Tensor) -> Tensor:
        """Build edge potentials of shape (batch, T, K, C, C). O(TKC²) memory."""
        batch, T, C = scores.shape
        K = self.max_duration

        # Cumulative scores for content computation
        scores_float = self._center_scores(scores, lengths)
        cum_scores = torch.zeros(batch, T + 1, C, dtype=torch.float64, device=scores.device)
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Build edge tensor with T positions (streaming can access positions 0 to T-1)
        edge = torch.full(
            (batch, T, K, C, C), float("-inf"), dtype=torch.float64, device=scores.device
        )

        for n in range(T):
            # Loop over valid durations k = 1, 2, ..., min(K, T - n)
            for k in range(1, min(K, T - n) + 1):
                # Content score for segment [n, n+k)
                content = cum_scores[:, n + k, :] - cum_scores[:, n, :]  # (batch, C)

                # Duration k uses index k-1 (0-based indexing)
                dur_idx = k - 1

                # Add duration bias
                segment_score = content + self.duration_bias[dur_idx]  # (batch, C)

                # Add transition (C_dest x C_src)
                # edge[n, dur_idx, c_dest, c_src] = segment_score[c_dest] + transition[c_src, c_dest]
                edge[:, n, dur_idx] = segment_score.unsqueeze(-1) + self.transition.T.unsqueeze(0)

        return edge

    def _build_differentiable_edge(self, scores: Tensor, lengths: Tensor) -> Tensor:
        """Build edge potentials via torch.stack (preserves autograd graph).

        Unlike ``_build_edge_tensor`` which uses in-place assignment, this method
        builds the edge tensor via ``torch.stack``, maintaining gradient flow from
        the edge back to ``self.transition`` and ``self.duration_bias``. Required
        for pytorch-struct DP methods that need proper autograd through the edge.

        Uses a large negative finite sentinel (-1e18) instead of ``-inf`` for
        invalid positions. This is critical: ``_dp_standard`` computes
        ``logsumexp`` over ALL K durations including invalid ones. With true
        ``-inf``, the backward pass encounters ``softmax([-inf,...,-inf])`` =
        ``0/0 = NaN``. A finite sentinel avoids this while being numerically
        equivalent (``exp(-1e18) = 0`` in float64).
        """
        batch, T, C = scores.shape
        K = self.max_duration

        scores_float = self._center_scores(scores, lengths)

        # Build cum_scores without in-place ops to preserve autograd graph
        cumsum_vals = torch.cumsum(scores_float, dim=1)
        zero_pad = torch.zeros(batch, 1, C, dtype=torch.float64, device=scores.device)
        cum_scores = torch.cat([zero_pad, cumsum_vals], dim=1)

        # Use finite sentinel instead of -inf to avoid NaN gradient in logsumexp.
        # _dp_standard processes ALL K durations per position (unlike _dp_scan_streaming
        # which only processes valid durations). For invalid durations, all C_src entries
        # are the sentinel value. logsumexp([-inf,...,-inf]) backward = softmax = 0/0 = NaN.
        # With -1e18, softmax gives uniform 1/C, but the upstream gradient is ~0 (since
        # exp(-1e18) = 0 in the outer logsumexp), so the gradient is clean.
        _NEGINF_SAFE = -1e18
        neg_inf = torch.full((batch, C, C), _NEGINF_SAFE, dtype=torch.float64, device=scores.device)

        edge_parts = []
        for n in range(T):
            k_parts = []
            for k_idx in range(K):
                end_pos = n + k_idx + 1
                if end_pos <= T:
                    content = cum_scores[:, end_pos, :] - cum_scores[:, n, :]
                    segment = content + self.duration_bias[k_idx]
                    edge_nk = segment.unsqueeze(-1) + self.transition.T.unsqueeze(0)
                else:
                    edge_nk = neg_inf
                k_parts.append(edge_nk)
            edge_parts.append(torch.stack(k_parts, dim=1))
        return torch.stack(edge_parts, dim=1)

    def _forward_exact(
        self, scores: Tensor, lengths: Tensor, semiring: str, algorithm: str = "scan"
    ) -> Tensor:
        """Compute partition via exact edge tensor. O(TKC²) memory.

        Args:
            algorithm: DP algorithm to use on the materialized edge tensor.
                ``"scan"`` uses ``_dp_scan_streaming`` (ring buffer, default).
                ``"standard"`` uses ``_dp_standard`` (pytorch-struct reference).
        """
        from .semimarkov import SemiMarkov
        from .semirings import LogSemiring, MaxSemiring

        SEMIRING_MAP = {"log": LogSemiring, "max": MaxSemiring}
        semiring_cls = SEMIRING_MAP[semiring]

        # SemiMarkov interprets edge shape (batch, N-1, K, C, C) as sequence of length N
        # Since our edge has T positions, SemiMarkov sees N = T + 1
        model = SemiMarkov(semiring_cls)
        if algorithm == "standard":
            # Use differentiable edge construction (torch.stack, not in-place)
            # so autograd flows from the edge back to model parameters.
            edge = self._build_differentiable_edge(scores, lengths)
            result = model._dp_standard(edge, lengths=lengths + 1, force_grad=False)
        else:
            # In-place edge construction is fine for _dp_scan_streaming
            edge = self._build_edge_tensor(scores, lengths)
            result = model.logpartition(edge, lengths=lengths + 1, use_linear_scan=True)
        return result[0].squeeze(0)

    def _forward_binary_tree_sharded(
        self, scores: Tensor, lengths: Tensor, semiring: str
    ) -> Tensor:
        """Compute partition via sharded binary tree. Memory-efficient reference implementation.

        Uses CheckpointShardSemiring to reduce peak memory by splitting large matmuls
        into smaller shards that are processed sequentially with gradient checkpointing.
        This is slower than streaming but provides a reference implementation for validation.
        """
        from .semimarkov import SemiMarkov
        from .semirings import LogSemiring, MaxSemiring
        from .semirings.checkpoint import CheckpointShardSemiring

        SEMIRING_MAP = {"log": LogSemiring, "max": MaxSemiring}
        base_semiring = SEMIRING_MAP[semiring]

        # Wrap semiring with sharded checkpointing for memory efficiency
        ShardedSemiring = CheckpointShardSemiring(base_semiring, max_size=10000)

        # Build edge tensor (still O(T*K*C^2) memory, but matmuls are sharded)
        edge = self._build_edge_tensor(scores, lengths)

        # Use binary tree algorithm with sharded semiring
        model = SemiMarkov(ShardedSemiring)
        result = model._dp_binary_tree(edge, lengths=lengths + 1, force_grad=True)
        return result[0].squeeze(0)

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_triton: bool = True,
        backend: str = "auto",
    ) -> dict:
        r"""Compute partition function from encoder hidden states.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                if projection is enabled, or :math:`(\text{batch}, T, C)` if projection is ``None``.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``
            backend (str, optional): Backend selection mode:

                - ``"auto"``: Streaming for log/max semirings, exact for others (default)
                - ``"streaming"``: Force streaming backend
                - ``"exact"``: Force exact backend via ``semimarkov.py``

        Returns:
            dict: Dictionary containing:

            - **partition** (Tensor): Log partition function of shape :math:`(\text{batch},)`.
            - **cum_scores** (Tensor): Cumulative scores of shape :math:`(\text{batch}, T+1, C)`
              for loss computation.
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)  # (batch, T, C)
        else:
            scores = hidden_states

        # Debug check for gradient corruption after projection
        # This catches NaN propagation from corrupted model parameters early
        if scores.requires_grad and torch.isnan(scores).any():
            raise ValueError(
                "NaN detected in projected scores. This typically indicates gradient "
                "corruption from a previous backward pass. Check if model parameters "
                "contain NaN values (e.g., via torch.isnan(param).any() for each param)."
            )

        # Select backend (validate value first so typos get the standard error)
        if backend == "auto":
            backend_type, use_triton_final = self._select_backend(T, "log", use_triton)
        elif backend == "streaming":
            backend_type, use_triton_final = "streaming", use_triton
        elif backend == "exact":
            backend_type, use_triton_final = "exact", False
        elif backend == "binary_tree_sharded":
            backend_type, use_triton_final = "binary_tree_sharded", False
        elif backend == "dp_standard":
            backend_type, use_triton_final = "dp_standard", False
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'auto', 'streaming', 'exact', 'dp_standard', or 'binary_tree_sharded'."
            )
        self._validate_backend_for_boundaries(backend)

        # Build cumulative scores for prefix-sum edge retrieval
        # CRITICAL: Use float64 for numerical stability at T > 100K
        scores_float = self._center_scores(scores, lengths)
        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float64, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Debug check for NaN in cumsum (can happen if scores have extreme values)
        if cum_scores.requires_grad and torch.isnan(cum_scores).any():
            raise ValueError(
                "NaN detected in cumulative scores after cumsum. This typically indicates "
                "extreme values in the input scores or projection layer weights."
            )

        # Compute boundary projections (None, None if not enabled)
        proj_start, proj_end = self._compute_boundary_projections(hidden_states)
        proj_start, proj_end, cum_scores = self._apply_sequence_boundaries(
            proj_start, proj_end, cum_scores, lengths, scores.device
        )

        if backend_type == "streaming":
            # Compute partition function via streaming algorithm
            partition = semi_crf_streaming_forward(
                cum_scores,
                self.transition,
                self.duration_bias,
                lengths,
                self.max_duration,
                semiring="log",
                use_triton=use_triton_final,
                num_warps=self.num_warps,
                proj_start=proj_start,
                proj_end=proj_end,
                precision=self.precision,
            )
        elif backend_type == "binary_tree_sharded":
            # Use sharded binary tree backend for memory-efficient reference implementation
            partition = self._forward_binary_tree_sharded(scores, lengths, "log")
        elif backend_type == "dp_standard":
            # Original pytorch-struct list-comprehension linear scan
            partition = self._forward_exact(scores, lengths, "log", algorithm="standard")
        else:
            # Use exact backend via semimarkov.py (_dp_scan_streaming)
            partition = self._forward_exact(scores, lengths, "log")

        return {
            "partition": partition,
            "cum_scores": cum_scores,
            "proj_start": proj_start,
            "proj_end": proj_end,
        }

    def compute_loss(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        labels: Tensor,
        use_triton: bool = True,
        backend: str = "auto",
        reduction: str = "mean",
    ) -> Tensor:
        r"""Compute negative log-likelihood loss.

        .. math::
            \text{NLL} = \log Z - \text{score}(y^*)

        where :math:`Z` is the partition function and :math:`y^*` is the gold segmentation.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`. Each position
                has a label ID. Segments are extracted by finding where labels change.
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``True``
            backend (str, optional): Backend selection mode: ``"auto"``, ``"streaming"``,
                or ``"exact"``. Default: ``"auto"``
            reduction (str, optional): Reduction mode: ``"mean"``, ``"sum"``, or ``"none"``.
                Default: ``"mean"``

        Returns:
            Tensor: NLL loss. Scalar if reduction is ``"mean"`` or ``"sum"``,
            shape :math:`(\text{batch},)` if ``"none"``.
        """
        # Validate labels (hidden_states and lengths validated in forward())
        validate_labels(
            labels,
            self.num_classes,
            batch_size=hidden_states.shape[0],
            seq_length=hidden_states.shape[1],
        )

        result = self.forward(hidden_states, lengths, use_triton, backend)
        partition = result["partition"]
        cum_scores = result["cum_scores"]
        proj_start = result.get("proj_start")
        proj_end = result.get("proj_end")

        # Score the gold segmentation
        gold_score = self._score_gold(cum_scores, labels, lengths, proj_start, proj_end)

        # NLL = partition - gold_score
        nll = partition - gold_score

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        return nll

    def parameter_penalty(self, p: float = 2.0) -> Tensor:
        r"""Compute Lp penalty on CRF parameters.

        Args:
            p (float, optional): Norm order (2.0 for L2, 1.0 for L1). Default: ``2.0``

        Returns:
            Tensor: Scalar penalty :math:`\|W_{\text{trans}}\|_p^p + \|W_{\text{dur}}\|_p^p`.
        """
        penalty = self.transition.norm(p=p).pow(p)

        # Only penalize duration bias if it has learnable parameters.
        # Imposed distributions (e.g., frozen geometric) produce a constant
        # offset with no gradient — including it is misleading.
        #
        # Note: any() on an empty iterator returns False, which correctly
        # excludes zero-parameter distributions (UniformDuration uses a
        # registered buffer; CallableDuration's _device_tracker has
        # requires_grad=False) without special-casing them.
        if any(param.requires_grad for param in self.duration_dist.parameters()):
            penalty = penalty + self.duration_bias.norm(p=p).pow(p)

        if self.proj_start_layer is not None:
            penalty = penalty + self.proj_start_layer.weight.norm(p=p).pow(p)
            penalty = penalty + self.proj_end_layer.weight.norm(p=p).pow(p)

        if self.pi_start is not None:
            penalty = penalty + self.pi_start.norm(p=p).pow(p)
            penalty = penalty + self.pi_end.norm(p=p).pow(p)

        return penalty

    def _score_gold(
        self,
        cum_scores: Tensor,
        labels: Tensor,
        lengths: Tensor,
        proj_start: Optional[Tensor] = None,
        proj_end: Optional[Tensor] = None,
    ) -> Tensor:
        """Score the gold segmentation. Returns shape (batch,).

        Args:
            proj_start: Optional start boundary scores of shape (batch, T, C).
            proj_end: Optional end boundary scores of shape (batch, T, C).
        """
        return score_gold_vectorized(
            cum_scores=cum_scores,
            labels=labels,
            lengths=lengths,
            transition=self.transition,
            duration_bias=self.duration_bias,
            max_duration=self.max_duration,
            proj_start=proj_start,
            proj_end=proj_end,
        )

    @torch.no_grad()
    def decode(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        use_triton: bool = True,
        backend: str = "auto",
    ) -> Tensor:
        r"""Compute Viterbi (max) score without path reconstruction.

        Args:
            hidden_states (Tensor): Shape :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Shape :math:`(\text{batch},)`.
            use_triton (bool, optional): Use Triton kernels. Default: ``True``
            backend (str, optional): ``"auto"``, ``"streaming"``, or ``"exact"``. Default: ``"auto"``

        Returns:
            Tensor: Best segmentation scores of shape :math:`(\text{batch},)`.
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Select backend (validate value first so typos get the standard error)
        if backend == "auto":
            backend_type, use_triton_final = self._select_backend(T, "max", use_triton)
        elif backend == "streaming":
            backend_type, use_triton_final = "streaming", use_triton
        elif backend == "exact":
            backend_type, use_triton_final = "exact", False
        elif backend == "binary_tree_sharded":
            backend_type, use_triton_final = "binary_tree_sharded", False
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'auto', 'streaming', 'exact', or 'binary_tree_sharded'."
            )
        self._validate_backend_for_boundaries(backend)

        # Build cumulative scores
        scores_float = self._center_scores(scores, lengths)
        cum_scores = torch.zeros(
            batch, T + 1, self.num_classes, dtype=torch.float64, device=scores.device
        )
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Compute boundary projections (None, None if not enabled)
        proj_start, proj_end = self._compute_boundary_projections(hidden_states)
        proj_start, proj_end, cum_scores = self._apply_sequence_boundaries(
            proj_start, proj_end, cum_scores, lengths, scores.device
        )

        if backend_type == "streaming":
            # Use max semiring for Viterbi
            max_score = semi_crf_streaming_forward(
                cum_scores,
                self.transition,
                self.duration_bias,
                lengths,
                self.max_duration,
                semiring="max",
                use_triton=use_triton_final,
                num_warps=self.num_warps,
                proj_start=proj_start,
                proj_end=proj_end,
                precision=self.precision,
            )
        elif backend_type == "binary_tree_sharded":
            # Use sharded binary tree backend for memory-efficient reference implementation
            max_score = self._forward_binary_tree_sharded(scores, lengths, "max")
        else:
            # Use exact backend via semimarkov.py
            max_score = self._forward_exact(scores, lengths, "max")

        return max_score

    @torch.no_grad()
    def decode_with_traceback(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        max_traceback_length: int = 10000,
        use_triton: bool = True,
        backend: str = "auto",
    ) -> ViterbiResult:
        r"""Viterbi decode with path reconstruction.

        Args:
            hidden_states (Tensor): Shape :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Shape :math:`(\text{batch},)`.
            max_traceback_length (int, optional): Maximum T for traceback. Longer
                sequences return empty segment lists. Default: ``10000``
            use_triton (bool, optional): Use Triton kernels for forward pass. Default: ``True``
            backend (str, optional): ``"auto"``, ``"streaming"``, ``"exact"``, or
                ``"binary_tree_sharded"``. Default: ``"auto"``

        Returns:
            ViterbiResult: Named tuple with ``scores`` :math:`(\text{batch},)` and
            ``segments`` (list of list of :class:`Segment`).
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape
        device = hidden_states.device

        # Project to label space if needed
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build cumulative scores
        scores_float = self._center_scores(scores, lengths)
        cum_scores = torch.zeros(batch, T + 1, self.num_classes, dtype=torch.float64, device=device)
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Select backend (validate value first so typos get the standard error)
        if backend == "auto":
            backend_type = "streaming"
        elif backend == "streaming":
            backend_type = "streaming"
        elif backend in ("exact", "binary_tree_sharded"):
            backend_type = backend
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Use 'auto', 'streaming', 'exact', or 'binary_tree_sharded'."
            )
        self._validate_backend_for_boundaries(backend)

        # Compute boundary projections once (None, None if not enabled)
        proj_start, proj_end = self._compute_boundary_projections(hidden_states)
        proj_start, proj_end, cum_scores = self._apply_sequence_boundaries(
            proj_start, proj_end, cum_scores, lengths, hidden_states.device
        )

        # Check which sequences need traceback
        needs_traceback = lengths <= max_traceback_length
        any_needs_traceback = needs_traceback.any().item()

        # Non-streaming backends use per-sequence traceback
        if backend_type in ("exact", "binary_tree_sharded"):
            all_segments: list[list[Segment]] = []
            max_scores_list = []

            for b in range(batch):
                seq_len = int(lengths[b].item())
                proj_start_b = proj_start[b : b + 1] if proj_start is not None else None
                proj_end_b = proj_end[b : b + 1] if proj_end is not None else None
                if seq_len > max_traceback_length:
                    # Skip traceback for long sequences
                    all_segments.append([])
                    # Still need to compute score - use streaming for efficiency
                    score = semi_crf_streaming_forward(
                        cum_scores[b : b + 1],
                        self.transition,
                        self.duration_bias,
                        lengths[b : b + 1],
                        self.max_duration,
                        semiring="max",
                        use_triton=False,
                        proj_start=proj_start_b,
                        proj_end=proj_end_b,
                    )
                    max_scores_list.append(score)
                else:
                    # Use _traceback_single for per-sequence Viterbi with backpointers
                    segments = self._traceback_single(
                        cum_scores[b : b + 1], seq_len, proj_start_b, proj_end_b
                    )
                    all_segments.append(segments)
                    # Compute max score from segments
                    if segments:
                        max_scores_list.append(
                            torch.tensor(
                                sum(seg.score for seg in segments),
                                device=device,
                                dtype=torch.float64,
                            )
                        )
                    else:
                        # Empty segments - compute score via forward pass
                        score = semi_crf_streaming_forward(
                            cum_scores[b : b + 1],
                            self.transition,
                            self.duration_bias,
                            lengths[b : b + 1],
                            self.max_duration,
                            semiring="max",
                            use_triton=False,
                            proj_start=proj_start_b,
                            proj_end=proj_end_b,
                        )
                        max_scores_list.append(score.squeeze(0))

            max_scores = torch.stack(max_scores_list)
            return ViterbiResult(scores=max_scores, segments=all_segments)

        # Streaming backend - use batched Viterbi with backpointers
        if any_needs_traceback:
            can_use_triton = (
                HAS_TRITON and use_triton and cum_scores.is_cuda and self.max_duration >= 3
            )
            # Single-boundary fallback: Triton requires both or neither
            if can_use_triton and (proj_start is None) != (proj_end is None):
                can_use_triton = False

            # Get max scores AND backpointers in a single forward pass
            if can_use_triton:
                max_scores, bp_k, bp_c, final_labels = semi_crf_streaming_viterbi_triton(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    lengths,
                    self.max_duration,
                    proj_start=proj_start,
                    proj_end=proj_end,
                )
            else:
                max_scores, bp_k, bp_c, final_labels = semi_crf_streaming_viterbi_with_backpointers(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    lengths,
                    self.max_duration,
                    proj_start=proj_start,
                    proj_end=proj_end,
                )

            # Fast O(T) traceback using backpointers
            all_segments = self._traceback_from_backpointers(
                bp_k, bp_c, final_labels, lengths, cum_scores, proj_start, proj_end
            )

            # Clear segments for sequences that exceeded max_traceback_length
            for b in range(batch):
                if lengths[b].item() > max_traceback_length:
                    all_segments[b] = []
        else:
            # No sequences need traceback - just compute scores
            max_scores = semi_crf_streaming_forward(
                cum_scores,
                self.transition,
                self.duration_bias,
                lengths,
                self.max_duration,
                semiring="max",
                use_triton=use_triton,
                num_warps=self.num_warps,
                proj_start=proj_start,
                proj_end=proj_end,
            )
            all_segments = [[] for _ in range(batch)]

        return ViterbiResult(scores=max_scores, segments=all_segments)

    def _traceback_single(
        self,
        cum_scores: Tensor,  # (1, T+1, C)
        seq_len: int,
        proj_start_b: Optional[Tensor] = None,  # (1, T, C)
        proj_end_b: Optional[Tensor] = None,  # (1, T, C)
    ) -> list[Segment]:
        """Viterbi traceback for single sequence. O(TC) memory."""
        device = cum_scores.device
        C = self.num_classes
        K = self.max_duration

        # Allocate alpha table and backpointers
        # alpha[t, c] = best score to reach position t ending in state c
        alpha = torch.full((seq_len + 1, C), NEG_INF, device=device, dtype=torch.float64)
        alpha[0, :] = 0.0  # Start: all states equally valid

        # Backpointers: for each (t, c), store (best_k, best_c_src)
        bp_k = torch.zeros((seq_len + 1, C), dtype=torch.long, device=device)
        bp_c = torch.zeros((seq_len + 1, C), dtype=torch.long, device=device)

        # Forward pass with backpointer storage
        for t in range(1, seq_len + 1):
            # Number of valid durations at this position: k = 1, 2, ..., min(K, t)
            k_eff = min(K, t)

            for k in range(1, k_eff + 1):
                start = t - k

                # Compute edge block for this (start, k)
                edge_block = compute_edge_block_streaming(
                    cum_scores,
                    self.transition,
                    self.duration_bias,
                    start,
                    k,
                    proj_start=proj_start_b,
                    proj_end=proj_end_b,
                )  # (1, C_dest, C_src)

                # scores[c_dest, c_src] = alpha[start, c_src] + edge[c_dest, c_src]
                candidate_scores = alpha[start, :].unsqueeze(0) + edge_block[0]  # (C_dest, C_src)

                # Find best c_src for each c_dest
                best_scores_from_src, best_c_src = candidate_scores.max(dim=-1)  # (C_dest,)

                # Update alpha and backpointers where this k gives better score
                better_mask = best_scores_from_src > alpha[t, :]
                alpha[t, :] = torch.where(better_mask, best_scores_from_src, alpha[t, :])
                bp_k[t, :] = torch.where(better_mask, k, bp_k[t, :])
                bp_c[t, :] = torch.where(better_mask, best_c_src, bp_c[t, :])

        # Traceback from final position
        final_scores = alpha[seq_len, :]
        best_final_c = final_scores.argmax().item()

        segments: list[Segment] = []
        t = seq_len
        c = best_final_c

        while t > 0:
            k = bp_k[t, c].item()
            c_prev = bp_c[t, c].item()

            if k == 0:
                # Safety check - shouldn't happen with proper initialization
                break

            start = t - k
            end = t - 1  # Segment is [start, end] inclusive

            # Compute segment score contribution
            # Always include transition, even for first segment - the forward pass
            # computes: max_{c_src} [alpha[0, c_src] + segment_score + transition[c_src, c_dest]]
            # where alpha[0, c_src] = 0, so transition from c_prev is part of the score
            seg_score = self._compute_segment_score(
                cum_scores[0],
                start,
                end,
                c,
                c_prev,
                proj_start_b=proj_start_b[0] if proj_start_b is not None else None,
                proj_end_b=proj_end_b[0] if proj_end_b is not None else None,
            )

            segments.append(Segment(start=start, end=end, label=c, score=seg_score))

            t = start
            c = c_prev

        # Reverse to get segments in order
        segments.reverse()
        return segments

    def _traceback_from_backpointers(
        self,
        bp_k: Tensor,
        bp_c: Tensor,
        final_labels: Tensor,
        lengths: Tensor,
        cum_scores: Tensor,
        proj_start: Optional[Tensor] = None,
        proj_end: Optional[Tensor] = None,
    ) -> list[list[Segment]]:
        """O(T) traceback using pre-computed backpointers."""
        batch = lengths.shape[0]
        all_segments: list[list[Segment]] = []

        for b in range(batch):
            seq_len = lengths[b].item()
            segments: list[Segment] = []

            if seq_len == 0:
                all_segments.append(segments)
                continue

            t = seq_len
            c = final_labels[b].item()

            while t > 0:
                # bp_k and bp_c are 0-indexed: position t corresponds to index t-1
                k = bp_k[b, t - 1, c].item()
                c_prev = bp_c[b, t - 1, c].item()

                if k == 0:
                    # Safety check - shouldn't happen with proper forward pass
                    break

                start = t - k
                end = t - 1  # Segment is [start, end] inclusive

                # Compute segment score
                # Always include transition, even for first segment - the forward pass
                # computes: max_{c_src} [alpha[0, c_src] + segment_score + transition[c_src, c_dest]]
                # where alpha[0, c_src] = 0, so transition from c_prev is part of the score
                seg_score = self._compute_segment_score(
                    cum_scores[b],
                    start,
                    end,
                    c,
                    c_prev,
                    proj_start_b=proj_start[b] if proj_start is not None else None,
                    proj_end_b=proj_end[b] if proj_end is not None else None,
                )

                segments.append(Segment(start=start, end=end, label=c, score=seg_score))

                t = start
                c = c_prev

            # Reverse to get segments in order
            segments.reverse()
            all_segments.append(segments)

        return all_segments

    def _compute_segment_score(
        self,
        cum_scores: Tensor,
        start: int,
        end: int,
        label: int,
        prev_label: Optional[int],
        proj_start_b: Optional[Tensor] = None,  # (T, C)
        proj_end_b: Optional[Tensor] = None,  # (T, C)
    ) -> float:
        """Compute content + duration_bias + transition + boundary for a segment."""
        # Content score
        content = (cum_scores[end + 1, label] - cum_scores[start, label]).item()

        # Duration bias: duration k uses index k-1
        duration = end - start + 1
        dur_idx = duration - 1  # Duration k uses index k-1
        dur_bias = self.duration_bias[dur_idx, label].item()

        # Transition (if not first segment)
        trans = 0.0
        if prev_label is not None:
            trans = self.transition[prev_label, label].item()

        boundary = 0.0
        if proj_start_b is not None:
            boundary += proj_start_b[start, label].item()
        if proj_end_b is not None:
            boundary += proj_end_b[end, label].item()

        return content + dur_bias + trans + boundary

    def extra_repr(self) -> str:
        parts = [
            f"num_classes={self.num_classes}",
            f"max_duration={self.max_duration}",
        ]
        if self.projection is not None:
            parts.append(f"hidden_dim={self.projection.in_features}")
        parts.append(f"duration_dist={self.duration_dist.__class__.__name__}")
        if self.proj_start_layer is not None:
            parts.append("use_boundary_projections=True")
        if self.pi_start is not None:
            parts.append("use_sequence_boundaries=True")
        return ", ".join(parts)
