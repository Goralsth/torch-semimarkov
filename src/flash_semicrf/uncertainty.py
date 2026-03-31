r"""Uncertainty quantification for Semi-Markov CRF.

This module provides uncertainty quantification methods for the
:class:`~flash_semicrf.nn.SemiMarkovCRFHead`, enabling clinical applications
where boundary confidence is critical.

Two approaches are supported:

1. **Streaming-compatible** (:math:`T \geq 10K`): Uses gradient-based marginals
   from backward pass
2. **Exact** (:math:`T < 10K`): Uses :meth:`SemiMarkov.marginals` and
   :class:`~flash_semicrf.semirings.EntropySemiring` with pre-computed edges

Important observation: The gradient of :math:`\log Z` w.r.t. cumulative scores gives marginal
information via the forward-backward algorithm. This works with the streaming API
for any sequence length.

See Also:
    :class:`UncertaintySemiMarkovCRFHead`: CRF head with uncertainty methods
    :class:`UncertaintyMixin`: Mixin class providing uncertainty methods
"""

from typing import Optional

import torch
from torch import Tensor

from .nn import SemiMarkovCRFHead
from .streaming import semi_crf_streaming_forward
from .validation import (
    validate_device_consistency,
    validate_hidden_states,
    validate_labels,
    validate_lengths,
)


class UncertaintyMixin:
    r"""Mixin providing uncertainty quantification for SemiMarkovCRFHead.

    Uses gradient-based marginals for streaming API compatibility (:math:`T \geq 10K`).
    Falls back to exact methods for short sequences when edges fit in memory.

    This mixin provides the following methods:

    - :meth:`compute_boundary_marginals`: :math:`P(\text{boundary at position } t)` for each position
    - :meth:`compute_position_marginals`: Per-position label distributions
    - :meth:`compute_entropy_streaming`: Approximate entropy from marginal distribution
    - :meth:`compute_entropy_exact`: Exact entropy via :class:`~flash_semicrf.semirings.EntropySemiring`
      (:math:`T < 10K` only)
    - :meth:`compute_loss_uncertainty_weighted`: Uncertainty-weighted NLL loss

    .. note::
        Classes using this mixin must have ``projection``, ``transition``,
        ``duration_bias``, ``num_classes``, and ``max_duration`` attributes.
    """

    def compute_boundary_marginals(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        backend: str = "auto",
        normalize: bool = True,
        use_streaming: Optional[bool] = None,
    ) -> Tensor:
        r"""compute_boundary_marginals(hidden_states, lengths, backend="auto", normalize=True) -> Tensor

        Compute :math:`P(\text{boundary at position } t)` for each position.

        For streaming mode: Uses gradient of :math:`\log Z` w.r.t.
        cumulative scores. For exact mode: Uses :meth:`SemiMarkov.marginals`
        if edges fit in memory.

        The gradient of the log partition function w.r.t. the cumulative scores
        encodes information about which positions contribute to the partition,
        which correlates with boundary probability.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            backend (str, optional): Backend selection mode:

                - ``"auto"``: Select based on memory heuristic (default)
                - ``"streaming"``: Force streaming gradient-based method
                - ``"exact"``: Force exact marginals via edge tensor

            normalize (bool, optional): If ``True``, normalize to :math:`[0, 1]` range.
                Default: ``True``
            use_streaming (bool, optional): Deprecated. Use ``backend`` instead.
                If provided, ``True`` maps to ``backend="streaming"`` and
                ``False`` maps to ``backend="exact"``.

        Returns:
            Tensor: Boundary probabilities of shape :math:`(\text{batch}, T)`.
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        _, T, _ = hidden_states.shape

        # Backward compatibility: map use_streaming to backend
        if use_streaming is not None:
            backend = "streaming" if use_streaming else "exact"

        # Select backend
        if backend == "auto":
            backend_type, _ = self._select_backend(T, "log", False)
        elif backend == "streaming":
            backend_type = "streaming"
        elif backend == "exact":
            backend_type = "exact"
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'streaming', or 'exact'.")

        if backend_type == "streaming":
            return self._compute_boundary_marginals_streaming(hidden_states, lengths, normalize)
        else:
            return self._compute_boundary_marginals_exact(hidden_states, lengths, normalize)

    def _compute_boundary_marginals_streaming(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        r"""_compute_boundary_marginals_streaming(hidden_states, lengths, normalize=True) -> Tensor

        Compute boundary marginals via streaming forward-backward algorithm.

        Uses the forward-backward algorithm with :math:`O(KC)` memory to compute
        true boundary marginals :math:`P(\text{segment starts at position } t)`.
        This matches the exact method but scales to any sequence length.

        The boundary marginal at position t is:

        .. math::
            P(\text{boundary at } t) = \sum_{k, c_{\text{dst}}, c_{\text{src}}}
            P(\text{segment}[t, k, c_{\text{dst}}, c_{\text{src}}])

        Args:
            hidden_states (Tensor): Encoder output of shape
                :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            normalize (bool, optional): If ``True``, normalize to :math:`[0, 1]` range
                per sequence. Default: ``True``

        Returns:
            Tensor: Boundary probabilities of shape :math:`(\text{batch}, T)`.
                Each value represents the probability that a segment starts at that position.

        .. note::
            Works with any sequence length via streaming API. Memory is :math:`O(KC)`
            independent of :math:`T`. Compute is 2x forward pass (forward + backward).

        Example::

            >>> model = UncertaintySemiMarkovCRFHead(num_classes=4, max_duration=10, hidden_dim=64)
            >>> hidden = torch.randn(2, 100, 64)
            >>> lengths = torch.tensor([100, 80])
            >>> marginals = model._compute_boundary_marginals_streaming(hidden, lengths)
            >>> marginals.shape
            torch.Size([2, 100])

        See Also:
            :func:`~flash_semicrf.streaming.semi_crf_streaming_marginals_pytorch`:
                Underlying implementation
            :meth:`_compute_boundary_marginals_exact`: Exact method (requires edge tensor)
        """
        from .streaming import semi_crf_streaming_marginals_pytorch

        batch, T, _ = hidden_states.shape

        # Project to label space if needed
        if self.projection is not None:
            with torch.no_grad():
                scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Zero-center scores using masked mean (see nn.py _center_scores)
        scores_float = self._center_scores(scores.detach(), lengths)

        # Build cumulative scores
        cum_scores = torch.zeros(
            batch,
            T + 1,
            self.num_classes,
            dtype=torch.float64,
            device=scores.device,
        )
        cum_scores[:, 1:] = torch.cumsum(scores_float, dim=1)

        # Compute true boundary marginals via forward-backward algorithm
        boundary_probs, _ = semi_crf_streaming_marginals_pytorch(
            cum_scores,
            self.transition.detach(),
            self.duration_bias.detach(),
            lengths,
            self.max_duration,
        )

        if normalize:
            # Normalize per sequence to [0, 1]
            max_val = boundary_probs.max(dim=-1, keepdim=True)[0] + 1e-8
            boundary_probs = boundary_probs / max_val

        return boundary_probs

    def _compute_boundary_marginals_exact(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        r"""_compute_boundary_marginals_exact(hidden_states, lengths, normalize=True) -> Tensor

        Compute boundary marginals via exact edge marginals.

        Uses :meth:`~flash_semicrf.SemiMarkov.marginals` to compute exact
        posterior marginals over all edge potentials, then aggregates to get
        the probability of a segment boundary at each position.

        Args:
            hidden_states (Tensor): Encoder output of shape
                :math:`(\text{batch}, T, \text{hidden\_dim})` or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            normalize (bool, optional): If ``True``, normalize to :math:`[0, 1]` range
                per sequence. Default: ``True``

        Returns:
            Tensor: Boundary probabilities of shape :math:`(\text{batch}, T)`.

        .. warning::
            Requires materializing the full edge tensor with :math:`O(TKC^2)` memory.
            Use streaming method for :math:`T > 10K`.

        See Also:
            :meth:`_compute_boundary_marginals_streaming`: Streaming alternative
        """
        from .semimarkov import SemiMarkov
        from .semirings import LogSemiring

        # Project to label space
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build edge tensor (WARNING: O(T*K*C^2) memory!)
        edge = self._build_edge_tensor(scores, lengths)

        # Compute marginals
        model = SemiMarkov(LogSemiring)
        # Edge tensor has T positions (0..T-1), SemiMarkov expects N=T+1 for lengths
        # Force linear scan to match streaming API indexing convention
        edge_marginals = model.marginals(edge, lengths=lengths + 1, use_linear_scan=True)

        # Aggregate to boundary probability
        # edge_marginals[b, n, k, c_dest, c_src] = P(segment starting at n with label c_dest)
        # Sum over k, c_dest, c_src to get P(any segment start at position n)
        boundary_probs = edge_marginals.sum(dim=(2, 3, 4))  # (batch, T)

        if normalize:
            max_val = boundary_probs.max(dim=-1, keepdim=True)[0] + 1e-8
            boundary_probs = boundary_probs / max_val

        return boundary_probs

    def compute_position_marginals(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""compute_position_marginals(hidden_states, lengths) -> Tensor

        Compute per-position label marginals :math:`P(\text{label}=c \text{ at position } t)`.

        Uses gradient of :math:`\log Z` w.r.t. projected scores to get per-position
        label distribution.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Label probabilities of shape :math:`(\text{batch}, T, C)`.
        """
        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape

        # Project to label space
        if self.projection is not None:
            with torch.no_grad():
                scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Use enable_grad() to ensure gradient computation works even if called
        # from within a no_grad() context
        with torch.enable_grad():
            # Make scores require grad
            scores_for_grad = scores.detach().requires_grad_(True)

            # Zero-center scores using masked mean (see nn.py _center_scores)
            scores_centered = self._center_scores(scores_for_grad, lengths)

            # Build cumulative scores
            cum_scores = torch.zeros(
                batch,
                T + 1,
                self.num_classes,
                dtype=torch.float64,
                device=scores_for_grad.device,
            )
            cum_scores[:, 1:] = torch.cumsum(scores_centered, dim=1)

            # Forward pass
            partition = semi_crf_streaming_forward(
                cum_scores,
                self.transition.detach(),
                self.duration_bias.detach(),
                lengths,
                self.max_duration,
                semiring="log",
                use_triton=False,
            )

            # Backward to get per-position, per-label gradients
            partition.sum().backward()

        # d(log Z)/d(score[t,c]) reflects the influence of label c at position t.
        # Centering means gradients can be negative, so use softmax to convert
        # to a proper distribution over classes at each position.
        grad = scores_for_grad.grad  # (batch, T, C)
        position_marginals = torch.softmax(grad, dim=-1)

        return position_marginals

    def compute_entropy_streaming(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""compute_entropy_streaming(hidden_states, lengths) -> Tensor

        Approximate entropy from marginal distribution.

        For very long sequences where :class:`~flash_semicrf.semirings.EntropySemiring`
        isn't available. Computes:

        .. math::
            H \approx -\sum_t P(\text{boundary}_t) \log P(\text{boundary}_t)

        as a proxy for segmentation uncertainty.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Entropy estimates of shape :math:`(\text{batch},)`.
        """
        # Get boundary marginals
        boundary_probs = self.compute_boundary_marginals(
            hidden_states, lengths, use_streaming=True, normalize=False
        )

        # Normalize to valid probability distribution
        boundary_probs = boundary_probs / (boundary_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute entropy: H = -sum(p * log(p))
        # Clamp to avoid log(0)
        log_probs = torch.log(boundary_probs.clamp(min=1e-10))
        entropy = -(boundary_probs * log_probs).sum(dim=-1)

        return entropy

    def compute_entropy_exact(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        r"""compute_entropy_exact(hidden_states, lengths) -> Tensor

        Compute exact entropy via :class:`~flash_semicrf.semirings.EntropySemiring` (:math:`T < 10K` only).

        Computes the exact Shannon entropy of the segmentation distribution:

        .. math::
            H(P) = -\sum_y P(y) \log P(y)

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.

        Returns:
            Tensor: Exact entropy values of shape :math:`(\text{batch},)`.

        Raises:
            MemoryError: If T is too large and edge tensor doesn't fit in memory.

        .. warning::
            Requires materializing edge tensor - will OOM for :math:`T > 10K`!
        """
        from .semimarkov import SemiMarkov
        from .semirings import EntropySemiring

        # Input validation
        validate_hidden_states(hidden_states)
        validate_lengths(lengths, hidden_states.shape[1], batch_size=hidden_states.shape[0])
        validate_device_consistency(hidden_states, lengths, names=["hidden_states", "lengths"])

        batch, T, _ = hidden_states.shape

        # Project to label space
        if self.projection is not None:
            scores = self.projection(hidden_states)
        else:
            scores = hidden_states

        # Build edge tensor (WARNING: O(T*K*C^2) memory!)
        edge = self._build_edge_tensor(scores, lengths)

        # Compute entropy using EntropySemiring
        # Force linear scan - binary tree algorithm has numerical issues with EntropySemiring
        # Edge tensor has T positions, SemiMarkov expects N=T+1, so pass lengths+1
        model = SemiMarkov(EntropySemiring)
        entropy = model.sum(edge, lengths=lengths + 1, use_linear_scan=True)

        return entropy

    def compute_loss_uncertainty_weighted(
        self,
        hidden_states: Tensor,
        lengths: Tensor,
        labels: Tensor,
        uncertainty_weight: float = 1.0,
        focus_mode: str = "high_uncertainty",
        use_triton: bool = False,
        reduction: str = "mean",
    ) -> Tensor:
        r"""compute_loss_uncertainty_weighted(hidden_states, lengths, labels, uncertainty_weight=1.0, focus_mode="high_uncertainty", use_triton=False, reduction="mean") -> Tensor

        Compute loss weighted by uncertainty for focused learning.

        Enables active learning by weighting the NLL loss by uncertainty:

        - ``"high_uncertainty"``: Higher weight on uncertain regions
        - ``"boundary_regions"``: Higher weight on boundary positions

        The weighted loss is:

        .. math::
            \mathcal{L}_{\text{weighted}} = (1 + \lambda \cdot u) \cdot \text{NLL}

        where :math:`\lambda` is the uncertainty weight and :math:`u` is the
        uncertainty measure.

        Args:
            hidden_states (Tensor): Encoder output of shape :math:`(\text{batch}, T, \text{hidden\_dim})`
                or :math:`(\text{batch}, T, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            labels (Tensor): Per-position labels of shape :math:`(\text{batch}, T)`.
            uncertainty_weight (float, optional): Scaling factor :math:`\lambda` for
                uncertainty weighting. Default: ``1.0``
            focus_mode (str, optional): Uncertainty mode: ``"high_uncertainty"`` or
                ``"boundary_regions"``. Default: ``"high_uncertainty"``
            use_triton (bool, optional): Whether to use Triton kernels. Default: ``False``
            reduction (str, optional): Reduction mode: ``"mean"``, ``"sum"``, or ``"none"``.
                Default: ``"mean"``

        Returns:
            Tensor: Weighted NLL loss.
        """
        # Input validation (hidden_states/lengths validated in called methods)
        validate_labels(
            labels,
            self.num_classes,
            batch_size=hidden_states.shape[0],
            seq_length=hidden_states.shape[1],
        )

        # First pass: compute uncertainty (no gradients needed)
        with torch.no_grad():
            if focus_mode == "high_uncertainty":
                # Use entropy as uncertainty measure
                uncertainty = self.compute_entropy_streaming(hidden_states.detach(), lengths)
                # Normalize to [0, 1]
                uncertainty = uncertainty / (uncertainty.max() + 1e-8)
            elif focus_mode == "boundary_regions":
                # Weight by average boundary probability
                boundary_probs = self.compute_boundary_marginals(
                    hidden_states.detach(), lengths, use_streaming=True
                )
                uncertainty = boundary_probs.mean(dim=-1)  # (batch,)
            else:
                raise ValueError(f"Unknown focus_mode: {focus_mode}")

        # Second pass: compute weighted loss
        # Standard forward + loss
        result = self.forward(hidden_states, lengths, use_triton)
        partition = result["partition"]
        cum_scores = result["cum_scores"]

        # Score the gold segmentation
        gold_score = self._score_gold(cum_scores, labels, lengths)

        # NLL = partition - gold_score
        nll = partition - gold_score

        # Apply uncertainty weighting
        weight = 1.0 + uncertainty_weight * uncertainty
        weighted_nll = nll * weight

        if reduction == "mean":
            return weighted_nll.mean()
        elif reduction == "sum":
            return weighted_nll.sum()
        return weighted_nll


class UncertaintySemiMarkovCRFHead(SemiMarkovCRFHead, UncertaintyMixin):
    r"""SemiMarkovCRFHead with uncertainty quantification methods.

    Combines the base :class:`~flash_semicrf.nn.SemiMarkovCRFHead` functionality
    with :class:`UncertaintyMixin` methods for clinical applications requiring
    boundary confidence estimates.

    This class inherits all functionality from :class:`SemiMarkovCRFHead` (forward,
    compute_loss, decode, decode_with_traceback) and adds uncertainty methods from
    :class:`UncertaintyMixin` (compute_boundary_marginals, compute_position_marginals,
    compute_entropy_streaming, compute_entropy_exact, compute_loss_uncertainty_weighted).

    Args:
        num_classes (int): Number of label classes (C).
        max_duration (int): Maximum segment duration (K).
        hidden_dim (int, optional): If provided, adds a projection layer from
            ``hidden_dim`` to ``num_classes``. Default: ``None``
        init_scale (float, optional): Scale for parameter initialization.
            Default: ``0.1``
        duration_distribution (str, DurationDistribution, optional): Duration
            distribution to use. Can be:

            - ``None`` or ``"learned"``: Fully learned bias (default)
            - ``"geometric"``: Geometric distribution with learnable rate
            - ``"negative_binomial"`` or ``"negbin"``: Negative binomial distribution
            - ``"poisson"``: Poisson-like distribution
            - ``"uniform"``: Uniform (no duration preference)
            - A :class:`~flash_semicrf.duration.DurationDistribution` instance

            Default: ``None`` (uses learned duration bias)
        edge_memory_threshold (float, optional): Memory threshold in bytes for
            automatic backend selection. Default: ``8e9`` (8GB)

    Attributes:
        transition (Parameter): Label transition scores of shape :math:`(C, C)`.
            Uses ``(source, destination)`` indexing: ``transition[i, j]`` is the
            score for transitioning FROM label ``i`` TO label ``j``.
        duration_dist (DurationDistribution): Duration distribution module.
        projection (Linear or None): Optional projection from encoder hidden dim.

    Properties:
        duration_bias: Returns the current duration bias tensor of shape :math:`(K, C)`.
            This is a property for backward compatibility - internally uses ``duration_dist()``.

    Transition Matrix Convention:
        The transition parameter has shape :math:`(C, C)` with indexing:

        - ``transition[i, j]`` = score for transitioning FROM label ``i`` TO label ``j``
        - Convention: ``(source, destination)`` ordering

        When building edge potentials internally, the matrix is transposed::

            edge[t, k, c_dest, c_src] = content[c_dest] + dur_bias[k, c_dest]
                                      + transition[c_src, c_dest]

        This matches the standard CRF transition convention used in the literature.

    Examples::

        >>> model = UncertaintySemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)
        >>> hidden = encoder(x)  # (batch, T, 512)
        >>> lengths = torch.full((batch,), T)
        >>>
        >>> # Standard loss (inherited from SemiMarkovCRFHead)
        >>> loss = model.compute_loss(hidden, lengths, labels)
        >>>
        >>> # Uncertainty-weighted loss (from UncertaintyMixin)
        >>> loss = model.compute_loss_uncertainty_weighted(hidden, lengths, labels)
        >>>
        >>> # Boundary marginals (from UncertaintyMixin)
        >>> boundary_probs = model.compute_boundary_marginals(hidden, lengths)
        >>>
        >>> # With geometric duration distribution
        >>> model = UncertaintySemiMarkovCRFHead(
        ...     num_classes=24, max_duration=100, hidden_dim=512,
        ...     duration_distribution="geometric"
        ... )

    See Also:
        :class:`~flash_semicrf.nn.SemiMarkovCRFHead`: Base CRF head without uncertainty
        :class:`UncertaintyMixin`: Mixin providing uncertainty methods
        :mod:`flash_semicrf.duration`: Available duration distributions
    """

    pass  # All functionality inherited from SemiMarkovCRFHead and UncertaintyMixin
