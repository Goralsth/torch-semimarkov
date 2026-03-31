r"""Triton forward kernels for streaming Semi-CRF.

This module contains a single unified Triton kernel for the forward pass
(log and max semiring, with optional backpointer tracking) and the launcher
functions that allocate buffers and dispatch it.

The kernel implements streaming edge computation where edge potentials are
computed on-the-fly from cumulative scores via prefix-sum decomposition:

.. math::
    \text{edge}[c_{\text{dst}}, c_{\text{src}}] =
    (\text{cum\_scores}[t, c_{\text{dst}}] - \text{cum\_scores}[t-k, c_{\text{dst}}])
    + \text{duration\_bias}[k, c_{\text{dst}}]
    + \text{transition}[c_{\text{src}}, c_{\text{dst}}]

The semiring variant and backpointer tracking are selected at compile time via
``IS_MAX_SEMIRING: tl.constexpr`` and ``TRACK_BACKPOINTERS: tl.constexpr``,
following the Flash Attention pattern for ``IS_CAUSAL`` / ``BIAS_TYPE``.

.. warning::
    **Minimum K Requirement**: These kernels require K >= 3 for correct operation.
    The ring buffer architecture assumes meaningful separation between timesteps:

    - **K=1**: Ring buffer aliasing (all t % 1 = 0). Use ``LinearCRFStreaming`` instead.
    - **K=2**: Ring buffer fragility (alternating slots). Use ``SemiCRFK2Streaming`` instead.
    - **K>=3**: Sufficient ring rotation for stable operation.

    The dispatch logic in ``autograd.py`` automatically routes K<3 to specialized
    PyTorch implementations. Do not call these kernels directly with K<3.

"""

import torch

from ..validation import validate_streaming_shapes
from .constants import NEG_INF
from .pytorch_reference import _compute_checkpoint_interval

# Triton is optional but will fall back to pytorch if not found.
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def _next_power_of_2(n: int) -> int:
    r"""_next_power_of_2(n) -> int

    Compute the smallest power of 2 greater than or equal to n.

    Used for padding tensor dimensions to powers of 2, which is required
    for efficient Triton kernel execution with vectorized loads.

    Args:
        n (int): Input value (must be positive for meaningful result).

    Returns:
        int: Smallest power of 2 :math:`\geq n`. Returns 1 for :math:`n \leq 0`.

    Examples::

        >>> _next_power_of_2(5)
        8
        >>> _next_power_of_2(8)
        8
        >>> _next_power_of_2(24)
        32
    """
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    p = 1
    while p < n:
        p *= 2
    return p


if HAS_TRITON:
    # NOTE: @triton.autotune removed - corrupts ring buffer during multi-config benchmarking.
    # See docs/debugging/DEBUGGING_NAN.md "Autotuning Limitations" for details on dev branch.
    @triton.jit
    def semi_crf_streaming_scan_kernel(
        # Inputs
        cum_scores_ptr,  # (batch, T+1, C) - cumulative projected scores
        transition_ptr,  # (C, C) or (K, C, C) - transition matrix
        duration_bias_ptr,  # (K, C) - duration-specific bias
        lengths_ptr,  # (batch,) - sequence lengths
        # Boundary projections (optional, may be null if HAS_BOUNDARIES=False)
        proj_start_ptr,  # (batch, T, C) - start boundary scores
        proj_end_ptr,  # (batch, T, C) - end boundary scores
        # Outputs
        out_ptr,  # (batch,) - partition function or Viterbi score
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer (read/write)
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints for backward
        # Backpointer outputs (only dereferenced when TRACK_BACKPOINTERS=True)
        bp_k_ptr,  # (batch, T, C) - best duration for each (t, c_dest)
        bp_c_ptr,  # (batch, T, C) - best source label for each (t, c_dest)
        final_labels_ptr,  # (batch,) - best final label (argmax of final alpha)
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length (T, not T+1)
        K: tl.constexpr,  # max segment duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        CHECKPOINT_INTERVAL: tl.constexpr,  # interval for saving ring buffer
        NUM_CKPTS: tl.constexpr,  # number of checkpoints
        HAS_BOUNDARIES: tl.constexpr,  # whether boundary projections are provided
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        # Strides for cum_scores (batch, T+1, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        # Strides for transition (C, C) or (K, C, C)
        stride_tr_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_tr_src,
        stride_tr_dst,
        # Strides for duration_bias (K, C)
        stride_db_k,
        stride_db_c,
        # Strides for proj_start/proj_end (batch, T, C) - only used if HAS_BOUNDARIES
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        # Strides for ring buffer (batch, K, C_PAD)
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        # Strides for ring checkpoints (batch, num_ckpts, K, C_PAD)
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
        # Log normalization checkpoints (only written when IS_MAX_SEMIRING=False)
        log_norm_ckpt_ptr,  # (batch, num_ckpts) - cumulative log normalization factors
        stride_lnc_b,
        stride_lnc_n,
        # Backpointer strides (only used when TRACK_BACKPOINTERS=True)
        stride_bp_b,
        stride_bp_t,
        stride_bp_c,
        # Semiring and mode flags (compile-time specialization, Flash Attention pattern)
        IS_MAX_SEMIRING: tl.constexpr,  # False=log (logsumexp), True=max (Viterbi)
        TRACK_BACKPOINTERS: tl.constexpr,  # True only when IS_MAX_SEMIRING and Viterbi decode
        EVEN_C: tl.constexpr,  # True when C is already a power of 2 (C == C_PAD)
        # Mixed precision support
        USE_FLOAT32: tl.constexpr = False,  # Use float32 for bulk compute
    ):
        r"""Unified streaming Semi-CRF forward scan kernel.

        Computes the forward pass for log and max semirings using an :math:`O(KC)`
        ring buffer. Semiring variant and backpointer mode are selected at compile
        time via ``IS_MAX_SEMIRING`` and ``TRACK_BACKPOINTERS`` constexpr flags.

        Edge potentials are computed on-the-fly via prefix-sum:

        .. math::
            \text{edge}[c_\text{dst}, c_\text{src}] =
            (\text{cum\_scores}[t, c_\text{dst}] - \text{cum\_scores}[t-k, c_\text{dst}])
            + \text{duration\_bias}[k, c_\text{dst}]
            + \text{transition}[c_\text{src}, c_\text{dst}]

        One program per batch element (``grid = (batch_size,)``).

        When ``IS_MAX_SEMIRING=False`` (log semiring), checkpoint normalization shifts
        alpha values at each checkpoint boundary to prevent unbounded growth at extreme T.
        ``accum_log_norm`` and the final partition value remain float64 regardless of
        ``USE_FLOAT32``.

        When ``IS_MAX_SEMIRING=True`` and ``TRACK_BACKPOINTERS=True``, the best duration
        ``k`` and source label ``c_src`` are recorded per :math:`(t, c_\text{dst})` for
        O(T) Viterbi traceback.
        """
        # Must match NEG_INF in constants.py
        NEG_INF: tl.constexpr = -1e9

        # Select compute dtype based on precision mode
        if USE_FLOAT32:
            COMPUTE_DTYPE: tl.constexpr = tl.float32
        else:
            COMPUTE_DTYPE: tl.constexpr = tl.float64

        # Batch index (one program per batch element)
        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        # 1D indices for labels (padded to power of 2)
        c_idx = tl.arange(0, C_PAD)

        # 2D indices for (C_dst, C_src) operations
        c_dst_idx = tl.arange(0, C_PAD)[:, None]  # (C_PAD, 1)
        c_src_idx = tl.arange(0, C_PAD)[None, :]  # (1, C_PAD)

        if EVEN_C:
            # C is already a power of 2: C == C_PAD, so every index in [0, C_PAD)
            # is valid. Define trivially-True masks so the compiler folds away all
            # downstream mask operations, and skip the min-clamping.
            c_mask = c_idx < C_PAD  # arange(0, C_PAD) < C_PAD — always True
            c_mask_2d = (c_dst_idx < C_PAD) & (c_src_idx < C_PAD)  # ditto
            c_idx_safe = c_idx
            c_dst_idx_safe = c_dst_idx
            c_src_idx_safe = c_src_idx
        else:
            c_mask = c_idx < C
            c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)
            # Clamp indices so masked threads do not form out-of-bounds addresses.
            #
            # IMPORTANT: Even with a load/store mask, Triton may still evaluate address
            # arithmetic for masked lanes. When C_PAD > C (non-power-of-2 class counts),
            # raw c_idx/c_dst_idx/c_src_idx can form invalid pointers for tensors whose
            # last dimension is C. This is undefined behavior and can appear as small,
            # configuration-dependent numerical drift (for example after unrelated cache
            # invalidation/recompilation). Always use *_safe indices for C-shaped tensors.
            c_idx_safe = tl.minimum(c_idx, C - 1)
            c_dst_idx_safe = tl.minimum(c_dst_idx, C - 1)
            c_src_idx_safe = tl.minimum(c_src_idx, C - 1)

        # Load sequence length
        # Cast to int32 to match loop variable type from tl.range (int32)
        # Avoids silent comparison failures between int32 and int64
        seq_len = tl.load(lengths_ptr + batch_idx).to(tl.int32)

        # Base pointers
        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        # Static transition matrix: load once before the t-loop.
        # Duration-dependent transitions (K, C, C) are loaded inside the k-loop.
        #
        # The tensor is stored with convention transition[c_src, c_dst], but the
        # edge formula needs edge[c_dst, c_src] = segment_score[c_dst] + transition[c_src, c_dst].
        # Loading with c_dst and c_src strides swapped produces the (C_PAD, C_PAD) matrix
        # indexed as [c_dst, c_src], i.e., transition transposed. No explicit tl.trans needed.
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx_safe * stride_tr_dst + c_src_idx_safe * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )  # (C_PAD, C_PAD) - this is transition.T

        # Ring buffer and checkpoint 0 are pre-initialized by the launcher:
        # - ring_buffer[:, 0, :C] = 0.0, rest = NEG_INF
        # - ring_checkpoints[:, 0, 0, :C] = 0.0, rest = NEG_INF
        # This avoids K iterations of conditional writes per batch element.

        # Track final alpha for each batch element
        final_alpha = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)

        # Log semiring: cumulative log normalization factor for numerical stability.
        # Tracks the total shift applied to alpha values at checkpoint boundaries.
        # MUST stay float64: grows to O(T) magnitude at extreme T.
        if not IS_MAX_SEMIRING:
            accum_log_norm = tl.zeros((), dtype=tl.float64)

        # Backpointer base offset (pre-computed outside t-loop)
        if TRACK_BACKPOINTERS:
            bp_base = batch_idx * stride_bp_b

        # Main forward loop: t = 1, 2, ..., T
        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            # Cast t to int32 to match seq_len type for consistent comparison
            active = t.to(tl.int32) <= seq_len

            # === Accumulator initialization (semiring-specific) ===
            if not IS_MAX_SEMIRING:
                # Log semiring: online logsumexp accumulators for alpha[t] over (k, c_src)
                m_alpha = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)
                l_alpha = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)
            else:
                # Max semiring: direct max accumulator
                alpha_t = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)
                if TRACK_BACKPOINTERS:
                    # Initialize backpointer tracking for this timestep
                    best_k_t = tl.zeros([C_PAD], dtype=tl.int32)
                    best_c_src_t = tl.zeros([C_PAD], dtype=tl.int32)

            # === Inner k-loop: accumulate over segment durations ===
            # Loop over valid segment durations k = 1, 2, ..., min(K, t)
            for k in tl.range(1, K + 1):
                # Duration k uses index k-1 in duration_bias
                k_valid = (k <= t) & (k <= K)
                start_pos = t - k

                # Ring slot for alpha[start_pos].
                # The ring has K slots addressed by position % K. At each iteration t,
                # alpha[t] is written to slot t % K. For k in [1, K], start_pos = t - k
                # ranges over t-1, t-2, ..., t-K. These K consecutive integers have K
                # distinct residues mod K, so each occupies a different slot and none
                # has been overwritten by a later timestep. The slot is start_pos % K.
                ring_k_idx = start_pos % K

                # Load alpha_prev from live ring buffer
                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )  # (C_PAD,) - alpha[start_pos, c_src]

                # === Compute edge block on-the-fly (prefix-sum) ===

                # Load cum_scores[t, :] and cum_scores[start_pos, :]
                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx_safe * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                # Content score = cum_scores[t, c_dst] - cum_scores[start, c_dst]
                content_score = cum_end - cum_start  # (C_PAD,)

                # Load duration bias: duration k uses index k-1
                dur_idx = k - 1
                dur_bias = tl.load(
                    duration_bias_ptr + dur_idx * stride_db_k + c_idx_safe * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )  # (C_PAD,)

                # Segment score = content_score + duration_bias
                segment_score = content_score + dur_bias  # (C_PAD,)

                # Add boundary scores if provided
                # Segment starts at start_pos, ends at t-1 (inclusive)
                if HAS_BOUNDARIES:
                    # proj_start[start_pos, :] - start boundary score
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx_safe * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    # proj_end[t-1, :] - end boundary score (t-1 is last position in segment)
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx_safe * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

                # Edge block: edge[c_dst, c_src] = segment_score[c_dst] + transition[c_src, c_dst]
                # segment_score is (C_PAD,), expand to (C_PAD, 1) for c_dst
                # transition_block is (C_PAD, C_PAD) as transition.T

                # For duration-dependent transitions, load transition[dur_idx] inside the loop
                # Duration k uses index k-1 (same convention as PyTorch reference)
                if HAS_DURATION_TRANSITIONS:
                    transition_block = tl.load(
                        transition_ptr
                        + dur_idx * stride_tr_k
                        + c_dst_idx_safe * stride_tr_dst
                        + c_src_idx_safe * stride_tr_src,
                        mask=c_mask_2d,
                        other=0.0,
                    )  # (C_PAD, C_PAD) - transition[dur_idx].T

                edge_block = segment_score[:, None] + transition_block  # (C_PAD, C_PAD)

                # === Compute scores and semiring reduction ===
                # scores[c_dst, c_src] = alpha_prev[c_src] + edge[c_dst, c_src]
                scores = alpha_prev[None, :] + edge_block  # (C_PAD, C_PAD)

                # Mask out invalid entries
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                if not IS_MAX_SEMIRING:
                    # Log semiring: logsumexp over c_src (axis=1) -> score_for_k (C_PAD,).
                    # The all-NEG_INF guard handles t < k (no valid start position):
                    # max_scores would be NEG_INF, so exp(NEG_INF - 0) underflows to 0,
                    # giving log(0) = nan. Return NEG_INF directly in that case.
                    max_scores = tl.max(scores, axis=1)
                    is_all_neginf = max_scores < (NEG_INF + 1.0)
                    max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
                    log_sum_exp = tl.log(tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1))
                    score_for_k = tl.where(is_all_neginf, NEG_INF, max_scores + log_sum_exp)

                    # Mask invalid durations and labels
                    score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                    # Online logsumexp: accumulate score_for_k into (m_alpha, l_alpha).
                    # Representation: alpha[t] = m_alpha + log(l_alpha), where m_alpha is
                    # the running max and l_alpha = sum(exp(x - m_alpha)) for each x seen.
                    # Cost: 2 exp + 0 log per k, vs 2 exp + 1 log per k for pairwise logsumexp.
                    score_for_k = score_for_k.to(COMPUTE_DTYPE)
                    m_new = tl.maximum(m_alpha, score_for_k)
                    is_m_neginf = m_alpha < (NEG_INF + 1.0)
                    is_score_neginf = score_for_k < (NEG_INF + 1.0)
                    m_new_safe = tl.where(is_m_neginf & is_score_neginf, 0.0, m_new)
                    # is_m_neginf means the accumulator is still empty (first valid k).
                    # The standard update l_alpha * exp(m_alpha - m_new_safe) = 0 * exp(-inf - ...)
                    # is mathematically 0 but can produce nan on some hardware. Start fresh.
                    l_alpha = tl.where(
                        is_m_neginf,
                        tl.exp(score_for_k - m_new_safe),
                        l_alpha * tl.exp(m_alpha - m_new_safe) + tl.exp(score_for_k - m_new_safe),
                    )
                    m_alpha = m_new

                else:
                    # Max semiring: max over c_src
                    score_for_k = tl.max(scores, axis=1)
                    if TRACK_BACKPOINTERS:
                        argmax_c_src = tl.argmax(scores, axis=1)
                    score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                    if TRACK_BACKPOINTERS:
                        # Track which k and c_src win for each c_dst
                        better_mask = score_for_k > alpha_t
                        best_k_t = tl.where(better_mask, k, best_k_t)
                        best_c_src_t = tl.where(better_mask, argmax_c_src, best_c_src_t)

                    # Max over k
                    alpha_t = tl.maximum(alpha_t, score_for_k)

            # === Finalize alpha_t for this timestep ===
            if not IS_MAX_SEMIRING:
                # Finalize online logsumexp: alpha_t = m + log(l)
                is_all_neginf = m_alpha < (NEG_INF + 1.0)
                # Guard: tl.log(l_alpha) is only safe when l_alpha > 0.
                # When all inputs were NEG_INF, l_alpha == 0 and log(0) = -inf/NaN.
                # The is_all_neginf guard short-circuits to NEG_INF before evaluating log.
                alpha_t = tl.where(is_all_neginf, NEG_INF, m_alpha + tl.log(l_alpha))
                alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)
            else:
                alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # === Store backpointers at position t-1 (0-indexed) ===
            if TRACK_BACKPOINTERS:
                bp_pos = t - 1
                tl.store(
                    bp_k_ptr + bp_base + bp_pos * stride_bp_t + c_idx_safe * stride_bp_c,
                    best_k_t,
                    mask=active & c_mask,
                )
                tl.store(
                    bp_c_ptr + bp_base + bp_pos * stride_bp_t + c_idx_safe * stride_bp_c,
                    best_c_src_t,
                    mask=active & c_mask,
                )

            # === Store to live ring buffer ===
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Memory barrier: ensure all warps have committed their ring buffer writes
            # before any warp begins the next t-iteration's ring loads.
            # Without this, at C_PAD = NVIDIA warp size (32) with num_warps > 1,
            # the ring buffer write is split across warps (8 elements each at num_warps=4).
            # A warp starting t+1 can then load stale values from the previous iteration
            # before the other warps have finished their stores for t.
            # Confirmed race: C=32 produced wrong partitions; all other C values passed
            # because C < 32 fits in 1 warp (no cross-warp issue) and C > 32 has enough
            # implicit pipeline stall from the additional instructions between store and load.
            # The backward kernel has an identical barrier after its beta ring store.
            tl.debug_barrier()

            # === Save checkpoint at interval boundaries ===
            # Checkpoint i stores the ring buffer state at position i * CHECKPOINT_INTERVAL
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
                if not IS_MAX_SEMIRING:
                    # ===== NORMALIZATION STEP (for numerical stability at extreme T) =====
                    # Following Flash Attention pattern: normalize at checkpoints to prevent
                    # alpha values from growing unbounded (e.g., to ±250,000 at T=100k)
                    #
                    # CRITICAL: Only normalize for ACTIVE sequences (t <= seq_len)
                    # For ended sequences, we must NOT update accum_log_norm or the
                    # partition function will be wrong (phantom shifts would be added)

                    # 1. Find max alpha value for normalization (over valid C only)
                    #    For inactive sequences, use NEG_INF to produce shift=0
                    alpha_for_norm = tl.where(active & c_mask, alpha_t, NEG_INF)
                    max_val = tl.max(alpha_for_norm)

                    # Use `active` rather than `max_val < (NEG_INF + 1.0)` to determine shift.
                    # For inactive sequences (t > seq_len) all alpha values are NEG_INF, so
                    # max_val is NEG_INF, but the float comparison against the NEG_INF constant
                    # produces incorrect results in Triton due to type ambiguity between the
                    # int32 loop predicate and the float64 max_val. `active` is the correct gate.
                    shift = tl.where(active, max_val, 0.0)

                    # 2. Update cumulative normalization factor
                    #    shift is 0 for inactive sequences, so no phantom updates
                    accum_log_norm = accum_log_norm + shift

                    # 3. CRITICAL: Update alpha_t register BEFORE final_alpha capture
                    #    This ensures consistency if seq_len falls on a checkpoint boundary
                    #    For inactive sequences, shift=0 so no change
                    alpha_t = alpha_t - shift

                    # 4. Normalize ALL K slots in ring buffer
                    #    The ring buffer is used for K more iterations after checkpoint,
                    #    so all slots must be shifted to maintain consistency
                    #    Only update for active sequences
                    for k_norm in tl.range(0, K):
                        ring_val = tl.load(
                            ring_base + k_norm * stride_ring_k + c_idx * stride_ring_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )
                        ring_val_shifted = ring_val - shift
                        tl.store(
                            ring_base + k_norm * stride_ring_k + c_idx * stride_ring_c,
                            ring_val_shifted,
                            mask=active & c_mask,  # Only update for active sequences
                        )

                    # 5. Save normalized ring buffer to checkpoint
                    #    Only save for active sequences with valid checkpoint index
                    for k_save in tl.range(0, K):
                        ring_val = tl.load(
                            ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )
                        # Only save if checkpoint index is valid AND sequence is active
                        save_mask = (ckpt_idx < NUM_CKPTS) & active & c_mask
                        tl.store(
                            ring_ckpt_base
                            + ckpt_idx * stride_ckpt_n
                            + k_save * stride_ckpt_k
                            + c_idx * stride_ckpt_c,
                            ring_val,
                            mask=save_mask,
                        )

                    # 6. Save cumulative log normalization factor
                    #    Only save for active sequences
                    if (ckpt_idx < NUM_CKPTS) & active:
                        tl.store(
                            log_norm_ckpt_ptr + batch_idx * stride_lnc_b + ckpt_idx * stride_lnc_n,
                            accum_log_norm,
                        )

                else:
                    # Max semiring: simple ring buffer checkpoint (no normalization needed)
                    # Only save for active sequences to avoid stale data
                    for k_save in tl.range(0, K):
                        ring_val = tl.load(
                            ring_base + k_save * stride_ring_k + c_idx * stride_ring_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )
                        # Only save if checkpoint index is valid AND sequence is active
                        save_mask = (ckpt_idx < NUM_CKPTS) & active & c_mask
                        tl.store(
                            ring_ckpt_base
                            + ckpt_idx * stride_ckpt_n
                            + k_save * stride_ckpt_k
                            + c_idx * stride_ckpt_c,
                            ring_val,
                            mask=save_mask,
                        )

            # Capture final alpha at sequence end (t == seq_len)
            # At iteration t, alpha_t represents segments ending at position t-1
            # For sequence of length L, we need alpha at t=L (segments ending at L-1)
            # Cast t to int32 to match seq_len type for consistent comparison
            is_final = t.to(tl.int32) == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # === Final reduction and output ===
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)

        if not IS_MAX_SEMIRING:
            # Log semiring: logsumexp over labels, then restore the cumulative normalization.
            # All-NEG_INF guard: if no valid path reached seq_len, log(sum(exp(-inf))) = log(0) = nan.
            max_val = tl.max(final_alpha_masked, axis=0)
            is_final_neginf = max_val < (NEG_INF + 1.0)
            max_val_safe = tl.where(is_final_neginf, 0.0, max_val)
            exp_fa = tl.where(c_mask, tl.exp(final_alpha - max_val_safe), 0.0)
            sum_exp = tl.sum(exp_fa, axis=0)
            raw_partition = tl.where(is_final_neginf, NEG_INF, max_val + tl.log(sum_exp))

            # Restore cumulative normalization to get the true partition function.
            # final_alpha holds normalized values; accum_log_norm is the total shift removed.
            # raw_partition is in COMPUTE_DTYPE (possibly float32). Cast to float64 before
            # adding accum_log_norm: at extreme T, accum_log_norm reaches ~250K, and adding
            # it to a float32 value loses low-order mantissa bits.
            partition = (raw_partition.to(tl.float64) + accum_log_norm).to(tl.float64)

            # DEBUG: Print partition (log_Z) for T=100k diagnostic
            # Uncomment these lines to diagnose numerical stability at extreme scale
            # if batch_idx == 0:
            #     tl.device_print("FWD log_Z=", partition)
            #     tl.device_print("FWD max_final_alpha=", max_val)

            tl.store(out_ptr + batch_idx, partition)

        else:
            # Max semiring: max over labels
            partition = tl.max(final_alpha_masked, axis=0)
            tl.store(out_ptr + batch_idx, partition)

            if TRACK_BACKPOINTERS:
                final_label = tl.argmax(final_alpha_masked, axis=0)
                tl.store(final_labels_ptr + batch_idx, final_label)

    def _prepare_forward_buffers(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        checkpoint_interval: int = None,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
        num_warps: int = 4,
        validate_cache: bool = True,
        precision: str = "float64",
    ) -> dict:
        """Shared buffer allocation and stride preparation for all forward kernel launchers.

        Handles cache validation, shape validation, contiguity enforcement, checkpoint
        interval computation, ring buffer allocation, and stride computation. The two
        public launcher functions call this helper and then add their semiring-specific
        outputs before dispatching the unified kernel.

        Returns:
            dict with keys: cum_scores, transition, duration_bias, lengths,
            proj_start, proj_end, ring_buffer, ring_checkpoints, batch, T, C,
            C_PAD, device, use_float32, compute_dtype, checkpoint_interval,
            num_checkpoints, has_boundaries, has_duration_transitions, and all
            stride_* values for shared tensor arguments.
        """
        from .triton_cache import TritonConfig, update_cache_sentinel, validate_triton_cache

        # Validate cache if requested
        if validate_cache:
            config = TritonConfig(num_warps=num_warps)
            validate_triton_cache(config)
            update_cache_sentinel(config)

        batch, T_plus_1, C = cum_scores.shape
        T = T_plus_1 - 1
        validate_streaming_shapes(K, C, batch, T, transition, duration_bias, proj_start, proj_end)
        device = cum_scores.device

        # Select dtype based on precision mode
        use_float32 = precision == "float32"
        compute_dtype = torch.float32 if use_float32 else torch.float64

        # Compute checkpoint interval if not provided
        if checkpoint_interval is None:
            checkpoint_interval = _compute_checkpoint_interval(
                T, K, compute_dtype="float32" if use_float32 else "float64"
            )
        else:
            checkpoint_interval = max(checkpoint_interval, K)

        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval

        # Pad C to next power of 2
        C_PAD = _next_power_of_2(C)

        # Validate boundary projections: require both or neither
        if (proj_start is None) != (proj_end is None):
            raise ValueError(
                "Triton kernels require both proj_start and proj_end, or neither. "
                f"Got proj_start={'provided' if proj_start is not None else 'None'}, "
                f"proj_end={'provided' if proj_end is not None else 'None'}. "
                "Use semi_crf_streaming_forward() for automatic PyTorch fallback."
            )

        # Determine if boundaries are provided
        has_boundaries = proj_start is not None and proj_end is not None

        # Determine if duration-dependent transitions
        has_duration_transitions = transition.ndim == 3

        # Ensure inputs are contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()

        # Handle boundary projections
        if has_boundaries:
            proj_start = proj_start.contiguous()
            proj_end = proj_end.contiguous()
            stride_ps_b, stride_ps_t, stride_ps_c = proj_start.stride()
        else:
            # Create dummy tensor for stride calculation (won't be accessed)
            proj_start = cum_scores[:, :T, :]  # Reuse cum_scores memory, won't be accessed
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0  # Strides don't matter when not used

        # Live ring buffer (will be L1/L2 cached for small K*C)
        # Initialize to NEG_INF, then set k=0 to 0.0 (initial alpha state)
        ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=compute_dtype)
        ring_buffer[:, 0, :C] = 0.0  # alpha[0, c] = 0.0 for all valid labels

        # Checkpoint storage for backward pass
        # Initialize to NEG_INF, then set checkpoint 0, k=0 to 0.0
        ring_checkpoints = torch.full(
            (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=compute_dtype
        )
        ring_checkpoints[:, 0, 0, :C] = 0.0  # Initial state at checkpoint 0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        # Handle transition strides for both (C, C) and (K, C, C)
        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0  # Not used for static transitions
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ring_b, stride_ring_k, stride_ring_c = ring_buffer.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_checkpoints.stride()

        return {
            "cum_scores": cum_scores,
            "even_c": (C == C_PAD),
            "transition": transition,
            "duration_bias": duration_bias,
            "lengths": lengths,
            "proj_start": proj_start,
            "proj_end": proj_end,
            "ring_buffer": ring_buffer,
            "ring_checkpoints": ring_checkpoints,
            "batch": batch,
            "T": T,
            "C": C,
            "C_PAD": C_PAD,
            "device": device,
            "use_float32": use_float32,
            "compute_dtype": compute_dtype,
            "checkpoint_interval": checkpoint_interval,
            "num_checkpoints": num_checkpoints,
            "has_boundaries": has_boundaries,
            "has_duration_transitions": has_duration_transitions,
            "stride_cs_b": stride_cs_b,
            "stride_cs_t": stride_cs_t,
            "stride_cs_c": stride_cs_c,
            "stride_tr_k": stride_tr_k,
            "stride_tr_src": stride_tr_src,
            "stride_tr_dst": stride_tr_dst,
            "stride_db_k": stride_db_k,
            "stride_db_c": stride_db_c,
            "stride_ps_b": stride_ps_b,
            "stride_ps_t": stride_ps_t,
            "stride_ps_c": stride_ps_c,
            "stride_ring_b": stride_ring_b,
            "stride_ring_k": stride_ring_k,
            "stride_ring_c": stride_ring_c,
            "stride_ckpt_b": stride_ckpt_b,
            "stride_ckpt_n": stride_ckpt_n,
            "stride_ckpt_k": stride_ckpt_k,
            "stride_ckpt_c": stride_ckpt_c,
        }

    def launch_streaming_triton_kernel(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        semiring: str = "log",
        checkpoint_interval: int = None,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
        num_warps: int = 4,
        validate_cache: bool = True,
        precision: str = "float64",
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        r"""launch_streaming_triton_kernel(cum_scores, transition, duration_bias, lengths, K, semiring="log", checkpoint_interval=None, proj_start=None, proj_end=None, num_warps=4, validate_cache=True) -> tuple[Tensor, Tensor, int, Tensor]

        Launch the streaming Triton kernel with proper buffer allocation.

        This function allocates the required buffers (ring buffer, checkpoints)
        and dispatches the appropriate Triton kernel based on the semiring.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` for
                static transitions, or :math:`(K, C, C)` for duration-dependent.
            duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            K (int): Maximum segment duration.
            semiring (str, optional): ``"log"`` or ``"max"``. Default: ``"log"``
            checkpoint_interval (int, optional): Interval for saving ring buffer.
                If ``None``, uses :math:`\sqrt{T \times K}`. Default: ``None``
            proj_start (Tensor, optional): Start boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            proj_end (Tensor, optional): End boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            num_warps (int, optional): Number of warps per block for Triton kernel.
                Higher values increase parallelism but also register pressure.
                Recommended range: 2-8. Default: ``4``
            validate_cache (bool, optional): If True, validate Triton cache
                consistency and warn on config changes. Default: ``True``

        Returns:
            tuple[Tensor, Tensor, int, Tensor]: Tuple of:
                - **partition** (Tensor): Partition function (log-space) or Viterbi scores
                  (max-space) of shape :math:`(\text{batch},)`.
                - **ring_checkpoints** (Tensor): Saved ring buffer states for backward pass
                  of shape :math:`(\text{batch}, \text{num\_ckpts}, K, C)`.
                - **checkpoint_interval** (int): Actual checkpoint interval used.
                - **log_norm_checkpoints** (Tensor): Cumulative log normalization factors
                  at each checkpoint of shape :math:`(\text{batch}, \text{num\_ckpts})`.

        .. note::
            Requires both or neither boundary tensor (proj_start, proj_end).
            For single-boundary support, use :func:`semi_crf_streaming_forward`
            which automatically falls back to the PyTorch path.

        See Also:
            :func:`launch_streaming_triton_backward`: Backward pass that uses checkpoints
                from this forward pass.
            :func:`launch_streaming_triton_kernel_max_bp`: Max-semiring variant with
                backpointer tracking for Viterbi decoding.
        """
        b = _prepare_forward_buffers(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            checkpoint_interval=checkpoint_interval,
            proj_start=proj_start,
            proj_end=proj_end,
            num_warps=num_warps,
            validate_cache=validate_cache,
            precision=precision,
        )

        # Partition always float64 (scalar per batch, adds accum_log_norm which is O(T))
        partition = torch.empty(b["batch"], device=b["device"], dtype=torch.float64)

        # Log normalization checkpoint storage for numerical stability at extreme T
        # Stores cumulative log normalization factor at each checkpoint boundary
        # MUST stay float64: these scalars grow to O(T) magnitude
        log_norm_checkpoints = torch.zeros(
            (b["batch"], b["num_checkpoints"]), device=b["device"], dtype=torch.float64
        )
        stride_lnc_b, stride_lnc_n = log_norm_checkpoints.stride()

        # Dummy backpointer tensors (TRACK_BACKPOINTERS=False — never dereferenced)
        _dummy_bp = torch.zeros(1, device=b["device"], dtype=torch.int32)

        is_max = semiring == "max"
        grid = (b["batch"],)
        with torch.cuda.device(b["device"]):
            semi_crf_streaming_scan_kernel[grid](
                b["cum_scores"],
                b["transition"],
                b["duration_bias"],
                b["lengths"],
                b["proj_start"],
                b["proj_end"],
                partition,
                b["ring_buffer"],
                b["ring_checkpoints"],
                _dummy_bp,  # bp_k_ptr (unused)
                _dummy_bp,  # bp_c_ptr (unused)
                _dummy_bp,  # final_labels_ptr (unused)
                b["batch"],
                b["T"],
                K,
                b["C"],
                b["C_PAD"],
                b["checkpoint_interval"],
                b["num_checkpoints"],
                b["has_boundaries"],
                b["has_duration_transitions"],
                b["stride_cs_b"],
                b["stride_cs_t"],
                b["stride_cs_c"],
                b["stride_tr_k"],
                b["stride_tr_src"],
                b["stride_tr_dst"],
                b["stride_db_k"],
                b["stride_db_c"],
                b["stride_ps_b"],
                b["stride_ps_t"],
                b["stride_ps_c"],
                b["stride_ring_b"],
                b["stride_ring_k"],
                b["stride_ring_c"],
                b["stride_ckpt_b"],
                b["stride_ckpt_n"],
                b["stride_ckpt_k"],
                b["stride_ckpt_c"],
                log_norm_checkpoints,
                stride_lnc_b,
                stride_lnc_n,
                0,  # stride_bp_b (unused)
                0,  # stride_bp_t (unused)
                0,  # stride_bp_c (unused)
                IS_MAX_SEMIRING=is_max,
                TRACK_BACKPOINTERS=False,
                EVEN_C=b["even_c"],
                USE_FLOAT32=b["use_float32"],
                num_warps=num_warps,
            )

        # Trim padding from checkpoints for return
        ring_checkpoints = b["ring_checkpoints"][:, :, :, : b["C"]]

        return partition, ring_checkpoints, b["checkpoint_interval"], log_norm_checkpoints

    def launch_streaming_triton_kernel_max_bp(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        checkpoint_interval: int = None,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
        num_warps: int = 4,
        validate_cache: bool = True,
        precision: str = "float64",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Launch Triton max kernel with backpointer tracking for Viterbi decoding.

        This function launches the backpointer-enabled max kernel for O(T) traceback.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` for
                static transitions, or :math:`(K, C, C)` for duration-dependent.
            duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            K (int): Maximum segment duration.
            checkpoint_interval (int, optional): Interval for saving ring buffer.
                If ``None``, uses :math:`\sqrt{T \times K}`. Default: ``None``
            proj_start (Tensor, optional): Start boundary scores. Default: ``None``
            proj_end (Tensor, optional): End boundary scores. Default: ``None``
            num_warps (int, optional): Number of warps per block for Triton kernel.
                Higher values increase parallelism but also register pressure.
                Recommended range: 2-8. Default: ``4``
            validate_cache (bool, optional): If True, validate Triton cache
                consistency and warn on config changes. Default: ``True``

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: Tuple of:
                - **viterbi_scores** (Tensor): Best path scores of shape
                  :math:`(\text{batch},)`.
                - **bp_k** (Tensor): Best duration backpointers for each
                  :math:`(t, c_{\text{dest}})` of shape :math:`(\text{batch}, T, C)`.
                - **bp_c** (Tensor): Best source label backpointers for each
                  :math:`(t, c_{\text{dest}})` of shape :math:`(\text{batch}, T, C)`.
                - **final_labels** (Tensor): Best final label (argmax of final alpha)
                  of shape :math:`(\text{batch},)`.

        .. note::
            Requires both or neither boundary tensor (proj_start, proj_end).
            For single-boundary support, use :func:`semi_crf_streaming_forward`
            which automatically falls back to the PyTorch path.

        See Also:
            :func:`launch_streaming_triton_kernel`: Log-semiring variant for computing
                partition function and gradients.
        """
        b = _prepare_forward_buffers(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            checkpoint_interval=checkpoint_interval,
            proj_start=proj_start,
            proj_end=proj_end,
            num_warps=num_warps,
            validate_cache=validate_cache,
            precision=precision,
        )

        # Allocate outputs
        partition = torch.empty(b["batch"], device=b["device"], dtype=b["compute_dtype"])
        bp_k = torch.zeros((b["batch"], b["T"], b["C"]), device=b["device"], dtype=torch.int32)
        bp_c = torch.zeros((b["batch"], b["T"], b["C"]), device=b["device"], dtype=torch.int32)
        final_labels = torch.zeros(b["batch"], device=b["device"], dtype=torch.int64)

        stride_bp_b, stride_bp_t, stride_bp_c = bp_k.stride()

        # Dummy log_norm tensor (IS_MAX_SEMIRING=True — never written to)
        _dummy_lnc = torch.zeros(1, device=b["device"], dtype=torch.float64)

        grid = (b["batch"],)
        with torch.cuda.device(b["device"]):
            semi_crf_streaming_scan_kernel[grid](
                b["cum_scores"],
                b["transition"],
                b["duration_bias"],
                b["lengths"],
                b["proj_start"],
                b["proj_end"],
                partition,
                b["ring_buffer"],
                b["ring_checkpoints"],
                bp_k,
                bp_c,
                final_labels,
                b["batch"],
                b["T"],
                K,
                b["C"],
                b["C_PAD"],
                b["checkpoint_interval"],
                b["num_checkpoints"],
                b["has_boundaries"],
                b["has_duration_transitions"],
                b["stride_cs_b"],
                b["stride_cs_t"],
                b["stride_cs_c"],
                b["stride_tr_k"],
                b["stride_tr_src"],
                b["stride_tr_dst"],
                b["stride_db_k"],
                b["stride_db_c"],
                b["stride_ps_b"],
                b["stride_ps_t"],
                b["stride_ps_c"],
                b["stride_ring_b"],
                b["stride_ring_k"],
                b["stride_ring_c"],
                b["stride_ckpt_b"],
                b["stride_ckpt_n"],
                b["stride_ckpt_k"],
                b["stride_ckpt_c"],
                _dummy_lnc,  # log_norm_ckpt_ptr (unused)
                0,  # stride_lnc_b (unused)
                0,  # stride_lnc_n (unused)
                stride_bp_b,
                stride_bp_t,
                stride_bp_c,
                IS_MAX_SEMIRING=True,
                TRACK_BACKPOINTERS=True,
                EVEN_C=b["even_c"],
                USE_FLOAT32=b["use_float32"],
                num_warps=num_warps,
            )

        return partition, bp_k, bp_c, final_labels

    def semi_crf_streaming_viterbi_triton(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        K: int,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""semi_crf_streaming_viterbi_triton(cum_scores, transition, duration_bias, lengths, K, proj_start=None, proj_end=None) -> tuple[Tensor, Tensor, Tensor, Tensor]

        Triton-accelerated Viterbi forward pass with backpointer tracking.

        This is the GPU-accelerated equivalent of
        :func:`semi_crf_streaming_viterbi_with_backpointers` from pytorch_reference.py.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` or
                :math:`(K, C, C)`.
            duration_bias (Tensor): Duration bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            K (int): Maximum segment duration.
            proj_start (Tensor, optional): Start boundary scores. Default: ``None``
            proj_end (Tensor, optional): End boundary scores. Default: ``None``

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: Same as
                :func:`semi_crf_streaming_viterbi_with_backpointers`:
                - **viterbi_scores** (Tensor): Best path scores ``(batch,)``.
                - **bp_k** (Tensor): Best durations ``(batch, T, C)``.
                - **bp_c** (Tensor): Best source labels ``(batch, T, C)``.
                - **final_labels** (Tensor): Best final labels ``(batch,)``.
        """
        return launch_streaming_triton_kernel_max_bp(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            checkpoint_interval=None,
            proj_start=proj_start,
            proj_end=proj_end,
        )
