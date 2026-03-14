r"""Triton forward kernels for streaming Semi-CRF.

This module contains the Triton kernels for the forward pass (log and max semiring)
and the launcher function that allocates buffers and dispatches the kernels.

The kernels implement streaming edge computation where edge potentials are
computed on-the-fly from cumulative scores via prefix-sum decomposition:

.. math::
    \text{edge}[c_{\text{dst}}, c_{\text{src}}] =
    (\text{cum\_scores}[t+k, c_{\text{dst}}] - \text{cum\_scores}[t, c_{\text{dst}}])
    + \text{duration\_bias}[k, c_{\text{dst}}]
    + \text{transition}[c_{\text{src}}, c_{\text{dst}}]

.. warning::
    **Minimum K Requirement**: These kernels require K >= 3 for correct operation.
    The ring buffer architecture assumes meaningful separation between timesteps:

    - **K=1**: Ring buffer aliasing (all t % 1 = 0). Use ``LinearCRFStreaming`` instead.
    - **K=2**: Ring buffer fragility (alternating slots). Use ``SemiCRFK2Streaming`` instead.
    - **K>=3**: Sufficient ring rotation for stable operation.

    The dispatch logic in ``autograd.py`` automatically routes K<3 to specialized
    PyTorch implementations. Do not call these kernels directly with K<3.

Functions:
    launch_streaming_triton_kernel: Main entry point for launching Triton forward kernels.
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
        out_ptr,  # (batch,) - partition function
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer (read/write)
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints for backward
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
        # Log normalization checkpoints for numerical stability at extreme T
        log_norm_ckpt_ptr,  # (batch, num_ckpts) - cumulative log normalization factors
        stride_lnc_b,
        stride_lnc_n,
        # Mixed precision support
        USE_FLOAT32: tl.constexpr = False,  # Use float32 for bulk compute (ring buffer, alpha)
    ):
        r"""Streaming Semi-CRF forward scan with on-the-fly edge computation (log semiring).

        This Triton kernel computes the forward pass of the Semi-CRF using logsumexp
        reductions. Edge potentials are computed on-the-fly from cumulative scores:

        .. code-block:: text

            edge[c_dst, c_src] = (cum_scores[t+k, c_dst] - cum_scores[t, c_dst])
                               + duration_bias[k, c_dst]
                               + transition[c_src, c_dst]

        Memory: Uses a ring buffer for alpha values with :math:`O(KC)` memory.
        Saves ring buffer checkpoints at regular intervals for backward pass.

        One program is launched per batch element (grid size = batch_size).

        When ``USE_FLOAT32=True``, bulk computation (ring buffer, alpha accumulators,
        logsumexp) uses float32 instead of float64. Scalar accumulators
        (``accum_log_norm``, partition) remain float64 for numerical stability.
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
        c_mask = c_idx < C

        # 2D indices for (C_dst, C_src) operations
        c_dst_idx = tl.arange(0, C_PAD)[:, None]  # (C_PAD, 1)
        c_src_idx = tl.arange(0, C_PAD)[None, :]  # (1, C_PAD)
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

        # Load transition matrix into registers: (C_PAD, C_PAD)
        # For static transitions (C, C): load once here
        # For duration-dependent (K, C, C): load inside k-loop
        # transition[c_src, c_dst] -> we need transition.T for edge computation
        # So we load transition_ptr[c_dst, c_src] effectively
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

        # Cumulative log normalization factor for numerical stability at extreme T
        # This tracks the total shift applied to alpha values at checkpoint boundaries
        # MUST stay float64: grows to O(T) magnitude at extreme T
        accum_log_norm = tl.zeros((), dtype=tl.float64)

        # Main forward loop: t = 1, 2, ..., T
        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            # Cast t to int32 to match seq_len type for consistent comparison
            active = t.to(tl.int32) <= seq_len

            # Online logsumexp accumulators for alpha[t] over (k, c_src)
            m_alpha = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)
            l_alpha = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)

            # Loop over valid segment durations k = 1, 2, ..., min(K, t)
            for k in tl.range(1, K + 1):
                # Duration k uses index k-1 in duration_bias
                k_valid = (k <= t) & (k <= K)
                start_pos = t - k

                # Ring index for alpha[start_pos]
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

                # === Compute scores and reduction ===
                # scores[c_dst, c_src] = alpha_prev[c_src] + edge[c_dst, c_src]
                scores = alpha_prev[None, :] + edge_block  # (C_PAD, C_PAD)

                # Mask out invalid entries
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Logsumexp over c_src (axis=1) -> (C_PAD,)
                # Guard against all-NEG_INF case to prevent undefined arithmetic
                # Flash Attention pattern: no epsilon needed inside log
                # The NEG_INF guard handles the zero case
                max_scores = tl.max(scores, axis=1)
                is_all_neginf = max_scores < (NEG_INF + 1.0)
                max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
                log_sum_exp = tl.log(tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1))
                score_for_k = tl.where(is_all_neginf, NEG_INF, max_scores + log_sum_exp)

                # Mask invalid durations and labels
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Online logsumexp: accumulate score_for_k into (m_alpha, l_alpha)
                # 2 exp + 0 log per k (vs pairwise: 2 exp + 1 log per k)
                score_for_k = score_for_k.to(COMPUTE_DTYPE)
                m_new = tl.maximum(m_alpha, score_for_k)
                is_m_neginf = m_alpha < (NEG_INF + 1.0)
                is_score_neginf = score_for_k < (NEG_INF + 1.0)
                m_new_safe = tl.where(is_m_neginf & is_score_neginf, 0.0, m_new)
                l_alpha = tl.where(
                    is_m_neginf,
                    tl.exp(score_for_k - m_new_safe),
                    l_alpha * tl.exp(m_alpha - m_new_safe) + tl.exp(score_for_k - m_new_safe),
                )
                m_alpha = m_new

            # Finalize online logsumexp: alpha_t = m + log(l)
            is_all_neginf = m_alpha < (NEG_INF + 1.0)
            # Guard: tl.log(l_alpha) is only safe when l_alpha > 0.
            # When all inputs were NEG_INF, l_alpha == 0 and log(0) = -inf/NaN.
            # The is_all_neginf guard short-circuits to NEG_INF before evaluating log.
            alpha_t = tl.where(is_all_neginf, NEG_INF, m_alpha + tl.log(l_alpha))
            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            # Checkpoint i stores the ring buffer state at position i * CHECKPOINT_INTERVAL
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
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

                # Guard against all-NEG_INF case (inactive sequence or numerical issue)
                # WORKAROUND: The comparison `max_val < (NEG_INF + 1.0)` fails silently in Triton
                # for variable-length sequences due to unclear type/comparison issues.
                # Instead, use `active` directly to determine if we should apply shift.
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

            # Capture final alpha at sequence end (t == seq_len)
            # At iteration t, alpha_t represents segments ending at position t-1
            # For sequence of length L, we need alpha at t=L (segments ending at L-1)
            # Cast t to int32 to match seq_len type for consistent comparison
            is_final = t.to(tl.int32) == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Final reduction: logsumexp over labels
        # Guard against all-NEG_INF case to prevent undefined arithmetic
        # Flash Attention pattern: no epsilon needed inside log
        # The NEG_INF guard handles the zero case
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        max_val = tl.max(final_alpha_masked, axis=0)
        is_final_neginf = max_val < (NEG_INF + 1.0)
        max_val_safe = tl.where(is_final_neginf, 0.0, max_val)
        exp_fa = tl.where(c_mask, tl.exp(final_alpha - max_val_safe), 0.0)
        sum_exp = tl.sum(exp_fa, axis=0)
        raw_partition = tl.where(is_final_neginf, NEG_INF, max_val + tl.log(sum_exp))

        # Add back cumulative normalization to get true partition function
        # final_alpha contains normalized values; accum_log_norm contains the total shift
        partition = (raw_partition.to(tl.float64) + accum_log_norm).to(tl.float64)

        # DEBUG: Print partition (log_Z) for T=100k diagnostic
        # Uncomment these lines to diagnose numerical stability at extreme scale
        # if batch_idx == 0:
        #     tl.device_print("FWD log_Z=", partition)
        #     tl.device_print("FWD max_final_alpha=", max_val)

        # Store result
        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_streaming_scan_kernel_max(
        # Same signature as log kernel
        cum_scores_ptr,
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,
        lengths_ptr,
        # Boundary projections (optional)
        proj_start_ptr,
        proj_end_ptr,
        out_ptr,
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer
        ring_ckpt_ptr,
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        HAS_BOUNDARIES: tl.constexpr,
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        stride_tr_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_tr_src,
        stride_tr_dst,
        stride_db_k,
        stride_db_c,
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
        # Log normalization checkpoints (not used for max semiring, but kept for API consistency)
        log_norm_ckpt_ptr,
        stride_lnc_b,
        stride_lnc_n,
        # Mixed precision support
        USE_FLOAT32: tl.constexpr = False,
    ):
        r"""Streaming Semi-CRF forward scan with max semiring (Viterbi decoding).

        Same structure as :func:`semi_crf_streaming_scan_kernel` but uses max
        instead of logsumexp for reductions. This implements the Viterbi algorithm
        for finding the most likely segmentation.

        See :func:`semi_crf_streaming_scan_kernel` for detailed documentation of
        parameters and the edge computation formula.
        """
        # Must match NEG_INF in constants.py
        NEG_INF: tl.constexpr = -1e9

        # Select compute dtype based on precision mode
        if USE_FLOAT32:
            COMPUTE_DTYPE: tl.constexpr = tl.float32
        else:
            COMPUTE_DTYPE: tl.constexpr = tl.float64

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        c_dst_idx = tl.arange(0, C_PAD)[:, None]
        c_src_idx = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        # Clamp indices so masked threads do not form out-of-bounds addresses.
        # See detailed note in semi_crf_streaming_scan_kernel above.
        c_idx_safe = tl.minimum(c_idx, C - 1)
        c_dst_idx_safe = tl.minimum(c_dst_idx, C - 1)
        c_src_idx_safe = tl.minimum(c_src_idx, C - 1)

        # Cast to int32 to match loop variable type from tl.range
        seq_len = tl.load(lengths_ptr + batch_idx).to(tl.int32)

        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        # Load static transitions once (duration-dependent loaded inside k-loop)
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx_safe * stride_tr_dst + c_src_idx_safe * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )

        # Ring buffer and checkpoint 0 are pre-initialized by the launcher:
        # - ring_buffer[:, 0, :C] = 0.0, rest = NEG_INF
        # - ring_checkpoints[:, 0, 0, :C] = 0.0, rest = NEG_INF
        # This avoids K iterations of conditional writes per batch element.

        final_alpha = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)

        for t in tl.range(1, T + 1):
            # Include t == seq_len to compute alpha at final position
            # Cast t to int32 to match seq_len type for consistent comparison
            active = t.to(tl.int32) <= seq_len
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)

            # Loop over valid segment durations k = 1, 2, ..., min(K, t)
            for k in tl.range(1, K + 1):
                # Duration k uses index k-1 in duration_bias
                k_valid = (k <= t) & (k <= K)
                start_pos = t - k
                ring_k_idx = start_pos % K

                # Load from live ring buffer
                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )

                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx_safe * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                content_score = cum_end - cum_start
                # Duration k uses index k-1
                dur_idx = k - 1
                dur_bias = tl.load(
                    duration_bias_ptr + dur_idx * stride_db_k + c_idx_safe * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )
                segment_score = content_score + dur_bias

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

                edge_block = segment_score[:, None] + transition_block

                scores = alpha_prev[None, :] + edge_block
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Max semiring: max over c_src
                score_for_k = tl.max(scores, axis=1)
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Max semiring: max over k
                alpha_t = tl.maximum(alpha_t, score_for_k)

            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            # Only save for active sequences to avoid stale data
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
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

        # Max semiring: max over labels
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        partition = tl.max(final_alpha_masked, axis=0)

        tl.store(out_ptr + batch_idx, partition)

    @triton.jit
    def semi_crf_streaming_scan_kernel_max_bp(
        # Same signature as max kernel, plus backpointer outputs
        cum_scores_ptr,
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,
        lengths_ptr,
        # Boundary projections (optional)
        proj_start_ptr,
        proj_end_ptr,
        out_ptr,
        ring_ptr,  # (batch, K, C_PAD) - live ring buffer
        ring_ckpt_ptr,
        # Backpointer outputs
        bp_k_ptr,  # (batch, T, C) - best duration for each (t, c_dest)
        bp_c_ptr,  # (batch, T, C) - best source label for each (t, c_dest)
        final_labels_ptr,  # (batch,) - best final label
        batch_size,
        T: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        HAS_BOUNDARIES: tl.constexpr,
        HAS_DURATION_TRANSITIONS: tl.constexpr,
        stride_cs_b,
        stride_cs_t,
        stride_cs_c,
        stride_tr_k,
        stride_tr_src,
        stride_tr_dst,
        stride_db_k,
        stride_db_c,
        stride_ps_b,
        stride_ps_t,
        stride_ps_c,
        stride_ring_b,
        stride_ring_k,
        stride_ring_c,
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
        stride_bp_b,
        stride_bp_t,
        stride_bp_c,
        # Mixed precision support
        USE_FLOAT32: tl.constexpr = False,
    ):
        r"""Streaming Semi-CRF forward scan with max semiring and backpointer tracking.

        Same as :func:`semi_crf_streaming_scan_kernel_max` but also outputs
        backpointers for O(T) traceback during Viterbi decoding.

        Additional outputs:
            bp_k: (batch, T, C) - best duration k for each (t, c_dest)
            bp_c: (batch, T, C) - best source label c_src for each (t, c_dest)
            final_labels: (batch,) - argmax of final alpha (best final label)
        """
        # Must match NEG_INF in constants.py
        NEG_INF: tl.constexpr = -1e9

        # Select compute dtype based on precision mode
        if USE_FLOAT32:
            COMPUTE_DTYPE: tl.constexpr = tl.float32
        else:
            COMPUTE_DTYPE: tl.constexpr = tl.float64

        batch_idx = tl.program_id(0)
        if batch_idx >= batch_size:
            return

        c_idx = tl.arange(0, C_PAD)
        c_mask = c_idx < C

        c_dst_idx = tl.arange(0, C_PAD)[:, None]
        c_src_idx = tl.arange(0, C_PAD)[None, :]
        c_mask_2d = (c_dst_idx < C) & (c_src_idx < C)

        # Clamp indices so masked threads do not form out-of-bounds addresses.
        # See detailed note in semi_crf_streaming_scan_kernel above.
        c_idx_safe = tl.minimum(c_idx, C - 1)
        c_dst_idx_safe = tl.minimum(c_dst_idx, C - 1)
        c_src_idx_safe = tl.minimum(c_src_idx, C - 1)

        # Cast to int32 to match loop variable type from tl.range
        seq_len = tl.load(lengths_ptr + batch_idx).to(tl.int32)

        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_base = ring_ptr + batch_idx * stride_ring_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b
        bp_base = batch_idx * stride_bp_b

        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b

        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx_safe * stride_tr_dst + c_src_idx_safe * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )

        final_alpha = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)

        for t in tl.range(1, T + 1):
            # Cast t to int32 to match seq_len type for consistent comparison
            active = t.to(tl.int32) <= seq_len
            alpha_t = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)

            # Initialize backpointer tracking for this timestep
            best_k_t = tl.zeros([C_PAD], dtype=tl.int32)
            best_c_src_t = tl.zeros([C_PAD], dtype=tl.int32)

            # Loop over valid segment durations k = 1, 2, ..., min(K, t)
            for k in tl.range(1, K + 1):
                # Duration k uses index k-1 in duration_bias
                k_valid = (k <= t) & (k <= K)
                start_pos = t - k
                ring_k_idx = start_pos % K

                alpha_prev = tl.load(
                    ring_base + ring_k_idx * stride_ring_k + c_idx * stride_ring_c,
                    mask=active & k_valid & c_mask,
                    other=NEG_INF,
                )

                cum_end = tl.load(
                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                cum_start = tl.load(
                    cum_scores_base + start_pos * stride_cs_t + c_idx_safe * stride_cs_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )

                content_score = cum_end - cum_start
                dur_idx = k - 1
                dur_bias = tl.load(
                    duration_bias_ptr + dur_idx * stride_db_k + c_idx_safe * stride_db_c,
                    mask=active & k_valid & c_mask,
                    other=0.0,
                )
                segment_score = content_score + dur_bias

                if HAS_BOUNDARIES:
                    start_score = tl.load(
                        proj_start_base + start_pos * stride_ps_t + c_idx_safe * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    end_pos_boundary = t - 1
                    end_score = tl.load(
                        proj_end_base + end_pos_boundary * stride_ps_t + c_idx_safe * stride_ps_c,
                        mask=active & k_valid & c_mask,
                        other=0.0,
                    )
                    segment_score = segment_score + start_score + end_score

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
                    )

                edge_block = segment_score[:, None] + transition_block

                scores = alpha_prev[None, :] + edge_block
                scores = tl.where(c_mask_2d, scores, NEG_INF)

                # Max over c_src WITH argmax tracking
                score_for_k = tl.max(scores, axis=1)
                argmax_c_src = tl.argmax(scores, axis=1)
                score_for_k = tl.where(k_valid & c_mask, score_for_k, NEG_INF)

                # Track which k wins for each c_dest
                better_mask = score_for_k > alpha_t
                best_k_t = tl.where(better_mask, k, best_k_t)
                best_c_src_t = tl.where(better_mask, argmax_c_src, best_c_src_t)

                # Max over k
                alpha_t = tl.maximum(alpha_t, score_for_k)

            alpha_t = tl.where(active & c_mask, alpha_t, NEG_INF)

            # Store backpointers at position t-1 (0-indexed)
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

            # Store to live ring buffer
            ring_t_idx = t % K
            tl.store(
                ring_base + ring_t_idx * stride_ring_k + c_idx * stride_ring_c,
                alpha_t,
                mask=active & c_mask,
            )

            # Save checkpoint at interval boundaries
            # Only save for active sequences to avoid stale data
            should_checkpoint = (t % CHECKPOINT_INTERVAL) == 0
            ckpt_idx = t // CHECKPOINT_INTERVAL
            if should_checkpoint:
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

            # Cast t to int32 to match seq_len type for consistent comparison
            is_final = t.to(tl.int32) == seq_len
            final_alpha = tl.where(is_final & c_mask, alpha_t, final_alpha)

        # Max over labels with argmax for final label
        final_alpha_masked = tl.where(c_mask, final_alpha, NEG_INF)
        partition = tl.max(final_alpha_masked, axis=0)
        final_label = tl.argmax(final_alpha_masked, axis=0)

        tl.store(out_ptr + batch_idx, partition)
        tl.store(final_labels_ptr + batch_idx, final_label)

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

        # Allocate outputs
        # Partition always float64 (scalar per batch, adds accum_log_norm which is O(T))
        partition = torch.empty(batch, device=device, dtype=torch.float64)

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

        # Log normalization checkpoint storage for numerical stability at extreme T
        # Stores cumulative log normalization factor at each checkpoint boundary
        # MUST stay float64: these scalars grow to O(T) magnitude
        log_norm_checkpoints = torch.zeros(
            (batch, num_checkpoints), device=device, dtype=torch.float64
        )

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
        stride_lnc_b, stride_lnc_n = log_norm_checkpoints.stride()

        # Launch kernel with device context for multi-GPU support
        grid = (batch,)
        kernel = (
            semi_crf_streaming_scan_kernel
            if semiring == "log"
            else semi_crf_streaming_scan_kernel_max
        )
        with torch.cuda.device(device):
            kernel[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                proj_start,
                proj_end,
                partition,
                ring_buffer,
                ring_checkpoints,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                has_boundaries,  # HAS_BOUNDARIES constexpr
                has_duration_transitions,  # HAS_DURATION_TRANSITIONS constexpr
                stride_cs_b,
                stride_cs_t,
                stride_cs_c,
                stride_tr_k,
                stride_tr_src,
                stride_tr_dst,
                stride_db_k,
                stride_db_c,
                stride_ps_b,
                stride_ps_t,
                stride_ps_c,
                stride_ring_b,
                stride_ring_k,
                stride_ring_c,
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
                log_norm_checkpoints,
                stride_lnc_b,
                stride_lnc_n,
                USE_FLOAT32=use_float32,
                num_warps=num_warps,
            )

        # Trim padding from checkpoints for return
        ring_checkpoints = ring_checkpoints[:, :, :, :C]

        return partition, ring_checkpoints, checkpoint_interval, log_norm_checkpoints

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
            proj_start = cum_scores[:, :T, :]
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0

        # Allocate outputs
        partition = torch.empty(batch, device=device, dtype=compute_dtype)
        bp_k = torch.zeros((batch, T, C), device=device, dtype=torch.int32)
        bp_c = torch.zeros((batch, T, C), device=device, dtype=torch.int32)
        final_labels = torch.zeros(batch, device=device, dtype=torch.int64)

        # Ring buffer
        ring_buffer = torch.full((batch, K, C_PAD), NEG_INF, device=device, dtype=compute_dtype)
        ring_buffer[:, 0, :C] = 0.0

        # Checkpoint storage
        ring_checkpoints = torch.full(
            (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=compute_dtype
        )
        ring_checkpoints[:, 0, 0, :C] = 0.0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ring_b, stride_ring_k, stride_ring_c = ring_buffer.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_checkpoints.stride()
        stride_bp_b, stride_bp_t, stride_bp_c = bp_k.stride()

        # Launch kernel
        grid = (batch,)
        with torch.cuda.device(device):
            semi_crf_streaming_scan_kernel_max_bp[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                proj_start,
                proj_end,
                partition,
                ring_buffer,
                ring_checkpoints,
                bp_k,
                bp_c,
                final_labels,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                has_boundaries,
                has_duration_transitions,
                stride_cs_b,
                stride_cs_t,
                stride_cs_c,
                stride_tr_k,
                stride_tr_src,
                stride_tr_dst,
                stride_db_k,
                stride_db_c,
                stride_ps_b,
                stride_ps_t,
                stride_ps_c,
                stride_ring_b,
                stride_ring_k,
                stride_ring_c,
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
                stride_bp_b,
                stride_bp_t,
                stride_bp_c,
                USE_FLOAT32=use_float32,
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
        r"""Triton-accelerated Viterbi forward pass with backpointer tracking.

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
