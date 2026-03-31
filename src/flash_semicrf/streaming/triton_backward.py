r"""Triton backward kernel for streaming Semi-CRF.

This module contains the Triton kernel for the backward pass and the launcher
function that allocates buffers and dispatches the kernel.

The backward pass uses the forward-backward algorithm with checkpointing:

1. Recompute alpha values from saved ring buffer checkpoints.
2. Compute beta backward while accumulating gradients.

Gradients are computed via marginal probabilities:

.. math::
    P(\text{segment}[t, k, c_{\text{dst}}, c_{\text{src}}]) =
    \frac{\exp(\alpha[t, c_{\text{src}}] + \text{edge} + \beta[t+k, c_{\text{dst}}])}
    {\exp(\log Z)}

.. warning::
    **Minimum K Requirement**: This kernel requires K >= 3 for correct operation.
    The ring buffer and checkpoint architecture assumes meaningful separation
    between timesteps. For K<3, use the specialized PyTorch implementations:

    - **K=1**: ``linear_crf_backward_pytorch`` in pytorch_reference.py
    - **K=2**: ``semi_crf_k2_backward_pytorch`` in pytorch_reference.py

    The dispatch logic in ``autograd.py`` automatically routes K<3 to these
    specialized implementations. Do not call this kernel directly with K<3.

.. note::
    The alpha recomputation path uses ``tl.debug_barrier()`` to ensure checkpoint
    values are visible across all warps before the backward pass. This prevents
    race conditions where segment 1 (processed first) overwrites values that
    segment 0 still needs. All tests pass with rtol=0.01.
"""

import torch

from ..validation import validate_streaming_shapes
from .constants import NEG_INF
from .triton_forward import _next_power_of_2

# Triton is optional
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    # NOTE: @triton.autotune removed - corrupts gradient buffers during benchmarking.
    # See docs/debugging/DEBUGGING_NAN.md "Autotuning Limitations" for details.
    @triton.jit
    def semi_crf_streaming_backward_kernel(
        # Inputs (from forward)
        cum_scores_ptr,  # (batch, T+1, C)
        transition_ptr,  # (C, C) or (K, C, C)
        duration_bias_ptr,  # (K, C)
        lengths_ptr,  # (batch,)
        log_Z_ptr,  # (batch,) - partition function values
        ring_ckpt_ptr,  # (batch, num_ckpts, K, C_PAD) - checkpoints from forward
        grad_output_ptr,  # (batch,) - upstream gradient
        # Boundary projections (optional, may be null if HAS_BOUNDARIES=False)
        proj_start_ptr,  # (batch, T, C) - start boundary scores
        proj_end_ptr,  # (batch, T, C) - end boundary scores
        # Working memory
        alpha_buffer_ptr,  # (batch, num_segments, SEGMENT_SIZE, C_PAD) - recomputed alpha (segment-isolated)
        beta_ring_ptr,  # (batch, K, C_PAD) - beta ring buffer
        # Outputs (gradients)
        grad_cum_scores_ptr,  # (batch, T+1, C)
        grad_tr_workspace_ptr,  # (batch, C, C) or (batch, K, C, C) - per-batch accumulator
        grad_db_workspace_ptr,  # (batch, K, C) - per-batch accumulator, no atomic needed
        grad_proj_start_ptr,  # (batch, T, C_PAD) - gradient for proj_start (only if HAS_BOUNDARIES)
        grad_proj_end_ptr,  # (batch, T, C_PAD) - gradient for proj_end (only if HAS_BOUNDARIES)
        boundary_marginals_ptr,  # (batch, T) - output for boundary marginals (if RETURN_BOUNDARY_MARGINALS)
        # Dimensions
        batch_size,
        T: tl.constexpr,  # max sequence length
        K: tl.constexpr,  # max segment duration
        C: tl.constexpr,  # actual num labels
        C_PAD: tl.constexpr,  # padded num labels (power of 2)
        CHECKPOINT_INTERVAL: tl.constexpr,
        NUM_CKPTS: tl.constexpr,
        SEGMENT_SIZE: tl.constexpr,  # = CHECKPOINT_INTERVAL + K
        HAS_BOUNDARIES: tl.constexpr,  # whether boundary projections are provided
        HAS_DURATION_TRANSITIONS: tl.constexpr,  # whether transitions are (K, C, C)
        RETURN_BOUNDARY_MARGINALS: tl.constexpr,  # whether to accumulate boundary marginals
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
        # Strides for grad_proj_start/grad_proj_end (batch, T, C_PAD) - only used if HAS_BOUNDARIES
        stride_gps_b,
        stride_gps_t,
        stride_gps_c,
        # Strides for ring checkpoints (batch, num_ckpts, K, C_PAD)
        stride_ckpt_b,
        stride_ckpt_n,
        stride_ckpt_k,
        stride_ckpt_c,
        # Strides for alpha buffer (batch, num_segments, SEGMENT_SIZE, C_PAD)
        stride_ab_b,
        stride_ab_seg,  # Segment stride for isolation between segments
        stride_ab_t,
        stride_ab_c,
        # Strides for beta ring (batch, K, C_PAD)
        stride_br_b,
        stride_br_k,
        stride_br_c,
        # Strides for grad_cum_scores (batch, T+1, C)
        stride_gcs_b,
        stride_gcs_t,
        stride_gcs_c,
        # Strides for grad_tr_workspace (batch, num_segments, C, C) or (batch, num_segments, K, C, C)
        stride_gtw_b,
        stride_gtw_seg,  # Segment stride for deterministic accumulation
        stride_gtw_k,  # Only used if HAS_DURATION_TRANSITIONS
        stride_gtw_src,
        stride_gtw_dst,
        # Strides for grad_db_workspace (batch, num_segments, K, C)
        stride_gdbw_b,
        stride_gdbw_seg,  # Segment stride for deterministic accumulation
        stride_gdbw_k,
        stride_gdbw_c,
        # Strides for boundary_marginals (batch, T) - only used if RETURN_BOUNDARY_MARGINALS
        stride_bm_b,
        stride_bm_t,
        # Log normalization checkpoints for numerical stability at extreme T
        log_norm_ckpt_ptr,  # (batch, num_ckpts) - cumulative log normalization factors
        stride_lnc_b,
        stride_lnc_n,
        # Autotuned parameter (must be at end for @triton.autotune)
        TILE_C: tl.constexpr,
        EVEN_C: tl.constexpr,  # True when C is already a power of 2 (C == C_PAD)
        # Mixed precision support
        USE_FLOAT32: tl.constexpr = False,  # Use float32 for bulk compute
    ):
        r"""Streaming Semi-CRF backward kernel with loop tiling and online logsumexp.

        Computes gradients for the Semi-CRF partition function using the
        forward-backward algorithm with memory-efficient streaming checkpoints.
        One program per batch element (``grid = (batch_size,)``).

        For each checkpoint segment (processed in reverse order):

        Phase 1 loads the ring buffer checkpoint for the segment start, then
        recomputes alpha values forward through the segment:

        .. math::
            \alpha[t, c] = \text{logsumexp}_k \left(
                \alpha[t-k, :] + \text{edge}[t-k \to t, :, c]
            \right)

        Phase 2 processes positions in reverse (t = seg_end-1 down to seg_start),
        computing marginal probabilities and accumulating gradients while updating
        beta:

        .. math::
            \beta[t, c] = \text{logsumexp}_k \left(
                \text{edge}[t \to t+k, c, :] + \beta[t+k, :]
            \right)

        .. note::

            **Numerical stability.** ``log_Z``, ``accum_log_norm``, and ``log_scale``
            remain float64 regardless of ``USE_FLOAT32`` because they involve
            O(T)-magnitude values. When ``USE_FLOAT32=True``, alpha/beta accumulators
            and gradient workspaces use float32; per-segment workspace isolation bounds
            the atomic add count, keeping accumulation error negligible. Log-marginals
            are clamped to [-700, 700] before ``exp()`` to prevent float64 overflow.
            Alpha, beta, and edge values are clamped to [-1e6, 1e6] before marginal
            computation.

        .. note::

            **Loop tiling for register pressure.** The marginal computation requires
            a :math:`(C_\text{PAD} \times C_\text{PAD})` matrix. At ``C_PAD=64`` this
            demands ~384 registers/thread, causing spilling. The kernel processes
            ``C_PAD`` in tiles of ``TILE_C`` (typically 16 or 32), loading a
            ``(TILE_C, C_PAD)`` tile per iteration with online logsumexp for the beta
            update. This reduces peak register demand to ~120/thread.

        .. warning::

            Multi-config autotuning is unsafe for this kernel. Triton's
            ``pre_hook``/``reset_to_zero`` runs only during benchmarking, not after
            config selection (Triton issue #7181). The final run then accumulates on
            uncleared buffers from the last benchmarked config, producing errors of
            ~10^23. Only one config is registered (num_warps=4, TILE_C=16, optimal
            for L40/L40S). Adding additional configs will corrupt results.

        Gradient scaling: ``grad_cum_scores`` is scaled by ``grad_output[batch_idx]``
        inside the kernel. ``grad_transition`` and ``grad_duration_bias`` are
        accumulated unscaled, then reduced by the launcher via
        ``einsum("bij, b -> ij", workspace, grad_output)``.
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

        # =====================================================================
        # PREAMBLE: indices, base pointers, static transition matrix
        # =====================================================================

        # Label indices
        c_idx = tl.arange(0, C_PAD)

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
            # Clamp indices to ensure address calculations stay within bounds
            # even for masked-out threads (indices C to C_PAD-1).
            # This prevents OOB pointer calculation which is undefined behavior.
            c_idx_safe = tl.minimum(c_idx, C - 1)
            c_dst_idx_safe = tl.minimum(c_dst_idx, C - 1)
            c_src_idx_safe = tl.minimum(c_src_idx, C - 1)

        # Load batch-specific values
        # Cast seq_len to int32 to match loop variable type from tl.range
        seq_len = tl.load(lengths_ptr + batch_idx).to(tl.int32)
        log_Z = tl.load(log_Z_ptr + batch_idx).to(tl.float64)
        grad_out = tl.load(grad_output_ptr + batch_idx).to(COMPUTE_DTYPE)

        # Base pointers
        cum_scores_base = cum_scores_ptr + batch_idx * stride_cs_b
        ring_ckpt_base = ring_ckpt_ptr + batch_idx * stride_ckpt_b
        alpha_buf_base = alpha_buffer_ptr + batch_idx * stride_ab_b
        beta_ring_base = beta_ring_ptr + batch_idx * stride_br_b
        grad_cs_base = grad_cum_scores_ptr + batch_idx * stride_gcs_b
        grad_tr_ws_base = grad_tr_workspace_ptr + batch_idx * stride_gtw_b
        grad_db_ws_base = grad_db_workspace_ptr + batch_idx * stride_gdbw_b

        # Boundary projection base pointers (only used if HAS_BOUNDARIES)
        if HAS_BOUNDARIES:
            proj_start_base = proj_start_ptr + batch_idx * stride_ps_b
            proj_end_base = proj_end_ptr + batch_idx * stride_ps_b
            grad_ps_base = grad_proj_start_ptr + batch_idx * stride_gps_b
            grad_pe_base = grad_proj_end_ptr + batch_idx * stride_gps_b

        # Load transition matrix into registers (only for static transitions).
        # Duration-dependent transitions are loaded inside the k-loops.
        # Use clamped indices to avoid OOB pointer calculation for masked-out threads.
        if not HAS_DURATION_TRANSITIONS:
            transition_block = tl.load(
                transition_ptr + c_dst_idx_safe * stride_tr_dst + c_src_idx_safe * stride_tr_src,
                mask=c_mask_2d,
                other=0.0,
            )  # (C_PAD, C_PAD) - this is transition.T

        # =====================================================================
        # BETA RING BUFFER INITIALIZATION
        # The beta ring uses 2*K slots (double-buffered) to avoid aliasing.
        # Only the slot corresponding to t=seq_len is set to 0.0 (boundary
        # condition: beta[seq_len, :] = 0 in log-space). All other slots are
        # NEG_INF so that segments extending past the sequence end contribute
        # nothing to the backward pass.
        # =====================================================================
        final_pos = seq_len
        final_ring_idx = final_pos % (2 * K)
        # Use tl.range (not static_range) to avoid compile-time explosion for large K.
        for k_init in tl.range(0, 2 * K):
            is_final = k_init == final_ring_idx
            init_val = tl.where(is_final & c_mask, 0.0, NEG_INF)
            tl.store(
                beta_ring_base + k_init * stride_br_k + c_idx * stride_br_c,
                init_val,
                mask=c_mask,
            )

        # =====================================================================
        # OUTER SEGMENT LOOP: process checkpoint segments in reverse order
        # =====================================================================
        # Use tl.range to avoid compile-time unrolling for large NUM_CKPTS.
        for ckpt_idx_loop in tl.range(0, NUM_CKPTS):
            ckpt_idx = NUM_CKPTS - 1 - ckpt_idx_loop
            seg_start = ckpt_idx * CHECKPOINT_INTERVAL
            seg_end = (ckpt_idx + 1) * CHECKPOINT_INTERVAL
            if seg_end > T:
                seg_end = T

            # Load cumulative log normalization factor for this checkpoint
            # IMPORTANT: Use ckpt_idx (logical index), NOT ckpt_idx_loop (loop counter)
            # This is needed to restore the scale when computing marginals
            log_norm_at_ckpt = tl.load(
                log_norm_ckpt_ptr + batch_idx * stride_lnc_b + ckpt_idx * stride_lnc_n
            ).to(tl.float64)

            # Segment-specific workspace slices for deterministic gradient accumulation.
            # Each segment writes to its own slice, eliminating cross-segment atomic contention.
            grad_tr_ws_seg = grad_tr_ws_base + ckpt_idx * stride_gtw_seg
            grad_db_ws_seg = grad_db_ws_base + ckpt_idx * stride_gdbw_seg
            # Alpha buffer also needs segment isolation to prevent race conditions.
            # Without it, segment 1 (processed first in backward) writes alpha[27]
            # to position 0, then segment 0 reads stale values instead of the checkpoint.
            alpha_buf_seg = alpha_buf_base + ckpt_idx * stride_ab_seg

            # log_norm_at_ckpt is used in Phase 2 Pass 1 to bridge the normalized
            # alpha values (shifted at checkpoint boundaries) and the full-scale log_Z.

            # Only process segments within sequence length
            if seg_start < seq_len:
                # =============================================================
                # PHASE 1: Recompute alpha for this segment
                #
                # The forward pass saves ring buffer checkpoints but not full
                # alpha arrays (that would be O(T*K*C) memory). Here we restore
                # alpha[seg_start] from the checkpoint, then replay the forward
                # recurrence through the segment to rebuild alpha[seg_start+1 ...
                # seg_end]. These values are needed in Phase 2 to compute
                # marginals: P(segment) = exp(alpha[t] + edge + beta[t+k] - log_Z).
                # =============================================================

                # --- Phase 1, step 1: Restore alpha[seg_start] from checkpoint ---
                # The checkpoint stores the full ring buffer state at seg_start.
                # Ring slot (seg_start % K) holds the alpha we need for position
                # seg_start; the other K-1 slots are prior positions used as
                # history during the forward recompute below.
                for k_slot in tl.range(0, K):
                    alpha_val = tl.load(
                        ring_ckpt_base
                        + ckpt_idx * stride_ckpt_n
                        + k_slot * stride_ckpt_k
                        + c_idx * stride_ckpt_c,
                        mask=c_mask,
                        other=NEG_INF,
                    )
                    # Buffer position 0 maps to absolute time seg_start.
                    if k_slot == seg_start % K:
                        tl.store(
                            alpha_buf_seg + 0 * stride_ab_t + c_idx * stride_ab_c,
                            alpha_val,
                            mask=c_mask,
                        )

                # Memory barrier: ensure alpha buffer writes are visible to all
                # warps before the forward recompute reads from them.
                # Without this, segment 1 (processed first in reverse) may write
                # values that segment 0 sees as stale.
                # This was responsible for 10-400% gradient errors in batch=1 tests.
                tl.debug_barrier()

                # --- Phase 1, step 2: Forward recompute alpha[seg_start+1 ... seg_end] ---
                # Replay the forward recurrence within the segment.
                # For each t, alpha_prev comes from one of two places:
                #   local_start >= 0: alpha buffer (within this segment)
                #   local_start < 0:  ring buffer checkpoint (prior segment)
                # Use tl.range to avoid compile-time unrolling for large SEGMENT_SIZE.
                for local_t in tl.range(1, SEGMENT_SIZE):
                    t = seg_start + local_t
                    # Only process if within segment and sequence bounds
                    if t < seg_end and t < seq_len:
                        # Online logsumexp accumulators for alpha_t over k durations
                        m_alpha = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)
                        l_alpha = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)

                        # Loop over valid segment durations k = 1, 2, ..., K
                        for k in tl.range(1, K + 1):
                            start_pos = t - k
                            # Only process valid start positions
                            if start_pos >= 0:
                                # --- Alpha lookup: segment buffer or prior checkpoint ---
                                local_start = start_pos - seg_start

                                if local_start >= 0 and local_start < SEGMENT_SIZE:
                                    alpha_prev = tl.load(
                                        alpha_buf_seg
                                        + local_start * stride_ab_t
                                        + c_idx * stride_ab_c,
                                        mask=c_mask,
                                        other=NEG_INF,
                                    )
                                else:
                                    # Position is before seg_start, get from current checkpoint
                                    # The checkpoint at ckpt_idx contains alpha[seg_start-K+1..seg_start]
                                    # at ring indices (seg_start-K+1) % K .. seg_start % K
                                    prev_ring_idx = start_pos % K
                                    alpha_prev = tl.load(
                                        ring_ckpt_base
                                        + ckpt_idx * stride_ckpt_n
                                        + prev_ring_idx * stride_ckpt_k
                                        + c_idx * stride_ckpt_c,
                                        mask=c_mask,
                                        other=NEG_INF,
                                    )

                                # Compute edge on-the-fly
                                # Use clamped indices to avoid OOB pointer calculation
                                cum_end = tl.load(
                                    cum_scores_base + t * stride_cs_t + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                cum_start = tl.load(
                                    cum_scores_base
                                    + start_pos * stride_cs_t
                                    + c_idx_safe * stride_cs_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                content_score = cum_end - cum_start

                                # Duration k uses index k-1
                                dur_idx = k - 1
                                dur_bias = tl.load(
                                    duration_bias_ptr
                                    + dur_idx * stride_db_k
                                    + c_idx_safe * stride_db_c,
                                    mask=c_mask,
                                    other=0.0,
                                )
                                segment_score = content_score + dur_bias

                                # Add boundary scores if provided
                                # Segment starts at start_pos, ends at t-1 (inclusive)
                                if HAS_BOUNDARIES:
                                    start_score = tl.load(
                                        proj_start_base
                                        + start_pos * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    end_pos_boundary = t - 1
                                    end_score = tl.load(
                                        proj_end_base
                                        + end_pos_boundary * stride_ps_t
                                        + c_idx_safe * stride_ps_c,
                                        mask=c_mask,
                                        other=0.0,
                                    )
                                    segment_score = segment_score + start_score + end_score

                                # Load dur_idx-indexed transition for duration-dependent case
                                # Duration k uses index k-1 (same convention as PyTorch reference)
                                # Use clamped indices to avoid OOB pointer calculation
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

                                # Logsumexp over c_src
                                # Guard against all-NEG_INF case to prevent undefined arithmetic
                                # Flash Attention pattern: no epsilon needed inside log
                                # The NEG_INF guard handles the zero case
                                max_scores = tl.max(scores, axis=1)
                                is_all_neginf = max_scores < (NEG_INF + 1.0)
                                max_scores_safe = tl.where(is_all_neginf, 0.0, max_scores)
                                log_sum_exp = tl.log(
                                    tl.sum(tl.exp(scores - max_scores_safe[:, None]), axis=1)
                                )
                                score_for_k = tl.where(
                                    is_all_neginf, NEG_INF, max_scores + log_sum_exp
                                )
                                score_for_k = tl.where(c_mask, score_for_k, NEG_INF)

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
                                    l_alpha * tl.exp(m_alpha - m_new_safe)
                                    + tl.exp(score_for_k - m_new_safe),
                                )
                                m_alpha = m_new

                        # Finalize online logsumexp: alpha_t = m + log(l)
                        is_all_neginf = m_alpha < (NEG_INF + 1.0)
                        # Guard: tl.log(l_alpha) is only safe when l_alpha > 0.
                        # When all inputs were NEG_INF, l_alpha == 0 and log(0) = -inf/NaN.
                        # The is_all_neginf guard short-circuits to NEG_INF before evaluating log.
                        alpha_t = tl.where(is_all_neginf, NEG_INF, m_alpha + tl.log(l_alpha))
                        alpha_t = tl.where(c_mask, alpha_t, NEG_INF)
                        tl.store(
                            alpha_buf_seg + local_t * stride_ab_t + c_idx * stride_ab_c,
                            alpha_t,
                            mask=c_mask,
                        )
                        # Memory barrier: same race as the forward ring buffer (see
                        # triton_forward.py comment). The next local_t iteration loads
                        # alpha_buf_seg[local_t] via alpha_prev[None, :] + edge_block,
                        # which is a C_PAD-element broadcast. Without the barrier, a warp
                        # entering local_t+1 can read stale values from another warp's
                        # store at local_t when C_PAD = warp_size (32).
                        tl.debug_barrier()

                # Memory barrier: ensure all Phase 1 alpha_buf_seg writes are visible
                # before Phase 2 reads them. Phase 2 loads alpha_t from alpha_buf_seg
                # and uses it in alpha_t_clamped[None, :] + edge_tile (C_PAD broadcast),
                # creating the same cross-warp race as the forward ring buffer.
                tl.debug_barrier()

                # =============================================================
                # PHASE 2: Beta backward pass and gradient accumulation
                #
                # Process positions t = seg_end-1 down to seg_start in reverse.
                # For each t, iterate over all valid k durations to:
                #   1. Compute marginals P(segment[t, t+k]) using the two-pass
                #      tiled algorithm (Pass 1: global max; Pass 2: marginals).
                #   2. Accumulate gradients for cum_scores, transition,
                #      duration_bias, and boundary projections.
                #   3. Update beta[t] via online logsumexp over k and c_dst.
                # Use tl.range to avoid compile-time unrolling for large CHECKPOINT_INTERVAL.
                # =============================================================
                for t_offset in tl.range(0, CHECKPOINT_INTERVAL):
                    t = seg_end - 1 - t_offset
                    # Only process valid positions
                    if t >= seg_start and t < seq_len and t >= 0:
                        # Get alpha[t] from buffer
                        local_t = t - seg_start
                        alpha_t = tl.load(
                            alpha_buf_seg + local_t * stride_ab_t + c_idx * stride_ab_c,
                            mask=c_mask,
                            other=NEG_INF,
                        )

                        # Compute beta[t] and gradients
                        # Online logsumexp accumulators for new_beta over k durations
                        m_new_beta = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)
                        l_new_beta = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)

                        # --- Tiled backward: process c_dst in tiles of TILE_C ---
                        # Avoids materializing the full (C_PAD, C_PAD) marginal matrix.
                        # Gradient accumulation and beta update run tile by tile.
                        # See kernel docstring for why TILE_C is needed for register pressure.

                        # === Gradient accumulators (atomic optimization) ===
                        # Accumulate grad_cum_scores[t] across all k and tiles, write once
                        grad_cs_t_local = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)

                        # Loop over valid segment durations k = 1, 2, ..., K
                        for k in tl.range(1, K + 1):
                            end_pos = t + k
                            # Only process valid end positions
                            if end_pos <= seq_len and end_pos <= T:
                                # Beta ring uses 2*K slots (double-buffered). The forward ring uses K
                                # slots and wraps at t % K. If we used the same K slots here, beta[t]
                                # and beta[t+K] would alias (both map to slot t % K = (t+K) % K = 0),
                                # causing beta[t+K] to overwrite the slot before beta[t] is read.
                                end_ring_idx = end_pos % (2 * K)
                                # Duration k uses index k-1
                                dur_idx = k - 1

                                # === Accumulators for this k iteration ===
                                # Boundary marginals accumulator (scalar)
                                # Must match COMPUTE_DTYPE for tl.range loop-carried variable type consistency
                                marginal_sum_all_k = tl.zeros((), dtype=COMPUTE_DTYPE)

                                # Online logsumexp accumulators for beta_k (indexed by c_src)
                                # Accumulates logsumexp over c_dst tiles
                                m_beta_k = tl.full([C_PAD], NEG_INF, dtype=COMPUTE_DTYPE)
                                l_beta_k = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)

                                # grad_duration_bias accumulator for this k (accumulate across tiles)
                                grad_db_k_local = tl.zeros([C_PAD], dtype=COMPUTE_DTYPE)

                                # Clamp alpha_t once per k (reused across tiles)
                                alpha_t_clamped = tl.minimum(tl.maximum(alpha_t, -1e6), 1e6)

                                # --- Pass 1: compute global max across all c_dst tiles ---
                                # global_max is a scalar max over the full (C_dst, C_src) log_joint
                                # matrix. The scale in the marginal computation is then:
                                #   scale = exp(global_max + log_norm_at_ckpt - log_Z)
                                # which does not require the global sum — log_Z already serves
                                # as the partition function. Only global_max is needed here.
                                #
                                # Must use tl.static_range: tl.range does not preserve scalar
                                # loop-carried state (global_max) across iterations, so each
                                # tile would use a different normalization value.
                                # static_range unrolls at compile time: O(C_PAD/TILE_C) iters,
                                # typically 2-8, which is acceptable.
                                global_max = tl.full((), NEG_INF, dtype=tl.float64)

                                for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
                                    # Tile indices
                                    c_dst_tile = tl.arange(0, TILE_C)
                                    c_dst_idx_tile = c_dst_tile_start + c_dst_tile
                                    c_dst_mask_tile = c_dst_idx_tile < C
                                    c_dst_idx_tile_safe = tl.minimum(c_dst_idx_tile, C - 1)
                                    tile_mask_2d = c_dst_mask_tile[:, None] & c_mask[None, :]

                                    # Load beta_next tile (TILE_C,)
                                    beta_tile = tl.load(
                                        beta_ring_base
                                        + end_ring_idx * stride_br_k
                                        + c_dst_idx_tile_safe * stride_br_c,
                                        mask=c_dst_mask_tile,
                                        other=NEG_INF,
                                    )

                                    # Load edge components to compute edge_tile
                                    cum_end_tile = tl.load(
                                        cum_scores_base
                                        + end_pos * stride_cs_t
                                        + c_dst_idx_tile_safe * stride_cs_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    cum_start_tile = tl.load(
                                        cum_scores_base
                                        + t * stride_cs_t
                                        + c_dst_idx_tile_safe * stride_cs_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    content_score_tile = cum_end_tile - cum_start_tile

                                    dur_bias_tile = tl.load(
                                        duration_bias_ptr
                                        + dur_idx * stride_db_k
                                        + c_dst_idx_tile_safe * stride_db_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    segment_score_tile = content_score_tile + dur_bias_tile

                                    # Add boundary scores if provided
                                    if HAS_BOUNDARIES:
                                        start_score_tile = tl.load(
                                            proj_start_base
                                            + t * stride_ps_t
                                            + c_dst_idx_tile_safe * stride_ps_c,
                                            mask=c_dst_mask_tile,
                                            other=0.0,
                                        )
                                        end_pos_boundary = end_pos - 1
                                        end_score_tile = tl.load(
                                            proj_end_base
                                            + end_pos_boundary * stride_ps_t
                                            + c_dst_idx_tile_safe * stride_ps_c,
                                            mask=c_dst_mask_tile,
                                            other=0.0,
                                        )
                                        segment_score_tile = (
                                            segment_score_tile + start_score_tile + end_score_tile
                                        )

                                    # Load transition tile (TILE_C, C_PAD)
                                    if HAS_DURATION_TRANSITIONS:
                                        transition_tile = tl.load(
                                            transition_ptr
                                            + dur_idx * stride_tr_k
                                            + c_dst_idx_tile_safe[:, None] * stride_tr_dst
                                            + c_idx_safe[None, :] * stride_tr_src,
                                            mask=tile_mask_2d,
                                            other=0.0,
                                        )
                                    else:
                                        transition_tile = tl.load(
                                            transition_ptr
                                            + c_dst_idx_tile_safe[:, None] * stride_tr_dst
                                            + c_idx_safe[None, :] * stride_tr_src,
                                            mask=tile_mask_2d,
                                            other=0.0,
                                        )

                                    # edge_tile: (TILE_C, C_PAD)
                                    edge_tile = segment_score_tile[:, None] + transition_tile

                                    # Clamp inputs for numerical stability
                                    beta_tile_clamped = tl.minimum(tl.maximum(beta_tile, -1e6), 1e6)
                                    edge_tile_clamped = tl.minimum(tl.maximum(edge_tile, -1e6), 1e6)

                                    # Compute log_joint for this tile
                                    log_joint_tile = (
                                        alpha_t_clamped[None, :]  # (1, C_PAD) for c_src
                                        + edge_tile_clamped  # (TILE_C, C_PAD)
                                        + beta_tile_clamped[:, None]  # (TILE_C, 1) for c_dst
                                    )

                                    # Mask invalid entries
                                    log_joint_masked = tl.where(
                                        tile_mask_2d, log_joint_tile, NEG_INF
                                    )

                                    # Update global max (scalar, reduce over entire tile)
                                    tile_max = tl.max(log_joint_masked)
                                    global_max = tl.maximum(global_max, tile_max)

                                # --- Scale computation (log_norm bridge) ---
                                # The forward pass shifts alpha at each checkpoint boundary
                                # (alpha[t] -= log_norm[t]) to keep values near 0 at extreme T.
                                # log_norm_at_ckpt holds the cumulative shift up to seg_start.
                                # To compute correct marginals exp(alpha + edge + beta - log_Z),
                                # we add log_norm_at_ckpt back before comparing to log_Z.
                                is_global_max_neginf = global_max < (NEG_INF + 1.0)
                                global_max_safe = tl.where(is_global_max_neginf, 0.0, global_max)

                                log_scale = (
                                    global_max_safe.to(tl.float64) + log_norm_at_ckpt - log_Z
                                )

                                # Clamp scale to prevent overflow/underflow
                                log_scale_clamped = tl.minimum(
                                    log_scale, 0.0
                                )  # Upper bound: scale ≤ 1
                                log_scale_clamped = tl.maximum(
                                    log_scale_clamped, -700.0
                                )  # Prevent underflow
                                scale = tl.exp(log_scale_clamped)  # Scalar

                                # Guard: If global_max was all NEG_INF, set scale to 0
                                scale = tl.where(is_global_max_neginf, 0.0, scale)

                                # Cast to COMPUTE_DTYPE for Pass 2 marginal/gradient computation.
                                # log_scale was computed in float64 (critical precision), but the
                                # resulting scale and global_max_safe are O(1) values safe for float32.
                                scale = scale.to(COMPUTE_DTYPE)
                                global_max_safe = global_max_safe.to(COMPUTE_DTYPE)

                                # --- Pass 2: marginals and gradient accumulation ---
                                # global_max_safe and scale are scalars fixed from Pass 1,
                                # ensuring all tiles use the same normalization.
                                for c_dst_tile_start in tl.static_range(0, C_PAD, TILE_C):
                                    # Tile indices
                                    c_dst_tile = tl.arange(0, TILE_C)
                                    c_dst_idx_tile = c_dst_tile_start + c_dst_tile
                                    c_dst_mask_tile = c_dst_idx_tile < C
                                    c_dst_idx_tile_safe = tl.minimum(c_dst_idx_tile, C - 1)
                                    tile_mask_2d = c_dst_mask_tile[:, None] & c_mask[None, :]

                                    # Load beta_next tile (TILE_C,)
                                    beta_tile = tl.load(
                                        beta_ring_base
                                        + end_ring_idx * stride_br_k
                                        + c_dst_idx_tile_safe * stride_br_c,
                                        mask=c_dst_mask_tile,
                                        other=NEG_INF,
                                    )

                                    # Compute segment_score tile (TILE_C,)
                                    cum_end_tile = tl.load(
                                        cum_scores_base
                                        + end_pos * stride_cs_t
                                        + c_dst_idx_tile_safe * stride_cs_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    cum_start_tile = tl.load(
                                        cum_scores_base
                                        + t * stride_cs_t
                                        + c_dst_idx_tile_safe * stride_cs_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    content_score_tile = cum_end_tile - cum_start_tile

                                    dur_bias_tile = tl.load(
                                        duration_bias_ptr
                                        + dur_idx * stride_db_k
                                        + c_dst_idx_tile_safe * stride_db_c,
                                        mask=c_dst_mask_tile,
                                        other=0.0,
                                    )
                                    segment_score_tile = content_score_tile + dur_bias_tile

                                    # Add boundary scores if provided
                                    if HAS_BOUNDARIES:
                                        start_score_tile = tl.load(
                                            proj_start_base
                                            + t * stride_ps_t
                                            + c_dst_idx_tile_safe * stride_ps_c,
                                            mask=c_dst_mask_tile,
                                            other=0.0,
                                        )
                                        end_pos_boundary = end_pos - 1
                                        end_score_tile = tl.load(
                                            proj_end_base
                                            + end_pos_boundary * stride_ps_t
                                            + c_dst_idx_tile_safe * stride_ps_c,
                                            mask=c_dst_mask_tile,
                                            other=0.0,
                                        )
                                        segment_score_tile = (
                                            segment_score_tile + start_score_tile + end_score_tile
                                        )

                                    # Load transition tile (TILE_C, C_PAD)
                                    # Rows = c_dst tile, Columns = all c_src
                                    # Duration k uses index k-1 (same convention as PyTorch reference)
                                    if HAS_DURATION_TRANSITIONS:
                                        transition_tile = tl.load(
                                            transition_ptr
                                            + dur_idx * stride_tr_k
                                            + c_dst_idx_tile_safe[:, None] * stride_tr_dst
                                            + c_idx_safe[None, :] * stride_tr_src,
                                            mask=tile_mask_2d,
                                            other=0.0,
                                        )
                                    else:
                                        transition_tile = tl.load(
                                            transition_ptr
                                            + c_dst_idx_tile_safe[:, None] * stride_tr_dst
                                            + c_idx_safe[None, :] * stride_tr_src,
                                            mask=tile_mask_2d,
                                            other=0.0,
                                        )

                                    # edge_tile: (TILE_C, C_PAD)
                                    edge_tile = segment_score_tile[:, None] + transition_tile

                                    # === Compute marginal tile (TILE_C, C_PAD) using GLOBAL statistics ===
                                    # Two-pass algorithm ensures all tiles use the same normalization:
                                    #   Pass 1 (lines 571-701): Compute global_max and scale across all tiles
                                    #   Pass 2 (lines 720-1037): Use global values for consistent marginal computation
                                    # This fixed a per-tile normalization bug that caused 10-400% gradient errors
                                    # when multiple tiles were active (e.g., C=32 with TILE_C=16 -> 2 tiles).
                                    beta_tile_clamped = tl.minimum(tl.maximum(beta_tile, -1e6), 1e6)
                                    edge_tile_clamped = tl.minimum(tl.maximum(edge_tile, -1e6), 1e6)

                                    # Step 1: Compute log_joint (without log_Z subtraction)
                                    log_joint_tile = (
                                        alpha_t_clamped[None, :]  # (1, C_PAD) for c_src
                                        + edge_tile_clamped  # (TILE_C, C_PAD)
                                        + beta_tile_clamped[:, None]  # (TILE_C, 1) for c_dst
                                    )

                                    # Step 2: Mask invalid entries
                                    log_joint_masked = tl.where(
                                        tile_mask_2d, log_joint_tile, NEG_INF
                                    )

                                    # Step 3: Compute relative log-marginal using GLOBAL max
                                    # This is the KEY FIX: global_max_safe is SCALAR, same for all tiles
                                    # Scalar broadcasts automatically to (TILE_C, C_PAD)
                                    log_marginal_rel = log_joint_masked - global_max_safe

                                    # Step 4: Compute unnormalized marginal (bounded in (0, 1])
                                    marginal_unnorm = tl.exp(log_marginal_rel)

                                    # Step 5: Final marginal using GLOBAL scale
                                    # scale is SCALAR, computed once in Pass 1: exp(global_max + log_norm - log_Z)
                                    # This ensures ALL tiles use the same scale factor
                                    marginal_tile = marginal_unnorm * scale
                                    marginal_tile = tl.where(tile_mask_2d, marginal_tile, 0.0).to(
                                        COMPUTE_DTYPE
                                    )

                                    # === Accumulate gradients from this tile ===

                                    # Boundary marginals: sum over both dims
                                    if RETURN_BOUNDARY_MARGINALS:
                                        marginal_sum_all_k += tl.sum(marginal_tile)

                                    # grad_cum_scores: sum over c_src -> (TILE_C,)
                                    marginal_sum_src_tile = tl.sum(marginal_tile, axis=1)
                                    marginal_sum_src_tile = tl.where(
                                        c_dst_mask_tile, marginal_sum_src_tile, 0.0
                                    ).to(COMPUTE_DTYPE)
                                    marginal_sum_src_tile_scaled = marginal_sum_src_tile * grad_out

                                    # grad_cum_scores[end_pos]: +marginal (end_pos varies by k, must use atomic)
                                    tl.atomic_add(
                                        grad_cs_base
                                        + end_pos * stride_gcs_t
                                        + c_dst_idx_tile * stride_gcs_c,
                                        marginal_sum_src_tile_scaled,
                                        mask=c_dst_mask_tile,
                                    )
                                    # grad_cum_scores[t]: -marginal (t is fixed for all k and tiles,
                                    # so we accumulate into grad_cs_t_local and do one atomic at the end).
                                    #
                                    # Problem: marginal_sum_src_tile_scaled is indexed by c_dst_tile
                                    # (TILE_C positions), but grad_cs_t_local is indexed by c_idx (C_PAD).
                                    # Triton has no scatter-assign, so we simulate it with a mask:
                                    #   scatter_mask[i, j] = True iff c_idx[i] == c_dst_idx_tile[j]
                                    # Broadcasting the TILE_C values into matching rows, then summing
                                    # over the tile axis, effectively does:
                                    #   grad_cs_t_local[c_dst_idx_tile[j]] -= values[j] for each j.
                                    # The final atomic write is deferred to after the k-loop, reducing
                                    # K*tiles atomics to 1 (e.g., 8000 -> 1 at K=1000, C=32, TILE_C=4).
                                    scatter_mask = (
                                        c_idx[:, None] == c_dst_idx_tile[None, :]
                                    )  # [C_PAD, TILE_C]
                                    scatter_values = tl.where(
                                        scatter_mask,
                                        marginal_sum_src_tile_scaled[
                                            None, :
                                        ],  # Broadcast to [C_PAD, TILE_C]
                                        0.0,
                                    )
                                    grad_cs_t_local -= tl.sum(scatter_values, axis=1).to(
                                        COMPUTE_DTYPE
                                    )  # Sum to [C_PAD]; cast needed: 0.0 literal in tl.where is f64

                                    # grad_transition: marginal_T_tile = (C_PAD, TILE_C)
                                    # Duration k uses index k-1 (same convention as forward pass)
                                    marginal_T_tile = tl.trans(marginal_tile)
                                    if HAS_DURATION_TRANSITIONS:
                                        tr_offsets_tile = (
                                            dur_idx * stride_gtw_k
                                            + c_idx[:, None] * stride_gtw_src
                                            + c_dst_idx_tile[None, :] * stride_gtw_dst
                                        )
                                    else:
                                        tr_offsets_tile = (
                                            c_idx[:, None] * stride_gtw_src
                                            + c_dst_idx_tile[None, :] * stride_gtw_dst
                                        )
                                    tile_mask_T = c_mask[:, None] & c_dst_mask_tile[None, :]
                                    # Use segment-specific workspace for deterministic accumulation
                                    tl.atomic_add(
                                        grad_tr_ws_seg + tr_offsets_tile,
                                        marginal_T_tile,
                                        mask=tile_mask_T,
                                    )

                                    # grad_duration_bias: unscaled (launcher applies grad_output via einsum).
                                    # Same scatter-sum pattern as grad_cum_scores[t] above.
                                    # Defers atomic to after the tile loop (1 write per k instead of per tile).
                                    scatter_mask_db = (
                                        c_idx[:, None] == c_dst_idx_tile[None, :]
                                    )  # [C_PAD, TILE_C]
                                    scatter_values_db = tl.where(
                                        scatter_mask_db,
                                        marginal_sum_src_tile[
                                            None, :
                                        ],  # Broadcast to [C_PAD, TILE_C]
                                        0.0,
                                    )
                                    grad_db_k_local += tl.sum(scatter_values_db, axis=1).to(
                                        COMPUTE_DTYPE
                                    )  # Sum to [C_PAD]; cast needed: 0.0 literal in tl.where is f64

                                    # grad_proj_start[t] and grad_proj_end[end_pos-1]
                                    if HAS_BOUNDARIES:
                                        # proj_start[t]: same position for all k, use atomic
                                        tl.atomic_add(
                                            grad_ps_base
                                            + t * stride_gps_t
                                            + c_dst_idx_tile * stride_gps_c,
                                            marginal_sum_src_tile_scaled,
                                            mask=c_dst_mask_tile,
                                        )
                                        # proj_end[end_pos-1]: varies by k, use atomic
                                        tl.atomic_add(
                                            grad_pe_base
                                            + (end_pos - 1) * stride_gps_t
                                            + c_dst_idx_tile * stride_gps_c,
                                            marginal_sum_src_tile_scaled,
                                            mask=c_dst_mask_tile,
                                        )

                                    # === Online logsumexp for beta_k (Flash Attention pattern) ===
                                    # Instead of materializing full (C_PAD, C_PAD) and reducing,
                                    # we accumulate logsumexp across tiles using online algorithm:
                                    #   m = running max, l = running sum of exp(x - m)
                                    #   For each new tile: m' = max(m, tile_max)
                                    #                      l' = l * exp(m - m') + tile_sum * exp(tile_max - m')
                                    # Final result: m + log(l)
                                    scores_for_beta_tile = edge_tile + beta_tile[:, None]
                                    scores_for_beta_tile = tl.where(
                                        tile_mask_2d, scores_for_beta_tile, NEG_INF
                                    )

                                    # Tile statistics: max and sum(exp) over c_dst dimension
                                    max_tile = tl.max(scores_for_beta_tile, axis=0)
                                    # Cast to COMPUTE_DTYPE to match accumulator m_beta_k.
                                    # tl.max returns the input dtype, but if scores_for_beta_tile
                                    # is float32, mixing it with the float64 m_beta_k accumulator
                                    # causes silent precision loss in the online update below,
                                    # producing errors of ~10^9 with 2+ tiles (observed empirically).
                                    max_tile = max_tile.to(COMPUTE_DTYPE)
                                    # NEG_INF guard: if all inputs are NEG_INF, max would be NEG_INF
                                    # and scores - max = 0, giving exp(0) = 1 (wrong!). Detect and handle.
                                    is_tile_neginf = max_tile < (NEG_INF + 1.0)
                                    max_tile_safe = tl.where(is_tile_neginf, 0.0, max_tile)

                                    sum_exp_tile = tl.sum(
                                        tl.exp(scores_for_beta_tile - max_tile_safe[None, :]),
                                        axis=0,
                                    )
                                    sum_exp_tile = tl.where(is_tile_neginf, 0.0, sum_exp_tile)
                                    # Cast to match accumulator precision.
                                    sum_exp_tile = sum_exp_tile.to(COMPUTE_DTYPE)

                                    # Merge this tile's statistics into the running accumulator.
                                    m_new = tl.maximum(m_beta_k, max_tile)
                                    is_m_neginf = m_beta_k < (NEG_INF + 1.0)
                                    m_new_safe = tl.where(is_m_neginf & is_tile_neginf, 0.0, m_new)

                                    # is_m_neginf means the accumulator is still empty (first non-trivial
                                    # tile). The standard update would compute l_beta_k * exp(m_beta_k -
                                    # m_new_safe) = 0 * exp(-inf - ...), which is 0 mathematically but
                                    # can evaluate to nan on some hardware. Short-circuit to start fresh.
                                    l_beta_k = tl.where(
                                        is_m_neginf,
                                        sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                        l_beta_k * tl.exp(m_beta_k - m_new_safe)
                                        + sum_exp_tile * tl.exp(max_tile - m_new_safe),
                                    )
                                    m_beta_k = m_new

                                # --- After all tiles: finalize beta_k via online logsumexp ---
                                # When l_beta_k == 0 (all inputs were NEG_INF), log(0) would produce
                                # -inf or nan. The guard short-circuits to NEG_INF directly.
                                is_beta_k_neginf = m_beta_k < (NEG_INF + 1.0)
                                beta_k = tl.where(
                                    is_beta_k_neginf,
                                    NEG_INF,
                                    m_beta_k + tl.log(l_beta_k),
                                )
                                beta_k = tl.where(c_mask, beta_k, NEG_INF)

                                # Write accumulated grad_duration_bias for this k (1 write instead of tiles atomics)
                                # Use dur_idx (not k) to handle K=1: k=1 maps to index 0
                                # Use segment-specific workspace for deterministic accumulation
                                tl.atomic_add(
                                    grad_db_ws_seg
                                    + dur_idx * stride_gdbw_k
                                    + c_idx * stride_gdbw_c,
                                    grad_db_k_local,
                                    mask=c_mask,
                                )

                                # Accumulate boundary marginals for this k
                                if RETURN_BOUNDARY_MARGINALS:
                                    tl.atomic_add(
                                        boundary_marginals_ptr
                                        + batch_idx * stride_bm_b
                                        + t * stride_bm_t,
                                        marginal_sum_all_k,
                                    )

                                # Online logsumexp: accumulate beta_k into (m_new_beta, l_new_beta)
                                # 2 exp + 0 log per k (vs pairwise: 2 exp + 1 log per k)
                                beta_k = beta_k.to(COMPUTE_DTYPE)
                                m_new = tl.maximum(m_new_beta, beta_k)
                                is_m_neginf = m_new_beta < (NEG_INF + 1.0)
                                is_bk_neginf = beta_k < (NEG_INF + 1.0)
                                m_new_safe = tl.where(is_m_neginf & is_bk_neginf, 0.0, m_new)
                                l_new_beta = tl.where(
                                    is_m_neginf,
                                    tl.exp(beta_k - m_new_safe),
                                    l_new_beta * tl.exp(m_new_beta - m_new_safe)
                                    + tl.exp(beta_k - m_new_safe),
                                )
                                m_new_beta = m_new

                        # --- Post-k: finalize beta[t], write grad_cum_scores[t], store to ring ---

                        # Finalize online logsumexp for new_beta.
                        # Guard: tl.log(l_new_beta) is only safe when l_new_beta > 0.
                        is_beta_neginf = m_new_beta < (NEG_INF + 1.0)
                        new_beta = tl.where(
                            is_beta_neginf, NEG_INF, m_new_beta + tl.log(l_new_beta)
                        )
                        new_beta = tl.where(c_mask, new_beta, NEG_INF)

                        # Write accumulated grad_cum_scores[t]: single atomic instead of
                        # K*tiles atomics (e.g., 8000 -> 1 at K=1000, C=32).
                        tl.atomic_add(
                            grad_cs_base + t * stride_gcs_t + c_idx * stride_gcs_c,
                            grad_cs_t_local,
                            mask=c_mask,
                        )

                        # Store beta[t] to the ring buffer.
                        # The ring uses 2*K slots (double-buffered) so that beta[t]
                        # and beta[t+K] never alias, even when t % K == (t+K) % K.
                        t_ring_idx = t % (2 * K)
                        tl.store(
                            beta_ring_base + t_ring_idx * stride_br_k + c_idx * stride_br_c,
                            new_beta,
                            mask=c_mask,
                        )

                        # Memory barrier: beta[t] must be visible before t-1 reads it.
                        # Without this, a tile at t-1 may read stale NEG_INF from
                        # ring initialization rather than the value just stored here.
                        tl.debug_barrier()

            # Sync all warps before beginning the next segment.
            tl.debug_barrier()

    def _compute_tile_c(C: int) -> int:
        """Compute adaptive TILE_C for c_dst dimension tiling.

        Hybrid strategy that balances compile time, register pressure, and correctness
        for use with tl.static_range (compile-time loop unrolling).

        Args:
            C: Actual number of classes (before padding).

        Returns:
            TILE_C value to use for kernel launch.

        Strategy:
            - Small C (≤16):    Use smaller tiles (4-8) for 2 iterations
            - Medium C (32-64): Use TILE_C=16 for 2-4 iterations
            - Large C (≥128):   Use TILE_C=32 for 4-8 iterations

        This balances:
            - Compile time:     Bounded to ≤8 unrolled iterations (even at C=256)
            - Register pressure: TILE_C ≤ 32 avoids spilling (~120 regs/thread)
            - Correctness:      tl.static_range handles scalar accumulation correctly

        Example:
            >>> _compute_tile_c(8)    # C_PAD=8   -> TILE_C=4  (2 iterations)
            4
            >>> _compute_tile_c(32)   # C_PAD=32  -> TILE_C=16 (2 iterations)
            16
            >>> _compute_tile_c(128)  # C_PAD=128 -> TILE_C=32 (4 iterations)
            32
            >>> _compute_tile_c(256)  # C_PAD=256 -> TILE_C=32 (8 iterations)
            32
        """
        C_PAD = _next_power_of_2(C)

        if C_PAD >= 256:
            return 32  # 256/32 = 8 iterations (max acceptable)
        elif C_PAD >= 128:
            return 32  # 128/32 = 4 iterations
        elif C_PAD >= 64:
            return 16  # 64/16 = 4 iterations
        elif C_PAD <= 8:
            return 4  # 8/4 = 2 iterations (minimal)
        elif C_PAD <= 16:
            return 8  # 16/8 = 2 iterations
        else:  # C_PAD == 32
            return 16  # 32/16 = 2 iterations

    def launch_streaming_triton_backward(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        log_Z: torch.Tensor,
        ring_checkpoints: torch.Tensor,
        log_norm_checkpoints: torch.Tensor,
        checkpoint_interval: int,
        grad_output: torch.Tensor,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
        return_boundary_marginals: bool = False,
        num_warps: int = 4,
        validate_cache: bool = True,
        precision: str = "float64",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""launch_streaming_triton_backward(cum_scores, transition, duration_bias, lengths, log_Z, ring_checkpoints, log_norm_checkpoints, checkpoint_interval, grad_output, proj_start=None, proj_end=None, return_boundary_marginals=False, num_warps=4, validate_cache=True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

        Launch the Triton backward kernel with proper buffer allocation.

        This function allocates working memory (alpha buffer, beta ring buffer)
        and dispatches the backward kernel. Gradients for shared parameters
        are accumulated per-batch then reduced via einsum.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` for
                static transitions, or :math:`(K, C, C)` for duration-dependent.
            duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            log_Z (Tensor): Partition values from forward of shape :math:`(\text{batch},)`.
            ring_checkpoints (Tensor): Saved ring buffer states of shape
                :math:`(\text{batch}, \text{num\_ckpts}, K, C)`. The class
                dimension must be ``C`` (not ``C_PAD``); the forward launcher
                trims padding before returning checkpoints.
            log_norm_checkpoints (Tensor): Cumulative log normalization factors
                at each checkpoint of shape :math:`(\text{batch}, \text{num\_ckpts})`.
            checkpoint_interval (int): Interval used during forward pass.
            grad_output (Tensor): Upstream gradient of shape :math:`(\text{batch},)`.
            proj_start (Tensor, optional): Start boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            proj_end (Tensor, optional): End boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            return_boundary_marginals (bool, optional): If ``True``, also compute
                and return boundary marginals. Default: ``False``
            num_warps (int, optional): Number of warps per block for Triton kernel.
                Higher values increase parallelism but also register pressure.
                Recommended range: 2-8. Default: ``4``
            validate_cache (bool, optional): If True, validate Triton cache
                consistency and warn on config changes. Default: ``True``

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple of:
                - **grad_cum_scores** (Tensor): Gradient for cumulative scores of shape
                  :math:`(\text{batch}, T+1, C)`.
                - **grad_transition** (Tensor): Gradient for transitions of shape
                  :math:`(C, C)` or :math:`(K, C, C)`.
                - **grad_duration_bias** (Tensor): Gradient for duration bias of shape
                  :math:`(K, C)`.
                - **grad_proj_start** (Tensor or None): Gradient for start projections of shape
                  :math:`(\text{batch}, T, C)` if boundaries provided, else ``None``.
                - **grad_proj_end** (Tensor or None): Gradient for end projections of shape
                  :math:`(\text{batch}, T, C)` if boundaries provided, else ``None``.
                - **boundary_marginals** (Tensor or None): Boundary marginal probabilities of shape
                  :math:`(\text{batch}, T)` if ``return_boundary_marginals=True``, else ``None``.

        .. note::
            Requires both or neither boundary tensor (proj_start, proj_end).
            For single-boundary support, use :func:`semi_crf_streaming_forward`
            which automatically falls back to the PyTorch path.

        See Also:
            :func:`launch_streaming_triton_kernel`: Forward pass that produces checkpoints
                for this backward pass.
            :func:`launch_streaming_triton_marginals`: Convenience wrapper for computing
                boundary marginals without gradients.
        """
        from .triton_cache import TritonConfig, update_cache_sentinel, validate_triton_cache

        # Extract dimensions to compute adaptive TILE_C early
        batch, T_plus_1, C = cum_scores.shape

        # Compute adaptive TILE_C based on number of classes
        # Forces multiple tiles at small C to reduce atomic contention
        tile_c = _compute_tile_c(C)
        # DEBUG
        # tile_c = C_PAD

        # Validate cache if requested (only validates num_warps)
        # Note: TILE_C is handled automatically by Triton's cache via tl.constexpr
        if validate_cache:
            config = TritonConfig(num_warps=num_warps)
            validate_triton_cache(config)
            update_cache_sentinel(config)
        T = T_plus_1 - 1
        K = duration_bias.shape[0]
        validate_streaming_shapes(K, C, batch, T, transition, duration_bias, proj_start, proj_end)

        # Validate checkpoint tensors
        if ring_checkpoints.ndim != 4:
            raise ValueError(
                f"ring_checkpoints must be 4D (batch, num_ckpts, K, C), got {ring_checkpoints.ndim}D"
            )
        if log_norm_checkpoints.ndim != 2:
            raise ValueError(
                f"log_norm_checkpoints must be 2D (batch, num_ckpts), got {log_norm_checkpoints.ndim}D"
            )
        if ring_checkpoints.shape[0] != batch:
            raise ValueError(
                f"ring_checkpoints batch dim must equal {batch}, got {ring_checkpoints.shape[0]}"
            )
        if log_norm_checkpoints.shape[0] != batch:
            raise ValueError(
                f"log_norm_checkpoints batch dim must equal {batch}, got {log_norm_checkpoints.shape[0]}"
            )
        if ring_checkpoints.shape[1] != log_norm_checkpoints.shape[1]:
            raise ValueError(
                f"ring_checkpoints and log_norm_checkpoints checkpoint count must match: "
                f"ring_checkpoints has {ring_checkpoints.shape[1]}, "
                f"log_norm_checkpoints has {log_norm_checkpoints.shape[1]}"
            )
        if ring_checkpoints.shape[2] != K:
            raise ValueError(
                f"ring_checkpoints K dim must equal K={K}, got {ring_checkpoints.shape[2]}"
            )
        if ring_checkpoints.shape[3] != C:
            raise ValueError(
                f"ring_checkpoints class dim must equal C={C}, got {ring_checkpoints.shape[3]}"
            )

        device = cum_scores.device
        use_float32 = precision == "float32"
        compute_dtype = torch.float32 if use_float32 else torch.float64

        num_checkpoints = ring_checkpoints.shape[1]
        C_PAD = _next_power_of_2(C)
        even_c = C == C_PAD

        # Compute segment size for alpha buffer
        segment_size = checkpoint_interval + K

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

        # Ensure contiguous
        cum_scores = cum_scores.contiguous()
        transition = transition.contiguous()
        duration_bias = duration_bias.contiguous()
        lengths = lengths.contiguous()
        log_Z = log_Z.contiguous()
        grad_output = grad_output.contiguous()

        # Handle boundary projections
        if has_boundaries:
            proj_start = proj_start.contiguous()
            proj_end = proj_end.contiguous()
            stride_ps_b, stride_ps_t, stride_ps_c = proj_start.stride()
            # Allocate gradient outputs for boundaries with C_PAD
            # (we slice back to C and convert to original dtype before returning)
            grad_proj_start = torch.zeros(batch, T, C_PAD, device=device, dtype=torch.float32)
            grad_proj_end = torch.zeros(batch, T, C_PAD, device=device, dtype=torch.float32)
            # Grad buffers have shape (batch, T, C_PAD) — different strides from
            # proj_start (batch, T, C) when C is not a power of 2.
            stride_gps_b, stride_gps_t, stride_gps_c = grad_proj_start.stride()
        else:
            # Create dummy tensors for stride calculation (won't be accessed)
            proj_start = cum_scores[:, :T, :]
            proj_end = cum_scores[:, :T, :]
            stride_ps_b, stride_ps_t, stride_ps_c = 0, 0, 0
            stride_gps_b, stride_gps_t, stride_gps_c = 0, 0, 0
            grad_proj_start = None
            grad_proj_end = None
            # Kernel uses proj_start/proj_end directly when HAS_BOUNDARY_PROJ=False

        # Pad checkpoints to C_PAD (match forward's compute_dtype)
        if ring_checkpoints.shape[-1] < C_PAD:
            ring_ckpts_padded = torch.full(
                (batch, num_checkpoints, K, C_PAD), NEG_INF, device=device, dtype=compute_dtype
            )
            ring_ckpts_padded[:, :, :, :C] = ring_checkpoints.to(compute_dtype)
        else:
            ring_ckpts_padded = ring_checkpoints.to(compute_dtype).contiguous()

        # Ensure log normalization checkpoints are contiguous
        log_norm_checkpoints = log_norm_checkpoints.contiguous()

        # Compute number of segments (one per checkpoint)
        num_segments = num_checkpoints

        # Allocate working memory
        # alpha_buffer must be segment-specific to prevent race conditions between segments.
        # Without segment isolation, segment 1 (processed first in backward) writes to
        # alpha_buffer[batch, 0, :], then segment 0 reads stale values instead of checkpoint.
        alpha_buffer = torch.full(
            (batch, num_segments, segment_size, C_PAD), NEG_INF, device=device, dtype=compute_dtype
        )
        beta_ring = torch.full((batch, 2 * K, C_PAD), NEG_INF, device=device, dtype=compute_dtype)

        # NUMERICAL STABILITY: Selective precision for gradient tensors.
        # - grad_cum_scores: per-position (no cross-T accumulation) -> always float32
        # - grad_transition, grad_duration_bias: per-segment workspace isolation bounds
        #   atomic add count, making compute_dtype sufficient. Host-side sum over segments
        #   and einsum with grad_output use float64 for final reduction.

        # Allocate gradient outputs with C_PAD
        # grad_cum_scores is per-position, doesn't accumulate across T -> float32 sufficient
        grad_cum_scores = torch.zeros(batch, T_plus_1, C_PAD, device=device, dtype=torch.float32)
        # grad_duration_bias: final output (reduced on host in float64)
        grad_duration_bias = torch.zeros(K, C, device=device, dtype=torch.float64)

        # Allocate per-batch-per-segment workspace buffers for deterministic gradients.
        # Each segment writes to its own workspace slice, eliminating atomic contention
        # across segments. Host-side sum over segments is deterministic.
        # PADDING: Use C_PAD to prevent OOB memory access from masked-out threads.
        if has_duration_transitions:
            # Duration-dependent: (batch, num_segments, K, C_PAD, C_PAD)
            grad_tr_workspace = torch.zeros(
                batch, num_segments, K, C_PAD, C_PAD, device=device, dtype=compute_dtype
            )
        else:
            # Static: (batch, num_segments, C_PAD, C_PAD)
            grad_tr_workspace = torch.zeros(
                batch, num_segments, C_PAD, C_PAD, device=device, dtype=compute_dtype
            )
        grad_db_workspace = torch.zeros(
            batch, num_segments, K, C_PAD, device=device, dtype=compute_dtype
        )

        # Allocate boundary marginals output if requested
        if return_boundary_marginals:
            boundary_marginals = torch.zeros(batch, T, device=device, dtype=compute_dtype)
            stride_bm_b, stride_bm_t = boundary_marginals.stride()
        else:
            boundary_marginals = grad_cum_scores[:, :T, 0]  # Dummy (won't be written)
            stride_bm_b, stride_bm_t = 0, 0

        # Get strides
        stride_cs_b, stride_cs_t, stride_cs_c = cum_scores.stride()

        # Handle transition strides for both (C, C) and (K, C, C)
        if has_duration_transitions:
            stride_tr_k, stride_tr_src, stride_tr_dst = transition.stride()
        else:
            stride_tr_k = 0  # Not used for static transitions
            stride_tr_src, stride_tr_dst = transition.stride()

        stride_db_k, stride_db_c = duration_bias.stride()
        stride_ckpt_b, stride_ckpt_n, stride_ckpt_k, stride_ckpt_c = ring_ckpts_padded.stride()
        # alpha_buffer now has segment dimension: (batch, num_segments, segment_size, C_PAD)
        stride_ab_b, stride_ab_seg, stride_ab_t, stride_ab_c = alpha_buffer.stride()
        stride_br_b, stride_br_k, stride_br_c = beta_ring.stride()
        stride_gcs_b, stride_gcs_t, stride_gcs_c = grad_cum_scores.stride()

        # Handle grad_tr_workspace strides for both shapes
        # Workspace now has segment dimension: (batch, num_segments, [K,] C_PAD, C_PAD)
        if has_duration_transitions:
            stride_gtw_b, stride_gtw_seg, stride_gtw_k, stride_gtw_src, stride_gtw_dst = (
                grad_tr_workspace.stride()
            )
        else:
            stride_gtw_k = 0  # Not used for static transitions
            stride_gtw_b, stride_gtw_seg, stride_gtw_src, stride_gtw_dst = (
                grad_tr_workspace.stride()
            )

        # Workspace: (batch, num_segments, K, C_PAD)
        stride_gdbw_b, stride_gdbw_seg, stride_gdbw_k, stride_gdbw_c = grad_db_workspace.stride()
        stride_lnc_b, stride_lnc_n = log_norm_checkpoints.stride()

        # Use actual gradients or dummies for kernel call
        grad_ps_for_kernel = grad_proj_start if has_boundaries else grad_cum_scores
        grad_pe_for_kernel = grad_proj_end if has_boundaries else grad_cum_scores

        # Launch kernel with device context for multi-GPU support
        # Note: tile_c was computed earlier for cache validation
        grid = (batch,)
        with torch.cuda.device(device):
            semi_crf_streaming_backward_kernel[grid](
                cum_scores,
                transition,
                duration_bias,
                lengths,
                log_Z,
                ring_ckpts_padded,
                grad_output,
                proj_start,
                proj_end,
                alpha_buffer,
                beta_ring,
                grad_cum_scores,
                grad_tr_workspace,
                grad_db_workspace,
                grad_ps_for_kernel,
                grad_pe_for_kernel,
                boundary_marginals,
                batch,
                T,
                K,
                C,
                C_PAD,
                checkpoint_interval,
                num_checkpoints,
                segment_size,
                has_boundaries,  # HAS_BOUNDARIES constexpr
                has_duration_transitions,  # HAS_DURATION_TRANSITIONS constexpr
                return_boundary_marginals,  # RETURN_BOUNDARY_MARGINALS constexpr
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
                stride_gps_b,
                stride_gps_t,
                stride_gps_c,
                stride_ckpt_b,
                stride_ckpt_n,
                stride_ckpt_k,
                stride_ckpt_c,
                stride_ab_b,
                stride_ab_seg,
                stride_ab_t,
                stride_ab_c,
                stride_br_b,
                stride_br_k,
                stride_br_c,
                stride_gcs_b,
                stride_gcs_t,
                stride_gcs_c,
                stride_gtw_b,
                stride_gtw_seg,
                stride_gtw_k,
                stride_gtw_src,
                stride_gtw_dst,
                stride_gdbw_b,
                stride_gdbw_seg,
                stride_gdbw_k,
                stride_gdbw_c,
                stride_bm_b,
                stride_bm_t,
                log_norm_checkpoints,
                stride_lnc_b,
                stride_lnc_n,
                TILE_C=tile_c,
                EVEN_C=even_c,
                USE_FLOAT32=use_float32,
                num_warps=num_warps,
            )

        # Compute weighted sum of per-batch gradients for shared parameters.
        #
        # Correct gradient semantics for shared parameters:
        #   grad_θ = Σ_b[grad_output[b] * Σ_{t,k}(marginal[b,t,k])]
        #
        # NOT the buggy formula:
        #   grad_θ = Σ_{b,t,k}(marginal[b,t,k]) * Σ_b(grad_output[b])  # WRONG!
        #
        # The difference matters when grad_output varies across batch elements
        # (e.g., masked sequences, weighted losses). With uniform grad_output=[1,1,...],
        # both formulas happen to give the same result, which is why tests using
        # .sum().backward() didn't catch the bug.
        #
        # We use einsum for memory efficiency: it fuses the multiply + reduce
        # without creating a large intermediate tensor. For K=1024, C=64, batch=16,
        # the naive broadcast approach would allocate ~268MB just to sum immediately.
        #
        # Notation: b=batch, s=segment, k=duration, i=src_state, j=dst_state, c=state
        #
        # DETERMINISTIC REDUCTION: Sum over segments first (deterministic order),
        # then einsum over batch. This eliminates non-determinism from cross-segment
        # atomic contention while preserving correct gradient semantics.
        #
        # Slice workspaces back to actual class count C before reduction
        # (they were allocated with C_PAD to prevent OOB memory access)
        # Host-side reduction ALWAYS uses float64 for numerical precision,
        # regardless of kernel compute_dtype. Upcast workspaces before sum/einsum.
        grad_output_f64 = grad_output.to(torch.float64)
        if has_duration_transitions:
            # (batch, num_segments, K, C_PAD, C_PAD) -> sum over segments -> (batch, K, C, C)
            grad_tr_workspace = grad_tr_workspace[:, :, :, :C, :C].to(torch.float64).sum(dim=1)
            grad_transition = torch.einsum("bkij, b -> kij", grad_tr_workspace, grad_output_f64)
        else:
            # (batch, num_segments, C_PAD, C_PAD) -> sum over segments -> (batch, C, C)
            grad_tr_workspace = grad_tr_workspace[:, :, :C, :C].to(torch.float64).sum(dim=1)
            grad_transition = torch.einsum("bij, b -> ij", grad_tr_workspace, grad_output_f64)

        # (batch, num_segments, K, C_PAD) -> sum over segments -> (batch, K, C)
        grad_db_workspace = grad_db_workspace[:, :, :, :C].to(torch.float64).sum(dim=1)
        grad_duration_bias = torch.einsum("bkc, b -> kc", grad_db_workspace, grad_output_f64)

        # Slice padded gradients back to actual class count C and convert to float64
        # (autograd layer will cast to input dtype)
        grad_cum_scores = grad_cum_scores[:, :, :C].to(torch.float64)
        grad_transition = grad_transition.to(torch.float64)
        grad_duration_bias = grad_duration_bias.to(torch.float64)

        if grad_proj_start is not None:
            grad_proj_start = grad_proj_start[:, :, :C].to(torch.float64)
            grad_proj_end = grad_proj_end[:, :, :C].to(torch.float64)

        return (
            grad_cum_scores,
            grad_transition,
            grad_duration_bias,
            grad_proj_start,
            grad_proj_end,
            boundary_marginals if return_boundary_marginals else None,
        )

    def launch_streaming_triton_marginals(
        cum_scores: torch.Tensor,
        transition: torch.Tensor,
        duration_bias: torch.Tensor,
        lengths: torch.Tensor,
        log_Z: torch.Tensor,
        ring_checkpoints: torch.Tensor,
        log_norm_checkpoints: torch.Tensor,
        checkpoint_interval: int,
        proj_start: torch.Tensor = None,
        proj_end: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""launch_streaming_triton_marginals(cum_scores, transition, duration_bias, lengths, log_Z, ring_checkpoints, log_norm_checkpoints, checkpoint_interval, proj_start=None, proj_end=None) -> Tensor

        Compute boundary marginals via Triton backward kernel.

        This is a convenience wrapper that runs the backward kernel with
        ``return_boundary_marginals=True`` and discards the gradient outputs.
        The boundary marginal at position t represents the probability that
        a segment starts at that position.

        Args:
            cum_scores (Tensor): Cumulative projected scores of shape
                :math:`(\text{batch}, T+1, C)`.
            transition (Tensor): Transition scores of shape :math:`(C, C)` for
                static transitions, or :math:`(K, C, C)` for duration-dependent.
            duration_bias (Tensor): Duration-specific bias of shape :math:`(K, C)`.
            lengths (Tensor): Sequence lengths of shape :math:`(\text{batch},)`.
            log_Z (Tensor): Partition values from forward of shape :math:`(\text{batch},)`.
            ring_checkpoints (Tensor): Saved ring buffer states of shape
                :math:`(\text{batch}, \text{num\_ckpts}, K, C)`. The class
                dimension must be ``C`` (not ``C_PAD``); the forward launcher
                trims padding before returning checkpoints.
            log_norm_checkpoints (Tensor): Cumulative log normalization factors of shape
                :math:`(\text{batch}, \text{num\_ckpts})`. Required for numerical stability
                at extreme sequence lengths.
            checkpoint_interval (int): Interval used during forward pass.
            proj_start (Tensor, optional): Start boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``
            proj_end (Tensor, optional): End boundary scores of shape
                :math:`(\text{batch}, T, C)`. Default: ``None``

        Returns:
            Tensor: Boundary marginal probabilities of shape :math:`(\text{batch}, T)`.
                Each value is the probability that a segment starts at that position.

        .. note::
            Requires CUDA and Triton. For CPU-compatible computation, use
            :func:`~flash_semicrf.streaming.semi_crf_streaming_marginals_pytorch`.

        Example::

            >>> # Setup (on CUDA)
            >>> cum_scores = torch.zeros(2, 101, 4, device='cuda')
            >>> cum_scores[:, 1:] = torch.cumsum(torch.randn(2, 100, 4, device='cuda'), dim=1)
            >>> transition = torch.randn(4, 4, device='cuda')
            >>> duration_bias = torch.randn(10, 4, device='cuda')  # K=10
            >>> lengths = torch.tensor([100, 80], device='cuda')
            >>> # Forward pass to get checkpoints
            >>> log_Z, ring_ckpts, interval, log_norm_ckpts = launch_streaming_triton_kernel(
            ...     cum_scores, transition, duration_bias, lengths, K=10
            ... )
            >>> # Compute boundary marginals
            >>> marginals = launch_streaming_triton_marginals(
            ...     cum_scores, transition, duration_bias, lengths,
            ...     log_Z, ring_ckpts, log_norm_ckpts, interval
            ... )
            >>> marginals.shape
            torch.Size([2, 100])

        See Also:
            :func:`launch_streaming_triton_backward`: Full backward pass with gradients
            :func:`~flash_semicrf.streaming.semi_crf_streaming_marginals_pytorch`:
                PyTorch reference implementation (CPU compatible)
        """
        batch = cum_scores.shape[0]
        # Use ones for grad_output since we only want marginals, not scaled gradients
        grad_output = torch.ones(batch, device=cum_scores.device, dtype=cum_scores.dtype)

        _, _, _, _, _, boundary_marginals = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_checkpoints,
            log_norm_checkpoints,
            checkpoint_interval,
            grad_output,
            proj_start,
            proj_end,
            return_boundary_marginals=True,
        )

        return boundary_marginals
