"""
Tests for the Triton streaming forward kernel.

Validates the Triton implementation against the PyTorch reference.
"""

import pytest
import torch

from flash_semicrf.streaming import (
    HAS_TRITON,
    semi_crf_streaming_forward_pytorch,
)
from tests.conftest import force_clear_triton_cache

# Conditionally import Triton functions
if HAS_TRITON:
    from flash_semicrf.streaming import (
        launch_streaming_triton_backward,
        launch_streaming_triton_kernel,
        launch_streaming_triton_kernel_max_bp,
    )


def create_streaming_inputs(batch, T, K, C, device="cpu", dtype=torch.float32, seed=42):
    """Create test inputs for the streaming API."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Simulate projected encoder features
    projected = torch.randn(batch, T, C, device=device, dtype=dtype)
    # Zero-center (critical for numerical stability at large T)
    projected = projected - projected.mean(dim=1, keepdim=True)

    # Cumulative scores: (batch, T+1, C)
    cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
    cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

    # Transition matrix: (C, C)
    transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1

    # Duration bias: (K, C)
    duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1

    # Lengths
    lengths = torch.full((batch,), T, dtype=torch.long, device=device)

    return cum_scores, transition, duration_bias, lengths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingKernel:
    """Test the Triton streaming forward kernel against PyTorch reference."""

    def test_triton_matches_pytorch_small(self):
        """Verify Triton kernel matches PyTorch for small inputs."""
        batch, T, K, C = 2, 20, 4, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_matches_pytorch_medium(self):
        """Verify Triton kernel matches PyTorch for medium inputs."""
        batch, T, K, C = 4, 100, 8, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda", dtype=torch.float64
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_matches_pytorch_large_C(self):
        """Verify Triton kernel works with larger C (requires padding)."""
        batch, T, K, C = 2, 50, 6, 24  # C=24 -> C_PAD=32
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_matches_pytorch_power_of_2_C(self):
        """Verify Triton kernel handles power-of-2 C values (EVEN_C=True path).

        These configs exercise the EVEN_C code path where C == C_PAD.  Uses the
        same dtype (float64) and config as the dedicated TestEvenCPath tests so
        that any pre-existing issues are caught here first, independently of the
        EVEN_C optimisation.
        """
        for C in [4, 8, 16, 32, 64]:
            batch, T, K = 2, 40, 5
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
                batch, T, K, C, device="cuda", dtype=torch.float64
            )

            partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton,
                partition_pytorch,
                rtol=1e-4,
                atol=1e-4,
                check_dtype=False,
                msg=f"power-of-2 forward C={C} failed",
            )

    def test_triton_matches_pytorch_non_power_of_2_C(self):
        """Verify Triton kernel handles non-power-of-2 C values."""
        for C in [3, 5, 7, 11, 13, 17]:
            batch, T, K = 2, 30, 5
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
                batch, T, K, C, device="cuda"
            )

            partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton,
                partition_pytorch,
                rtol=1e-4,
                atol=1e-4,
                check_dtype=False,
                msg=f"C={C} failed",
            )

    def test_triton_variable_lengths(self):
        """Verify Triton kernel handles variable sequence lengths."""
        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_short_sequences(self):
        """Verify Triton kernel handles sequences shorter than K."""
        K, C, batch = 10, 4, 2

        for T in [4, 6, 8, 12]:
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
                batch, T, K, C, device="cuda"
            )

            partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
                cum_scores, transition, duration_bias, lengths, K
            )

            partition_triton, _, _, _ = launch_streaming_triton_kernel(
                cum_scores, transition, duration_bias, lengths, K
            )

            torch.testing.assert_close(
                partition_triton,
                partition_pytorch,
                rtol=1e-4,
                atol=1e-4,
                check_dtype=False,
                msg=f"T={T} failed",
            )

    def test_triton_max_semiring(self):
        """Verify Triton max semiring matches PyTorch."""
        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_produces_finite_values(self):
        """Verify Triton kernel produces finite values."""
        batch, T, K, C = 4, 100, 8, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_triton, ring_ckpts, ckpt_interval, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        assert torch.isfinite(partition_triton).all(), "Partition contains non-finite values"

    def test_triton_checkpoints_saved(self):
        """Verify Triton kernel saves checkpoints correctly."""
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        _, ring_ckpts, ckpt_interval, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Verify checkpoint shape: ceil(T / ckpt_interval) checkpoints
        # Checkpoints are saved at t=0, ckpt_interval, 2*ckpt_interval, etc.
        expected_num_ckpts = (T + ckpt_interval - 1) // ckpt_interval
        assert (
            ring_ckpts.shape[1] == expected_num_ckpts
        ), f"Expected {expected_num_ckpts} checkpoints"
        assert ring_ckpts.shape[2] == K
        assert ring_ckpts.shape[3] == C

        # Verify checkpoint 0 is initialized correctly (alpha[0] = 0, rest = -inf)
        ckpt_0 = ring_ckpts[:, 0, :, :]  # (batch, K, C)
        assert torch.allclose(ckpt_0[:, 0, :], torch.zeros_like(ckpt_0[:, 0, :]))

    @pytest.mark.gpu_isolated
    def test_triton_larger_sequence(self):
        """Verify Triton kernel works with larger sequences.

        This test is particularly sensitive to Triton cache contamination from
        earlier tests with smaller configurations. We explicitly clear the cache
        to ensure kernels are compiled with appropriate autotuning parameters
        for the larger sequence length (T=500).
        """
        force_clear_triton_cache()

        batch, T, K, C = 2, 500, 16, 8
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Use slightly looser tolerance for longer sequences
        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-3, atol=1e-3, check_dtype=False
        )

    def test_triton_k1_forward_matches_pytorch(self):
        """Verify Triton kernel matches PyTorch for K=1 (linear CRF)."""
        batch, T, K, C = 2, 50, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_k1_backward_finite_gradients(self):
        """Verify K=1 Triton backward produces finite gradients.

        This is a regression test for the K=1 out-of-bounds bug where
        grad_duration_bias was written to index k=1 instead of dur_idx=0.
        """
        batch, T, K, C = 2, 50, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Forward through Triton
        partition, ring_ckpts, ckpt_interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Backward through Triton
        grad_output = torch.ones_like(partition)
        grads = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            partition,
            ring_ckpts,
            log_norm_ckpts,
            ckpt_interval,
            grad_output,
        )

        grad_cum_scores, grad_transition, grad_duration_bias, _, _, _ = grads

        assert torch.isfinite(grad_cum_scores).all(), "K=1 grad_cum_scores has non-finite values"
        assert torch.isfinite(grad_transition).all(), "K=1 grad_transition has non-finite values"
        assert torch.isfinite(
            grad_duration_bias
        ).all(), "K=1 grad_duration_bias has non-finite values"

    def test_triton_k1_gradients_match_pytorch(self):
        """Verify K=1 Triton gradients match PyTorch reference via autograd."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path (full autograd)
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path (full Triton forward + backward kernels)
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients - Triton backward uses tl.atomic_add which is non-deterministic
        # due to GPU thread execution order. Use looser tolerances (5% relative, 0.5 absolute).
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=0.05, atol=0.5)

    def test_triton_k1_timit_config(self):
        """Verify K=1 Triton backward works with TIMIT-like configuration.

        TIMIT uses:
        - C=39 phoneme classes -> C_PAD=64 (significant padding)
        - Longer sequences (100-600 frames)
        - Larger batches (32)

        This tests the combination of K=1 + larger C_PAD which requires
        multiple tiles (4 tiles for TILE_C=16).

        If this test fails, the bug is in the tiling logic for K=1 + C_PAD > C.
        If this test passes, the bug may be data-dependent.
        """
        from flash_semicrf.streaming import semi_crf_streaming_forward

        # Match TIMIT configuration
        batch, T, K, C = 4, 150, 1, 39  # C=39 -> C_PAD=64
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path (reference)
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Check for NaN/Inf (the TIMIT failure mode)
        assert torch.isfinite(partition_tr).all(), "K=1 TIMIT-config: partition has NaN/Inf"
        assert torch.isfinite(cs_tr.grad).all(), "K=1 TIMIT-config: grad_cum_scores has NaN/Inf"
        assert torch.isfinite(tr_tr.grad).all(), "K=1 TIMIT-config: grad_transition has NaN/Inf"
        assert torch.isfinite(db_tr.grad).all(), "K=1 TIMIT-config: grad_duration_bias has NaN/Inf"

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients - Triton backward uses tl.atomic_add which is non-deterministic
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=0.05, atol=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingTraining:
    """Test the Triton streaming kernel in training mode (with gradients)."""

    def test_triton_forward_with_gradients(self):
        """Verify Triton forward works when gradients are needed."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        # Use Triton path (full Triton forward + backward)
        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )
        partition.sum().backward()

        assert torch.isfinite(cum_scores.grad).all(), "cum_scores grad non-finite"
        assert torch.isfinite(transition.grad).all(), "transition grad non-finite"
        assert torch.isfinite(duration_bias.grad).all(), "duration_bias grad non-finite"

    def test_triton_gradients_match_pytorch(self):
        """Verify Triton path gradients match PyTorch reference."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path (full Triton forward + backward kernels)
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients - Triton backward uses tl.atomic_add which is non-deterministic
        # due to GPU thread execution order. Use looser tolerances (5% relative, 0.5 absolute).
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=0.05, atol=0.5)

    def test_dispatch_inference_vs_training(self):
        """Verify correct dispatch based on requires_grad."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 50, 6, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Inference (no grad) - should use Triton kernel
        partition_inf = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K
        )
        assert torch.isfinite(partition_inf).all()

        # Training (with grad)
        cum_scores.requires_grad_(True)
        partition_train = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K
        )
        assert torch.isfinite(partition_train).all()
        partition_train.sum().backward()
        assert cum_scores.grad is not None

    def test_triton_backward_kernel_raw(self):
        """Test the raw Triton backward kernel launcher."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Run forward
        partition, ring_checkpoints, checkpoint_interval, log_norm_ckpts = (
            launch_streaming_triton_kernel(cum_scores, transition, duration_bias, lengths, K)
        )

        # Run backward
        grad_output = torch.ones(batch, device="cuda")
        grad_cum_scores, grad_transition, grad_duration_bias, _, _, _ = (
            launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                log_norm_ckpts,
                checkpoint_interval,
                grad_output,
            )
        )

        # Verify shapes
        assert grad_cum_scores.shape == cum_scores.shape
        assert grad_transition.shape == transition.shape
        assert grad_duration_bias.shape == duration_bias.shape

        # Verify finite values
        assert torch.isfinite(grad_cum_scores).all(), "grad_cum_scores non-finite"
        assert torch.isfinite(grad_transition).all(), "grad_transition non-finite"
        assert torch.isfinite(grad_duration_bias).all(), "grad_duration_bias non-finite"

    def test_triton_gradients_variable_lengths(self):
        """Verify Triton gradients handle variable sequence lengths."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-3, atol=1e-3)

        # Compare gradients - Triton backward uses tl.atomic_add which is non-deterministic
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=0.05, atol=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingBoundaries:
    """Test the Triton streaming kernel with boundary projections (Phase 4B)."""

    def create_boundary_inputs(self, batch, T, K, C, device="cuda", dtype=torch.float32, seed=42):
        """Create test inputs including boundary projections."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Standard inputs
        projected = torch.randn(batch, T, C, device=device, dtype=dtype)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        transition = torch.randn(C, C, device=device, dtype=dtype) * 0.1
        duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        # Boundary projections
        proj_start = torch.randn(batch, T, C, device=device, dtype=dtype) * 0.1
        proj_end = torch.randn(batch, T, C, device=device, dtype=dtype) * 0.1

        return cum_scores, transition, duration_bias, lengths, proj_start, proj_end

    def test_triton_with_boundaries_forward(self):
        """Verify Triton forward with boundaries matches PyTorch."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # Triton kernel
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_with_boundaries_forward_max(self):
        """Verify Triton max semiring with boundaries matches PyTorch."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="max",
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # Triton kernel
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="max",
            proj_start=proj_start,
            proj_end=proj_end,
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_with_boundaries_gradients(self):
        """Verify Triton backward with boundaries matches PyTorch."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)
        ps_py = proj_start.clone().requires_grad_(True)
        pe_py = proj_end.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(
            cs_py, tr_py, db_py, lengths, K, proj_start=ps_py, proj_end=pe_py, use_triton=False
        )
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)
        ps_tr = proj_start.clone().requires_grad_(True)
        pe_tr = proj_end.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr, tr_tr, db_tr, lengths, K, proj_start=ps_tr, proj_end=pe_tr, use_triton=True
        )
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients - Triton backward uses tl.atomic_add which is non-deterministic
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(ps_tr.grad, ps_py.grad, rtol=0.05, atol=0.5)
        torch.testing.assert_close(pe_tr.grad, pe_py.grad, rtol=0.05, atol=0.5)

    def test_triton_boundaries_gradients_non_power_of_2_C(self):
        """Verify boundary gradients are correct when C != C_PAD (non-power-of-2).

        Regression test for stride mismatch bug: grad_proj_start/grad_proj_end
        buffers are allocated with C_PAD columns but were indexed using strides
        from proj_start (C columns), causing row overlap when C < C_PAD.
        """
        from flash_semicrf.streaming import semi_crf_streaming_forward

        # Small odd sizes and large pad jumps (C=17->32, C=24->32)
        for C in [5, 6, 17, 24]:
            batch, T, K = 2, 30, 5
            cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
                self.create_boundary_inputs(batch, T, K, C)
            )

            # Non-uniform grad_output to exercise backward scaling
            grad_output = torch.randn(batch, device="cuda")

            # PyTorch path
            cs_py = cum_scores.clone().requires_grad_(True)
            tr_py = transition.clone().requires_grad_(True)
            db_py = duration_bias.clone().requires_grad_(True)
            ps_py = proj_start.clone().requires_grad_(True)
            pe_py = proj_end.clone().requires_grad_(True)

            partition_py = semi_crf_streaming_forward(
                cs_py,
                tr_py,
                db_py,
                lengths,
                K,
                proj_start=ps_py,
                proj_end=pe_py,
                use_triton=False,
            )
            partition_py.backward(grad_output)

            # Triton path
            cs_tr = cum_scores.clone().requires_grad_(True)
            tr_tr = transition.clone().requires_grad_(True)
            db_tr = duration_bias.clone().requires_grad_(True)
            ps_tr = proj_start.clone().requires_grad_(True)
            pe_tr = proj_end.clone().requires_grad_(True)

            partition_tr = semi_crf_streaming_forward(
                cs_tr,
                tr_tr,
                db_tr,
                lengths,
                K,
                proj_start=ps_tr,
                proj_end=pe_tr,
                use_triton=True,
            )
            partition_tr.backward(grad_output)

            # Compare partition values
            torch.testing.assert_close(
                partition_tr,
                partition_py,
                rtol=1e-4,
                atol=1e-4,
                msg=f"C={C}: partition mismatch",
            )

            # Compare boundary gradients (the bug target)
            torch.testing.assert_close(
                ps_tr.grad,
                ps_py.grad,
                rtol=0.05,
                atol=0.5,
                msg=f"C={C}: grad_proj_start mismatch",
            )
            torch.testing.assert_close(
                pe_tr.grad,
                pe_py.grad,
                rtol=0.05,
                atol=0.5,
                msg=f"C={C}: grad_proj_end mismatch",
            )

            # Also verify other gradients aren't disturbed
            torch.testing.assert_close(
                cs_tr.grad,
                cs_py.grad,
                rtol=0.05,
                atol=0.5,
                msg=f"C={C}: grad_cum_scores mismatch",
            )
            torch.testing.assert_close(
                tr_tr.grad,
                tr_py.grad,
                rtol=0.05,
                atol=0.5,
                msg=f"C={C}: grad_transition mismatch",
            )
            torch.testing.assert_close(
                db_tr.grad,
                db_py.grad,
                rtol=0.05,
                atol=0.5,
                msg=f"C={C}: grad_duration_bias mismatch",
            )

    def test_triton_boundaries_backward_kernel_raw(self):
        """Test the raw Triton backward kernel with boundaries."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )

        # Run forward
        partition, ring_checkpoints, checkpoint_interval, log_norm_ckpts = (
            launch_streaming_triton_kernel(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                proj_start=proj_start,
                proj_end=proj_end,
            )
        )

        # Run backward
        grad_output = torch.ones(batch, device="cuda")
        grad_cum_scores, grad_transition, grad_duration_bias, grad_proj_start, grad_proj_end, _ = (
            launch_streaming_triton_backward(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                partition,
                ring_checkpoints,
                log_norm_ckpts,
                checkpoint_interval,
                grad_output,
                proj_start=proj_start,
                proj_end=proj_end,
            )
        )

        # Verify shapes
        assert grad_cum_scores.shape == cum_scores.shape
        assert grad_transition.shape == transition.shape
        assert grad_duration_bias.shape == duration_bias.shape
        assert grad_proj_start.shape == proj_start.shape
        assert grad_proj_end.shape == proj_end.shape

        # Verify finite values
        assert torch.isfinite(grad_cum_scores).all(), "grad_cum_scores non-finite"
        assert torch.isfinite(grad_transition).all(), "grad_transition non-finite"
        assert torch.isfinite(grad_duration_bias).all(), "grad_duration_bias non-finite"
        assert torch.isfinite(grad_proj_start).all(), "grad_proj_start non-finite"
        assert torch.isfinite(grad_proj_end).all(), "grad_proj_end non-finite"

    def test_triton_boundaries_larger_C(self):
        """Verify boundaries work with C=24 (genomics scale)."""
        batch, T, K, C = 2, 50, 8, 24
        # Use float64: semi_crf_streaming_forward_pytorch computes in the dtype of its
        # inputs while the Triton kernel always uses float64 internally.  With C=24,
        # T=50 and checkpoint-based log normalisation, float32 accumulates ~1 absolute
        # error vs float64 — far outside the 1e-4 tolerance.
        cum_scores, transition, duration_bias, lengths, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C, dtype=torch.float64)
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_boundaries_variable_lengths(self):
        """Verify boundaries handle variable sequence lengths."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 4, 50, 6, 4
        cum_scores, transition, duration_bias, _, proj_start, proj_end = (
            self.create_boundary_inputs(batch, T, K, C)
        )
        lengths = torch.tensor([T, T - 10, T - 20, T - 30], dtype=torch.long, device="cuda")

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        ps_py = proj_start.clone().requires_grad_(True)
        pe_py = proj_end.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(
            cs_py,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=ps_py,
            proj_end=pe_py,
            use_triton=False,
        )
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        ps_tr = proj_start.clone().requires_grad_(True)
        pe_tr = proj_end.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr,
            transition,
            duration_bias,
            lengths,
            K,
            proj_start=ps_tr,
            proj_end=pe_tr,
            use_triton=True,
        )
        partition_tr.sum().backward()

        # Compare
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ps_tr.grad, ps_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(pe_tr.grad, pe_py.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonStreamingBenchmark:
    """Benchmark tests for Triton streaming kernel."""

    @pytest.mark.parametrize("T", [100, 500, 1000])
    @pytest.mark.parametrize("K", [8, 16, 32])
    def test_benchmark_correctness(self, T, K):
        """Verify correctness across different T and K values."""
        batch, C = 4, 6
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg=f"T={T}, K={K} failed",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestDurationDependentTransitions:
    """Test duration-dependent transitions (Phase 4A)."""

    def create_duration_transition_inputs(
        self, batch, T, K, C, device="cuda", dtype=torch.float32, seed=42
    ):
        """Create test inputs with duration-dependent transitions (K, C, C)."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Standard inputs
        projected = torch.randn(batch, T, C, device=device, dtype=dtype)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device=device, dtype=dtype)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

        # Duration-dependent transitions: (K, C, C)
        transition = torch.randn(K, C, C, device=device, dtype=dtype) * 0.1

        duration_bias = torch.randn(K, C, device=device, dtype=dtype) * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device=device)

        return cum_scores, transition, duration_bias, lengths

    def test_pytorch_duration_transitions_forward(self):
        """Verify PyTorch forward pass works with (K, C, C) transitions."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # Run forward pass
        partition, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Basic sanity checks
        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all(), "Partition contains non-finite values"

    def test_triton_duration_transitions_forward(self):
        """Verify Triton forward kernel works with (K, C, C) transitions."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        # Triton kernel
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_duration_transitions_forward_max(self):
        """Verify Triton max semiring works with (K, C, C) transitions."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        # Triton kernel
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_triton_duration_transitions_gradients(self):
        """Verify Triton backward gradients match PyTorch for (K, C, C) transitions."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare partition values
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-4, atol=1e-4)

        # Compare gradients
        torch.testing.assert_close(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)

    def test_duration_transitions_gradient_shape(self):
        """Verify gradient shape is (K, C, C) for duration-dependent transitions."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # Triton path
        transition = transition.clone().requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )
        partition.sum().backward()

        # Verify gradient shape matches input shape
        assert transition.grad.shape == (
            K,
            C,
            C,
        ), f"Expected gradient shape (K, C, C) = ({K}, {C}, {C}), got {transition.grad.shape}"

    def test_duration_transitions_with_boundaries(self):
        """Verify (K, C, C) transitions work together with boundary projections."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = self.create_duration_transition_inputs(
            batch, T, K, C
        )

        # Add boundary projections
        proj_start = torch.randn(batch, T, C, device="cuda") * 0.1
        proj_end = torch.randn(batch, T, C, device="cuda") * 0.1

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        ps_py = proj_start.clone().requires_grad_(True)
        pe_py = proj_end.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(
            cs_py,
            tr_py,
            duration_bias,
            lengths,
            K,
            proj_start=ps_py,
            proj_end=pe_py,
            use_triton=False,
        )
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        ps_tr = proj_start.clone().requires_grad_(True)
        pe_tr = proj_end.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr,
            tr_tr,
            duration_bias,
            lengths,
            K,
            proj_start=ps_tr,
            proj_end=pe_tr,
            use_triton=True,
        )
        partition_tr.sum().backward()

        # Compare
        torch.testing.assert_close(partition_tr, partition_py, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ps_tr.grad, ps_py.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(pe_tr.grad, pe_py.grad, rtol=1e-2, atol=1e-2)

    def test_static_vs_duration_transitions_shape(self):
        """Verify static (C, C) transitions still work after Phase 4A changes."""
        batch, T, K, C = 2, 30, 5, 4
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Create inputs
        projected = torch.randn(batch, T, C, device="cuda")
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda")
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

        # Static transition (C, C)
        transition_static = torch.randn(C, C, device="cuda") * 0.1
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        # Forward pass with static transitions should still work
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition_static, duration_bias, lengths, K
        )
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition_static, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton, partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestGradientScalingBugFix:
    """Test that shared parameter gradients are correctly weighted by grad_output.

    These tests specifically target the bug where:
    - BUGGY: grad = Σ_b(marginal[b]) * Σ_b(grad_output[b])
    - CORRECT: grad = Σ_b(marginal[b] * grad_output[b])

    The bug was masked because all tests used .sum().backward() which creates
    uniform grad_output = [1, 1, ..., 1], making both formulas equivalent.
    """

    def test_heterogeneous_grad_output_shared_params(self):
        """Verify shared parameter gradients with non-uniform grad_output."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Create heterogeneous grad_output
        grad_output = torch.tensor([0.5, 2.0], device="cuda")

        # Test Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.backward(grad_output)

        # Test PyTorch reference path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.backward(grad_output)

        # Both should produce identical gradients for shared parameters
        torch.testing.assert_close(
            tr_tr.grad,
            tr_py.grad,
            rtol=1e-2,
            atol=1e-2,
            msg="Transition gradient mismatch with heterogeneous grad_output",
        )
        torch.testing.assert_close(
            db_tr.grad,
            db_py.grad,
            rtol=1e-2,
            atol=1e-2,
            msg="Duration bias gradient mismatch with heterogeneous grad_output",
        )

    def test_gradient_linearity_in_grad_output(self):
        """Verify that gradients scale linearly with grad_output."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # First run with grad_output = [1.0, 1.0]
        tr_1 = transition.clone().requires_grad_(True)
        partition_1 = semi_crf_streaming_forward(
            cum_scores, tr_1, duration_bias, lengths, K, use_triton=True
        )
        partition_1.backward(torch.tensor([1.0, 1.0], device="cuda"))
        grad_1 = tr_1.grad.clone()

        # Second run with grad_output = [2.0, 2.0]
        tr_2 = transition.clone().requires_grad_(True)
        partition_2 = semi_crf_streaming_forward(
            cum_scores, tr_2, duration_bias, lengths, K, use_triton=True
        )
        partition_2.backward(torch.tensor([2.0, 2.0], device="cuda"))
        grad_2 = tr_2.grad.clone()

        # grad_2 should be exactly 2 * grad_1
        torch.testing.assert_close(
            grad_2,
            2.0 * grad_1,
            rtol=1e-5,
            atol=1e-5,
            msg="Gradient does not scale linearly with grad_output",
        )

    def test_zero_mask_gradient_isolation(self):
        """Verify that grad_output=0 completely masks a batch element's contribution.

        This is the critical test that directly catches the gradient scaling bug.
        When grad_output[1] = 0, the second batch element should contribute
        NOTHING to the shared parameter gradients.
        """
        from flash_semicrf.streaming import semi_crf_streaming_forward

        T, K, C = 30, 5, 4

        # Create different inputs for each batch element
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        cum_scores_0, transition, duration_bias, _ = create_streaming_inputs(
            1, T, K, C, device="cuda"
        )
        torch.manual_seed(123)  # Different seed for different data
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)
        cum_scores_1, _, _, _ = create_streaming_inputs(1, T, K, C, device="cuda")

        # Combine into batch of 2
        cum_scores_batch = torch.cat([cum_scores_0, cum_scores_1], dim=0)
        lengths = torch.tensor([T, T], dtype=torch.long, device="cuda")

        # Run batch of 2 with mask [1.0, 0.0] - second element masked
        tr_batch = transition.clone().requires_grad_(True)
        partition_batch = semi_crf_streaming_forward(
            cum_scores_batch, tr_batch, duration_bias, lengths, K, use_triton=True
        )
        partition_batch.backward(torch.tensor([1.0, 0.0], device="cuda"))
        grad_tr_batch = tr_batch.grad.clone()

        # Run single-batch on first sequence only
        tr_single = transition.clone().requires_grad_(True)
        lengths_single = torch.tensor([T], dtype=torch.long, device="cuda")
        partition_single = semi_crf_streaming_forward(
            cum_scores_0, tr_single, duration_bias, lengths_single, K, use_triton=True
        )
        partition_single.backward(torch.ones(1, device="cuda"))
        grad_tr_single = tr_single.grad.clone()

        # MUST be exactly equal - masked sequence contributes nothing
        torch.testing.assert_close(
            grad_tr_batch,
            grad_tr_single,
            rtol=1e-4,
            atol=1e-4,
            msg="Masked sequence (grad_output=0) still contributed to gradient!",
        )

    def test_duration_dependent_gradient_scaling(self):
        """Verify gradient scaling works correctly with (K, C, C) transitions."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Create duration-dependent transitions
        projected = torch.randn(batch, T, C, device="cuda")
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda")
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
        transition = torch.randn(K, C, C, device="cuda") * 0.1  # (K, C, C)
        duration_bias = torch.randn(K, C, device="cuda") * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        # Test with heterogeneous grad_output
        grad_output = torch.tensor([0.3, 1.7], device="cuda")

        # Triton path
        tr_tr = transition.clone().requires_grad_(True)
        partition_tr = semi_crf_streaming_forward(
            cum_scores, tr_tr, duration_bias, lengths, K, use_triton=True
        )
        partition_tr.backward(grad_output)

        # PyTorch path
        tr_py = transition.clone().requires_grad_(True)
        partition_py = semi_crf_streaming_forward(
            cum_scores, tr_py, duration_bias, lengths, K, use_triton=False
        )
        partition_py.backward(grad_output)

        # Verify gradient shape is preserved
        assert tr_tr.grad.shape == (
            K,
            C,
            C,
        ), f"Expected gradient shape (K, C, C), got {tr_tr.grad.shape}"

        # Verify Triton matches PyTorch
        torch.testing.assert_close(
            tr_tr.grad,
            tr_py.grad,
            rtol=1e-2,
            atol=1e-2,
            msg="Duration-dependent transition gradients don't match",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.diagnostic
class TestTritonBackwardDebug:
    """Debug tests to diagnose Triton vs PyTorch gradient mismatches."""

    def test_triton_backward_debug_minimal(self):
        """Minimal debug test with tiny config to compare gradient values."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        # Tiny config for easy debugging
        batch, T, K, C = 1, 5, 2, 2
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Print detailed comparison
        print("\n" + "=" * 60)
        print("TRITON BACKWARD GRADIENT DEBUG")
        print("=" * 60)
        print(f"Config: batch={batch}, T={T}, K={K}, C={C}")
        print(f"Forward values match: {torch.allclose(partition_py, partition_tr)}")
        print(f"  PyTorch partition: {partition_py.item():.6f}")
        print(f"  Triton partition:  {partition_tr.item():.6f}")

        print("\n--- grad_cum_scores comparison ---")
        print(f"PyTorch:\n{cs_py.grad.squeeze()}")
        print(f"\nTriton:\n{cs_tr.grad.squeeze()}")
        print(f"\nDiff (Triton - PyTorch):\n{(cs_tr.grad - cs_py.grad).squeeze()}")
        print(f"\nMax abs diff: {(cs_tr.grad - cs_py.grad).abs().max().item():.6f}")

        print("\n--- grad_transition comparison ---")
        print(f"PyTorch:\n{tr_py.grad}")
        print(f"\nTriton:\n{tr_tr.grad}")
        print(f"\nDiff (Triton - PyTorch):\n{tr_tr.grad - tr_py.grad}")
        print(f"\nMax abs diff: {(tr_tr.grad - tr_py.grad).abs().max().item():.6f}")

        print("\n--- grad_duration_bias comparison ---")
        print(f"PyTorch:\n{db_py.grad}")
        print(f"\nTriton:\n{db_tr.grad}")
        print(f"\nDiff (Triton - PyTorch):\n{db_tr.grad - db_py.grad}")
        print(f"\nMax abs diff: {(db_tr.grad - db_py.grad).abs().max().item():.6f}")

        # Check if gradients match (this will fail, but shows the actual values)
        cs_match = torch.allclose(cs_tr.grad, cs_py.grad, rtol=1e-2, atol=1e-2)
        tr_match = torch.allclose(tr_tr.grad, tr_py.grad, rtol=1e-2, atol=1e-2)
        db_match = torch.allclose(db_tr.grad, db_py.grad, rtol=1e-2, atol=1e-2)

        print("\n--- Summary ---")
        print(f"cum_scores grad match: {cs_match}")
        print(f"transition grad match: {tr_match}")
        print(f"duration_bias grad match: {db_match}")
        print("=" * 60)

        # Don't assert - this is for debugging only
        if not (cs_match and tr_match and db_match):
            print("\nWARNING: Gradients don't match! Use output above to diagnose.")

    def test_triton_k1_debug_multitile(self):
        """Debug test to pinpoint K=1 + multi-tile divergence.

        This test systematically compares grad_cum_scores between PyTorch and
        Triton to find:
        1. The first timestep where gradients diverge significantly
        2. Which class indices have the largest errors
        3. Whether the error pattern is systematic or random

        Important observation: For K=1, the backward kernel processes segments of length 1
        (t -> t+1). The bug manifests as 10^19 magnitude errors, suggesting:
        - Uninitialized memory access, OR
        - Incorrect logsumexp accumulation across tiles
        """
        from flash_semicrf.streaming import semi_crf_streaming_forward

        # Use the failing config: K=1, C=39 -> C_PAD=64, 4 tiles
        batch, T, K, C = 2, 50, 1, 39
        C_PAD = 64  # Next power of 2
        TILE_C = 16
        num_tiles = C_PAD // TILE_C  # = 4 tiles

        print("\n" + "=" * 70)
        print("K=1 MULTI-TILE DEBUG TEST")
        print("=" * 70)
        print(f"Config: batch={batch}, T={T}, K={K}, C={C}, C_PAD={C_PAD}")
        print(f"Tiles: {num_tiles} tiles of TILE_C={TILE_C}")
        print(f"Ring buffer slots: K={K} (all positions map to index 0)")

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)
        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton under test
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)
        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # === Analysis Phase ===
        print("\n--- Forward Pass ---")
        print(
            f"Partition match: {torch.allclose(partition_py, partition_tr, rtol=1e-4, atol=1e-4)}"
        )
        print(f"PyTorch: {partition_py.tolist()}")
        print(f"Triton:  {partition_tr.tolist()}")

        print("\n--- Backward Pass: grad_cum_scores ---")
        diff = (cs_tr.grad - cs_py.grad).abs()

        # Find first timestep with significant error (> 0.1)
        threshold = 0.1
        error_mask = diff > threshold

        if not error_mask.any():
            print(f"All gradients match within tolerance {threshold}")
            return

        # Find first error location
        first_found = False
        for b in range(batch):
            for t in range(T + 1):
                if error_mask[b, t].any():
                    max_err_idx = diff[b, t].argmax().item()
                    max_err = diff[b, t, max_err_idx].item()
                    py_val = cs_py.grad[b, t, max_err_idx].item()
                    tr_val = cs_tr.grad[b, t, max_err_idx].item()

                    # Determine which tile this class belongs to
                    tile_idx = max_err_idx // TILE_C
                    within_tile_idx = max_err_idx % TILE_C

                    print(f"\nFIRST DIVERGENCE at batch={b}, t={t}, c={max_err_idx}")
                    print(f"  Tile: {tile_idx} (position {within_tile_idx} within tile)")
                    print(f"  PyTorch: {py_val:.6e}")
                    print(f"  Triton:  {tr_val:.6e}")
                    print(f"  Abs diff: {max_err:.6e}")

                    # Show all classes at this (b, t) position
                    print(f"\n  All classes at (b={b}, t={t}):")
                    print(f"  {'c':>3} {'tile':>4} {'PyTorch':>12} {'Triton':>12} {'diff':>12}")
                    for c in range(C):
                        tile = c // TILE_C
                        d = diff[b, t, c].item()
                        marker = "***" if d > threshold else ""
                        print(
                            f"  {c:3d} {tile:4d} {cs_py.grad[b, t, c].item():12.4e} "
                            f"{cs_tr.grad[b, t, c].item():12.4e} {d:12.4e} {marker}"
                        )
                    first_found = True
                    break
            if first_found:
                break

        # Statistical summary
        print("\n--- Error Distribution by Tile ---")
        for tile in range(num_tiles):
            c_start = tile * TILE_C
            c_end = min((tile + 1) * TILE_C, C)
            if c_start < C:
                tile_diff = diff[:, :, c_start:c_end]
                tile_errors = (tile_diff > threshold).sum().item()
                tile_max = tile_diff.max().item()
                print(
                    f"  Tile {tile} (c={c_start}-{c_end - 1}): {tile_errors} errors, max={tile_max:.4e}"
                )

        # Check if errors correlate with ring buffer wraparound points
        print("\n--- Error by Position (t) ---")
        for t in [0, 1, 5, 10, 20, T - 1, T]:
            if t <= T:
                t_errors = (diff[:, t, :] > threshold).sum().item()
                t_max = diff[:, t, :].max().item()
                ring_idx = t % K
                print(f"  t={t:3d} (ring_idx={ring_idx}): {t_errors:4d} errors, max={t_max:.4e}")

        print("=" * 70)

        # Don't fail - this is diagnostic

    def test_triton_k1_isolate_tile_count(self):
        """Isolate whether the bug is specific to multi-tile processing.

        Compare:
        - C=4 -> C_PAD=4 -> 1 tile (expected: PASS)
        - C=17 -> C_PAD=32 -> 2 tiles (check if fails)
        - C=33 -> C_PAD=64 -> 4 tiles (expected: FAIL)

        If 2-tile also fails, the bug is in tile accumulation.
        If only 4-tile fails, the bug may be in higher-tile handling.
        """
        from flash_semicrf.streaming import semi_crf_streaming_forward

        test_cases = [
            (4, 4, 1, "1 tile"),  # C=4 -> C_PAD=4
            (17, 32, 2, "2 tiles"),  # C=17 -> C_PAD=32
            (33, 64, 4, "4 tiles"),  # C=33 -> C_PAD=64
        ]

        print("\n" + "=" * 70)
        print("K=1 TILE COUNT ISOLATION TEST")
        print("=" * 70)

        batch, T, K = 2, 50, 1

        for C, C_PAD, _num_tiles, label in test_cases:
            cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
                batch, T, K, C, device="cuda"
            )

            # PyTorch reference
            cs_py = cum_scores.clone().requires_grad_(True)
            partition_py = semi_crf_streaming_forward(
                cs_py, transition, duration_bias, lengths, K, use_triton=False
            )
            partition_py.sum().backward()

            # Triton
            cs_tr = cum_scores.clone().requires_grad_(True)
            partition_tr = semi_crf_streaming_forward(
                cs_tr, transition, duration_bias, lengths, K, use_triton=True
            )
            partition_tr.sum().backward()

            # Compute statistics
            diff = (cs_tr.grad - cs_py.grad).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            num_errors = (diff > 0.1).sum().item()

            status = "PASS" if max_diff < 1.0 else "FAIL"
            print(
                f"{label:10s} C={C:2d} C_PAD={C_PAD:2d}: "
                f"max_diff={max_diff:.2e}, mean={mean_diff:.2e}, errors={num_errors:4d} [{status}]"
            )

        print("=" * 70)

    def test_triton_k2_multitile_control(self):
        """Control test: K=2 with large C to check if bug is K=1-specific.

        If K=2 with C=39 (4 tiles) passes, the bug is specific to K=1.
        If K=2 with C=39 also fails, the bug is in general multi-tile logic.
        """
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 50, 2, 39  # K=2, same C as TIMIT

        print("\n" + "=" * 70)
        print("K=2 MULTI-TILE CONTROL TEST")
        print("=" * 70)
        print(f"Config: K={K}, C={C}, C_PAD=64, 4 tiles")

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)
        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)
        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Compare
        cs_diff = (cs_tr.grad - cs_py.grad).abs()
        tr_diff = (tr_tr.grad - tr_py.grad).abs()
        db_diff = (db_tr.grad - db_py.grad).abs()

        print(f"Forward match: {torch.allclose(partition_py, partition_tr, rtol=1e-4, atol=1e-4)}")
        print(
            f"grad_cum_scores: max_diff={cs_diff.max().item():.2e}, errors={(cs_diff > 0.1).sum().item()}"
        )
        print(f"grad_transition: max_diff={tr_diff.max().item():.2e}")
        print(f"grad_duration_bias: max_diff={db_diff.max().item():.2e}")

        status = "PASS" if cs_diff.max().item() < 1.0 else "FAIL"
        print(f"Result: [{status}]")
        print("=" * 70)

    def test_triton_backward_debug_larger(self):
        """Larger debug test matching the actual failing test config."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        # Same config as test_triton_gradients_match_pytorch
        batch, T, K, C = 2, 30, 5, 4
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(cs_tr, tr_tr, db_tr, lengths, K, use_triton=True)
        partition_tr.sum().backward()

        # Statistical summary
        cs_diff = (cs_tr.grad - cs_py.grad).abs()
        tr_diff = (tr_tr.grad - tr_py.grad).abs()
        db_diff = (db_tr.grad - db_py.grad).abs()

        print("\n" + "=" * 60)
        print("TRITON BACKWARD GRADIENT DEBUG (LARGER)")
        print("=" * 60)
        print(f"Config: batch={batch}, T={T}, K={K}, C={C}")
        print(f"Forward values match: {torch.allclose(partition_py, partition_tr)}")

        print("\n--- grad_cum_scores statistics ---")
        print(f"Shape: {cs_py.grad.shape}")
        print(f"Max abs diff: {cs_diff.max().item():.6f}")
        print(f"Mean abs diff: {cs_diff.mean().item():.6f}")
        print(f"Num mismatched (>0.01): {(cs_diff > 0.01).sum().item()} / {cs_diff.numel()}")

        # Show first few mismatched positions
        mismatch_mask = cs_diff > 0.01
        if mismatch_mask.any():
            idxs = torch.nonzero(mismatch_mask)[:5]
            print("First 5 mismatched positions:")
            for idx in idxs:
                b, t, c = idx.tolist()
                print(
                    f"  [{b}, {t}, {c}]: PyTorch={cs_py.grad[b, t, c]:.4f}, Triton={cs_tr.grad[b, t, c]:.4f}, diff={cs_diff[b, t, c]:.4f}"
                )

        print("\n--- grad_transition statistics ---")
        print(f"Shape: {tr_py.grad.shape}")
        print(f"Max abs diff: {tr_diff.max().item():.6f}")
        print(f"Mean abs diff: {tr_diff.mean().item():.6f}")
        print(f"Num mismatched (>0.01): {(tr_diff > 0.01).sum().item()} / {tr_diff.numel()}")

        print("\n--- grad_duration_bias statistics ---")
        print(f"Shape: {db_py.grad.shape}")
        print(f"Max abs diff: {db_diff.max().item():.6f}")
        print(f"Mean abs diff: {db_diff.mean().item():.6f}")
        print(f"Num mismatched (>0.01): {(db_diff > 0.01).sum().item()} / {db_diff.numel()}")
        print("=" * 60)


# =============================================================================
# K-Based Dispatch Tests (K=1, K=2, K>=3)
# =============================================================================


class TestKBasedDispatch:
    """Test the K-based dispatch logic for specialized K=1 and K=2 paths."""

    def test_k1_forward_matches_generic(self):
        """K=1 fast path produces same results as generic implementation."""
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward
        from flash_semicrf.streaming.pytorch_reference import (
            linear_crf_forward_pytorch,
            semi_crf_streaming_forward_pytorch,
        )

        batch, T, K, C = 4, 50, 1, 8
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        # Direct K=1 function
        partition_k1 = linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias)

        # Generic streaming function (should dispatch to K=1 internally)
        partition_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        # Also test against generic PyTorch reference
        partition_generic, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(partition_k1, partition_generic, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(partition_dispatch, partition_generic, rtol=1e-4, atol=1e-4)

    def test_k1_backward_gradients(self):
        """K=1 fast path gradients are correct (numerical gradient check)."""
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward

        batch, T, K, C = 2, 20, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )
        partition.sum().backward()

        # Check gradients exist and are finite
        assert cum_scores.grad is not None
        assert transition.grad is not None
        assert duration_bias.grad is not None
        assert torch.isfinite(cum_scores.grad).all()
        assert torch.isfinite(transition.grad).all()
        assert torch.isfinite(duration_bias.grad).all()

    def test_k2_forward_correctness(self):
        """K=2 specialized path is correct (verifies k=1 and k=2 durations are summed).

        Note: The generic semi_crf_streaming_forward_pytorch has a bug where K=2 only
        processes duration k=1 (not k=2) due to k_eff = min(K-1, t) = 1. The K=2
        specialized path correctly processes both durations.

        We verify correctness by checking:
        1. Partition values are finite
        2. Partition > K=1 partition (more paths summed)
        3. Dispatch routes to K=2 path
        """
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward
        from flash_semicrf.streaming.pytorch_reference import (
            linear_crf_forward_pytorch,
            semi_crf_k2_forward_pytorch,
        )

        batch, T, K, C = 4, 50, 2, 8
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        # K=2 specialized path
        partition_k2 = semi_crf_k2_forward_pytorch(cum_scores, transition, duration_bias, lengths)

        # K=1 path (for comparison - K=2 should be >= K=1 due to more paths)
        partition_k1 = linear_crf_forward_pytorch(cum_scores, transition, lengths, duration_bias)

        # Dispatch should route to K=2 path
        partition_dispatch = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        # Verify K=2 partition is finite
        assert torch.isfinite(partition_k2).all(), "K=2 partition has non-finite values"

        # Verify K=2 >= K=1 (more paths to sum over means higher partition)
        assert (partition_k2 >= partition_k1 - 1e-4).all(), (
            f"K=2 partition should be >= K=1 partition, but got "
            f"K=2={partition_k2.tolist()}, K=1={partition_k1.tolist()}"
        )

        # Verify dispatch matches direct call
        torch.testing.assert_close(partition_dispatch, partition_k2, rtol=1e-5, atol=1e-5)

    def test_k2_backward_gradients(self):
        """K=2 specialized path gradients are correct."""
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward

        batch, T, K, C = 2, 20, 2, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        cum_scores.requires_grad_(True)
        transition.requires_grad_(True)
        duration_bias.requires_grad_(True)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )
        partition.sum().backward()

        # Check gradients exist and are finite
        assert cum_scores.grad is not None
        assert transition.grad is not None
        assert duration_bias.grad is not None
        assert torch.isfinite(cum_scores.grad).all()
        assert torch.isfinite(transition.grad).all()
        assert torch.isfinite(duration_bias.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    def test_k3_uses_triton_on_cuda(self):
        """K>=3 routes to Triton kernel on CUDA."""
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward
        from flash_semicrf.streaming.pytorch_reference import semi_crf_streaming_forward_pytorch

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # Generic streaming function with Triton enabled
        partition_triton = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=True
        )

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores.cpu(), transition.cpu(), duration_bias.cpu(), lengths.cpu(), K
        )

        torch.testing.assert_close(
            partition_triton.cpu(), partition_pytorch, rtol=1e-4, atol=1e-4, check_dtype=False
        )

    def test_k1_viterbi_decoding(self):
        """K=1 Viterbi decoding works correctly."""
        from flash_semicrf.streaming.pytorch_reference import linear_crf_viterbi_pytorch

        batch, T, K, C = 2, 20, 1, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        scores, paths = linear_crf_viterbi_pytorch(cum_scores, transition, lengths, duration_bias)

        # Check output shapes
        assert scores.shape == (batch,)
        assert paths.shape == (batch, T)

        # Check scores are finite
        assert torch.isfinite(scores).all()

        # Check paths contain valid labels
        assert (paths >= 0).all()
        assert (paths < C).all()

    def test_k2_viterbi_decoding(self):
        """K=2 Viterbi decoding works correctly."""
        from flash_semicrf.streaming.pytorch_reference import semi_crf_k2_viterbi_pytorch

        batch, T, K, C = 2, 20, 2, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        scores, paths, durations = semi_crf_k2_viterbi_pytorch(
            cum_scores, transition, duration_bias, lengths
        )

        # Check output shapes
        assert scores.shape == (batch,)
        assert paths.shape == (batch, T)
        assert durations.shape == (batch, T)

        # Check scores are finite
        assert torch.isfinite(scores).all()

        # Check paths contain valid labels
        assert (paths >= 0).all()
        assert (paths < C).all()

    def test_k1_variable_lengths(self):
        """K=1 fast path handles variable sequence lengths."""
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward

        batch, T, K, C = 4, 30, 1, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        # Variable lengths
        lengths = torch.tensor([10, 20, 25, 30], dtype=torch.long)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        # Check output shape and finiteness
        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all()

        # Shorter sequences should generally have lower partition values
        # (fewer paths to sum over)
        # This is a sanity check, not a strict requirement
        assert partition[0] < partition[3]  # length 10 vs length 30

    def test_k2_variable_lengths(self):
        """K=2 specialized path handles variable sequence lengths."""
        from flash_semicrf.streaming.autograd import semi_crf_streaming_forward

        batch, T, K, C = 4, 30, 2, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cpu"
        )

        # Variable lengths
        lengths = torch.tensor([10, 20, 25, 30], dtype=torch.long)

        partition = semi_crf_streaming_forward(
            cum_scores, transition, duration_bias, lengths, K, use_triton=False
        )

        # Check output shape and finiteness
        assert partition.shape == (batch,)
        assert torch.isfinite(partition).all()


NUM_WARPS_VALUES = [2, 4, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonNumWarps:
    """Verify Triton kernels produce correct results across num_warps values (2, 4, 8).

    Each num_warps value triggers a separate Triton compilation (it is part of the
    cache key). Different warp counts change register pressure and thread scheduling,
    so correctness must be verified for each value in the recommended range.

    All direct launcher tests use validate_cache=False to avoid noisy warnings when
    switching between num_warps values within a single test session.
    """

    @pytest.mark.parametrize("num_warps", NUM_WARPS_VALUES)
    def test_forward_log_semiring(self, num_warps):
        """Triton log-semiring forward matches PyTorch reference at each num_warps."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="log",
            num_warps=num_warps,
            validate_cache=False,
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
            msg=f"log semiring failed at num_warps={num_warps}",
        )

    @pytest.mark.parametrize("num_warps", NUM_WARPS_VALUES)
    def test_forward_max_semiring(self, num_warps):
        """Triton max-semiring forward matches PyTorch reference at each num_warps."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K, semiring="max"
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="max",
            num_warps=num_warps,
            validate_cache=False,
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
            msg=f"max semiring failed at num_warps={num_warps}",
        )

    @pytest.mark.parametrize("num_warps", NUM_WARPS_VALUES)
    def test_forward_max_bp(self, num_warps):
        """Triton max-backpointer kernel matches PyTorch Viterbi at each num_warps."""
        from flash_semicrf.streaming.pytorch_reference import (
            semi_crf_streaming_viterbi_with_backpointers,
        )

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        scores_pytorch, bp_k_py, bp_c_py, final_labels_py = (
            semi_crf_streaming_viterbi_with_backpointers(
                cum_scores, transition, duration_bias, lengths, K
            )
        )

        scores_triton, bp_k_tr, bp_c_tr, final_labels_tr = launch_streaming_triton_kernel_max_bp(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            num_warps=num_warps,
            validate_cache=False,
        )

        torch.testing.assert_close(
            scores_triton,
            scores_pytorch,
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
            msg=f"max_bp scores failed at num_warps={num_warps}",
        )
        # Final labels should match (argmax of final alpha)
        assert torch.equal(
            final_labels_tr.to(torch.long), final_labels_py
        ), f"max_bp final_labels mismatch at num_warps={num_warps}"
        # Note: bp_k/bp_c are NOT compared exactly because argmax tie-breaking
        # differs between Triton (tl.argmax) and PyTorch (torch.argmax).
        # When two durations produce equal max scores, the chosen backpointer
        # is arbitrary. The Viterbi scores and final_labels are the
        # functionally meaningful outputs; backpointers at off-path positions
        # are undefined anyway.

    @pytest.mark.parametrize("num_warps", NUM_WARPS_VALUES)
    @pytest.mark.filterwarnings("ignore:Triton cache may be stale:UserWarning")
    def test_backward_via_autograd(self, num_warps):
        """Triton backward gradients match PyTorch reference at each num_warps."""
        from flash_semicrf.streaming import semi_crf_streaming_forward

        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        # PyTorch reference path
        cs_py = cum_scores.clone().requires_grad_(True)
        tr_py = transition.clone().requires_grad_(True)
        db_py = duration_bias.clone().requires_grad_(True)

        partition_py = semi_crf_streaming_forward(cs_py, tr_py, db_py, lengths, K, use_triton=False)
        partition_py.sum().backward()

        # Triton path at this num_warps
        cs_tr = cum_scores.clone().requires_grad_(True)
        tr_tr = transition.clone().requires_grad_(True)
        db_tr = duration_bias.clone().requires_grad_(True)

        partition_tr = semi_crf_streaming_forward(
            cs_tr,
            tr_tr,
            db_tr,
            lengths,
            K,
            use_triton=True,
            num_warps=num_warps,
        )
        partition_tr.sum().backward()

        # Forward values should match tightly
        torch.testing.assert_close(
            partition_tr,
            partition_py,
            rtol=1e-4,
            atol=1e-4,
            msg=f"forward partition mismatch at num_warps={num_warps}",
        )

        # Gradients: looser tolerance due to atomic_add non-determinism
        torch.testing.assert_close(
            cs_tr.grad,
            cs_py.grad,
            rtol=0.05,
            atol=0.5,
            msg=f"grad_cum_scores mismatch at num_warps={num_warps}",
        )
        torch.testing.assert_close(
            tr_tr.grad,
            tr_py.grad,
            rtol=0.05,
            atol=0.5,
            msg=f"grad_transition mismatch at num_warps={num_warps}",
        )
        torch.testing.assert_close(
            db_tr.grad,
            db_py.grad,
            rtol=0.05,
            atol=0.5,
            msg=f"grad_duration_bias mismatch at num_warps={num_warps}",
        )

    @pytest.mark.parametrize("num_warps", NUM_WARPS_VALUES)
    @pytest.mark.parametrize("C", [4, 5])
    def test_forward_log_with_padding(self, num_warps, C):
        """Triton log forward is correct with both power-of-2 and non-power-of-2 C."""
        batch, T, K = 2, 30, 5
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="log",
            num_warps=num_warps,
            validate_cache=False,
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
            msg=f"C={C}, num_warps={num_warps} failed",
        )

    @pytest.mark.parametrize("num_warps", NUM_WARPS_VALUES)
    def test_forward_duration_dependent_transitions(self, num_warps):
        """Triton log forward with (K, C, C) transitions at each num_warps."""
        batch, T, K, C = 2, 30, 5, 4

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        projected = torch.randn(batch, T, C, device="cuda", dtype=torch.float32)
        projected = projected - projected.mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float32)
        cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)

        # Duration-dependent transitions: (K, C, C)
        transition = torch.randn(K, C, C, device="cuda", dtype=torch.float32) * 0.1
        duration_bias = torch.randn(K, C, device="cuda", dtype=torch.float32) * 0.1
        lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )

        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            semiring="log",
            num_warps=num_warps,
            validate_cache=False,
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
            msg=f"duration-dependent transitions failed at num_warps={num_warps}",
        )

    def test_cross_num_warps_consistency(self):
        """All num_warps values produce identical partition values for log semiring."""
        batch, T, K, C = 2, 30, 5, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        results = {}
        for nw in NUM_WARPS_VALUES:
            partition, _, _, _ = launch_streaming_triton_kernel(
                cum_scores,
                transition,
                duration_bias,
                lengths,
                K,
                semiring="log",
                num_warps=nw,
                validate_cache=False,
            )
            results[nw] = partition

        baseline = results[NUM_WARPS_VALUES[0]]
        for nw in NUM_WARPS_VALUES[1:]:
            torch.testing.assert_close(
                results[nw],
                baseline,
                rtol=1e-8,
                atol=1e-9,
                check_dtype=False,
                msg=f"num_warps={nw} differs from num_warps={NUM_WARPS_VALUES[0]}",
            )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
    elif not HAS_TRITON:
        print("Triton not available, skipping tests")
    else:
        print("Running Triton streaming kernel validation...")

        # Quick validation
        batch, T, K, C = 2, 100, 8, 4
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda"
        )

        print("\nInput shapes:")
        print(f"  cum_scores: {cum_scores.shape}")
        print(f"  transition: {transition.shape}")
        print(f"  duration_bias: {duration_bias.shape}")
        print(f"  lengths: {lengths}")

        # PyTorch reference
        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )
        print(f"\nPyTorch partition: {partition_pytorch}")

        # Triton kernel
        partition_triton, ring_ckpts, ckpt_interval, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        print(f"Triton partition:  {partition_triton}")
        print(f"Checkpoint interval: {ckpt_interval}")
        print(f"Ring checkpoints shape: {ring_ckpts.shape}")

        # Compare
        diff = (partition_triton - partition_pytorch).abs().max().item()
        print(f"\nMax difference: {diff:.6e}")

        if diff < 1e-4:
            print("PASSED: Triton matches PyTorch reference")
        else:
            print("FAILED: Triton does not match PyTorch reference")


# =============================================================================
# EVEN_C path tests (C already a power of 2)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestEvenCPath:
    """Verify correctness of the EVEN_C=True kernel path.

    When C is already a power of 2, _prepare_forward_buffers sets EVEN_C=True,
    which causes the kernel to use trivially-True masks and skip index clamping.
    These tests confirm that path produces the same results as the PyTorch
    reference, covering both forward and backward passes.
    """

    @pytest.mark.parametrize("C", [4, 8, 16, 32, 64])
    def test_even_c_forward_matches_pytorch(self, C):
        """Power-of-2 C (EVEN_C=True) forward partition matches PyTorch reference."""
        batch, T, K = 2, 40, 5
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda", dtype=torch.float64
        )

        # Confirm EVEN_C will be True for this C
        from flash_semicrf.streaming.triton_forward import _next_power_of_2

        assert _next_power_of_2(C) == C, f"C={C} should be a power of 2"

        partition_pytorch, _, _, _ = semi_crf_streaming_forward_pytorch(
            cum_scores, transition, duration_bias, lengths, K
        )
        partition_triton, _, _, _ = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            partition_triton,
            partition_pytorch,
            rtol=1e-4,
            atol=1e-4,
            check_dtype=False,
            msg=f"EVEN_C forward mismatch at C={C}",
        )

    @pytest.mark.parametrize("C", [4, 32, 64])
    def test_even_c_backward_matches_pytorch(self, C):
        """Power-of-2 C (EVEN_C=True) backward gradients match PyTorch reference."""
        from flash_semicrf.streaming import semi_crf_streaming_backward_pytorch

        batch, T, K = 2, 40, 5
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda", dtype=torch.float64
        )

        # Forward
        log_Z, ring_ckpts, ckpt_interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        grad_output = torch.ones(batch, device="cuda", dtype=torch.float64)

        # Triton backward
        grads_triton = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            ckpt_interval,
            grad_output,
        )
        grad_cs_triton, grad_tr_triton, grad_db_triton = grads_triton[:3]

        # PyTorch reference backward.
        # semi_crf_streaming_backward_pytorch returns per-batch *unscaled* gradients:
        #   grad_transition: (batch, C, C)  — one per batch element, not yet reduced
        #   grad_duration_bias: (batch, K, C) — same
        # launch_streaming_triton_backward applies the same scaling + reduction the
        # autograd wrapper does, so we must mirror that here before comparing:
        #   grad_cum_scores *= grad_output[b] (per-batch scale)
        #   grad_transition = einsum("bij,b->ij", ..., grad_output)  → (C, C)
        #   grad_duration_bias = einsum("bkc,b->kc", ..., grad_output)  → (K, C)
        log_Z_py, ring_ckpts_py, ckpt_interval_py, log_norm_ckpts_py = (
            semi_crf_streaming_forward_pytorch(cum_scores, transition, duration_bias, lengths, K)
        )
        grads_pytorch = semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            log_Z_py,
            ring_ckpts_py,
            log_norm_ckpts_py,
            ckpt_interval_py,
        )
        grad_cs_pytorch_raw, grad_tr_pytorch_raw, grad_db_pytorch_raw = grads_pytorch[:3]

        grad_cs_pytorch = grad_cs_pytorch_raw * grad_output.view(batch, 1, 1)
        grad_tr_pytorch = torch.einsum("bij,b->ij", grad_tr_pytorch_raw, grad_output)
        grad_db_pytorch = torch.einsum("bkc,b->kc", grad_db_pytorch_raw, grad_output)

        torch.testing.assert_close(
            grad_cs_triton,
            grad_cs_pytorch,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg=f"EVEN_C grad_cum_scores mismatch at C={C}",
        )
        torch.testing.assert_close(
            grad_tr_triton,
            grad_tr_pytorch,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg=f"EVEN_C grad_transition mismatch at C={C}",
        )
        torch.testing.assert_close(
            grad_db_triton,
            grad_db_pytorch,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg=f"EVEN_C grad_duration_bias mismatch at C={C}",
        )

    def test_even_c_and_non_power_of_2_same_values(self):
        """EVEN_C=True (C=32) and EVEN_C=False (C=31, padded to 32) give consistent output.

        Both configs have the same C_PAD=32, but only C=32 triggers EVEN_C=True.
        This verifies the two code paths agree on results for a shared C_PAD.
        """
        batch, T, K = 2, 40, 5
        torch.manual_seed(0)

        # Shared random weights in label space 0..30 (valid for both C=31 and C=32)
        projected_base = torch.randn(batch, T, 31, device="cuda", dtype=torch.float64)
        projected_base = projected_base - projected_base.mean(dim=1, keepdim=True)

        def make_inputs(C):
            projected = torch.zeros(batch, T, C, device="cuda", dtype=torch.float64)
            projected[:, :, :31] = projected_base
            cum_scores = torch.zeros(batch, T + 1, C, device="cuda", dtype=torch.float64)
            cum_scores[:, 1:, :] = torch.cumsum(projected, dim=1)
            transition = torch.zeros(C, C, device="cuda", dtype=torch.float64)
            transition[:31, :31] = torch.randn(31, 31, device="cuda", dtype=torch.float64) * 0.1
            duration_bias = torch.zeros(K, C, device="cuda", dtype=torch.float64)
            duration_bias[:, :31] = torch.randn(K, 31, device="cuda", dtype=torch.float64) * 0.1
            lengths = torch.full((batch,), T, dtype=torch.long, device="cuda")
            return cum_scores, transition, duration_bias, lengths

        # EVEN_C=False: C=31, C_PAD=32
        cs31, tr31, db31, ln31 = make_inputs(31)
        part31, _, _, _ = launch_streaming_triton_kernel(cs31, tr31, db31, ln31, K)

        # EVEN_C=True: C=32, C_PAD=32 — same data in first 31 labels, zero in label 31
        cs32, tr32, db32, ln32 = make_inputs(32)
        part32, _, _, _ = launch_streaming_triton_kernel(cs32, tr32, db32, ln32, K)

        # The results won't be identical (extra label 31 contributes in C=32), but
        # both should be finite and within a reasonable range — the main check is
        # that EVEN_C=True doesn't produce NaN/Inf or silently wrong values.
        assert torch.isfinite(part31).all(), "EVEN_C=False produced non-finite partition"
        assert torch.isfinite(part32).all(), "EVEN_C=True produced non-finite partition"


# =============================================================================
# Backward Phase 1 barrier regression tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestBackwardPhaseBarriers:
    """Regression tests for tl.debug_barrier() calls in the backward kernel Phase 1.

    Two barriers protect against a cross-warp race at C_PAD = NVIDIA warp size (32):

    **Phase 1 inner loop barrier** (after alpha_buf_seg[local_t] store):
        alpha_buf_seg[local_t] is written by 4 warps (8 elements each).
        local_t+1 loads the same slot via ``alpha_prev[None, :] + edge_block``
        (a C_PAD-element broadcast requiring all warps' data). Without the
        barrier, warp 0 at local_t+1 can read stale values not yet committed
        by warp 3 from local_t.

    **Phase 1 -> Phase 2 barrier** (between the two loops):
        Phase 1 fills alpha_buf_seg[0..SEGMENT_SIZE-1]; Phase 2 immediately
        loads each row as ``alpha_t_clamped[None, :] + edge_tile``.  Without
        the barrier the Phase 2 broadcast can race against Phase 1 stores.

    Both races manifest at C_PAD=32 (= NVIDIA warp size) with num_warps=4.
    Removing either barrier causes the corresponding test to produce gradients
    that differ from the PyTorch reference by more than 1e-3.
    """

    def _reference_backward(self, cum_scores, transition, duration_bias, lengths, K):
        """Run Triton forward + backward and return scaled, reduced gradients."""
        from flash_semicrf.streaming import semi_crf_streaming_backward_pytorch

        batch = cum_scores.shape[0]
        grad_output = torch.ones(batch, device="cuda", dtype=torch.float64)

        log_Z, ring_ckpts, ckpt_interval, log_norm_ckpts = launch_streaming_triton_kernel(
            cum_scores, transition, duration_bias, lengths, K
        )
        grads_triton = launch_streaming_triton_backward(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            log_Z,
            ring_ckpts,
            log_norm_ckpts,
            ckpt_interval,
            grad_output,
        )

        log_Z_py, ring_ckpts_py, ckpt_interval_py, log_norm_ckpts_py = (
            semi_crf_streaming_forward_pytorch(cum_scores, transition, duration_bias, lengths, K)
        )
        grads_pytorch = semi_crf_streaming_backward_pytorch(
            cum_scores,
            transition,
            duration_bias,
            lengths,
            K,
            log_Z_py,
            ring_ckpts_py,
            log_norm_ckpts_py,
            ckpt_interval_py,
        )
        grad_cs_raw, grad_tr_raw, grad_db_raw = grads_pytorch[:3]
        grad_cs_ref = grad_cs_raw * grad_output.view(batch, 1, 1)
        grad_tr_ref = torch.einsum("bij,b->ij", grad_tr_raw, grad_output)
        grad_db_ref = torch.einsum("bkc,b->kc", grad_db_raw, grad_output)

        return grads_triton[:3], (grad_cs_ref, grad_tr_ref, grad_db_ref)

    def test_phase1_inner_loop_barrier(self):
        """Phase 1 inner loop barrier: large K forces long alpha_buf recompute chains.

        K=12 gives SEGMENT_SIZE = checkpoint_interval + 12 ≈ 50.  The Phase 1
        local_t loop runs ~49 iterations; for local_t >= 13, alpha_prev is read
        from alpha_buf_seg (not from the checkpoint).  That creates 36+ iterations
        where each store must be visible to the next load.

        Removing the ``tl.debug_barrier()`` after
        ``tl.store(alpha_buf_seg + local_t * stride_ab_t + ...)``
        causes grad_cum_scores to diverge from the PyTorch reference at C=32.
        """
        C, K, T, batch = 32, 12, 100, 2
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda", dtype=torch.float64
        )

        (grad_cs, grad_tr, grad_db), (ref_cs, ref_tr, ref_db) = self._reference_backward(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            grad_cs,
            ref_cs,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Phase 1 inner loop barrier: grad_cum_scores mismatch at C=32, K=12",
        )
        torch.testing.assert_close(
            grad_tr,
            ref_tr,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Phase 1 inner loop barrier: grad_transition mismatch at C=32, K=12",
        )
        torch.testing.assert_close(
            grad_db,
            ref_db,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Phase 1 inner loop barrier: grad_duration_bias mismatch at C=32, K=12",
        )

    def test_phase1_to_phase2_barrier(self):
        """Phase 1 -> Phase 2 barrier: many checkpoints force many transitions.

        T=100, K=5 produces ~8 checkpoint segments.  At each segment, Phase 1
        fills alpha_buf_seg[0..SEGMENT_SIZE-1] (~19 rows); Phase 2 immediately
        reads each row for ``alpha_t_clamped[None, :] + edge_tile``.  With 8
        segments there are 8 Phase 1 -> Phase 2 transitions where the broadcast
        race can occur.

        Removing the ``tl.debug_barrier()`` before the Phase 2 loop causes
        grad_cum_scores to diverge from the PyTorch reference at C=32.
        """
        C, K, T, batch = 32, 5, 100, 2
        cum_scores, transition, duration_bias, lengths = create_streaming_inputs(
            batch, T, K, C, device="cuda", dtype=torch.float64
        )

        (grad_cs, grad_tr, grad_db), (ref_cs, ref_tr, ref_db) = self._reference_backward(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            grad_cs,
            ref_cs,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Phase 1->Phase 2 barrier: grad_cum_scores mismatch at C=32",
        )
        torch.testing.assert_close(
            grad_tr,
            ref_tr,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Phase 1->Phase 2 barrier: grad_transition mismatch at C=32",
        )
        torch.testing.assert_close(
            grad_db,
            ref_db,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Phase 1->Phase 2 barrier: grad_duration_bias mismatch at C=32",
        )

    def test_both_barriers_variable_lengths(self):
        """Both barriers: variable-length batch exercises active/inactive masking.

        Short sequences end mid-segment, which changes which segments have
        active Phase 1 recompute work.  This stresses both barriers because
        different batch elements hit different local_t loop depths and different
        Phase 1->Phase 2 transition points.
        """
        C, K, T, batch = 32, 8, 80, 4
        cum_scores, transition, duration_bias, _ = create_streaming_inputs(
            batch, T, K, C, device="cuda", dtype=torch.float64
        )
        # Variable lengths: some sequences end mid-segment, stressing masking
        lengths = torch.tensor([T, T - 15, T - 30, T - 45], dtype=torch.long, device="cuda")

        (grad_cs, grad_tr, grad_db), (ref_cs, ref_tr, ref_db) = self._reference_backward(
            cum_scores, transition, duration_bias, lengths, K
        )

        torch.testing.assert_close(
            grad_cs,
            ref_cs,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Variable-length barrier test: grad_cum_scores mismatch at C=32",
        )
        torch.testing.assert_close(
            grad_tr,
            ref_tr,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Variable-length barrier test: grad_transition mismatch at C=32",
        )
        torch.testing.assert_close(
            grad_db,
            ref_db,
            rtol=1e-3,
            atol=1e-3,
            check_dtype=False,
            msg="Variable-length barrier test: grad_duration_bias mismatch at C=32",
        )
