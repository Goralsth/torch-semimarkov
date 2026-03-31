"""Tests for SemiMarkovCRFHead nn.Module."""

import pytest
import torch

from flash_semicrf import SemiMarkovCRFHead, semi_crf_streaming_forward
from flash_semicrf.uncertainty import UncertaintySemiMarkovCRFHead


class TestBackendRouting:
    """Tests for T-based backend routing heuristic."""

    def test_should_use_streaming_small_tensor(self):
        """Test heuristic returns False for small edge tensors."""
        # T=1000, K=100, C=24 -> 230MB edge tensor
        crf = SemiMarkovCRFHead(num_classes=24, max_duration=100)
        assert crf._should_use_streaming(1000) is False

    def test_should_use_streaming_large_tensor(self):
        """Test heuristic returns True for large edge tensors."""
        # T=50000, K=100, C=24 -> 11.5GB edge tensor (exceeds 8GB threshold)
        crf = SemiMarkovCRFHead(num_classes=24, max_duration=100)
        assert crf._should_use_streaming(50000) is True

    def test_should_use_streaming_custom_threshold(self):
        """Test heuristic respects custom threshold."""
        # T=5000, K=100, C=24 -> 1.15GB edge tensor
        # With 1GB threshold, should use streaming
        crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, edge_memory_threshold=1e9)
        assert crf._should_use_streaming(5000) is True

        # With 8GB threshold, should use exact
        crf2 = SemiMarkovCRFHead(num_classes=24, max_duration=100, edge_memory_threshold=8e9)
        assert crf2._should_use_streaming(5000) is False

    def test_select_backend_always_streaming_for_log(self):
        """Test auto backend always selects streaming for log semiring."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        # Small T — previously would have returned exact
        backend_type, use_triton = crf._select_backend(T=100, semiring="log", use_triton=True)
        assert backend_type == "streaming"
        assert use_triton is True

        # Large T
        backend_type, use_triton = crf._select_backend(T=100000, semiring="log", use_triton=True)
        assert backend_type == "streaming"
        assert use_triton is True

    def test_select_backend_always_streaming_for_max(self):
        """Test auto backend always selects streaming for max semiring."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        backend_type, use_triton = crf._select_backend(T=100, semiring="max", use_triton=True)
        assert backend_type == "streaming"
        assert use_triton is True

    def test_select_backend_use_triton_false(self):
        """Test streaming with use_triton=False passes through correctly."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        backend_type, use_triton = crf._select_backend(T=100, semiring="log", use_triton=False)
        assert backend_type == "streaming"
        assert use_triton is False

    def test_select_backend_semiring_constraint(self):
        """Test that non-log/max semirings require exact backend."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        backend_type, use_triton = crf._select_backend(T=100, semiring="entropy", use_triton=True)
        assert backend_type == "exact"
        assert use_triton is False

    def test_select_backend_semiring_error_large_t(self):
        """Test error when semiring requires exact but T is too large."""
        # With small threshold to force streaming
        crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, edge_memory_threshold=1e6)
        with pytest.raises(ValueError, match="requires exact backend"):
            crf._select_backend(T=1000, semiring="entropy", use_triton=True)

    def test_forward_backend_auto(self):
        """Test forward with auto backend selection."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        # Auto always selects streaming for log/max semirings
        result = crf(hidden_states, lengths, backend="auto", use_triton=False)
        assert result["partition"].shape == (2,)

    def test_forward_backend_streaming(self):
        """Test forward with forced streaming backend."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        result = crf(hidden_states, lengths, backend="streaming", use_triton=False)
        assert result["partition"].shape == (2,)

    def test_forward_backend_exact(self):
        """Test forward with forced exact backend."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        result = crf(hidden_states, lengths, backend="exact", use_triton=False)
        assert result["partition"].shape == (2,)

    def test_forward_backend_invalid(self):
        """Test forward with invalid backend raises error."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        with pytest.raises(ValueError, match="Unknown backend"):
            crf(hidden_states, lengths, backend="invalid")

    def test_streaming_exact_equivalence(self):
        """Test that streaming and exact backends produce same results."""
        torch.manual_seed(42)
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        result_streaming = crf(hidden_states, lengths, backend="streaming", use_triton=False)
        result_exact = crf(hidden_states, lengths, backend="exact", use_triton=False)

        torch.testing.assert_close(
            result_streaming["partition"],
            result_exact["partition"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_decode_backend_routing(self):
        """Test decode respects backend parameter."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        score_streaming = crf.decode(hidden_states, lengths, backend="streaming", use_triton=False)
        score_exact = crf.decode(hidden_states, lengths, backend="exact", use_triton=False)

        torch.testing.assert_close(score_streaming, score_exact, rtol=1e-4, atol=1e-4)

    def test_compute_loss_backend_routing(self):
        """Test compute_loss respects backend parameter."""
        torch.manual_seed(42)
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss_streaming = crf.compute_loss(
            hidden_states, lengths, labels, backend="streaming", use_triton=False
        )
        loss_exact = crf.compute_loss(
            hidden_states, lengths, labels, backend="exact", use_triton=False
        )

        torch.testing.assert_close(loss_streaming, loss_exact, rtol=1e-4, atol=1e-4)

    def test_forward_auto_uses_streaming_small_t(self, monkeypatch):
        """Behavioral regression: backend='auto' executes streaming for small T.

        Monkeypatches semi_crf_streaming_forward to verify it is actually called
        by forward() and decode() when backend='auto', even for small T.
        """
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        # Track whether the streaming function was called
        call_log = []
        original_fn = semi_crf_streaming_forward

        def tracking_wrapper(*args, **kwargs):
            call_log.append(True)
            return original_fn(*args, **kwargs)

        monkeypatch.setattr("flash_semicrf.nn.semi_crf_streaming_forward", tracking_wrapper)

        # forward() with auto should call streaming
        call_log.clear()
        result = crf(hidden_states, lengths, backend="auto", use_triton=False)
        assert len(call_log) == 1, "forward(backend='auto') did not use streaming path"
        assert result["partition"].shape == (2,)

        # decode() with auto should also call streaming
        call_log.clear()
        crf.decode(hidden_states, lengths, backend="auto", use_triton=False)
        assert len(call_log) == 1, "decode(backend='auto') did not use streaming path"


class TestUncertaintyBackendRouting:
    """Tests for backend routing in UncertaintySemiMarkovCRFHead."""

    def test_should_use_streaming(self):
        """Test heuristic in uncertainty module."""
        crf = UncertaintySemiMarkovCRFHead(num_classes=24, max_duration=100)
        assert crf._should_use_streaming(1000) is False
        assert crf._should_use_streaming(50000) is True

    def test_forward_backend_auto(self):
        """Test uncertainty forward with auto backend."""
        crf = UncertaintySemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        result = crf(hidden_states, lengths, backend="auto", use_triton=False)
        assert result["partition"].shape == (2,)

    def test_compute_loss_backend_routing(self):
        """Test uncertainty compute_loss backend parameter."""
        torch.manual_seed(42)
        crf = UncertaintySemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss_streaming = crf.compute_loss(
            hidden_states, lengths, labels, backend="streaming", use_triton=False
        )
        loss_exact = crf.compute_loss(
            hidden_states, lengths, labels, backend="exact", use_triton=False
        )

        torch.testing.assert_close(loss_streaming, loss_exact, rtol=1e-4, atol=1e-4)

    def test_compute_boundary_marginals_backend(self):
        """Test compute_boundary_marginals backend parameter."""
        torch.manual_seed(42)
        crf = UncertaintySemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        # Both backends should produce valid boundary marginals
        marginals_streaming = crf.compute_boundary_marginals(
            hidden_states, lengths, backend="streaming", normalize=False
        )
        marginals_exact = crf.compute_boundary_marginals(
            hidden_states, lengths, backend="exact", normalize=False
        )

        assert marginals_streaming.shape == (2, 20)
        assert marginals_exact.shape == (2, 20)

        # Streaming and exact should produce near-identical values
        torch.testing.assert_close(marginals_streaming, marginals_exact, rtol=0.01, atol=1e-5)


class TestSemiMarkovCRFHead:
    """Tests for SemiMarkovCRFHead module."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for fast tests."""
        return {
            "num_classes": 4,
            "max_duration": 8,
            "hidden_dim": 16,
            "batch": 2,
            "T": 20,
        }

    @pytest.fixture
    def crf_head(self, small_config):
        """Create a CRF head for testing."""
        return SemiMarkovCRFHead(
            num_classes=small_config["num_classes"],
            max_duration=small_config["max_duration"],
            hidden_dim=small_config["hidden_dim"],
        )

    def test_init_with_hidden_dim(self, small_config):
        """Test initialization with projection layer."""
        crf = SemiMarkovCRFHead(
            num_classes=small_config["num_classes"],
            max_duration=small_config["max_duration"],
            hidden_dim=small_config["hidden_dim"],
        )

        assert crf.projection is not None
        assert crf.projection.in_features == small_config["hidden_dim"]
        assert crf.projection.out_features == small_config["num_classes"]
        assert crf.transition.shape == (
            small_config["num_classes"],
            small_config["num_classes"],
        )
        assert crf.duration_bias.shape == (
            small_config["max_duration"],
            small_config["num_classes"],
        )

    def test_init_without_hidden_dim(self, small_config):
        """Test initialization without projection layer."""
        crf = SemiMarkovCRFHead(
            num_classes=small_config["num_classes"],
            max_duration=small_config["max_duration"],
        )

        assert crf.projection is None

    def test_forward_shape(self, crf_head, small_config):
        """Test forward pass output shapes."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)

        result = crf_head(hidden_states, lengths, use_triton=False)

        assert "partition" in result
        assert "cum_scores" in result
        assert result["partition"].shape == (batch,)
        assert result["cum_scores"].shape == (batch, T + 1, num_classes)

    def test_forward_matches_streaming_api(self, small_config):
        """Test that forward matches direct call to semi_crf_streaming_forward."""
        num_classes = small_config["num_classes"]
        max_duration = small_config["max_duration"]
        batch = small_config["batch"]
        T = small_config["T"]

        # Create CRF without projection (input is already num_classes)
        crf = SemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
        )

        # Input scores (directly in label space)
        scores = torch.randn(batch, T, num_classes)
        lengths = torch.full((batch,), T)

        # Forward via CRFHead
        result = crf(scores, lengths, use_triton=False)

        # Manual computation (must apply same zero-centering as CRFHead)
        scores_centered = scores.double() - scores.double().mean(dim=1, keepdim=True)
        cum_scores = torch.zeros(batch, T + 1, num_classes, dtype=torch.float64)
        cum_scores[:, 1:] = torch.cumsum(scores_centered, dim=1)

        partition_manual = semi_crf_streaming_forward(
            cum_scores,
            crf.transition,
            crf.duration_bias,
            lengths,
            max_duration,
            use_triton=False,
        )

        torch.testing.assert_close(result["partition"], partition_manual)

    def test_compute_loss_shape(self, crf_head, small_config):
        """Test compute_loss output shape."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        # Mean reduction (default)
        loss = crf_head.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert loss.shape == ()  # Scalar

        # Sum reduction
        loss_sum = crf_head.compute_loss(
            hidden_states, lengths, labels, use_triton=False, reduction="sum"
        )
        assert loss_sum.shape == ()

        # No reduction
        loss_none = crf_head.compute_loss(
            hidden_states, lengths, labels, use_triton=False, reduction="none"
        )
        assert loss_none.shape == (batch,)

    def test_compute_loss_positive(self, crf_head, small_config):
        """Test that NLL loss is positive (partition >= gold_score)."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        loss = crf_head.compute_loss(
            hidden_states, lengths, labels, use_triton=False, reduction="none"
        )

        # NLL should be non-negative (partition >= any single path score)
        assert (loss >= -1e-5).all(), f"NLL should be non-negative, got {loss}"

    def test_gradients_flow(self, crf_head, small_config):
        """Test that gradients flow through compute_loss."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T, hidden_dim, requires_grad=True)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        loss = crf_head.compute_loss(hidden_states, lengths, labels, use_triton=False)
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert crf_head.transition.grad is not None
        assert crf_head.duration_bias.grad is not None
        assert crf_head.projection.weight.grad is not None

    def test_score_gold_single_segment(self, small_config):
        """Test gold scoring with single segment (all same label)."""
        num_classes = small_config["num_classes"]
        max_duration = small_config["max_duration"]
        batch = 1
        T = 5

        crf = SemiMarkovCRFHead(num_classes=num_classes, max_duration=max_duration)

        # All zeros label
        labels = torch.zeros(batch, T, dtype=torch.long)
        lengths = torch.tensor([T])

        # Simple scores: all 1.0 for label 0
        scores = torch.zeros(batch, T, num_classes)
        scores[:, :, 0] = 1.0

        cum_scores = torch.zeros(batch, T + 1, num_classes, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        gold_score = crf._score_gold(cum_scores, labels, lengths)

        # Expected: content = T * 1.0 = 5.0, duration_bias[5, 0], no transitions
        # (duration_bias[k] stores bias for segments of duration k)
        expected_content = 5.0
        expected_duration = crf.duration_bias[T, 0].item()  # duration=5 uses index 5
        expected = expected_content + expected_duration

        assert abs(gold_score.item() - expected) < 1e-5

    def test_score_gold_multiple_segments(self, small_config):
        """Test gold scoring with multiple segments."""
        num_classes = 4
        max_duration = 10
        batch = 1
        T = 6

        crf = SemiMarkovCRFHead(num_classes=num_classes, max_duration=max_duration)

        # Labels: [0, 0, 1, 1, 1, 2] -> segments: (0-1, label 0), (2-4, label 1), (5, label 2)
        labels = torch.tensor([[0, 0, 1, 1, 1, 2]])
        lengths = torch.tensor([T])

        # Simple scores
        scores = torch.ones(batch, T, num_classes)
        cum_scores = torch.zeros(batch, T + 1, num_classes, dtype=torch.float32)
        cum_scores[:, 1:] = torch.cumsum(scores.float(), dim=1)

        gold_score = crf._score_gold(cum_scores, labels, lengths)

        # Expected components (duration_bias[k] stores bias for segments of duration k):
        # Segment 1 (0-1, label 0, dur=2): content = 2*1 = 2, dur_bias[2, 0]
        # Segment 2 (2-4, label 1, dur=3): content = 3*1 = 3, dur_bias[3, 1], trans[0, 1]
        # Segment 3 (5-5, label 2, dur=1): content = 1*1 = 1, dur_bias[1, 2], trans[1, 2]
        expected_content = 2 + 3 + 1
        expected_duration = (
            crf.duration_bias[2, 0].item()  # dur=2 uses index 2
            + crf.duration_bias[3, 1].item()  # dur=3 uses index 3
            + crf.duration_bias[1, 2].item()  # dur=1 uses index 1
        )
        expected_transition = crf.transition[0, 1].item() + crf.transition[1, 2].item()
        expected = expected_content + expected_duration + expected_transition

        assert abs(gold_score.item() - expected) < 1e-5

    def test_decode_shape(self, crf_head, small_config):
        """Test decode output shape."""
        batch = small_config["batch"]
        T = small_config["T"]
        hidden_dim = small_config["hidden_dim"]

        hidden_states = torch.randn(batch, T, hidden_dim)
        lengths = torch.full((batch,), T)

        max_score = crf_head.decode(hidden_states, lengths, use_triton=False)

        assert max_score.shape == (batch,)

    def test_extra_repr(self, crf_head, small_config):
        """Test string representation."""
        repr_str = crf_head.extra_repr()
        assert f"num_classes={small_config['num_classes']}" in repr_str
        assert f"max_duration={small_config['max_duration']}" in repr_str
        assert f"hidden_dim={small_config['hidden_dim']}" in repr_str

    def test_variable_lengths(self, crf_head, small_config):
        """Test with variable sequence lengths."""
        batch = 3
        T_max = small_config["T"]
        hidden_dim = small_config["hidden_dim"]
        num_classes = small_config["num_classes"]

        hidden_states = torch.randn(batch, T_max, hidden_dim)
        lengths = torch.tensor([T_max, T_max // 2, T_max // 4])
        labels = torch.randint(0, num_classes, (batch, T_max))

        # Should not error
        result = crf_head(hidden_states, lengths, use_triton=False)
        loss = crf_head.compute_loss(hidden_states, lengths, labels, use_triton=False)

        assert result["partition"].shape == (batch,)
        assert loss.shape == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSemiMarkovCRFHeadGPU:
    """GPU-specific tests for SemiMarkovCRFHead."""

    @pytest.fixture
    def gpu_config(self):
        """GPU configuration."""
        return {
            "num_classes": 8,
            "max_duration": 16,
            "hidden_dim": 32,
            "batch": 4,
            "T": 100,
        }

    def test_forward_gpu(self, gpu_config):
        """Test forward pass on GPU."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"], gpu_config["T"], gpu_config["hidden_dim"]
        ).cuda()
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()

        result = crf(hidden_states, lengths, use_triton=True)

        assert result["partition"].device.type == "cuda"
        assert result["partition"].shape == (gpu_config["batch"],)

    def test_compute_loss_gpu(self, gpu_config):
        """Test compute_loss on GPU with Triton kernels."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"], gpu_config["T"], gpu_config["hidden_dim"]
        ).cuda()
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()
        labels = torch.randint(
            0, gpu_config["num_classes"], (gpu_config["batch"], gpu_config["T"])
        ).cuda()

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=True)

        assert loss.device.type == "cuda"
        assert loss >= 0  # NLL should be non-negative

    def test_gradients_gpu(self, gpu_config):
        """Test gradient computation on GPU."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"],
            gpu_config["T"],
            gpu_config["hidden_dim"],
            requires_grad=True,
            device="cuda",
        )
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()
        labels = torch.randint(
            0, gpu_config["num_classes"], (gpu_config["batch"], gpu_config["T"])
        ).cuda()

        # Retain grad since hidden_states becomes non-leaf after projection in compute_loss
        hidden_states.retain_grad()

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=True)
        loss.backward()

        assert hidden_states.grad is not None
        assert crf.transition.grad is not None
        assert crf.duration_bias.grad is not None

    def test_triton_vs_pytorch(self, gpu_config):
        """Test that Triton and PyTorch implementations match."""
        crf = SemiMarkovCRFHead(
            num_classes=gpu_config["num_classes"],
            max_duration=gpu_config["max_duration"],
            hidden_dim=gpu_config["hidden_dim"],
        ).cuda()

        hidden_states = torch.randn(
            gpu_config["batch"], gpu_config["T"], gpu_config["hidden_dim"]
        ).cuda()
        lengths = torch.full((gpu_config["batch"],), gpu_config["T"]).cuda()

        # Forward pass
        result_triton = crf(hidden_states, lengths, use_triton=True)
        result_pytorch = crf(hidden_states, lengths, use_triton=False)

        torch.testing.assert_close(
            result_triton["partition"],
            result_pytorch["partition"],
            rtol=1e-4,
            atol=1e-4,
        )


class TestBoundaryProjections:
    """Tests for use_boundary_projections=True in SemiMarkovCRFHead."""

    def test_boundary_projections_loss_finite(self):
        """compute_loss with use_boundary_projections=True produces finite loss
        and both boundary layers receive non-zero gradients."""
        torch.manual_seed(42)
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_boundary_projections=True
        )
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

        loss.backward()
        assert crf.proj_start_layer.weight.grad is not None
        assert crf.proj_end_layer.weight.grad is not None
        assert crf.proj_start_layer.weight.grad.abs().sum() > 0
        assert crf.proj_end_layer.weight.grad.abs().sum() > 0

    def test_boundary_projections_hidden_dim_none_raises(self):
        """use_boundary_projections=True with hidden_dim=None raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim"):
            SemiMarkovCRFHead(num_classes=4, max_duration=8, use_boundary_projections=True)

    def test_boundary_projections_zero_weights_equals_no_boundary(self):
        """Boundary model with zeroed weights matches no-boundary model loss."""
        torch.manual_seed(7)
        crf_base = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        crf_boundary = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_boundary_projections=True
        )

        # Copy all shared parameters from base to boundary model
        with torch.no_grad():
            crf_boundary.transition.copy_(crf_base.transition)
            crf_boundary.duration_dist.load_state_dict(crf_base.duration_dist.state_dict())
            crf_boundary.projection.weight.copy_(crf_base.projection.weight)
            crf_boundary.projection.bias.copy_(crf_base.projection.bias)
            # Zero out boundary weights so they contribute nothing
            crf_boundary.proj_start_layer.weight.zero_()
            crf_boundary.proj_end_layer.weight.zero_()

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss_base = crf_base.compute_loss(hidden_states, lengths, labels, use_triton=False)
        loss_boundary = crf_boundary.compute_loss(hidden_states, lengths, labels, use_triton=False)

        torch.testing.assert_close(loss_base, loss_boundary, rtol=1e-5, atol=1e-5)

    def test_boundary_projections_non_streaming_backend_raises(self):
        """Boundary guard raises ValueError for non-streaming backends."""
        torch.manual_seed(0)
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_boundary_projections=True
        )
        hidden_states = torch.randn(2, 10, 16)
        lengths = torch.full((2,), 10)

        # forward() also accepts dp_standard
        for bad_backend in ("exact", "binary_tree_sharded", "dp_standard"):
            with pytest.raises(ValueError, match="[Bb]oundary"):
                crf(hidden_states, lengths, backend=bad_backend, use_triton=False)

        # decode() and decode_with_traceback() don't accept dp_standard
        for bad_backend in ("exact", "binary_tree_sharded"):
            with pytest.raises(ValueError, match="[Bb]oundary"):
                crf.decode(hidden_states, lengths, backend=bad_backend, use_triton=False)
            with pytest.raises(ValueError, match="[Bb]oundary"):
                crf.decode_with_traceback(
                    hidden_states, lengths, backend=bad_backend, use_triton=False
                )

    def test_boundary_projections_decode_with_traceback(self):
        """Sum of segment scores matches Viterbi score from decode_with_traceback."""
        torch.manual_seed(99)
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_boundary_projections=True
        )
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        result = crf.decode_with_traceback(hidden_states, lengths, use_triton=False)

        for b in range(2):
            segments = result.segments[b]
            assert len(segments) > 0
            seg_score_sum = sum(seg.score for seg in segments)
            viterbi_score = result.scores[b].item()
            assert (
                abs(seg_score_sum - viterbi_score) < 1e-4
            ), f"Batch {b}: seg_score_sum={seg_score_sum:.6f} != viterbi_score={viterbi_score:.6f}"

    def test_boundary_grad_with_frozen_geometric_duration(self):
        """With frozen geometric duration, boundary layers still get non-zero gradients."""
        from flash_semicrf.duration import GeometricDuration

        torch.manual_seed(5)
        dur = GeometricDuration(max_duration=8, num_classes=4, learn_rate=False)
        crf = SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=16,
            duration_distribution=dur,
            use_boundary_projections=True,
        )
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert torch.isfinite(loss)
        loss.backward()

        assert crf.proj_start_layer.weight.grad.abs().sum() > 0
        assert crf.proj_end_layer.weight.grad.abs().sum() > 0
        # Duration dist parameters should have no grad (frozen buffers)
        for param in crf.duration_dist.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0

    def test_boundary_grad_finite_difference(self):
        """Finite-difference gradient check on proj_start_layer.weight."""
        torch.manual_seed(13)
        crf = SemiMarkovCRFHead(
            num_classes=3, max_duration=4, hidden_dim=8, use_boundary_projections=True
        )
        # Use double precision for FD check
        crf = crf.double()
        hidden_states = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
        lengths = torch.tensor([6])
        labels = torch.randint(0, 3, (1, 6))

        def loss_fn(hs):
            return crf.compute_loss(hs, lengths, labels, use_triton=False)

        torch.autograd.gradcheck(loss_fn, (hidden_states,), eps=1e-5, atol=1e-3, rtol=1e-3)

    def test_extra_repr_includes_boundary(self):
        """extra_repr includes 'use_boundary_projections=True' when enabled."""
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_boundary_projections=True
        )
        assert "use_boundary_projections=True" in crf.extra_repr()

        crf_no_boundary = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        assert "use_boundary_projections" not in crf_no_boundary.extra_repr()


class TestScoreGoldBoundary:
    """Tests for boundary projection terms in score_gold_vectorized."""

    def test_score_gold_T1_boundary(self):
        """Verify T=1 branch in score_gold_vectorized with boundary projections."""
        from flash_semicrf.helpers import score_gold_vectorized

        torch.manual_seed(0)
        batch, C = 2, 4
        cum_scores = torch.randn(batch, 2, C, dtype=torch.float64)
        labels = torch.tensor([[1], [2]])  # (batch, T=1)
        lengths = torch.tensor([1, 1])
        transition = torch.zeros(C, C, dtype=torch.float64)
        duration_bias = torch.zeros(4, C, dtype=torch.float64)
        proj_start = torch.randn(batch, 1, C, dtype=torch.float64)
        proj_end = torch.randn(batch, 1, C, dtype=torch.float64)

        score_no_boundary = score_gold_vectorized(
            cum_scores, labels, lengths, transition, duration_bias, max_duration=4
        )
        score_with_boundary = score_gold_vectorized(
            cum_scores,
            labels,
            lengths,
            transition,
            duration_bias,
            max_duration=4,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # Expected delta: proj_start[b, 0, label[b]] + proj_end[b, 0, label[b]]
        for b in range(batch):
            lbl = labels[b, 0].item()
            expected_delta = proj_start[b, 0, lbl].item() + proj_end[b, 0, lbl].item()
            actual_delta = (score_with_boundary[b] - score_no_boundary[b]).item()
            assert abs(actual_delta - expected_delta) < 1e-9

    def test_score_gold_K1_boundary(self):
        """Verify K=1 branch in score_gold_vectorized with boundary projections."""
        from flash_semicrf.helpers import score_gold_vectorized

        torch.manual_seed(1)
        batch, T, C = 2, 6, 4
        cum_scores = torch.randn(batch, T + 1, C, dtype=torch.float64)
        labels = torch.randint(0, C, (batch, T))
        lengths = torch.full((batch,), T)
        transition = torch.zeros(C, C, dtype=torch.float64)
        duration_bias = torch.zeros(1, C, dtype=torch.float64)  # K=1
        proj_start = torch.randn(batch, T, C, dtype=torch.float64)
        proj_end = torch.randn(batch, T, C, dtype=torch.float64)

        score_no_boundary = score_gold_vectorized(
            cum_scores, labels, lengths, transition, duration_bias, max_duration=1
        )
        score_with_boundary = score_gold_vectorized(
            cum_scores,
            labels,
            lengths,
            transition,
            duration_bias,
            max_duration=1,
            proj_start=proj_start,
            proj_end=proj_end,
        )

        # For K=1, each position t is its own segment: delta = sum_t proj_start[b,t,lbl] + proj_end[b,t,lbl]
        for b in range(batch):
            expected_delta = 0.0
            for t in range(T):
                lbl = labels[b, t].item()
                expected_delta += proj_start[b, t, lbl].item() + proj_end[b, t, lbl].item()
            actual_delta = (score_with_boundary[b] - score_no_boundary[b]).item()
            assert abs(actual_delta - expected_delta) < 1e-9


class TestParameterPenaltyDurationGating:
    """Tests for parameter_penalty() gating on learnable duration parameters."""

    def test_parameter_penalty_excludes_frozen_duration(self):
        """parameter_penalty excludes duration bias when duration is frozen."""
        from flash_semicrf.duration import GeometricDuration

        dur_frozen = GeometricDuration(max_duration=8, num_classes=4, learn_rate=False)
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, duration_distribution=dur_frozen)

        # With frozen duration, penalty should only include transition
        penalty = crf.parameter_penalty()
        expected = crf.transition.norm(p=2).pow(2)
        torch.testing.assert_close(penalty, expected)

    def test_parameter_penalty_includes_learnable_duration(self):
        """parameter_penalty includes duration bias when duration is learnable."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8)  # default: learned

        penalty = crf.parameter_penalty()
        expected = crf.transition.norm(p=2).pow(2) + crf.duration_bias.norm(p=2).pow(2)
        torch.testing.assert_close(penalty, expected)

    def test_parameter_penalty_includes_learnable_geometric_rate(self):
        """parameter_penalty includes duration bias when geometric has learn_rate=True."""
        from flash_semicrf.duration import GeometricDuration

        dur_learnable = GeometricDuration(max_duration=8, num_classes=4, learn_rate=True)
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, duration_distribution=dur_learnable)

        penalty = crf.parameter_penalty()
        expected = crf.transition.norm(p=2).pow(2) + crf.duration_bias.norm(p=2).pow(2)
        torch.testing.assert_close(penalty, expected)

    def test_parameter_penalty_includes_boundary_projections(self):
        """parameter_penalty always includes boundary projections when enabled."""
        from flash_semicrf.duration import GeometricDuration

        # With both frozen duration and boundary projections
        dur_frozen = GeometricDuration(max_duration=8, num_classes=4, learn_rate=False)
        crf = SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=16,
            duration_distribution=dur_frozen,
            use_boundary_projections=True,
        )

        penalty = crf.parameter_penalty()
        expected = (
            crf.transition.norm(p=2).pow(2)
            + crf.proj_start_layer.weight.norm(p=2).pow(2)
            + crf.proj_end_layer.weight.norm(p=2).pow(2)
        )
        torch.testing.assert_close(penalty, expected)


class TestTracebackSingleRegression:
    """Regression tests for _traceback_single 2D/3D shape fix."""

    def test_traceback_single_exact_backend(self):
        """Regression: _traceback_single must work with 3D cum_scores.

        Prior to the fix, _traceback_single received cum_scores[b] (2D, shape
        (T+1, C)), but compute_edge_block_streaming indexes cum_scores[:, t+k, :]
        requiring a batch dimension. This test exercises the exact-backend path
        (decode_with_traceback accepts 'exact' at the backend dispatch),
        which is the only path that calls _traceback_single directly.
        """
        torch.manual_seed(0)
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        # use_triton=False for determinism/CI portability — backend="exact" doesn't
        # use Triton, but being explicit avoids ambiguity if backend selection is
        # ever refactored and removes dependency on Triton availability in CI.
        result = crf.decode_with_traceback(
            hidden_states, lengths, backend="exact", use_triton=False
        )
        assert len(result.segments) == 2
        for segs in result.segments:
            assert len(segs) > 0


class TestSequenceBoundaries:
    """Tests for use_sequence_boundaries=True in SemiMarkovCRFHead."""

    def test_loss_finite(self):
        """compute_loss with use_sequence_boundaries=True produces finite loss
        and pi_start/pi_end receive non-zero gradients."""
        torch.manual_seed(42)
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_sequence_boundaries=True
        )
        # Set non-zero values so gradients are meaningful
        with torch.no_grad():
            crf.pi_start.fill_(0.1)
            crf.pi_end.fill_(-0.1)

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

        loss.backward()
        assert crf.pi_start.grad is not None
        assert crf.pi_end.grad is not None
        assert crf.pi_start.grad.abs().sum() > 0
        assert crf.pi_end.grad.abs().sum() > 0

    def test_hidden_dim_none_ok(self):
        """use_sequence_boundaries=True works without hidden_dim (standalone params)."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, use_sequence_boundaries=True)
        assert crf.pi_start is not None
        assert crf.pi_end is not None
        assert crf.pi_start.shape == (4,)
        assert crf.pi_end.shape == (4,)

    def test_zero_equals_no_boundary(self):
        """Sequence boundary model with zeroed pi matches no-boundary model loss."""
        torch.manual_seed(7)
        crf_base = SemiMarkovCRFHead(num_classes=4, max_duration=8, hidden_dim=16)
        crf_seq = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_sequence_boundaries=True
        )

        # Copy all shared parameters
        with torch.no_grad():
            crf_seq.transition.copy_(crf_base.transition)
            crf_seq.duration_dist.load_state_dict(crf_base.duration_dist.state_dict())
            crf_seq.projection.weight.copy_(crf_base.projection.weight)
            crf_seq.projection.bias.copy_(crf_base.projection.bias)
            # pi_start and pi_end are already zero-initialized

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss_base = crf_base.compute_loss(hidden_states, lengths, labels, use_triton=False)
        loss_seq = crf_seq.compute_loss(hidden_states, lengths, labels, use_triton=False)

        torch.testing.assert_close(loss_base, loss_seq, rtol=1e-5, atol=1e-5)

    def test_non_streaming_backend_raises(self):
        """Boundary guard raises ValueError for non-streaming backends."""
        torch.manual_seed(0)
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_sequence_boundaries=True
        )
        hidden_states = torch.randn(2, 10, 16)
        lengths = torch.full((2,), 10)

        for bad_backend in ("exact", "binary_tree_sharded", "dp_standard"):
            with pytest.raises(ValueError, match="[Bb]oundary"):
                crf(hidden_states, lengths, backend=bad_backend, use_triton=False)

    def test_decode_with_traceback(self):
        """Sum of segment scores matches Viterbi score from decode_with_traceback."""
        torch.manual_seed(99)
        crf = SemiMarkovCRFHead(
            num_classes=4, max_duration=8, hidden_dim=16, use_sequence_boundaries=True
        )
        with torch.no_grad():
            crf.pi_start.fill_(0.5)
            crf.pi_end.fill_(-0.3)

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)

        result = crf.decode_with_traceback(hidden_states, lengths, use_triton=False)

        for b in range(2):
            segments = result.segments[b]
            assert len(segments) > 0
            seg_score_sum = sum(seg.score for seg in segments)
            viterbi_score = result.scores[b].item()
            assert (
                abs(seg_score_sum - viterbi_score) < 1e-4
            ), f"Batch {b}: seg_score_sum={seg_score_sum:.6f} != viterbi_score={viterbi_score:.6f}"

    def test_extra_repr(self):
        """extra_repr includes 'use_sequence_boundaries=True' when enabled."""
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, use_sequence_boundaries=True)
        assert "use_sequence_boundaries=True" in crf.extra_repr()

        crf_no = SemiMarkovCRFHead(num_classes=4, max_duration=8)
        assert "use_sequence_boundaries" not in crf_no.extra_repr()

    def test_parameter_penalty(self):
        """parameter_penalty includes pi_start and pi_end norms."""
        torch.manual_seed(0)
        crf = SemiMarkovCRFHead(num_classes=4, max_duration=8, use_sequence_boundaries=True)
        with torch.no_grad():
            crf.pi_start.fill_(1.0)
            crf.pi_end.fill_(2.0)

        penalty = crf.parameter_penalty()
        # penalty should include transition + duration + pi_start + pi_end
        expected = (
            crf.transition.norm(p=2).pow(2)
            + crf.duration_bias.norm(p=2).pow(2)
            + crf.pi_start.norm(p=2).pow(2)
            + crf.pi_end.norm(p=2).pow(2)
        )
        torch.testing.assert_close(penalty, expected)

    def test_variable_lengths(self):
        """pi_end is applied at per-sequence final position, not global T-1."""
        torch.manual_seed(42)
        C = 4
        crf = SemiMarkovCRFHead(
            num_classes=C, max_duration=5, hidden_dim=16, use_sequence_boundaries=True
        )
        with torch.no_grad():
            crf.pi_end.fill_(10.0)  # Large value to make effect detectable

        hidden_states = torch.randn(2, 20, 16)
        # Different lengths: sequences end at different positions
        lengths = torch.tensor([15, 20])
        labels = torch.zeros(2, 20, dtype=torch.long)

        # Should not error — pi_end must handle variable lengths
        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_grad_finite_difference(self):
        """Finite-difference gradient check on pi_start and pi_end."""
        torch.manual_seed(13)
        crf = SemiMarkovCRFHead(
            num_classes=3, max_duration=4, hidden_dim=8, use_sequence_boundaries=True
        )
        crf = crf.double()
        hidden_states = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
        lengths = torch.tensor([6])
        labels = torch.randint(0, 3, (1, 6))

        def loss_fn(hs):
            return crf.compute_loss(hs, lengths, labels, use_triton=False)

        torch.autograd.gradcheck(loss_fn, (hidden_states,), eps=1e-5, atol=1e-3, rtol=1e-3)


class TestBothBoundaryModes:
    """Tests for use_boundary_projections=True AND use_sequence_boundaries=True."""

    def test_both_modes_loss_finite(self):
        """Both flags True: finite loss, all boundary params get gradients."""
        torch.manual_seed(42)
        crf = SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=16,
            use_boundary_projections=True,
            use_sequence_boundaries=True,
        )
        with torch.no_grad():
            crf.pi_start.fill_(0.2)
            crf.pi_end.fill_(-0.2)

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss = crf.compute_loss(hidden_states, lengths, labels, use_triton=False)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss}"

        loss.backward()
        # Projection layers get gradients
        assert crf.proj_start_layer.weight.grad.abs().sum() > 0
        assert crf.proj_end_layer.weight.grad.abs().sum() > 0
        # Scalar vectors get gradients
        assert crf.pi_start.grad.abs().sum() > 0
        assert crf.pi_end.grad.abs().sum() > 0

    def test_both_modes_additive(self):
        """Zeroed pi matches projection-only loss (pi adds nothing when zero)."""
        torch.manual_seed(7)
        crf_proj = SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=16,
            use_boundary_projections=True,
        )
        crf_both = SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=16,
            use_boundary_projections=True,
            use_sequence_boundaries=True,
        )

        # Copy all shared parameters
        with torch.no_grad():
            crf_both.transition.copy_(crf_proj.transition)
            crf_both.duration_dist.load_state_dict(crf_proj.duration_dist.state_dict())
            crf_both.projection.weight.copy_(crf_proj.projection.weight)
            crf_both.projection.bias.copy_(crf_proj.projection.bias)
            crf_both.proj_start_layer.weight.copy_(crf_proj.proj_start_layer.weight)
            crf_both.proj_end_layer.weight.copy_(crf_proj.proj_end_layer.weight)
            # pi_start and pi_end already zero

        hidden_states = torch.randn(2, 20, 16)
        lengths = torch.full((2,), 20)
        labels = torch.randint(0, 4, (2, 20))

        loss_proj = crf_proj.compute_loss(hidden_states, lengths, labels, use_triton=False)
        loss_both = crf_both.compute_loss(hidden_states, lengths, labels, use_triton=False)

        torch.testing.assert_close(loss_proj, loss_both, rtol=1e-5, atol=1e-5)

    def test_both_modes_parameter_penalty(self):
        """parameter_penalty includes all boundary terms when both modes active."""
        torch.manual_seed(0)
        crf = SemiMarkovCRFHead(
            num_classes=4,
            max_duration=8,
            hidden_dim=16,
            use_boundary_projections=True,
            use_sequence_boundaries=True,
        )
        with torch.no_grad():
            crf.pi_start.fill_(1.0)
            crf.pi_end.fill_(2.0)

        penalty = crf.parameter_penalty()
        expected = (
            crf.transition.norm(p=2).pow(2)
            + crf.duration_bias.norm(p=2).pow(2)
            + crf.proj_start_layer.weight.norm(p=2).pow(2)
            + crf.proj_end_layer.weight.norm(p=2).pow(2)
            + crf.pi_start.norm(p=2).pow(2)
            + crf.pi_end.norm(p=2).pow(2)
        )
        torch.testing.assert_close(penalty, expected)
