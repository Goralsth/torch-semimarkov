# API reference

For practical usage examples showing how to integrate these APIs with upstream encoders, see the [Integration guide](../guides/workflow_integration.md).

## SemiMarkovCRFHead (Recommended)

The high-level module for most use cases. Uses the streaming Triton kernel internally,
which is the recommended approach for both training and inference:

```python
from flash_semicrf import SemiMarkovCRFHead

class SemiMarkovCRFHead(nn.Module):
    def __init__(
        self,
        num_classes: int,              # Number of label classes (C)
        max_duration: int,             # Maximum segment duration (K)
        hidden_dim: int = None,        # Optional: projection from encoder dim
        init_scale: float = 0.1,       # Parameter initialization scale
        duration_distribution: str = None,  # "learned", "geometric", "poisson", etc.
        edge_memory_threshold: float = 8e9,  # Memory threshold for non-log/max semirings (8GB)
        num_warps: int = 4,            # Triton kernel parallelism (2-8 recommended)
    ):
        """
        CRF head for Semi-Markov sequence labeling.

        Compatible with DDP - gradients sync automatically via standard PyTorch.
        Memory: O(KC) independent of sequence length T.

        Transition Matrix Convention:
            transition[i, j] = score for transitioning FROM label i TO label j.

        Note:
            For T > 100K, use float32 precision for numerical stability.
        """

    def forward(
        self,
        hidden_states,  # (batch, T, hidden_dim) or (batch, T, C)
        lengths,        # (batch,) sequence lengths
        use_triton=True,
        backend="auto",  # "auto", "streaming", "exact", or "binary_tree_sharded"
    ) -> dict:
        """
        Compute partition function.

        Args:
            backend: Backend selection mode:
                - "auto": Streaming for log/max semirings, exact for others (default)
                - "streaming": Force streaming backend (genome-scale)
                - "exact": Force exact backend via semimarkov.py
                - "binary_tree_sharded": Memory-efficient reference with checkpointing

        Returns:
            dict with 'partition' (batch,) and 'cum_scores' (batch, T+1, C)
        """

    def compute_loss(
        self,
        hidden_states,
        lengths,
        labels,         # (batch, T) per-position labels
        use_triton=True,
        backend="auto",  # "auto", "streaming", "exact", or "binary_tree_sharded"
        reduction="mean",
    ) -> Tensor:
        """Compute negative log-likelihood loss."""

    def decode(
        self,
        hidden_states,
        lengths,
        use_triton=True,
        backend="auto",  # "auto", "streaming", "exact", or "binary_tree_sharded"
    ) -> Tensor:
        """Viterbi decoding - returns best score (batch,)."""

    def decode_with_traceback(
        self,
        hidden_states,
        lengths,
        max_traceback_length=10000,
        use_triton=True,
    ) -> ViterbiResult:
        """Viterbi with path reconstruction. Returns (scores, segments)."""

    def parameter_penalty(self, p: float = 2.0) -> Tensor:
        """
        Compute Lp penalty on CRF parameters for regularization.

        Returns: ||transition||_p^p + ||duration_bias||_p^p
        """
```

**Example usage:**

```python
from flash_semicrf import SemiMarkovCRFHead

# Create CRF head
crf = SemiMarkovCRFHead(
    num_classes=24,
    max_duration=100,
    hidden_dim=512,
    duration_distribution="geometric",  # or "learned", "poisson", etc.
)

# Encoder output
hidden = encoder(x)  # (batch, T, 512)
lengths = torch.full((batch,), T)

# Forward pass
result = crf(hidden, lengths)
partition = result['partition']  # (batch,)

# Training with labels
labels = torch.randint(0, 24, (batch, T))
loss = crf.compute_loss(hidden, lengths, labels)
loss.backward()

# Viterbi decoding with traceback
result = crf.decode_with_traceback(hidden, lengths)
for seg in result.segments[0]:
    print(f"[{seg.start}, {seg.end}] label={seg.label}")

# Regularization
reg_loss = loss + 0.01 * crf.parameter_penalty()
```

## Code Execution Flow

Understanding how data flows through the CRF head helps with debugging and optimization.

### Training Flow (compute_loss)

```text
hidden_states (batch, T, hidden_dim)
       │
       ▼ [projection layer, if hidden_dim provided]
   scores (batch, T, C)
       │
       ▼ [zero-center for numerical stability]
   scores_centered = scores - scores.mean(dim=1)
       │
       ▼ [cumulative sum in float32]
   cum_scores (batch, T+1, C)
       │
       ▼ [backend selection: streaming for log/max, exact for others]
       │
       ├── streaming (log/max semirings — always default)
       │        │
       │        ▼
       │   semi_crf_streaming_forward()
       │        │
       │        ▼ [Triton kernel on CUDA, PyTorch reference on CPU]
       │   partition (batch,)
       │
       └── exact (non-log/max semirings or explicitly requested)
                │
                ▼
           _build_edge_tensor() → edge (batch, T, K, C, C)
                │
                ▼
           SemiMarkov.logpartition()
                │
                ▼
           partition (batch,)
       │
       ▼ [score gold segmentation via label changes]
   gold_score = score_gold_vectorized(cum_scores, labels, ...)
       │
       ▼
   NLL = partition - gold_score
```

### Inference Flow (decode_with_traceback)

```text
hidden_states (batch, T, hidden_dim)
       │
       ▼ [same preprocessing as training]
   cum_scores (batch, T+1, C)
       │
       ▼ [streaming Viterbi with backpointer storage]
   semi_crf_streaming_viterbi_with_backpointers()
       │
       ├── max_scores (batch,)
       ├── bp_k (batch, T, C)  ← best duration at each (position, label)
       ├── bp_c (batch, T, C)  ← best source label at each (position, label)
       └── final_labels (batch,)
       │
       ▼ [O(T) traceback using backpointers]
   _traceback_from_backpointers()
       │
       ▼
   ViterbiResult(scores, List[List[Segment]])
```

### Key Files

| Component | File |
| --------- | ---- |
| CRF head module | [nn.py](../src/flash_semicrf/nn.py) |
| Streaming kernels | [streaming/autograd.py](../src/flash_semicrf/streaming/autograd.py) |
| Triton forward | [streaming/triton_forward.py](../src/flash_semicrf/streaming/triton_forward.py) |
| Triton backward | [streaming/triton_backward.py](../src/flash_semicrf/streaming/triton_backward.py) |
| Gold scoring | [helpers.py](../src/flash_semicrf/helpers.py) |

## UncertaintySemiMarkovCRFHead

Extended CRF head with uncertainty quantification:

```python
from flash_semicrf import UncertaintySemiMarkovCRFHead

class UncertaintySemiMarkovCRFHead(SemiMarkovCRFHead):
    """
    SemiMarkovCRFHead with uncertainty methods.

    Additional methods for boundary confidence and active learning.
    Inherits all parameters from SemiMarkovCRFHead including accum_dtype and num_warps.
    """

    def compute_boundary_marginals(
        self,
        hidden_states,
        lengths,
        backend="auto",  # "auto", "streaming", or "exact"
        normalize=True,
    ) -> Tensor:
        """
        P(boundary at position t) for each position.

        Args:
            backend: Backend selection mode:
                - "auto": Streaming for log/max semirings, exact for others (default)
                - "streaming": Force streaming forward-backward algorithm
                - "exact": Force exact marginals via edge tensor

        Returns: (batch, T) boundary probabilities

        Note: use_streaming parameter is deprecated, use backend instead.
        """

    def compute_position_marginals(
        self,
        hidden_states,
        lengths,
    ) -> Tensor:
        """
        P(label=c at position t) for each position.

        Returns: (batch, T, C) label probabilities
        """

    def compute_entropy_streaming(
        self,
        hidden_states,
        lengths,
    ) -> Tensor:
        """
        Approximate entropy from marginals (works for T > 10K).

        Returns: (batch,) entropy estimates
        """

    def compute_entropy_exact(
        self,
        hidden_states,
        lengths,
    ) -> Tensor:
        """
        Exact entropy via EntropySemiring (T < 10K only).

        Computes H(P) = -sum_y P(y) log P(y) using the entropy semiring.
        Requires building the full edge tensor, so only works for short sequences.

        Returns: (batch,) exact entropy values
        """

    def compute_loss_uncertainty_weighted(
        self,
        hidden_states,
        lengths,
        labels,
        uncertainty_weight=1.0,
        focus_mode="high_uncertainty",  # or "boundary_regions"
        use_triton=False,
        reduction="mean",
    ) -> Tensor:
        """
        Uncertainty-weighted loss for active learning.

        L_weighted = (1 + lambda * uncertainty) * NLL
        """
```

**Example usage:**

```python
from flash_semicrf import UncertaintySemiMarkovCRFHead

model = UncertaintySemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)

# Boundary confidence for decision support (auto-selects streaming for large T)
boundary_probs = model.compute_boundary_marginals(hidden, lengths)

# Force streaming backend for genome-scale sequences
boundary_probs = model.compute_boundary_marginals(hidden, lengths, backend="streaming")

# Entropy computation
entropy_approx = model.compute_entropy_streaming(hidden, lengths)  # T > 10K
entropy_exact = model.compute_entropy_exact(hidden, lengths)       # T < 10K

# Uncertainty-weighted training for active learning
loss = model.compute_loss_uncertainty_weighted(hidden, lengths, labels)
```

## Streaming API (Recommended for Training)

The streaming API is the **recommended approach for training** on GPU. It uses Triton
kernels that compute edges on-the-fly, providing both memory efficiency and reliable
gradient computation.

For very long sequences (T = 100K - 400K+) where edge tensor cannot fit in memory:

```python
from flash_semicrf import semi_crf_streaming_forward

def semi_crf_streaming_forward(
    cum_scores,       # (batch, T+1, C) cumulative projected scores
    transition,       # (C, C) or (K, C, C) transition matrix
    duration_bias,    # (K, C) duration bias
    lengths,          # (batch,) sequence lengths
    K,                # max segment duration
    semiring="log",   # "log" (partition) or "max" (Viterbi)
    use_triton=True,  # Use Triton kernel if available
    num_warps=4,      # Triton kernel parallelism (2-8 recommended)
) -> Tensor:
    """
    Memory-efficient Semi-CRF forward with on-the-fly edge computation.

    Memory: O(T*C) for cum_scores vs O(T*K*C^2) for edge tensor
    - T=400K, K=3K, C=24: 38 MB vs 2.76 TB

    Args:
        num_warps: Number of warps per block for Triton kernels.
            Higher values increase parallelism but also register pressure.
            Recommended range: 2-8. Default: 4

    Returns:
        partition: (batch,) log partition function or Viterbi score
    """
```

**Example usage:**

```python
from flash_semicrf import semi_crf_streaming_forward

# Pre-project features (outside kernel)
projected = hidden @ W_content  # (batch, T, C)
projected = projected - projected.mean(dim=1, keepdim=True)  # Zero-center!

# Build cumulative scores
cum_scores = torch.zeros(batch, T+1, C, dtype=torch.float32)
cum_scores[:, 1:] = torch.cumsum(projected.float(), dim=1)

# Streaming forward (edges computed on-the-fly)
partition = semi_crf_streaming_forward(
    cum_scores, transition, duration_bias, lengths, max_duration
)
```

## Streaming Internals (Advanced)

Low-level components of the streaming API for advanced use cases.

### compute_edge_block_streaming

Compute edge potentials for a single (position, duration) pair on-the-fly:

```python
from flash_semicrf.streaming import compute_edge_block_streaming

def compute_edge_block_streaming(
    cum_scores,      # (batch, T+1, C)
    transition,      # (C, C)
    duration_bias,   # (K, C)
    start: int,      # Segment start position
    k: int,          # Segment duration
) -> Tensor:
    """
    Compute edge block for segment [start, start+k) on-the-fly.

    Returns: (batch, C_dest, C_src) edge potentials
    """
```

### Autograd Functions

PyTorch autograd functions for streaming Semi-CRF with custom backward passes:

```python
from flash_semicrf.streaming import SemiCRFStreaming, SemiCRFStreamingTriton

# Pure PyTorch (CPU or GPU without Triton)
class SemiCRFStreaming(torch.autograd.Function):
    """
    O(KC) memory streaming forward-backward.

    Uses gradient checkpointing to recompute alpha values during backward,
    trading compute for memory.
    """

# Triton-accelerated (GPU with Triton)
class SemiCRFStreamingTriton(torch.autograd.Function):
    """
    O(KC) memory with hand-written Triton backward kernels.

    Hand-written Triton forward and backward kernels.
    """
```

### Triton Launchers (Conditionally Available)

When Triton is installed, these low-level kernel launchers are available:

```python
from flash_semicrf.streaming import HAS_TRITON

if HAS_TRITON:
    from flash_semicrf.streaming import (
        launch_streaming_triton_kernel,      # Forward pass
        launch_streaming_triton_backward,    # Backward pass
        launch_streaming_triton_kernel_max_bp,  # Viterbi with backpointers
        semi_crf_streaming_viterbi_triton,   # Full Viterbi decoding
    )
```

### Constants

```python
from flash_semicrf.streaming import NEG_INF

NEG_INF  # Large negative value for log-space operations (-1e38)
```

## Duration Distributions

Flexible parameterization for segment duration priors:

```python
from flash_semicrf import (
    create_duration_distribution,
    LearnedDuration,
    GeometricDuration,
    NegativeBinomialDuration,
    PoissonDuration,
    UniformDuration,
    CallableDuration,
)

# Factory function
dur = create_duration_distribution(
    "geometric",      # or "learned", "negative_binomial", "poisson", "uniform"
    max_duration=100,
    num_classes=24,
    init_logit=-1.0,  # Distribution-specific kwargs
)
bias = dur()  # Returns (K, C) tensor
```

**Available distributions:**

| Distribution | Formula | Use case |
|--------------|---------|----------|
| `LearnedDuration` | Fully learned | Most flexible, default |
| `GeometricDuration` | P(k) ~ p(1-p)^(k-1) | Exponential decay, numerically stable |
| `NegativeBinomialDuration` | Generalizes geometric | Peaked distributions |
| `PoissonDuration` | P(k) ~ lambda^k/k! | Characteristic segment length |
| `UniformDuration` | P(k) = const | No duration preference |
| `CallableDuration` | User-defined | Full customization |

**Numerical stability note:**

`NegativeBinomialDuration` with very small `r` values (init_log_r < -10) can cause
numerical instability. A runtime warning is emitted when non-finite values are detected.
Use `GeometricDuration` as a stable alternative.

## SemiMarkov (Low-level API)

Low-level API with semiring abstraction for advanced use cases:

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring, MaxSemiring

class SemiMarkov(semiring):
    def logpartition(
        self,
        log_potentials,  # (batch, N-1, K, C, C) edge potentials
        lengths=None,    # (batch,) sequence lengths
        force_grad=False,
        use_linear_scan=None,  # None=auto, True=O(N) scan, False=O(log N) tree
        use_vectorized=False,  # If True, O(TKC) memory but 2-3x faster
        use_banded=False,      # Prototype: banded matrix optimization
    ) -> Tuple[Tensor, List[Tensor], None]:
        """
        Compute log partition function.

        Algorithm selection (use_linear_scan):
            - None (default): Auto-select based on KC > 200
            - True: O(N) linear scan with O(KC) ring buffer memory
            - False: O(log N) binary tree (WARNING: O((KC)³) memory per matmul)

        Memory modes:
            - use_vectorized=False (default): O(KC) streaming ring buffer
            - use_vectorized=True: O(TKC) but 2-3x faster

        Returns:
            v: (ssize, batch,) log partition values
            edges: list containing input potentials for gradient computation
            charts: None (streaming scan does not store charts)
        """

    def marginals(
        self,
        log_potentials,
        lengths=None,
    ) -> Tensor:
        """
        Compute edge marginals via backward pass.

        Returns: (batch, N-1, K, C, C) marginal probabilities
        """

    @staticmethod
    def hsmm(init, trans_z, trans_l, emission) -> Tensor:
        """
        Convert HSMM parameters to Semi-Markov edge potentials.

        Args:
            init: (C,) initial state distribution
            trans_z: (C, C) state transition matrix
            trans_l: (C, K) duration distribution per state
            emission: (batch, N, K, C) emission scores

        Returns:
            edge: (batch, N, K, C, C) edge potentials
        """
```

## Semirings

```python
from flash_semicrf.semirings import (
    LogSemiring,           # Standard log-space (sum-product)
    MaxSemiring,           # Viterbi decoding (max-product)
    StdSemiring,           # Standard arithmetic
    KMaxSemiring,          # Top-k paths
    EntropySemiring,       # Entropy computation
    KLDivergenceSemiring,  # KL divergence D_KL(P || Q)
    CrossEntropySemiring,  # Cross-entropy H(P, Q)
)

from flash_semicrf.semirings.checkpoint import (
    CheckpointSemiring,       # Gradient checkpointing
    CheckpointShardSemiring,  # Sharded checkpointing
)
```

## Pre-computed Edge Tensor API

For pre-computed edge tensors, use `SemiMarkov.logpartition` directly. This unlocks
all 7 semirings (not just log/max as in the streaming API).

> **Note:** For most use cases, the streaming API (`SemiMarkovCRFHead` or
> `semi_crf_streaming_forward`) is recommended — it computes edges on-the-fly
> with O(KC) memory and supports both training and inference.

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring, MaxSemiring, EntropySemiring

# Pre-computed edge tensor (batch, T-1, K, C, C) - must fit in memory
crf = SemiMarkov(LogSemiring)
log_Z, _ = crf.logpartition(edge, lengths=lengths)

# Viterbi score (max semiring)
crf_max = SemiMarkov(MaxSemiring)
best_score, _ = crf_max.logpartition(edge, lengths=lengths)

# Entropy (not available via streaming API)
crf_ent = SemiMarkov(EntropySemiring)
entropy, _ = crf_ent.logpartition(edge, lengths=lengths)
```

## Helper Types

```python
from flash_semicrf import Segment, ViterbiResult

@dataclass
class Segment:
    """A single segment from Viterbi decoding."""
    start: int   # Start position (inclusive)
    end: int     # End position (inclusive)
    label: int   # Label class
    score: float # Segment score contribution

class ViterbiResult(NamedTuple):
    """Result from decode_with_traceback."""
    scores: Tensor              # (batch,) best scores
    segments: List[List[Segment]]  # Per-batch segment lists
```

## Sparse Matrix Backends (Experimental)

These classes provide memory-efficient sparse representations for Semi-Markov
structures. They are primarily used for benchmarking and experimentation.

### BandedMatrix

Lightweight banded matrix representation for CPU/PyTorch operations:

```python
from flash_semicrf import BandedMatrix

@dataclass
class BandedMatrix:
    """
    Banded matrix representation for memory-efficient sparse operations.

    Stores only the non-zero diagonals of a banded matrix in a compact format.
    Supports log-semiring and max-semiring matrix multiplication.
    """
    data: Tensor   # (batch, n, lu+ld+1)
    lu: int        # Number of upper diagonals
    ld: int        # Number of lower diagonals
    fill: float = 0.0

    @classmethod
    def from_dense(cls, dense, lu, ld, fill=0.0) -> BandedMatrix:
        """Extract banded view from dense square matrix."""

    def to_dense(self) -> Tensor:
        """Expand back to dense matrix."""

    def transpose(self) -> BandedMatrix:
        """Transpose the banded matrix."""

    def multiply_log(self, other) -> BandedMatrix:
        """Log-semiring matrix multiplication."""

    def multiply_max(self, other) -> BandedMatrix:
        """Max-semiring matrix multiplication."""
```

**Example:**

```python
from flash_semicrf import BandedMatrix

# Create banded matrix from dense
dense = torch.randn(2, 10, 10)
banded = BandedMatrix.from_dense(dense, lu=2, ld=2)
print(banded.data.shape)  # (2, 10, 5)

# Convert back to dense
reconstructed = banded.to_dense()
```

### BlockTriangularMatrix

Block-triangular sparse representation exploiting duration constraint `k1 + k2 <= span`:

```python
from flash_semicrf import BlockTriangularMatrix, block_triang_matmul

@dataclass
class BlockTriangularMatrix:
    """
    Block-triangular matrix over duration states.

    Stores only blocks (k1, k2) satisfying the duration constraint,
    reducing memory from O(K²C²) to O(K(K+1)/2 * C²).
    """
    values: Tensor        # (batch, num_blocks, C, C)
    block_indices: Tensor # (num_blocks, 2) - (k1, k2) coordinates
    K: int
    C: int

    @classmethod
    def from_dense(cls, dense, K, C, span, duration_mask=None) -> BlockTriangularMatrix:
        """Compress dense to block-triangular representation."""

    def to_dense(self, semiring=None) -> Tensor:
        """Expand back to dense matrix."""

def block_triang_matmul(left, right, semiring, span) -> BlockTriangularMatrix:
    """Sparse semiring matrix multiplication."""
```

**Example:**

```python
from flash_semicrf import BlockTriangularMatrix

dense = torch.randn(2, 12, 12)  # K=4, C=3
bt = BlockTriangularMatrix.from_dense(dense, K=4, C=3, span=4)
print(bt.values.shape)  # (2, 10, 3, 3) - only 10 blocks satisfy k1+k2 <= 4
```

## Banded Utilities

Utilities for analyzing and optimizing banded matrix structures:

```python
from flash_semicrf import (
    measure_effective_bandwidth,
    snake_ordering,
    rcm_ordering_from_adjacency,
    apply_permutation,
)

def measure_effective_bandwidth(adj, fill_value=None) -> int:
    """
    Compute maximum distance from diagonal of any non-fill entry.

    Args:
        adj: Adjacency matrix (n, n) or (batch, n, n), or BandedMatrix
        fill_value: Value representing empty/non-edges (auto-detected if None)

    Returns:
        Maximum distance from diagonal
    """

def snake_ordering(K, C) -> Tensor:
    """
    Generate snake ordering permutation for (K, C) state space.

    Interleaves low and high duration states to reduce bandwidth:
    [0, K-1, 1, K-2, 2, K-3, ...] for each label.

    Returns: Permutation tensor of shape (K*C,)
    """

def rcm_ordering_from_adjacency(adj) -> tuple[Tensor, bool]:
    """
    Compute Reverse Cuthill-McKee ordering (requires SciPy).

    Minimizes bandwidth of sparse matrices.

    Returns: (permutation, success) - success=False if SciPy unavailable
    """

def apply_permutation(potentials, perm) -> Tensor:
    """Apply permutation to both dimensions of a matrix."""
```

**Example:**

```python
from flash_semicrf import measure_effective_bandwidth, snake_ordering

adj = torch.eye(5)
print(measure_effective_bandwidth(adj))  # 0

adj[0, 4] = 1  # Add off-diagonal entry
print(measure_effective_bandwidth(adj))  # 4

# Generate snake ordering for K=10, C=3
perm = snake_ordering(10, 3)
```

## Memory Comparison

| Scenario | Edge tensor size | Streaming API size |
|----------|------------------|-------------------|
| T=1K, K=32, C=24 | 18 MB | 96 KB |
| T=10K, K=100, C=24 | 5.5 GB | 960 KB |
| T=400K, K=3K, C=24 | **2.76 TB** | 38 MB |

**Recommended usage:**

| Use case | API | Why |
|----------|-----|-----|
| Training (any T) | `SemiMarkovCRFHead` or `semi_crf_streaming_forward` | Hand-written Triton forward and backward kernels |
| Inference (recommended) | `SemiMarkovCRFHead` or `semi_crf_streaming_forward` | O(KC) memory, faster than pre-computed edges |
| Pre-computed edges / all semirings | `SemiMarkov.logpartition` | All 7 semirings, edge tensor must fit in memory |
| Very long sequences (T > 10K) | Streaming API | Edge tensor cannot fit in memory |

## See Also

- [Backends and Triton kernel](backends.md) - Detailed kernel behavior and backend selection
- [Integration guide](../guides/workflow_integration.md) - End-to-end training examples with BERT, Mamba, CNNs
- [Streaming internals](../internals/streaming_internals.md) - Low-level algorithm details and numerical stability
- [Uncertainty and focused learning](../guides/uncertainty_and_focused_learning.md) - Boundary confidence and active learning
- [Parameter guide: T, K, C](../guides/parameter_guide.md) - Understanding sequence length, duration, and state dimensions
- [Semirings guide](../guides/semirings.md) - Mathematical context for semiring operations
