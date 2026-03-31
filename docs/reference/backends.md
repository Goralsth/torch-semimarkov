# Backends and Triton kernel

This project provides GPU-accelerated semi-CRF inference backends using custom Triton kernels.

## Backend summary

| Backend | Time | DP memory | Semirings | Best for |
|---------|------|-----------|-----------|----------|
| `streaming` (recommended) | O(TKC²) | O(KC) | log, max | **Default** - Training and inference |
| `exact` | O(TKC²) | O(TKC²) | All 7 | Full semiring support via `SemiMarkov` class |

## Selecting Backends via SemiMarkovCRFHead

The `SemiMarkovCRFHead` class supports automatic and manual backend selection via the `backend` parameter:

```python
from flash_semicrf import SemiMarkovCRFHead

crf = SemiMarkovCRFHead(num_classes=24, max_duration=100, hidden_dim=512)

# Automatic selection (default) - uses streaming for large T
result = crf.forward(hidden, lengths, backend="auto")

# Force streaming backend (recommended for genome-scale)
result = crf.forward(hidden, lengths, backend="streaming")

# Force exact backend (for semirings beyond log/max)
result = crf.forward(hidden, lengths, backend="exact")
```

The automatic backend selection always uses streaming for log/max semirings (the common case). The exact backend is only used for other semirings (entropy, KL divergence, etc.) or when explicitly requested. On GPU with Triton installed, streaming uses the fused Triton kernel; otherwise it falls back to the PyTorch reference implementation.

## Recommendation

**Default: Streaming API** (`semi_crf_streaming_forward` or `SemiMarkovCRFHead`)

- O(KC) memory via ring buffer (edges computed on-the-fly)
- Hand-written Triton forward and backward kernels
- Supports log and max semirings

**Use the exact backend** (`SemiMarkov.logpartition`) when you have pre-computed edge
tensors and need access to all 7 semirings (entropy, KL divergence, cross-entropy,
counting, K-best — in addition to log and max).

## Streaming Triton Kernel (Recommended)

The streaming API computes edge potentials on-the-fly from O(TxC) cumulative scores,
eliminating the need for the O(TxKxC²) edge tensor.

**Why streaming is faster than pre-computed edges:**
- Memory bandwidth is the bottleneck, not compute
- Computing edges on-the-fly from O(TxC) cumulative scores is faster than loading
  O(TxKxC²) pre-computed edges from memory
- Linear batch scaling: memory grows as O(batchxTxC), not O(batchxTxKxC²)

**Training advantages:**
- Hand-written Triton forward and backward kernels
- No RecursionError from deep computational graphs

### Supported semirings

The streaming API supports **Log** and **Max** semirings:

- `semiring="log"` (default): Log partition function (sum-product)
- `semiring="max"`: Viterbi score (max-product)

For all 7 semirings (including entropy, KL divergence, cross-entropy, counting, K-best),
use `SemiMarkov.logpartition` with pre-computed edge tensors via the `exact` backend.

## Exact Backend (Pre-computed Edge Tensors)

Use `SemiMarkov.logpartition` when you have pre-computed edge tensors and need access
to semirings beyond log/max:

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring, MaxSemiring, EntropySemiring

# Pre-computed edge tensor (batch, T-1, K, C, C) - must fit in memory
crf = SemiMarkov(LogSemiring)
log_Z, _ = crf.logpartition(edge, lengths=lengths)

# Access all 7 semirings
crf_ent = SemiMarkov(EntropySemiring)
entropy, _ = crf_ent.logpartition(edge, lengths=lengths)
```

### Streaming API example

```python
from flash_semicrf.streaming import semi_crf_streaming_forward

# Streaming: compute edges on-the-fly (recommended for both training and inference)
cum_scores = cumsum(projected, dim=1)  # O(TxC) - much smaller!
partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
partition.sum().backward()  # Hand-written Triton backward kernel
```
