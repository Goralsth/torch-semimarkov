<div align="center">

<img src="https://raw.githubusercontent.com/biobenkj/flash-semicrf/main/flash_semicrf_logo.png" alt="flash-semicrf" width="420" />

Structured Sequence Decoding with Memory-Efficient Semi-CRF Inference

[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20NC-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/biobenkj/flash-semicrf/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/biobenkj/flash-semicrf/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/biobenkj/flash-semicrf/branch/main/graph/badge.svg)](https://codecov.io/gh/biobenkj/flash-semicrf)

[Install](#installation) | [Quick Start](#quick-start) | [Docs](docs/) | [FAQ](docs/faq) | [Examples](examples/) | [GitHub](https://github.com/biobenkj/flash-semicrf)

</div>

## Overview

Semi-Markov CRFs are powerful models for sequences with natural segment structure—speech (phone/word boundaries), biomedical signals (ECG/EEG events), digital pathology (tissue regions), NLP (named entities), time series (regimes), and genomics (genes, exons, chromatin states). However, their inference algorithms are resource-intensive. The segment-level forward pass requires $O(TKC^2)$ time and critically $O(TKC)$ memory, where $T$ is sequence length, $K$ is maximum segment duration, and $C$ is the number of states. For long sequences, this memory footprint quickly exceeds GPU capacity.

Existing implementations navigate this through various tradeoffs—bounding $K$, chunked processing, or filtering heuristics. This package takes a different approach:

Streaming the linear scan reduces DP working memory to $O(KC)$, avoiding the $O(TKC^2)$ edge tensor (see [streaming internals](docs/internals/streaming_internals.md) for details).

This makes Semi-Markov CRF inference practical for long sequences—chromosome-scale genomics, multi-hour clinical recordings, or large document collections—without architectural compromises.

**flash-semicrf** provides:

- **Streaming Semi-CRF inference** with $O(KC)$ memory via ring buffer and on-the-fly edge computation
  - PyTorch reference implementation (CPU/GPU, always available)
  - Triton fused kernel (GPU, 2-5x faster when available)

## Why Semi-CRFs?

Neural sequence models - Transformers, Mamba SSMs, CNNs, LSTMs—produce per-position representations, but many tasks require segment-level predictions with structural constraints. Standard per-position prediction heads have limitations:

- No guarantee of valid segmentations (gaps, overlaps, implausible boundaries)
- Duration constraints require post-hoc heuristics
- No principled uncertainty over segment boundaries

Semi-Markov CRFs bridge this gap as a **structured decoder layer**. The potential function scores an entire segment spanning positions $s$ to $e$:

$$\psi(x_{s:e}, c', c, d) = \underbrace{\psi_{\text{emit}}(x_{s:e}, c)}_{\text{input content}} + \underbrace{\psi_{\text{trans}}(c', c)}_{\text{transition structure}} + \underbrace{\psi_{\text{dur}}(c, d)}_{\text{duration prior}}$$

Each term encodes a distinct constraint: does the input *content* support this segment label? Is this *transition* structurally valid? Is this *duration* plausible for this segment type?

This formulation provides:

- **Valid segmentations by construction** — segments tile the sequence exactly, eliminating post-processing
- **Explicit duration modeling** — encode priors like "named entities rarely exceed 10 tokens" or "exons are typically 50–300 bp"
- **Segment-level posteriors** — principled uncertainty quantification over whole segments, not just positions
- **Transition constraints** — encode structural grammars (e.g., which state transitions are valid)

## Application Domains

Semi-CRFs excel when sequences have inherent segment structure with meaningful durations:

- **Speech & Audio** — phone/word boundaries, speaker diarization, music structure
- **Biomedical Signals** — ECG event detection, EEG sleep staging, activity recognition
- **Digital Pathology** — tissue region segmentation in whole slide images, tumor margin detection
- **Natural Language** — named entity recognition with spans, discourse segmentation, chunking
- **Time Series** — regime detection, anomaly localization, process phase identification
- **Genomics** — gene structure annotation, chromatin states, transposable element detection

Each domain benefits from explicit duration modeling and valid-by-construction segmentations.

## Installation

```bash
# Basic installation
pip install flash-semicrf

# Development installation
git clone https://github.com/biobenkj/flash-semicrf.git
cd flash-semicrf
pip install -e ".[dev]"

# Optional Triton kernel for GPU acceleration
pip install triton
```

## Quick Start

```python
import torch
from flash_semicrf import SemiMarkovCRFHead

# Create Semi-CRF decoder (integrates with Transformer, Mamba, CNN, etc.)
crf = SemiMarkovCRFHead(
    num_classes=24,      # C: number of segment labels
    max_duration=100,    # K: ring buffer size; max segment length is K-1=99
    hidden_dim=512       # matches encoder output
)

# Move to GPU for Triton-accelerated inference (falls back to CPU gracefully)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crf = crf.to(device)

# Encoder output (from Mamba, Transformer, CNN, etc.)
batch, T = 4, 1000
hidden_states = torch.randn(batch, T, 512, device=device)
lengths = torch.full((batch,), T, device=device)

# Training: compute NLL loss
labels = torch.randint(0, 24, (batch, T), device=device)
loss = crf.compute_loss(hidden_states, lengths, labels)
loss.backward()

# Inference: partition function or Viterbi decoding
log_Z = crf(hidden_states, lengths)['partition']
viterbi_score = crf.decode(hidden_states, lengths)
```

Works with PyTorch Lightning and DDP out of the box—see [examples/lightning_integration.py](examples/lightning_integration.py).

For the low-level API with explicit edge tensors and semiring control, see the [API reference](docs/reference/api.md).

## Tensor Conventions

**Edge tensor indexing:** `edge[batch, position, duration, c_dest, c_src]`

This library follows **destination-first** convention for edge tensors, where `edge[..., j, i]` represents the potential for transitioning **from** label `i` **to** label `j`. This differs from some other CRF libraries that use source-first ordering.

**Example:**
```python
# edge[b, t, k, j, i] represents:
#   - Batch item b
#   - Segment starting at position t
#   - Duration k (segment spans positions t to t+k-1)
#   - Transition FROM label i TO label j
```

**Transition matrix:** Similarly, `transition[c_src, c_dest]` stores the score for transitioning from `c_src` to `c_dest`.

**Duration bias indexing:** `duration_bias[k, c]` stores the log-probability bias for segments of duration `k` with label `c`.

- Index 0 is unused (no segments of duration 0)
- Valid durations: 1 to K-1 (where K = `max_duration`)
- Durations ≥ K are clamped to K-1

```python
# A segment spanning positions [t, t+2] (3 positions, duration=3)
# uses duration_bias[3, label]
```

**Special case K=1:** When `max_duration=1`, the model behaves like a standard HMM where all segments have duration 1. In this case, `duration_bias[0]` stores the bias for duration 1 (due to clamping).

### GPU Acceleration (Triton)

When Triton is installed, flash-semicrf uses custom GPU kernels with fused edge computation that significantly accelerate both forward and backward passes.

**How it works:**

The streaming API computes edge potentials on-the-fly from cumulative scores rather than materializing the full `O(T x K x C²)` edge tensor. The Triton kernel fuses this edge computation with the DP scan:

```
# Edge computed on-the-fly (never materialized):
content = cum_scores[t+k, c] - cum_scores[t, c]  # O(1) lookup
edge[t, k, c_dst, c_src] = content + duration_bias[k, c] + transition[c_src, c_dst]
```

Both the PyTorch reference and Triton implementations use the same streaming algorithm: ring buffer for O(KC) DP state, prefix-sum edge decomposition, and checkpoint-based backward pass. The Triton kernel fuses these operations for higher GPU throughput.

Key optimizations:

- **Fused edge computation** — computes edges on-the-fly via prefix-sum, avoiding the full edge tensor
- **O(KC) memory** — ring buffer for DP state, independent of sequence length
- **Separate forward/backward kernels** — the forward-backward algorithm requires computing marginals from both α (forward) and β (backward) messages, so these are necessarily separate kernel launches with checkpointing
- **Checkpointing** — trades compute for memory by recomputing alpha values during backward

**Usage:**

The `SemiMarkovCRFHead` uses Triton automatically when available—pass `use_triton=True` (the default on GPU) to `forward()`, `compute_loss()`, or `decode()`.

For direct access to the streaming kernel:

```python
from flash_semicrf.streaming import semi_crf_streaming_forward

# Cumulative scores from encoder (see docs for zero-centering requirements)
partition = semi_crf_streaming_forward(
    cum_scores, transition, duration_bias, lengths, K
)
```

For performance characteristics, see [Benchmarking](docs/reference/benchmarks.md).

## Documentation

- [Integration guide](docs/guides/workflow_integration.md) — how to use flash-semicrf with BERT, Mamba, CNNs, and other encoders
- [Parameter guide: T, K, C](docs/guides/parameter_guide.md) — understanding sequence length, duration, and state dimensions
- [Semirings guide](docs/guides/semirings.md) — context and intuition for semirings used in flash-semicrf
- [Uncertainty and focused learning](docs/guides/uncertainty_and_focused_learning.md) — boundary confidence, active learning, and clinical applications
- [Backends and Triton kernel](docs/reference/backends.md) — algorithm selection and GPU acceleration
- [API reference](docs/reference/api.md) — detailed API documentation
- [Benchmarking](docs/reference/benchmarks.md) — performance measurement
- [FAQ](docs/faq) — background and frequently asked questions
- [AI disclosure](docs/disclosure.md)

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=flash_semicrf --cov-report=term-missing
```

Tests run CPU-only by default. GPU tests require CUDA and are skipped in CI.

## Implementation Status

| Component | Status |
|-----------|--------|
| **Streaming Algorithm** | O(KC) memory, ring buffer + prefix-sum edges |
| **PyTorch Backend** | Reference implementation, CPU/GPU |
| **Triton Backend** | Fused GPU kernel, Log/Max semirings, 2-5x faster |
| **Semirings** | Log, Max, Std, KMax, Entropy, CrossEntropy, KLDivergence |

## Acknowledgments

This library builds on [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) by Alexander Rush. GPU kernels are written using [Triton](https://github.com/triton-lang/triton).

## License

PolyForm Noncommercial License - see [LICENSE](LICENSE.md) for details.
