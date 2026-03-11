"""
flash-semicrf: Efficient Semi-Markov CRF Inference for PyTorch

This package provides optimized implementations of Semi-Markov CRF inference
algorithms for genomic sequence annotation at chromosome scale (T=400K+).

Key Features:
- O(KC) memory streaming scan (independent of sequence length T)
- Streaming API for on-the-fly edge computation via prefix-sum decomposition
- Triton GPU acceleration for inference
- Multiple backend algorithms for benchmarking (banded, block-triangular)

Original pytorch-struct: https://github.com/harvardnlp/pytorch-struct
License: MIT (see LICENSE file)

Modifications:
- Added streaming linear scan with O(KC) memory
- Added streaming API for T=400K+ sequences with on-the-fly edge computation
- Added banded and block-triangular backends for benchmarking
- Optimized for genomic segmentation use cases
"""

# Lightning integration (optional — requires lightning>=2.0 or pytorch-lightning>=2.0)
import importlib.util as _ilu

from .banded import BandedMatrix
from .banded_utils import (
    apply_permutation,
    measure_effective_bandwidth,
    rcm_ordering_from_adjacency,
    snake_ordering,
)
from .blocktriangular import BlockTriangularMatrix, block_triang_matmul
from .duration import (
    CallableDuration,
    DurationDistribution,
    GeometricDuration,
    LearnedDuration,
    NegativeBinomialDuration,
    PoissonDuration,
    UniformDuration,
    create_duration_distribution,
)
from .nn import Segment, SemiMarkovCRFHead, ViterbiResult
from .semimarkov import SemiMarkov
from .streaming import (
    SemiCRFStreaming,
    compute_edge_block_streaming,
    semi_crf_streaming_forward,
)
from .uncertainty import UncertaintyMixin, UncertaintySemiMarkovCRFHead

if _ilu.find_spec("lightning") or _ilu.find_spec("pytorch_lightning"):
    from .lightning import SemiCRFLightningModule, pad_and_collate  # noqa: F401

    _LIGHTNING_EXPORTS = ["SemiCRFLightningModule", "pad_and_collate"]
else:
    _LIGHTNING_EXPORTS = []
del _ilu

__version__ = "0.2.0"

__all__ = [
    # Core API
    "SemiMarkov",
    # Streaming API
    "semi_crf_streaming_forward",
    "SemiCRFStreaming",
    "compute_edge_block_streaming",
    # Neural network modules
    "SemiMarkovCRFHead",
    "Segment",
    "ViterbiResult",
    # Uncertainty quantification
    "UncertaintyMixin",
    "UncertaintySemiMarkovCRFHead",
    # Duration distributions
    "DurationDistribution",
    "LearnedDuration",
    "GeometricDuration",
    "NegativeBinomialDuration",
    "PoissonDuration",
    "UniformDuration",
    "CallableDuration",
    "create_duration_distribution",
    # Lightning integration (optional)
    *_LIGHTNING_EXPORTS,
    # Benchmark backends
    "BandedMatrix",
    "BlockTriangularMatrix",
    "block_triang_matmul",
    "measure_effective_bandwidth",
    "snake_ordering",
    "rcm_ordering_from_adjacency",
    "apply_permutation",
]
