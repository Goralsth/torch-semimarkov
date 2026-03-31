# Integration with upstream encoders

This guide shows how to use `flash-semicrf` as a structured prediction layer
on top of upstream encoders like transformers, Mamba SSMs, CNNs, or any
sequence model.

## Contents

- [Conceptual model](#conceptual-model) — how the pieces fit together
- [Two parameterization approaches](#two-parameterization-approaches) — direct edge potentials vs. HSMM factorization
- [Integration examples](#integration-examples) — BERT, Mamba, CNN, BiLSTM
- [What to do with the outputs](#what-to-do-with-the-outputs) — training, loss computation, decoding
- [Performance tips](#performance-tips) — Triton kernel, vectorized scan, batching
- [Memory considerations](#memory-considerations) — streaming vs. vectorized tradeoffs
- [Common patterns](#common-patterns) — genomics, NLP, time series
- [Beyond point predictions](#beyond-point-predictions-uncertainty-at-segment-boundaries) — uncertainty quantification at segment boundaries

## Conceptual model

```
Input sequence (batch, T, input_dim)
        │
        ▼
┌────────────────────┐
│      Encoder       │   BERT, Mamba, CNN, BiLSTM, etc.
│ (any architecture) │
└────────────────────┘
        │
        ▼
Hidden states (batch, T, hidden_dim)
        │
        ▼
┌───────────────────┐
│  Projection head  │   Linear layers to semi-CRF parameters
└───────────────────┘
        │
        ▼
Edge potentials (batch, T-1, K, C, C)  -or-  HSMM params
        │
        ▼
┌───────────────────┐
│    SemiMarkov     │   Structured inference (this library)
└───────────────────┘
        │
        ▼
Log partition / Viterbi path / Marginals
```

The encoder is **your choice**. This library handles the final structured
prediction layer.

> **Note**: The code examples in this guide are simplified illustrations to
> demonstrate the integration pattern, not production-ready implementations.
> Parameter choices (hidden dimensions, layer counts, K, C values) are
> placeholders and you should tune these for your specific task and data.

## Two parameterization approaches

### 1. Direct edge potentials

Predict the full `(batch, T-1, K, C, C)` tensor directly from encoder outputs.
Each position gets its own learned potentials for all duration and label
combinations.

**What the dimensions mean:**
- `batch`: Independent sequences
- `T-1`: Each position n (from 0 to T-2) predicts segments that *end* at n+1
- `K`: Possible segment durations (1 to K)
- `C, C`: Transition from previous label (last dim) to current label (second-to-last)

So `edge[b, n, k, c2, c1]` is the score for: "at position n in sequence b,
a segment of duration k with label c2 ends here, and the previous segment
had label c1."

**When to use direct edge potentials:**
- When transition patterns vary by position (e.g., near sequence boundaries)
- When you have enough data to learn position-specific patterns
- When the HSMM factorization is too restrictive for your domain
- When you want the encoder to learn everything end-to-end

**Projection head definition:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectEdgeHead(nn.Module):
    """Project encoder hidden states to semi-CRF edge potentials."""

    def __init__(self, hidden_dim, K, C):
        super().__init__()
        self.K = K
        self.C = C
        self.edge_proj = nn.Linear(hidden_dim, K * C * C)

    def forward(self, hidden_states):
        # hidden_states: (batch, T, hidden_dim)
        batch, T, _ = hidden_states.shape
        # Use positions 0..T-2 to predict edges
        edge_hidden = hidden_states[:, :-1, :]  # (batch, T-1, hidden_dim)
        edge_flat = self.edge_proj(edge_hidden)  # (batch, T-1, K*C*C)
        return edge_flat.view(batch, T - 1, self.K, self.C, self.C)
```

**Using it with flash-semicrf:**

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring

# Build model
head = DirectEdgeHead(hidden_dim=256, K=8, C=5)
crf = SemiMarkov(LogSemiring)

# Forward pass: encoder -> head -> semi-CRF
hidden_states = encoder(x)              # your encoder
edge = head(hidden_states)              # project to edge potentials
log_Z, _ = crf.logpartition(edge, lengths=lengths)  # <-- flash-semicrf
```

**Variant: Context window for edge prediction**

Instead of predicting edges from a single position, you can use a window of
context around each position:

```python
class ContextEdgeHead(nn.Module):
    """Predict edge potentials using local context window."""

    def __init__(self, hidden_dim, K, C, context_size=5):
        super().__init__()
        self.K = K
        self.C = C
        self.context_size = context_size
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * context_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K * C * C)
        )

    def forward(self, hidden_states):
        batch, T, H = hidden_states.shape
        pad = self.context_size // 2

        # Pad and unfold to get context windows
        padded = F.pad(hidden_states, (0, 0, pad, pad))
        windows = padded.unfold(1, self.context_size, 1)
        windows = windows.reshape(batch, T, -1)

        edge_flat = self.edge_mlp(windows[:, :-1])
        return edge_flat.view(batch, T - 1, self.K, self.C, self.C)
```

**Inspecting learned edge potentials**

After training, you can visualize what the model predicts at specific
positions to understand position-dependent patterns:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# After a forward pass on some data
edge = head(encoder(x))  # (batch, T-1, K, C, C)

# Average edge potentials across a batch for a specific position
pos = 50
avg_edge_at_pos = edge[:, pos].mean(dim=0)  # (K, C, C)

# Plot transition preferences at this position for duration k=1
labels = ["5'UTR", "CDS", "intron", "3'UTR", "intergenic"]
trans_at_pos = avg_edge_at_pos[1]  # duration=1, shape (C, C)

sns.heatmap(trans_at_pos.detach().cpu(), annot=True, fmt=".1f",
            xticklabels=labels, yticklabels=labels, cmap="RdBu_r")
plt.title(f"Learned transition scores at position {pos}")
plt.xlabel("Previous label")
plt.ylabel("Current label")
```

This lets you see if the model learns position-specific behavior, like
different transition patterns near the start vs. middle of sequences.

### 2. HSMM factorization

The Hidden Semi-Markov Model (HSMM) factorization breaks edge potentials into
interpretable components:

- **Initial distribution** `(C,)`: Which segment type is likely to start the sequence?
- **Transitions** `(C, C)`: When a segment ends, what type comes next?
  (e.g., exon → intron is common, exon → exon is rare)
- **Durations** `(C, K)`: How long do segments of each type tend to be?
  (e.g., exons are typically shorter than introns)
- **Emissions** `(batch, T, K, C)`: Given the input at positions n to n+k,
  how well does label c explain the data? This is what your encoder predicts.

This factorization has two practical advantages:

1. **Parameter efficiency**: Instead of learning `T * K * C * C` values,
   you learn `C + C*C + C*K` shared parameters plus `T * K * C` emissions.
   The encoder only needs to predict emissions; transitions and durations
   are learned as small parameter matrices.

2. **Interpretability**: After training, you can extract and visualize the
   learned parameters to understand what the model learned about your domain.

**Projection head definition:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_semicrf import SemiMarkov

class HSMMHead(nn.Module):
    """HSMM-factorized projection head for semi-CRF."""

    def __init__(self, hidden_dim, K, C):
        super().__init__()
        self.K = K
        self.C = C

        # Learnable transition parameters (shared across sequences)
        self.init_logits = nn.Parameter(torch.zeros(C))
        self.trans_logits = nn.Parameter(torch.zeros(C, C))
        self.dur_logits = nn.Parameter(torch.zeros(C, K))

        # Per-position emissions from encoder
        self.emission_proj = nn.Linear(hidden_dim, K * C)

    def forward(self, hidden_states):
        # hidden_states: (batch, T, hidden_dim)
        batch, T, _ = hidden_states.shape

        # Emission scores from encoder
        emission_flat = self.emission_proj(hidden_states)
        emission = emission_flat.view(batch, T, self.K, self.C)

        # Normalize transition parameters to log-probabilities
        init_z = F.log_softmax(self.init_logits, dim=-1)
        trans_z = F.log_softmax(self.trans_logits, dim=-1)
        dur_z = F.log_softmax(self.dur_logits, dim=-1)

        # Combine into edge potentials using flash-semicrf helper
        edge = SemiMarkov.hsmm(init_z, trans_z, dur_z, emission)  # <-- flash-semicrf
        return edge
```

**Using it with flash-semicrf:**

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring

# Build model
head = HSMMHead(hidden_dim=256, K=8, C=5)
crf = SemiMarkov(LogSemiring)

# Forward pass: encoder -> head -> semi-CRF
hidden_states = encoder(x)              # your encoder
edge = head(hidden_states)              # project to edge potentials (uses SemiMarkov.hsmm)
log_Z, _ = crf.logpartition(edge, lengths=lengths)  # <-- flash-semicrf
```

**Inspecting a trained model**

After training, you can extract and visualize the learned parameters:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract learned parameters from trained head
trans_probs = torch.softmax(head.trans_logits, dim=-1).detach().cpu()
dur_probs = torch.softmax(head.dur_logits, dim=-1).detach().cpu()

# Label names for a splicing model
labels = ["5'UTR", "CDS", "intron", "3'UTR", "intergenic"]

# Plot transition matrix and duration distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(trans_probs, annot=True, fmt=".2f",
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", ax=axes[0])
axes[0].set_xlabel("Next segment")
axes[0].set_ylabel("Current segment")
axes[0].set_title("Learned transition probabilities")

for i, label in enumerate(labels):
    axes[1].plot(dur_probs[i], label=label)
axes[1].set_xlabel("Duration (positions)")
axes[1].set_ylabel("Probability")
axes[1].set_title("Learned duration distributions")
axes[1].legend()

plt.tight_layout()
plt.savefig("learned_splicing_structure.png")
```

This might reveal, for example:
- CDS → intron and intron → CDS have high probability (expected for splicing)
- intron → intron is near zero (introns don't self-transition)
- Introns have a broader duration distribution than exons
- 5'UTR → CDS is common, but CDS → 5'UTR is rare (directional gene structure)

## Integration examples

### With HuggingFace Transformers (BERT, RoBERTa, etc.)

```python
from transformers import AutoModel
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring

class BERTSemiCRF(nn.Module):
    def __init__(self, model_name, K, C):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = HSMMHead(self.encoder.config.hidden_size, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, input_ids, attention_mask, lengths):
        # Encode
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, T, hidden_dim)

        # Project to semi-CRF potentials
        edge = self.head(hidden)

        # Compute log partition (for training loss)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge

    def decode(self, input_ids, attention_mask, lengths):
        """Viterbi decoding for inference."""
        from flash_semicrf.semirings import MaxSemiring

        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        edge = self.head(hidden)

        crf_max = SemiMarkov(MaxSemiring)
        score, _ = crf_max.logpartition(edge, lengths=lengths)
        # Use marginals or backtrack for best path
        return score
```

### With Mamba SSM

```python
from mamba_ssm import Mamba

class MambaSemiCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, C, num_layers=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = HSMMHead(hidden_dim, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, x, lengths):
        # x: (batch, T, input_dim) - e.g., one-hot DNA or embeddings
        h = self.embed(x)

        for layer in self.layers:
            h = layer(h) + h  # residual
        h = self.norm(h)

        edge = self.head(h)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge
```

### With a simple CNN (for genomics)

```python
class CNNSemiCRF(nn.Module):
    """CNN encoder for DNA sequences with semi-CRF decoder."""

    def __init__(self, K, C, hidden_dim=256):
        super().__init__()
        # DNA: 4 channels (A, C, G, T one-hot)
        self.conv = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7),
        )
        self.head = HSMMHead(hidden_dim, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, x_onehot, lengths):
        # x_onehot: (batch, T, 4) -> transpose for conv1d
        h = self.conv(x_onehot.transpose(1, 2))  # (batch, hidden, T)
        h = h.transpose(1, 2)  # (batch, T, hidden)

        edge = self.head(h)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge
```

### With bidirectional LSTM

```python
class BiLSTMSemiCRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, K, C, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.head = HSMMHead(hidden_dim, K, C)
        self.crf = SemiMarkov(LogSemiring)

    def forward(self, x, lengths):
        # Pack for variable length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        edge = self.head(h)
        log_Z, _ = self.crf.logpartition(edge, lengths=lengths)
        return log_Z, edge
```

## What to do with the outputs

### Understanding `log_Z` (the log partition function)

When you call `crf.logpartition(edge, lengths)`, you get back `log_Z`, a tensor
of shape `(batch,)`. This is the **log partition function** - the log of the
sum of exponentiated scores over all possible segmentations:

```
log_Z = log( Σ_{all segmentations y} exp(score(y)) )
```

By itself, `log_Z` is just a number. Its purpose is as a **normalizer** that
turns raw scores into probabilities. You'll use it in two main ways:

1. **Training**: Compute the loss as `log_Z - score(gold_segmentation)`
2. **Inference**: Compare `log_Z` across different inputs or use marginals

### Training: Computing the loss

The standard training objective is **conditional log-likelihood**. The loss
for a single example is:

```
loss = log_Z - score(gold_segmentation)
```

This pushes the model to assign high scores to the correct segmentation
relative to all other possible segmentations.

```python
def compute_loss(model, x, lengths, gold_edges):
    """
    Negative log-likelihood loss for semi-CRF.

    Args:
        x: Input sequences
        lengths: Sequence lengths
        gold_edges: Ground truth edge indicators (batch, T-1, K, C, C)
                   with 1s marking true segment boundaries
    """
    log_Z, edge = model(x, lengths)

    # Score of gold segmentation: sum of potentials at gold edges
    gold_score = (edge * gold_edges).sum(dim=(1, 2, 3, 4))

    # NLL = log_Z - gold_score
    loss = (log_Z - gold_score).mean()
    return loss
```

**Preparing your labels:**

Your ground truth labels need to be converted to edge format. Use
`SemiMarkov.to_parts()`:

```python
# labels: (batch, T) tensor
# -1 = continuation of previous segment
# 0, 1, 2, ... = segment of that class starts here
#
# Example: [0, -1, -1, 1, -1, 2] means:
#   - Class 0 segment from position 0-2
#   - Class 1 segment from position 3-4
#   - Class 2 segment at position 5

gold_edges = SemiMarkov.to_parts(labels, extra=(C, K))
# gold_edges: (batch, T-1, K, C, C) with 1s at segment boundaries
```

**Complete training loop:**

```python
model = MambaSemiCRF(input_dim=4, hidden_dim=256, K=8, C=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in dataloader:
        x, labels, lengths = batch

        # Convert labels to edge format
        gold_edges = SemiMarkov.to_parts(labels, extra=(C, K)).float().to(x.device)

        # Forward pass
        log_Z, edge = model(x, lengths)

        # Compute loss
        gold_score = (edge * gold_edges).sum(dim=(1, 2, 3, 4))
        loss = (log_Z - gold_score).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference: Decoding the best segmentation

To get the most likely segmentation (Viterbi decoding), use `MaxSemiring`:

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import MaxSemiring

def decode(model, x, lengths):
    """Get the most likely segmentation."""
    # Get edge potentials from your model
    hidden = model.encoder(x)
    edge = model.head(hidden)

    # Viterbi decoding with MaxSemiring
    crf_max = SemiMarkov(MaxSemiring)
    best_score, _ = crf_max.logpartition(edge, lengths=lengths)

    # Get the actual segmentation via marginals
    _, edge_marginals = crf_max.marginals(edge, lengths=lengths)

    # edge_marginals now contains 1s at the best path edges
    return edge_marginals, best_score
```

**Converting predictions back to labels:**

```python
# edge_marginals: (batch, T-1, K, C, C) with 1s at predicted boundaries
predicted_labels, (C, K) = SemiMarkov.from_parts(edge_marginals)
# predicted_labels: (batch, T) with -1 for continuation, class indices for starts
```

**Extracting segment boundaries:**

```python
def extract_segments(labels):
    """Convert label sequence to list of (start, end, class) tuples."""
    segments = []
    current_start = 0
    current_class = labels[0].item()

    for i, label in enumerate(labels[1:], 1):
        if label.item() != -1:  # New segment starts
            segments.append((current_start, i, current_class))
            current_start = i
            current_class = label.item()

    # Don't forget the last segment
    segments.append((current_start, len(labels), current_class))
    return segments

# Example output: [(0, 3, 0), (3, 5, 1), (5, 6, 2)]
# Meaning: class 0 from 0-3, class 1 from 3-5, class 2 from 5-6
```

## Performance tips

1. **Use `SemiMarkovCRFHead` for most use cases** (recommended):
   ```python
   from flash_semicrf import SemiMarkovCRFHead

   # Create CRF head with automatic backend selection
   crf = SemiMarkovCRFHead(num_classes=C, max_duration=K, hidden_dim=hidden_dim)

   # Forward pass (auto uses streaming; Triton on GPU, PyTorch reference on CPU)
   result = crf(hidden_states, lengths)
   partition = result['partition']

   # Training with loss
   loss = crf.compute_loss(hidden_states, lengths, labels)

   # Force specific backend if needed
   result = crf(hidden_states, lengths, backend="streaming")  # For genome-scale
   result = crf(hidden_states, lengths, backend="exact")      # For small T
   ```

2. **For very long sequences (T > 10K)**, the streaming API is essential:
   ```python
   from flash_semicrf.streaming import semi_crf_streaming_forward

   # Streaming: compute edges on-the-fly (O(KC) memory)
   partition = semi_crf_streaming_forward(cum_scores, transition, duration_bias, lengths, K)
   partition.sum().backward()  # Hand-written Triton backward kernel
   ```

3. **Use `SemiMarkov.logpartition` for pre-computed edge tensors**:
   ```python
   from flash_semicrf import SemiMarkov
   from flash_semicrf.semirings import LogSemiring, MaxSemiring, EntropySemiring

   # If you have a pre-computed edge tensor (batch, T-1, K, C, C)
   crf = SemiMarkov(LogSemiring)
   log_Z, _ = crf.logpartition(edge, lengths=lengths)

   # Viterbi decoding (max semiring)
   crf_max = SemiMarkov(MaxSemiring)
   best_score, _ = crf_max.logpartition(edge, lengths=lengths)

   # Access ALL 7 semirings (not possible with streaming API)
   crf_ent = SemiMarkov(EntropySemiring)
   entropy, _ = crf_ent.logpartition(edge, lengths=lengths)
   ```

   > **Note**: The streaming API (tips 1-2) only supports log and max semirings.
   > For entropy, KL divergence, cross-entropy, counting, or K-best, use
   > `SemiMarkov.logpartition` with pre-computed edge tensors. This requires
   > the edge tensor to fit in memory.

4. **Choose K carefully**: Memory and compute scale with K. Use empirical
   quantiles (p95/p99) of segment lengths rather than maximum.

5. **Batch similar lengths together** to minimize padding waste.

6. **First training call overhead**: The first call with `requires_grad=True`
   incurs a small warmup cost as Triton compiles and caches kernels.

## Memory considerations

The semi-CRF layer adds minimal overhead to your model:

- **Streaming scan** (default): O(KC) memory for the ring buffer, independent
  of sequence length. For typical values (K=100, C=8), this is negligible.
- **Vectorized scan**: O(TKC) memory, scaling with sequence length. Use when
  you have memory headroom as it is slightly faster than the streaming approach under various use cases.

In practice, your encoder (transformer, Mamba, etc.) will dominate memory.
The semi-CRF inference layer is now lightweight by comparison.

## Common patterns

### Genomics: Long sequences, moderate K and C

```python
# Gene structure annotation
T = 10000      # 10kb chunks
K = 500        # Max exon/intron length in tokens
C = 5          # exon, intron, UTR, intergenic, etc.

# Use streaming scan (O(KC) = O(2500) memory)
model = MambaSemiCRF(input_dim=4, hidden_dim=256, K=K, C=C)
```

### NLP: Shorter sequences, small K

```python
# Named entity recognition with duration
T = 512        # BERT max length
K = 16         # Max entity span
C = 9          # BIO tags or entity types

model = BERTSemiCRF("bert-base-uncased", K=K, C=C)
```

### Time series segmentation

```python
# Activity recognition
T = 1000       # 1000 timesteps
K = 100        # Max activity duration
C = 6          # Activity classes

model = BiLSTMSemiCRF(input_dim=6, hidden_dim=128, K=K, C=C)
```

## Beyond point predictions: Uncertainty at segment boundaries

A key advantage of semi-CRFs over simpler decoders (like per-position softmax)
is that they define a **distribution over segmentations**, not just a single
prediction. This gives you principled uncertainty estimates about where segment
boundaries actually are.

### Why this matters

When you decode with a standard classifier, you get a label per position, but
no sense of how confident the model is about boundary placement. With a
semi-CRF, the model explicitly reasons about segments as units, and you can
ask: "How probable is it that a segment boundary occurs at position n?"

This is valuable for:
- **Knowing when to trust predictions**: High uncertainty regions may need
  manual review or additional data
- **Scientific applications**: In genomics, knowing that an exon-intron
  boundary is uncertain within a 10bp window is more honest than reporting
  a single coordinate
- **Active learning**: Sample regions where boundary uncertainty is high
- **Downstream analysis**: Propagate uncertainty into variant effect
  predictions, isoform quantification, etc.

### Computing marginal probabilities

The `marginals()` method computes posterior probabilities over edges:

```python
from flash_semicrf import SemiMarkov
from flash_semicrf.semirings import LogSemiring

model = SemiMarkov(LogSemiring)
log_Z, edge_marginals = model.marginals(edge, lengths=lengths)

# edge_marginals has shape (batch, T-1, K, C, C)
# edge_marginals[b, n, k, c2, c1] = P(segment of label c2, duration k,
#                                     ending at position n+1,
#                                     following a segment of label c1)
```

### Visualizing boundary uncertainty

You can aggregate marginals to see where the model is confident vs. uncertain
about segment boundaries:

```python
# Probability of ANY segment boundary at each position
# Sum over all durations, current labels, and previous labels
boundary_prob = edge_marginals.sum(dim=(2, 3, 4))  # (batch, T-1)

# Plot boundary probability along a sequence
plt.figure(figsize=(12, 3))
plt.plot(boundary_prob[0].detach().cpu())
plt.xlabel("Position")
plt.ylabel("P(segment boundary)")
plt.title("Boundary probability along sequence")

# High values = confident boundary; low values = mid-segment
# Intermediate values = uncertain boundary location
```

### Comparing to the Viterbi path

The Viterbi path (via `MaxSemiring`) gives you the single most probable
segmentation, but it discards uncertainty information:

```python
from flash_semicrf.semirings import MaxSemiring

# Viterbi: single best path (no uncertainty)
model_max = SemiMarkov(MaxSemiring)
best_score, _ = model_max.logpartition(edge, lengths=lengths)

# Marginals: full posterior (preserves uncertainty)
model_log = SemiMarkov(LogSemiring)
log_Z, marginals = model_log.marginals(edge, lengths=lengths)
```

In regions where the posterior is peaked, Viterbi and marginals will agree.
In ambiguous regions, the marginals will show spread across multiple
boundary positions, while Viterbi arbitrarily picks one.

### Entropy as a global uncertainty measure

For a single scalar measure of segmentation uncertainty, you can compute the
entropy of the posterior distribution using `EntropySemiring`:

```python
from flash_semicrf.semirings import EntropySemiring

model_ent = SemiMarkov(EntropySemiring)
entropy, _ = model_ent.logpartition(edge, lengths=lengths)

# Higher entropy = more uncertainty about the segmentation
# Lower entropy = model is confident in a particular segmentation
```

This is useful for flagging sequences that may need additional review or
where predictions should be treated with caution.
