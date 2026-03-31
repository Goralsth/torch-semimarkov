# Benchmarking

Run the included benchmarks to reproduce paper results.

## Basic Usage

```bash
# Memory analysis across backends (default: forward+backward combined)
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --backends triton_streaming,linear_scan_streaming
```

## Triton-Accelerated Scan

Compare Triton kernel vs PyTorch reference implementations:

```bash
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 256,512,1024 \
    --K 8,12,16 \
    --C 6,9,12 \
    --backends triton_streaming,linear_scan_streaming
```

Note: Triton backends support `Log` and `Max` semirings.

## Phase-Separated Timing

Measure forward and backward passes separately:

```bash
# Forward pass only
python benchmarks/benchmark_memory_analysis.py \
    --phases forward \
    --backends linear_scan_streaming,triton_streaming

# Backward pass only
python benchmarks/benchmark_memory_analysis.py \
    --phases backward \
    --backends linear_scan_streaming,triton_streaming

# All phases for comparison
python benchmarks/benchmark_memory_analysis.py \
    --phases forward,backward,both \
    --backends linear_scan_streaming,triton_streaming
```

## Semiring Comparison

Compare different semirings (Log, Max, Entropy):

```bash
python benchmarks/benchmark_memory_analysis.py \
    --semirings Log,Max,Entropy \
    --backends triton_streaming,linear_scan_streaming \
    --T 128,256,512 \
    --K 4,8,12 \
    --C 3,6
```

Note: `triton_streaming` supports `Log` and `Max` semirings only.

## Full Benchmark Suite

Run comprehensive benchmarks across all dimensions:

```bash
python benchmarks/benchmark_memory_analysis.py \
    --device cuda:0 \
    --T 128,256,512,1024 \
    --K 4,8,12,16,20,24 \
    --C 3,6,9,12 \
    --B 4 \
    --repeats 5 \
    --phases forward,backward,both \
    --semirings Log,Max \
    --backends linear_scan_streaming,triton_streaming \
    --output-dir results/
```

## Available Options

### Backends

| Backend | Description | Memory | Semirings |
|---------|-------------|--------|-----------|
| `linear_scan_streaming` | Streaming O(N) scan (PyTorch reference) | O(KC) | All |
| `triton_streaming` | Fused Triton GPU kernel with hand-written forward and backward | O(KC) | Log, Max |
| `linear_scan_vectorized` | Vectorized linear scan | O(KC) | All |
| `binary_tree_sharded` | Sharded binary tree with checkpointing | O(KC) | All |

### Semirings

| Semiring | Description | Use Case |
|----------|-------------|----------|
| `Log` | Log-space (logsumexp, +) | Partition function, marginals |
| `Max` | Max-plus (max, +) | Viterbi decoding |
| `Entropy` | Entropy computation | Uncertainty quantification |

### Phases

| Phase | Description |
|-------|-------------|
| `forward` | Time forward pass only |
| `backward` | Time backward pass only |
| `both` | Time forward + backward together (default) |

## Output Files

The benchmark produces three output files in `--output-dir`:

- `benchmark_full.csv`: Complete results with all metrics
- `heatmap_data.json`: Data for OOM feasibility heatmaps
- `memory_breakdown.csv`: Memory breakdown by category

## Analysis and Plotting

Use the analysis script to generate plots and derived metrics from benchmark results:

```bash
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --format pdf

# Compare all backends against a specific baseline
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --baseline linear_scan_streaming
```

### Generated Plots

| Plot | Description |
|------|-------------|
| `scalability_T_*.pdf` | Time vs sequence length (log-log) - reveals O(N) vs O(log N) |
| `scalability_KC_*.pdf` | Time vs state-space size |
| `throughput_*.pdf` | Positions/sec vs KC |
| `memory_efficiency_*.pdf` | Memory per state-position |
| `backward_forward_ratio_*.pdf` | Cost of backward pass vs forward |
| `semiring_overhead_*.pdf` | Overhead of Max/Entropy vs Log |
| `time_ratio_baseline_*.pdf` | Time ratio vs baseline backend |
| `memory_ratio_baseline_*.pdf` | Memory ratio vs baseline backend |
| `time_ratio_heatmap_*.pdf` | Heatmap of time ratios (backend x KC) |
| `memory_ratio_heatmap_*.pdf` | Heatmap of memory ratios (backend x KC) |

### Generated Tables

| Table | Description |
|-------|-------------|
| `summary_stats.csv` | Aggregate statistics by backend/semiring/phase |
| `backward_forward_ratios.csv` | Backward/forward time ratios per config |
| `semiring_overhead.csv` | Overhead relative to LogSemiring |
| `crossover_points.csv` | KC thresholds where streaming beats other backends |
| `baseline_ratios.csv` | Time and memory ratios vs baseline backend |

### Example: Generate all analysis

```bash
# Run benchmarks
python benchmarks/benchmark_memory_analysis.py \
    --phases forward,backward \
    --semirings Log,Max \
    --backends linear_scan_streaming,triton_streaming \
    --T 128,256,512,1024 \
    --K 4,8,12 \
    --C 3,6,9 \
    --output-dir results/

# Analyze results
python benchmarks/analyze_benchmarks.py \
    --input results/benchmark_full.csv \
    --output-dir results/plots/ \
    --format pdf
```

### Example: Quick Triton vs Streaming Comparison

```bash
python benchmarks/benchmark_memory_analysis.py \
    --backends triton_streaming,linear_scan_streaming \
    --T 256,512,1024 \
    --K 8,16 \
    --C 6,12 \
    --phases forward \
    --output-dir results/quick/
```
