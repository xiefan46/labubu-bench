# CuTeDSL Grouped GEMM FP8 Benchmark Results

**Date**: 2026-02-03
**GPU**: B200 (SM100, ~4500 TFLOPs/s FP8 peak)
**Kernel**: `moe_grouped_gemm_fp8_cutedsl` (persistent grouped GEMM with float32 per-128-block scales)
**Baseline**: `group_gemm_fp8_nt_groupwise` (trtllm cubin kernel)
**dtype**: FP8 e4m3fn input, BF16 output

## A/B Comparison: CuTeDSL vs trtllm cubin

**Summary (81 configs): CuTeDSL / trtllm ratio: min=0.03x  median=0.21x  max=0.49x**

### DeepSeek-V3 Relevant Sizes (E=8)

| Stage | m | N | K | CuTeDSL | trtllm | ratio |
|-------|---|------|------|---------|--------|-------|
| GEMM1 | 64 | 4096 | 7168 | 27.4 | 337.7 | 0.08x |
| GEMM1 | 128 | 4096 | 7168 | 320.9 | 675.2 | 0.48x |
| GEMM1 | 512 | 4096 | 7168 | 614.9 | 1753.3 | 0.35x |
| GEMM2 | 64 | 7168 | 2048 | 14.1 | 287.5 | 0.05x |
| GEMM2 | 128 | 7168 | 2048 | 217.5 | 554.3 | 0.39x |
| GEMM2 | 512 | 7168 | 2048 | 483.3 | 1379.6 | 0.35x |

### Full Results by E (N=7168, K=7168 — largest config)

| E | m | CuTeDSL | trtllm | ratio |
|---|---|---------|--------|-------|
| 1 | 64 | 23.9 | 221.5 | 0.11x |
| 1 | 128 | 99.7 | 414.8 | 0.24x |
| 1 | 512 | 300.4 | 1195.8 | 0.25x |
| 4 | 64 | 39.9 | 317.2 | 0.13x |
| 4 | 128 | 285.4 | 667.5 | 0.43x |
| 4 | 512 | 596.0 | 1696.3 | 0.35x |
| 8 | 64 | 44.2 | 398.3 | 0.11x |
| 8 | 128 | 393.8 | 804.0 | 0.49x |
| 8 | 512 | 723.7 | 1713.1 | 0.42x |

### Key Patterns

- **m=64 is catastrophic**: 0.03x-0.13x — padded to 128, half compute wasted + kernel launch overhead dominates
- **m=128 best ratio**: up to 0.49x — single MMA tile in M, no waste
- **m=512 ratio drops to 0.25-0.42x**: trtllm scales better with more M tiles
- **Larger K improves ratio**: more K-tiles amortizes per-tile overhead better (K=7168 > K=2048)
- **trtllm peak: 1753 TFLOPs/s** (39% of FP8 peak) vs **CuTeDSL peak: 724 TFLOPs/s** (16%)

## Root Cause Analysis: Why 2-5x Slower

The gap is **NOT** primarily from per-K-tile TMEM readout. The trtllm cubin kernel also does per-K-tile scale application (float32 scales are not hardware-supported). The real bottlenecks are:

### 1. Per-K-tile ACCUMULATE=False Reset (Major)
Our kernel sets `ACCUMULATE=False` for every K-tile, meaning the MMA pipeline restarts from scratch each time. The trtllm kernel likely accumulates multiple K-tiles in TMEM before reading out, only resetting when needed for scale boundaries.

**Impact**: ~2x overhead from MMA pipeline restart penalty

### 2. No Async TMA/MMA Overlap for Scales (Major)
We load a_scale and b_scale synchronously in the epilogue warp. The trtllm kernel likely prefetches scales via TMA into SMEM concurrently with MMA execution.

**Impact**: Scale loads add latency to the critical path

### 3. Single epi_tidx for a_scale Indexing (Moderate)
Our epilogue applies `a_scale[k_tile, m_flat]` where `m_flat` is computed from a single thread index. This may cause uncoalesced memory access or incorrect scale broadcast across the M tile.

### 4. Python-side Padding Overhead (Minor for m=64)
For m=64, we allocate [E, 128, K] zero-padded tensors on every call. For small problems, this host overhead is non-trivial.

## Potential Optimizations (Priority Order)

1. **Multi-K-tile TMEM accumulation**: Accumulate 2-4 K-tiles in TMEM before reading out. Apply combined scale (product of per-K scales). Only works if scale variation is small enough to avoid overflow.
2. **Async scale prefetch**: Load scales into SMEM via TMA in the TMA warp pipeline, not in the epilogue path.
3. **ACCUMULATE=True for within-scale-group K-tiles**: If consecutive K-tiles can share a scale approximation, keep TMEM accumulating.
4. **Tile size tuning**: Try 128x256 or 256x128 MMA tiles.
5. **Remove Python padding overhead**: Pre-allocate padded buffers, or pad inside the kernel.
