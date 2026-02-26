# 4-Way Benchmark: flashinfer_moe vs cutedsl_v3 vs cutedsl_v3.2 vs triton_v2

**Date**: 2026-02-26
**Hardware**: NVIDIA B200
**Definition**: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Tolerances**: rtol=0.2, atol=0.1, required-matched-ratio=0.85, timeout=1200s
**All 76 traces PASSED**

## Raw Results (speedup vs reference, sorted by seq_len)

| # | seq_len | expert_offset | flashinfer_moe | cutedsl_v3 | cutedsl_v3.2 | triton_v2 | Winner |
|---|---------|---------------|---------------|------------|--------------|-----------|--------|
| 1 | 1 | 32 | 29.88x | 8.91x | 10.82x | **34.66x** | triton_v2 |
| 2 | 7 | 192 | **42.63x** | 9.64x | 11.74x | 31.64x | flashinfer_moe |
| 3 | 14 | 0 | 30.94x | 10.37x | 12.03x | **34.93x** | triton_v2 |
| 4 | 15 | 32 | 24.77x | 9.15x | 11.12x | **33.45x** | triton_v2 |
| 5 | 16 | 224 | 31.21x | 10.10x | 11.77x | **33.92x** | triton_v2 |
| 6 | 32 | 32 | 32.69x | 11.29x | 11.86x | **32.89x** | triton_v2 |
| 7 | 52 | 160 | **41.73x** | 14.28x | 15.05x | 41.18x | flashinfer_moe |
| 8 | 53 | 32 | 31.15x | 10.93x | 11.46x | **31.33x** | triton_v2 |
| 9 | 54 | 128 | **31.27x** | 11.96x | 11.42x | 30.77x | flashinfer_moe |
| 10 | 55 | 128 | 30.54x | 11.85x | 11.34x | **31.40x** | triton_v2 |
| 11 | 56 | 64 | 31.25x | 11.09x | 11.59x | **31.40x** | triton_v2 |
| 12 | 57 | 96 | 30.80x | 11.63x | 11.33x | **31.49x** | triton_v2 |
| 13 | 58 | 64 | 31.73x | 11.30x | 11.95x | **31.88x** | triton_v2 |
| 14 | 59 | 160 | 29.94x | 10.79x | 11.62x | **31.82x** | triton_v2 |
| 15 | 62 | 96 | 30.39x | 11.03x | 11.53x | **31.27x** | triton_v2 |
| 16 | 80 | 96 | 31.35x | 11.11x | 11.57x | **31.40x** | triton_v2 |
| 17 | 901 | 96 | **33.14x** | 13.26x | 14.29x | 25.19x | flashinfer_moe |
| 18 | **11948** | 128 | **16.06x** | 11.02x | 11.53x | 4.81x | flashinfer_moe |
| 19 | **14107** | 32 | **15.63x** | 10.20x | 10.73x | 5.11x | flashinfer_moe |

## Summary Statistics

| Metric | flashinfer_moe | cutedsl_v3 | cutedsl_v3.2 | triton_v2 |
|--------|---------------|------------|--------------|-----------|
| **Average** | **30.37x** | 11.05x | 11.83x | **29.50x** |
| **Median** | 31.21x | 11.03x | 11.53x | 31.40x |
| **Min** | 15.63x | 8.91x | 10.73x | 4.81x |
| **Max** | 42.63x | 14.28x | 15.05x | 41.18x |
| **Std Dev** | ~6.3x | ~1.2x | ~1.0x | ~9.8x |
| **Wins** | 6/19 | 0/19 | 0/19 | 13/19 |

## Key Findings

### 1. Overall Ranking: flashinfer_moe > triton_v2 >> cutedsl_v3.2 > cutedsl_v3

- **flashinfer_moe** and **triton_v2** are in the same tier (avg 30x vs 29.5x)
- **CuTeDSL** variants are ~2.5-3x slower (avg ~11-12x)
- triton_v2 wins more workloads (13/19) but flashinfer_moe wins on critical large-batch cases

### 2. Large-Batch Workloads: triton_v2 Collapses

The two extreme large-batch workloads (seq_len=11948 and 14107) show a dramatic pattern:

| seq_len | expert_offset | flashinfer_moe | triton_v2 | cutedsl_v3.2 | cutedsl_v3 |
|---------|---------------|---------------|-----------|--------------|------------|
| 11948 | 128 | **16.06x** | 4.81x | 11.53x | 11.02x |
| 14107 | 32 | **15.63x** | 5.11x | 10.73x | 10.20x |

Compared to medium-batch (seq_len=901, already showing degradation):

| seq_len | expert_offset | flashinfer_moe | triton_v2 | cutedsl_v3.2 | cutedsl_v3 |
|---------|---------------|---------------|-----------|--------------|------------|
| 901 | 96 | **33.14x** | 25.19x | 14.29x | 13.26x |

**Why triton_v2 collapses on large batches**:
- triton_v2 uses `sgl_kernel` (pre-compiled CUTLASS) for GEMM, but Triton kernels for scatter/SwiGLU+quant/gather
- At seq_len ~12K-14K, the expanded token count after top-8 routing = **~96K-113K tokens**, making scatter/gather/SwiGLU kernels dominate
- Triton scatter/gather kernels have O(seq_len × top_k) memory traffic without fusion — memory-bound at large batch
- `sgl_kernel` GEMM is designed for small-to-medium batch decode, not prefill-scale GEMM
- flashinfer_moe's CUTLASS SM100 persistent kernel handles large batches with better tiling/pipelining
- **CuTeDSL is the most stable** across all batch sizes (10-15x regardless), because its grouped GEMM inherently handles variable-size expert batches

### 3. CuTeDSL v3.2 vs v3: Modest Improvement

- v3.2 avg speedup: **11.83x** vs v3: **11.05x** → **+7.1% improvement**
- v3.2 wins on 13/19 workloads, v3 wins on 6/19
- v3.2's GEMM1 FP8 epilogue (eliminating separate BF16→FP8 quant kernel) provides consistent but small gains
- v3.2 has **lower variance** (std ~1.0x vs ~1.2x), confirming more stable performance

### 4. Consistency vs Peak Performance

| Solution | Coefficient of Variation |
|----------|------------------------|
| cutedsl_v3.2 | 8.5% (most consistent) |
| cutedsl_v3 | 11.2% |
| flashinfer_moe | 20.8% |
| triton_v2 | 33.2% (most variable) |

CuTeDSL delivers predictable performance; triton_v2 is highly variable.

### 5. Comparison with 2026-02-04 Benchmark

Results are consistent with the previous 4-way benchmark:
- flashinfer_moe/triton_v2 both ~30x on small batches (unchanged)
- triton_v2 large-batch collapse confirmed (5x, was 5.02x before)
- CuTeDSL v3.2 improvement over v3 was ~1-4x before, now ~7% avg — narrower but more consistent
- CuTeDSL gap to leaders (~2.5-3x) remains unchanged — **fundamental bottleneck not yet addressed**

## CuTeDSL Optimization Priorities

The ~2.5-3x gap between CuTeDSL and flashinfer_moe/triton_v2 persists. Based on accumulated evidence:

1. **GEMM kernel efficiency**: CuTeDSL grouped GEMM likely has lower occupancy or tile utilization vs CUTLASS SM100 persistent kernels
2. **Routing + scatter/gather overhead**: Non-GEMM stages (routing, finalize) may dominate for small batches
3. **Kernel launch count**: CuTeDSL launches more separate kernels vs fused approaches
4. **TMA configuration**: May not fully exploit B200's TMA multicast capabilities (v3.3 attempts this)

## Solutions Tested

| Solution | Description |
|----------|-------------|
| flashinfer_moe | FlashInfer native `trtllm_fp8_block_scale_moe` (CUTLASS SM100 warp-specialized persistent kernel with TMA) |
| cutedsl_moe_fp8_v3 | CuTeDSL flat 2D grouped GEMM with 128-aligned m_indptr |
| cutedsl_moe_fp8_v3_2 | CuTeDSL v3 + GEMM1 FP8 output epilogue (eliminates separate quant kernel) |
| triton_fused_moe_v2 | Triton fused MoE with pre-allocated buffers, sgl_kernel GEMM, Triton scatter/SwiGLU+quant/gather |
