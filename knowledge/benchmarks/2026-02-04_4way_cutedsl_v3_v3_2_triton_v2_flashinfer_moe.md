# 4-Way Benchmark: CuTeDSL v3 vs v3.2 vs Triton v2 vs FlashInfer MoE

**Date**: 2026-02-04
**Hardware**: NVIDIA B200
**Environment**: torch 2.9.1+cu128, triton 3.5.1, cuda 12.8
**Definition**: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Tolerances**: rtol=0.2, atol=0.1, required-matched-ratio=0.85

## Results (speedup vs reference)

| seq_len | flashinfer_moe | triton_v2 | cutedsl_v3 | cutedsl_v3.2 | winner |
|---------|---------------|-----------|------------|--------------|--------|
| 1       | 33.32x        | **35.24x** | 8.21x     | 10.68x       | triton_v2 |
| 7       | **45.31x**    | 30.26x    | 8.43x     | 10.67x       | flashinfer_moe |
| 14      | 31.76x        | **33.87x** | 11.57x   | 12.12x       | triton_v2 |
| 15      | 30.72x        | **32.58x** | 10.25x   | 12.36x       | triton_v2 |
| 16      | **34.57x**    | 26.17x    | 14.31x   | 14.96x       | flashinfer_moe |
| 32      | 32.40x        | **32.79x** | 9.32x    | 10.90x       | triton_v2 |
| 52      | 31.21x        | **34.30x** | 9.19x    | 12.15x       | triton_v2 |
| 53      | 32.69x        | **34.04x** | 12.30x   | 12.55x       | triton_v2 |
| 54 (a)  | 30.63x        | **37.23x** | 7.49x    | 13.27x       | triton_v2 |
| 54 (b)  | 32.92x        | **34.59x** | 12.44x   | 13.14x       | triton_v2 |
| 55      | 31.30x        | **32.49x** | 11.66x   | 11.65x       | triton_v2 |
| 56      | 31.64x        | **33.41x** | 11.62x   | 12.26x       | triton_v2 |
| 58      | 33.64x        | **36.10x** | 13.08x   | 13.81x       | triton_v2 |
| 60      | 30.81x        | **33.63x** | 11.81x   | 12.16x       | triton_v2 |
| 62      | **34.35x**    | 31.67x    | 7.32x    | 11.73x       | flashinfer_moe |
| 80      | **38.42x**    | 24.30x    | 10.49x   | 11.35x       | flashinfer_moe |
| 901     | **15.78x**    | 5.33x     | 10.09x   | 9.29x        | flashinfer_moe |
| 11948   | **16.14x**    | 5.02x     | 10.89x   | 11.46x       | flashinfer_moe |
| 14107   | **16.14x**    | 5.02x     | 10.89x   | 11.46x       | flashinfer_moe |

## Key Findings

### 1. flashinfer_moe (trtllm CUTLASS SM100 kernel)
- Best overall across all seq_len ranges
- Dominates large batches: 15-16x at seq_len=901/11948 (vs triton 5x)
- Competitive on small batches: 30-45x (vs triton 30-37x)
- Uses CUTLASS warp-specialized persistent kernel with TMA

### 2. triton_fused_moe_v2
- Fastest on most small seq_len (1-60): 30-37x
- Collapses on large batches: 5x at seq_len=901/11948
- Pre-allocated buffers eliminate per-call torch.empty overhead
- Uses sgl_kernel for GEMM, Triton for scatter/SwiGLU+quant/gather

### 3. cutedsl_moe_fp8_v3.2 vs v3 (FP8 epilogue)
- v3.2 consistently faster than v3: typically +1-4x improvement
- Biggest wins where v3 was weakest (e.g., 7.49x→13.27x at seq_len=54)
- Confirms eliminating BF16→FP8 quantization kernel is beneficial
- Exception: large batches (901) where v3.2 slightly slower (9.29x vs 10.09x)

### 4. CuTeDSL gap analysis
- CuTeDSL v3.2 is ~2.5-3x slower than flashinfer_moe/triton_v2 on small batches
- Gap narrows on large batches (v3.2 11.46x vs flashinfer 16.14x = 1.4x gap)
- Main bottleneck is NOT GEMM1 quantization (fixed by v3.2)
- Likely bottlenecks: routing overhead, scatter/gather kernels, kernel launch count

## Solutions Tested

| Solution | Description |
|----------|-------------|
| flashinfer_moe | FlashInfer native `trtllm_fp8_block_scale_moe` (CUTLASS SM100 persistent kernel) |
| triton_fused_moe_v2 | Triton fused MoE with pre-allocated buffers, sgl_kernel GEMM |
| cutedsl_moe_fp8_v3 | CuTeDSL flat grouped GEMM with 128-aligned m_indptr |
| cutedsl_moe_fp8_v3_2 | CuTeDSL v3 + GEMM1 FP8 output epilogue (eliminates separate quant kernel) |
