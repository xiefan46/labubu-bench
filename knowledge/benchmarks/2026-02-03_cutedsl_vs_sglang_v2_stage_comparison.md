# CuTeDSL vs SGLang v2: Stage-by-Stage MoE Comparison

**Date:** 2026-02-03
**Hardware:** NVIDIA B200
**Config:** T=2048, E_global=256, E_local=32, top_k=8, H=7168, I=2048

## Per-Stage Timing (median ms, 10 iters)

| Stage | CuTeDSL | SGLang v2 | Ratio | Verdict |
|-------|---------|-----------|-------|---------|
| 1. Routing | 1.269 | 0.233 | 5.45x | CuTeDSL slow |
| 2. GEMM1 | 4.241 | 0.614 | **6.91x** | **Major bottleneck** |
| 3. SwiGLU+Requant | 0.084 | 0.303 | 0.28x | CuTeDSL wins |
| 4. GEMM2 | 3.012 | 0.336 | **8.95x** | **Major bottleneck** |
| 5. Finalize | 0.063 | 0.075 | 0.83x | ~Parity |
| Workspace alloc | 0.026 | --- | --- | |
| **End-to-end** | **8.155** | **1.535** | **5.31x** | |

## Bottleneck Analysis

### GEMM1 + GEMM2 = 7.25ms (89% of total) vs SGLang 0.95ms

The grouped GEMM is the dominant bottleneck. Root causes:

1. **Python-side data reshaping overhead** in `moe_grouped_gemm_fp8_cutedsl`:
   - Pads `[total_M, K]` → `[E, max_M, K]` with a Python for-loop (line 1196-1201)
   - Similarly pads `a_scale` with a for-loop (line 1213-1220)
   - Extracts output back with a for-loop (line 1264-1271)

2. **GEMM1 extra requant step**: `moe_gemm1_fp8` runs grouped GEMM → BF16, then calls `_quantize_output_fp8` (pure Python) to convert BF16→FP8. SGLang's GEMM directly outputs BF16 (no intermediate requant).

3. **Kernel performance**: CuTeDSL grouped GEMM may itself be slower than sgl_kernel's CUTLASS-based `fp8_blockwise_scaled_grouped_mm`.

### Routing: 1.27ms vs 0.23ms (5.45x)

After vectorization fix (was 327ms before), still 5x slower than `moe_fused_gate` + `prepare_moe_input` (fused CUDA kernels in sgl_kernel). The remaining gap is:
- `fused_topk_deepseek` kernel call + torch tensor ops (argsort, bincount, scatter)
- vs sgl_kernel's single fused `prepare_moe_input` CUDA kernel

### SwiGLU + Finalize: CuTeDSL wins

- SwiGLU fused CuTeDSL kernel: 0.084ms vs 0.303ms (3.6x faster)
- Finalize: 0.063ms vs 0.075ms (comparable)

## Precision

- cosine_sim = 0.998808 (good)
- max_abs_err = 88064 (large, likely due to FP8 quantization differences)

## Optimization Priority

1. **GEMM1/GEMM2 Python overhead**: Eliminate for-loops in `moe_grouped_gemm_fp8_cutedsl` (reshape/unreshape). Use vectorized pad/gather.
2. **GEMM1 requant**: Fuse BF16→FP8 quantization into the GEMM kernel output path (avoid `_quantize_output_fp8`).
3. **Routing**: Consider using `prepare_moe_input` from sgl_kernel directly, or write a CUDA permutation kernel.
4. **GEMM kernel perf**: Profile CuTeDSL kernel vs sgl_kernel grouped GEMM in isolation.

## Historical Context

| Version | Routing | Total | Notes |
|---------|---------|-------|-------|
| v0 (Python loop) | 327ms | 336ms | `_compute_permutation_indices` Python for loop |
| v1 (vectorized) | 1.27ms | 8.15ms | This measurement |
| SGLang v2 | 0.23ms | 1.54ms | Target baseline |
