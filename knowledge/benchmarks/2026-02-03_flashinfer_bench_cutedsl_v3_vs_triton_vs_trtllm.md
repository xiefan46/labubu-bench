# flashinfer-bench: CuTeDSL v3 vs Triton v2 vs flashinfer_moe (trtllm)

## Date: 2026-02-03
## GPU: NVIDIA B200

## Results

All 19 workloads × 3 solutions = 57 traces, all PASSED.

| Solution | Speedup range | Typical | Notes |
|----------|--------------|---------|-------|
| flashinfer_moe (trtllm cubin) | 15-48x | ~35x | Baseline best |
| triton_fused_moe_v2 | 5-39x | ~28x | sgl_kernel GEMM |
| cutedsl_moe_fp8_v3 | 6-13x | ~9x | CuTeDSL flat kernel |

## Per-Stage Analysis (T=1024)

From `compare_cutedsl_v2_v3_triton.py`:

| Stage | CD v2 | CD v3 | Triton v2 | v3/v2 | v3/Tr |
|-------|-------|-------|-----------|-------|-------|
| Routing | 0.200 | 0.200 | 0.136 | 1.00x | 1.47x |
| GEMM1 | 0.709 | 0.754 | 0.350 | 1.06x | 2.15x |
| SwiGLU+Req | 0.032 | 0.033 | 0.068 | 1.03x | 0.49x |
| GEMM2 | 0.342 | 0.106 | 0.197 | 0.31x | 0.54x |
| Finalize | 0.032 | 0.044 | 0.050 | 1.35x | 0.88x |
| **E2E** | **1.598** | **1.282** | **0.765** | **0.80x** | **1.68x** |

## Key Findings

1. **v3 GEMM2 is excellent**: 3.2x faster than v2, 1.9x faster than Triton v2 (sgl_kernel)
2. **v3 GEMM1 is the bottleneck**: Python scatter_to_flat_padded overhead makes it 6% slower than v2, 2.15x slower than Triton
3. **SwiGLU is already optimized**: CuTeDSL kernel is 2x faster than Triton's
4. **Routing gap**: CuTeDSL routing (0.200ms) vs Triton (0.136ms) — 47% slower

## Next Optimization Targets (v3.2+)

1. **v3.2: GEMM1 FP8 output** — eliminate `_quantize_output_fp8` round-trip (~0.1-0.2ms)
2. **v3.3: CUDA scatter kernel** — replace Python gather+scatter (~0.3ms savings in GEMM1)
3. **v3.4: Routing cleanup** — reduce Python overhead (~0.05ms)
