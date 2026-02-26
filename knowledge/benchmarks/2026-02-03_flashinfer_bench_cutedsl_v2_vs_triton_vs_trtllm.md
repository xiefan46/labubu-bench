# flashinfer-bench: CuTeDSL v2 vs Triton v2 vs flashinfer_moe (trtllm) (2026-02-03)

## Summary

Ran all 19 MoE workloads on B200 via `flashinfer-bench run` with three solutions.
All 57 traces PASSED (rtol=0.2, atol=0.1, required-matched-ratio=0.85).

**flashinfer_moe (trtllm cubin C++ pipeline) is the clear winner** at 30-60x speedup.
CuTeDSL v2 lags at 8-13x — roughly 1/4 of trtllm and 1/3 of Triton v2.

## Command

```bash
flashinfer-bench run --local ~/labubu-bench/flashinfer-trace \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions cutedsl_moe_fp8_v2 triton_fused_moe_v2 flashinfer_moe \
  --timeout 1200 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85
```

## Results (speedup vs reference)

| Workload | CuTeDSL v2 | flashinfer_moe (trtllm) | Triton v2 |
|----------|-----------|------------------------|-----------|
| b8f4f012 | 8.38x | **30.47x** | 25.94x |
| e05c6c03 | 7.17x | **46.05x** | 35.77x |
| 6230e838 | 9.10x | **38.75x** | 29.84x |
| 8f1ff9f1 | 9.70x | 30.58x | **32.73x** |
| 1a4c6ba1 | 13.42x | **33.43x** | 26.61x |
| a7c2bcfd | 9.68x | **53.29x** | 43.22x |
| 2e69caee | 8.42x | **59.94x** | 41.08x |
| 8cba5890 | 9.23x | **54.77x** | 42.04x |
| 5e8dc11c | 9.62x | **15.74x** | 5.00x |
| 58a34f27 | 10.46x | **16.27x** | 4.67x |
| 5eadab1e | 8.63x | **38.54x** | 31.47x |
| eedc63b2 | 8.62x | **38.06x** | 31.15x |
| e626d3e6 | 10.23x | **37.39x** | 30.41x |
| e626d3e6 | 10.31x | **37.74x** | 31.09x |
| 4822167c | 10.25x | **37.57x** | 30.35x |
| 81955b1e | 10.09x | **37.75x** | 30.64x |
| 76010cb4 | 9.09x | **37.55x** | 30.59x |
| fc378037 | 9.17x | **37.38x** | 30.59x |
| f7d6ac7c | 9.34x | **38.99x** | 30.52x |

## Key Observations

1. **flashinfer_moe (trtllm)** wins on 18/19 workloads. Pre-compiled cubin GEMMs + fused scatter into GEMM1 + C++ pipeline with zero Python overhead.

2. **Triton v2** is 2nd place on most workloads (26-43x), using sgl_kernel's `fp8_blockwise_scaled_grouped_mm` with flat `[total_M, K]` input.

3. **CuTeDSL v2** is 3rd at 8-13x. The gap vs our manual benchmark (1.8x vs Triton) is larger here (~3x) because:
   - Real workloads have more experts activated (not just 2/32 like our biased routing benchmark)
   - More active experts → larger `max_M` padding → more wasted bandwidth in `[E, max_M, K]` batched layout
   - Scatter/gather overhead scales with total_M

4. **Small-batch advantage**: On workloads `5e8dc11c` and `58a34f27`, CuTeDSL v2 (9.6x, 10.5x) beats Triton v2 (5.0x, 4.7x) by ~2x. These are likely small seq_len workloads where Triton's kernel launch overhead dominates and CuTeDSL's persistent kernel + fused SwiGLU pays off.

5. **Triton v2 occasionally beats trtllm**: On workload `8f1ff9f1`, Triton (32.73x) edges out trtllm (30.58x).

## Gap Analysis: CuTeDSL v2 vs trtllm (~3-4x gap)

| Source | Impact | Fix |
|--------|--------|-----|
| Batched `[E, max_M, K]` padding + scatter/gather | High | Phase 2: flat-input kernel with m_indptr |
| Python-level pipeline (5 separate kernel launches) | Medium | Fuse into C++ runner like trtllm |
| GEMM1 outputs BF16 then requantizes to FP8 | Medium | Fuse FP8 requant into GEMM epilogue |
| Scatter not fused into GEMM1 | Medium | Add routeAct-style A-gather in kernel |
| CuTeDSL kernel vs trtllm cubin (micro-arch tuning) | Low-Medium | Tile size tuning, cluster shapes |

## Next Steps

- Phase 2: flat-input kernel (eliminate padding, match trtllm data layout)
- Profile real workload distribution to understand expert activation patterns
- Consider fusing scatter into GEMM1 (trtllm's `routeAct=true` approach)
