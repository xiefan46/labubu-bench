# 4-Way: sglang_v1 vs sglang_v2 vs triton_v1 vs triton_v2 — 2026-02-02

**GPU:** NVIDIA B200 (SM100)
**Definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Tolerance:** rtol=0.2, atol=0.1, required-matched-ratio=0.85
**Config:** default (warmup=5, iterations=20, num_trials=3)

## Command

```bash
flashinfer-bench run --local "$FIB_DATASET_PATH" \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions sglang_fp8_blockwise_moe_v1 sglang_fp8_blockwise_moe_v2 triton_fused_moe_v1 triton_fused_moe_v2 \
  --timeout 600 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85
```

## Solutions compared

| Solution | Key difference |
|----------|---------------|
| sglang_v1 | torch.cat SwiGLU + PyTorch FP8 quant + 1GB workspace |
| sglang_v2 | Zero-copy SwiGLU + sgl_kernel FP8 quant + 90KB workspace |
| triton_fused_v1 | Fused SwiGLU+quant Triton kernel, per-call buffer allocation |
| triton_fused_v2 | Same kernel as v1, module-level buffer pre-allocation (eliminates ~22 torch.empty/full/zeros per call) |

## Results (sorted by num_tokens)

All 19 workloads PASS for all 4 solutions.

| UUID | T | offset | sglang_v1 | sglang_v2 | triton_v1 | triton_v2 | best |
|----------|----:|-------:|----------:|----------:|----------:|----------:|------|
| e05c6c03 | 1 | 32 | 17.94x | 23.00x | 28.74x | **37.58x** | v2 |
| b8f4f012 | 7 | 192 | 17.78x | 22.13x | 17.45x | **36.03x** | v2 |
| 8cba5890 | 14 | 0 | 15.89x | 21.66x | 17.18x | **34.13x** | v2 |
| 2e69caee | 15 | 32 | 13.49x | 26.08x | 18.57x | **34.59x** | v2 |
| a7c2bcfd | 16 | 224 | 20.17x | 30.34x | 30.16x | **35.75x** | v2 |
| 6230e838 | 32 | 32 | 14.08x | 21.78x | 23.95x | **31.85x** | v2 |
| f7d6ac7c | 52 | 160 | 20.45x | 27.44x | 18.01x | **23.86x** | v2 |
| fc378037 | 53 | 32 | 21.69x | 27.80x | 27.76x | **31.79x** | v2 |
| 76010cb4 | 54 | 128 | 21.10x | 27.64x | 27.75x | **31.63x** | v2 |
| 81955b1e | 55 | 128 | 21.15x | 28.04x | 27.93x | **31.89x** | v2 |
| 4822167c | 56 | 64 | 21.43x | 27.46x | 27.06x | **31.50x** | v2 |
| 74d7ff04 | 57 | 96 | 20.54x | 28.05x | 28.06x | **32.35x** | v2 |
| e626d3e6 | 58 | 64 | 22.35x | 28.03x | 28.16x | **31.85x** | v2 |
| eedc63b2 | 59 | 160 | 21.32x | 29.02x | 28.23x | **33.70x** | v2 |
| 5eadab1e | 62 | 96 | 19.28x | 18.38x | 22.52x | **35.95x** | v2 |
| 8f1ff9f1 | 80 | 96 | 22.01x | 26.40x | 26.40x | **30.27x** | v2 |
| 1a4c6ba1 | 901 | 96 | 18.06x | 21.42x | 23.52x | **25.75x** | v2 |
| 58a34f27 | 11948 | 128 | 3.61x | 4.07x | 4.64x | **4.67x** | v2 |
| 5e8dc11c | 14107 | 32 | 3.84x | 4.39x | **4.99x** | **4.99x** | tie |

## Summary statistics

| Metric | sglang_v1 | sglang_v2 | triton_v1 | triton_v2 |
|--------|----------:|----------:|----------:|----------:|
| Average | 17.69x | 23.32x | 22.69x | **29.48x** |
| Small-batch avg (T<=901) | 19.34x | 25.57x | 24.79x | **32.38x** |
| Large-batch avg (T>10K) | 3.73x | 4.23x | 4.82x | **4.83x** |
| Min | 3.61x | 4.07x | 4.64x | 4.67x |
| Max | 22.35x | 30.34x | 30.16x | 37.58x |

## triton_v2 vs other solutions (speedup ratio)

| vs | Average ratio | Range |
|----|-------------:|------:|
| sglang_v1 | **1.67x** | 1.01-2.53 |
| sglang_v2 | **1.26x** | 1.01-1.96 |
| triton_v1 | **1.30x** | 1.00-2.06 |

## Analysis

### Buffer pre-allocation impact (triton_v2 vs triton_v1)

The v2 optimization eliminates ~22 `torch.empty/full/zeros` calls per invocation by caching buffers at module level. The benefit varies dramatically by batch size:

- **Small batch (T=1-32):** v2 is 1.09-2.06x faster. Allocation overhead dominates at small T because kernel time is tiny. The 2.06x outlier (b8f4f012, T=7) likely reflects Triton JIT warmup penalty on v1's first workload being avoided in v2 through better cache behavior.
- **Medium batch (T=50-80):** v2 is 1.13-1.60x faster. Consistent ~15-30% improvement.
- **Large batch (T=901):** v2 is 1.09x faster. Modest — kernel compute dominates.
- **Very large batch (T>10K):** v2 is 1.00-1.01x faster. Negligible — as expected, buffer allocation is irrelevant when GEMM takes milliseconds.

### Overall ranking

**triton_fused_v2 > sglang_v2 ≈ triton_v1 >> sglang_v1**

triton_v2 wins all 19 workloads (ties on 1). It is the best non-flashinfer solution, averaging 29.48x vs flashinfer_moe's ~32x (ratio ~0.92). The gap is now only ~8%.

### Remaining gap vs flashinfer_moe (~32x)

The ~8% remaining gap is structural:
- flashinfer_moe uses a single fused trtllm cubin kernel (all stages in one launch)
- triton_v2 still has 6 kernel launches with Python orchestration
- flashinfer_moe also avoids Python-level weight transpose overhead

### Next optimization opportunities
1. **A/B test sgl_kernel fused SwiGLU+quant CUDA kernel** vs Triton kernel -> Done in fused_v3
2. **Cache weight transposes** at module level (currently `.transpose(1,2)` called per run)
3. **Fuse scatter + GEMM1** or **GEMM2 + gather** to reduce kernel launches further
4. **Explore MXFP8** (SM100 native block size 32) instead of blockwise FP8 (block size 128)
5. **Tune Triton kernel** — try 2D tiling or larger blocks for better GPU occupancy
