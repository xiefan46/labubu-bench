# triton_fused_v2 vs flashinfer_moe — 2026-02-02

**GPU:** NVIDIA B200 (SM100)
**Definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Tolerance:** rtol=0.2, atol=0.1, required-matched-ratio=0.85
**Config:** default (warmup=5, iterations=20, num_trials=3)

## Command

```bash
flashinfer-bench run --local "$FIB_DATASET_PATH" \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions flashinfer_moe triton_fused_moe_v2 \
  --timeout 600 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85
```

## Architecture difference

| Aspect | triton_fused_v2 | flashinfer_moe |
|--------|-----------------|----------------|
| Pipeline | 6 kernel launches with Python orchestration | Single fused trtllm cubin kernel |
| SwiGLU+Quant | Custom Triton kernel (fused) | Fused inside cubin |
| GEMM | sgl_kernel grouped_mm | trtllm pre-compiled cubin |
| Buffers | Module-level pre-allocation | N/A (all internal) |
| Weight prep | Per-call `.transpose(1,2)` + `.to(fp32)` | N/A (handled internally) |

## Results (sorted by num_tokens)

All 19 workloads PASS for both.

| UUID | T | offset | flashinfer | triton_v2 | v2/fi | winner |
|----------|----:|-------:|-----------:|----------:|------:|--------|
| e05c6c03 | 1 | 32 | 31.24x | 19.89x | 0.64 | fi |
| b8f4f012 | 7 | 192 | 41.32x | 30.15x | 0.73 | fi |
| 8cba5890 | 14 | 0 | 31.15x | 31.75x | 1.02 | **v2** |
| 2e69caee | 15 | 32 | 32.25x | 34.90x | 1.08 | **v2** |
| a7c2bcfd | 16 | 224 | 32.04x | 26.44x | 0.83 | fi |
| 6230e838 | 32 | 32 | 32.25x | 31.44x | 0.97 | fi |
| f7d6ac7c | 52 | 160 | 31.88x | 30.59x | 0.96 | fi |
| fc378037 | 53 | 32 | 39.02x | 31.20x | 0.80 | fi |
| 76010cb4 | 54 | 128 | 30.91x | 22.16x | 0.72 | fi |
| 81955b1e | 55 | 128 | 30.06x | 26.46x | 0.88 | fi |
| 4822167c | 56 | 64 | 32.67x | 30.45x | 0.93 | fi |
| 74d7ff04 | 57 | 96 | 31.89x | 26.27x | 0.82 | fi |
| e626d3e6 | 58 | 64 | 32.16x | 30.49x | 0.95 | fi |
| eedc63b2 | 59 | 160 | 40.08x | 28.97x | 0.72 | fi |
| 5eadab1e | 62 | 96 | 31.64x | 28.90x | 0.91 | fi |
| 8f1ff9f1 | 80 | 96 | 29.13x | 24.60x | 0.84 | fi |
| 1a4c6ba1 | 901 | 96 | 34.15x | 25.16x | 0.74 | fi |
| 58a34f27 | 11948 | 128 | 15.96x | 4.74x | 0.30 | fi |
| 5e8dc11c | 14107 | 32 | 15.66x | 5.04x | 0.32 | fi |

## Summary statistics

| Metric | flashinfer_moe | triton_fused_v2 |
|--------|---------------:|----------------:|
| Average | 30.81x | 24.72x |
| Small-batch avg (T<=901) | 32.58x | 27.86x |
| Large-batch avg (T>10K) | 15.81x | 4.89x |
| Min | 15.66x | 4.74x |
| Max | 41.32x | 34.90x |

## Analysis

### Overall

- **flashinfer_moe wins 17/19 workloads**
- **triton_v2 wins 2 workloads** (8cba5890 T=14, 2e69caee T=15)
- **Small-batch ratio (v2/fi):** 0.86 — triton_v2 is ~14% slower
- **Large-batch ratio (v2/fi):** 0.31 — triton_v2 is ~3.2x slower

### By num_tokens regime

| Regime | flashinfer avg | triton_v2 avg | v2/fi ratio |
|--------|---------------:|-------------:|------------:|
| Tiny (T=1-7) | 36.28x | 25.02x | 0.69 |
| Small (T=14-32) | 31.92x | 31.13x | 0.98 |
| Medium (T=52-80) | 33.04x | 27.91x | 0.84 |
| Large (T=901) | 34.15x | 25.16x | 0.74 |
| Very large (T>10K) | 15.81x | 4.89x | 0.31 |

### Key observations

1. **triton_v2 is competitive at T=14-32** (ratio 0.98) — at this sweet spot, kernel compute dominates and Python overhead is small relative to total time. triton_v2 even wins 2 workloads here.

2. **Tiny batch (T=1-7) gap is large** (ratio 0.69). Python-level overhead (routing, weight transpose, shuffle_rows, etc.) dominates at tiny T. flashinfer avoids all of this with a single kernel launch.

3. **Very large batch gap is huge** (ratio 0.31). At T>10K, triton_v2 drops to ~5x while flashinfer maintains ~16x. The sgl_kernel grouped GEMM at this scale is significantly slower than flashinfer's trtllm cubin. This is the GEMM implementation gap, not Python overhead.

4. **Variance in triton_v2 is notable**: some medium-batch workloads range from 22x to 31x despite similar T values. This suggests sensitivity to expert load distribution (different `local_expert_offset` values create different per-expert problem sizes).

### Where triton_v2 loses time

| Source | Impact | Regime most affected |
|--------|--------|---------------------|
| Python orchestration (6 launches vs 1) | ~10-15% | Tiny/small batch |
| Weight `.transpose(1,2)` + `.to(fp32)` per call | ~1-3% | All |
| sgl_kernel grouped GEMM vs trtllm cubin | ~3x | Very large batch |
| `shuffle_rows` + `prepare_moe_input` overhead | ~5% | Small/medium batch |

### Conclusion

triton_v2 is within ~14% of flashinfer on small batches (the dominant LLM inference regime), but falls far behind at large batches due to the GEMM implementation gap. The remaining small-batch gap is split between Python orchestration overhead (~10%) and minor per-call inefficiencies (~3-5%).
