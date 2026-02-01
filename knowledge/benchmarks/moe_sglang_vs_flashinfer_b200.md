# MoE Benchmark: sglang_fp8_blockwise_moe vs flashinfer_moe on B200

**Date:** 2026-02-02
**GPU:** NVIDIA B200 (SM100)
**Definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Tolerance:** rtol=0.2, atol=0.1, required-matched-ratio=0.85

## Results (sorted by seq_len)

Both solutions pass all 19 workloads. `routed_scaling_factor=2.5` for all.

| UUID | seq_len | offset | sglang | flashinfer | ratio |
|----------|--------:|-------:|-------:|-----------:|------:|
| e05c6c03 | 1 | 32 | 15.53x | 32.14x | 0.48 |
| b8f4f012 | 7 | 192 | 20.87x | 44.85x | 0.47 |
| 8cba5890 | 14 | 0 | 13.99x | 34.13x | 0.41 |
| 2e69caee | 15 | 32 | 18.79x | 27.39x | 0.69 |
| a7c2bcfd | 16 | 224 | 19.29x | 31.67x | 0.61 |
| 6230e838 | 32 | 32 | 24.61x | 33.68x | 0.73 |
| f7d6ac7c | 52 | 160 | 21.62x | 34.23x | 0.63 |
| fc378037 | 53 | 32 | 23.74x | 36.76x | 0.65 |
| 76010cb4 | 54 | 128 | 21.09x | 33.67x | 0.63 |
| 81955b1e | 55 | 128 | 20.96x | 32.57x | 0.64 |
| 4822167c | 56 | 64 | 21.39x | 33.50x | 0.64 |
| 74d7ff04 | 57 | 96 | 22.15x | 33.51x | 0.66 |
| e626d3e6 | 58 | 64 | 22.03x | 32.76x | 0.67 |
| eedc63b2 | 59 | 160 | 21.14x | 34.65x | 0.61 |
| 5eadab1e | 62 | 96 | 20.56x | 26.14x | 0.79 |
| 8f1ff9f1 | 80 | 96 | 22.38x | 33.79x | 0.66 |
| 1a4c6ba1 | 901 | 96 | 20.27x | 34.78x | 0.58 |
| 58a34f27 | 11948 | 128 | 3.79x | 15.77x | 0.24 |
| 5e8dc11c | 14107 | 32 | 3.97x | 15.88x | 0.25 |

## Workload distribution

- **Small batch (seq_len 1-32):** 6 workloads
- **Medium batch (seq_len 50-80):** 10 workloads
- **Large batch (seq_len 901):** 1 workload
- **Very large batch (seq_len 11948-14107):** 2 workloads
- **local_expert_offset** ranges from 0 to 224 (covers different EP ranks)
- **routed_scaling_factor** = 2.5 for all (DeepSeek-V3 default)

## Performance analysis

### Overall
- **flashinfer_moe** average: ~32x speedup
- **sglang_fp8_blockwise_moe** average: ~19x speedup
- FlashInfer is ~1.7x faster on average

### By seq_len regime
- **Small/Medium (seq_len <= 80):** sglang ~20x, flashinfer ~33x (ratio ~0.6)
- **Large (seq_len ~900):** sglang 20x, flashinfer 35x (ratio 0.58)
- **Very large (seq_len >10000):** sglang ~4x, flashinfer ~16x (ratio ~0.25)

The performance gap widens significantly at large seq_len, where sglang drops to ~4x while flashinfer maintains ~16x. This suggests the sglang solution's Python-level overhead (routing, quantization, memory copies) scales poorly with token count.

### Why flashinfer_moe is faster
- Fully fused CUDA kernel: routing + GEMM + SwiGLU + output accumulation in one kernel launch
- No intermediate memory materialization between stages
- Uses pre-compiled trtllm cubin kernels optimized for B200/SM100

### sglang_fp8_blockwise_moe bottlenecks
- Python-level routing (`moe_fused_gate`) is a separate kernel launch
- `torch.cat` to swap SwiGLU halves causes extra memory copy of [m*topk, 2*I] tensor
- PyTorch `_per_token_group_quant_fp8` for intermediate quantization (non-fused)
- Multiple separate kernel launches: prepare_moe_input, shuffle_rows, GEMM1, silu_and_mul, quant, GEMM2, apply_shuffle_mul_sum
- 1GB workspace allocation is wasteful (SGLang uses 90KB)

### Potential optimizations for sglang solution
1. Replace `torch.cat` swap with manual SwiGLU in PyTorch (avoid copy)
2. Use `sgl_kernel.per_token_group_quant_fp8` instead of custom PyTorch implementation
3. Reduce workspace from 1GB to 90KB
4. Pre-allocate buffers to avoid repeated allocation per call
5. For very large seq_len: the intermediate tensor [m*topk, 2*I] becomes huge; fusing SwiGLU+quant would help most
