# flashinfer_moe nsys Profile — 2026-02-02

**GPU:** NVIDIA B200 (SM100)
**Definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Config:** default (19 workloads x ~99 iterations each = ~1883 pipeline invocations)

## Command

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --output /root/moe_flashinfer \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  flashinfer-bench run --local "$FIB_DATASET_PATH" \
    --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --solutions flashinfer_moe \
    --timeout 600 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85

/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys stats /root/moe_flashinfer.nsys-rep
```

## Pipeline kernel identification

`routingMainKernel` has **1883 instances**, matching the pipeline invocation count. Other pipeline kernels are identified by instance counts summing to 1883.

flashinfer_moe uses **trtllm cubin kernels** for GEMM, with different tile configurations selected per batch size (549 + 1151 + 183 = 1883).

## Pipeline kernel breakdown

### GEMM1 (FP8 in -> FP8 out, `bmm_E4m3_E4m3E4m3_Fp32`)

| Tile config | Instances | Total (ms) | Avg (us) |
|-------------|----------:|-----------:|---------:|
| t128x64x128 | 549 | 351.6 | 640.5 |
| t128x8x128 | 1151 | 75.8 | 65.9 |
| t128x16x128 | 183 | 27.5 | 150.0 |
| **Subtotal** | **1883** | **454.9** | |

### GEMM2 (FP8 in -> BF16 out, `bmm_Bfloat16_E4m3E4m3_Fp32`)

| Tile config | Instances | Total (ms) | Avg (us) |
|-------------|----------:|-----------:|---------:|
| t128x64x128 | 549 | 218.2 | 397.4 |
| t128x8x128 | 1151 | 38.3 | 33.3 |
| t128x16x128 | 183 | 13.1 | 71.7 |
| **Subtotal** | **1883** | **269.6** | |

### SwiGLU activation (`activationDeepSeekKernel`)

Three template instantiations for different problem sizes:

| Variant | Instances | Total (ms) | Avg (us) |
|---------|----------:|-----------:|---------:|
| Large (FP8 output) | 785 | 310.2 | 395.2 |
| Small | 915 | 6.2 | 6.8 |
| Medium | 183 | 1.6 | 8.7 |
| **Subtotal** | **1883** | **318.0** | |

### Routing

| Kernel | Instances | Total (ms) | Avg (us) |
|--------|----------:|-----------:|---------:|
| `routingMainKernel` | 1883 | 20.7 | 11.0 |
| `routingIndicesClusterKernel` | 1517 | 5.5 | 3.6 |
| `routingIndicesCoopKernel` | 366 | 2.0 | 5.5 |
| **Subtotal** | | **28.2** | |

### Finalize (gather + weighted reduce)

| Kernel | Instances | Total (ms) | Avg (us) |
|--------|----------:|-----------:|---------:|
| `finalizeKernelVecLoad` | 549 | 49.6 | 90.3 |
| `finalizeKernel` | 1098 | 6.1 | 5.6 |
| **Subtotal** | **1647** | **55.7** | |

Note: finalize instance count (1647) < 1883 because some small-batch workloads may use a different finalize path or skip vectorized loads.

### Pipeline summary

| Stage | Time (ms) | % of pipeline |
|-------|----------:|--------------:|
| GEMM1 (FP8->FP8) | 454.9 | 40.4% |
| SwiGLU (activation) | 318.0 | 28.2% |
| GEMM2 (FP8->BF16) | 269.6 | 23.9% |
| Finalize (gather) | 55.7 | 4.9% |
| Routing | 28.2 | 2.5% |
| **Pipeline total** | **~1126** | **100%** |

## Head-to-head comparison with triton_v2

| Stage | triton_v2 (ms) | flashinfer (ms) | flashinfer speedup |
|-------|---------------:|----------------:|-------------------:|
| GEMM total | 2602.7 | 724.5 | **3.6x** |
| SwiGLU | 286.1 | 318.0 | **0.9x** (triton faster) |
| Routing + Scatter | 282.6 | 28.2 | **10x** |
| Gather / Finalize | 119.6 | 55.7 | **2.1x** |
| **Pipeline total** | **~3310** | **~1126** | **2.9x** |

## Key findings

### 1. GEMM is the dominant gap (3.6x)

trtllm cubin FP8 grouped BMM kernels are 3.6x faster than sgl_kernel's cutlass grouped GEMM. This is the single biggest reason flashinfer_moe outperforms triton_v2. The cubin kernels are pre-compiled and heavily optimized for SM100, while sgl_kernel uses a more general cutlass GroupProblemShape dispatch.

GEMM breakdown comparison:
- **triton_v2 GEMM1:** 2410 ms (72.8% of pipeline) — cutlass grouped GEMM with large K=7168
- **flashinfer GEMM1:** 454.9 ms (40.4%) — trtllm cubin with adaptive tile selection
- **triton_v2 GEMM2:** 192.7 ms (5.8%)
- **flashinfer GEMM2:** 269.6 ms (23.9%) — flashinfer's GEMM2 is actually slower in share-of-pipeline because GEMM1 is so much faster

### 2. Our Triton SwiGLU kernel beats flashinfer's CUDA kernel

triton_v2's `swiglu_quant_kernel` (286.1 ms) is **10% faster** than flashinfer's `activationDeepSeekKernel` (318.0 ms). This validates that our Triton kernel is well-optimized — further Triton kernel tuning has diminishing returns.

### 3. Routing is 10x more compact in flashinfer

triton_v2 uses 6 separate kernels for routing (gate + offsets + sizes + sort + shuffle_fp8 + shuffle_float = 282.6 ms). flashinfer uses 3 tightly integrated routing kernels totaling just 28.2 ms. The difference comes from:
- flashinfer fuses gate computation with index generation
- No separate scatter/shuffle step (data movement is integrated into GEMM dispatch)
- CUB-based operations for efficient parallel prefix sums

### 4. GEMM share differs significantly

In triton_v2, GEMM is 78.6% of pipeline time. In flashinfer, GEMM is 64.3%. Because flashinfer's GEMM is so efficient, SwiGLU becomes the second-largest cost (28.2%), suggesting that for flashinfer, SwiGLU optimization would yield more gains than for triton_v2.

### 5. Non-pipeline overhead dominates total GPU time

The largest kernels by total time are framework overhead:
- `unrolled_elementwise_kernel` (copy): 13.9 s (26.9%)
- `elementwise_kernel` (comparison): 10.1 s (19.5%)
- `vectorized_elementwise_kernel` (arithmetic): 8.5 s (16.4%)
- `cutlass3x_sm100_simt_sgemm` (FP32 reference GEMM): ~10 s total

These are all from the benchmark framework's reference implementation and correctness comparison, not from the flashinfer_moe solution.

## Optimization implications for triton_v2

Based on the comparison:

| Optimization target | Gap | Potential pipeline improvement | Feasibility |
|--------------------|-----|-------------------------------|-------------|
| Replace GEMM with trtllm cubin | 3.6x on GEMM (1878 ms gap) | ~57% faster pipeline | Hard (proprietary cubin) |
| Fuse routing kernels | 10x on routing (254 ms gap) | ~8% faster pipeline | Medium |
| SwiGLU kernel | Already faster than flashinfer | None | N/A |
| Fuse finalize | 2.1x on finalize (64 ms gap) | ~2% faster pipeline | Low priority |

**Conclusion:** The performance gap between triton_v2 and flashinfer_moe is overwhelmingly driven by GEMM efficiency (trtllm cubin vs cutlass grouped GEMM). Our Triton SwiGLU kernel is already competitive. Routing fusion offers moderate (~8%) gains. To close the gap significantly, we would need access to faster FP8 grouped GEMM implementations.
