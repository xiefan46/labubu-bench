# triton_fused_v2 nsys Profile — 2026-02-02

**GPU:** NVIDIA B200 (SM100)
**Definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Config:** default (19 workloads × ~100 iterations each ≈ 1949 pipeline invocations)

## Command

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --output /root/moe_triton_v2 \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  flashinfer-bench run --local "$FIB_DATASET_PATH" \
    --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --solutions triton_fused_moe_v2 \
    --timeout 600 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85

/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys stats /root/moe_triton_v2.nsys-rep
```

## Pipeline kernel breakdown

Extracted from CUDA GPU Kernel Summary, filtering to kernels belonging to our 6-stage pipeline (identified by ~1949 instances matching pipeline invocation count):

| Stage | Kernel | Instances | Total (ms) | Avg (us) | % of pipeline |
|-------|--------|----------:|-----------:|---------:|--------------:|
| K1 | `moe_fused_gate_kernel` | 1949 | 42.5 | 21.8 | 1.3% |
| K2a | `compute_expert_offsets` | 1949 | 6.2 | 3.2 | 0.2% |
| K2b | `compute_problem_sizes` | 1949 | 25.1 | 12.9 | 0.8% |
| K2c | `compute_arg_sorts` | 1949 | 53.5 | 27.4 | 1.6% |
| K2d | `shuffleRowsKernel<fp8>` (hidden_states) | 1949 | 113.4 | 58.2 | 3.4% |
| K2e | `shuffleRowsKernel<float>` (scales) | 1949 | 41.9 | 21.5 | 1.3% |
| K3 | `GemmUniversal<GroupProblemShape>` (GEMM1) | 1098 | 2410.0 | 2194.9 | 72.8% |
| K3' | `get_group_gemm_starts` (GEMM1 setup) | 2799+1098 | 10.4 | — | 0.3% |
| K4 | **`swiglu_quant_kernel`** (Triton) | 1948 | 286.1 | 146.9 | 8.6% |
| K5 | `GemmUniversal<GroupProblemShape>` (GEMM2) | 2798 | 192.7 | 68.9 | 5.8% |
| K6 | `apply_shuffle_mul_sum_kernel` | 1948 | 119.6 | 61.4 | 3.6% |
| | **Pipeline total** | | **~3310** | | **100%** |

注：GEMM1 和 GEMM2 的 instance 数不同 (1098 vs 2798) 是因为 cutlass grouped GEMM 根据 problem size 选择不同 tile 配置。GEMM1 output 是 [T*8, 4096] (大矩阵)，GEMM2 output 是 [T*8, 7168] (更大)，所以 GEMM2 拆成了更多 kernel instance。

## Key findings

### 1. GEMM 占绝对主导（~79%）

GEMM1 (72.8%) + GEMM2 (5.8%) = **78.6%** 的 pipeline GPU 时间。这解释了为什么 Python-level 优化（buffer 预分配等）在大 batch 时效果微弱。

GEMM1 比 GEMM2 慢很多（2195us vs 69us avg）因为：
- GEMM1: [T*8, 7168] × [E, 7168, 4096] → 大 K 维度
- GEMM2: [T*8, 2048] × [E, 2048, 7168] → 小 K 维度

### 2. Triton swiglu_quant_kernel 只占 8.6%

这意味着即使我们把 Triton kernel 优化 2x，整体也只提升 ~4%。2D tiling 等优化 ROI 有限。

### 3. Scatter/Gather 占 ~10%

- shuffleRows (fp8 + float): 4.7%
- apply_shuffle_mul_sum: 3.6%
- compute_arg_sorts + offsets + sizes: 2.6%

这部分是 sgl_kernel 实现，优化空间有限。

### 4. 非 pipeline 开销巨大

profile 中最大的 kernel 类别是 `unrolled_elementwise_kernel` (25.8%) 和其他 elementwise ops (18.7% + 15.7%)，共 **60.2%** 总 GPU 时间。这些来自：
- Benchmark 框架的 reference 实现运行
- 正确性比较（output 对比）
- Weight transpose `.transpose(1,2)` 和 `.to(torch.float32)` 的隐式 copy

### 5. D2D memcpy 巨量

- 7.6 TB total D2D copy, 2.48 seconds
- 大部分来自 `shuffle_rows`（scatter tokens by expert）和 weight transpose

### 6. cudaStreamSynchronize 占 72% CUDA API 时间

41.4 秒花在 stream sync 上，这是 benchmark 框架 per-iteration 计时导致的（`torch.cuda.synchronize()`），不是我们的 solution overhead。

## Optimization implications

| 优化方向 | 目标 kernel | 占比 | 预期全局提升 | 值得做？ |
|----------|------------|-----:|------------:|---------|
| 缓存 weight transpose | elementwise copy | ~2-3% | 1-2% | 是（简单） |
| Triton kernel 2D tiling | swiglu_quant | 8.6% | 2-4% | 可能 |
| 减少 kernel launch | Python gap | N/A | 5-10% (小batch) | 中等 |
| 优化 GEMM | cutlass grouped GEMM | 78.6% | 大 | 很难（需换实现） |

**结论：在当前架构下（sgl_kernel grouped GEMM + Python 编排），大的性能提升空间已不多。剩余优化主要是缓存 weight transpose（简单）和减少 Python 编排开销（对小 batch 有效）。要大幅追平 flashinfer_moe，需要换到更快的 GEMM 实现或完全 fused kernel。**
