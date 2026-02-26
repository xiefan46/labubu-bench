# CuTeDSL FP8 MoE 可行性分析

**Date:** 2026-02-02
**Goal:** 纯 CuTeDSL 实现 DeepSeek FP8 MoE，严格匹配 trtllm 结果

## 核心发现

### 1. trtllm 的 FP8 GEMM 不使用硬件 block-scale MMA

trtllm cubin 使用 **software scale application**（非 MX 格式硬件指令）：
- FP8×FP8 → FP32 accumulation（标准 MMA，非 `tcgen05.mma.kind.block_scale`）
- float32 per-128-block scales 在 epilogue 中 software 应用
- `deepSeekFp8 = true` flag 控制 software dequant 行为

### 2. FlashInfer CuTeDSL `grouped_gemm_nt_masked` 不支持 DeepSeek scale 格式

| 参数 | CuTeDSL 支持 | DeepSeek 需要 |
|------|-------------|--------------|
| ab_dtype | Float8E4M3FN ✅ | Float8E4M3FN ✅ |
| sf_dtype | Float8E8M0FNU / Float8E4M3FN ❌ | **float32** ❌ |
| sf_vec_size | 16 或 32 ❌ | **128** ❌ |
| Scale layout | CuTe atom `(l, rm, rk, 32, 4, 4)` | MN-major `[N//128, M]` |

硬性限制在 `blockscaled_gemm.py:2234`：sf_dtype 只接受 Float8 类型。

### 3. FlashInfer 已有 C++ CUTLASS 实现完全匹配 DeepSeek 需求

**`group_gemm_fp8_nt_groupwise`** (`include/flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh`):
- FP8_E4M3 × FP8_E4M3 → float32 accumulation → BF16 output
- **float32 scales**，block_size=128
- 支持 MN-major 和 K-major scale layout
- Grouped GEMM（每个 expert 独立 problem shape）
- SM100 (Blackwell) 专用
- 使用 `Sm100BlockwiseScaleConfig` + `KernelPtrArrayTmaWarpSpecializedBlockwise{1,2}SmSm100` scheduler
- Python API: `flashinfer.gemm.group_gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, m_indptr, scale_major_mode, out_dtype)`

另外还有 **`batch_deepgemm_fp8_nt_groupwise`**：
- 支持 masked_m（每个 expert 不同 token 数）
- 等价于 CuTeDSL `grouped_gemm_nt_masked` 的 masked_m 功能

### 4. SM100 FP8 MMA 两种模式

| 模式 | MMA 指令 | Scale 格式 | CuTeDSL 实现 | C++ 实现 |
|------|---------|-----------|-------------|---------|
| 硬件 block-scale | `tcgen05.mma.kind.block_scale` | E8M0, block=32 (MX) | `blockscaled_gemm.py` ✅ | N/A |
| Software scale | `tcgen05.mma` (标准 f8f6f4) | float32, block=128 | `gemm_allreduce_two_shot.py` (无 group) | `group_gemm_fp8_groupwise_sm100.cuh` ✅ |

## 方案选择

### 方案 A：混合架构（C++ GEMM + CuTeDSL/Python 辅助 kernels）⭐ 推荐

直接使用已有的 `group_gemm_fp8_nt_groupwise` / `batch_deepgemm_fp8_nt_groupwise` 做 GEMM1 和 GEMM2，
围绕它构建 MoE pipeline：

- Routing: 需要实现 DeepSeek-V3 routing（sigmoid + bias + grouped top-k）
- Scatter: token reorder（可用 Python/Triton/CUDA）
- GEMM1: `group_gemm_fp8_nt_groupwise` → BF16 output
- SwiGLU + FP8 requant: 需要自定义 kernel（CuTeDSL 或 Triton）
- GEMM2: `group_gemm_fp8_nt_groupwise` → BF16 output
- Finalize: gather + weighted reduce

**优点**: GEMM 精度严格匹配（与 trtllm 使用相同的 CUTLASS blockwise scale 机制），开发量小
**缺点**: 不是"纯 CuTeDSL"，GEMM 部分是 C++ CUTLASS

### 方案 B：纯 CuTeDSL（需要新写 GEMM kernel）

以 `gemm_allreduce_two_shot.py` 为基础，改造为支持 float32 scale 的 grouped GEMM：

1. 使用 `make_trivial_tiled_mma()`（标准 FP8 MMA，非 block-scale）
2. 添加 float32 scale tensor 的 TMA 加载
3. 在 epilogue 中 apply scale（类似 C++ 版本的 `Sm100BlockwiseScaleConfig`）
4. 添加 grouped GEMM 支持（多 expert，masked_m）

**优点**: 纯 Python CuTeDSL，可灵活定制 epilogue（fuse activation 等）
**缺点**: 工作量大，需要深入理解 CUTLASS 3.x persistent scheduling + TMA + TMEM

### 方案 C：CuTeDSL GEMM + FP8→MX 格式转换

在 GEMM 之前把 DeepSeek 的 (FP8, float32 scale, block=128) 转换为 MX 的 (FP8, E8M0 scale, block=32)。

**已排除**: 用户要求严格匹配 trtllm 结果，MX 格式转换会引入精度损失。

## 关键 API 参考

### C++ CUTLASS（已有，可直接用）

```python
from flashinfer.gemm import group_gemm_fp8_nt_groupwise, batch_deepgemm_fp8_nt_groupwise

# Grouped GEMM: 所有 expert 的 token 拼接
out = group_gemm_fp8_nt_groupwise(
    a_fp8,       # [total_tokens, K] FP8
    b_fp8,       # [num_experts, N, K] FP8
    a_scale,     # [K//128, total_tokens] float32 (MN-major)
    b_scale,     # [num_experts, N//128, K//128] float32 (MN-major)
    m_indptr,    # [num_experts+1] int32, token offsets
    scale_major_mode="MN",
    out_dtype=torch.bfloat16,
)

# Batched GEMM with masked_m: 每个 expert 独立 batch
out = batch_deepgemm_fp8_nt_groupwise(
    a_fp8,       # [num_experts, max_m, K] FP8
    b_fp8,       # [num_experts, N, K] FP8
    a_scale,     # [num_experts, max_m, K//128] float32 (K-major)
    b_scale,     # [num_experts, N//128, K//128] float32
    masked_m,    # [num_experts] int32
    expected_m,  # int
    out_dtype=torch.bfloat16,
)
```

### CuTeDSL（现有，仅 MX 格式）

```python
from flashinfer.cute_dsl import grouped_gemm_nt_masked

# 仅支持 MX 格式 (E8M0 scale, block=32)，不支持 DeepSeek float32 scale
out = grouped_gemm_nt_masked(
    lhs=(a, sfa), rhs=(b, sfb), out=c, masked_m=masked_m,
    ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
    sf_vec_size=32, c_dtype="bfloat16",
)
```
