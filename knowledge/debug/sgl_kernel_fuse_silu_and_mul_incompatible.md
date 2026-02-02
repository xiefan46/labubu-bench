# sgl_kernel fuse_silu_and_mul 与 blockwise grouped GEMM 不兼容

**Date:** 2026-02-02
**Attempted in:** sglang_fused_moe_v3 (已删除)

## 背景

尝试用 `sgl_per_token_group_quant_8bit_v2(fuse_silu_and_mul=True)` 替换自定义 Triton swiglu_quant_kernel，以 A/B 测试 sgl_kernel 的 CUDA 实现是否更快。

## 结果

**19/19 INCORRECT_NUMERICAL** — 全部失败。

## 根因

`sgl_per_token_group_quant_8bit_v2` 的 CUDA dispatch 逻辑（`per_token_group_quant_8bit_v2.cu` line 461-478）：

```cpp
if (is_column_major) {
  if (scale_ue8m0) {
    if (fuse_silu_and_mul) {
      // ← 实际的 fused SwiGLU+Quant 路径
    }
  } else {
    // float scales, column_major — 无 fuse 分支
  }
} else {
  // row_major — 无 fuse 分支，fuse_silu_and_mul 被静默忽略
}
```

**`fuse_silu_and_mul=True` 只在 `scale_ue8m0=True` + `column_major_scales=True` 时才生效。** 否则 SwiGLU 融合被静默跳过，c1 原始数据被直接量化。

## 为什么不能加上 scale_ue8m0=True

即使加上这两个参数，输出的 scale 格式是 **UE8M0**（unsigned 8-bit exponent），而下游的 `fp8_blockwise_scaled_grouped_mm` 需要 **float32** scale。格式不兼容，无法作为 drop-in 替换。

## 测试用例的佐证

`sgl-kernel/tests/test_per_token_group_quant_8bit.py` 中，所有 `fuse_silu_and_mul=True` 的 config 都搭配了：
```python
column_major_scales=True,
scale_tma_aligned=True,
scale_ue8m0=True,
```

没有任何 `fuse_silu_and_mul=True` + `scale_ue8m0=False` 的组合。

## 结论

`sgl_kernel` 的 fused SwiGLU+Quant 路径是为 **MXFP8 GEMM**（如 cuBLAS MXFP8 path）设计的，scale 格式（UE8M0 + column-major）与我们使用的 `fp8_blockwise_scaled_grouped_mm`（float32 row-major scales）不兼容。

**这条优化路线走不通。** 保留自定义 Triton kernel 方案（triton_fused_v2）。

## 后续方向

继续基于 Triton kernel 优化：
1. 缓存 weight transpose + dtype 转换
2. Triton kernel 2D tiling 改善 SM 利用率
3. nsys profiling 确认各 kernel 耗时占比
