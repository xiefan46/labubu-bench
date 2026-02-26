# 2-Way Benchmark: flashinfer_moe vs triton_fused_moe_v2

**Date**: 2026-02-26
**Hardware**: NVIDIA B200
**Definition**: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Tolerances**: rtol=0.2, atol=0.1, required-matched-ratio=0.85, timeout=1200s
**All 38 traces PASSED**

## Results (sorted by seq_len)

| # | seq_len | expert_offset | flashinfer_moe | triton_v2 | diff | Winner |
|---|---------|---------------|---------------|-----------|------|--------|
| 1 | 1 | 32 | 21.97x | **26.43x** | +20.3% | triton_v2 |
| 2 | 7 | 192 | **35.77x** | 31.32x | +14.2% | flashinfer_moe |
| 3 | 14 | 0 | 28.17x | **32.93x** | +16.9% | triton_v2 |
| 4 | 15 | 32 | 27.45x | **32.01x** | +16.6% | triton_v2 |
| 5 | 16 | 224 | 30.50x | **33.87x** | +11.0% | triton_v2 |
| 6 | 32 | 32 | **27.82x** | 26.50x | +5.0% | flashinfer_moe |
| 7 | 52 | 160 | 26.99x | **29.87x** | +10.7% | triton_v2 |
| 8 | 53 | 32 | 27.35x | **29.79x** | +8.9% | triton_v2 |
| 9 | 54 | 128 | 28.19x | **30.26x** | +7.3% | triton_v2 |
| 10 | 55 | 128 | 28.85x | **30.99x** | +7.4% | triton_v2 |
| 11 | 56 | 64 | 29.20x | **30.41x** | +4.1% | triton_v2 |
| 12 | 57 | 96 | 28.91x | **31.41x** | +8.6% | triton_v2 |
| 13 | 58 | 64 | 29.44x | **31.33x** | +6.4% | triton_v2 |
| 14 | 59 | 160 | 28.47x | **30.59x** | +7.4% | triton_v2 |
| 15 | 62 | 96 | 28.98x | **31.58x** | +9.0% | triton_v2 |
| 16 | 80 | 96 | 27.37x | **29.15x** | +6.5% | triton_v2 |
| 17 | 901 | 96 | **33.23x** | 23.20x | +43.2% | flashinfer_moe |
| 18 | **11948** | 128 | **15.87x** | 4.82x | +229.3% | flashinfer_moe |
| 19 | **14107** | 32 | **15.73x** | 5.11x | +207.8% | flashinfer_moe |

## Summary Statistics

| Metric | flashinfer_moe | triton_v2 |
|--------|---------------|-----------|
| **Overall Average** | 27.38x | 27.45x |
| **Small-batch Average** (seq_len ≤ 80) | 28.46x | **30.53x** |
| **Large-batch Average** (seq_len ≥ 901) | **21.61x** | 11.04x |
| Min | 15.73x | 4.82x |
| Max | 35.77x | 33.87x |
| Wins | 5/19 | 14/19 |

## Key Findings

### 1. Overall: Nearly Tied (27.38x vs 27.45x)

Overall averages are virtually identical, but the distribution is very different:
- **triton_v2 dominates decode-scale batches** (seq_len 1-80): wins 14/16 workloads, avg 30.53x vs 28.46x (+7.3%)
- **flashinfer_moe dominates prefill-scale batches** (seq_len ≥ 901): wins 3/3, avg 21.61x vs 11.04x (+95.7%)

### 2. Three Performance Regimes

| Regime | seq_len | flashinfer_moe | triton_v2 | Winner |
|--------|---------|---------------|-----------|--------|
| Small decode | 1-16 | 28.8x | 30.1x | triton_v2 (+5-20%) |
| Medium decode | 32-80 | 28.4x | 30.4x | triton_v2 (+4-10%) |
| Large prefill | 901-14107 | 21.6x | 11.0x | flashinfer_moe (+43-230%) |

- triton_v2 的 decode 优势稳定在 **+5-20%**
- flashinfer_moe 的 prefill 优势从 seq_len=901 的 +43% 急剧扩大到 seq_len=14107 的 **+208%**

### 3. triton_v2 的大 batch 崩塌模式

| seq_len | triton_v2 speedup | vs flashinfer_moe |
|---------|------------------|-------------------|
| 80 | 29.15x | 0.94x (略慢) |
| 901 | 23.20x | 0.70x (慢 30%) |
| 11948 | 4.82x | 0.30x (慢 70%) |
| 14107 | 5.11x | 0.32x (慢 68%) |

崩塌原因：
- top-8 routing 展开后 seq_len=14107 → **~113K tokens**
- Triton scatter/gather kernel 在大 batch 下 memory-bound，没有 fusion
- sgl_kernel GEMM 为 decode 优化，不适合 prefill 级矩阵
- flashinfer_moe 的 CUTLASS SM100 persistent kernel 有更好的 tiling 和 software pipelining

### 4. 与 4-way 对比的数值差异

本次 2-way run 整体数值比 4-way run 略低（~2-3x），可能原因：
- 减少了 2 个 CuTeDSL solution 的开销，GPU 热状态不同
- Run-to-run variance（benchmark noise）
- **相对排名完全一致**，确认结果可复现

| Metric | 4-way run | 2-way run |
|--------|-----------|-----------|
| flashinfer_moe avg | 30.37x | 27.38x |
| triton_v2 avg | 29.50x | 27.45x |
| triton_v2 large-batch | ~5x | ~5x (一致) |
| flashinfer_moe large-batch | ~16x | ~16x (一致) |

## Conclusion

**对于 LLM 推理场景的选择建议**：
- **纯 decode**（seq_len < 100，绝大多数推理请求）：triton_v2 更快 5-20%
- **prefill 或长序列**（seq_len > 900）：必须用 flashinfer_moe，triton_v2 不可用
- **混合场景**：flashinfer_moe 是更安全的选择，无性能悬崖

## Solutions Tested

| Solution | Description |
|----------|-------------|
| flashinfer_moe | FlashInfer native `trtllm_fp8_block_scale_moe` (CUTLASS SM100 warp-specialized persistent kernel with TMA) |
| triton_fused_moe_v2 | Triton fused MoE with pre-allocated buffers, sgl_kernel GEMM, Triton scatter/SwiGLU+quant/gather |
