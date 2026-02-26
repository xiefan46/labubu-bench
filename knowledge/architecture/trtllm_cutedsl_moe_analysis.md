# TensorRT-LLM CuTeDSL MoE Kernel Analysis

**Date:** 2026-02-03
**Source:** TensorRT-LLM PRs #8880, #9288, #10130, #10987, #10429 + local clone analysis

## Overview

TensorRT-LLM has a comprehensive CuTeDSL-based MoE implementation for Blackwell (SM100) targeting NVFP4 quantization. FP8 block-scale support exists but currently uses a **reference (non-optimized) Python implementation**, not CuTeDSL kernels.

## PR Summary

| PR | Title | Status | Key Contribution |
|----|-------|--------|-----------------|
| #8880 | CuTeDSL NVFP4 Grouped GEMM (Part 1) | Merged 2025-11 | Base grouped GEMM kernel + MoE routing utilities |
| #9288 | CuTeDSL NVFP4 Grouped GEMM (Part 2: SwiGLU + Finalize Fusion) | Merged 2025-11 | SwiGLU fusion + finalize fusion kernels |
| #10130 | CuTeDSL FP8 GEMM for Blackwell | Open (not merged) | Dense FP8 GEMM + batched GEMM + grouped GEMM with block-scaling |
| #10987 | TMA.RED to improve memory bandwidth | Merged 2026-01 | Block-reduction (UBLKRED) replacing fine-grained atomics, 50% improvement |
| #10429 | Raster M for gather FC1 kernel | Merged 2026-01 | Configurable raster_along_m for tile scheduling in gather variant |

## Architecture: Kernel Files

All CuTeDSL kernels live in:
```
tensorrt_llm/_torch/cute_dsl_kernels/blackwell/
├── blockscaled_contiguous_grouped_gemm.py                    # Base grouped GEMM
├── blockscaled_contiguous_grouped_gemm_swiglu_fusion.py      # GEMM1 + SwiGLU
├── blockscaled_contiguous_grouped_gemm_finalize_fusion.py    # GEMM2 + Finalize (unpermute + weighted reduce)
├── blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py  # GEMM1 + SwiGLU with gather (no pre-permutation)
├── dense_blockscaled_gemm_persistent.py                      # Dense GEMM (not grouped)
├── custom_pipeline.py                                        # Pipeline abstractions
└── utils.py                                                  # Helper functions (silu, atomic ops, TMA.RED)
```

## Key Design: "Contiguous Grouped" Layout

**This is the most important architectural insight for our work.**

TRT-LLM uses a "contiguous grouped" layout where:
- Matrix A is `[M, K, 1]` -- all experts' tokens concatenated along M dimension (flat)
- Matrix B is `[N, K, L]` -- L is the number of experts (grouped dimension)
- Matrix C is `[M, N, 1]` -- output is also flat

```
Matrix A/C Memory Layout:
   Group 0    Group 1   Group 2
  +---------+---------+---------+
  |         |         |         |
 K| ValidM0 | ValidM1 | ValidM2 |
  |         |         |         |
  +---------+---------+---------+
  |<-        ValidM           ->|
```

This is essentially identical to our flat A/C layout with `m_indptr` -- except TRT-LLM uses `tile_idx_to_group_idx` instead of `m_indptr` for the tile-to-expert mapping. Each ValidM is aligned to 128 or 256 (based on MMA tile M size).

### Tile Scheduling

The kernel uses `tile_idx_to_group_idx[tile_id] -> expert_id` to look up which expert's B weights to use for each tile. A dedicated scheduler warp dispatches tile info to consumer warps via a tile_info pipeline.

Key tensors:
- `tile_idx_to_group_idx`: [num_tiles] int32 -- maps tile ID to expert
- `num_non_exiting_tiles`: [1] int32 -- number of valid tiles
- `tile_idx_to_mn_limit`: [num_tiles] int32 -- M/N limits per tile (for masking)

## Warp Specialization (7 warps)

| Warp ID | Role |
|---------|------|
| 0-3 | Epilogue warps (4 warps) |
| 4 | MMA warp |
| 5 | TMA warp (DMA loads) |
| 6 | Scheduler warp |

## Four Kernel Variants

### 1. Base Grouped GEMM
- Standard `C = alpha * (SFA * A) * (SFB * B)`
- Input A is already permuted (tokens pre-sorted by expert)
- TMA loads for A, B, SFA, SFB

### 2. GEMM1 + SwiGLU Fusion
- Weight B has interleaved up/gate: `[up_0:64, gate_64:128, up_128:192, ...]`
- Epilogue applies: `output = up * silu(gate)` (SwiGLU activation)
- Output C is `[M, N/2]` (halved due to SwiGLU)
- Optional NVFP4 dynamic quantization in epilogue (generate SFC + quantize)

### 3. Gather GEMM1 + SwiGLU Fusion
- Input A is **not pre-permuted** -- uses `token_id_mapping` for gather
- Uses LDGSTS (Load Global to Shared with Swizzle) instead of TMA for A and SFA
- Uses TMA for B and SFB (weights are already organized by expert)
- Has `raster_along_m` option for tile scheduling orientation
- 11 warps total: warps 4-7 for LDGSTS, warp 8 for MMA, warp 9 for TMA, warp 10 for scheduler

### 4. GEMM2 + Finalize Fusion
- Fuses the unpermute/finalize operation into the GEMM2 epilogue
- Epilogue logic:
  a. Use `permuted_idx_to_expanded_idx` to map permuted row -> (token_id, topk_idx)
  b. Load `router_scale` from global memory directly to registers
  c. Apply: `Final = router_scale * alpha * acc`
  d. **Atomic add** to output buffer (multiple experts contribute to same token)
- Uses vectorized atomic adds: `red.global.v4.bf16x2.add.noftz` (8 bf16 values at once)
- TMA.RED optimization (PR #10987): bulk reduction `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16` for 50% bandwidth improvement

## FP8 Block-Scale Status

**Critical finding: FP8 block-scale MoE does NOT have CuTeDSL kernels yet.**

In `fused_moe_cute_dsl.py`, the `run_moe_fp8_block_scales()` method uses:
- `torch.ops.trtllm.fp8_quantize_1x128(x)` for quantization
- `cute_dsl_fp8_group_blockwise_gemm_ref()` -- a **Python reference implementation** that loops over experts with `torch.einsum`
- Standard `torch.compile`-d `swiglu_fused_moe()` for activation
- `torch.ops.trtllm.moe_finalize_scale_op()` for finalize

The reference GEMM is pure PyTorch, running one expert at a time:
```python
for i in range(len_offset_array - 1):
    start, end = offset_array[i], offset_array[i + 1]
    ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end], updated_b[:, :, i])
```

**PR #10130 (FP8 CuTeDSL GEMM) is still OPEN** -- it adds dense FP8 GEMM and batched FP8 GEMM but has not been merged. This suggests FP8 CuTeDSL grouped GEMM is not yet production-ready in TRT-LLM.

### PR #10130 Deep Dive: `BlockwiseGemmKernel` (FP8 Dense/Batched GEMM)

**File**: `blockwise_gemm/blockwise_gemm.py` (2565 lines, single class)

**Key architecture differences from NVFP4 kernels**:

| Aspect | NVFP4 kernels | PR #10130 FP8 kernel |
|--------|--------------|---------------------|
| MMA type | `make_blockscaled_trivial_tiled_mma` (hardware block_scale) | `make_trivial_tiled_mma` (standard dense) |
| Scale factors | SFA/SFB loaded via TMA→SMEM→S2T→TMEM | SFA/SFB loaded via `CopyG2SOp` (LDG)→SMEM by dedicated Scale warp |
| Scale application | Hardware in MMA pipeline (zero cost) | Software in `acc_update` warps: `final = acc * scale_a * scale_b + final` |
| Scale granularity | `sf_vec_size=16/32`, SF dtype=FP8 | `1×128` blocks, SF dtype=Float32 |
| Warps | 7 (epilog×4, MMA, TMA, Scheduler) | **12** (acc_update×4, epilog×4, MMA, TMA, Scale, Scheduler) |
| Pipelines | 3 (AB, ACC, tile_info) | **5** (AB, Scale, ACC, Epilogue, tile_info) |

**Warp specialization (12 warps, 384 threads)**:

| Warp ID | Role | Description |
|---------|------|-------------|
| 0-3 | acc_update warps | Load partial acc from TMEM, multiply by SFA*SFB, accumulate across K-tiles, write final acc back to TMEM |
| 4-7 | Epilogue warps | Load final acc from TMEM, type convert, TMA store to GMEM |
| 8 | MMA warp | Execute `tcgen05.mma` (standard, no block_scale) |
| 9 | TMA warp | TMA loads for A, B (no SFA/SFB — those go through Scale warp) |
| 10 | Scale warp | Load SFA, SFB from GMEM→SMEM via `CopyG2SOp` (predicated copy) |
| 11 | Scheduler warp | Persistent tile dispatch via `StaticPersistentTileScheduler` |

**Scale application flow**:
```
Per K-tile iteration:
  1. Scale warp: LDG SFA[m_block, k_tile], SFB[n_block, k_tile] → SMEM
  2. MMA warp: A × B → partial acc in TMEM (raw FP8 dot product)
  3. acc_update warps: TMEM→RMEM, multiply by SFA*SFB from SMEM, accumulate to final_acc, RMEM→TMEM
After all K-tiles:
  4. Epilogue warps: TMEM→RMEM, type convert to BF16, TMA store to GMEM
```

**TMEM usage**: Two accumulator regions in TMEM:
- `tCtAcc_base` (offset 0): Per-K-tile partial accumulator from MMA
- `tCtAcc_final` (offset 384): Running final accumulator with scale corrections applied

**Critical observation**: PR #10130 does NOT add grouped GEMM (the commit `10e4ca9ae` "Revert grouped gemm related changes" explicitly removed it). Only dense and batched FP8 GEMM are included. The MoE `run_moe_fp8_block_scales()` path is unchanged — still uses reference Python einsum.

## Techniques Applicable to Our Work

### 1. Tile-Based Expert Dispatch (vs m_indptr)
TRT-LLM's `tile_idx_to_group_idx` is semantically equivalent to our `m_indptr` approach but with a different representation. Instead of storing expert boundary offsets, they store a per-tile expert ID lookup. This avoids binary search in the kernel.

### 2. Fused SwiGLU in Epilogue
The SwiGLU fusion with interleaved weights `[up_0:64, gate_64:128, ...]` is a powerful optimization:
- Single GEMM computes both up and gate projections
- Epilogue applies activation without extra kernel launch
- Output is half the width (N/2)

### 3. Fused Finalize with Atomic Add
The GEMM2 + Finalize fusion eliminates a separate finalize kernel:
- Atomic adds directly to the original (unpermuted) output buffer
- Vectorized atomics (`v4.bf16x2` = 8 elements per instruction)
- TMA.RED bulk reduction for even better bandwidth

### 4. Gather in GEMM1 (No Pre-Permutation)
The gather variant (`blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py`) avoids pre-permuting input activations by doing the gather during GMEM-to-SMEM load using LDGSTS. This saves a separate permute/scatter kernel.

### 5. CUDA Graph Support
Padding support for fixed-shape allocations:
- Pad `tile_idx_to_group_idx` and A/C to maximum sizes
- `num_non_exiting_tiles` controls how many tiles are actually processed
- Scheduler warp exits early when `tile_idx >= num_non_exiting_tiles`

### 6. Grid Dependency Control (PDL)
Uses `griddepcontrol.wait` and `griddepcontrol.launch_dependents` PTX instructions for programmatic dependent launch -- allows next kernel to start early without full grid completion.

### 7. Async Output Memset with Aux Streams
Uses auxiliary CUDA streams to memset the output buffer asynchronously while GEMM1 is running, so GEMM2's atomic adds start from zero without serialization.

## Implications for Our FP8 MoE Competition Entry

1. **TRT-LLM does NOT have production CuTeDSL FP8 grouped GEMM** -- their FP8 MoE path uses reference Python code. This means we're operating in uncharted territory, same as them.

2. **The NVFP4 CuTeDSL kernels are excellent architectural references** for kernel structure (warp specialization, tile scheduling, pipeline stages) even though they target a different quantization format.

3. **Our approach should combine**:
   - Flat A/C layout (matching TRT-LLM's "contiguous grouped")
   - tile_idx_to_expert_idx dispatch (matching their tile scheduling)
   - Standard FP8 MMA (not hardware block-scale) with software per-128-block scaling in epilogue
   - SwiGLU and finalize fusion in epilogue for end-to-end optimization

4. **Key differences from NVFP4 kernels**:
   - FP8 uses standard `tcgen05.mma` (not `tcgen05.mma.kind.block_scale`)
   - float32 scales applied in epilogue (not MX format in TMEM)
   - Scale block size is 128 (not 16 or 32)
