# CuTeDSL v3 GEMM Kernel Optimization Directions

**Date**: 2026-02-09
**Context**: CuTeDSL v3.2 achieves 9-15x speedup but lags behind flashinfer_moe (15-45x) and triton_v2 (24-37x)

## Current CuTeDSL v3 Configuration

From `moe_grouped_gemm_cutedsl_v3.py`:

```python
mma_tiler_mn = (128, 128)      # Fixed MMA tile size
cluster_shape_mn = (1, 1)      # Single CTA, no clustering
num_acc_stage = 1              # Single accumulator stage (cycled per K-tile)
num_ab_stage = auto            # Computed from SMEM capacity (~4-8 stages)
```

**Warp specialization (6 warps total):**
- Warps 0-3: Epilogue (4 warps for TMEM→register→SMEM→GMEM)
- Warp 4: MMA (single warp for tcgen05 MMA)
- Warp 5: TMA (single warp for GMEM→SMEM loads)

## trtllm CUTLASS SM100 Kernel Configuration

From `moe_gemm_template_dispatch_tma_ws.h`:

**Supported tile sizes (M × N × K):**
- 64×32×128, 64×64×128, 64×128×128, 64×256×128
- 128×16×128, 128×32×128, 128×64×128, **128×128×128**, 128×256×128

**Runtime cluster shapes:**
- 1×1×1 (single CTA)
- 2×1×1 (2 CTAs in M dimension)
- 1×2×1 (2 CTAs in N dimension)
- 2×2×1 (4 CTAs)

**K-tile depth:**
- Ktile = 512 for FP8 (4× deeper than 128)

**Mainloop schedule:**
- `COOPERATIVE` for tile_M ≥ 128 (ping-pong for 128×128)
- `PINGPONG` for smaller tiles

## Gap Analysis

| Aspect | CuTeDSL v3 | trtllm CUTLASS | Impact |
|--------|------------|----------------|--------|
| Tile size | Fixed 128×128 | Tuned per problem | Medium |
| Cluster shape | 1×1 only | Dynamic 1×1 to 2×2 | **High** |
| K-tile depth | 128 | 512 | Medium |
| Mainloop | Simple TMA WS | COOPERATIVE | **High** |
| Scheduler | MaskedScheduler | Specialized MoE scheduler | Medium |

## Optimization Directions

### Direction 1: Cluster Shape (High Impact, Medium Effort)

Enable 2×1 or 1×2 clustering for TMA multicast reuse:
- **2×1**: Share A loads across 2 CTAs processing different N tiles
- **1×2**: Share B loads across 2 CTAs processing different M tiles

**Changes required:**
1. Modify `cluster_shape_mn = (2, 1)` or `(1, 2)` in API
2. Ensure TMA multicast masks are configured correctly (already in kernel)
3. Update grid computation for cluster-aware scheduling

**Expected benefit:** 1.3-1.5x on memory-bound small batches

### Direction 2: Tile Size Tuning (Medium Impact, Low Effort)

Try different tile configurations based on problem size:
- **Small N (< 256):** 128×64 or 64×64 for better occupancy
- **Large N (≥ 256):** 128×256 or 256×128 for compute density

**Implementation:**
```python
# Dynamic tile selection based on N
if N <= 64:
    mma_tiler_mn = (128, 64)
elif N <= 128:
    mma_tiler_mn = (128, 128)
else:
    mma_tiler_mn = (128, 256)
```

### Direction 3: K-tile Pipeline Depth (Medium Impact, Medium Effort)

Increase K-tile depth from 128 to 256 or 512:
- Current: Process 128 K elements per pipeline stage
- Target: Process 256-512 K elements for better TMA/MMA overlap

**Changes required:**
1. Modify `mma_inst_tile_k` computation
2. Increase `num_ab_stage` SMEM allocation
3. Adjust scale loading granularity

### Direction 4: Mainloop Schedule (High Impact, High Effort)

Implement COOPERATIVE mainloop for 128×128 tiles:
- Uses special ping-pong buffering between producer/consumer warps
- Better overlap between TMA loads and MMA compute

**Current flow (per K-tile):**
```
TMA wait → MMA compute → Epilogue (scale + accumulate)
```

**COOPERATIVE flow:**
```
TMA warp: prefetch K+1 while MMA processes K
Epilogue: overlap with next tile's TMA
```

### Direction 5: Scheduler Optimization (Low Impact, Low Effort)

Reduce MaskedScheduler overhead:
- Cache `work_tile.is_valid_tile` check results
- Pre-compute expert tile boundaries outside loop
- Use `cutlass.range` with larger unroll factors

### Direction 6: Scale Loading Optimization (Low Impact, Low Effort)

Cache scales in shared memory:
- Current: Each thread loads its own `a_scale_val` and `b_scale_val` per K-tile
- Optimization: Load scales cooperatively, broadcast via SMEM

```python
# Current (per-thread loads)
b_scale_val = mBScale[expert_idx, n_tile_idx, k_tile]  # 32 redundant loads
a_scale_val = mAScale[k_tile, m_flat]                   # divergent loads

# Optimized (shared + broadcast)
if threadIdx.x == 0:
    smem_b_scale = mBScale[expert_idx, n_tile_idx, k_tile]
__syncthreads()
b_scale_val = smem_b_scale  # single load, broadcast
```

## Recommended Implementation Order

1. **Cluster shape (2×1)** — Highest ROI, mostly parameter change
2. **Tile size tuning** — Quick experiment with different sizes
3. **K-tile depth** — Requires SMEM layout changes
4. **Mainloop schedule** — Major rewrite, save for later
5. **Scale caching** — Minor optimization, low priority

## Quick Experiment: Cluster Shape

To test 2×1 clustering, modify `moe_grouped_gemm_cutedsl_v3.py`:

```python
# Line 1218-1219: Change cluster shape
mma_tiler_mn = (128, 128)
cluster_shape_mn = (2, 1)  # Was (1, 1)
```

Then benchmark:
```bash
flashinfer-bench run --local ~/labubu-bench/flashinfer-trace \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions cutedsl_moe_fp8_v3 \
  --timeout 1200 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85
```

## References

- trtllm CUTLASS kernels: `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/`
- CuTeDSL v3 kernel: `flashinfer/cute_dsl/moe_grouped_gemm_cutedsl_v3.py`
- MaskedScheduler: `flashinfer/cute_dsl/blockscaled_gemm.py`
