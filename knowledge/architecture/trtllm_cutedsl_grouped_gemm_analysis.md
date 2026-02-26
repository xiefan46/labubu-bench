# TensorRT-LLM CuTeDSL Grouped GEMM Kernel Analysis for FP8 Adaptation

## 1. File Inventory

All kernel files live under:
`/Users/fxie/Desktop/projects/labubu-bench/projects/TensorRT-LLM/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/`

| File | Purpose |
|------|---------|
| `blockscaled_contiguous_grouped_gemm.py` | Base grouped GEMM (C = alpha * SFA*A * SFB*B) |
| `blockscaled_contiguous_grouped_gemm_swiglu_fusion.py` | GEMM1 + SwiGLU activation fusion |
| `blockscaled_contiguous_grouped_gemm_finalize_fusion.py` | GEMM2 + router_scale + atomic scatter-add |
| `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` | GEMM1 + SwiGLU with LDGSTS gather for A |
| `custom_pipeline.py` | Pipeline classes: PipelineTmaUmma, PipelineUmmaAsync, PipelineCpAsyncUmma |
| `utils.py` | Helpers: make_ptr, silu_f32, vectorized_atomic_add_bf16x8/fp32x2, blk_reduce_*, griddepcontrol |

Orchestrator: `/Users/fxie/Desktop/projects/labubu-bench/projects/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py`

---

## 2. Base Grouped GEMM Kernel (`Sm100BlockScaledContiguousGroupedGemmKernel`)

### 2.1 Class Structure and Configuration

```python
class Sm100BlockScaledContiguousGroupedGemmKernel:
    def __init__(self, sf_vec_size, mma_tiler_mn, cluster_shape_mn):
        self.sf_vec_size = sf_vec_size          # 16 (NVF4) or 32 (MXF8/MXF4)
        self.acc_dtype = cutlass.Float32
        self.use_2cta_instrs = mma_tiler_mn[0] == 256  # True for M=256
        self.mma_tiler = (*mma_tiler_mn, 1)     # K deferred to _setup_attributes
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.occupancy = 1
```

### 2.2 Warp Specialization (7 warps total)

| Warp ID | Role | Threads |
|---------|------|---------|
| 0-3 | Epilogue warps (4 warps) | 128 |
| 4 | MMA warp | 32 |
| 5 | TMA warp (loads A, B, SFA, SFB) | 32 |
| 6 | Scheduler warp (tile dispatch) | 32 |

Total: 224 threads per CTA.

Named barriers:
- `barrier_id=1`: CTA sync (all 224 threads)
- `barrier_id=2`: Epilogue sync (128 threads, warps 0-3)
- `barrier_id=3`: TMEM alloc (MMA + epilogue warps)
- `barrier_id=4`: Scheduler sync (32 threads)

Register allocation:
- `num_regs_uniform_warps = 64` (MMA, TMA warps)
- `num_regs_sched_warps = 64` (scheduler)
- `num_regs_epilogue_warps = 216` (epilogue warps get much more register budget)

### 2.3 MMA Tile Configuration

```python
# K dimension computed from tiled_mma:
mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
mma_inst_tile_k = 4
# Final MMA tiler: (M, N, mma_inst_shape_k * 4)
# For NVF4 (sf_vec_size=16): K = 64 * 4 = 256

self.cta_tile_shape_mnk = (
    mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),  # M/CTA_count
    mma_tiler[1],   # N
    mma_tiler[2],   # K
)
```

For typical config `mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1)`:
- MMA tiler: (256, 128, K_tile)
- CTA tile: (128, 128, K_tile) [divided by 2 for 2CTA]
- Cluster: (2, 1) -> 2 CTAs along M

### 2.4 Pipeline Stages

```python
# ACC stages: 1 if N=256, else 2
num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
num_c_stage = 2
num_tile_stage = 2
# AB stages: computed from SMEM capacity after allocating C, info, barriers
```

Three pipelines:
1. **ab_pipeline** (`PipelineTmaUmma`): TMA producer -> UMMA consumer, for A/B/SFA/SFB loading
2. **acc_pipeline** (`PipelineUmmaAsync`): UMMA producer -> Async thread consumer, for accumulator handoff
3. **tile_info_pipeline** (`PipelineAsync`): Scheduler producer -> all consumer warps, for tile dispatch

### 2.5 Memory Hierarchy (SM100 Features)

**Block-scaled MMA flow:**
```
GMEM -> [TMA] -> SMEM (A, B)
GMEM -> [TMA] -> SMEM (SFA, SFB)
SMEM -> [tcgen05.cp] -> TMEM (SFA, SFB)
SMEM + TMEM -> [tcgen05.mma.block_scale] -> TMEM (accumulator)
TMEM -> [tcgen05.ld] -> RMEM (registers)
RMEM -> epilogue ops -> SMEM -> [TMA S2G] -> GMEM (C)
```

Key SM100-specific features:
- **TMEM (Tensor Memory)**: Holds scale factors and accumulators, 512 columns total
- **tcgen05.mma.kind.block_scale**: Reads A/B from SMEM, SFA/SFB from TMEM, writes accumulator to TMEM
- **tcgen05.cp (Cp4x32x128bOp)**: SMEM-to-TMEM copy for scale factors
- **tcgen05.ld**: TMEM-to-register load for epilogue
- **TMA multicast**: Cluster-level data reuse across CTAs

### 2.6 tile_idx_to_group_idx Usage

The grouped GEMM uses a contiguous layout where all tokens for all experts are packed consecutively in M dimension:

```
Group 0    Group 1    Group 2
|--ValidM0--|--ValidM1--|--ValidM2--|
```

The `tile_idx_to_group_idx` tensor (shape: `[num_tiles]`) maps each M-tile to its expert/group index. In the scheduler warp:

```python
# Scheduler warp dispatches tile info to consumers
expert_idx = tile_idx_to_group_idx[tile_idx]
# Stored in sInfo as (bidx_m, bidx_n, expert_idx, valid_flag)
```

The expert_idx selects which B weight matrix and which alpha scalar to use. B is indexed as `B[expert_idx, :, :]`.

### 2.7 A/B/C Tensor Layout

- **A**: `(M, K, 1)` - row-major ("K" major mode), M concatenates all groups
- **B**: `(N, K, L)` - col-major ("K" major mode), L = num_experts
- **C**: `(M, N, 1)` - row-major ("N"), same M concatenation as A
- **SFA**: Block-scaled layout via `BlockScaledBasicChunk`, `M x ceil_div(K, sf_vec_size) x L`
- **SFB**: Block-scaled layout, `N x ceil_div(K, sf_vec_size) x L`

### 2.8 Scale Factor Loading

Scale factors go through a special atom layout (`blockscaled_utils.tile_atom_to_shape_SF`) and are loaded:
1. TMA: GMEM -> SMEM (via `make_tiled_tma_atom_A/B` with `internal_type=cutlass.Int16`)
2. S2T Copy: SMEM -> TMEM (via `tcgen05.Cp4x32x128bOp`)

TMEM column allocation:
```python
sf_atom_mn = 32
num_sfa_tmem_cols = (cta_tile_M / 32) * mma_inst_tile_k
num_sfb_tmem_cols = (cta_tile_N_sfb / 32) * mma_inst_tile_k
num_accumulator_tmem_cols = cta_tile_N * num_acc_stage  # or overlap formula
# Total TMEM: 512 columns (full SM100 TMEM capacity)
```

### 2.9 Epilogue Structure (Base Kernel)

```python
# Load accumulator from TMEM -> registers
cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
acc_vec = tTR_rAcc.load()

# Apply alpha scaling and type conversion
result = epilogue_op(acc_vec * alpha_val)
tTR_rC.store(result.to(c_dtype))

# Store via TMA: registers -> SMEM -> GMEM
cute.copy(tiled_copy_r2s, tTR_rC, sC_subtile)
cpasync.store(tma_atom_c, gC_subtile, sC_subtile)
```

---

## 3. SwiGLU Fusion Kernel (`Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel`)

### 3.1 Key Differences from Base

1. **Interleaved weight layout**: B has 2x N columns, interleaved as `[up_0:64, gate_0:64, up_64:128, gate_128:192, ...]` with granularity=64
2. **Output C has N/2 columns** (SwiGLU reduces dimension by 2)
3. **Epilogue tile**: Fixed at `(128, 64)` with `epi_tile_cnt = (cta_M/128, cta_N_c/64)`
4. **`epi_tile_n_required = 2 * epi_tile_n`**: Because each SwiGLU step consumes 2 subtiles
5. **Additional parameter**: `vectorized_f32` for packed f32x2 operations
6. **Optional quantization**: Can generate SFC and quantize to Float4E2M1FN (for GEMM2 input)

### 3.2 SwiGLU Epilogue (Core Pattern)

```python
# Process accumulator in pairs of subtiles (up, gate)
for subtile_idx in cutlass.range(0, subtile_cnt, 2):
    real_subtile_idx = subtile_idx // 2

    # Load two adjacent accumulator subtiles from TMEM
    tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
    tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]
    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
    cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

    acc_vec_up = tTR_rAcc_up.load()
    acc_vec_gate = tTR_rAcc_gate.load()

    # SwiGLU: output = (alpha * up) * silu(alpha * gate)
    # Vectorized path (f32x2 packed operations):
    LOG2_E = 1.4426950408889634
    for i in range(0, size, 2):
        acc_vec_up_alpha = mul_packed_f32x2(
            (acc_vec_up[i], acc_vec_up[i+1]), (alpha, alpha))
        acc_vec_gate_alpha = mul_packed_f32x2(
            (acc_vec_gate[i], acc_vec_gate[i+1]), (alpha, alpha))

        # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        # Using exp2 for better numerical behavior:
        #   sigmoid(x) = 1/(1+exp(-x)) = 1/(1+exp2(-x*log2(e)))
        log2e_neg = mul_packed_f32x2(acc_vec_gate_alpha, (-LOG2_E, -LOG2_E))
        exp_result = (exp2(log2e_neg[0], fastmath=True),
                      exp2(log2e_neg[1], fastmath=True))
        plus_one = add_packed_f32x2(exp_result, (1.0, 1.0))
        sigmoid = (rcp_approx(plus_one[0]), rcp_approx(plus_one[1]))
        silu_result = mul_packed_f32x2(sigmoid, acc_vec_gate_alpha)
        output = mul_packed_f32x2(silu_result, acc_vec_up_alpha)
```

Scalar (non-vectorized) fallback:
```python
for i in range(size):
    acc_vec_up_alpha = acc_vec_up[i] * alpha_val
    acc_vec_gate_alpha = acc_vec_gate[i] * alpha_val
    tCompute[i] = acc_vec_up_alpha * silu_f32(acc_vec_gate_alpha, fastmath=True)
```

### 3.3 Optional Quantization (NVF4 Output)

When `generate_sfc=True` (c_dtype is Float4E2M1FN):
1. Compute per-vector absolute max from SwiGLU result
2. Generate scale factor C (SFC) based on max values
3. Store SFC to global memory
4. Quantize output by scaling with reciprocal of SFC

---

## 4. Finalize Fusion Kernel (`Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel`)

### 4.1 Key Differences

1. **Scatter-add epilogue**: Instead of TMA store, uses atomic adds to scatter results back
2. **Extra parameters**: `permuted_idx_to_expanded_idx`, `token_final_scales`, `tile_idx_to_mn_limit`
3. **sInfo has 5 fields**: `(bidx_m, bidx_n, expert_idx, valid_flag, mn_limit)` (vs 4 in base)
4. **No TMA C store**: C is written via vectorized atomics or TMA.RED (block reduce)
5. **`use_blkred` option**: Choose between vectorized per-element atomics vs TMA.RED bulk reduce
6. **`raster_along_m` option**: Tile scheduling can prioritize M or N dimension
7. **Hooked PersistentTileSchedulerParams**: Uses `FastDivmod` for efficient 3-way decomposition

### 4.2 Finalize Epilogue (Core Pattern)

```python
# Map permuted row back to original token
tile_m_start = tile_info[0] * cta_tile_M
permuted_row = tile_m_start + epi_tidx
expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
is_valid_row = permuted_row < tile_info[4]  # mn_limit

if is_valid_row:
    # Decode token and topk indices from expanded_idx
    token_idx = expanded_idx // topK
    topk_idx = expanded_idx % topK

    # Load router scale from global memory (direct G2R, no SMEM)
    token_scale = token_final_scales[(token_idx, topk_idx)]
    alpha_val = alpha_val * token_scale  # Combined: alpha * router_scale

# For each accumulator subtile:
acc_vec = tTR_rAcc.load()
acc_vec_final = alpha_val * acc_vec  # Scale includes both alpha and router_scale
tTR_rC.store(acc_vec_final.to(out_dtype))

# Scatter-add to output (different token rows may map to same output row)
scatter_out = cute.domain_offset((token_idx, 0, 0), out)
```

### 4.3 Atomic Add Patterns

**Vectorized atomics (non-blkred path, default):**
```python
if out_dtype == BFloat16:
    # 8-element vectorization: red.global.v4.bf16x2.add.noftz
    # PTX inline asm with 4 x bf16x2 packed registers
    vectorized_atomic_add_bf16x8(rOut_epi_packed, scatter_out_offset)
elif out_dtype == Float32:
    # 2-element vectorization: red.global.v2.f32.add
    vectorized_atomic_add_fp32x2(rOut_epi_packed, scatter_out_offset)
else:
    # Scalar fallback: red.global.add.f32 or red.add.noftz.bf16
    atomic_add_func(rOut_epi_packed, scatter_out_offset)
```

**Block reduce path (use_blkred=True):**
```python
# Step 1: registers -> SMEM (convert to output dtype first)
tRS_rC.store(acc_vec_final.to(out_dtype))
cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, subtile_idx, None)])

# Step 2: TMA.RED from SMEM -> GMEM (bulk async reduce)
# One copy per row (entire N dimension at once)
if out_dtype == BFloat16:
    blk_reduce_bf16(scatter_out_offset, sC[epi_tidx, None, 0], copy_size)
    # PTX: cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16
elif out_dtype == Float32:
    blk_reduce_fp32(scatter_out_offset, sC[epi_tidx, None, 0], copy_size)
```

### 4.4 C SMEM Layout for Finalize

Different from base kernel - uses a simple padded layout (not TMA-compatible swizzled):
```python
swizzled_pad = 16 // (out_dtype.width // 8)
c_smem_layout_staged = make_layout(
    (cta_tile_M, cta_tile_N, num_c_stage),
    stride=(cta_tile_N + swizzled_pad, 1, cta_tile_M * (cta_tile_N + 8))
)
```

### 4.5 Epilogue Data Layout for Scatter

```python
# Layout determines how accumulator elements map to vectorized atomic operations
if out_dtype == BFloat16:
    # (ttr_racc_size//8, 4_bf16x2_groups, 2_elements_per_group)
    epi_layout = make_layout((ttr_racc_size // 8, 4, 2), stride=(8, 2, 1))
    epi_loop_size = ttr_racc_size // 8
    element_offset = 8  # 8 BF16 elements per vectorized atomic
elif out_dtype == Float32:
    epi_layout = make_layout((ttr_racc_size // 2, 2), stride=(2, 1))
    epi_loop_size = ttr_racc_size // 2
    element_offset = 2  # 2 FP32 elements per vectorized atomic
```

---

## 5. Gather Variant (`BlockScaledContiguousGatherGroupedGemmKernel`)

### 5.1 Key Differences from SwiGLU Kernel

1. **LDGSTS replaces TMA for A loading**: Uses `CopyG2SOp` (cp.async) instead of TMA for A and SFA
2. **token_id_mapping**: Maps permuted rows to original token IDs for gather during load
3. **More warps**: 12 warps total (vs 7 in base)
4. **Separate pipelines**: A pipeline (`PipelineCpAsyncUmma`) and B pipeline (`PipelineTmaUmma`)
5. **Same SwiGLU epilogue** as non-gather variant

### 5.2 Warp Allocation (12 warps)

| Warp ID | Role | Threads |
|---------|------|---------|
| 0-3 | Epilogue warps | 128 |
| 4-7 | LDGSTS A/SFA warps (4 warps) | 128 |
| 8 | MMA warp | 32 |
| 9 | TMA B/SFB warp | 32 |
| 10 | Scheduler warp | 32 |
| 11 | Sync Transform warp (2CTA only) | 32 |

Total: 384 threads per CTA.

### 5.3 LDGSTS A Loading Configuration

```python
# LDGSTS copy atom: 8x LDGSTS.128 per thread for A (32 elements per load)
a_atom_copy = make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    a_dtype, num_bits_per_copy=128,
)
a_thread_layout = make_layout((16, 8), stride=(8, 1))  # 128 threads total
a_value_layout = make_layout((1, 32), stride=(32, 1))

# SFA: 4x LDGSTS.32 per thread (4 scale elements per thread)
sfa_atom_copy = make_copy_atom(cpasync.CopyG2SOp(), sfa_dtype, num_bits_per_copy=32)
```

### 5.4 Token Gather Logic

```python
# Each of 128 threads in the LDGSTS warpgroup loads 8 token offsets
for i in range(8):
    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
    a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
    a_predicate_tensor[i] = (permuted_row < mn_limit)  # Guard against padding
    a_token_offset_tensor[i] = a_token_offset_tensor[i] // topk  # Map to original token

# For SFA: 1 token offset per thread (different addressing pattern)
sfa_token_offset_tensor[0] = gToken_ml_tile[sfa_offset] // topk

# LDGSTS with gather: compute global memory address from token offset
A_gmem_slice_offset = A_gmem_thread_offset + a_token_offset_tensor[i] * stride
tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
# Predicated copy skips invalid tokens (padding tokens marked as -1)
```

### 5.5 Pipeline Setup

```python
# A pipeline: PipelineCpAsyncUmma
# Producer: 4 LDGSTS warps (128 threads) | Consumer: MMA warp
a_pipeline = PipelineCpAsyncUmma.create(
    num_stages=num_ab_stage,
    producer_group=CooperativeGroup(Agent.Thread, 128),
    consumer_group=CooperativeGroup(Agent.Thread, ...),
    cta_layout_vmnk=cluster_layout_vmnk,
)

# B pipeline: PipelineTmaUmma (same as base)
# Producer: TMA warp (32 threads) | Consumer: MMA warp
b_pipeline = PipelineTmaUmma.create(
    num_stages=num_ab_stage, ...
)
```

---

## 6. Pipeline Orchestrator (`fused_moe_cute_dsl.py`)

### 6.1 NVF4 MoE Flow

```python
def run_moe_nvfp4_impl(self, x, token_selected_experts, token_final_scales, x_sf, moe_output, ...):
    # Step 1: Sort tokens by expert (tile-aligned)
    tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, \
        permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = \
        torch.ops.trtllm.moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=num_slots,
            top_k=experts_per_token,
            local_expert_offset=slot_start,
            local_num_experts=expert_size_per_partition,
            tile_tokens_dim=tile_size,  # 128 or 256
        )

    # Step 2: GEMM1 + SwiGLU (gather variant)
    x, x_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell(
        input=x,                    # [total_tokens, K] in NVF4
        weight=w3_w1_weight,        # [num_experts, 2*intermediate, K] (interleaved up/gate)
        input_scale=x_sf,           # NVF4 scale factors
        weight_scale=fc1_weight_block,
        alpha=fc1_global,           # per-expert global scale
        tile_idx_to_group_idx=tile_idx_to_expert_idx,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        ...
    )
    # Output: x = [total_padded_tokens, intermediate] (NVF4), x_sf = output scales

    # Step 3: Async memset output buffer (in aux stream, overlapped)
    # Zeroes out moe_output for scatter-add in step 4
    with torch.cuda.stream(aux_stream):
        event_main.wait()
        torch.ops.trtllm.moe_output_memset_inplace(input=moe_output, ...)
        event_memset.record()
    event_memset.wait()

    # Step 4: GEMM2 + Finalize (scatter-add with router_scale)
    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input=x,                    # [total_padded_tokens, intermediate]
        weight=w2_weight,           # [num_experts, hidden, intermediate]
        output=moe_output,          # [seq_len, hidden] (scatter-add target, pre-zeroed)
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        token_final_scales=token_final_scales,  # [num_tokens, topK]
        ...
    )
```

### 6.2 FP8 Block-Scale MoE Flow (Current Reference Implementation)

```python
def run_moe_fp8_block_scales(self, x, token_selected_experts, token_final_scales, ...):
    # Step 1: Permute tokens by expert (different from NVF4 moe_sort)
    (permuted_row_to_unpermuted_row, permuted_token_selected_experts,
     x, expert_first_token_offset, permuted_token_final_scales,
     unpermuted_row_to_permuted_row) = torch.ops.trtllm.moe_permute_op(
        x, token_selected_experts, token_final_scales, ...)

    # Step 2: Quantize input to FP8 with 1x128 block scales
    x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)

    # Step 3: GEMM1 (REFERENCE ONLY - Python loop, NOT CuTeDSL)
    x = cute_dsl_fp8_group_blockwise_gemm_ref(
        a=x, b=w3_w1_weight, a_sf=x_sf, b_sf=fc1_scales,
        offset_array=expert_first_token_offset)

    # Step 4: SwiGLU activation (separate torch.compile kernel)
    x = swiglu_fused_moe(x)  # x, gate = x.chunk(2); F.silu(gate) * x

    # Step 5: Re-quantize for GEMM2
    x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)

    # Step 6: GEMM2 (REFERENCE ONLY)
    x = cute_dsl_fp8_group_blockwise_gemm_ref(
        a=x, b=w2_weight, a_sf=x_sf, b_sf=fc2_scales,
        offset_array=expert_first_token_offset)

    # Step 7: Finalize (unpermute + scale)
    x = torch.ops.trtllm.moe_finalize_scale_op(x, ...)
```

**CRITICAL**: The FP8 path currently uses a Python reference implementation (`cute_dsl_fp8_group_blockwise_gemm_ref`) with explicit loops over expert groups. It does NOT use CuTeDSL kernels yet. This is the gap we want to fill.

### 6.3 FP8 Reference GEMM Implementation Details

```python
def cute_dsl_fp8_group_blockwise_gemm_ref(a, b, a_sf, b_sf, offset_array):
    # a: [M, K] in FP8, a_sf: per-1x128-block scales (Float32)
    # b: [L, N, K] in FP8, b_sf: [L, N//128, K//128] block scales (Float32)
    # offset_array: [num_groups+1] expert boundaries

    # Scale factor layout for SM100:
    # a_sf is [K//128, M] (transposed from usual [M, K//128])
    input_scale_tmp = a_sf.permute(1, 0)  # -> [M, K//128]

    # Expand block-wise scales to per-element
    for i in range(num_groups - 1):
        start, end = offset_array[i], offset_array[i+1]
        ref[start:end] = einsum("mk,nk->mn",
            (a_scaled)[start:end], (b_scaled)[:, :, i])
    return ref.to(bfloat16)
```

### 6.4 Autotuning

The NVF4 path supports autotuning between tile_size=128 and tile_size=256:
```python
def get_valid_tactics(self, inputs, profile, **kwargs):
    return [128, 256]  # Corresponding to MMA tiler M dimension
```

The `runner_tactic_comb_checker` ensures all sub-kernels (GEMM1 SwiGLU, GEMM2 Finalize) use consistent tile sizes.

---

## 7. FP8 Adaptation Analysis

### 7.1 What Changes for FP8 Block-Scale MMA

The kernels already support FP8 via `make_blockscaled_trivial_tiled_mma`:
- **MXF8**: `A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32`

For our FP8 per-128-block scaling (DeepSeek-style), the key differences:

| Aspect | NVFP4 (current) | FP8 Block-Scale (target) |
|--------|-----------------|--------------------------|
| Data type | Float4E2M1FN | Float8E4M3FN |
| SF data type | Float8E4M3FN/Float8E8M0FNU | Float32 (per-128-block) |
| SF vec size | 16 | 128 |
| MMA kind | block_scale (hardware) | Standard FP8 MMA + software scale |
| A/B bytes per element | 0.5 | 1.0 (2x larger per element) |
| Scale factor density | 1 per 16 elements | 1 per 128 elements (8x sparser) |
| TMEM for scales | Required (SFA/SFB in TMEM) | Not needed (scales in registers) |

### 7.2 Option A: Use `tcgen05.mma.block_scale` with MXF8 Mode

This is the simplest adaptation:
1. Change `sf_vec_size` from 16 to 32
2. Change data types to `Float8E4M3FN`
3. Scale factor dtype becomes `Float8E8M0FNU` (power-of-2 exponent)
4. **Problem**: Our FP8 block scales are Float32 per-128-block, not Float8E8M0FNU per-32-element. The hardware block_scale MMA only supports 8-bit scale factors with specific vector sizes.

### 7.3 Option B: Standard FP8 MMA + Software Dequant in Epilogue (RECOMMENDED)

Use standard (non-block-scaled) `tcgen05.mma` with FP8 operands:
1. Replace `make_blockscaled_trivial_tiled_mma` with standard `make_trivial_tiled_mma` for FP8
2. MMA produces FP32 accumulator (already in TMEM) - this is the raw dot product without scale corrections
3. **Scale application strategy**: Since per-128-block scales mean each K-128-element chunk has one scale, and the MMA K-tile is typically 256 elements:
   - Each K-tile spans 2 scale blocks
   - Need to accumulate partial results per-scale-block and multiply
   - OR: accept that the FP32 accumulator holds the raw FP8 dot product, and apply all scale corrections in the epilogue (outer product of scale_A * scale_B)

4. **Simplified approach (if K_tile == 128)**: Each MMA K-tile aligns exactly with one scale block. Apply `scale_A_block * scale_B_block` as a scalar multiplier to the partial accumulator after each K-tile.

### 7.4 Option C: Decomposed K-tile with Per-Block Scale Application

For K_tile > 128 (e.g., K_tile = 256):
1. Decompose each K-tile MMA into sub-tiles of 128 elements
2. After each 128-element sub-tile, apply per-block scales
3. This effectively creates a nested loop: outer loop over K-tiles, inner loop over 128-element scale blocks

### 7.5 Epilogue Changes

**SwiGLU fusion**: Works unchanged for FP8 - the epilogue operates on FP32 accumulators regardless of input type. The core computation `output = (alpha * up) * silu(alpha * gate)` is dtype-agnostic once the accumulator is in FP32.

Only difference: if we need FP8 re-quantization of the SwiGLU output (for GEMM2 input), we generate FP8 output with per-128-block Float32 scales instead of NVF4 output with UE8M0 scales.

**Finalize fusion**: Works unchanged - operates on FP32 accumulators, applies router_scale, does scatter-add. The atomic operations (`vectorized_atomic_add_bf16x8`) are independent of input type.

### 7.6 SM100-Specific Features Used

1. **TMA (Tensor Memory Access)**: For loading A, B from GMEM -> SMEM
   - Also for multicast across cluster CTAs
   - For storing C from SMEM -> GMEM (S2G)
   - For TMA.RED (bulk reduce) in finalize kernel
2. **TMEM (Tensor Memory)**: 512 columns per SM
   - In block_scale mode: holds SFA, SFB, and accumulators
   - In standard MMA mode: holds only accumulators (more TMEM for accumulator double-buffering)
3. **tcgen05.mma.kind.block_scale**: Hardware block-scaled MMA (NVF4/MXF8 specific)
4. **tcgen05.mma** (standard): FP8/FP16 MMA without hardware scale (for our FP8 path)
5. **tcgen05.cp**: SMEM-to-TMEM copy for scale factors
6. **tcgen05.ld**: TMEM-to-register load for epilogue
7. **LDGSTS (cp.async)**: For gather-based A loading (gather variant)
8. **griddepcontrol**: PDL (Programmatic Dependent Launch) for kernel chaining
9. **Named barriers**: For warp-specialized synchronization
10. **Cluster**: Multi-CTA cooperation for data reuse

### 7.7 Critical Insight for FP8 Adaptation

The `make_blockscaled_trivial_tiled_mma` call is the central MMA configuration:

```python
tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
    self.a_dtype,        # Can be Float8E4M3FN
    self.a_major_mode,
    self.b_major_mode,
    self.sf_dtype,       # Must be Float8E8M0FNU or Float8E4M3FN
    self.sf_vec_size,    # 16 or 32
    self.cta_group,
    self.mma_inst_shape_mn,
)
```

For FP8 with Float32 per-128-block scales (our target), we **CANNOT** use `block_scale` MMA directly because:
1. SF dtype must be FP8 (Float8E8M0FNU or Float8E4M3FN), not Float32
2. SF vec size must be 16 or 32, not 128
3. The hardware block_scale path has a specific TMEM layout requirement for scale factors

**Recommended approach**: Use standard (non-block-scaled) FP8 MMA, and apply the Float32 per-128-block scales in software. This means:
1. The mainloop structure stays the same (TMA loads, MMA warp, pipeline stages)
2. **No SFA/SFB SMEM/TMEM buffers needed** -> more SMEM available for AB stages
3. **No S2T copy for scale factors** -> simpler mainloop, the TMA warp only loads A and B
4. Scale factors loaded separately (e.g., by epilogue warps from GMEM to registers)
5. Scale application: multiply accumulator by `row_scale * col_scale` in epilogue
6. SwiGLU and Finalize fusion patterns work exactly as-is

---

## 8. Summary of Key Code Patterns for Reuse

### 8.1 Persistent Tile Scheduling with Grouped GEMM
- `StaticPersistentTileScheduler` with `FastDivmod` for efficient tile decomposition
- Scheduler warp dispatches `(tile_m, tile_n, expert_idx, valid_flag, [mn_limit])` through `tile_info_pipeline`
- `tile_idx_to_group_idx` maps M-tiles to experts (critical for grouped GEMM)
- `num_non_exiting_tiles` enables early exit for padded tiles (CUDA graph support)

### 8.2 Warp Specialization Pattern
```python
warp_idx = cute.arch.warp_idx()
if warp_idx == sched_warp_id:
    # Tile scheduling loop: read tile_idx_to_group_idx, write sInfo
elif warp_idx == tma_warp_id:
    # TMA producer loop: load A, B, [SFA, SFB] via TMA
elif warp_idx == mma_warp_id:
    # MMA consumer loop: [S2T copy for scales], execute MMA
elif warp_idx in epilog_warp_ids:
    # Epilogue consumer loop: T2R load, alpha/SwiGLU/finalize, store
```

### 8.3 Overlapping Accumulator Pattern
When `overlapping_accum = True` (N=256):
- Double-buffer accumulator in TMEM
- Early release of accumulator buffer during epilogue subtile iteration
- Reverse subtile iteration on alternate stages
```python
if self.overlapping_accum:
    acc_stage_index = acc_consumer_state.phase  # 0 or 1
    reverse_subtile = (acc_stage_index == 0)
    if subtile_idx // 2 == self.iter_acc_early_release_in_epilogue:
        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
```

### 8.4 Pipeline Creation Pattern
```python
# TMA -> UMMA pipeline (for A/B loading)
ab_pipeline = PipelineTmaUmma.create(
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    num_stages=num_ab_stage,
    producer_group=CooperativeGroup(Agent.Thread),
    consumer_group=CooperativeGroup(Agent.Thread, num_mcast_ctas),
    tx_count=num_tma_load_bytes,
    cta_layout_vmnk=cluster_layout_vmnk,
)

# UMMA -> Async pipeline (for accumulator handoff to epilogue)
acc_pipeline = PipelineUmmaAsync.create(
    barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    num_stages=num_acc_stage,
    producer_group=CooperativeGroup(Agent.Thread),
    consumer_group=CooperativeGroup(Agent.Thread, num_epi_threads),
    cta_layout_vmnk=cluster_layout_vmnk,
)

# LDGSTS -> UMMA pipeline (for gather-based A loading)
a_pipeline = PipelineCpAsyncUmma.create(
    num_stages=num_ab_stage,
    producer_group=CooperativeGroup(Agent.Thread, 128),  # 4 LDGSTS warps
    consumer_group=CooperativeGroup(Agent.Thread, ...),
    cta_layout_vmnk=cluster_layout_vmnk,
)
```

### 8.5 Tile Info Pipeline Pattern
```python
# Scheduler writes tile info to SMEM
# sInfo layout: (5_fields, num_tile_stage) or (4_fields, num_tile_stage)
# Fields: tile_m_idx, tile_n_idx, expert_idx, valid_flag, [mn_limit]
for idx in range(num_fields):
    sInfo[(idx, tile_info_producer_state.index)] = tile_info[idx]
tile_info_pipeline.producer_commit(tile_info_producer_state)

# Consumers read tile info from SMEM
tile_info_pipeline.consumer_wait(tile_info_consumer_state)
for idx in range(num_fields):
    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
tile_info_pipeline.consumer_release(tile_info_consumer_state)
```

### 8.6 PDL (Programmatic Dependent Launch)
```python
# At kernel start (first warp):
if TRTLLM_ENABLE_PDL:
    griddepcontrol_wait()  # Wait for previous kernel to finish

# Before epilogue (hint to launch dependent kernel early):
griddepcontrol_launch_dependents()
```
