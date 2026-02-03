# CuTeDSL v3 — Flat Kernel Architecture

## Overview

v3 replaces the 3D batched A/C tensors `[E, max_M, K/N]` with 2D flat tensors `[total_padded_M, K/N]` where expert boundaries are at 128-aligned offsets defined by `m_indptr_tiles`.

## Key Design: L=1 Trick

CuTe TMA and MMA infrastructure expect 3D tensors `(M, K/N, L)`. Instead of rewriting the entire TMA pipeline for 2D, v3 uses an **L=1 trick**:

- A: logical `(M_total, K, 1)` → physical `[1, M_total, K]` = flat `[M_total, K]`
- C: logical `(M_total, N, 1)` → physical `[1, M_total, N]` = flat `[M_total, N]`
- B: logical `(N, K, E)` → physical `[E, N, K]` (unchanged)

This preserves TMA compatibility. Since L=0 always, the L stride never contributes to addressing.

## MaskedScheduler + Dummy c_sched

The MaskedScheduler needs a 3D c tensor for tile counting. v3 passes a **dummy** `c_sched` with shape `(M_total, N, E)` and a dummy pointer (16). The scheduler only uses its layout for computing `problem_shape_ntile_mnl`, never dereferences the data pointer.

## Global M Tile Computation

The scheduler still produces `(local_m_tile, n_tile, expert_idx)`. In the kernel:
```
global_m_tile = m_indptr_tiles[expert_idx] + local_m_tile
```
- TMA warp: `tAgA[(None, global_m_tile, None, 0)]` (A), `tBgB[(None, n_tile, None, expert_idx)]` (B)
- Epilogue: `bSG_gC_tile = bSG_gC[(None, None, None, global_m_tile, n_tile, 0)]`
- a_scale: `mAScale[k_tile, global_m_tile * 128 + epi_tidx]`

## Memory Savings

| Routing pattern | v2 (batched) | v3 (flat) | Savings |
|-----------------|-------------|-----------|---------|
| Uniform (256/expert, 32E) | 32×256 = 8192 | 32×256 = 8192 | 0% |
| Skewed (512/expert, 16 active) | 32×512 = 16384 | 16×512 = 8192 | 50% |
| Very sparse (2/32 active, 128 each) | 32×128 = 4096 | 2×128 = 256 | 94% |

## File Layout

```
moe_grouped_gemm_cutedsl_v3.py  — Sm100FlatGroupedGemmKernel + wrapper + API
moe_grouped_gemm_fp8_v3.py     — GEMM1/GEMM2 wrappers with scatter/unsatter
moe_pipeline_v3.py              — Full pipeline: Routing → Scatter → GEMM1 → SwiGLU → GEMM2 → Unsatter → Finalize
```

## Pipeline Data Flow

1. **Routing**: produces `m_indptr [E+1]` (NOT 128-aligned)
2. **Compute aligned boundaries**: `aligned_m = ceil(masked_m/128)*128`, `m_indptr_aligned`, `m_indptr_tiles`
3. **Scatter**: `a_gathered[max_padded, K]` → `a_flat[total_padded_M, K]` (vectorized via searchsorted)
4. **GEMM1**: flat kernel → `c_flat[total_padded_M, 2*I]` (BF16) → FP8 quantize
5. **SwiGLU**: pointwise on flat layout (padding rows are zeros → no effect)
6. **GEMM2**: flat kernel → `out_flat[total_padded_M, H]` (BF16)
7. **Unsatter**: `out_flat[total_padded_M, H]` → `out_grouped[max_padded, H]` (gather via dst_row)
8. **Finalize**: standard weighted reduce

## Compilation Cache Key

`(M_total, N, K, E, c_dtype_str, mma_tiler_mn, cluster_shape_mn, sm_count, sm_version)`

M_total varies per call (depends on routing). Same as v1 where max_M varied.

## Code Review Findings (v3.1)

Thorough review completed. Key findings:

### Verified Correct
- **L=1 trick**: TMA addressing verified. `base + 0*(M_total*K) + global_m_tile*128*K + k = base + global_m_tile*128*K + k`. L stride never contributes since L=0 always.
- **Dummy c_sched (ptr=16)**: MaskedScheduler only uses `problem_shape_ntile_mnl[1]` (N//128) from `c_sched`. The M and E dimensions in `problem_shape_ntile_mnl` are computed but never used in iteration logic. The scheduler iterates per-expert via `masked_m[batch_idx]`, not via c_sched's M dimension.
- **MLIR serialization**: c_sched's ptr=16 survives `__extract_mlir_values__`/`__new_from_mlir_values__` because `cute.zipped_divide` only reads layout, not data.
- **m_indptr_tiles[expert_idx]**: 1D CuTe tensor Int32 element access on GPU works identically to `masked_m[batch_idx]` in v1 (established pattern).
- **a_scale indexing**: `mAScale[k_tile, global_m_tile * 128 + epi_tidx]` correctly indexes flat scale array. `epi_tidx` maps 1:1 to M rows within tile (128 threads, 128 M rows).
- **b_scale indexing**: `mBScale[expert_idx, n_tile_idx, k_tile]` unchanged from v1.
- **SwiGLU on flat padded**: Operates pointwise; padding zeros → SwiGLU(0)=0 → no effect. Verified `moe_swiglu_fp8_requant` handles arbitrary batch dim.
- **`_quantize_output_fp8`**: Handles flat padded input (vectorized PyTorch ops, per-row independent).
- **scatter_to_flat_padded**: `searchsorted(m_indptr[1:], right=True)` correctly handles expert boundaries including empty experts. Verified with trace-through example.
- **gather_from_flat_padded**: Inverse of scatter via `c_flat[dst_row]`.

### Edge Cases Handled
- All experts empty: early exit with zero tensor
- Some experts empty (masked_m=0): scheduler skips (0 tiles), flat layout has no rows for them
- Single token (seq_len=1): alignment padding to 128 works correctly
- Destination passing (pre-allocated output): supported via `output` parameter

### Known Limitations
- Compilation cache: `M_total` varies per call → new compilation per unique `total_padded_M` (same as v1's `max_M`)
- `gather_from_flat_padded`: `max_padded` parameter is unused (documentation-only)

## Test Coverage

| Test | What it validates |
|------|-------------------|
| `test_v3_vs_v1` (6 seq_lens) | End-to-end correctness, cosine_sim >= 0.99 |
| `test_v3_small_batch` | Edge case seq_len=1 |
| `test_v3_output_preallocated` | Destination passing + correctness |
| `test_v3_sparse_routing` | Sparse routing (8/32 active), v3 memory advantage |
| `test_v3_memory_savings` | Verifies flat layout memory savings |
| `test_v3_alignment_edge_cases` | Mixed alignment, all empty, exact 128, single token |
| `test_v3_scatter_gather_roundtrip` | Scatter/gather are exact inverses |
| `bench_cute_dsl_moe_v3.py` | Performance benchmark v1 vs v2 vs v3 |

## Next Steps (v3.2+)

- v3.2: GEMM1 FP8 output in kernel epilogue (eliminate `_quantize_output_fp8` round-trip)
- v3.3: Scatter kernel (eliminate Python gather + scatter)
- v3.4: Routing cleanup (GPU-side aligned m_indptr computation)
