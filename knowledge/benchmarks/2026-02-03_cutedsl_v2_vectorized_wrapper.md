# CuTeDSL v2 Vectorized Wrapper Benchmark (2026-02-03)

## Summary

Phase 1 of the v2 optimization: replaced Python for-loops in GEMM padding/unpadding
with vectorized torch ops (searchsorted + scatter/gather). **4.1x end-to-end speedup**.

## Key Results

- **v2 vs v1 precision**: cosine_sim = 1.000000 (bitwise identical)
- **GEMM1**: 3.139ms → 0.608ms (**5.2x** speedup)
- **GEMM2**: 2.432ms → 0.330ms (**7.4x** speedup)
- **End-to-end**: 5.702ms → 1.384ms (**4.1x** speedup)
- **v2 vs SGLang**: 1.6x (down from 6.7x)

## Per-Stage Timing (T=1024, B200, median ms)

| Stage | CD v1 | CD v2 | SGLang v2 | Triton v2 | v1/SG | v2/SG | v1/v2 |
|-------|-------|-------|-----------|-----------|-------|-------|-------|
| Routing | 0.282 | 0.282 | 0.190 | 0.190 | 1.5x | 1.5x | 1.0x |
| GEMM1 | 3.139 | 0.608 | 0.360 | 0.360 | 8.7x | 1.7x | 5.2x |
| SwiGLU+Req | 0.028 | 0.027 | 0.162 | 0.077 | 0.2x | 0.2x | 1.0x |
| GEMM2 | 2.432 | 0.330 | 0.195 | 0.195 | 12.5x | 1.7x | 7.4x |
| Finalize | 0.041 | 0.040 | 0.049 | 0.049 | 0.9x | 0.8x | 1.0x |
| **End-to-end** | **5.702** | **1.384** | **0.855** | **0.755** | **6.7x** | **1.6x** | **4.1x** |

## What v2 Changed

Replaced three `for e in range(E)` Python loops (each with `.item()` GPU→CPU sync)
with vectorized torch ops:
- `torch.searchsorted` to map each row to its expert
- Index assignment (`a_batched[dest_flat] = a`) for A padding
- Index assignment (`a_scale_flat[:, dest_flat] = a_scale`) for scale padding
- Index selection (`c_flat[dest_flat]`) for output extraction

Total: ~192 GPU→CPU syncs eliminated → ~0 syncs (only 1 `.item()` for max_M_raw).

## Remaining Gap Analysis (v2 vs SGLang: 1.6x)

The 1.6x gap is the **kernel architecture overhead** (Layer 2):
- CuTeDSL kernel requires `[E, max_M, K]` batched input → zero-padded rows waste bandwidth
- SGLang uses flat `[total_M, K]` + `m_indptr` → no padding at all
- The scatter/gather ops themselves add ~0.2-0.3ms overhead

Phase 2 (flat-input kernel with m_indptr) would eliminate this remaining gap.

## Notes

- 30/32 experts had 0 tokens at T=1024 (only 2 experts active due to routing bias)
- v2 first-call time: 0.03s (no JIT recompile needed — same kernel as v1)
- CuTeDSL SwiGLU is 6x faster than SGLang's (0.027ms vs 0.162ms)
