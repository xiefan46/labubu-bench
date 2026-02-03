# CuTeDSL v3 (Flat Kernel) Benchmark Results

## Date: 2026-02-03
## GPU: NVIDIA B200
## Timing: CUDA events (median of 20 iterations)

## v1 vs v2 vs v3 End-to-End Pipeline

| seq_len | v1 (ms) | v2 (ms) | v3 (ms) | v3/v2 | v3/v1 | cos_sim |
|---------|---------|---------|---------|-------|-------|---------|
| 1       | 5.125   | 1.551   | 1.250   | 0.806 | 0.244 | 1.0000  |
| 4       | 5.077   | 1.527   | 1.251   | 0.819 | 0.246 | 1.0000  |
| 16      | 5.093   | 1.536   | 1.256   | 0.818 | 0.247 | 1.0000  |
| 64      | 5.048   | 1.551   | 1.251   | 0.807 | 0.248 | 1.0000  |
| 256     | 5.102   | 1.607   | 1.289   | 0.802 | 0.253 | 1.0000  |
| 1024    | 5.444   | 1.658   | 1.304   | 0.786 | 0.239 | 1.0000  |

## Key Findings

- **v3 vs v2**: ~19-21% faster across all seq_lens (v3/v2 ratio 0.79-0.82)
- **v3 vs v1**: ~4x faster (v3/v1 ratio 0.24-0.25)
- **Correctness**: Bit-exact with v1 (cosine similarity = 1.000000, zero diff)
- **Stable across seq_lens**: Performance improvement consistent from seq_len=1 to 1024

## v3 Architecture

- Flat 2D A/C tensors `[total_padded_M, K/N]` instead of batched `[E, max_M, K/N]`
- 128-aligned expert boundaries via `m_indptr_tiles`
- L=1 trick for TMA compatibility
- Memory savings: 75%+ for skewed routing patterns

## What v3 Eliminates (vs v2)

- Padding all experts to max_M (only pad each to 128-alignment)
- Less total memory for A/C tensors
- Simpler scatter/gather logic (flat indexing)

## Test Results

12/12 tests passed:
- 6 seq_len correctness tests (all cosine_sim = 1.000000)
- small_batch (seq_len=1)
- output_preallocated (destination passing)
- sparse_routing (8/32 active experts)
- memory_savings (75% savings verified)
- alignment_edge_cases (mixed alignment patterns)
- scatter_gather_roundtrip (exact inverse property)
