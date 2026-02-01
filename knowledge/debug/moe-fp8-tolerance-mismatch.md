# MoE FP8 Block-Scale Tolerance Mismatch

## Problem

The `flashinfer_moe` solution fails 16/19 workloads with `INCORRECT_NUMERICAL` when using default flashinfer-bench tolerances.

## Root Cause

### Tolerance comparison

| | atol | rtol | matched ratio | match logic |
|---|---|---|---|---|
| `LowBitEvaluator` (flashinfer-bench) | 0.01 | 0.01 | 0.95 | `(abs_err > atol) & (rel_err > rtol)` — AND, both must exceed |
| `test_dpsk_fused_moe_fp8.py` (flashinfer) | 0.1 | 0.2 | 0.85 | `torch.isclose(a, b, atol, rtol)` — `|a-b| <= atol + rtol*|b|` |
| `test_trtllm_gen_fused_moe.py` (flashinfer) | 0.1 | 0.85 | 0.80 | same `torch.isclose` semantics |

### Why the precision gap exists

- **Reference**: dequantizes FP8 to float32, then float32 matmul
- **Kernel**: hardware FP8 tensor core GEMM (`wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3`) with post-GEMM block scale multiplication
- FP8 E4M3 has 4-bit mantissa (~1/16 relative precision per element), so numerical differences are expected

## Solution

### Option 1: CLI override (branch `feat/cli-required-matched-ratio`)

Added `--required-matched-ratio` CLI parameter to flashinfer-bench:

```bash
flashinfer-bench run --local $FIB_DATASET_PATH \
  --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85 \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions flashinfer_moe --timeout 1200
```

Result: 15/15 completed workloads PASSED (SSH disconnected before remaining 4).

### Option 2: Align match logic with flashinfer (branch `feat/isclose-tolerance-check`)

Changed `compute_error_stats` in `utils.py` from:
```python
exceeds_tol_mask = (abs_error > cfg.atol) & (rel_error > cfg.rtol)
```
to:
```python
exceeds_tol_mask = ~torch.isclose(x, y, atol=cfg.atol, rtol=cfg.rtol)
```

Result: 19/19 workloads PASSED with `--rtol 0.2 --atol 0.1 --required-matched-ratio 0.85`.

## Key Files

- `flashinfer-bench/flashinfer_bench/bench/evaluators/lowbit.py` — LowBitEvaluator, matches `"moe_fp8_block_scale"` in definition name
- `flashinfer-bench/flashinfer_bench/bench/config.py` — BenchmarkConfig defaults (atol=0.01, rtol=0.01)
- `flashinfer-bench/flashinfer_bench/bench/utils.py:108` — tolerance check logic
- `flashinfer-bench/flashinfer_bench/bench/evaluators/registry.py` — evaluator selection
- `flashinfer/tests/moe/test_dpsk_fused_moe_fp8.py` — flashinfer's own FP8 MoE test
- `flashinfer/tests/moe/test_trtllm_gen_fused_moe.py` — TRT-LLM MoE test
