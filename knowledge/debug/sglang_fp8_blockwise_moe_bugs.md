# sglang_fp8_blockwise_moe: Bugs Found and Fixed

**Date:** 2026-02-02

## Bug 1: SwiGLU activation order mismatch

**Symptom:** INCORRECT_NUMERICAL on all 19 workloads (cosine similarity ~0.5)

**Root cause:** sgl_kernel's `silu_and_mul` computes `silu(first_half) * second_half`, but the reference implementation computes `silu(second_half) * first_half`. These are mathematically different.

**Evidence:**
- sgl_kernel source: `F.silu(x[..., :d]) * x[..., d:]` (activation.py)
- Reference: `silu_X2 * X1` where `X1 = G1[:, :I]`, `X2 = G1[:, I:]`
- Confirmed via debug_moe.py SwiGLU test

**Fix:** Swap the two halves of GEMM1 output before calling `silu_and_mul`:
```python
c1_swapped = torch.cat([c1[:, n:], c1[:, :n]], dim=1)
silu_and_mul(c1_swapped, intermediate)
```

**Commit:** 1919900

## Bug 2: Routing weight scaling factor not applied

**Symptom:** After SwiGLU fix, E2E test still failing. Output values ~2.5x too small.

**Root cause:** `moe_fused_gate` with `apply_routed_scaling_factor_on_output=False` normalizes weights to sum=1.0 but does NOT multiply by `routed_scaling_factor`. The reference expects weights summing to `routed_scaling_factor` (e.g., 2.5).

**Evidence:**
```
Weight sums per token:
  Ref:  [2.5, 2.5, 2.5, 2.5]
  SGL:  [1.0, 1.0, 1.0, 1.0]
```

**Fix:** Multiply topk_weights by routed_scaling_factor after moe_fused_gate:
```python
topk_weights = topk_weights * routed_scaling_factor
```

**Commit:** f4d2bc4

## Key lesson: moe_fused_gate parameter semantics

- `apply_routed_scaling_factor_on_output=False`: weights normalized to sum=1, scaling_factor NOT applied
- `apply_routed_scaling_factor_on_output=True`: weights normalized to sum=1, then scaling_factor applied to the final MoE output (not the weights)
- Neither option gives weights summing to `routed_scaling_factor` directly
- Must manually multiply weights when the reference expects scaled weights

## Debug methodology

Created `debug_moe.py` with isolated component tests:
1. **Routing test**: Compare moe_fused_gate vs reference routing (expert IDs + weights)
2. **SwiGLU test**: Verify silu_and_mul computation order
3. **E2E test**: Full pipeline comparison with small random inputs
