# MoE JIT Compilation Timeout

## Problem

`flashinfer_moe` solution times out (default 300s) on first run due to JIT compilation.

## Root Cause

FlashInfer uses JIT compilation (TVM-FFI) — CUDA kernels are compiled on first use. The MoE kernel compilation for SM100 (Blackwell) takes several minutes. After first compilation, cubins are cached and subsequent runs are fast.

## Solution

Use `--timeout 1200` (20 minutes) to allow JIT warmup:

```bash
flashinfer-bench run --local $FIB_DATASET_PATH \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions flashinfer_moe --timeout 1200
```

## Key Files

- `flashinfer/flashinfer/jit/fused_moe.py:212` — `gen_trtllm_gen_fused_moe_sm100_module()` JIT compilation
- `flashinfer/flashinfer/fused_moe/core.py:930` — `get_trtllm_moe_sm100_module()` module loading with `@functools.cache`
