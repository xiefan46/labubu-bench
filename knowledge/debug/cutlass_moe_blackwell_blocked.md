# CUTLASS MoE FP8 Block-Scale: Blackwell (SM100) Not Supported

## Date: 2026-02-01

## Summary

Attempted to create a CUTLASS-based MoE solution using `cutlass_fused_moe` with `use_deepseek_fp8_block_scale=True`. Found it is explicitly blocked for Blackwell (B200/SM100) GPUs.

## Blocking Code

`flashinfer/fused_moe/core.py:867-871`:

```python
if use_deepseek_fp8_block_scale:
    if device_arch != "90":
        raise NotImplementedError("FP8 block scaling not yet implemented for Blackwell.")
```

Only SM90 (Hopper/H100) is supported. SM100 (Blackwell/B200) raises `NotImplementedError`.

## Decision

Skipped this approach. No implementation made.

## Revisit Condition

When FlashInfer adds Blackwell support for `cutlass_fused_moe` with FP8 block-scale, revisit this plan.
