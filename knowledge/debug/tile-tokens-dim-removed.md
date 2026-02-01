# tile_tokens_dim Parameter Removed from FlashInfer API

## Problem

The `flashinfer_moe` solution in flashinfer-trace passes `tile_tokens_dim` kwarg to `trtllm_fp8_block_scale_moe`, but this parameter has been removed from the API.

## History

1. **PR #1980** (commit bb6b6208): Added autotuner that deprecated manual `tile_tokens_dim` specification
2. **PR #2086** (commit 9a79b786): Fully removed `tile_tokens_dim` from the public API

## Solution

Patched in `setup.sh` — removes `tile_tokens_dim` related code from the solution JSON:
- Removes `_next_power_of_2` helper
- Removes `_get_tile_tokens_dim` helper
- Removes `tile_tokens_dim` local variable assignment
- Removes `tile_tokens_dim=tile_tokens_dim` kwarg in function call

## Key Files

- `setup.sh:82-120` — patch logic
- `flashinfer-trace/solutions/moe/.../flashinfer_wrapper_9sdjf3.json` — solution JSON
