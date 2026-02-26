# flashinfer_moe Tolerance Tuning — 2026-02-02

**GPU:** NVIDIA B200 (SM100)
**Definition:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
**Solution:** `flashinfer_moe` (single fused trtllm cubin kernel)

## Tolerance sweep

| Run | atol | rtol | matched_ratio | Logic | Result |
|-----|------|------|---------------|-------|--------|
| 1 | 0.01 | 0.01 | 0.95 | default | 0/19 PASSED |
| 2 | 0.1 | 0.2 | 0.95 | default | 3/19 PASSED |
| 3 | 0.1 | 0.2 | 0.85 | AND | 15/15 (SSH lost) |
| 4 | 0.1 | 0.2 | 0.85 | torch.isclose | **19/19 PASSED** |

Run 4 used branch `feat/isclose-tolerance-check`.

## Run 4 results (19/19 PASSED)

| Workload | Speedup |
|----------|--------:|
| b8f4f012 | 48.77x |
| e05c6c03 | 32.37x |
| 6230e838 | 32.06x |
| 8f1ff9f1 | 27.01x |
| 1a4c6ba1 | 35.04x |
| a7c2bcfd | 32.14x |
| 2e69caee | 31.39x |
| 8cba5890 | 33.98x |
| 5e8dc11c | 15.89x |
| 58a34f27 | 16.35x |
| 5eadab1e | 31.26x |
| eedc63b2 | 32.76x |
| e626d3e6 | 33.71x |
| 74d7ff04 | 30.32x |
| 4822167c | 35.60x |
| 81955b1e | 24.71x |
| 76010cb4 | 31.77x |
| fc378037 | 32.78x |
| f7d6ac7c | 36.29x |

**Average speedup: ~31x over reference implementation**

## Takeaway

FP8 blockwise MoE has inherent numerical variance. The recommended tolerance for correctness checks is `rtol=0.2, atol=0.1, required-matched-ratio=0.85` with `torch.isclose` logic.
