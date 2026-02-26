#!/usr/bin/env python3
"""Quick smoke test for triton_fused_v3 MoE solution.

Usage:
    CUDA_LAUNCH_BLOCKING=1 python scripts/test_triton_v3.py
"""

import sys
import os

# Add solutions dir so triton_fused_v3 is importable as a package
SOL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "solutions",
    "moe",
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
)
sys.path.insert(0, os.path.abspath(SOL_DIR))

import torch
from triton_fused_v3.main import run

T, E, H, I, BLOCK = 4, 32, 7168, 2048, 128
device = "cuda:0"

print(f"Testing triton_fused_v3 with T={T}, E={E}, H={H}, I={I}")

out = run(
    routing_logits=torch.randn(T, 256, device=device, dtype=torch.float32),
    routing_bias=torch.zeros(256, device=device, dtype=torch.float32),
    hidden_states=torch.randn(T, H, device=device, dtype=torch.float8_e4m3fn),
    hidden_states_scale=torch.ones(H // BLOCK, T, device=device, dtype=torch.float32),
    gemm1_weights=torch.randn(E, 2 * I, H, device=device, dtype=torch.float8_e4m3fn),
    gemm1_weights_scale=torch.ones(
        E, 2 * I // BLOCK, H // BLOCK, device=device, dtype=torch.float32
    ),
    gemm2_weights=torch.randn(E, H, I, device=device, dtype=torch.float8_e4m3fn),
    gemm2_weights_scale=torch.ones(
        E, H // BLOCK, I // BLOCK, device=device, dtype=torch.float32
    ),
    local_expert_offset=0,
    routed_scaling_factor=2.5,
)
print(f"OK! Output shape: {out.shape}, dtype: {out.dtype}")
print(f"Output stats: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}")
