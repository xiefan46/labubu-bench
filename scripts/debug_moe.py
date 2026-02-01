#!/usr/bin/env python3
"""Quick sanity check for trtllm_fp8_block_scale_moe output."""
import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

seq_len = 7
num_experts = 256
local_num_experts = 32
hidden_size = 7168
intermediate_size = 2048
block_size = 128

routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32, device="cuda")
routing_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
hidden_states = torch.randn(seq_len, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")
hidden_states_scale = torch.randn(
    hidden_size // block_size, seq_len, dtype=torch.float32, device="cuda"
).abs()
gemm1_weights = torch.randn(
    local_num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float8_e4m3fn, device="cuda"
)
gemm1_weights_scale = torch.randn(
    local_num_experts,
    (2 * intermediate_size) // block_size,
    hidden_size // block_size,
    dtype=torch.float32,
    device="cuda",
).abs()
gemm2_weights = torch.randn(
    local_num_experts, hidden_size, intermediate_size, dtype=torch.float8_e4m3fn, device="cuda"
)
gemm2_weights_scale = torch.randn(
    local_num_experts,
    hidden_size // block_size,
    intermediate_size // block_size,
    dtype=torch.float32,
    device="cuda",
).abs()

out = trtllm_fp8_block_scale_moe(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    num_experts,
    8,  # top_k
    8,  # n_group
    4,  # topk_group
    intermediate_size,
    0,  # local_expert_offset
    local_num_experts,
    1.0,  # routed_scaling_factor
    routing_method_type=2,
    use_shuffled_weight=False,
)
print(f"output shape: {out.shape}, dtype: {out.dtype}")
print(f"has nan: {out.isnan().any()}, has inf: {out.isinf().any()}")
print(f"abs max: {out.abs().max():.4f}, abs mean: {out.abs().mean():.4f}")
