"""Triton fused MoE: 6-kernel pipeline for DeepSeek-V3 FP8 blockwise MoE.

Kernel pipeline:
  K1: moe_fused_gate           — DeepSeek-V3 routing (sgl_kernel)
  K2: prepare_moe_input +      — Build expert offsets, scatter tokens by expert
      shuffle_rows               (sgl_kernel)
  K3: fp8_blockwise_scaled_    — GEMM1: FP8 [T*topk, H] x [E, H, 2I] -> BF16
      grouped_mm                 (sgl_kernel)
  K4: swiglu_quant_kernel      — Fused SwiGLU + FP8 quant (Triton)
  K5: fp8_blockwise_scaled_    — GEMM2: FP8 [T*topk, I] x [E, I, H] -> BF16
      grouped_mm                 (sgl_kernel)
  K6: apply_shuffle_mul_sum    — Weighted gather + accumulate (sgl_kernel)

Main optimization: K4 fuses SwiGLU activation and FP8 quantization into a
single Triton kernel, eliminating intermediate BF16 materialization and
reducing kernel launch overhead vs sglang v2's 2-3 separate kernels.
"""

import torch
from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    prepare_moe_input,
    shuffle_rows,
)

from .routing import route_tokens
from .swiglu_quant_kernel import swiglu_quant_kernel

# Fixed DeepSeek-V3/R1 geometry
E_LOCAL = 32
TOP_K = 8
H = 7168  # hidden_size
I = 2048  # intermediate_size
BLOCK = 128


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    device = hidden_states.device
    T = hidden_states.shape[0]

    # Convert scalar inputs
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())

    # --- K1: DeepSeek-V3 routing ---
    topk_weights, topk_ids = route_tokens(
        routing_logits, routing_bias, local_expert_offset, routed_scaling_factor
    )

    m = T
    topk = TOP_K
    k = H
    n = I
    num_experts = E_LOCAL

    # --- K2: Prepare MoE input + scatter ---
    expert_offsets = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((m * topk,), dtype=torch.int32, device=device)
    c_map = torch.empty((m * topk,), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    a_scale = hidden_states_scale.to(torch.float32).T.contiguous()
    rep_a_q = shuffle_rows(hidden_states.contiguous(), a_map, (m * topk, k))
    rep_a_scales = shuffle_rows(a_scale, a_map, (m * topk, k // BLOCK))

    # --- Weight transpose to SGLang convention ---
    w1_q = gemm1_weights.transpose(1, 2)
    w2_q = gemm2_weights.transpose(1, 2)
    w1_scale = gemm1_weights_scale.to(torch.float32).transpose(1, 2)
    w2_scale = gemm2_weights_scale.to(torch.float32).transpose(1, 2)

    # --- Allocate strides and scratch buffers ---
    ab_strides1 = torch.full((num_experts,), k, device=device, dtype=torch.int64)
    c_strides1 = torch.full((num_experts,), 2 * n, device=device, dtype=torch.int64)
    ab_strides2 = torch.full((num_experts,), n, device=device, dtype=torch.int64)
    c_strides2 = torch.full((num_experts,), k, device=device, dtype=torch.int64)

    workspace = torch.empty(90000, device=device, dtype=torch.uint8)
    a_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    b_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    out_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    a_scales_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    b_scales_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    a_sf_layout = torch.empty((num_experts, 5), dtype=torch.int32, device=device)
    w_sf_layout = torch.empty((num_experts, 5), dtype=torch.int32, device=device)

    # --- K3: GEMM1 — FP8 [T*topk, H] x [E, H, 2I] -> BF16 [T*topk, 2I] ---
    c1 = torch.empty((m * topk, 2 * n), device=device, dtype=torch.bfloat16)
    fp8_blockwise_scaled_grouped_mm(
        c1,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        rep_a_q,
        w1_q,
        rep_a_scales,
        w1_scale,
        ab_strides1,
        ab_strides1,
        c_strides1,
        a_sf_layout,
        w_sf_layout,
        problem_sizes1,
        expert_offsets[:-1],
        workspace,
    )

    # --- K4: Fused SwiGLU + FP8 quantization (Triton) ---
    M_total = m * topk
    intermediate_q = torch.empty((M_total, n), device=device, dtype=torch.float8_e4m3fn)
    a2_scale = torch.empty(
        (M_total, n // BLOCK), dtype=torch.float32, device=device
    )

    grid = (M_total,)
    swiglu_quant_kernel[grid](
        c1,
        intermediate_q,
        a2_scale,
        M_total,
        n,
        c1.stride(0),
        intermediate_q.stride(0),
        a2_scale.stride(0),
        GROUP_SIZE=BLOCK,
        BLOCK_N=BLOCK,
    )

    # --- K5: GEMM2 — FP8 [T*topk, I] x [E, I, H] -> BF16 [T*topk, H] ---
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)
    fp8_blockwise_scaled_grouped_mm(
        c2,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        intermediate_q,
        w2_q,
        a2_scale,
        w2_scale,
        ab_strides2,
        ab_strides2,
        c_strides2,
        a_sf_layout,
        w_sf_layout,
        problem_sizes2,
        expert_offsets[:-1],
        workspace,
    )

    # --- K6: Weighted gather + accumulate ---
    output = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(torch.bfloat16))

    return output
