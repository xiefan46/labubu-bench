"""Fused MoE v3: sgl_kernel fused SwiGLU+Quant replacing Triton kernel.

Same pipeline as triton_fused_v2 (buffer pre-allocation), but K4 uses
sgl_kernel's native CUDA fused SwiGLU+FP8 quantization instead of
the custom Triton kernel. A/B test to compare kernel implementations.

Kernel pipeline:
  K1: moe_fused_gate           — DeepSeek-V3 routing (sgl_kernel)
  K2: prepare_moe_input +      — Build expert offsets, scatter tokens by expert
      shuffle_rows               (sgl_kernel)
  K3: fp8_blockwise_scaled_    — GEMM1: FP8 [T*topk, H] x [E, H, 2I] -> BF16
      grouped_mm                 (sgl_kernel)
  K4: sgl_per_token_group_     — Fused SwiGLU + FP8 quant (sgl_kernel CUDA)
      quant_fp8 (fused)
  K5: fp8_blockwise_scaled_    — GEMM2: FP8 [T*topk, I] x [E, I, H] -> BF16
      grouped_mm                 (sgl_kernel)
  K6: apply_shuffle_mul_sum    — Weighted gather + accumulate (sgl_kernel)
"""

import torch
from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    prepare_moe_input,
    sgl_per_token_group_quant_fp8,
    shuffle_rows,
)

from .routing import route_tokens

# Fixed DeepSeek-V3/R1 geometry
E_LOCAL = 32
TOP_K = 8
H = 7168  # hidden_size
I = 2048  # intermediate_size
BLOCK = 128
FP8_MIN = -448.0
FP8_MAX = 448.0

# Module-level buffer caches (persist across run() calls)
_static = {}   # device -> dict of fixed-shape buffers
_dynamic = {}  # (M, device) -> dict of M-dependent buffers


def _init_static(device):
    """Allocate all fixed-shape buffers (15 tensors, independent of M)."""
    s = {}
    num_experts = E_LOCAL

    # Stride arrays for grouped GEMM
    s["ab_strides1"] = torch.full((num_experts,), H, device=device, dtype=torch.int64)
    s["c_strides1"] = torch.full(
        (num_experts,), 2 * I, device=device, dtype=torch.int64
    )
    s["ab_strides2"] = torch.full((num_experts,), I, device=device, dtype=torch.int64)
    s["c_strides2"] = torch.full((num_experts,), H, device=device, dtype=torch.int64)

    # Workspace for grouped GEMM
    s["workspace"] = torch.empty(90000, device=device, dtype=torch.uint8)

    # Pointer arrays
    s["a_ptrs"] = torch.empty((num_experts,), dtype=torch.int64, device=device)
    s["b_ptrs"] = torch.empty((num_experts,), dtype=torch.int64, device=device)
    s["out_ptrs"] = torch.empty((num_experts,), dtype=torch.int64, device=device)
    s["a_scales_ptrs"] = torch.empty((num_experts,), dtype=torch.int64, device=device)
    s["b_scales_ptrs"] = torch.empty((num_experts,), dtype=torch.int64, device=device)

    # Scale factor layouts
    s["a_sf_layout"] = torch.empty((num_experts, 5), dtype=torch.int32, device=device)
    s["w_sf_layout"] = torch.empty((num_experts, 5), dtype=torch.int32, device=device)

    # Expert offsets
    s["expert_offsets"] = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=device
    )

    # Problem sizes
    s["problem_sizes1"] = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=device
    )
    s["problem_sizes2"] = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=device
    )

    return s


def _get_static(device):
    """Get or create fixed-shape buffers for the given device."""
    if device not in _static:
        _static[device] = _init_static(device)
    return _static[device]


def _init_dynamic(M, device):
    """Allocate all M-dependent buffers (7 tensors)."""
    d = {}
    MT = M * TOP_K

    # Scatter/gather maps
    d["a_map"] = torch.empty((MT,), dtype=torch.int32, device=device)
    d["c_map"] = torch.empty((MT,), dtype=torch.int32, device=device)

    # GEMM1 output
    d["c1"] = torch.empty((MT, 2 * I), device=device, dtype=torch.bfloat16)

    # SwiGLU+quant outputs
    d["intermediate_q"] = torch.empty(
        (MT, I), device=device, dtype=torch.float8_e4m3fn
    )
    d["a2_scale"] = torch.empty((MT, I // BLOCK), dtype=torch.float32, device=device)

    # GEMM2 output
    d["c2"] = torch.empty((MT, H), device=device, dtype=torch.bfloat16)

    # Final output (needs zero_() each call)
    d["output"] = torch.zeros((M, H), device=device, dtype=torch.bfloat16)

    return d


def _get_dynamic(M, device):
    """Get or create M-dependent buffers."""
    key = (M, device)
    if key not in _dynamic:
        _dynamic[key] = _init_dynamic(M, device)
    return _dynamic[key]


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

    # --- Get cached buffers ---
    s = _get_static(device)
    d = _get_dynamic(T, device)

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
    expert_offsets = s["expert_offsets"]
    problem_sizes1 = s["problem_sizes1"]
    problem_sizes2 = s["problem_sizes2"]
    a_map = d["a_map"]
    c_map = d["c_map"]

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

    # --- Cached strides and scratch buffers ---
    ab_strides1 = s["ab_strides1"]
    c_strides1 = s["c_strides1"]
    ab_strides2 = s["ab_strides2"]
    c_strides2 = s["c_strides2"]
    workspace = s["workspace"]
    a_ptrs = s["a_ptrs"]
    b_ptrs = s["b_ptrs"]
    out_ptrs = s["out_ptrs"]
    a_scales_ptrs = s["a_scales_ptrs"]
    b_scales_ptrs = s["b_scales_ptrs"]
    a_sf_layout = s["a_sf_layout"]
    w_sf_layout = s["w_sf_layout"]

    # --- K3: GEMM1 — FP8 [T*topk, H] x [E, H, 2I] -> BF16 [T*topk, 2I] ---
    c1 = d["c1"]
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

    # --- K4: Fused SwiGLU + FP8 quantization (sgl_kernel CUDA) ---
    intermediate_q = d["intermediate_q"]
    a2_scale = d["a2_scale"]

    sgl_per_token_group_quant_fp8(
        c1,
        intermediate_q,
        a2_scale,
        BLOCK,
        1e-10,
        FP8_MIN,
        FP8_MAX,
        fuse_silu_and_mul=True,
        enable_v2=True,
    )

    # --- K5: GEMM2 — FP8 [T*topk, I] x [E, I, H] -> BF16 [T*topk, H] ---
    c2 = d["c2"]
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
    output = d["output"]
    output.zero_()
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(torch.bfloat16))

    return output
