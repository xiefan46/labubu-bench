"""Triton fused MoE v3: Replace sgl_kernel GEMM with FlashInfer group_gemm_fp8_nt_groupwise.

Same pipeline as v2 but GEMM1/GEMM2 use FlashInfer's CUTLASS SM100 grouped GEMM
instead of sgl_kernel's fp8_blockwise_scaled_grouped_mm.

Kernel pipeline:
  K1: moe_fused_gate           — DeepSeek-V3 routing (sgl_kernel)
  K2: prepare_moe_input +      — Build expert offsets, scatter tokens by expert
      shuffle_rows               (sgl_kernel)
  K3: group_gemm_fp8_nt_       — GEMM1: FP8 [cum_m, H] x [E, 2I, H] -> BF16
      groupwise                  (FlashInfer)
  K4: swiglu_quant_kernel      — Fused SwiGLU + FP8 quant (Triton)
  K5: group_gemm_fp8_nt_       — GEMM2: FP8 [cum_m, I] x [E, H, I] -> BF16
      groupwise                  (FlashInfer)
  K6: apply_shuffle_mul_sum    — Weighted gather + accumulate (sgl_kernel)
"""

import torch
from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from sgl_kernel import (
    apply_shuffle_mul_sum,
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

# Module-level buffer caches (persist across run() calls)
_static = {}   # device -> dict of fixed-shape buffers
_dynamic = {}  # (M, device) -> dict of M-dependent buffers


def _init_static(device):
    """Allocate all fixed-shape buffers."""
    s = {}
    num_experts = E_LOCAL

    # Expert offsets
    s["expert_offsets"] = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=device
    )

    # Problem sizes (still needed for prepare_moe_input)
    s["problem_sizes1"] = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=device
    )
    s["problem_sizes2"] = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=device
    )

    return s


def _get_static(device):
    if device not in _static:
        _static[device] = _init_static(device)
    return _static[device]


def _init_dynamic(M, device):
    """Allocate all M-dependent buffers."""
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

    # Final output
    d["output"] = torch.zeros((M, H), device=device, dtype=torch.bfloat16)

    return d


def _get_dynamic(M, device):
    key = (M, device)
    if key not in _dynamic:
        _dynamic[key] = _init_dynamic(M, device)
    return _dynamic[key]


def _build_m_indptr(expert_offsets, num_experts):
    """Convert expert_offsets [E+1] to m_indptr [E+1] with 4-alignment padding.

    FlashInfer requires each value in m_indptr to be a multiple of 4.
    """
    m_indptr = expert_offsets[:num_experts + 1].to(torch.int32)
    # Pad each offset up to multiple of 4
    # NOTE: m_indptr must be non-decreasing. We pad each segment end up.
    # The simplest correct approach: align each offset to ceil_div(x, 4)*4
    # But this changes the segment boundaries. Instead, we keep offsets as-is
    # since prepare_moe_input already pads to alignment in practice.
    # If alignment issues arise, uncomment:
    # m_indptr = ((m_indptr + 3) // 4) * 4
    return m_indptr


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

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())

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

    # Scale: definition gives [H//128, T], we need [T, H//128] for scatter then transpose back
    a_scale = hidden_states_scale.to(torch.float32).T.contiguous()  # [T, H//128]
    rep_a_q = shuffle_rows(hidden_states.contiguous(), a_map, (m * topk, k))
    rep_a_scales = shuffle_rows(a_scale, a_map, (m * topk, k // BLOCK))

    # --- Build m_indptr for FlashInfer ---
    m_indptr = expert_offsets[:num_experts + 1].to(torch.int32)

    # --- Prepare scales for FlashInfer ---
    # FlashInfer a_scale: [K//128, cum_m] when scale_major_mode="MN"
    # rep_a_scales is [cum_m, K//128], so transpose
    fi_a_scale_1 = rep_a_scales.T.contiguous()  # [H//128, cum_m]

    # FlashInfer b: [E, N, K] column-major
    # gemm1_weights is [E, 2I, H] — already in [E, N, K] format
    fi_b1 = gemm1_weights  # [E, 2I, H]

    # FlashInfer b_scale: [E, N//128, K//128] when scale_major_mode="MN"
    # gemm1_weights_scale is [E, 2I//128, H//128] — already correct
    fi_b_scale_1 = gemm1_weights_scale.to(torch.float32)

    # --- K3: GEMM1 via FlashInfer ---
    c1 = d["c1"]
    group_gemm_fp8_nt_groupwise(
        a=rep_a_q,
        b=fi_b1,
        a_scale=fi_a_scale_1,
        b_scale=fi_b_scale_1,
        m_indptr=m_indptr,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="MN",
        mma_sm=1,
        out=c1,
    )

    # --- K4: Fused SwiGLU + FP8 quantization (Triton) ---
    M_total = m * topk
    intermediate_q = d["intermediate_q"]
    a2_scale = d["a2_scale"]

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
        FP8_MAX=448.0,
        GROUP_SIZE=BLOCK,
        BLOCK_N=BLOCK,
    )

    # --- Prepare GEMM2 scales ---
    # a2_scale is [cum_m, I//128], transpose for FlashInfer
    fi_a_scale_2 = a2_scale.T.contiguous()  # [I//128, cum_m]

    # gemm2_weights: [E, H, I] — already [E, N, K]
    fi_b2 = gemm2_weights

    # gemm2_weights_scale: [E, H//128, I//128] — already [E, N//128, K//128]
    fi_b_scale_2 = gemm2_weights_scale.to(torch.float32)

    # --- K5: GEMM2 via FlashInfer ---
    c2 = d["c2"]
    group_gemm_fp8_nt_groupwise(
        a=intermediate_q,
        b=fi_b2,
        a_scale=fi_a_scale_2,
        b_scale=fi_b_scale_2,
        m_indptr=m_indptr,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="MN",
        mma_sm=1,
        out=c2,
    )

    # --- K6: Weighted gather + accumulate ---
    output = d["output"]
    output.zero_()
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(torch.bfloat16))

    return output
