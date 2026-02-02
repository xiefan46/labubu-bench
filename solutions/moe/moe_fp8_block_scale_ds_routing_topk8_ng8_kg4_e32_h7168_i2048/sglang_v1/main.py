import torch
from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    moe_fused_gate,
    prepare_moe_input,
    shuffle_rows,
    silu_and_mul,
)

# Fixed DeepSeek-V3/R1 geometry
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
H = 7168       # hidden_size
I = 2048       # intermediate_size
BLOCK = 128
FP8_MAX = 448.0


def _per_token_group_quant_fp8(x, group_size=128):
    """Quantize bf16/fp16 tensor to FP8 E4M3 with per-token-group scales."""
    shape = x.shape
    x_grouped = x.reshape(-1, shape[-1] // group_size, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / FP8_MAX
    x_q = (x_grouped / scale).reshape(shape).to(torch.float8_e4m3fn)
    scales = scale.squeeze(-1).to(torch.float32)
    return x_q, scales


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

    # --- Step 1: DeepSeek-V3 routing via moe_fused_gate ---
    topk_weights, topk_ids = moe_fused_gate(
        routing_logits.to(torch.float32).contiguous(),
        routing_bias.to(torch.float32).contiguous(),
        N_GROUP,
        TOPK_GROUP,
        TOP_K,
        num_fused_shared_experts=0,
        routed_scaling_factor=float(routed_scaling_factor),
        apply_routed_scaling_factor_on_output=False,
    )

    topk_weights = topk_weights * routed_scaling_factor

    # --- Step 2: Map global expert IDs to local [0, E_LOCAL) ---
    local_mask = (topk_ids >= local_expert_offset) & (
        topk_ids < local_expert_offset + E_LOCAL
    )
    local_ids = (topk_ids - local_expert_offset).clamp(0, E_LOCAL - 1)
    topk_weights = topk_weights * local_mask.to(topk_weights.dtype)
    topk_ids = local_ids.to(torch.int32)

    m = T
    topk = TOP_K
    k = H
    n = I
    num_experts = E_LOCAL

    # --- Step 3: Prepare MoE input ---
    expert_offsets = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((m * topk,), dtype=torch.int32, device=device)
    c_map = torch.empty((m * topk,), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
        a_map, c_map, num_experts, n, k,
    )

    # --- Step 4: Prepare activations ---
    a_scale = hidden_states_scale.to(torch.float32).T.contiguous()
    rep_a_q = shuffle_rows(hidden_states.contiguous(), a_map, (m * topk, k))
    rep_a_scales = shuffle_rows(a_scale, a_map, (m * topk, k // BLOCK))

    # --- Step 5: Transpose weights to SGLang convention ---
    w1_q = gemm1_weights.transpose(1, 2)
    w2_q = gemm2_weights.transpose(1, 2)
    w1_scale = gemm1_weights_scale.to(torch.float32).transpose(1, 2)
    w2_scale = gemm2_weights_scale.to(torch.float32).transpose(1, 2)

    # --- Step 6: Allocate strides and scratch buffers ---
    ab_strides1 = torch.full((num_experts,), k, device=device, dtype=torch.int64)
    c_strides1 = torch.full((num_experts,), 2 * n, device=device, dtype=torch.int64)
    ab_strides2 = torch.full((num_experts,), n, device=device, dtype=torch.int64)
    c_strides2 = torch.full((num_experts,), k, device=device, dtype=torch.int64)

    workspace = torch.empty(1024 * 1024 * 1024, device=device, dtype=torch.uint8)
    a_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    b_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    out_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    a_scales_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    b_scales_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=device)
    a_sf_layout = torch.empty((num_experts, 5), dtype=torch.int32, device=device)
    w_sf_layout = torch.empty((num_experts, 5), dtype=torch.int32, device=device)

    # --- Step 7: GEMM1 ---
    c1 = torch.empty((m * topk, 2 * n), device=device, dtype=torch.bfloat16)
    fp8_blockwise_scaled_grouped_mm(
        c1,
        a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        rep_a_q, w1_q, rep_a_scales, w1_scale,
        ab_strides1, ab_strides1, c_strides1,
        a_sf_layout, w_sf_layout,
        problem_sizes1, expert_offsets[:-1],
        workspace,
    )

    # --- Step 8: SwiGLU activation (torch.cat swap + silu_and_mul) ---
    c1_swapped = torch.cat([c1[:, n:], c1[:, :n]], dim=1)
    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.bfloat16)
    silu_and_mul(c1_swapped, intermediate)

    # --- Step 9: Quantize intermediate to FP8 for GEMM2 (PyTorch) ---
    intermediate_q, a2_scale = _per_token_group_quant_fp8(intermediate, BLOCK)

    # --- Step 10: GEMM2 ---
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)
    fp8_blockwise_scaled_grouped_mm(
        c2,
        a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        intermediate_q, w2_q, a2_scale, w2_scale,
        ab_strides2, ab_strides2, c_strides2,
        a_sf_layout, w_sf_layout,
        problem_sizes2, expert_offsets[:-1],
        workspace,
    )

    # --- Step 11: Weighted sum of expert outputs ---
    output = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(torch.bfloat16))

    return output
