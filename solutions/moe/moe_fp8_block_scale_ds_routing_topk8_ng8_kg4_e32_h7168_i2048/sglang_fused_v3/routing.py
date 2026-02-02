"""DeepSeek-V3 routing with local expert remapping.

Uses sgl_kernel's moe_fused_gate for the fused sigmoid+bias+group+topk routing,
then maps global expert IDs to local [0, E_LOCAL) range.
"""

import torch
from sgl_kernel import moe_fused_gate

# Fixed DeepSeek-V3/R1 geometry
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4


def route_tokens(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """Compute routing weights and local expert assignments.

    Returns:
        topk_weights: [T, TOP_K] BF16 — routing weights (scaled, zeroed for non-local)
        topk_ids: [T, TOP_K] INT32 — local expert IDs in [0, E_LOCAL)
    """
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

    # Map global expert IDs to local [0, E_LOCAL)
    local_mask = (topk_ids >= local_expert_offset) & (
        topk_ids < local_expert_offset + E_LOCAL
    )
    local_ids = (topk_ids - local_expert_offset).clamp(0, E_LOCAL - 1)
    topk_weights = topk_weights * local_mask.to(topk_weights.dtype)
    topk_ids = local_ids.to(torch.int32)

    return topk_weights, topk_ids
