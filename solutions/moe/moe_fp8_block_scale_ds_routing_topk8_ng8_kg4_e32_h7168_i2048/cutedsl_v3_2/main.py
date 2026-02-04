import torch
from flashinfer.cute_dsl.moe_pipeline_v3_2 import cutedsl_fp8_moe_v3_2


NUM_EXPERTS_GLOBAL = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = 128


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
    output: torch.Tensor = None,
):
    seq_len, num_experts = routing_logits.shape
    local_num_experts = gemm1_weights.shape[0]

    assert num_experts == NUM_EXPERTS_GLOBAL
    assert hidden_states.shape == (seq_len, HIDDEN_SIZE)

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    else:
        local_expert_offset = int(local_expert_offset)

    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    return cutedsl_fp8_moe_v3_2(
        routing_logits=routing_logits.to(torch.float32).contiguous(),
        routing_bias=routing_bias.contiguous() if routing_bias is not None else None,
        hidden_states=hidden_states.contiguous(),
        hidden_states_scale=hidden_states_scale.to(torch.float32).contiguous(),
        gemm1_weights=gemm1_weights.contiguous(),
        gemm1_weights_scale=gemm1_weights_scale.to(torch.float32).contiguous(),
        gemm2_weights=gemm2_weights.contiguous(),
        gemm2_weights_scale=gemm2_weights_scale.to(torch.float32).contiguous(),
        num_experts_global=NUM_EXPERTS_GLOBAL,
        num_local_experts=local_num_experts,
        local_expert_offset=local_expert_offset,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE_SIZE,
        routed_scaling_factor=routed_scaling_factor,
        output=output,
    )
