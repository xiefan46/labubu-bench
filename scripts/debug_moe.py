#!/usr/bin/env python3
"""Compare trtllm_fp8_block_scale_moe output against reference implementation."""
import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

# ── Constants (DeepSeek-V3 geometry) ──
T = 7  # seq_len
E_global = 256
E_local = 32
H = 7168
I = 2048
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4

torch.manual_seed(42)

# ── Generate inputs ──
routing_logits = torch.full((T, E_global), -10.0, dtype=torch.float32, device="cuda")
# Force routing to select local experts 0..7 (all in group 0) by giving them high logits
for i in range(TOP_K):
    routing_logits[:, i] = 10.0 + float(i)
routing_bias = torch.zeros(E_global, dtype=torch.bfloat16, device="cuda")
hidden_states = torch.randn(T, H, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
hidden_states_scale = torch.randn(H // BLOCK, T, dtype=torch.float32, device="cuda").abs() + 0.01
gemm1_weights = torch.randn(E_local, 2 * I, H, dtype=torch.float32, device="cuda").to(
    torch.float8_e4m3fn
)
gemm1_weights_scale = (
    torch.randn(E_local, (2 * I) // BLOCK, H // BLOCK, dtype=torch.float32, device="cuda").abs()
    + 0.01
)
gemm2_weights = torch.randn(E_local, H, I, dtype=torch.float32, device="cuda").to(
    torch.float8_e4m3fn
)
gemm2_weights_scale = (
    torch.randn(E_local, H // BLOCK, I // BLOCK, dtype=torch.float32, device="cuda").abs() + 0.01
)
local_expert_offset = 0
routed_scaling_factor = 1.0


# ── Reference implementation (from definition JSON) ──
@torch.no_grad()
def reference_run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    device = hidden_states.device

    # 1) FP8 block-scale dequantization
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)  # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1).repeat(1, 1, BLOCK).reshape(T, H).contiguous()
    )
    A = A_fp32 * A_scale_expanded  # [T, H]

    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)
    W13 = W13_fp32 * S13_expanded

    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)
    W2 = W2_fp32 * S2_expanded

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # 3) Local expert compute
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue
        sel_mask_per_token = (topk_idx == ge).any(dim=1)
        if not sel_mask_per_token.any():
            continue
        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)
        W13_e = W13[le]
        W2_e = W2[le]
        G1 = A_e.matmul(W13_e.t())
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1
        O = C.matmul(W2_e.t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


# ── Run reference ──
print("Running reference...")
ref_out = reference_run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
)
print(f"  ref shape={ref_out.shape}, dtype={ref_out.dtype}")
print(f"  ref abs_max={ref_out.float().abs().max():.4f}, abs_mean={ref_out.float().abs().mean():.4f}")

# ── Run solution (flashinfer kernel) ──
print("\nRunning flashinfer kernel...")
sol_out = trtllm_fp8_block_scale_moe(
    routing_logits.contiguous(),
    routing_bias.contiguous(),
    hidden_states.contiguous(),
    hidden_states_scale.to(torch.float32).contiguous(),
    gemm1_weights.contiguous(),
    gemm1_weights_scale.to(torch.float32).contiguous(),
    gemm2_weights.contiguous(),
    gemm2_weights_scale.to(torch.float32).contiguous(),
    E_global,
    TOP_K,
    N_GROUP,
    TOPK_GROUP,
    I,
    local_expert_offset,
    E_local,
    routed_scaling_factor,
    routing_method_type=2,
    use_shuffled_weight=False,
)
print(f"  sol shape={sol_out.shape}, dtype={sol_out.dtype}")
print(f"  sol abs_max={sol_out.float().abs().max():.4f}, abs_mean={sol_out.float().abs().mean():.4f}")

# ── Compare ──
print("\n── Comparison ──")
ref_f = ref_out.float()
sol_f = sol_out.float()
abs_err = (ref_f - sol_f).abs()
rel_err = abs_err / (ref_f.abs() + 1e-8)

print(f"max_abs_error: {abs_err.max():.4f}")
print(f"mean_abs_error: {abs_err.mean():.4f}")
print(f"max_rel_error: {rel_err.max():.4f}")
print(f"mean_rel_error: {rel_err.mean():.4f}")

# Per-element tolerance check (same as benchmark: both atol AND rtol must exceed)
atol, rtol = 0.01, 0.01
exceeds = (abs_err > atol) & (rel_err > rtol)
matched_ratio = 1.0 - exceeds.float().mean().item()
print(f"matched_ratio: {matched_ratio:.4f}")

# Show first few mismatched values
if exceeds.any():
    idxs = torch.nonzero(exceeds, as_tuple=False)[:5]
    print(f"\nFirst {len(idxs)} mismatched positions:")
    for idx in idxs:
        i, j = idx[0].item(), idx[1].item()
        print(
            f"  [{i},{j}] ref={ref_f[i,j]:.6f} sol={sol_f[i,j]:.6f} "
            f"abs={abs_err[i,j]:.6f} rel={rel_err[i,j]:.6f}"
        )
