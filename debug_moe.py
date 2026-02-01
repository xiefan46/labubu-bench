"""Debug script to isolate INCORRECT_NUMERICAL issues in sglang_fp8_blockwise_moe.

Run on GPU server:
    python debug_moe.py

Tests each component independently against the reference implementation.
"""
import torch
torch.manual_seed(42)

# ============================================================
# 1. Test routing: moe_fused_gate vs reference
# ============================================================
def test_routing():
    from sgl_kernel import moe_fused_gate

    T, E_GLOBAL = 8, 256
    N_GROUP, TOPK_GROUP, TOP_K = 8, 4, 8
    routed_scaling_factor = 2.5

    logits = torch.randn(T, E_GLOBAL, dtype=torch.float32, device="cuda")
    bias = torch.randn(E_GLOBAL, dtype=torch.float32, device="cuda")

    # --- Reference routing ---
    s = torch.sigmoid(logits)
    s_with_bias = s + bias
    group_size = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, ref_topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, ref_topk_idx, 1.0)
    ref_weights = s * M
    ref_weights_sum = ref_weights.sum(dim=1, keepdim=True) + 1e-20
    ref_weights = (ref_weights / ref_weights_sum) * routed_scaling_factor

    # Gather per-topk weights for comparison
    ref_topk_weights = ref_weights.gather(1, ref_topk_idx)

    # --- sgl_kernel routing ---
    sgl_topk_weights, sgl_topk_ids = moe_fused_gate(
        logits.contiguous(),
        bias.contiguous(),
        N_GROUP, TOPK_GROUP, TOP_K,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )

    # Compare: expert selections
    ref_experts_set = [set(ref_topk_idx[t].tolist()) for t in range(T)]
    sgl_experts_set = [set(sgl_topk_ids[t].tolist()) for t in range(T)]
    expert_match = sum(1 for a, b in zip(ref_experts_set, sgl_experts_set) if a == b)

    print("=" * 60)
    print("ROUTING TEST")
    print("=" * 60)
    print(f"Expert selection match: {expert_match}/{T} tokens")

    # Compare weights (need to align by expert ID)
    for t in range(min(3, T)):
        print(f"\nToken {t}:")
        print(f"  Ref experts:  {sorted(ref_topk_idx[t].tolist())}")
        print(f"  SGL experts:  {sorted(sgl_topk_ids[t].tolist())}")
        # Find common experts and compare weights
        common = ref_experts_set[t] & sgl_experts_set[t]
        if common:
            for eid in sorted(common):
                ref_w = ref_weights[t, eid].item()
                sgl_idx = (sgl_topk_ids[t] == eid).nonzero(as_tuple=True)[0]
                sgl_w = sgl_topk_weights[t, sgl_idx].item() if len(sgl_idx) > 0 else 0
                print(f"  Expert {eid}: ref_w={ref_w:.6f}, sgl_w={sgl_w:.6f}, diff={abs(ref_w - sgl_w):.6e}")

    # Overall weight comparison (sum of weights per token)
    ref_sum = ref_topk_weights.sum(dim=1)
    sgl_sum = sgl_topk_weights.sum(dim=1)
    print(f"\nWeight sums per token:")
    print(f"  Ref:  {ref_sum[:4].tolist()}")
    print(f"  SGL:  {sgl_sum[:4].tolist()}")
    print(f"  Expected (routed_scaling_factor): {routed_scaling_factor}")

    return expert_match == T


# ============================================================
# 2. Test SwiGLU order
# ============================================================
def test_swiglu():
    from sgl_kernel import silu_and_mul

    n = 64
    x = torch.randn(4, 2 * n, dtype=torch.bfloat16, device="cuda")

    # sgl_kernel
    out_sgl = torch.empty(4, n, dtype=torch.bfloat16, device="cuda")
    silu_and_mul(x, out_sgl)

    # Reference: silu(second_half) * first_half
    X1 = x[:, :n].float()
    X2 = x[:, n:].float()
    ref_out = (torch.nn.functional.silu(X2) * X1).to(torch.bfloat16)

    # sgl_kernel convention: silu(first_half) * second_half
    sgl_convention = (torch.nn.functional.silu(X1) * X2).to(torch.bfloat16)

    match_ref = torch.allclose(out_sgl.float(), ref_out.float(), atol=1e-3, rtol=1e-3)
    match_sgl = torch.allclose(out_sgl.float(), sgl_convention.float(), atol=1e-3, rtol=1e-3)

    print("\n" + "=" * 60)
    print("SWIGLU ORDER TEST")
    print("=" * 60)
    print(f"silu_and_mul matches silu(second)*first (reference): {match_ref}")
    print(f"silu_and_mul matches silu(first)*second (sgl convention): {match_sgl}")

    if match_sgl and not match_ref:
        print("=> CONFIRMED: Need to swap halves before calling silu_and_mul")

    # Test with swapped input
    x_swapped = torch.cat([x[:, n:], x[:, :n]], dim=1)
    out_swapped = torch.empty(4, n, dtype=torch.bfloat16, device="cuda")
    silu_and_mul(x_swapped, out_swapped)
    match_swapped = torch.allclose(out_swapped.float(), ref_out.float(), atol=1e-3, rtol=1e-3)
    print(f"After swap, silu_and_mul matches reference: {match_swapped}")

    return match_swapped


# ============================================================
# 3. End-to-end small test with random data
# ============================================================
def test_e2e_small():
    """Small end-to-end test comparing our solution vs reference implementation."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "projects/flashinfer-bench/flashinfer_trace/tests/references"))

    from test_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 import (
        run as ref_run,
        generate_random_inputs_moe,
    )

    device = "cuda"
    seq_len = 4
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=256,
        num_local_experts=32,
        hidden_size=7168,
        intermediate_size=2048,
        use_bias=True,
        local_expert_offset=0,
        routed_scaling_factor=2.5,
        device=device,
    )

    # Reference output
    ref_out = ref_run(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    # Load and run our solution
    import json
    sol_dir = "flashinfer-trace/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    sol_path = os.path.join(sol_dir, "sglang_fp8_blockwise_moe.json")
    if not os.path.exists(sol_path):
        print(f"\nSolution JSON not found at {sol_path}, skipping e2e test")
        return False

    sol = json.load(open(sol_path))
    code = sol["sources"][0]["content"]

    # Execute solution code to get the run function
    ns = {}
    exec(code, ns)
    sol_run = ns["run"]

    sol_out = sol_run(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=inputs["local_expert_offset"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
    )

    ref_f32 = ref_out.float()
    sol_f32 = sol_out.float()

    abs_diff = (ref_f32 - sol_f32).abs()
    cos_sim = torch.nn.functional.cosine_similarity(ref_f32.flatten(), sol_f32.flatten(), dim=0).item()

    atol, rtol = 0.1, 0.2
    ok = (abs_diff <= atol + rtol * sol_f32.abs())
    hit_ratio = ok.float().mean().item()

    print("\n" + "=" * 60)
    print("END-TO-END TEST (small)")
    print("=" * 60)
    print(f"Ref output range: [{ref_f32.min().item():.4f}, {ref_f32.max().item():.4f}]")
    print(f"Sol output range: [{sol_f32.min().item():.4f}, {sol_f32.max().item():.4f}]")
    print(f"Max abs diff:  {abs_diff.max().item():.6e}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Hit ratio: {hit_ratio * 100:.2f}%  (need >= 85%)")

    # Check if output is all zeros
    if sol_f32.abs().max().item() < 1e-6:
        print("WARNING: Solution output is all zeros!")
    if ref_f32.abs().max().item() < 1e-6:
        print("WARNING: Reference output is all zeros!")

    return hit_ratio >= 0.85


if __name__ == "__main__":
    print("Testing sglang_fp8_blockwise_moe components...\n")

    routing_ok = test_routing()
    swiglu_ok = test_swiglu()

    try:
        e2e_ok = test_e2e_small()
    except Exception as e:
        print(f"\nE2E test failed with error: {e}")
        import traceback
        traceback.print_exc()
        e2e_ok = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Routing:  {'PASS' if routing_ok else 'FAIL'}")
    print(f"SwiGLU:   {'PASS' if swiglu_ok else 'FAIL'}")
    print(f"E2E:      {'PASS' if e2e_ok else 'FAIL'}")
