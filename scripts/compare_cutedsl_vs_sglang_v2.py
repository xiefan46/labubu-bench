"""
Stage-by-stage performance and precision comparison:
  CuTeDSL MoE pipeline  vs  SGLang v2 MoE pipeline

Both pipelines are decomposed into 5 aligned stages:
  1. Routing     — expert selection + index computation
  2. GEMM1       — FP8 grouped GEMM (incl. A-gather/shuffle)
  3. SwiGLU+Req  — activation + requantization
  4. GEMM2       — FP8 grouped GEMM
  5. Finalize    — weighted reduce / scatter-add

Usage:
    python scripts/compare_cutedsl_vs_sglang_v2.py [--seq-len 1024]
"""

import argparse
import time

import torch
import torch.nn.functional as F

# ── CuTeDSL imports ──
from flashinfer.cute_dsl.moe_activation import moe_swiglu_fp8_requant
from flashinfer.cute_dsl.moe_finalize import moe_finalize
from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_gemm1_fp8, moe_gemm2_fp8
from flashinfer.cute_dsl.moe_pipeline import (
    allocate_moe_workspace,
    cutedsl_fp8_moe,
    moe_routing_deepseek,
)

# ── SGLang imports ──
from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    moe_fused_gate,
    prepare_moe_input,
    sgl_per_token_group_quant_fp8,
    shuffle_rows,
)

# ── Constants (DeepSeek-V3) ──
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
H = 7168
I = 2048
BLOCK = 128
FP8_MIN, FP8_MAX = -448.0, 448.0


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def cuda_time(fn, warmup=3, iters=10):
    """Median CUDA-event time in ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def make_inputs(T, device):
    """Create random MoE inputs.

    Biases routing logits toward local experts (0..E_LOCAL-1) so they
    actually receive tokens.  Without this, DeepSeek-V3 group routing
    with 256 experts rarely selects the first 32.
    """
    routing_logits = torch.randn(T, E_GLOBAL, dtype=torch.float32, device=device)
    # Boost local expert logits so routing selects them
    routing_logits[:, :E_LOCAL] += 5.0
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(T, H, device=device).to(torch.float8_e4m3fn)
    hs_scale = torch.randn(H // BLOCK, T, dtype=torch.float32, device=device).abs() + 0.01
    g1w = torch.randn(E_LOCAL, 2 * I, H, device=device).to(torch.float8_e4m3fn)
    g1ws = torch.randn(E_LOCAL, 2 * I // BLOCK, H // BLOCK, dtype=torch.float32, device=device).abs() + 0.01
    g2w = torch.randn(E_LOCAL, H, I, device=device).to(torch.float8_e4m3fn)
    g2ws = torch.randn(E_LOCAL, H // BLOCK, I // BLOCK, dtype=torch.float32, device=device).abs() + 0.01
    return routing_logits, routing_bias, hidden_states, hs_scale, g1w, g1ws, g2w, g2ws


# ═══════════════════════════════════════════════════════════════════
# SGLang v2 — stage decomposition
# ═══════════════════════════════════════════════════════════════════


def sglang_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T):
    """Stage 1: Routing (moe_fused_gate + local masking + prepare_moe_input)."""
    topk_weights, topk_ids = moe_fused_gate(
        routing_logits.to(torch.float32).contiguous(),
        routing_bias.to(torch.float32).contiguous(),
        N_GROUP, TOPK_GROUP, TOP_K,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )
    topk_weights = topk_weights * routed_scaling_factor

    local_mask = (topk_ids >= 0) & (topk_ids < E_LOCAL)
    local_ids = (topk_ids - 0).clamp(0, E_LOCAL - 1)
    topk_weights = topk_weights * local_mask.to(topk_weights.dtype)
    topk_ids = local_ids.to(torch.int32)

    expert_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E_LOCAL, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E_LOCAL, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((T * TOP_K,), dtype=torch.int32, device=device)
    c_map = torch.empty((T * TOP_K,), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
        a_map, c_map, E_LOCAL, I, H,
    )
    return topk_weights, topk_ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map


def sglang_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, a_map, problem_sizes1, expert_offsets, T, device):
    """Stage 2: GEMM1 (shuffle_rows + grouped GEMM)."""
    a_scale = hs_scale.to(torch.float32).T.contiguous()
    rep_a_q = shuffle_rows(hidden_states.contiguous(), a_map, (T * TOP_K, H))
    rep_a_scales = shuffle_rows(a_scale, a_map, (T * TOP_K, H // BLOCK))

    w1_q = g1w.transpose(1, 2)
    w1_scale = g1ws.to(torch.float32).transpose(1, 2)

    ab_strides1 = torch.full((E_LOCAL,), H, device=device, dtype=torch.int64)
    c_strides1 = torch.full((E_LOCAL,), 2 * I, device=device, dtype=torch.int64)

    workspace = torch.empty(90000, device=device, dtype=torch.uint8)
    a_ptrs = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    b_ptrs = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    out_ptrs = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    a_scales_ptrs = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    b_scales_ptrs = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    a_sf_layout = torch.empty((E_LOCAL, 5), dtype=torch.int32, device=device)
    w_sf_layout = torch.empty((E_LOCAL, 5), dtype=torch.int32, device=device)

    c1 = torch.empty((T * TOP_K, 2 * I), device=device, dtype=torch.bfloat16)
    fp8_blockwise_scaled_grouped_mm(
        c1,
        a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        rep_a_q, w1_q, rep_a_scales, w1_scale,
        ab_strides1, ab_strides1, c_strides1,
        a_sf_layout, w_sf_layout,
        problem_sizes1, expert_offsets[:-1],
        workspace,
    )
    return c1, workspace, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a_sf_layout, w_sf_layout


def sglang_stage3_swiglu_requant(c1):
    """Stage 3: SwiGLU + FP8 requant."""
    n = I
    intermediate = (F.silu(c1[:, n:]) * c1[:, :n]).to(torch.bfloat16)
    intermediate_q = torch.empty_like(intermediate, dtype=torch.float8_e4m3fn)
    a2_scale = torch.empty(
        (intermediate.shape[0], intermediate.shape[1] // BLOCK),
        dtype=torch.float32, device=intermediate.device,
    )
    sgl_per_token_group_quant_fp8(
        intermediate, intermediate_q, a2_scale,
        BLOCK, 1e-10, FP8_MIN, FP8_MAX, enable_v2=False,
    )
    return intermediate_q, a2_scale


def sglang_stage4_gemm2(
    intermediate_q, a2_scale, g2w, g2ws, problem_sizes2, expert_offsets,
    workspace, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
    a_sf_layout, w_sf_layout, T, device,
):
    """Stage 4: GEMM2."""
    w2_q = g2w.transpose(1, 2)
    w2_scale = g2ws.to(torch.float32).transpose(1, 2)

    ab_strides2 = torch.full((E_LOCAL,), I, device=device, dtype=torch.int64)
    c_strides2 = torch.full((E_LOCAL,), H, device=device, dtype=torch.int64)

    c2 = torch.empty((T * TOP_K, H), device=device, dtype=torch.bfloat16)
    fp8_blockwise_scaled_grouped_mm(
        c2,
        a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        intermediate_q, w2_q, a2_scale, w2_scale,
        ab_strides2, ab_strides2, c_strides2,
        a_sf_layout, w_sf_layout,
        problem_sizes2, expert_offsets[:-1],
        workspace,
    )
    return c2


def sglang_stage5_finalize(c2, c_map, topk_weights, T, device):
    """Stage 5: Weighted reduce."""
    output = torch.zeros((T, H), device=device, dtype=torch.bfloat16)
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(torch.bfloat16))
    return output


# ═══════════════════════════════════════════════════════════════════
# CuTeDSL — stage decomposition
# ═══════════════════════════════════════════════════════════════════


def cutedsl_stage1_routing(routing_logits, routing_bias, routed_scaling_factor):
    """Stage 1: Routing."""
    return moe_routing_deepseek(
        routing_logits, routing_bias,
        num_local_experts=E_LOCAL, local_expert_offset=0,
        n_group=N_GROUP, topk_group=TOPK_GROUP, top_k=TOP_K,
        routed_scaling_factor=routed_scaling_factor, pad_to=4,
    )


def cutedsl_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr, ws, mp):
    """Stage 2: GEMM1 (A-gather + grouped GEMM → FP8)."""
    return moe_gemm1_fp8(
        hidden_states, g1w, hs_scale, g1ws,
        rr.m_indptr, rr.permuted_idx_to_token_idx,
        gemm1_out=ws.gemm1_out[:mp],
        gemm1_out_scale=ws.gemm1_scale[:, :mp],
    )


def cutedsl_stage3_swiglu(g1_out, g1_scale, ws, mp):
    """Stage 3: SwiGLU + FP8 requant (fused CuTeDSL kernel)."""
    return moe_swiglu_fp8_requant(
        g1_out, g1_scale,
        act_out=ws.act_out[:mp],
        act_scale=ws.act_scale[:, :mp],
    )


def cutedsl_stage4_gemm2(a_out, g2w, a_scale, g2ws, rr, ws, mp):
    """Stage 4: GEMM2."""
    return moe_gemm2_fp8(
        a_out, g2w, a_scale, g2ws,
        rr.m_indptr, gemm2_out=ws.gemm2_out[:mp],
    )


def cutedsl_stage5_finalize(g2_out, rr):
    """Stage 5: Finalize (CuTeDSL kernel)."""
    return moe_finalize(
        g2_out, rr.topk_values, rr.topk_indices,
        rr.expanded_idx_to_permuted_idx, E_LOCAL, 0, H,
    )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024, help="Number of tokens (T)")
    parser.add_argument("--iters", type=int, default=10, help="Timing iterations")
    args = parser.parse_args()

    T = args.seq_len
    device = torch.device("cuda:0")
    routed_scaling_factor = 2.5

    print(f"T={T}, E_global={E_GLOBAL}, E_local={E_LOCAL}, top_k={TOP_K}, H={H}, I={I}")
    print()

    routing_logits, routing_bias, hidden_states, hs_scale, g1w, g1ws, g2w, g2ws = make_inputs(T, device)

    # ── JIT warmup ──
    print("=== JIT Warmup ===")
    t0 = time.time()
    cutedsl_out = cutedsl_fp8_moe(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws,
        num_experts_global=E_GLOBAL, num_local_experts=E_LOCAL,
        top_k=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
        intermediate_size=I, routed_scaling_factor=routed_scaling_factor,
    )
    torch.cuda.synchronize()
    print(f"CuTeDSL first call: {time.time() - t0:.2f}s")

    # SGLang warmup (run once end-to-end)
    t0 = time.time()
    s_rt = sglang_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
    s_g1 = sglang_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, s_rt[5], s_rt[3], s_rt[2], T, device)
    s_act = sglang_stage3_swiglu_requant(s_g1[0])
    s_g2 = sglang_stage4_gemm2(s_act[0], s_act[1], g2w, g2ws, s_rt[4], s_rt[2], s_g1[1], s_g1[2], s_g1[3], s_g1[4], s_g1[5], s_g1[6], s_g1[7], s_g1[8], T, device)
    sglang_out = sglang_stage5_finalize(s_g2, s_rt[6], s_rt[0], T, device)
    torch.cuda.synchronize()
    print(f"SGLang v2 first call: {time.time() - t0:.2f}s")

    # ── End-to-end precision comparison ──
    print("\n=== End-to-End Precision ===")
    diff = (cutedsl_out.float() - sglang_out.float()).abs()
    print(f"CuTeDSL vs SGLang v2:  max_abs_err={diff.max().item():.4f}  "
          f"mean_abs_err={diff.mean().item():.6f}  "
          f"cosine_sim={F.cosine_similarity(cutedsl_out.float().flatten(), sglang_out.float().flatten(), dim=0).item():.6f}")

    # ── Per-stage timing ──
    ITERS = args.iters

    # Prepare CuTeDSL routing result + workspace (for stages 2-5)
    rr = cutedsl_stage1_routing(routing_logits, routing_bias, routed_scaling_factor)
    mp = rr.max_padded_tokens
    ws = allocate_moe_workspace(mp, H, I, device)

    # Check for 0-token experts (known CuTeDSL bug)
    masked_m = (rr.m_indptr[1:] - rr.m_indptr[:-1]).cpu()
    if (masked_m == 0).any():
        zero_experts = (masked_m == 0).sum().item()
        print(f"\n  WARNING: {zero_experts} experts have 0 tokens — CuTeDSL GEMM will crash.")
        print(f"  Try a larger --seq-len (current: {T}).\n")

    # Prepare SGLang routing result (for stages 2-5)
    s_routing = sglang_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
    s_topk_weights, s_topk_ids, s_expert_offsets, s_ps1, s_ps2, s_a_map, s_c_map = s_routing

    # Collect all stage timings: list of (name, cd_time, sg_time)
    stage_timings = []

    # Stage 1: Routing
    cd_t = cuda_time(
        lambda: cutedsl_stage1_routing(routing_logits, routing_bias, routed_scaling_factor),
        iters=ITERS,
    )
    sg_t = cuda_time(
        lambda: sglang_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T),
        iters=ITERS,
    )
    stage_timings.append(("1. Routing", cd_t, sg_t))

    # Stage 2: GEMM1
    try:
        g1_out, g1_scale = cutedsl_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr, ws, mp)
        cd_t = cuda_time(
            lambda: cutedsl_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr, ws, mp),
            iters=ITERS,
        )
    except Exception as e:
        cd_t = float("nan")
        g1_out = g1_scale = None
        print(f"  [CuTeDSL GEMM1 failed: {e}]")

    s_gemm1_result = sglang_stage2_gemm1(
        hidden_states, hs_scale, g1w, g1ws, s_a_map, s_ps1, s_expert_offsets, T, device,
    )
    s_c1 = s_gemm1_result[0]
    sg_t = cuda_time(
        lambda: sglang_stage2_gemm1(
            hidden_states, hs_scale, g1w, g1ws, s_a_map, s_ps1, s_expert_offsets, T, device,
        ),
        iters=ITERS,
    )
    stage_timings.append(("2. GEMM1", cd_t, sg_t))

    # Stage 3: SwiGLU + Requant
    if g1_out is not None:
        a_out, a_scale = cutedsl_stage3_swiglu(g1_out, g1_scale, ws, mp)
        cd_t = cuda_time(
            lambda: cutedsl_stage3_swiglu(g1_out, g1_scale, ws, mp),
            iters=ITERS,
        )
    else:
        cd_t = float("nan")
        a_out = a_scale = None

    s_act_result = sglang_stage3_swiglu_requant(s_c1)
    s_int_q, s_a2_scale = s_act_result
    sg_t = cuda_time(lambda: sglang_stage3_swiglu_requant(s_c1), iters=ITERS)
    stage_timings.append(("3. SwiGLU+Requant", cd_t, sg_t))

    # Stage 4: GEMM2
    if a_out is not None:
        g2_out = cutedsl_stage4_gemm2(a_out, g2w, a_scale, g2ws, rr, ws, mp)
        cd_t = cuda_time(
            lambda: cutedsl_stage4_gemm2(a_out, g2w, a_scale, g2ws, rr, ws, mp),
            iters=ITERS,
        )
    else:
        cd_t = float("nan")
        g2_out = None

    s_g2 = sglang_stage4_gemm2(
        s_int_q, s_a2_scale, g2w, g2ws, s_ps2, s_expert_offsets,
        s_gemm1_result[1], s_gemm1_result[2], s_gemm1_result[3],
        s_gemm1_result[4], s_gemm1_result[5], s_gemm1_result[6],
        s_gemm1_result[7], s_gemm1_result[8], T, device,
    )
    sg_t = cuda_time(
        lambda: sglang_stage4_gemm2(
            s_int_q, s_a2_scale, g2w, g2ws, s_ps2, s_expert_offsets,
            s_gemm1_result[1], s_gemm1_result[2], s_gemm1_result[3],
            s_gemm1_result[4], s_gemm1_result[5], s_gemm1_result[6],
            s_gemm1_result[7], s_gemm1_result[8], T, device,
        ),
        iters=ITERS,
    )
    stage_timings.append(("4. GEMM2", cd_t, sg_t))

    # Stage 5: Finalize
    if g2_out is not None:
        cd_t = cuda_time(lambda: cutedsl_stage5_finalize(g2_out, rr), iters=ITERS)
    else:
        cd_t = float("nan")

    sg_t = cuda_time(
        lambda: sglang_stage5_finalize(s_g2, s_c_map, s_topk_weights, T, device),
        iters=ITERS,
    )
    stage_timings.append(("5. Finalize", cd_t, sg_t))

    # Workspace alloc overhead (CuTeDSL only)
    alloc_t = cuda_time(lambda: allocate_moe_workspace(mp, H, I, device), iters=ITERS)

    # End-to-end
    cd_e2e = cuda_time(
        lambda: cutedsl_fp8_moe(
            routing_logits, routing_bias, hidden_states, hs_scale,
            g1w, g1ws, g2w, g2ws,
            num_experts_global=E_GLOBAL, num_local_experts=E_LOCAL,
            top_k=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
            intermediate_size=I, routed_scaling_factor=routed_scaling_factor,
        ),
        iters=ITERS,
    )

    # SGLang v2 end-to-end (full run function from solution)
    def sglang_e2e():
        r = sglang_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
        g = sglang_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, r[5], r[3], r[2], T, device)
        a = sglang_stage3_swiglu_requant(g[0])
        g2 = sglang_stage4_gemm2(a[0], a[1], g2w, g2ws, r[4], r[2], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], T, device)
        return sglang_stage5_finalize(g2, r[6], r[0], T, device)

    sg_e2e = cuda_time(sglang_e2e, iters=ITERS)

    # ── Print results with percentage columns ──
    # Compute totals from per-stage sums (excluding NaN)
    cd_stage_total = sum(t for _, t, _ in stage_timings if t == t)  # NaN != NaN
    sg_stage_total = sum(t for _, _, t in stage_timings)

    print(f"\n=== Per-Stage Timing ({ITERS} iters, median ms) ===")
    print(f"{'Stage':<24} {'CuTeDSL':>8} {'%':>6} {'SGLang v2':>10} {'%':>6} {'Ratio':>8}")
    print("-" * 68)

    for name, cd, sg in stage_timings:
        cd_pct = f"{cd / cd_stage_total * 100:5.1f}%" if cd == cd and cd_stage_total > 0 else "  N/A"
        sg_pct = f"{sg / sg_stage_total * 100:5.1f}%" if sg_stage_total > 0 else "  N/A"
        ratio_str = f"{cd/sg:>7.2f}x" if cd == cd else "   N/A"
        cd_str = f"{cd:>8.3f}" if cd == cd else "     N/A"
        print(f"{name:<24} {cd_str} {cd_pct:>6} {sg:>10.3f} {sg_pct:>6} {ratio_str}")

    print(f"{'   Workspace alloc':<24} {alloc_t:>8.3f} {'':>6} {'---':>10} {'':>6} {'':>8}")

    print("-" * 68)
    cd_e2e_pct = f"{cd_e2e / cd_e2e * 100:5.1f}%" if cd_e2e == cd_e2e else "  N/A"
    sg_e2e_pct = f"{sg_e2e / sg_e2e * 100:5.1f}%" if sg_e2e > 0 else "  N/A"
    print(f"{'End-to-end':<24} {cd_e2e:>8.3f} {'':>6} {sg_e2e:>10.3f} {'':>6} {cd_e2e/sg_e2e:>7.2f}x")
    print(f"{'Sum of stages':<24} {cd_stage_total:>8.3f} {'':>6} {sg_stage_total:>10.3f} {'':>6}")


if __name__ == "__main__":
    main()
