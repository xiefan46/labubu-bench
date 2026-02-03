"""
Stage-by-stage performance comparison:
  CuTeDSL v2  vs  CuTeDSL v3  vs  Triton v2

All three pipelines are decomposed into 5 aligned stages:
  1. Routing     — expert selection + index computation
  2. GEMM1       — FP8 grouped GEMM (incl. A-gather/shuffle)
  3. SwiGLU+Req  — activation + requantization
  4. GEMM2       — FP8 grouped GEMM
  5. Finalize    — weighted reduce / scatter-add

Usage:
    python scripts/compare_cutedsl_v2_v3_triton.py [--seq-len 1024]

AI-assisted implementation (Claude).
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

# ── CuTeDSL imports ──
from flashinfer.cute_dsl.moe_activation import moe_swiglu_fp8_requant
from flashinfer.cute_dsl.moe_finalize import moe_finalize
from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v2 import moe_gemm1_fp8_v2, moe_gemm2_fp8_v2
from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v3 import (
    moe_gemm1_fp8_v3,
    moe_gemm2_fp8_v3,
    gather_from_flat_padded,
    compute_aligned_m_indptr,
)
from flashinfer.cute_dsl.moe_pipeline import allocate_moe_workspace
from flashinfer.cute_dsl.moe_pipeline_v2 import cutedsl_fp8_moe_v2
from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3
from flashinfer.cute_dsl.moe_routing import moe_routing_sglang

# ── SGLang / sgl_kernel imports (for Triton v2) ──
from sgl_kernel import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    moe_fused_gate,
    prepare_moe_input,
    sgl_per_token_group_quant_fp8,
    shuffle_rows,
)

# ── Triton v2 SwiGLU kernel ──
sys.path.insert(
    0,
    "solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/triton_fused_v2",
)
from swiglu_quant_kernel import swiglu_quant_kernel  # noqa: E402

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
    actually receive tokens.
    """
    routing_logits = torch.randn(T, E_GLOBAL, dtype=torch.float32, device=device)
    routing_logits[:, :E_LOCAL] += 20.0
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(T, H, device=device).to(torch.float8_e4m3fn)
    hs_scale = torch.randn(H // BLOCK, T, dtype=torch.float32, device=device).abs() + 0.01
    g1w = torch.randn(E_LOCAL, 2 * I, H, device=device).to(torch.float8_e4m3fn)
    g1ws = torch.randn(E_LOCAL, 2 * I // BLOCK, H // BLOCK, dtype=torch.float32, device=device).abs() + 0.01
    g2w = torch.randn(E_LOCAL, H, I, device=device).to(torch.float8_e4m3fn)
    g2ws = torch.randn(E_LOCAL, H // BLOCK, I // BLOCK, dtype=torch.float32, device=device).abs() + 0.01
    return routing_logits, routing_bias, hidden_states, hs_scale, g1w, g1ws, g2w, g2ws


# ═══════════════════════════════════════════════════════════════════
# Triton v2 — stage decomposition
# ═══════════════════════════════════════════════════════════════════


def triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T):
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


def triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, a_map, problem_sizes1, expert_offsets, T, device):
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


def triton_stage3_swiglu_requant(c1):
    """Stage 3: SwiGLU + FP8 requant (Triton kernel)."""
    M_total = c1.shape[0]
    intermediate_q = torch.empty(
        (M_total, I), device=c1.device, dtype=torch.float8_e4m3fn
    )
    a2_scale = torch.empty(
        (M_total, I // BLOCK), dtype=torch.float32, device=c1.device
    )
    grid = (M_total,)
    swiglu_quant_kernel[grid](
        c1,
        intermediate_q,
        a2_scale,
        M_total,
        I,
        c1.stride(0),
        intermediate_q.stride(0),
        a2_scale.stride(0),
        FP8_MAX=448.0,
        GROUP_SIZE=BLOCK,
        BLOCK_N=BLOCK,
    )
    return intermediate_q, a2_scale


def triton_stage4_gemm2(
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


def triton_stage5_finalize(c2, c_map, topk_weights, T, device):
    """Stage 5: Weighted reduce."""
    output = torch.zeros((T, H), device=device, dtype=torch.bfloat16)
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(torch.bfloat16))
    return output


# ═══════════════════════════════════════════════════════════════════
# CuTeDSL v2 — stage decomposition
# ═══════════════════════════════════════════════════════════════════


def cutedsl_stage1_routing(routing_logits, routing_bias, routed_scaling_factor):
    """Stage 1: Routing (sgl_kernel backend). Shared by v2 and v3."""
    return moe_routing_sglang(
        routing_logits, routing_bias,
        num_local_experts=E_LOCAL, local_expert_offset=0,
        n_group=N_GROUP, topk_group=TOPK_GROUP, top_k=TOP_K,
        routed_scaling_factor=routed_scaling_factor,
        intermediate_size=I, hidden_size=H,
    )


def cutedsl_v2_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr, ws, mp):
    """Stage 2 v2: GEMM1 (A-gather + vectorized grouped GEMM -> FP8)."""
    return moe_gemm1_fp8_v2(
        hidden_states, g1w, hs_scale, g1ws,
        rr.m_indptr, rr.permuted_idx_to_token_idx,
        gemm1_out=ws.gemm1_out[:mp],
        gemm1_out_scale=ws.gemm1_scale[:, :mp],
    )


def cutedsl_v2_stage3_swiglu(g1_out, g1_scale, ws, mp):
    """Stage 3: SwiGLU + FP8 requant (fused CuTeDSL kernel)."""
    return moe_swiglu_fp8_requant(
        g1_out, g1_scale,
        act_out=ws.act_out[:mp],
        act_scale=ws.act_scale[:, :mp],
    )


def cutedsl_v2_stage4_gemm2(a_out, g2w, a_scale, g2ws, rr, ws, mp):
    """Stage 4 v2: GEMM2 (vectorized wrapper)."""
    return moe_gemm2_fp8_v2(
        a_out, g2w, a_scale, g2ws,
        rr.m_indptr, gemm2_out=ws.gemm2_out[:mp],
    )


def cutedsl_v2_stage5_finalize(g2_out, rr):
    """Stage 5: Finalize (CuTeDSL kernel)."""
    return moe_finalize(
        g2_out, rr.topk_values, rr.topk_indices,
        rr.expanded_idx_to_permuted_idx, E_LOCAL, 0, H,
    )


# ═══════════════════════════════════════════════════════════════════
# CuTeDSL v3 — stage decomposition
# ═══════════════════════════════════════════════════════════════════


def cutedsl_v3_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr):
    """Stage 2 v3: GEMM1 (A-gather + scatter to flat + grouped GEMM -> FP8)."""
    return moe_gemm1_fp8_v3(
        hidden_states, g1w, hs_scale, g1ws,
        rr.m_indptr, rr.permuted_idx_to_token_idx,
    )


def cutedsl_v3_stage3_swiglu(g1_out, g1_scale):
    """Stage 3 v3: SwiGLU + FP8 requant on flat padded layout."""
    return moe_swiglu_fp8_requant(g1_out, g1_scale)


def cutedsl_v3_stage4_gemm2(a_out, g2w, a_scale, g2ws, masked_m, m_indptr_tiles):
    """Stage 4 v3: GEMM2 on flat padded layout."""
    return moe_gemm2_fp8_v3(a_out, g2w, a_scale, g2ws, masked_m, m_indptr_tiles)


def cutedsl_v3_stage5_finalize(g2_out_flat, dst_row, max_padded, rr):
    """Stage 5 v3: Unsatter + Finalize."""
    g2_out = gather_from_flat_padded(g2_out_flat, dst_row, max_padded)
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
    cutedsl_v2_out = cutedsl_fp8_moe_v2(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws,
        num_experts_global=E_GLOBAL, num_local_experts=E_LOCAL,
        top_k=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
        intermediate_size=I, routed_scaling_factor=routed_scaling_factor,
    )
    torch.cuda.synchronize()
    print(f"CuTeDSL v2 first call: {time.time() - t0:.2f}s")

    t0 = time.time()
    cutedsl_v3_out = cutedsl_fp8_moe_v3(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws,
        num_experts_global=E_GLOBAL, num_local_experts=E_LOCAL,
        top_k=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
        intermediate_size=I, routed_scaling_factor=routed_scaling_factor,
    )
    torch.cuda.synchronize()
    print(f"CuTeDSL v3 first call: {time.time() - t0:.2f}s")

    # Triton v2 warmup
    t0 = time.time()
    tr_rt = triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
    tr_g1 = triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, tr_rt[5], tr_rt[3], tr_rt[2], T, device)
    triton_stage3_swiglu_requant(tr_g1[0])
    torch.cuda.synchronize()
    print(f"Triton v2 first call: {time.time() - t0:.2f}s")

    # ── End-to-end precision comparison ──
    print("\n=== End-to-End Precision ===")
    diff_v3_v2 = (cutedsl_v3_out.float() - cutedsl_v2_out.float()).abs()
    cos_v3_v2 = F.cosine_similarity(
        cutedsl_v3_out.float().flatten(), cutedsl_v2_out.float().flatten(), dim=0
    ).item()
    print(f"CuTeDSL v3 vs v2:  max_abs_err={diff_v3_v2.max().item():.4f}  "
          f"mean_abs_err={diff_v3_v2.mean().item():.6f}  "
          f"cosine_sim={cos_v3_v2:.6f}")

    # ── Per-stage timing ──
    ITERS = args.iters

    # Prepare CuTeDSL routing result (shared by v2 and v3)
    rr = cutedsl_stage1_routing(routing_logits, routing_bias, routed_scaling_factor)
    mp = rr.max_padded_tokens

    # Prepare v2 workspace
    ws = allocate_moe_workspace(mp, H, I, device)

    # Prepare Triton routing result
    tr_routing = triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
    tr_topk_weights, tr_topk_ids, tr_expert_offsets, tr_ps1, tr_ps2, tr_a_map, tr_c_map = tr_routing

    # Collect stage timings: list of (name, v2_time, v3_time, triton_time)
    stage_timings = []

    # ── Stage 1: Routing ──
    # CuTeDSL v2 and v3 share the same routing
    cd_t = cuda_time(
        lambda: cutedsl_stage1_routing(routing_logits, routing_bias, routed_scaling_factor),
        iters=ITERS,
    )
    v2_t = cd_t
    v3_t = cd_t  # same routing
    tr_t = cuda_time(
        lambda: triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T),
        iters=ITERS,
    )
    stage_timings.append(("1. Routing", v2_t, v3_t, tr_t))

    # ── Stage 2: GEMM1 ──
    # v2
    g1_out_v2, g1_scale_v2 = cutedsl_v2_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr, ws, mp)
    v2_t = cuda_time(
        lambda: cutedsl_v2_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr, ws, mp),
        iters=ITERS,
    )

    # v3
    g1_v3_result = cutedsl_v3_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr)
    g1_out_v3, g1_scale_v3, m_indptr_aligned, m_indptr_tiles, masked_m, dst_row = g1_v3_result
    v3_t = cuda_time(
        lambda: cutedsl_v3_stage2_gemm1(hidden_states, g1w, hs_scale, g1ws, rr),
        iters=ITERS,
    )

    # Triton v2
    tr_gemm1_result = triton_stage2_gemm1(
        hidden_states, hs_scale, g1w, g1ws, tr_a_map, tr_ps1, tr_expert_offsets, T, device,
    )
    tr_c1 = tr_gemm1_result[0]
    tr_t = cuda_time(
        lambda: triton_stage2_gemm1(
            hidden_states, hs_scale, g1w, g1ws, tr_a_map, tr_ps1, tr_expert_offsets, T, device,
        ),
        iters=ITERS,
    )
    stage_timings.append(("2. GEMM1", v2_t, v3_t, tr_t))

    # ── Stage 3: SwiGLU + Requant ──
    # v2
    a_out_v2, a_scale_v2 = cutedsl_v2_stage3_swiglu(g1_out_v2, g1_scale_v2, ws, mp)
    v2_t = cuda_time(
        lambda: cutedsl_v2_stage3_swiglu(g1_out_v2, g1_scale_v2, ws, mp),
        iters=ITERS,
    )

    # v3 (same kernel, different input shape — flat padded)
    a_out_v3, a_scale_v3 = cutedsl_v3_stage3_swiglu(g1_out_v3, g1_scale_v3)
    v3_t = cuda_time(
        lambda: cutedsl_v3_stage3_swiglu(g1_out_v3, g1_scale_v3),
        iters=ITERS,
    )

    # Triton v2
    tr_act_result = triton_stage3_swiglu_requant(tr_c1)
    tr_int_q, tr_a2_scale = tr_act_result
    tr_t = cuda_time(lambda: triton_stage3_swiglu_requant(tr_c1), iters=ITERS)
    stage_timings.append(("3. SwiGLU+Requant", v2_t, v3_t, tr_t))

    # ── Stage 4: GEMM2 ──
    # v2
    g2_out_v2 = cutedsl_v2_stage4_gemm2(a_out_v2, g2w, a_scale_v2, g2ws, rr, ws, mp)
    v2_t = cuda_time(
        lambda: cutedsl_v2_stage4_gemm2(a_out_v2, g2w, a_scale_v2, g2ws, rr, ws, mp),
        iters=ITERS,
    )

    # v3
    g2_out_v3_flat = cutedsl_v3_stage4_gemm2(a_out_v3, g2w, a_scale_v3, g2ws, masked_m, m_indptr_tiles)
    v3_t = cuda_time(
        lambda: cutedsl_v3_stage4_gemm2(a_out_v3, g2w, a_scale_v3, g2ws, masked_m, m_indptr_tiles),
        iters=ITERS,
    )

    # Triton v2
    tr_g2 = triton_stage4_gemm2(
        tr_int_q, tr_a2_scale, g2w, g2ws, tr_ps2, tr_expert_offsets,
        tr_gemm1_result[1], tr_gemm1_result[2], tr_gemm1_result[3],
        tr_gemm1_result[4], tr_gemm1_result[5], tr_gemm1_result[6],
        tr_gemm1_result[7], tr_gemm1_result[8], T, device,
    )
    tr_t = cuda_time(
        lambda: triton_stage4_gemm2(
            tr_int_q, tr_a2_scale, g2w, g2ws, tr_ps2, tr_expert_offsets,
            tr_gemm1_result[1], tr_gemm1_result[2], tr_gemm1_result[3],
            tr_gemm1_result[4], tr_gemm1_result[5], tr_gemm1_result[6],
            tr_gemm1_result[7], tr_gemm1_result[8], T, device,
        ),
        iters=ITERS,
    )
    stage_timings.append(("4. GEMM2", v2_t, v3_t, tr_t))

    # ── Stage 5: Finalize ──
    # v2
    v2_t = cuda_time(lambda: cutedsl_v2_stage5_finalize(g2_out_v2, rr), iters=ITERS)

    # v3 (unsatter + finalize)
    v3_t = cuda_time(
        lambda: cutedsl_v3_stage5_finalize(g2_out_v3_flat, dst_row, mp, rr),
        iters=ITERS,
    )

    # Triton v2
    tr_t = cuda_time(
        lambda: triton_stage5_finalize(tr_g2, tr_c_map, tr_topk_weights, T, device),
        iters=ITERS,
    )
    stage_timings.append(("5. Finalize", v2_t, v3_t, tr_t))

    # Workspace alloc overhead (CuTeDSL v2 only; v3 allocates inside GEMM1)
    alloc_t = cuda_time(lambda: allocate_moe_workspace(mp, H, I, device), iters=ITERS)

    # ── End-to-end timing ──
    common_kwargs = dict(
        num_experts_global=E_GLOBAL, num_local_experts=E_LOCAL,
        top_k=TOP_K, n_group=N_GROUP, topk_group=TOPK_GROUP,
        intermediate_size=I, routed_scaling_factor=routed_scaling_factor,
    )

    v2_e2e = cuda_time(
        lambda: cutedsl_fp8_moe_v2(
            routing_logits, routing_bias, hidden_states, hs_scale,
            g1w, g1ws, g2w, g2ws, **common_kwargs,
        ),
        iters=ITERS,
    )

    v3_e2e = cuda_time(
        lambda: cutedsl_fp8_moe_v3(
            routing_logits, routing_bias, hidden_states, hs_scale,
            g1w, g1ws, g2w, g2ws, **common_kwargs,
        ),
        iters=ITERS,
    )

    def triton_e2e():
        r = triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
        g = triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, r[5], r[3], r[2], T, device)
        a = triton_stage3_swiglu_requant(g[0])
        g2 = triton_stage4_gemm2(
            a[0], a[1], g2w, g2ws, r[4], r[2],
            g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], T, device,
        )
        return triton_stage5_finalize(g2, r[6], r[0], T, device)

    tr_e2e = cuda_time(triton_e2e, iters=ITERS)

    # ── Print results ──
    v2_total = sum(t for _, t, _, _ in stage_timings)
    v3_total = sum(t for _, _, t, _ in stage_timings)
    tr_total = sum(t for _, _, _, t in stage_timings)

    print(f"\n=== Per-Stage Timing ({ITERS} iters, median ms) ===")
    hdr = (f"{'Stage':<20} "
           f"{'CD v2':>8} {'%':>6}  "
           f"{'CD v3':>8} {'%':>6}  "
           f"{'Triton':>8} {'%':>6}  "
           f"{'v3/v2':>6} {'v3/Tr':>6} {'v2/Tr':>6}")
    print(hdr)
    print("-" * len(hdr))

    for name, v2, v3, tr in stage_timings:
        v2_pct = f"{v2 / v2_total * 100:5.1f}%" if v2_total > 0 else "  N/A"
        v3_pct = f"{v3 / v3_total * 100:5.1f}%" if v3_total > 0 else "  N/A"
        tr_pct = f"{tr / tr_total * 100:5.1f}%" if tr_total > 0 else "  N/A"
        v3_v2 = f"{v3/v2:>5.2f}x" if v2 > 0 else "  N/A"
        v3_tr = f"{v3/tr:>5.2f}x" if tr > 0 else "  N/A"
        v2_tr = f"{v2/tr:>5.2f}x" if tr > 0 else "  N/A"
        print(f"{name:<20} "
              f"{v2:>8.3f} {v2_pct:>6}  "
              f"{v3:>8.3f} {v3_pct:>6}  "
              f"{tr:>8.3f} {tr_pct:>6}  "
              f"{v3_v2} {v3_tr} {v2_tr}")

    print(f"{'  Workspace alloc':<20} {alloc_t:>8.3f} {'':>6}  {'---':>8} {'':>6}  {'---':>8} {'':>6}")

    print("-" * len(hdr))
    print(f"{'End-to-end':<20} "
          f"{v2_e2e:>8.3f} {'':>6}  "
          f"{v3_e2e:>8.3f} {'':>6}  "
          f"{tr_e2e:>8.3f} {'':>6}  "
          f"{v3_e2e/v2_e2e:>5.2f}x {v3_e2e/tr_e2e:>5.2f}x {v2_e2e/tr_e2e:>5.2f}x")
    print(f"{'Sum of stages':<20} "
          f"{v2_total:>8.3f} {'':>6}  "
          f"{v3_total:>8.3f} {'':>6}  "
          f"{tr_total:>8.3f} {'':>6}")


if __name__ == "__main__":
    main()
