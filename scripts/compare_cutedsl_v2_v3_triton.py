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
# Triton v2 — pre-allocated buffers + stage decomposition
# Mirrors triton_fused_v2/main.py: module-level buffer caching
# to eliminate per-call torch.empty overhead.
# ═══════════════════════════════════════════════════════════════════

# Module-level buffer caches (same pattern as real triton_fused_v2 solution)
_tr_static = {}   # device -> dict of fixed-shape buffers
_tr_dynamic = {}  # (M, device) -> dict of M-dependent buffers


def _tr_init_static(device):
    """Allocate all fixed-shape buffers (independent of M)."""
    s = {}
    s["ab_strides1"] = torch.full((E_LOCAL,), H, device=device, dtype=torch.int64)
    s["c_strides1"] = torch.full((E_LOCAL,), 2 * I, device=device, dtype=torch.int64)
    s["ab_strides2"] = torch.full((E_LOCAL,), I, device=device, dtype=torch.int64)
    s["c_strides2"] = torch.full((E_LOCAL,), H, device=device, dtype=torch.int64)
    s["workspace"] = torch.empty(90000, device=device, dtype=torch.uint8)
    s["a_ptrs"] = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    s["b_ptrs"] = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    s["out_ptrs"] = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    s["a_scales_ptrs"] = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    s["b_scales_ptrs"] = torch.empty((E_LOCAL,), dtype=torch.int64, device=device)
    s["a_sf_layout"] = torch.empty((E_LOCAL, 5), dtype=torch.int32, device=device)
    s["w_sf_layout"] = torch.empty((E_LOCAL, 5), dtype=torch.int32, device=device)
    s["expert_offsets"] = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device)
    s["problem_sizes1"] = torch.empty((E_LOCAL, 3), dtype=torch.int32, device=device)
    s["problem_sizes2"] = torch.empty((E_LOCAL, 3), dtype=torch.int32, device=device)
    return s


def _tr_get_static(device):
    if device not in _tr_static:
        _tr_static[device] = _tr_init_static(device)
    return _tr_static[device]


def _tr_init_dynamic(M, device):
    """Allocate all M-dependent buffers."""
    d = {}
    MT = M * TOP_K
    d["a_map"] = torch.empty((MT,), dtype=torch.int32, device=device)
    d["c_map"] = torch.empty((MT,), dtype=torch.int32, device=device)
    d["c1"] = torch.empty((MT, 2 * I), device=device, dtype=torch.bfloat16)
    d["intermediate_q"] = torch.empty((MT, I), device=device, dtype=torch.float8_e4m3fn)
    d["a2_scale"] = torch.empty((MT, I // BLOCK), dtype=torch.float32, device=device)
    d["c2"] = torch.empty((MT, H), device=device, dtype=torch.bfloat16)
    d["output"] = torch.zeros((M, H), device=device, dtype=torch.bfloat16)
    return d


def _tr_get_dynamic(M, device):
    key = (M, device)
    if key not in _tr_dynamic:
        _tr_dynamic[key] = _tr_init_dynamic(M, device)
    return _tr_dynamic[key]


def triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T):
    """Stage 1: Routing (moe_fused_gate + local masking + prepare_moe_input).

    Uses pre-allocated buffers for expert_offsets, problem_sizes, a_map, c_map.
    """
    s = _tr_get_static(device)
    d = _tr_get_dynamic(T, device)

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

    prepare_moe_input(
        topk_ids, s["expert_offsets"], s["problem_sizes1"], s["problem_sizes2"],
        d["a_map"], d["c_map"], E_LOCAL, I, H,
    )
    return topk_weights, topk_ids


def triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, T, device):
    """Stage 2: GEMM1 (shuffle_rows + grouped GEMM). Pre-allocated buffers."""
    s = _tr_get_static(device)
    d = _tr_get_dynamic(T, device)

    a_scale = hs_scale.to(torch.float32).T.contiguous()
    rep_a_q = shuffle_rows(hidden_states.contiguous(), d["a_map"], (T * TOP_K, H))
    rep_a_scales = shuffle_rows(a_scale, d["a_map"], (T * TOP_K, H // BLOCK))

    w1_q = g1w.transpose(1, 2)
    w1_scale = g1ws.to(torch.float32).transpose(1, 2)

    fp8_blockwise_scaled_grouped_mm(
        d["c1"],
        s["a_ptrs"], s["b_ptrs"], s["out_ptrs"],
        s["a_scales_ptrs"], s["b_scales_ptrs"],
        rep_a_q, w1_q, rep_a_scales, w1_scale,
        s["ab_strides1"], s["ab_strides1"], s["c_strides1"],
        s["a_sf_layout"], s["w_sf_layout"],
        s["problem_sizes1"], s["expert_offsets"][:-1],
        s["workspace"],
    )
    return d["c1"]


def triton_stage3_swiglu_requant(c1, T, device):
    """Stage 3: SwiGLU + FP8 requant (Triton kernel). Pre-allocated buffers."""
    d = _tr_get_dynamic(T, device)
    M_total = T * TOP_K

    grid = (M_total,)
    swiglu_quant_kernel[grid](
        c1,
        d["intermediate_q"],
        d["a2_scale"],
        M_total,
        I,
        c1.stride(0),
        d["intermediate_q"].stride(0),
        d["a2_scale"].stride(0),
        FP8_MAX=448.0,
        GROUP_SIZE=BLOCK,
        BLOCK_N=BLOCK,
    )
    return d["intermediate_q"], d["a2_scale"]


def triton_stage4_gemm2(intermediate_q, a2_scale, g2w, g2ws, T, device):
    """Stage 4: GEMM2. Pre-allocated buffers."""
    s = _tr_get_static(device)
    d = _tr_get_dynamic(T, device)

    w2_q = g2w.transpose(1, 2)
    w2_scale = g2ws.to(torch.float32).transpose(1, 2)

    fp8_blockwise_scaled_grouped_mm(
        d["c2"],
        s["a_ptrs"], s["b_ptrs"], s["out_ptrs"],
        s["a_scales_ptrs"], s["b_scales_ptrs"],
        intermediate_q, w2_q, a2_scale, w2_scale,
        s["ab_strides2"], s["ab_strides2"], s["c_strides2"],
        s["a_sf_layout"], s["w_sf_layout"],
        s["problem_sizes2"], s["expert_offsets"][:-1],
        s["workspace"],
    )
    return d["c2"]


def triton_stage5_finalize(topk_weights, T, device):
    """Stage 5: Weighted reduce. Pre-allocated buffers."""
    d = _tr_get_dynamic(T, device)
    d["output"].zero_()
    apply_shuffle_mul_sum(d["c2"], d["output"], d["c_map"], topk_weights.to(torch.bfloat16))
    return d["output"]


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

    # Triton v2 warmup (also initializes pre-allocated buffers)
    t0 = time.time()
    tr_topk_w, tr_topk_i = triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
    tr_c1 = triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, T, device)
    triton_stage3_swiglu_requant(tr_c1, T, device)
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

    # Prepare Triton routing result (populates pre-allocated buffers)
    tr_topk_weights, _ = triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)

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

    # Triton v2 (pre-allocated buffers)
    tr_c1 = triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, T, device)
    tr_t = cuda_time(
        lambda: triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, T, device),
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

    # Triton v2 (pre-allocated buffers)
    tr_int_q, tr_a2_scale = triton_stage3_swiglu_requant(tr_c1, T, device)
    tr_t = cuda_time(lambda: triton_stage3_swiglu_requant(tr_c1, T, device), iters=ITERS)
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

    # Triton v2 (pre-allocated buffers)
    tr_g2 = triton_stage4_gemm2(tr_int_q, tr_a2_scale, g2w, g2ws, T, device)
    tr_t = cuda_time(
        lambda: triton_stage4_gemm2(tr_int_q, tr_a2_scale, g2w, g2ws, T, device),
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

    # Triton v2 (pre-allocated output buffer)
    tr_t = cuda_time(
        lambda: triton_stage5_finalize(tr_topk_weights, T, device),
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
        tw, _ = triton_stage1_routing(routing_logits, routing_bias, routed_scaling_factor, device, T)
        c1 = triton_stage2_gemm1(hidden_states, hs_scale, g1w, g1ws, T, device)
        iq, asc = triton_stage3_swiglu_requant(c1, T, device)
        triton_stage4_gemm2(iq, asc, g2w, g2ws, T, device)
        return triton_stage5_finalize(tw, T, device)

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
