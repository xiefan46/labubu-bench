"""
Per-stage profiling of CuTeDSL FP8 MoE pipeline.

Measures JIT compilation overhead, then CUDA-event-timed breakdown
of each pipeline stage (routing, GEMM1, SwiGLU, GEMM2, finalize, alloc).

Usage:
    python scripts/profile_cutedsl_moe.py
"""

import time

import torch

from flashinfer.cute_dsl.moe_activation import moe_swiglu_fp8_requant
from flashinfer.cute_dsl.moe_finalize import moe_finalize
from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_gemm1_fp8, moe_gemm2_fp8
from flashinfer.cute_dsl.moe_pipeline import (
    allocate_moe_workspace,
    cutedsl_fp8_moe,
    moe_routing_deepseek,
)

device = torch.device("cuda:0")
T, H, I = 256, 7168, 2048
E_global, E_local = 256, 32
BLOCK = 128

# --- Fake inputs ---
routing_logits = torch.randn(T, E_global, dtype=torch.float32, device=device)
routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)
hidden_states = torch.randn(T, H, device=device).to(torch.float8_e4m3fn)
hidden_states_scale = (
    torch.randn(H // BLOCK, T, dtype=torch.float32, device=device).abs() + 0.01
)
gemm1_w = torch.randn(E_local, 2 * I, H, device=device).to(torch.float8_e4m3fn)
gemm1_ws = (
    torch.randn(E_local, 2 * I // BLOCK, H // BLOCK, dtype=torch.float32, device=device).abs()
    + 0.01
)
gemm2_w = torch.randn(E_local, H, I, device=device).to(torch.float8_e4m3fn)
gemm2_ws = (
    torch.randn(E_local, H // BLOCK, I // BLOCK, dtype=torch.float32, device=device).abs()
    + 0.01
)

MOE_KWARGS = dict(
    num_experts_global=E_global,
    num_local_experts=E_local,
    top_k=8,
    n_group=8,
    topk_group=4,
    intermediate_size=I,
    routed_scaling_factor=2.5,
)


def cuda_time(fn, warmup=3, iters=10):
    """Median CUDA-event time in ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# === Warmup (JIT compilation) ===
print("=== Warmup (JIT compilation) ===")
t0 = time.time()
out = cutedsl_fp8_moe(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_w,
    gemm1_ws,
    gemm2_w,
    gemm2_ws,
    **MOE_KWARGS,
)
torch.cuda.synchronize()
print(f"First call (with JIT): {time.time() - t0:.2f}s")

t0 = time.time()
out = cutedsl_fp8_moe(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_w,
    gemm1_ws,
    gemm2_w,
    gemm2_ws,
    **MOE_KWARGS,
)
torch.cuda.synchronize()
print(f"Second call (cached): {time.time() - t0:.4f}s")

# === Per-stage profiling ===
print("\n=== Per-stage CUDA timing (10 iters, median) ===")


# Stage 1: Routing
def run_routing():
    return moe_routing_deepseek(
        routing_logits,
        routing_bias,
        num_local_experts=E_local,
        local_expert_offset=0,
        n_group=8,
        topk_group=4,
        top_k=8,
        routed_scaling_factor=2.5,
        pad_to=4,
    )


routing_ms = cuda_time(run_routing)
print(f"  Routing:           {routing_ms:.3f} ms")

rr = run_routing()
mp = rr.max_padded_tokens
ws = allocate_moe_workspace(mp, H, I, device)


# Stage 2: GEMM1
def run_gemm1():
    return moe_gemm1_fp8(
        hidden_states,
        gemm1_w,
        hidden_states_scale,
        gemm1_ws,
        rr.m_indptr,
        rr.permuted_idx_to_token_idx,
        gemm1_out=ws.gemm1_out[:mp],
        gemm1_out_scale=ws.gemm1_scale[:, :mp],
    )


gemm1_ms = cuda_time(run_gemm1)
print(f"  GEMM1:             {gemm1_ms:.3f} ms")

g1_out, g1_scale = run_gemm1()


# Stage 3: SwiGLU + Requant
def run_act():
    return moe_swiglu_fp8_requant(
        g1_out,
        g1_scale,
        act_out=ws.act_out[:mp],
        act_scale=ws.act_scale[:, :mp],
    )


act_ms = cuda_time(run_act)
print(f"  SwiGLU+Requant:    {act_ms:.3f} ms")

a_out, a_scale = run_act()


# Stage 4: GEMM2
def run_gemm2():
    return moe_gemm2_fp8(
        a_out,
        gemm2_w,
        a_scale,
        gemm2_ws,
        rr.m_indptr,
        gemm2_out=ws.gemm2_out[:mp],
    )


gemm2_ms = cuda_time(run_gemm2)
print(f"  GEMM2:             {gemm2_ms:.3f} ms")

g2_out = run_gemm2()


# Stage 5: Finalize
def run_finalize():
    return moe_finalize(
        g2_out,
        rr.topk_values,
        rr.topk_indices,
        rr.expanded_idx_to_permuted_idx,
        E_local,
        0,
        H,
    )


fin_ms = cuda_time(run_finalize)
print(f"  Finalize:          {fin_ms:.3f} ms")

# Workspace allocation
alloc_ms = cuda_time(lambda: allocate_moe_workspace(mp, H, I, device))
print(f"  Workspace alloc:   {alloc_ms:.3f} ms")

# Summary
total = routing_ms + gemm1_ms + act_ms + gemm2_ms + fin_ms + alloc_ms
print(f"\n  Sum of stages:     {total:.3f} ms")

e2e_ms = cuda_time(
    lambda: cutedsl_fp8_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_w,
        gemm1_ws,
        gemm2_w,
        gemm2_ws,
        **MOE_KWARGS,
    )
)
print(f"  End-to-end:        {e2e_ms:.3f} ms")
