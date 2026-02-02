"""Fused SwiGLU + FP8 per-token-group quantization Triton kernel.

Replaces three separate operations in the MoE pipeline:
  1. SwiGLU: silu(gate) * up
  2. FP8 quantization with per-token-group scales
  3. Intermediate BF16 materialization

Input:  c1 [M, 2*N] BF16 — GEMM1 output with interleaved [up | gate] layout
Output: out [M, N] FP8 E4M3 — quantized SwiGLU result
        scales [M, N // GROUP_SIZE] FP32 — per-group quantization scales
"""

import triton
import triton.language as tl


@triton.jit
def swiglu_quant_kernel(
    c1_ptr,
    out_ptr,
    scale_ptr,
    M,
    N,
    stride_c1_m,
    stride_out_m,
    stride_scale_m,
    FP8_MAX: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused SwiGLU + FP8 E4M3 quantization.

    Each program processes one row of the input. Iterates over columns in
    blocks of BLOCK_N (which equals GROUP_SIZE for per-group quantization).

    Layout: c1[:, :N] = up projection, c1[:, N:] = gate projection.
    SwiGLU: silu(gate) * up = sigmoid(gate) * gate * up.
    """
    row = tl.program_id(0)

    c1_row_ptr = c1_ptr + row * stride_c1_m
    out_row_ptr = out_ptr + row * stride_out_m
    scale_row_ptr = scale_ptr + row * stride_scale_m

    num_groups = tl.cdiv(N, GROUP_SIZE)

    for group_idx in range(0, num_groups):
        col_start = group_idx * GROUP_SIZE
        offsets = col_start + tl.arange(0, BLOCK_N)
        mask = offsets < N

        # Load up (c1[:, :N]) and gate (c1[:, N:]) projections
        up = tl.load(c1_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(c1_row_ptr + N + offsets, mask=mask, other=0.0).to(tl.float32)

        # SwiGLU: silu(gate) * up = sigmoid(gate) * gate * up
        x = tl.sigmoid(gate) * gate * up

        # Per-group FP8 quantization
        amax = tl.max(tl.abs(x))
        # Avoid division by zero
        amax = tl.maximum(amax, 1e-10)
        scale = amax / FP8_MAX
        x_q = x / scale

        # Store quantized values and scale
        tl.store(out_row_ptr + offsets, x_q.to(tl.float8e4nv), mask=mask)
        tl.store(scale_row_ptr + group_idx, scale)
