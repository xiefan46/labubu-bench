#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# labubu-bench setup script (re-entrant)
# Usage: git clone labubu-bench && cd labubu-bench && bash setup.sh
#
# Final structure:
#   ../miniconda3/          (conda, parallel to labubu-bench)
#   ./flashinfer-trace/     (dataset, inside labubu-bench)
#   ./projects/flashinfer/
#   ./projects/flashinfer-bench/
#   ./projects/flashinfer-bench-starter-kit/
# ============================================================

# ---------- Colors ----------
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

step() { echo -e "\n${BOLD}${CYAN}=> $1${RESET}"; }

# ---------- Miniconda (parallel to labubu-bench) ----------
step "Installing Miniconda"
if [ ! -d ../miniconda3 ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ../miniconda3
    rm -f Miniconda3-latest-Linux-x86_64.sh
fi

source ../miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---------- Conda environment ----------
step "Setting up conda environment"
if ! conda env list | grep -q "fi-bench"; then
    conda create -n fi-bench python=3.12 -y
fi
conda activate fi-bench

# ---------- Python packages ----------
step "Installing Python packages"
pip install flashinfer-python   # requires CUDA, SM80+ (A100/H100/B200)
pip install safetensors torch

# ---------- Git LFS ----------
step "Installing Git LFS"
apt-get update
apt-get install git-lfs -y
git lfs install

# ---------- Dataset ----------
step "Cloning flashinfer-trace dataset"
if [ ! -d flashinfer-trace ]; then
    git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
fi

# ---------- Projects ----------
step "Cloning projects"
mkdir -p projects
cd projects

if [ ! -d flashinfer ]; then
    git clone https://github.com/xiefan46/flashinfer.git
    cd flashinfer && git remote add upstream https://github.com/flashinfer-ai/flashinfer.git && cd ..
fi

if [ ! -d flashinfer-bench ]; then
    git clone https://github.com/xiefan46/flashinfer-bench.git
fi
cd flashinfer-bench && git checkout feat/cli-required-matched-ratio && pip install -v -e . && cd ..

if [ ! -d flashinfer-bench-starter-kit ]; then
    git clone https://github.com/xiefan46/flashinfer-bench-starter-kit.git
    cd flashinfer-bench-starter-kit && git remote add upstream https://github.com/flashinfer-ai/flashinfer-bench-starter-kit.git && cd ..
fi

# ---------- SGLang sgl-kernel (for sglang_fp8_blockwise_moe solution) ----------
step "Installing sgl_kernel"
pip install sgl-kernel

# ---------- Patch: fix flashinfer_moe solution bugs in flashinfer-trace ----------
step "Patching flashinfer-trace dataset"
MOE_SOL_DIR="$(cd .. && pwd)/flashinfer-trace/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
if [ -f "$MOE_SOL_DIR/flashinfer_wrapper_9sdjf3.json" ]; then
    python3 << PATCH_EOF
import json, re

f = "$MOE_SOL_DIR/flashinfer_wrapper_9sdjf3.json"
d = json.load(open(f))
patched = False

# Fix 1: missing destination_passing_style
if d["spec"].get("destination_passing_style") is not False:
    d["spec"]["destination_passing_style"] = False
    print("Patched: set destination_passing_style=false")
    patched = True

# Fix 2: tile_tokens_dim is not a public API param, remove it from source
for src in d.get("sources", []):
    if src["path"] == "main.py" and "tile_tokens_dim" in src["content"]:
        code = src["content"]
        # Remove _next_power_of_2 helper
        code = re.sub(r'\ndef _next_power_of_2\(.*?\n(?=\ndef |\n@|\Z)', '\n', code, flags=re.DOTALL)
        # Remove _get_tile_tokens_dim helper
        code = re.sub(r'\ndef _get_tile_tokens_dim\(.*?\n(?=\ndef |\n@|\Z)', '\n', code, flags=re.DOTALL)
        # Remove tile_tokens_dim local variable assignment
        code = re.sub(r'    tile_tokens_dim = _get_tile_tokens_dim\(.*?\n', '', code)
        # Remove tile_tokens_dim=tile_tokens_dim kwarg in function call
        code = re.sub(r'        tile_tokens_dim=tile_tokens_dim,\n', '', code)
        src["content"] = code
        print("Patched: removed tile_tokens_dim from main.py")
        patched = True

if patched:
    json.dump(d, open(f, "w"), indent=2)
else:
    print("Already patched, skipping")
PATCH_EOF
fi

# ---------- Add sglang_fp8_blockwise_moe solution ----------
step "Adding sglang_fp8_blockwise_moe solution"
SGLANG_SOL="$MOE_SOL_DIR/sglang_fp8_blockwise_moe.json"
if [ ! -f "$SGLANG_SOL" ]; then
    python3 << SGLANG_SOL_EOF
import json, os

MAIN_PY = r'''import torch
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
    # Applies sigmoid + bias + group selection (top-2 per group) +
    # top-4 groups + global top-8 experts, with weight normalization.
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
    # topk_weights: [T, TOP_K] float32, topk_ids: [T, TOP_K] int32

    # --- Step 2: Map global expert IDs to local [0, E_LOCAL) ---
    # Tokens routed to non-local experts get weight=0 (no contribution)
    local_mask = (topk_ids >= local_expert_offset) & (
        topk_ids < local_expert_offset + E_LOCAL
    )
    local_ids = (topk_ids - local_expert_offset).clamp(0, E_LOCAL - 1)
    topk_weights = topk_weights * local_mask.to(topk_weights.dtype)
    topk_ids = local_ids.to(torch.int32)

    m = T
    topk = TOP_K
    k = H   # hidden_size
    n = I   # intermediate_size
    num_experts = E_LOCAL

    # --- Step 3: Prepare MoE input (expert offsets, problem sizes, permutations) ---
    expert_offsets = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((m * topk,), dtype=torch.int32, device=device)
    c_map = torch.empty((m * topk,), dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # --- Step 4: Prepare activations ---
    # hidden_states: [T, H] fp8, hidden_states_scale: [H//128, T] float32
    # Transpose scale to [T, H//128] then shuffle both by a_map
    a_scale = hidden_states_scale.to(torch.float32).T.contiguous()
    rep_a_q = shuffle_rows(hidden_states.contiguous(), a_map, (m * topk, k))
    rep_a_scales = shuffle_rows(a_scale, a_map, (m * topk, k // BLOCK))

    # --- Step 5: Transpose weights to SGLang convention ---
    # SGLang CUTLASS kernel expects B as column-major [K, N] via transposed view
    # Definition [E, 2I, K] -> view as [E, K, 2I] (non-contiguous, column-major)
    w1_q = gemm1_weights.transpose(1, 2)
    w2_q = gemm2_weights.transpose(1, 2)   # [E, K, I] -> [E, I, K]
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

    # --- Step 7: GEMM1 (gate+up projection) ---
    # A[m*topk, K] @ W1[K, 2I] -> C1[m*topk, 2I]
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

    # --- Step 8: SwiGLU activation ---
    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.bfloat16)
    silu_and_mul(c1, intermediate)

    # --- Step 9: Quantize intermediate to FP8 for GEMM2 ---
    intermediate_q, a2_scale = _per_token_group_quant_fp8(intermediate, BLOCK)

    # --- Step 10: GEMM2 (down projection) ---
    # A[m*topk, I] @ W2[I, K] -> C2[m*topk, K]
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
'''

sol = {
    "name": "sglang_fp8_blockwise_moe",
    "definition": "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    "description": "Solution wrapping SGLang sgl_kernel fp8_blockwise MoE primitives with DeepSeek-V3 routing.",
    "author": "sglang",
    "spec": {
        "language": "python",
        "target_hardware": ["NVIDIA B200"],
        "dependencies": ["sgl_kernel"],
        "entry_point": "main.py::run",
        "destination_passing_style": False,
    },
    "sources": [{"path": "main.py", "content": MAIN_PY.strip() + "\n"}],
}

out = "$SGLANG_SOL"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(sol, open(out, "w"), indent=2)
print(f"Created: {out}")
SGLANG_SOL_EOF
else
    echo "Already exists, skipping"
fi

# ---------- Export FIB_DATASET_PATH to .bashrc ----------
FIB_DATASET_PATH="$(cd .. && pwd)/flashinfer-trace"
if ! grep -q "FIB_DATASET_PATH" ~/.bashrc 2>/dev/null; then
    echo "export FIB_DATASET_PATH=\"$FIB_DATASET_PATH\"" >> ~/.bashrc
fi
export FIB_DATASET_PATH

# ---------- Quick start hints ----------
echo -e "
${BOLD}${GREEN}========================================${RESET}
${BOLD}${GREEN}  Setup complete! Example commands:${RESET}
${BOLD}${GREEN}========================================${RESET}

${BOLD}${YELLOW}⚠  Run this first in every new terminal:${RESET}
${BOLD}${GREEN}   source ../miniconda3/bin/activate && conda activate fi-bench${RESET}

${YELLOW}# Run all benchmarks:${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH

${YELLOW}# Run MOE benchmark (MLSys 2026 competition):${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

${YELLOW}# Run MOE benchmark with flashinfer_moe solution only (use --timeout for JIT warmup):${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 --solutions flashinfer_moe --timeout 1200

${YELLOW}# Run MOE benchmark with relaxed FP8 tolerances (matching FlashInfer's own test thresholds):${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 --solutions flashinfer_moe --timeout 1200 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85

${YELLOW}# Run MOE benchmark with sglang_fp8_blockwise_moe solution (uses sgl_kernel):${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 --solutions sglang_fp8_blockwise_moe --timeout 600 --rtol 0.2 --atol 0.1 --required-matched-ratio 0.85

${YELLOW}# Resume an interrupted run:${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --resume

${YELLOW}# Custom benchmark parameters:${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --warmup-runs 10 --iterations 100 --num-trials 5
"
