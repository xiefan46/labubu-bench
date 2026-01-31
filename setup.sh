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
cd flashinfer-bench && pip install -v -e . && cd ..

if [ ! -d flashinfer-bench-starter-kit ]; then
    git clone https://github.com/xiefan46/flashinfer-bench-starter-kit.git
    cd flashinfer-bench-starter-kit && git remote add upstream https://github.com/flashinfer-ai/flashinfer-bench-starter-kit.git && cd ..
fi

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

${YELLOW}# Run MOE benchmark with flashinfer_moe solution only:${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 --solutions flashinfer_moe

${YELLOW}# Resume an interrupted run:${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --resume

${YELLOW}# Custom benchmark parameters:${RESET}
flashinfer-bench run --local $FIB_DATASET_PATH --warmup-runs 10 --iterations 100 --num-trials 5
"
