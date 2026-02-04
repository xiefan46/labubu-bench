#!/usr/bin/env bash
set -uo pipefail

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
RED='\033[0;31m'
RESET='\033[0m'

step() { echo -e "\n${BOLD}${CYAN}=> $1${RESET}"; }
warn() { echo -e "${YELLOW}WARNING: $1${RESET}"; }
fail() { echo -e "${RED}FAILED: $1${RESET}"; }

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ERRORS=0

# Helper: run a command, warn on failure but continue
run_or_warn() {
    if ! "$@"; then
        warn "Command failed: $*"
        ERRORS=$((ERRORS + 1))
    fi
}

# ---------- Miniconda (parallel to labubu-bench) ----------
step "Installing Miniconda"
if [ ! -d ../miniconda3 ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ../miniconda3
    rm -f Miniconda3-latest-Linux-x86_64.sh
fi

source ../miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---------- Conda environment ----------
step "Setting up conda environment"
if ! conda env list | grep -q "fi-bench"; then
    conda create -n fi-bench python=3.12 -y
fi
conda activate fi-bench

# ---------- Python packages ----------
step "Installing Python packages"
run_or_warn pip install safetensors torch pytest
# Pin cuda-python to match CUDA 12.8 driver; cuda-python>=13.x causes
# cudaErrorInsufficientDriver on machines with CUDA 12.8 drivers.
run_or_warn pip install "cuda-python>=12.8,<13"

# ---------- Git LFS + Nsight Systems ----------
step "Installing Git LFS and Nsight Systems"
run_or_warn apt-get update
run_or_warn apt-get install git-lfs -y
apt-get install nsight-systems-cli -y 2>/dev/null || warn "nsight-systems-cli not available, skipping"
run_or_warn git lfs install

# ---------- Dataset ----------
step "Cloning flashinfer-trace dataset"
if [ ! -d flashinfer-trace ]; then
    run_or_warn git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
fi

# ---------- Projects ----------
step "Cloning projects"
mkdir -p projects
cd projects

if [ ! -d flashinfer ]; then
    run_or_warn git clone --recursive https://github.com/xiefan46/flashinfer.git
    if [ -d flashinfer ]; then
        (cd flashinfer && git remote add upstream https://github.com/flashinfer-ai/flashinfer.git 2>/dev/null || true)
    fi
fi
if [ -d flashinfer ]; then
    (cd flashinfer && run_or_warn git checkout feat/cutedsl-fp8-moe && run_or_warn git submodule update --init --recursive && run_or_warn pip install --no-build-isolation -e . -v)
fi

if [ ! -d flashinfer-bench ]; then
    run_or_warn git clone https://github.com/xiefan46/flashinfer-bench.git
fi
if [ -d flashinfer-bench ]; then
    (cd flashinfer-bench && run_or_warn git checkout feat/cli-required-matched-ratio && run_or_warn pip install -v -e .)
fi

if [ ! -d flashinfer-bench-starter-kit ]; then
    run_or_warn git clone https://github.com/xiefan46/flashinfer-bench-starter-kit.git
    if [ -d flashinfer-bench-starter-kit ]; then
        (cd flashinfer-bench-starter-kit && git remote add upstream https://github.com/flashinfer-ai/flashinfer-bench-starter-kit.git 2>/dev/null || true)
    fi
fi

cd "$REPO_ROOT"

# ---------- SGLang sgl-kernel (for sglang_fp8_blockwise_moe solution) ----------
step "Installing sgl_kernel"
run_or_warn pip install sgl-kernel

# ============================================================
# Everything below this line is patch/pack/copy — MUST always run
# ============================================================

# ---------- Patch: fix flashinfer_moe solution bugs in flashinfer-trace ----------
step "Patching flashinfer-trace dataset"
if [ -d "$REPO_ROOT/flashinfer-trace/solutions" ]; then
    # Find the flashinfer_moe JSON anywhere under flashinfer-trace/solutions/ (directory structure may vary)
    FLASHINFER_MOE_JSON=$(find "$REPO_ROOT/flashinfer-trace/solutions" -name '*.json' -exec grep -l '"flashinfer_moe"' {} \; 2>/dev/null | head -1)
    if [ -n "$FLASHINFER_MOE_JSON" ]; then
        python3 << PATCH_EOF
import json, re

f = "$FLASHINFER_MOE_JSON"
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
    else
        echo "flashinfer_moe JSON not found in dataset, skipping patch"
    fi
else
    echo "flashinfer-trace/solutions not found, skipping patch"
fi

# ---------- Clear flashinfer-bench solution cache ----------
step "Clearing solution build cache"
if [ -d ~/.cache/flashinfer_bench/cache/python ]; then
    rm -rf ~/.cache/flashinfer_bench/cache/python/fib_python_*
    echo "Cleared python solution cache"
fi

# ---------- Pack & copy custom solutions to flashinfer-trace dataset ----------
step "Packing custom solutions"
REPO_SOL_DIR="$REPO_ROOT/solutions"
if [ -d "$REPO_SOL_DIR" ]; then
    find "$REPO_SOL_DIR" -name 'pack_solution.py' | while read pack_script; do
        pack_dir="$(dirname "$pack_script")"
        echo "Packing: $pack_script"
        (cd "$pack_dir" && python pack_solution.py --all)
    done
fi

step "Copying custom solutions"
if [ -d "$REPO_SOL_DIR" ]; then
    find "$REPO_SOL_DIR" -name '*.json' ! -name 'pack_solution.py' | while read src; do
        # Mirror the directory structure: solutions/moe/def_name/sol.json -> flashinfer-trace/solutions/moe/def_name/sol.json
        rel="${src#$REPO_SOL_DIR/}"
        dst="$REPO_ROOT/flashinfer-trace/solutions/$rel"
        # Skip if a solution with the same name already exists anywhere in the dataset (from HuggingFace)
        sol_name=$(python3 -c "import json; print(json.load(open('$src')).get('name',''))" 2>/dev/null || echo "")
        if [ -n "$sol_name" ]; then
            existing=$(find "$REPO_ROOT/flashinfer-trace/solutions" -name '*.json' -exec grep -l "\"name\": \"$sol_name\"" {} \; 2>/dev/null | head -1)
            if [ -n "$existing" ] && [ "$existing" != "$dst" ]; then
                echo "Skipped: $rel (already exists as $existing)"
                continue
            fi
        fi
        mkdir -p "$(dirname "$dst")"
        cp -f "$src" "$dst"
        echo "Copied: $rel"
    done
else
    echo "No custom solutions directory found, skipping"
fi

# ---------- Export FIB_DATASET_PATH to .bashrc ----------
FIB_DATASET_PATH="$REPO_ROOT/flashinfer-trace"
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

if [ $ERRORS -gt 0 ]; then
    warn "$ERRORS step(s) had non-fatal errors (see above)"
fi
