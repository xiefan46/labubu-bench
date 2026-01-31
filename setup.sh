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

# ---------- Miniconda (parallel to labubu-bench) ----------
if [ ! -d ../miniconda3 ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ../miniconda3
    rm -f Miniconda3-latest-Linux-x86_64.sh
fi

source ../miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---------- Conda environment ----------
if ! conda env list | grep -q "fi-bench"; then
    conda create -n fi-bench python=3.12 -y
fi
conda activate fi-bench

# ---------- Python packages ----------
pip install flashinfer-python   # requires CUDA, SM80+ (A100/H100/B200)
pip install safetensors torch

# ---------- Git LFS ----------
apt-get update
apt-get install git-lfs -y
git lfs install

# ---------- Dataset ----------
if [ ! -d flashinfer-trace ]; then
    git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
fi

# ---------- Projects ----------
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
