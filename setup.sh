#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# labubu-bench setup script
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
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ../miniconda3

source ../miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---------- Conda environment ----------
conda create -n fi-bench python=3.12 -y
conda activate fi-bench

# ---------- Python packages ----------
pip install flashinfer-python   # requires CUDA, SM80+ (A100/H100/B200)
pip install safetensors torch

# ---------- Git LFS ----------
apt-get update
apt-get install git-lfs -y
git lfs install

# ---------- Dataset ----------
git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace

# ---------- Projects ----------
mkdir -p projects
cd projects

git clone https://github.com/xiefan46/flashinfer.git
cd flashinfer && git remote add upstream https://github.com/flashinfer-ai/flashinfer.git && cd ..

git clone https://github.com/xiefan46/flashinfer-bench.git
cd flashinfer-bench && pip install -v -e . && cd ..

git clone -b moe-kernel-optimizations https://github.com/xiefan46/flashinfer-bench-starter-kit.git
cd flashinfer-bench-starter-kit && git remote add upstream https://github.com/flashinfer-ai/flashinfer-bench-starter-kit.git && cd ..
