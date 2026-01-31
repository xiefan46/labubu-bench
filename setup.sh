#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# labubu-bench setup script
# Sets up Miniconda, creates conda env, installs dependencies,
# and prepares flashinfer-bench.
# ============================================================

# ---------- Miniconda ----------
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$(pwd)/miniconda3"

# Initialize the current shell
source "$(pwd)/miniconda3/bin/activate"

# Accept Anaconda TOS
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

# ---------- Datasets & benchmark repo ----------
git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace

git clone https://github.com/flashinfer-ai/flashinfer-bench.git
cd flashinfer-bench
pip install -v -e .
