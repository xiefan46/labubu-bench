# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

labubu-bench is a monorepo for the [MLSys 2026 FlashInfer AI Kernel Competition](https://mlsys26.flashinfer.ai/), orchestrating three interconnected GPU kernel optimization projects for LLM inference:

- **projects/flashinfer/** — High-performance GPU kernel library (attention, GEMM, MoE, sampling, communication). Uses JIT compilation via TVM-FFI.
- **projects/flashinfer-bench/** — Benchmarking framework with a Definition→Solution→Workload→Trace pipeline. Supports AI-driven kernel optimization.
- **projects/flashinfer-bench-starter-kit/** — MLSys2026 competition template for kernel generation on NVIDIA Blackwell GPUs.

## Setup

```bash
git clone https://github.com/xiefan46/labubu-bench.git
cd labubu-bench
bash setup.sh
```

This installs miniconda (parallel to repo), creates `fi-bench` conda env, clones all three projects under `projects/`, and downloads the flashinfer-trace dataset.

## Repository Structure

```
labubu-bench/
├── setup.sh                  # Environment setup script
├── flashinfer-trace/         # HuggingFace dataset (created by setup.sh)
└── projects/
    ├── flashinfer/           # origin: xiefan46/flashinfer, upstream: flashinfer-ai/flashinfer
    ├── flashinfer-bench/     # origin: xiefan46/flashinfer-bench
    └── flashinfer-bench-starter-kit/  # origin: xiefan46/flashinfer-bench-starter-kit, upstream: flashinfer-ai/flashinfer-bench-starter-kit
```

The `projects/` directory is gitignored — each sub-project is an independent git repo, not a submodule.

## Sub-Project Development

Each sub-project has its own `CLAUDE.md` with detailed guidance. Refer to those when working within a specific project:

- `projects/flashinfer/CLAUDE.md` — JIT compilation, CUDA kernel development, testing, benchmarking
- `projects/flashinfer-bench/CLAUDE.md` — Model addition, kernel definitions, solution schema, agent workflows

### flashinfer-bench key commands

```bash
cd projects/flashinfer-bench
pip install -v -e .                          # editable install
pytest tests/                                # run all tests
pytest tests/data/test_definition.py -k "test_name"  # single test
pre-commit run --all-files                   # lint (black, isort, yamlfmt, taplo)
```

### flashinfer key commands

```bash
cd projects/flashinfer
pip install -e . --no-build-isolation        # editable install (JIT mode, no reinstall needed for changes)
pytest tests/test_norm.py -k "test_name"     # single test
```

### flashinfer-bench-starter-kit key commands

```bash
cd projects/flashinfer-bench-starter-kit
python scripts/run_local.py                  # local GPU benchmark
python scripts/pack_solution.py              # pack for submission
```

## Development Rules

- **setup.sh must be re-entrant**: Any modification to `setup.sh` must ensure the script can be run multiple times safely. Use existence checks (`if [ ! -d ... ]`, `if ! conda env list | grep -q ...`, etc.) to skip already-completed steps.
- **Ask before implementing**: For complex analysis, design, or coding tasks, clarify any uncertainties with the user before starting implementation.
- **Persist important context**: User preferences, lessons learned, and behavioral guidelines that should survive across sessions must be written into `CLAUDE.md` or `knowledge/`. If something would be lost when a new context starts, save it now.
- **Proactive communication**: If unsure about something, or have suggestions (e.g., new knowledge categories, workflow improvements), proactively discuss with the user rather than staying silent.

## Knowledge Base

The `knowledge/` directory contains accumulated project knowledge that persists across sessions.

```
knowledge/
├── debug/          # Project-specific error records and solutions (e.g., tolerance mismatches, JIT timeouts)
├── papers/         # Key research papers (DeepSeek V3, FlashInfer, etc.)
├── benchmarks/     # Performance results and comparison data across configurations
└── architecture/   # System architecture notes (call chains, evaluation pipelines, etc.)
```

### Knowledge base rules (MUST follow)

1. **Read when relevant**: Before debugging, benchmarking, or investigating an issue, judge whether `knowledge/` likely has useful prior findings. If so, search the relevant subdirectory (e.g., `knowledge/debug/` for bugs, `knowledge/benchmarks/` for perf data). Don't read everything blindly — use judgment on what's relevant.
2. **Write after work**: After completing any debugging, benchmarking, architecture investigation, or significant implementation, proactively write findings to the appropriate `knowledge/` subdirectory. Do NOT wait for the user to remind you.
3. **What to record**: Bug root causes and fixes, benchmark results, API/library quirks discovered, architectural decisions and rationale, workload parameter analysis, any insight that would save time if encountered again.
4. **Update existing files**: If new information relates to an existing knowledge file, update that file rather than creating a new one.

## Architecture Notes

- FlashInfer uses a **JIT compilation** model — CUDA kernels are compiled on first use, not at install time. Header templates live in `include/flashinfer/`, Python bindings in `csrc/`.
- FlashInfer-Bench's core abstraction: **Definition** (kernel interface spec) → **Solution** (implementation) → **Workload** (input config) → **Trace** (execution record with correctness/perf data).
- Supported op types: `rmsnorm`, `gemm`, `gqa_ragged`, `gqa_paged`, `mla_paged`, `dsa_paged`, `gdn`, `moe`, `sampling`.
- Kernel definitions and workloads are stored as JSON in `flashinfer-bench/flashinfer_trace/definitions/` and `flashinfer-bench/flashinfer_trace/workloads/`.
