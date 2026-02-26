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

## Adding a New Solution

Solutions live in `solutions/<op_type>/<definition_name>/` within labubu-bench (NOT directly in flashinfer-trace). `setup.sh` runs `pack_solution.py --all` to generate JSON files and copies them into `flashinfer-trace/solutions/`.

### Steps

1. **Create source directory**: `solutions/<op_type>/<def_name>/<variant_name>/main.py` (and any other source files).
2. **Add variant to `config.toml`** in the same `<def_name>/` directory:
   ```toml
   [variants.<variant_key>]
   name = "<solution_name>"
   description = "..."
   source_dir = "<variant_name>"
   language = "python"
   entry_point = "main.py::run"
   dependencies = ["flashinfer"]          # or ["sgl_kernel", "triton"], etc.
   # destination_passing_style = false    # only if solution does NOT return output directly
   ```
3. **Commit and push** to labubu-bench. On the remote server, `setup.sh` will:
   - Run `pack_solution.py --all` → generates `<solution_name>.json`
   - Copy all `*.json` to `flashinfer-trace/solutions/<op_type>/<def_name>/`

### Key rules
- **Never commit solution JSON directly into flashinfer-trace** — always go through the `solutions/` + `config.toml` + `pack_solution.py` pipeline.
- `pack_solution.py` uses `pack_solution_from_files()` from flashinfer-bench, which reads all source files from the variant directory and embeds them into the JSON.
- The `run()` function signature must match the Definition's input spec exactly.

## Architecture Notes

- FlashInfer uses a **JIT compilation** model — CUDA kernels are compiled on first use, not at install time. Header templates live in `include/flashinfer/`, Python bindings in `csrc/`.
- FlashInfer-Bench's core abstraction: **Definition** (kernel interface spec) → **Solution** (implementation) → **Workload** (input config) → **Trace** (execution record with correctness/perf data).
- Supported op types: `rmsnorm`, `gemm`, `gqa_ragged`, `gqa_paged`, `mla_paged`, `dsa_paged`, `gdn`, `moe`, `sampling`.
- Kernel definitions and workloads are stored as JSON in `flashinfer-bench/flashinfer_trace/definitions/` and `flashinfer-bench/flashinfer_trace/workloads/`.
