"""Show error messages from trace files for a given solution.

Usage:
    python show_errors.py                          # show all errors
    python show_errors.py triton_fused_moe_v1      # filter by solution name
"""

import json
import os
import sys
from pathlib import Path

FIB_DATASET_PATH = os.environ.get("FIB_DATASET_PATH", "")
DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# Traces are stored as traces/moe/<definition>.jsonl
TRACE_FILE = Path(FIB_DATASET_PATH) / "traces" / "moe" / f"{DEFINITION}.jsonl"
# Fallback: traces/moe/<definition>/ directory
TRACE_DIR = Path(FIB_DATASET_PATH) / "traces" / "moe" / DEFINITION

solution_filter = sys.argv[1] if len(sys.argv) > 1 else None


def read_traces(path):
    """Read a trace file (JSON or JSONL)."""
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def collect_trace_files():
    """Find all trace files."""
    files = []
    if TRACE_FILE.exists():
        files.append(TRACE_FILE)
    if TRACE_DIR.exists() and TRACE_DIR.is_dir():
        files.extend(sorted(f for f in TRACE_DIR.iterdir() if f.is_file()))
    return files


trace_files = collect_trace_files()
if not trace_files:
    print(f"No trace files found at:")
    print(f"  {TRACE_FILE}")
    print(f"  {TRACE_DIR}/")
    print("Is FIB_DATASET_PATH set?")
    sys.exit(1)

found = 0
for tf in trace_files:
    try:
        traces = read_traces(tf)
    except Exception as e:
        print(f"Failed to parse {tf.name}: {e}")
        continue

    for t in traces:
        sol = t.get("solution", "")
        ev = t.get("evaluation", {})
        status = ev.get("status", t.get("status", ""))
        log = ev.get("log", "")
        error = t.get("error", "")

        if solution_filter and solution_filter not in sol:
            continue
        if "ERROR" not in status:
            continue

        found += 1
        msg = log or error or "(no error detail)"
        print(f"=== [{found}] {sol} | {status} ===")
        print(msg[:3000])
        print()

        if found >= 3:
            print(f"(showing first 3 errors)")
            sys.exit(0)

if found == 0:
    print("No errors found" + (f" for '{solution_filter}'" if solution_filter else ""))
