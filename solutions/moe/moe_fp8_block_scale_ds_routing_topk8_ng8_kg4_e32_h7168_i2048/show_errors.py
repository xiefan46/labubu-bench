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
TRACE_DIR = Path(FIB_DATASET_PATH) / "traces" / "moe" / DEFINITION

solution_filter = sys.argv[1] if len(sys.argv) > 1 else None


def read_traces(path):
    """Read a trace file (JSON or JSONL)."""
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


if not TRACE_DIR.exists():
    print(f"Trace directory not found: {TRACE_DIR}")
    print("Is FIB_DATASET_PATH set?")
    sys.exit(1)

found = 0
for trace_file in sorted(TRACE_DIR.iterdir()):
    if not trace_file.is_file():
        continue
    try:
        traces = read_traces(trace_file)
    except Exception as e:
        print(f"Failed to parse {trace_file.name}: {e}")
        continue

    for t in traces:
        sol = t.get("solution", "")
        status = t.get("status", "")
        error = t.get("error", "")

        if solution_filter and solution_filter not in sol:
            continue
        if not error:
            continue

        found += 1
        print(f"--- {sol} | {status} ---")
        print(error)
        print()

        if found >= 3:
            print(f"(showing first 3 errors, {TRACE_DIR} may have more)")
            sys.exit(0)

if found == 0:
    print("No errors found" + (f" for '{solution_filter}'" if solution_filter else ""))
