#!/usr/bin/env python3
"""Check correctness details from benchmark trace files."""
import json
import sys
import os

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default path
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "flashinfer-trace/traces/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl",
        )

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            workload = d.get("workload", "?")
            for ev in d.get("evaluations", []):
                status = ev.get("status", "?")
                c = ev.get("correctness", {})
                print(f"[{i}] workload={workload[:36]}  status={status}")
                if c:
                    print(f"     max_abs_error={c.get('max_absolute_error')}")
                    print(f"     max_rel_error={c.get('max_relative_error')}")
                    print(f"     extra={c.get('extra')}")
                log = ev.get("log", "")
                if log:
                    # Print first 500 chars of log
                    print(f"     log={log[:500]}")
                print()

if __name__ == "__main__":
    main()
