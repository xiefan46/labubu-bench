#!/usr/bin/env python3
"""Check correctness details from benchmark trace files."""
import json
import sys
import os
import glob


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Auto-find jsonl files under traces/
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "flashinfer-trace/traces",
        )
        files = glob.glob(os.path.join(base, "**/*.jsonl"), recursive=True)
        if not files:
            print(f"No .jsonl files found under {base}")
            sys.exit(1)
        path = files[0]
        print(f"Reading: {path}\n")

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)

            # Extract workload identifier
            wl = d.get("workload", {})
            if isinstance(wl, dict):
                axes = wl.get("axes", {})
                wl_str = str(axes)
            else:
                wl_str = str(wl)[:40]

            # Handle both "evaluation" (singular) and "evaluations" (plural)
            ev = d.get("evaluation")
            evs = d.get("evaluations", [])
            if ev:
                evs = [ev] if isinstance(ev, dict) else ev

            for ev in evs:
                status = ev.get("status", "?")
                c = ev.get("correctness") or {}
                print(f"[{i}] workload={wl_str}  status={status}")
                if c:
                    print(f"     max_abs_error={c.get('max_absolute_error')}")
                    print(f"     max_rel_error={c.get('max_relative_error')}")
                    extra = c.get("extra")
                    if extra:
                        print(f"     extra={extra}")
                log = ev.get("log", "")
                if log:
                    print(f"     log={log[:500]}")
                print()


if __name__ == "__main__":
    main()
