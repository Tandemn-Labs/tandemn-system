#!/usr/bin/env python3
"""Repeat an existing JSONL workload N times with unique IDs, for stress/teardown testing."""
import json, sys

src = sys.argv[1] if len(sys.argv) > 1 else "examples/workloads/sharegpt-numreq_200-avginputlen_2926-avgoutputlen_100.jsonl"
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
dst = sys.argv[3] if len(sys.argv) > 3 else "examples/workloads/stress_1000.jsonl"

lines = open(src).readlines()
with open(dst, "w") as f:
    seq = 0
    for r in range(repeats):
        for line in lines:
            obj = json.loads(line)
            obj["custom_id"] = f"stress-{seq:05d}"
            seq += 1
            f.write(json.dumps(obj) + "\n")

print(f"Wrote {seq} requests to {dst}")
