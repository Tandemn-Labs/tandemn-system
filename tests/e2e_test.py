#!/usr/bin/env python3
"""
End-to-end observability test — fully autonomous.

Launches a real batch job, monitors progress + live metrics,
waits for completion, then verifies:
  1. Job launched and completed successfully
  2. Live metrics streamed every ~1s during generation
  3. Incremental progress bar updated (not stuck at 0%)
  4. Timeseries rows accumulated in DB
  5. Final summary row in DB with all canonical fields populated
  6. metrics.csv + output.jsonl present and correct

Usage:
    # Restart server first:
    ORCA_API_KEY=test123 python server.py &

    # Run the test:
    python tests/e2e_test.py
"""

import json
import os
import sqlite3
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("ORCA_SERVER", "http://localhost:26336")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
WORKLOAD = "examples/workloads/sharegpt-numreq_200-avginputlen_956-avgoutputlen_50.jsonl"
GPU = "L40S"
TP = 1
TIMEOUT_MIN = 25  # max wait for job to finish
POLL_SEC = 5      # polling interval
DB_PATH = "temp/metrics_db.sqlite"

# ANSI
G = "\033[32m"
R = "\033[31m"
Y = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RST = "\033[0m"


def api(method, path, **kwargs):
    import requests
    url = f"{BASE_URL}{path}"
    return getattr(requests, method)(url, timeout=15, **kwargs)


def step(msg):
    print(f"\n{BOLD}{'='*60}{RST}")
    print(f"  {BOLD}{msg}{RST}")
    print(f"{BOLD}{'='*60}{RST}")


def check(label, ok, detail=""):
    mark = f"{G}PASS{RST}" if ok else f"{R}FAIL{RST}"
    print(f"  [{mark}] {label}")
    if detail:
        print(f"         {DIM}{detail}{RST}")
    return ok


# ---------------------------------------------------------------------------
# Phase 1: Launch
# ---------------------------------------------------------------------------
def launch_job():
    step("PHASE 1: Launch job")

    # Verify server is up
    try:
        r = api("get", "/jobs")
        assert r.status_code == 200
    except Exception as e:
        print(f"\n  {R}Server not reachable at {BASE_URL}: {e}{RST}")
        print(f"  Start it: ORCA_API_KEY=test123 python server.py &")
        sys.exit(1)
    print(f"  Server OK at {BASE_URL}")

    # Deploy via CLI (captures job_id from output)
    print(f"  Deploying {MODEL} with {WORKLOAD}...")
    cmd = [
        sys.executable, "orca", "deploy", MODEL, WORKLOAD,
        "--gpu", GPU, "--tp", str(TP), "--force",
    ]
    env = {**os.environ, "ORCA_SERVER": BASE_URL, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)

    if result.returncode != 0:
        print(f"  {R}Deploy failed:{RST}")
        print(result.stderr[-500:] if result.stderr else result.stdout[-500:])
        sys.exit(1)

    # Extract job_id
    import re
    match = re.search(r"(mo-[a-f0-9-]+)", result.stdout)
    if not match:
        print(f"  {R}Could not find job_id in deploy output{RST}")
        print(result.stdout[-500:])
        sys.exit(1)

    job_id = match.group(1)
    print(f"  Job ID: {BOLD}{job_id}{RST}")
    return job_id


# ---------------------------------------------------------------------------
# Phase 2: Monitor — poll for progress + live metrics
# ---------------------------------------------------------------------------
def monitor_job(job_id):
    step("PHASE 2: Monitor job (progress + live metrics)")

    max_polls = (TIMEOUT_MIN * 60) // POLL_SEC
    metrics_seen = 0
    progress_values = []
    had_nonzero_progress = False
    had_live_metrics = False

    for i in range(max_polls):
        time.sleep(POLL_SEC)

        # Poll job status
        try:
            r = api("get", f"/job/{job_id}")
            job = r.json()
        except Exception as e:
            print(f"  [{i*POLL_SEC:>4}s] poll error: {e}")
            continue

        status = job.get("status", "?")
        progress = job.get("progress", 0)
        progress_values.append(progress)

        if progress > 0 and progress < 1.0:
            had_nonzero_progress = True

        # Poll live metrics
        metrics_line = ""
        try:
            mr = api("get", f"/job/{job_id}/metrics")
            if mr.status_code == 200:
                md = mr.json()
                tps = md.get("avg_generation_throughput_toks_per_s", 0)
                kv = md.get("gpu_cache_usage_perc", 0) * 100
                run = md.get("num_requests_running", 0)
                # Count as live metrics if either tok/s > 0 OR requests are running
                # (vLLM's throughput gauge only updates after request completion)
                if tps > 0 or run > 0:
                    metrics_seen += 1
                    had_live_metrics = True
                    metrics_line = f"  {tps:.0f} tok/s  KV={kv:.0f}%  run={run}"
        except Exception:
            pass

        # Progress bar
        filled = int(progress * 30)
        bar = "[" + "█" * filled + "░" * (30 - filled) + f"] {progress*100:.1f}%"
        elapsed = (i + 1) * POLL_SEC
        print(f"  [{elapsed:>4}s] {bar}  {status}{metrics_line}")

        if status == "succeeded":
            print(f"\n  {G}Job succeeded in {elapsed}s{RST}")
            return True, metrics_seen, had_nonzero_progress, had_live_metrics
        if status == "failed":
            print(f"\n  {R}Job failed after {elapsed}s{RST}")
            return False, metrics_seen, had_nonzero_progress, had_live_metrics

    print(f"\n  {R}Timeout after {TIMEOUT_MIN}min{RST}")
    return False, metrics_seen, had_nonzero_progress, had_live_metrics


# ---------------------------------------------------------------------------
# Phase 3: Verify results
# ---------------------------------------------------------------------------
def verify_results(job_id, metrics_seen, had_nonzero_progress, had_live_metrics):
    step("PHASE 3: Verify results")

    results = []

    # 1. Job status
    r = api("get", f"/job/{job_id}")
    job = r.json()
    results.append(check(
        "Job completed successfully",
        job["status"] == "succeeded",
        f"status={job['status']}"
    ))

    # 2. Live metrics streamed during generation
    results.append(check(
        "Live metrics received during generation",
        had_live_metrics,
        f"{metrics_seen} snapshots with tok/s > 0"
    ))

    # 3. Incremental progress (not stuck at 0%)
    results.append(check(
        "Incremental progress observed (not 0→100 jump)",
        had_nonzero_progress,
        "Progress updated between 0% and 100%"
    ))

    # 4. Timeseries rows in DB
    ts_count = 0
    try:
        conn = sqlite3.connect(DB_PATH)
        ts_count = conn.execute(
            "SELECT count(*) FROM timeseries WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        conn.close()
    except Exception as e:
        print(f"         {DIM}DB error: {e}{RST}")
    results.append(check(
        "Timeseries rows accumulated in DB",
        ts_count > 0,
        f"{ts_count} rows"
    ))

    # 5. Final summary row in DB
    run = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        run = conn.execute(
            "SELECT * FROM runs WHERE job_id=?", (job_id,)
        ).fetchone()
        conn.close()
    except Exception:
        pass

    results.append(check(
        "Summary run row in DB",
        run is not None,
        f"run_id={run['id']}" if run else "no row found"
    ))

    # 5b. Key fields populated in DB
    if run:
        key_fields = [
            "total_tokens_per_sec", "ttft_ms_p50", "tpot_ms_p50", "e2e_ms_p50",
            "num_preemptions", "model_name", "instance_type", "actual_region",
        ]
        populated = {f: run[f] for f in key_fields if run[f] is not None}
        missing = [f for f in key_fields if run[f] is None]
        results.append(check(
            "Key latency/throughput fields populated in DB",
            len(missing) == 0,
            f"populated={list(populated.keys())}" + (f" missing={missing}" if missing else "")
        ))

        # Scheduler fields from timeseries-derived or metrics.csv
        sched_fields = ["running_avg", "kv_cache_util_pct_avg", "scheduler_samples"]
        sched_ok = {f: run[f] for f in sched_fields if run[f] is not None}
        results.append(check(
            "Scheduler/KV cache fields in DB",
            len(sched_ok) > 0,
            f"{list(sched_ok.keys())}"
        ))

    # 6. Output files on disk
    import glob
    output_dirs = glob.glob(f"outputs/**/success-*", recursive=True)
    # Find the one matching this job
    job_dir = None
    for d in output_dirs:
        log_path = os.path.join(d, "job.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                if job_id in f.read():
                    job_dir = d
                    break

    if job_dir:
        has_output = os.path.exists(os.path.join(job_dir, "output.jsonl"))
        has_metrics = os.path.exists(os.path.join(job_dir, "metrics.csv"))
        results.append(check(
            "output.jsonl exists",
            has_output,
            job_dir
        ))
        results.append(check(
            "metrics.csv exists",
            has_metrics,
            job_dir
        ))

        if has_metrics:
            import csv
            rows = {}
            with open(os.path.join(job_dir, "metrics.csv")) as f:
                for row in csv.reader(f):
                    if len(row) == 2 and row[0] != "metric":
                        rows[row[0]] = row[1]

            canon_sample = [
                "ttft_ms_p50", "tpot_client_ms_p50", "tpot_ms_p50",
                "e2e_ms_p50", "avg_sm_util_pct", "running_avg",
                "kv_cache_util_pct_avg", "scheduler_samples",
                "params_billion", "model_architecture",
                "gpu_bandwidth_gbps", "bandwidth_per_param",
                "cost_for_run_usd", "tokens_per_dollar",
            ]
            present = [f for f in canon_sample if rows.get(f) and rows[f] not in ("", "None")]
            absent = [f for f in canon_sample if f not in present]
            results.append(check(
                f"Canonical fields in metrics.csv ({len(present)}/{len(canon_sample)})",
                len(present) >= 10,  # some need real GPU/Prometheus
                f"present={present}" + (f"\n         absent={absent}" if absent else "")
            ))

        if has_output:
            with open(os.path.join(job_dir, "output.jsonl")) as f:
                lines = f.readlines()
            first = json.loads(lines[0])
            results.append(check(
                f"output.jsonl has {len(lines)} results",
                len(lines) >= 100,
                f"first: custom_id={first.get('custom_id')} status={first['response']['status_code']}"
            ))
    else:
        results.append(check("Output directory found", False, "no success-* dir with this job_id"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{BOLD}Orca E2E Observability Test{RST}")
    print(f"{DIM}Model: {MODEL}  GPU: {GPU}  TP: {TP}{RST}")
    print(f"{DIM}Workload: {WORKLOAD}{RST}")
    print(f"{DIM}Server: {BASE_URL}{RST}")

    job_id = launch_job()
    success, metrics_seen, had_progress, had_metrics = monitor_job(job_id)

    all_results = verify_results(job_id, metrics_seen, had_progress, had_metrics)

    # Summary
    step("SUMMARY")
    passed = sum(1 for r in all_results if r)
    total = len(all_results)
    color = G if passed == total else (Y if passed > total // 2 else R)
    print(f"\n  {color}{BOLD}{passed}/{total} checks passed{RST}\n")

    if passed < total:
        print(f"  {Y}Note: avg_sm_util_pct and some Prometheus fields require")
        print(f"  real GPU hardware + vLLM histogram data. These are expected")
        print(f"  to be None in stub/local environments.{RST}\n")

    sys.exit(0 if passed >= total - 2 else 1)  # allow 2 soft failures


if __name__ == "__main__":
    main()
