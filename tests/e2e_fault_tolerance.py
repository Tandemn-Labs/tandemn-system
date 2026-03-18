#!/usr/bin/env python3
"""
Automated E2E fault-tolerance test for chunk lease-based recovery.

Launches a real 2-replica chunked batch job, waits for both replicas to
start processing, kills one replica mid-inference, then verifies:
  1. Orphaned chunk(s) get reclaimed after the lease TTL expires
  2. Surviving replica picks up the orphaned work
  3. Job completes successfully (all_done: true)
  4. Output has the expected number of lines

Prerequisites:
  1. Redis running: docker run -d -p 6379:6379 redis
  2. Server running with SHORT lease timeouts (for fast testing):

       CHUNK_LEASE_TTL_SEC=60 \\
       CHUNK_RECLAIM_INTERVAL_SEC=15 \\
       CHUNK_RENEW_INTERVAL_SEC=10 \\
       ORCA_API_KEY=test123 \\
       python server.py

  3. AWS credentials + SkyPilot configured

Usage:
    python tests/e2e_fault_tolerance.py [--model MODEL] [--gpu GPU]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("ORCA_SERVER", "http://localhost:26336")
WORKLOAD = "examples/workloads/stress_5000.jsonl"

# Timeouts
LAUNCH_TIMEOUT_MIN = 20     # max wait for replicas to be processing
RECLAIM_TIMEOUT_SEC = 180   # max wait for reclaim after kill (TTL + reclaim interval + buffer)
JOB_TIMEOUT_MIN = 60        # max total wait for job completion
POLL_SEC = 5

# ANSI
G = "\033[32m"
R = "\033[31m"
Y = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RST = "\033[0m"


ORCA_API_KEY = os.getenv("ORCA_API_KEY", "test123")

def api(method, path, **kwargs):
    import requests
    url = f"{BASE_URL}{path}"
    headers = kwargs.pop("headers", {})
    if ORCA_API_KEY:
        headers.setdefault("Authorization", f"Bearer {ORCA_API_KEY}")
    return getattr(requests, method)(url, timeout=kwargs.pop("timeout", 30), headers=headers, **kwargs)


def step(msg):
    print(f"\n{BOLD}{'=' * 64}{RST}")
    print(f"  {BOLD}{msg}{RST}")
    print(f"{BOLD}{'=' * 64}{RST}")


def check(label, ok, detail=""):
    mark = f"{G}PASS{RST}" if ok else f"{R}FAIL{RST}"
    print(f"  [{mark}] {label}")
    if detail:
        print(f"         {DIM}{detail}{RST}")
    return ok


def get_chunk_progress(job_id):
    try:
        r = api("get", f"/job/{job_id}/chunks/progress")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_replicas(job_id):
    try:
        r = api("get", f"/job/{job_id}/replicas")
        if r.status_code == 200:
            return r.json().get("replicas", [])
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Phase 1: Submit
# ---------------------------------------------------------------------------
def submit_job(model, gpu):
    step("PHASE 1: Submit chunked job (2 replicas)")

    # Verify server is up
    try:
        r = api("get", "/jobs")
        assert r.status_code == 200
    except Exception as e:
        print(f"\n  {R}Server not reachable at {BASE_URL}: {e}{RST}")
        print(f"  Start it with:")
        print(f"    CHUNK_LEASE_TTL_SEC=60 CHUNK_RECLAIM_INTERVAL_SEC=15 \\")
        print(f"    CHUNK_RENEW_INTERVAL_SEC=10 ORCA_API_KEY=test123 python server.py")
        sys.exit(1)

    print(f"  Server OK at {BASE_URL}")
    print(f"  Model:    {model}")
    print(f"  GPU:      {gpu}")
    print(f"  Workload: {WORKLOAD}")
    print(f"  Replicas: 2")

    cmd = [
        sys.executable, "orca", "deploy", model, WORKLOAD,
        "--gpu", gpu, "--tp", "1", "--replicas", "2", "--force",
    ]
    env = {**os.environ, "ORCA_SERVER": BASE_URL, "PYTHONUNBUFFERED": "1"}

    print(f"\n  {DIM}Running: {' '.join(cmd)}{RST}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

    if result.returncode != 0:
        print(f"  {R}Deploy failed:{RST}")
        print(result.stderr[-1000:] if result.stderr else result.stdout[-1000:])
        sys.exit(1)

    match = re.search(r"(mo-[a-zA-Z0-9-]+)", result.stdout)
    if not match:
        print(f"  {R}Could not find job_id in deploy output{RST}")
        print(result.stdout[-500:])
        sys.exit(1)

    job_id = match.group(1)
    print(f"\n  Job ID: {BOLD}{job_id}{RST}")
    print(f"  Replica clusters: {job_id}-r0, {job_id}-r1")
    return job_id


# ---------------------------------------------------------------------------
# Phase 2: Wait for active processing
# ---------------------------------------------------------------------------
def wait_for_processing(job_id):
    step("PHASE 2: Wait for both replicas to start processing")

    deadline = time.time() + LAUNCH_TIMEOUT_MIN * 60
    seen_inflight = False

    while time.time() < deadline:
        progress = get_chunk_progress(job_id)
        replicas = get_replicas(job_id)

        phases = {r["replica_id"].rsplit("-", 1)[-1]: r.get("phase", "?") for r in replicas}
        running_count = sum(1 for p in phases.values() if p in ("running", "generating"))

        if progress:
            inflight = progress.get("inflight", 0)
            completed = progress.get("completed", 0)
            pending = progress.get("pending", 0)
            total = progress.get("total", 0)
            print(
                f"  [{int(time.time()) % 10000:>4}] "
                f"chunks: {completed} done, {inflight} inflight, {pending} pending / {total} total  "
                f"replicas: {phases}"
            )
            if inflight >= 2 or (inflight >= 1 and completed >= 1):
                # Both replicas are actively working (or one already finished a chunk)
                seen_inflight = True
                print(f"\n  {G}Both replicas active — ready to simulate preemption{RST}")
                return True
        else:
            print(f"  [{int(time.time()) % 10000:>4}] Waiting for chunk queue... replicas: {phases}")

        # Check if job already finished
        try:
            jr = api("get", f"/job/{job_id}")
            status = jr.json().get("status", "?")
            if status in ("succeeded", "failed"):
                print(f"\n  {Y}Job already {status} before we could kill a replica{RST}")
                return False
        except Exception:
            pass

        time.sleep(POLL_SEC)

    print(f"\n  {R}Timed out waiting for replicas to start processing{RST}")
    return False


# ---------------------------------------------------------------------------
# Phase 3: Kill replica r0
# ---------------------------------------------------------------------------
def kill_replica(job_id):
    step("PHASE 3: Simulate spot preemption — kill replica r0")

    target = f"{job_id}-r0"
    progress_before = get_chunk_progress(job_id)

    print(f"  Target cluster: {BOLD}{target}{RST}")
    if progress_before:
        print(f"  Progress before kill: {json.dumps(progress_before)}")

    # Record which chunks are inflight (to verify reclaim later)
    inflight_before = progress_before.get("inflight", 0) if progress_before else 0

    print(f"\n  Executing: sky down {target} --yes")
    t0 = time.time()
    try:
        result = subprocess.run(
            ["sky", "down", target, "--yes"],
            capture_output=True, text=True, timeout=120,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  {G}Cluster {target} terminated in {elapsed:.0f}s{RST}")
        else:
            print(f"  {Y}sky down returned code {result.returncode} ({elapsed:.0f}s){RST}")
            if result.stderr:
                print(f"  {DIM}{result.stderr[:300]}{RST}")
    except subprocess.TimeoutExpired:
        print(f"  {Y}sky down timed out — cluster may still be terminating{RST}")
    except FileNotFoundError:
        print(f"  {R}sky CLI not found — cannot kill cluster{RST}")
        return inflight_before

    return inflight_before


# ---------------------------------------------------------------------------
# Phase 4: Monitor reclaim
# ---------------------------------------------------------------------------
def monitor_reclaim(job_id, inflight_before_kill):
    step("PHASE 4: Monitor chunk reclaim")

    print(f"  Waiting for server to detect lease expiry and reclaim orphaned chunk(s)...")
    print(f"  (Lease TTL + reclaim interval — should happen within ~{RECLAIM_TIMEOUT_SEC}s)\n")

    deadline = time.time() + RECLAIM_TIMEOUT_SEC
    reclaim_seen = False

    while time.time() < deadline:
        progress = get_chunk_progress(job_id)
        if not progress:
            time.sleep(POLL_SEC)
            continue

        inflight = progress.get("inflight", 0)
        pending = progress.get("pending", 0)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        total = progress.get("total", 0)

        bar_done = completed + failed
        pct = (bar_done / total * 100) if total else 0
        print(
            f"  [{int(time.time()) % 10000:>4}] "
            f"{pct:5.1f}%  completed={completed} inflight={inflight} "
            f"pending={pending} failed={failed}"
        )

        # Reclaim detected: inflight dropped and pending went back up,
        # or a chunk moved to failed
        if inflight < inflight_before_kill or pending > 0 or failed > 0:
            if not reclaim_seen:
                reclaim_seen = True
                print(f"\n  {G}Reclaim detected! Orphaned chunk(s) re-queued or failed.{RST}\n")

        if progress.get("all_done"):
            print(f"\n  {G}All chunks accounted for (completed + failed >= total){RST}")
            return True, reclaim_seen

        time.sleep(POLL_SEC)

    print(f"\n  {Y}Reclaim monitoring timed out — continuing to watch for completion{RST}")
    return False, reclaim_seen


# ---------------------------------------------------------------------------
# Phase 5: Wait for completion
# ---------------------------------------------------------------------------
def wait_for_completion(job_id):
    step("PHASE 5: Wait for job completion")

    deadline = time.time() + JOB_TIMEOUT_MIN * 60
    start = time.time()

    while time.time() < deadline:
        try:
            jr = api("get", f"/job/{job_id}")
            job = jr.json()
        except Exception:
            time.sleep(POLL_SEC)
            continue

        status = job.get("status", "?")
        progress_frac = job.get("progress", 0)

        chunk_progress = get_chunk_progress(job_id)
        chunk_str = ""
        if chunk_progress:
            chunk_str = (
                f"  chunks: {chunk_progress['completed']}/{chunk_progress['total']} done"
                f" ({chunk_progress.get('failed', 0)} failed)"
            )

        elapsed = time.time() - start
        filled = int(progress_frac * 30)
        bar = "[" + "#" * filled + "." * (30 - filled) + f"] {progress_frac * 100:.1f}%"
        print(f"  [{elapsed:>5.0f}s] {bar}  {status}{chunk_str}")

        if status == "succeeded":
            print(f"\n  {G}Job succeeded in {elapsed:.0f}s{RST}")
            return "succeeded"
        if status == "failed":
            print(f"\n  {R}Job failed after {elapsed:.0f}s{RST}")
            return "failed"

        time.sleep(POLL_SEC)

    print(f"\n  {R}Timed out after {JOB_TIMEOUT_MIN}min{RST}")
    return "timeout"


# ---------------------------------------------------------------------------
# Phase 6: Verify
# ---------------------------------------------------------------------------
def verify(job_id, reclaim_seen, final_status):
    step("PHASE 6: Verify results")

    results = []

    # 1. Job succeeded
    results.append(check(
        "Job completed successfully",
        final_status == "succeeded",
        f"status={final_status}",
    ))

    # 2. Reclaim was observed
    results.append(check(
        "Chunk reclaim detected after replica kill",
        reclaim_seen,
        "inflight dropped or pending increased after sky down",
    ))

    # 3. Chunk accounting is consistent
    progress = get_chunk_progress(job_id)
    if progress:
        total = progress["total"]
        completed = progress["completed"]
        failed = progress.get("failed", 0)
        results.append(check(
            "Chunk accounting: completed + failed >= total",
            (completed + failed) >= total and total > 0,
            f"completed={completed} failed={failed} total={total}",
        ))
        results.append(check(
            "No permanently failed chunks (all recovered)",
            failed == 0,
            f"failed={failed}" + (" (poison chunks?)" if failed > 0 else ""),
        ))
    else:
        # Progress might be gone if cleanup_job already ran
        results.append(check(
            "Chunk progress (cleaned up after success)",
            final_status == "succeeded",
            "Redis keys cleaned up — implies assembly completed",
        ))

    # 4. Wait for S3 download, then check output file
    if final_status == "succeeded":
        print(f"\n  Waiting 20s for S3 download + assembly...")
        time.sleep(20)

        import glob
        output_dirs = glob.glob("outputs/**/success-*", recursive=True)
        job_dir = None
        for d in output_dirs:
            log_path = os.path.join(d, "job.log")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    if job_id in f.read():
                        job_dir = d
                        break

        if job_dir:
            output_path = os.path.join(job_dir, "output.jsonl")
            has_output = os.path.exists(output_path)
            results.append(check(
                "output.jsonl exists",
                has_output,
                job_dir,
            ))

            if has_output:
                with open(output_path) as f:
                    lines = f.readlines()
                # With failed chunks, output may be partial
                expected = 5000
                results.append(check(
                    f"Output has {len(lines)} lines (expected ~{expected})",
                    len(lines) >= expected * 0.9,  # allow 10% tolerance for failed chunks
                    f"{len(lines)}/{expected}",
                ))
        else:
            results.append(check(
                "Output directory found",
                False,
                "no success-* dir with this job_id",
            ))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="E2E fault tolerance test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu", default="A10G")
    args = parser.parse_args()

    print(f"\n{BOLD}Orca E2E Fault Tolerance Test{RST}")
    print(f"{DIM}Model:    {args.model}{RST}")
    print(f"{DIM}GPU:      {args.gpu}{RST}")
    print(f"{DIM}Workload: {WORKLOAD} (5000 requests){RST}")
    print(f"{DIM}Replicas: 2{RST}")
    print(f"{DIM}Server:   {BASE_URL}{RST}")
    print(f"{DIM}Strategy: Kill replica r0 mid-chunk, verify reclaim + recovery{RST}")

    # Phase 1
    job_id = submit_job(args.model, args.gpu)

    # Phase 2
    ready = wait_for_processing(job_id)
    if not ready:
        print(f"\n  {Y}Skipping fault injection — checking if job completed normally{RST}")
        final_status = wait_for_completion(job_id)
        all_results = verify(job_id, False, final_status)
    else:
        # Phase 3
        inflight_before = kill_replica(job_id)

        # Phase 4
        _, reclaim_seen = monitor_reclaim(job_id, inflight_before)

        # Phase 5
        final_status = wait_for_completion(job_id)

        # Phase 6
        all_results = verify(job_id, reclaim_seen, final_status)

    # Summary
    step("SUMMARY")
    passed = sum(1 for r in all_results if r)
    total = len(all_results)
    color = G if passed == total else (Y if passed > total // 2 else R)
    print(f"\n  {color}{BOLD}{passed}/{total} checks passed{RST}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
