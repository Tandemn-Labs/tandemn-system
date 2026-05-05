# Plan: Spot Instance Support with On-Demand Fallback

## Context

GPU benchmarks on A100/H100 are expensive ($30-98/hr). Spot instances are 60-90% cheaper but can be preempted. We want: try spot first, if it fails (no capacity or mid-run preemption), automatically fall back to on-demand. Fully backward compatible — default behavior unchanged.

**Approach:** Keep using `sky launch` (not `sky jobs launch`) for simplicity. Same SCP-based result fetching. The only change is a retry loop: spot first → on-demand fallback.

---

## Files to modify

| File | Changes |
|------|---------|
| `roofline/vllm/automatic_launch_1.py` | `generate_yaml()` accepts `use_spot` param; `run_cluster_benchmarks()` gets spot→on-demand retry logic; generated benchmark script uploads results to S3 as backup |
| `roofline/vllm/automatic_benchmark_1.py` | Add `--spot` flag, pass through to `run_cluster_benchmarks()` |

---

## Change 1: `generate_yaml()` — accept `use_spot` parameter

**Where:** `generate_yaml()` function signature (line ~380) and the YAML template (line ~498)

**What:** Add `use_spot=False` parameter. Replace hardcoded `use_spot: false` with the parameter value.

```python
def generate_yaml(gpus_per_node, num_nodes, cluster_name, experiments, gpu_type=DEFAULT_GPU_TYPE, s3_models=False, use_spot=False):
```

In the YAML template:
```yaml
  use_spot: {str(use_spot).lower()}
```

No other YAML changes needed. `sky launch --retry-until-up` already handles spot capacity retries.

---

## Change 2: `run_cluster_benchmarks()` — spot-first with on-demand fallback

**Where:** `run_cluster_benchmarks()` (line ~2570)

**What:** Add `use_spot=False` parameter. When `use_spot=True`, implement a two-attempt strategy:

```
Attempt 1 (spot):
  - Generate YAML with use_spot=true
  - sky launch WITHOUT --retry-until-up (fail fast if no spot capacity)
  - If succeeds → SCP results → done
  - If fails → go to Attempt 2

Attempt 2 (on-demand fallback):
  - Log: "⚠️ Spot failed, falling back to on-demand"
  - Regenerate YAML with use_spot=false
  - sky launch WITH --retry-until-up (standard on-demand behavior)
  - SCP results → done
```

**Implementation:** Wrap the existing try block in a loop over `[(True, False), (False, True)]` when `use_spot=True`:

```python
def run_cluster_benchmarks(cluster_config, experiments, parent_dir=None, dry_run=True,
                           gpu_type=DEFAULT_GPU_TYPE, s3_models=False, use_spot=False):
    ...
    # Determine launch attempts
    if use_spot:
        attempts = [
            {'use_spot': True, 'retry_until_up': False, 'label': 'spot'},
            {'use_spot': False, 'retry_until_up': True, 'label': 'on-demand (fallback)'},
        ]
    else:
        attempts = [
            {'use_spot': False, 'retry_until_up': True, 'label': 'on-demand'},
        ]

    for attempt_idx, attempt in enumerate(attempts):
        try:
            # Generate YAML with current spot setting
            yaml_content = generate_yaml(..., use_spot=attempt['use_spot'])
            yaml_path.write_text(yaml_content)

            # Build launch command
            cmd = ["sky", "launch", "-y", "-c", cluster_name, str(yaml_path)]
            if attempt['retry_until_up']:
                cmd.insert(3, "--retry-until-up")

            logger.info(f"🚀 Attempt {attempt_idx+1}: Launching {cluster_name} ({attempt['label']})...")

            # ... existing Popen + stream logic ...

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "sky launch")

            # Success — fetch results via SCP (existing logic)
            ...
            break  # Don't try next attempt

        except Exception as e:
            # Teardown failed spot cluster before retrying
            logger.info(f"🗑️  Tearing down failed {attempt['label']} cluster...")
            subprocess.run(["sky", "down", "-y", cluster_name], check=False)

            if attempt_idx < len(attempts) - 1:
                logger.warning(f"⚠️  {attempt['label'].capitalize()} failed: {e}")
                logger.info(f"🔄 Retrying with {attempts[attempt_idx+1]['label']}...")
                continue  # Try next attempt
            else:
                # Last attempt failed — use existing error handling
                ... (existing error handler code)
```

**Key details:**
- Spot attempt: no `--retry-until-up` — if spot isn't available, fail fast instead of waiting forever
- On-demand attempt: with `--retry-until-up` — standard behavior, waits for capacity
- Between attempts: `sky down -y` to clean up the failed spot cluster before retrying
- Cluster name stays the same across attempts (SkyPilot reuses the name)
- If `use_spot=False` (default): single attempt, exact current behavior

---

## Change 3: S3 result backup from benchmark script (belt-and-suspenders)

**Where:** Inside the f-string template, at the end of `main()` (line ~2558)

**What:** After saving results to `/tmp/benchmark_results.json`, also upload to S3. This protects against the edge case where the benchmark completes but the instance dies before SCP.

```python
# At end of main() in the generated script:
# Upload results to S3 as backup (survives spot preemption)
try:
    import subprocess as _sp
    s3_result_path = f"s3://{S3_RESULTS_BUCKET}/benchmark_results/{CLUSTER_NAME}/benchmark_results.json"
    _sp.run(["aws", "s3", "cp", RESULTS_FILE, s3_result_path], check=True)
    print(f"📤 Results backed up to {s3_result_path}")
    # Also upload timeseries
    _sp.run(f"aws s3 cp /tmp/timeseries_*.json s3://{S3_RESULTS_BUCKET}/benchmark_results/{CLUSTER_NAME}/",
            shell=True, check=False)
except Exception as e:
    print(f"⚠️  S3 backup failed (non-fatal): {e}")
```

New constants injected into the template:
```python
S3_RESULTS_BUCKET = "{s3_results_bucket}"  # Same as DEFAULT_S3_BUCKET
CLUSTER_NAME = "{cluster_name}"
```

**In the launcher's result-fetching logic**, add an S3 fallback:
```python
# Try SCP first (fast, works for on-demand)
try:
    subprocess.run(["scp", f"{cluster_name}:/tmp/benchmark_results.json", str(local_results)], check=True)
except:
    # SCP failed (spot preempted?) — try S3 backup
    logger.info("📥 SCP failed, trying S3 backup...")
    s3_path = f"s3://{DEFAULT_S3_BUCKET}/benchmark_results/{cluster_name}/benchmark_results.json"
    subprocess.run(["aws", "s3", "cp", s3_path, str(local_results)], check=True)
```

---

## Change 4: `automatic_benchmark_1.py` — add `--spot` flag

**Where:** `main()` argument parser

```python
parser.add_argument(
    "--spot", action="store_true",
    help="Try spot instances first, fall back to on-demand if unavailable or preempted",
)
```

Pass through to `run_cluster_benchmarks(..., use_spot=args.spot)`.

Update the dry-run plan renderer to show spot vs on-demand pricing.

---

## Change 5: Record spot/on-demand in results

**Where:** `benchmark_config` dict in the generated script + result dict in launcher

Add to `benchmark_config`:
```python
'instance_spot': attempt['use_spot'],  # True if running on spot
'instance_spot_fallback': attempt_idx > 0,  # True if this was a fallback from spot
```

This way the CSV records whether each experiment ran on spot or on-demand.

---

## What happens in each failure scenario

| Scenario | What happens |
|----------|-------------|
| Spot available, benchmark succeeds | Spot runs, SCP results, teardown. Same as on-demand but cheaper. |
| No spot capacity | `sky launch` fails fast (no --retry-until-up). Teardown. Retry with on-demand. |
| Spot preempted during setup/install | `sky launch` fails. Teardown. Retry with on-demand. |
| Spot preempted during benchmark | `sky launch` returns error. SCP fails. Try S3 backup. If no results, teardown. Retry with on-demand. |
| Spot preempted after benchmark completes | SCP might fail but S3 backup has results. Fetch from S3. No retry needed. |
| On-demand fails | Existing error handling — records as `cluster_error`. |
| `--spot` not passed | Exact current behavior. No changes. |

---

## Backward compatibility

- Default `use_spot=False` everywhere — no behavior change unless `--spot` is explicitly passed
- `generate_yaml()` signature adds `use_spot=False` default — all existing callers unaffected
- `run_cluster_benchmarks()` signature adds `use_spot=False` default — all existing callers unaffected
- S3 backup upload is a no-op addition to the generated script (try/except, non-fatal)
- The `automatic_launch_1.py` CLI (`main()`) doesn't get a `--spot` flag — only the new `automatic_benchmark_1.py` wrapper exposes it

---

## Verification

1. `python -m py_compile` on both files
2. Dry-run: `python automatic_benchmark_1.py --gpu L40S --spot` — verify plan shows "spot (with on-demand fallback)"
3. Test spot launch: `python automatic_benchmark_1.py --gpu A10G --spot --run --models Qwen/Qwen3-32B --tp-pp 4,2 --io 128,128`
4. Test fallback: temporarily use a GPU type with no spot capacity, verify it falls back to on-demand
5. Test without `--spot`: verify exact same behavior as before
