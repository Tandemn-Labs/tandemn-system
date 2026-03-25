# Roadmap

What's coming next for Orca.

---

## Shipped

### Chunked Distributed Batch Inference
- CLI-side chunking into Redis queue, N replicas pull chunks independently
- Rate-limited injection via KV cache `max_concurrency`
- Lease-based fault tolerance: chunk leasing with TTL, auto-reclaim, max retries
- Output assembly: per-chunk S3 outputs combined into ordered `output.jsonl`
- ReplicaWatchdog: heartbeat-based dead replica detection (45s) + force-reclaim

### Hot-Swap Replicas (`orca swap`)
- Change GPU/TP/PP mid-job without losing progress
- New replicas join the same Redis queue, old replicas torn down after threshold met
- E2E tested: A10G → L40S swap, 5000/5000 requests, zero data loss

### Live Observability
- Per-token LiveTokenCounter (SSE-based, smooth throughput vs bursty Prometheus counters)
- 10-second ring buffer window for instantaneous throughput
- GPU SM/MemBW utilization from pynvml
- Per-replica metrics with `--replica` and `--compare` flags
- Full metrics port from server runner to chunked runner (build_metrics, histograms, cost)

---

## Next Up

### Orca Scale
- `orca scale <job_id> --add 3 --gpu L40S --tp 4` — add replicas without killing old ones
- `orca scale <job_id> --remove r0 r1` — force-reclaim + tear down specific replicas
- Heterogeneous pools: A10G + H100 replicas on the same queue simultaneously
- Foundation already built (force_reclaim, same-queue architecture)

### Runner Dedup
- Extract ~800 shared lines from `vllm_batch_runner_chunked.py` and `vllm_batch_runner_server.py` into `runner_common.py`
- GPUMonitor, MetricsPoller, LiveTokenCounter, build_metrics, send_one, GPU_SPECS

### Test Coverage
- Unit tests for `_assemble_output` (S3 download + combine + aggregate)
- Unit tests for `aggregate_replica_summaries` field-specific aggregation
- Unit tests for `build_metrics` counter delta computation
- Integration tests for swap endpoint + `_swap_monitor`

### Reliability
- Graceful SIGTERM handling in runner (finish current chunk, upload, then exit)
- `_post_summary` retry logic (currently single attempt, metrics lost on failure)
- Dynamic `max_concurrency` from real output lengths after first chunk (Approach C)
- `orca swap --cancel` command

---

## Solver Expansion

- **More GPU types.** Add profiling baselines for H200, B200, MI300X, L4, and A10G to the roofline placement solver. Each new GPU type requires memory modeling coefficients and a performance database entry.
- **Multi-region cost optimization.** Factor in cross-region spot pricing variance — same GPU can be 40% cheaper in a different AZ. The solver should search across regions, not just instance types.
- **Workload-aware placement.** Use input token length distribution (not just averages) to estimate prefill vs. decode balance and pick configs that match. Long-context workloads need different TP/PP than short-prompt workloads.

---

## Multi-Cloud and On-Prem

- **GCP and Azure.** SkyPilot already supports them. Wire up quota tracking, instance type mappings, and region selection for GCP (TPU v4/v5e, A100, L4) and Azure (ND A100, NC H100).
- **On-prem Kubernetes.** For teams with their own GPU clusters. Deploy vLLM pods via Kubernetes instead of SkyPilot, using the same solver for TP/PP sizing and the same metrics pipeline. Requires a K8s job controller and a way to discover available GPU resources.
- **Hybrid scheduling.** Use on-prem GPUs for baseline capacity, burst to cloud spot for overflow. The solver should factor in on-prem availability before considering cloud.

---

## Online Inference with Dynamo

- **Dynamo integration.** Use NVIDIA Dynamo as the serving runtime for online (non-batch) inference workloads. Dynamo handles request routing, KV cache transfer across nodes, and disaggregated prefill/decode.
- **Orca as the deployment layer.** Orca's solver picks the cluster size and config, SkyPilot provisions it, Dynamo runs on it. The control plane manages autoscaling signals based on live queue depth and latency SLOs.

---

## Koi Integration

- **Koi (coming soon).** Integration with Tandemn's Koi platform for unified job management, billing, and multi-tenant access control.

---

## Observability and UX

- **`orca download <job_id>`** — Download results directly from the CLI.
- **Cost tracking across swap.** Track GPU hours and cost from all replicas (including swapped-out ones) for accurate total job cost.
- **Cost tracking dashboard.** Surface cumulative spend per model, per GPU type, and per time period in the CLI and API.
- **Alerting.** Webhook or Slack notification when a job completes, fails, or exceeds its SLO deadline.

---

## Engine and Performance

- **Prefix caching optimization.** For workloads with shared system prompts, measure and report prefix cache hit rates. Auto-enable prefix caching when the solver detects high prompt overlap.
- **Speculative decoding.** Support draft model configuration in the solver. Estimate the speedup from spec decode and include it in placement cost calculations.
- **Multi-LoRA batch inference.** Support batches that target different LoRA adapters on the same base model. The runner groups requests by adapter and manages LoRA loading.

---

## Infrastructure

- **Job queue.** Today each `orca deploy` launches immediately. Add a server-side job queue with priority scheduling, so multiple users can submit jobs and the control plane sequences them based on quota availability and priority.
- **Cluster reuse.** Instead of launching a fresh cluster per job, reuse warm clusters for back-to-back jobs with the same model. Skip the 5-10 minute provisioning and model loading time.
- **Replica self-teardown after chunk completion.** When a replica finishes all its chunks from the Redis queue, it should shut itself down within 1 minute instead of sitting idle. SkyPilot's `autodown` doesn't fire because the sky job stays "active" after the runner exits. Fix: runner should call `sky down` on itself or signal the control plane to tear it down immediately.
