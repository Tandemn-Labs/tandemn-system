# Roadmap

What's coming next for Orca.

---

## Chunked Distributed Batch Inference

The big one. Today, a single vLLM replica processes the entire input file sequentially. The next version splits workloads into chunks and distributes them across multiple spot replicas with fault tolerance.

### Chunking and Distribution

- **CLI-side chunking.** The CLI detects large input files, splits them into chunks (1000 lines each), uploads each chunk to S3, and populates a Redis queue on the control plane with chunk IDs.
- **Replica pull model.** Each vLLM replica pulls its next chunk from the Redis queue. Two chunks are present on a replica at any time: one actively running, one in a prefetch buffer. The gap between chunk transitions should be near-zero.
- **Rate-limited injection.** When a chunk arrives at a replica, it does not blast all requests into vLLM at once. Instead, it reads `max_num_seqs` from the vLLM server config and sustains exactly that many concurrent requests. At any time `t`, the vLLM server should have `max_num_seqs` in-flight requests (currently hardcoded to 256) as long as work remains.

### Fault Tolerance

- **Chunk leasing.** Each chunk is leased to a replica with a TTL. If the replica finishes, it marks the chunk complete. If the replica is preempted or crashes, the lease expires and another replica picks it up. No work is lost.
- **Replica snapshotting.** The control plane maintains a live map of which replica is running which chunk, with health checks. This is the foundation for production-grade spot preemption recovery.

### Output Assembly

- vLLM writes responses directly to S3 (per-chunk output files).
- Once all chunks are complete, a lightweight CPU instance is launched via SkyPilot to combine per-chunk outputs into a single ordered output file, then self-terminates.
- Alternative: the control plane itself assembles outputs if the dataset is small enough to avoid launching a combiner.

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

- **`orca download <job_id>`** — Download results directly from the CLI (fetches from S3 via presigned URL or streaming download). No need to open the AWS console or write `aws s3 cp` commands.
- **Failure detection.** If the vLLM server crashes or OOMs, detect it within seconds and update job status to `failed` with a reason (currently relies on the runner reporting back; should also detect via health check timeout).
- **Cost tracking dashboard.** The analytics DB already stores cost-per-run. Surface cumulative spend per model, per GPU type, and per time period in the CLI and API.
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
