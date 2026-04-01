# Orca
<div align="center">
<img src="data/orca.png" width="400" alt="Orca">
</div>

Batch inference on large models means choosing the right GPU, figuring out tensor and pipeline parallelism, picking a region with available quota, and hoping it all fits in memory before your deadline. Most teams just guess.

Orca handles all of that. Give it a model name, a JSONL file, and a deadline. The placement solver sizes the job automatically; picking the instance type, parallelism configuration, and AWS region, then launches on spot via SkyPilot. While the job runs, Orca streams real-time throughput, latency, and scheduler metrics back to your terminal. Output lands in S3 when it's done.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- AWS credentials configured (`aws configure` or `~/.aws/credentials`) — **currently supports AWS only**
- Redis (for multi-replica chunked jobs): `docker run -d -p 6379:6379 redis`

---

## Quick Start

```bash
git clone --recurse-submodules https://github.com/Tandemn-Labs/Tandemn-orca.git
cd Tandemn-orca
bash setup.sh
```

The setup script installs dependencies, checks AWS/Redis, and creates your `.env`. Or do it manually:

<details>
<summary>Manual installation</summary>

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
sky check
```

Create a `.env` file:
```bash
S3_UPLOAD_BUCKET=your-s3-bucket       # must exist in your AWS account
HF_TOKEN=hf_your_token_here          # for gated models (Llama, Gemma, etc.)
```
</details>

## Supported Models

Orca works with **any HuggingFace model that vLLM supports**. You can deploy any model by specifying `--gpu` and `--tp` manually:

```bash
./orca deploy <any-hf-model> input.jsonl --gpu A10G --tp 1
```

The **automatic placement solver** (picks GPU/TP/PP for you based on your SLO) has profiling data for:

| Model | Parameters | Profiled GPUs |
|-------|-----------|---------------|
| Llama 3.3 70B (FP8) | 70B | H100 |
| Llama 3.1 8B (FP8) | 8B | H100 |
| DeepSeek-R1-Distill-Llama-70B | 70B | L40S, A100 |
| Llama 2 70B | 70B | A100 |
| Llama 3 70B Instruct | 70B | A100 |

For models not in this list, use `--gpu` and `--tp` to specify your config, or use `./orca plan` to see what the solver recommends.

---

## Quick Start

Start the control plane with a Cloudflare tunnel (so EC2 replicas can reach your laptop):

```bash
python server.py --tunnel
```

This starts the server on `http://localhost:26336` and prints a public tunnel URL. The tunnel URL is automatically set as `ORCA_SERVER_URL`.

Run a batch job:

```bash
./orca deploy Qwen/Qwen2.5-7B-Instruct examples/workloads/demo_batch.jsonl --slo 4
```

Orca parses the input file, runs the placement solver, and launches on the cheapest viable spot configuration. No GPU selection required.

For multi-replica with chunking:

```bash
./orca deploy Qwen/Qwen2.5-7B-Instruct input.jsonl --gpu A10G --tp 1 --replicas 2 --chunk-size 100
```

Track progress:

```bash
./orca progress
./orca metrics <job_id> --watch
```

---

## CLI

### Deployment

```
./orca deploy <model> <input>     Run a batch job (solver picks GPU automatically)
./orca plan   <model> <input>     Show placement plan without launching
```

Options for `deploy` and `plan`:

```
--slo <hours>           Deadline: plain hours (4), fractional (0.5h), or minutes (30m) (default: 4)
--max-output-tokens N   Max tokens per response (default: 1024)
--gpu <type>            Override GPU type (e.g. A100, L40S, H100)
--tp / --pp             Override tensor / pipeline parallelism
--replicas N            Number of replica clusters (default: 1)
--chunk-size N          Lines per chunk (default: 1000)
--force                 Skip feasibility check and launch anyway
--persist               Keep cluster alive after job completes
--on-demand             Use on-demand instances instead of spot
```

### Scale and Kill Replicas

Add or remove replicas from a running job:

```
./orca add <job_id> 2                        # Add 2 replicas (inherit GPU config)
./orca add <job_id> 3 --gpu L40S --tp 4      # Add 3 L40S replicas (heterogeneous fleet)
./orca kill <job_id> --replica <rid>          # Kill a specific replica
./orca kill <job_id> --replica r0 --replica r1  # Kill multiple replicas
```

New replicas join the same Redis chunk queue. Killed replicas' inflight chunks are reclaimed and returned to pending.

### Hot-Swap Replicas

Replace all replicas with a new GPU config mid-job (atomic: waits for readiness before killing old):

```
./orca swap <job_id> --gpu A100 --tp 4 --replicas 2
./orca swap <job_id> --gpu L40S --tp 1 --ready-threshold 2 --on-demand
```

Swap composes `add` + `kill` — new replicas launch first, old ones are torn down after the new ones start inferring.

### Monitoring

```
./orca web                        Open real-time web dashboard in browser
./orca progress [job_id]          Live progress bar with throughput and queue depth
./orca status                     List all jobs
./orca metrics <job_id> [-w]      Latest vLLM metrics snapshot (--watch for 2s refresh)
./orca metrics <job_id> --replica <rid>   Per-replica metrics
./orca metrics <job_id> --compare         Aggregated + per-replica side by side
./orca stream <job_id>            Stream live metrics table (1 event/sec via SSE)
./orca logs [cluster]             Stream logs from a SkyPilot cluster
./orca clusters                   Show active clusters
```

The web dashboard (`orca web`) shows a real-time single-page UI with:
- **Workload panel** — model, prompts, status, chunk progress
- **Chain visualization** — SVG replica nodes with phase colors and animated data flow
- **Cost bar** — accrued cost, projected total, ETA, throughput (from SkyPilot pricing)
- **Quota sidebar** — live AWS GPU quota utilization per region/family (auto-discovered on startup)
- **Event log** — synthetic events for job status changes, chunk milestones, replica phase transitions
- **Charts** (toggle) — throughput, KV cache, scheduler, GPU utilization, latency, completions with linked crosshairs

The dashboard uses SSE for real-time updates, with automatic polling fallback for environments that buffer SSE (e.g., Cloudflare tunnels). Panels are resizable via tmux-style drag splitters.

### Operations

```
./orca destroy <job_id>           Tear down clusters + Redis state for a job
./orca destroy --all              Tear down ALL Orca clusters
```

### Analytics

```
./orca history [--model X] [--gpu Y]   Browse completed runs
./orca inspect <run_id>                Full run report (latency, throughput, cost, GPU util)
./orca inspect <run_id> --replicas     Per-replica summaries side by side
./orca timeseries <run_id>             Scheduler timeseries for a completed run
```

---

## How It Works

```
         ./orca deploy Qwen/Qwen2.5-72B batch.jsonl --slo 4 --replicas 2

                              +---------------------+
                              |   Control Plane      |
                              |   (server.py)        |
                              +---------------------+
                              |  1. Parse input      |
                              |  2. Roofline solver  |
                              |  3. Quota check      |
                              |  4. Chunk + Redis    |
                              |  5. SkyPilot launch  |
                              +----------+----------+
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
                    v                    v                    v
             +----------+        +----------+        +----------+
             | Replica 0|        | Replica 1|  ...   | Replica N|
             | vLLM V1  |        | vLLM V1  |        | vLLM V1  |
             +----+-----+        +----+-----+        +----+-----+
                  |                    |                    |
                  +------+  Redis  +--+--------------------+
                         |  Queue  |
                         +---------+
                  pull → process → upload → complete
                              |
                              v
                   S3 (per-chunk outputs → assembled output.jsonl + metrics.csv)
```

**Placement solver.** Uses a roofline model to estimate throughput and memory requirements across GPU types and TP/PP configurations. Picks the cheapest option that completes within your SLO. Automatically falls back to alternative regions and instance types if the primary launch fails.

**Chunked multi-replica.** Input is split into chunks (default 1000 lines each) and queued in Redis. N independent replicas pull chunks, process them via vLLM, and upload outputs to S3. Lease-based fault tolerance: if a replica dies, its inflight chunks are reclaimed and re-queued. A ReplicaWatchdog detects dead replicas via heartbeat within 45 seconds.

**Hot-swap.** `orca swap` launches new replicas with a different GPU/TP/PP config on the same Redis queue. Once the new replicas start processing, old replicas are torn down. Zero chunks lost in the transition.

**Quota tracking.** Real-time quota usage across AWS regions. Orca won't try to launch where you have no capacity.

**Observability.** Each replica's sidecar pushes Prometheus snapshots + GPU utilization + per-token live counters to the control plane every 5 seconds. Orca computes throughput from a fixed 10-second ring buffer window, extracts histogram quantiles (TTFT, TPOT, E2E, queue/prefill/decode/inference), and tracks KV cache utilization and scheduler state. All metrics are persisted to SQLite for post-run analysis.

**Teardown.** Clusters are destroyed by default after job completion. Use `--persist` to keep them alive. `orca destroy --all` cleans up clusters, Redis state, and S3 uploads.

---

## Supported Hardware

| GPU | AWS Instance | VRAM |
|-----|-------------|------|
| A100 80GB | p4d.24xlarge, p4de.24xlarge | 8x 80GB |
| H100 80GB | p5.48xlarge | 8x 80GB |
| L40S 48GB | g6e.12xlarge / 24xlarge / 48xlarge | 4x / 4x / 8x 48GB |
| A10G 24GB | g5.12xlarge / 48xlarge | 4x / 8x 24GB |

The solver searches across all GPU types and parallelism configs (TP 1-8, PP 1-4) to find the cheapest placement that fits your model and meets the deadline.

---

## Input Format

Standard OpenAI batch JSONL:

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "placeholder", "messages": [{"role": "user", "content": "Your prompt here"}], "max_tokens": 256}}
```

Local files are uploaded to S3 automatically. S3 URIs (`s3://...`) are passed through directly.

Sample workloads are in `examples/workloads/`:

```bash
examples/workloads/demo_batch.jsonl                    # 30 requests, quick test
examples/workloads/sharegpt-numreq_200-*.jsonl         # 200 ShareGPT conversations
examples/workloads/stress_5000.jsonl                   # 5000 requests, stress test
```

Generate larger workloads:
```bash
python examples/workloads/make_long_workload.py examples/workloads/sharegpt-numreq_200-avginputlen_956-avgoutputlen_50.jsonl 25 /tmp/stress_5k.jsonl
```

---

## Running Locally

EC2 replicas need to reach your control plane for metrics, chunk pulling, etc. `python server.py --tunnel` handles this automatically using a free Cloudflare tunnel (no account required).

If you prefer manual tunnel setup or a persistent URL:

```bash
# Option A: Manual Cloudflare tunnel
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:26336

# Then set in .env:
ORCA_SERVER_URL=https://random-words.trycloudflare.com
```

```bash
# Option B: Persistent setup
# Use Tailscale (mesh VPN, stable IPs) or deploy on a small EC2 in the same VPC
```

Optional security for public tunnels:
```bash
# .env
ORCA_API_KEY=some-secret-key   # all replica↔server communication will require this token
```

---

## API

The control plane exposes a REST API at `http://localhost:26336`:

| Endpoint | Description |
|----------|-------------|
| **Jobs** | |
| `POST /submit/batch` | Submit batch inference job |
| `POST /test/placement` | Run solver only (no launch) |
| `GET /jobs` | List all jobs |
| `GET /job/{id}` | Job status and progress |
| `POST /job/{id}/phase` | Update job lifecycle phase |
| **Metrics** | |
| `GET /job/{id}/metrics` | Latest metrics snapshot |
| `GET /job/{id}/metrics/stream` | SSE metrics stream |
| `POST /job/{id}/metrics/ingest` | Sidecar metrics ingest (replica → server) |
| `POST /job/{id}/metrics/summary` | Per-replica build_metrics summary |
| `GET /job/{id}/throughput` | Sustained throughput (rolling window) |
| **Replicas** | |
| `GET /job/{id}/replicas` | Per-replica state (phase, region, metrics) |
| `GET /job/{id}/replicas/{rid}/metrics` | Metrics for a specific replica |
| `GET /job/{id}/replicas/summaries` | Per-replica completion summaries |
| `POST /job/{id}/scale` | Add replicas to a running job |
| `POST /job/{id}/kill` | Kill specific replicas |
| `POST /job/{id}/swap` | Hot-swap replicas to new GPU config |
| **Chunks** | |
| `GET /job/{id}/chunks/progress` | Chunk-level progress (pending/inflight/completed/failed) |
| `POST /job/{id}/chunks/pull` | Pull next chunk (replica-facing) |
| `POST /job/{id}/chunks/complete` | Mark chunk completed |
| `POST /job/{id}/chunks/renew` | Renew chunk lease |
| **Dashboard** | |
| `GET /dashboard` | Web dashboard (HTML) |
| `GET /dashboard/poll` | Dashboard data (JSON, polling fallback) |
| `GET /dashboard/stream` | Dashboard data (SSE, real-time) |
| **Analytics** | |
| `GET /analytics/runs` | List completed runs |
| `GET /analytics/runs/{id}` | Full run report |
| `GET /analytics/runs/{id}/timeseries` | Scheduler timeseries |
| `GET /quota/status` | Quota usage across regions |

### Monitoring Endpoint Details

<details>
<summary><strong>GET /job/{id}/metrics</strong> — Aggregated metrics snapshot</summary>

**Response:**
```json
{
  "job_id": "mo-qwen7b-a1b2",
  "timestamp": 1711612800.0,
  "avg_generation_throughput_toks_per_s": 1450.5,
  "avg_prompt_throughput_toks_per_s": 320.0,
  "gpu_cache_usage_perc": 0.42,
  "num_requests_running": 64,
  "num_requests_waiting": 0,
  "num_requests_swapped": 0,
  "request_success_total": 3500,
  "num_preemptions_total": 0,
  "generation_tokens_total": 2800000,
  "prompt_tokens_total": 350000,
  "gpu_sm_util_pct": 95.2,
  "gpu_mem_bw_util_pct": 61.0,
  "ttft_ms_p50": 45.0,
  "ttft_ms_p95": 120.0,
  "tpot_ms_p50": 8.5,
  "tpot_ms_p95": 15.0
}
```
</details>

<details>
<summary><strong>GET /job/{id}/replicas</strong> — Per-replica state</summary>

**Response:**
```json
{
  "replicas": [
    {
      "replica_id": "mo-qwen7b-a1b2-r0",
      "phase": "running",
      "region": "us-east-2",
      "market": "spot",
      "instance_type": "g5.xlarge",
      "has_metrics": true
    }
  ]
}
```
Phases: `launching` → `running` → `completed` | `failed` | `killed` | `swapped_out`
</details>

<details>
<summary><strong>GET /job/{id}/chunks/progress</strong> — Chunk-level progress</summary>

**Response:**
```json
{
  "total": 10,
  "pending": 3,
  "inflight": 2,
  "completed": 5,
  "failed": 0,
  "all_done": false
}
```
</details>

<details>
<summary><strong>POST /job/{id}/scale</strong> — Add replicas to a running job</summary>

**Request:**
```json
{
  "count": 2,
  "gpu_type": "L40S",
  "tp_size": 4,
  "pp_size": 1,
  "on_demand": false,
  "force": false
}
```
`gpu_type`, `tp_size`, `pp_size` optional — inherited from existing job if omitted.

**Response:**
```json
{
  "status": "scaling",
  "new_replicas": ["mo-qwen7b-a1b2-v2-r0", "mo-qwen7b-a1b2-v2-r1"],
  "version": 2
}
```
</details>

<details>
<summary><strong>POST /job/{id}/kill</strong> — Kill specific replicas</summary>

**Request:**
```json
{
  "replica_ids": ["mo-qwen7b-a1b2-r0"]
}
```

**Response:**
```json
{
  "status": "killing",
  "killed": ["mo-qwen7b-a1b2-r0"],
  "skipped": [],
  "reclaimed": 3
}
```
Inflight chunks reclaimed to pending queue. Clusters tear down in background.
</details>

<details>
<summary><strong>POST /job/{id}/swap</strong> — Hot-swap replicas</summary>

**Request:**
```json
{
  "gpu_type": "H100",
  "tp_size": 4,
  "num_replicas": 2,
  "ready_threshold": 1,
  "force": false
}
```

**Response:**
```json
{
  "status": "swapping",
  "old_replicas": ["mo-qwen7b-a1b2-r0"],
  "new_replicas": ["mo-qwen7b-a1b2-v2-r0", "mo-qwen7b-a1b2-v2-r1"],
  "ready_threshold": 1,
  "version": 2
}
```
New replicas launch first. Old replicas killed after `ready_threshold` new ones start inferring.
</details>

<details>
<summary><strong>GET /dashboard/poll</strong> — Full dashboard payload</summary>

**Response:**
```json
{
  "jobs": [{ "job_id": "...", "status": "...", "model_name": "...", "progress": 0.5, ... }],
  "metrics": { "job_id": { "avg_generation_throughput_toks_per_s": 1450.5, ... } },
  "chunks": { "job_id": { "total": 10, "completed": 5, "inflight": 2, ... } },
  "replicas": { "job_id": [{ "replica_id": "...", "phase": "running", "region": "us-east-2", ... }] },
  "cost": { "job_id": { "accrued_usd": 0.15, "projected_total_usd": 0.43, "eta_sec": 2482 } },
  "events": [{ "ts": 1711612800.0, "level": "info", "message": "..." }],
  "timeseries": { "job_id": [{ "timestamp": ..., "avg_generation_throughput_toks_per_s": ... }] },
  "quota": [{ "Region": "us-east-2", "Family": "G", "Market": "spot", "Used": 4, "Baseline": 128 }]
}
```
</details>

---

## Roadmap

See [ROADMAP.md](ROADMAP.md).

---

## Contact

- Hetarth -- hetarth@tandemn.com
- Mankeerat -- mankeerat@tandemn.com

---

## License

MIT (depends on SkyPilot under Apache License 2.0)
