# Tandemn System

**Automated batch inference orchestration for large language models on AWS spot infrastructure.**

Tandemn System (currently referenced in this repository as `orca`) is a self-hosted batch inference orchestration system that takes a model name, a JSONL workload, and a deadline — and handles everything else. A placement solver selects the optimal GPU type, tensor and pipeline parallelism configuration, and AWS region based on your SLO. Jobs launch on spot instances via SkyPilot, with chunked multi-replica execution, real-time observability, and output delivered to your S3 bucket. Your data never leaves your infrastructure.

> **Naming note:** The product is called **Tandemn System**, while parts of this repository, CLI, and internal code use the codename **Orca**. In this README, **Tandemn System** refers to the platform, and **`orca`** refers to the current CLI / implementation.

> **Deployment model:** Tandemn System is fully open-source and self-hosted. You run it on your own AWS account. There is no managed tier or external data plane.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Naming](#naming)
- [Supported Models](#supported-models)
- [Supported Hardware](#supported-hardware)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Security](#security)
- [Architecture](#architecture)
- [Input Format](#input-format)
- [REST API](#rest-api)
- [Analytics](#analytics)
- [Roadmap](#roadmap)
- [Contact & Support](#contact--support)

---

## Prerequisites

Before installing, ensure the following are in place:

| Requirement | Notes |
|-------------|-------|
| Python 3.12+ | Required for the control plane |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [SkyPilot](https://skypilot.readthedocs.io/) | Installed automatically by `setup.sh`; run `sky check` to verify cloud access |
| AWS credentials | `aws configure` or `~/.aws/credentials` — **AWS only at this time** |
| AWS IAM permissions | EC2 (launch/terminate), S3 (read/write to your bucket), service quota read access |
| S3 bucket | Must exist in your AWS account; set as `S3_UPLOAD_BUCKET` in `.env` |
| Redis | Required for multi-replica chunked jobs: `docker run -d -p 6379:6379 redis` |
| HuggingFace token | Required for gated models (Llama, Gemma, etc.); set as `HF_TOKEN` in `.env` |

> **AWS IAM note:** At minimum, your credentials need `ec2:*` on the instance types you plan to use, `s3:GetObject` / `s3:PutObject` on your bucket, and `servicequotas:GetServiceQuota` for quota tracking. A least-privilege IAM policy example is available in [`docs/iam-policy.json`](docs/iam-policy.json).

---

## Installation

```bash
git clone --recurse-submodules https://github.com/Tandemn-Labs/Tandemn-orca.git
cd Tandemn-orca
bash setup.sh
```

`setup.sh` installs Python dependencies, verifies AWS and Redis connectivity, and creates your `.env` file.

<details>
<summary>Manual installation</summary>

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
sky check
```

Create a `.env` file in the project root:

```bash
S3_UPLOAD_BUCKET=your-s3-bucket       # Must exist in your AWS account
HF_TOKEN=hf_your_token_here           # Required for gated models
ORCA_API_KEY=your-secret-key          # Recommended; see Security section
KOI_SERVICE_URL=http://localhost:8090 # Optional: Koi placement engine
```

</details>

---

## Naming

This repository currently uses the legacy name `orca` in parts of the codebase, CLI, environment variables, and examples.

- **Product name:** Tandemn System
- **Current CLI:** `orca`
- **Repo internals / env vars:** may still reference `orca`

Over time, repository naming may be updated for consistency, but current commands and interfaces remain unchanged.

---

## Supported Models

Tandemn System supports **any HuggingFace model compatible with vLLM**. Specify `--gpu` and `--tp` manually to run any model:

```bash
./orca deploy <any-hf-model> input.jsonl --gpu A10G --tp 1
```

The **automatic placement solver** — which selects GPU type, TP/PP configuration, and region on your behalf — has profiling data for the following models:

| Model | Parameters | Profiled GPUs |
|-------|------------|---------------|
| Llama 3.3 70B (FP8) | 70B | H100 |
| Llama 3.1 8B (FP8) | 8B | H100 |
| DeepSeek-R1-Distill-Llama-70B | 70B | L40S, A100 |
| Llama 2 70B | 70B | A100 |
| Llama 3 70B Instruct | 70B | A100 |

For models not listed, use `./orca plan` to see solver recommendations, or override manually with `--gpu` and `--tp`.

**Koi integration.** When `KOI_SERVICE_URL` is set, `orca plan` and `orca deploy` call the [Koi](https://github.com/Tandemn-Labs/koi) LLM-powered placement engine in parallel with the roofline solver. Both recommendations are shown side by side. For deploy, a prompt lets you choose between them (`--skip-dangerously` auto-picks Koi). Falls back to roofline silently if Koi is unreachable.

---

## Supported Hardware

| GPU | AWS Instance | VRAM |
|-----|-------------|------|
| A100 80GB | p4d.24xlarge, p4de.24xlarge | 8× 80GB |
| H100 80GB | p5.48xlarge | 8× 80GB |
| L40S 48GB | g6e.12xlarge / 24xlarge / 48xlarge | 4× / 4× / 8× 48GB |
| A10G 24GB | g5.12xlarge / 48xlarge | 4× / 8× 24GB |

The solver searches across all GPU types and parallelism configurations (TP 1–8, PP 1–4) to find the cheapest placement that fits your model in memory and meets your deadline.

---

## Quick Start

### Step 1 — Start the control plane

The Tandemn System control plane must be reachable by EC2 replicas. The `--tunnel` flag opens a free Cloudflare tunnel automatically (no account required):

```bash
python server.py --tunnel
```

This starts the server at `http://localhost:26336` and prints a public tunnel URL, which is automatically set as `ORCA_SERVER_URL`.

### Step 2 — Run a batch job

```bash
./orca deploy Qwen/Qwen2.5-7B-Instruct examples/workloads/demo_batch.jsonl --slo 4
```

Tandemn System parses the input file, runs the placement solver, and launches on the cheapest viable spot configuration. No GPU selection required.

### Step 3 — Monitor progress

```bash
./orca progress          # Live progress bar with throughput and queue depth
./orca web               # Open real-time web dashboard in browser
```

### Multi-replica with chunking

```bash
./orca deploy Qwen/Qwen2.5-7B-Instruct input.jsonl --gpu A10G --tp 1 --replicas 2 --chunk-size 100
```

---

## CLI Reference

### Deployment

```bash
./orca deploy <model> <input>     Run a batch job (solver picks GPU automatically)
./orca plan   <model> <input>     Show placement plan without launching
```

**Options for `deploy` and `plan`:**

| Flag | Description | Default |
|------|-------------|---------|
| `--slo <hours>` | Deadline: plain hours (`4`), fractional (`0.5h`), or minutes (`30m`) | `4` |
| `--max-output-tokens N` | Max tokens per response | `1024` |
| `--gpu <type>` | Override GPU type (e.g. `A100`, `L40S`, `H100`) | solver |
| `--tp / --pp` | Override tensor / pipeline parallelism | solver |
| `--replicas N` | Number of replica clusters | `1` |
| `--chunk-size N` | Lines per chunk | `1000` |
| `--force` | Skip feasibility check and launch anyway | — |
| `--persist` | Keep cluster alive after job completes | — |
| `--on-demand` | Use on-demand instances instead of spot | — |

---

### Scale and Kill Replicas

Add or remove replicas from a running job:

```bash
./orca add <job_id> 2                           # Add 2 replicas (inherit GPU config)
./orca add <job_id> 3 --gpu L40S --tp 4         # Add 3 L40S replicas (heterogeneous fleet)
./orca kill <job_id> --replica <rid>            # Kill a specific replica
./orca kill <job_id> --replica r0 --replica r1  # Kill multiple replicas
```

New replicas join the same Redis chunk queue. Killed replicas' in-flight chunks are reclaimed and returned to pending.

---

### Hot-Swap Replicas

Replace all replicas with a new GPU configuration mid-job. New replicas launch first; old replicas are torn down only after the new ones begin processing:

```bash
./orca swap <job_id> --gpu A100 --tp 4 --replicas 2
./orca swap <job_id> --gpu L40S --tp 1 --ready-threshold 2 --on-demand
```

Zero chunks are lost during a swap.

---

### Monitoring

```bash
./orca web                                     # Open real-time web dashboard
./orca progress [job_id]                       # Live progress bar with throughput and queue depth
./orca status                                  # List all jobs
./orca metrics <job_id> [-w]                   # Latest vLLM metrics snapshot (--watch for 2s refresh)
./orca metrics <job_id> --replica <rid>        # Per-replica metrics
./orca metrics <job_id> --compare              # Aggregated + per-replica side by side
./orca stream <job_id>                         # Stream live metrics table (1 event/sec via SSE)
./orca logs [cluster]                          # Stream logs from a SkyPilot cluster
./orca clusters                                # Show active clusters
```

**Web dashboard** (`./orca web`) provides a real-time single-page UI including:

- **Workload panel** — model, prompts, status, chunk progress
- **Chain visualization** — SVG replica nodes with phase colors and animated data flow
- **Cost bar** — accrued cost, projected total, ETA, throughput (sourced from SkyPilot pricing)
- **Quota sidebar** — live AWS GPU quota utilization per region and instance family, auto-discovered on startup
- **Event log** — job status changes, chunk milestones, replica phase transitions
- **Charts** (toggle) — throughput, KV cache, scheduler, GPU utilization, latency, and completions with linked crosshairs

The dashboard uses SSE for real-time updates with automatic polling fallback for environments that buffer SSE (e.g., Cloudflare tunnels). Panels are resizable via drag splitters.

---

### Operations

```bash
./orca destroy <job_id>     # Tear down clusters and Redis state for a job
./orca destroy --all        # Tear down ALL `orca` clusters
```

Clusters are destroyed by default after job completion. Use `--persist` to keep them alive.

---

### Analytics

```bash
./orca history [--model X] [--gpu Y]    # Browse completed runs
./orca inspect <run_id>                 # Full run report: latency, throughput, cost, GPU utilization
./orca inspect <run_id> --replicas      # Per-replica summaries side by side
./orca timeseries <run_id>              # Scheduler timeseries for a completed run
```

---

## Security

The Tandemn System control plane accepts connections from EC2 replicas over the network. In local development with a Cloudflare tunnel, this means your endpoint is publicly reachable. **We strongly recommend setting an API key in all environments.**

### API Key Authentication

Set `ORCA_API_KEY` in your `.env`:

```bash
ORCA_API_KEY=your-secret-key
```

All replica-to-server communication will require this token. The key is distributed to replicas automatically at launch.

### Tunnel Options

| Option | Use case |
|--------|----------|
| `python server.py --tunnel` | Local development; ephemeral URL via Cloudflare (no account required) |
| Manual Cloudflare tunnel | Persistent or custom URL |
| Tailscale | Stable mesh VPN; recommended for teams |
| Same-VPC EC2 | Production; no public exposure |

**Manual Cloudflare tunnel:**

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:26336
```

Then set in `.env`:

```bash
ORCA_SERVER_URL=https://your-tunnel.trycloudflare.com
```

**Production recommendation:** Deploy the control plane on a small EC2 instance in the same VPC as your inference replicas. This avoids public exposure entirely and eliminates tunnel latency.

---

## Architecture

```bash
$ ./orca deploy Qwen/Qwen2.5-72B batch.jsonl --slo 4 --replicas 2

                                 ┌──────────────────────────────────────┐
                                 │             Control Plane            │
                                 │              server.py               │
                                 ├──────────────────────────────────────┤
                                 │ 1. Parse deployment request          │
                                 │ 2. Run roofline solver               │
                                 │ 3. Validate quota / capacity         │
                                 │ 4. Chunk input + enqueue in Redis    │
                                 │ 5. Launch replicas via SkyPilot      │
                                 └──────────────────┬───────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
          │    Replica 0     │            │    Replica 1     │            │    Replica N     │
          │     vLLM V1      │    ...     │     vLLM V1      │    ...     │     vLLM V1      │
          └────────┬─────────┘            └────────┬─────────┘            └────────┬─────────┘
                   │                               │                               │
                   └───────────────┬───────────────┴───────────────┬───────────────┘
                                   │                               │
                                   ▼                               │
                         ┌──────────────────────┐                  │
                         │      Redis Queue     │◄─────────────────┘
                         │   chunk coordination │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    pull chunk → run inference → upload result → mark complete
                                    │
                                    ▼
                ┌────────────────────────────────────────────────────────────┐
                │                          S3                                │
                │   per-chunk outputs → assembled output.jsonl + metrics.csv │
                └────────────────────────────────────────────────────────────┘
```

**Placement solver.** Uses a roofline model to estimate throughput and memory requirements across GPU types and TP/PP configurations. Selects the cheapest option that completes within your SLO, with automatic fallback to alternative regions and instance types if the primary launch fails.

**Chunked multi-replica execution.** Input is split into chunks (default: 1,000 lines each) and queued in Redis. Independent replicas pull chunks, process them via vLLM, and upload outputs to S3. Lease-based fault tolerance ensures that if a replica dies, its in-flight chunks are reclaimed and re-queued within 45 seconds via a ReplicaWatchdog heartbeat.

**Hot-swap.** `orca swap` launches replacement replicas with a different GPU/TP/PP configuration on the same Redis queue. Old replicas are torn down only after the new replicas begin processing. Zero chunks are lost.

**Quota tracking.** Real-time GPU quota usage is tracked across AWS regions. Tandemn System will not attempt to launch where you have insufficient capacity.

**Observability.** Each replica's sidecar pushes Prometheus snapshots, GPU utilization, and per-token counters to the control plane every 5 seconds. Tandemn System computes throughput from a 10-second rolling window, extracts histogram quantiles (TTFT, TPOT, E2E, queue/prefill/decode/inference latency), and tracks KV cache utilization and scheduler state. All metrics are persisted to SQLite for post-run analysis.

---

## Input Format

Standard OpenAI batch JSONL format:

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "placeholder", "messages": [{"role": "user", "content": "Your prompt here"}], "max_tokens": 256}}
```

Local files are uploaded to S3 automatically. S3 URIs (`s3://...`) are passed through directly.

**Sample workloads** are included in `examples/workloads/`:

| File | Size | Purpose |
|------|------|---------|
| `demo_batch.jsonl` | 30 requests | Quick smoke test |
| `sharegpt-numreq_200-*.jsonl` | 200 requests | Realistic ShareGPT conversations |
| `stress_5000.jsonl` | 5,000 requests | Load and stress testing |

Generate larger workloads:

```bash
python examples/workloads/make_long_workload.py \
  examples/workloads/sharegpt-numreq_200-avginputlen_956-avgoutputlen_50.jsonl \
  25 /tmp/stress_5k.jsonl
```

---

## REST API

The control plane exposes a REST API at `http://localhost:26336`.

### Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| **Jobs** | | |
| `/submit/batch` | POST | Submit a batch inference job |
| `/test/placement` | POST | Run solver only (no launch) |
| `/jobs` | GET | List all jobs |
| `/job/{id}` | GET | Job status and progress |
| `/job/{id}/phase` | POST | Update job lifecycle phase |
| **Metrics** | | |
| `/job/{id}/metrics` | GET | Latest aggregated metrics snapshot |
| `/job/{id}/metrics/stream` | GET | SSE metrics stream |
| `/job/{id}/metrics/ingest` | POST | Sidecar metrics ingest (replica → server) |
| `/job/{id}/metrics/summary` | POST | Per-replica build_metrics summary |
| `/job/{id}/throughput` | GET | Sustained throughput (rolling window) |
| **Replicas** | | |
| `/job/{id}/replicas` | GET | Per-replica state (phase, region, metrics) |
| `/job/{id}/replicas/{rid}/metrics` | GET | Metrics for a specific replica |
| `/job/{id}/replicas/summaries` | GET | Per-replica completion summaries |
| `/job/{id}/scale` | POST | Add replicas to a running job |
| `/job/{id}/kill` | POST | Kill specific replicas |
| `/job/{id}/swap` | POST | Hot-swap replicas to new GPU config |
| **Chunks** | | |
| `/job/{id}/chunks/progress` | GET | Chunk-level progress (pending/inflight/completed/failed) |
| `/job/{id}/chunks/pull` | POST | Pull next chunk (replica-facing) |
| `/job/{id}/chunks/complete` | POST | Mark chunk completed |
| `/job/{id}/chunks/renew` | POST | Renew chunk lease |
| **Dashboard** | | |
| `/dashboard` | GET | Web dashboard (HTML) |
| `/dashboard/poll` | GET | Dashboard data (JSON, polling fallback) |
| `/dashboard/stream` | GET | Dashboard data (SSE, real-time) |
| **Analytics** | | |
| `/analytics/runs` | GET | List completed runs |
| `/analytics/runs/{id}` | GET | Full run report |
| `/analytics/runs/{id}/timeseries` | GET | Scheduler timeseries |
| `/quota/status` | GET | Quota usage across regions |

---

### Endpoint Details

<details>
<summary><strong>GET /job/{id}/metrics</strong> — Aggregated metrics snapshot</summary>

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

Replica phases: `launching` → `running` → `completed` | `failed` | `killed` | `swapped_out`
</details>

<details>
<summary><strong>GET /job/{id}/chunks/progress</strong> — Chunk-level progress</summary>

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

`gpu_type`, `tp_size`, and `pp_size` are optional — inherited from the existing job if omitted.

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

In-flight chunks are reclaimed to the pending queue. Clusters tear down in the background.
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

New replicas launch first. Old replicas are killed after `ready_threshold` new replicas start inferring.
</details>

<details>
<summary><strong>GET /dashboard/poll</strong> — Full dashboard payload</summary>

```json
{
  "jobs": [{ "job_id": "...", "status": "...", "model_name": "...", "progress": 0.5 }],
  "metrics": { "job_id": { "avg_generation_throughput_toks_per_s": 1450.5 } },
  "chunks": { "job_id": { "total": 10, "completed": 5, "inflight": 2 } },
  "replicas": { "job_id": [{ "replica_id": "...", "phase": "running", "region": "us-east-2" }] },
  "cost": { "job_id": { "accrued_usd": 0.15, "projected_total_usd": 0.43, "eta_sec": 2482 } },
  "events": [{ "ts": 1711612800.0, "level": "info", "message": "..." }],
  "timeseries": { "job_id": [{ "timestamp": "...", "avg_generation_throughput_toks_per_s": "..." }] },
  "quota": [{ "Region": "us-east-2", "Family": "G", "Market": "spot", "Used": 4, "Baseline": 128 }]
}
```
</details>

<details>
<summary><strong>GET /resources</strong> — Instance catalog + quota pools for Koi</summary>

**Response:**
```json
{
  "vpc_id": "orca-cluster",
  "snapshot_time": "2026-03-31T20:02:40.791092",
  "instances": [
    {
      "instance_type": "g6e.12xlarge",
      "gpu_type": "L40S",
      "gpus_per_instance": 4,
      "vcpus": 48,
      "quota_family": "G",
      "gpu_memory_gb": 48.0,
      "interconnect": "PCIe",
      "cost_per_instance_hour_usd": 10.4926
    }
  ],
  "quotas": [
    {
      "family": "G",
      "region": "us-east-1",
      "market": "on_demand",
      "baseline_vcpus": 192,
      "used_vcpus": 0
    }
  ]
}
```
- **instances**: Multi-GPU instance types with GPU specs and SkyPilot pricing. Filtered to Koi-supported GPUs (H100, A100, L40S, L4, A10G).
- **quotas**: Raw per-(family, region, market) vCPU limits from AWS. Koi's Oracle joins instances to quotas via `quota_family` to compute how many instances of each type can be launched.
</details>

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and known limitations.

---

## Contact & Support

Tandemn System is developed and maintained by [Tandemn Labs](https://tandemn.com).

For bugs and feature requests, open an issue on [GitHub](https://github.com/Tandemn-Labs/Tandemn-orca/issues). For enterprise inquiries or support agreements, contact us directly at admin@tandemn.com.

---

## License

MIT License. Depends on [SkyPilot](https://github.com/skypilot-org/skypilot), which is licensed under Apache 2.0.
