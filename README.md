# Orca
<div align="center">
<img src="data/orca.png" width="400" alt="Orca">
</div>

Batch inference on large models means choosing the right GPU, figuring out tensor and pipeline parallelism, picking a region with available quota, and hoping it all fits in memory before your deadline. Most teams just guess.

Orca handles all of that. Give it a model name, a JSONL file, and a deadline. Its roofline-based placement solver sizes the job automatically — picking the instance type, parallelism configuration, and AWS region — then launches on spot via SkyPilot. While the job runs, Orca streams real-time throughput, latency, and scheduler metrics back to your terminal. Output lands in S3 when it's done.

---

## Prerequisites

- Python 3.11+
- AWS credentials (`~/.aws/credentials`)
- SkyPilot configured for AWS
- Redis (for multi-replica chunked jobs): `docker run -d -p 6379:6379 redis`

---

## Installation

```bash
git clone --recurse-submodules https://github.com/Tandemn-Labs/Tandemn-orca.git
cd Tandemn-orca
pip install -r requirements.txt
```

```bash
# .env
S3_UPLOAD_BUCKET=your-s3-bucket
HF_TOKEN=your_hf_token        # for gated models (Llama, etc.)
```

---

## Quick Start

Start the control plane:

```bash
python server.py
```

Run a batch job:

```bash
./orca deploy Qwen/Qwen2.5-72B-Instruct input.jsonl --slo 4
```

Orca parses the input file, runs the placement solver, and launches on the cheapest viable spot configuration. No GPU selection required.

Track progress:

```bash
./orca progress
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
--replicas N            Number of replica clusters (enables chunked multi-replica mode)
--chunk-size N          Lines per chunk (default: 1000)
--chunked               Force chunked pipeline (even with 1 replica)
--force                 Skip feasibility check and launch anyway
--persist               Keep cluster alive after job completes
--on-demand             Use on-demand instances instead of spot
```

### Hot-Swap Replicas

Change GPU type, TP/PP, or replica count mid-job without losing progress:

```
./orca swap <job_id> --gpu A100 --tp 4 --replicas 2
./orca swap <job_id> --gpu L40S --tp 1 --ready-threshold 2 --on-demand
```

New replicas join the same Redis chunk queue. Old replicas are killed after the new ones start processing.

### Monitoring

```
./orca progress [job_id]          Live progress bar with throughput and queue depth
./orca status                     List all jobs
./orca metrics <job_id> [-w]      Latest vLLM metrics snapshot (--watch for 2s refresh)
./orca metrics <job_id> --replica <rid>   Per-replica metrics
./orca metrics <job_id> --compare         Aggregated + per-replica side by side
./orca stream <job_id>            Stream live metrics table (1 event/sec via SSE)
./orca logs [cluster]             Stream logs from a SkyPilot cluster
./orca clusters                   Show active clusters
```

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

---

## Trying Locally?

The control plane runs on your laptop, but EC2 replicas need to reach it (for metrics, chunk pulling, etc.). The easiest way is a Cloudflare Tunnel — one command, no account required, no firewall changes:

```bash
# Install once
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared

# Expose your local server (run alongside python server.py)
./cloudflared tunnel --url http://localhost:26336
# → https://random-words.trycloudflare.com
```

Then add the URL to your `.env`:

```bash
ORCA_SERVER_URL=https://random-words.trycloudflare.com
ORCA_API_KEY=some-secret-key   # optional but recommended since the tunnel is public
```

Restart the server and the EC2 replicas will be able to call back to your laptop. The URL changes each time you restart `cloudflared` — copy the new one into `.env` and restart `server.py`.

For a persistent setup, use [Tailscale](https://tailscale.com) (mesh VPN, stable IPs, encrypted) or deploy the control plane on a small EC2 instance in the same VPC.

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
| `POST /job/{id}/swap` | Hot-swap replicas to new GPU config |
| **Chunks** | |
| `GET /job/{id}/chunks/progress` | Chunk-level progress (pending/inflight/completed/failed) |
| `POST /job/{id}/chunks/pull` | Pull next chunk (replica-facing) |
| `POST /job/{id}/chunks/complete` | Mark chunk completed |
| `POST /job/{id}/chunks/renew` | Renew chunk lease |
| **Analytics** | |
| `GET /analytics/runs` | List completed runs |
| `GET /analytics/runs/{id}` | Full run report |
| `GET /analytics/runs/{id}/timeseries` | Scheduler timeseries |
| `GET /quota/status` | Quota usage across regions |

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
