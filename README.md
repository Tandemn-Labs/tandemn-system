# Orca
<div align="center">
<img src="https://raw.githubusercontent.com/Tandemn-Labs/Tandemn-orca/demo/thapar-cli/data/orca.png" width="400" alt="Orca">
</div>

Batch inference on large models means choosing the right GPU, figuring out tensor and pipeline parallelism, picking a region with available quota, and hoping it all fits in memory before your deadline. Most teams just guess.

Orca handles all of that. Give it a model name, a JSONL file, and a deadline. Its roofline-based placement solver sizes the job automatically — picking the instance type, parallelism configuration, and AWS region — then launches on spot via SkyPilot. While the job runs, Orca streams real-time throughput, latency, and scheduler metrics back to your terminal. Output lands in S3 when it's done.

---

## Prerequisites

- Python 3.11+
- AWS credentials (`~/.aws/credentials`)
- SkyPilot configured for AWS

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
orca deploy Qwen/Qwen2.5-72B-Instruct input.jsonl --slo 4
```

Orca parses the input file, runs the placement solver, and launches on the cheapest viable spot configuration. No GPU selection required.

Track progress:

```bash
orca progress
```

---

## CLI

### Deployment

```
orca deploy <model> <input>     Run a batch job (solver picks GPU automatically)
orca plan   <model> <input>     Show placement plan without launching
```

Options for `deploy` and `plan`:

```
--slo <hours>           Deadline in hours (default: 4)
--max-output-tokens N   Max tokens per response (default: 1024)
--gpu <type>            Override GPU type (e.g. A100, L40S, H100)
--tp / --pp             Override tensor / pipeline parallelism
--force                 Skip feasibility check and launch anyway
--persist               Keep cluster alive after job completes
--on-demand             Use on-demand instances instead of spot
```

### Monitoring

```
orca progress [job_id]          Live progress bar with throughput and queue depth
orca status                     List all jobs
orca metrics <job_id> [-w]      Latest vLLM metrics snapshot (--watch for 2s refresh)
orca stream <job_id>            Stream live metrics table (1 event/sec via SSE)
orca logs [cluster]             Stream logs from a SkyPilot cluster
orca clusters                   Show active clusters
```

### Analytics

```
orca history [--model X] [--gpu Y]   Browse completed runs
orca inspect <run_id>                Full run report (latency, throughput, cost, GPU util)
orca timeseries <run_id>             Scheduler timeseries for a completed run
```

---

## How It Works

```
                   orca deploy Qwen/Qwen2.5-72B-Instruct batch.jsonl --slo 4
                                         |
                                         v
                              +---------------------+
                              |   Control Plane      |
                              |   (server.py)        |
                              +---------------------+
                              |  1. Parse input      |
                              |  2. Roofline solver  |
                              |  3. Quota check      |
                              |  4. SkyPilot launch  |
                              +----------+----------+
                                         |
                          +--------------+--------------+
                          |                             |
                          v                             v
                   +-------------+              +--------------+
                   |  EC2 Spot   |  metrics/s   |  Metrics     |
                   |  vLLM V1    | -----------> |  Collector   |
                   |  + runner   |              |  (SQLite)    |
                   +------+------+              +--------------+
                          |
                          v
                    S3 (output.jsonl + metrics.csv)
```

**Placement solver.** Uses a roofline model to estimate throughput and memory requirements across GPU types and TP/PP configurations. Picks the cheapest option that completes within your SLO. Automatically falls back to alternative regions and instance types if the primary launch fails.

**Quota tracking.** Real-time quota usage across AWS regions. Orca won't try to launch where you have no capacity.

**Observability.** The runner on the EC2 node pushes Prometheus snapshots to the control plane every second. Orca computes throughput from counter deltas, extracts histogram quantiles (TTFT, TPOT, E2E, queue/prefill/decode/inference), and tracks KV cache utilization and scheduler state. All metrics are persisted to SQLite for post-run analysis.

**Teardown.** Clusters are destroyed by default after job completion. Use `--persist` to keep them alive.

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

## API

The control plane exposes a REST API at `http://localhost:26336`:

| Endpoint | Description |
|----------|-------------|
| `POST /submit/batch` | Submit batch inference job |
| `POST /test/placement` | Run solver only (no launch) |
| `GET /job/{id}` | Job status and progress |
| `GET /job/{id}/metrics` | Latest metrics snapshot |
| `GET /job/{id}/metrics/stream` | SSE metrics stream |
| `GET /job/{id}/throughput` | Sustained throughput (rolling window) |
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
