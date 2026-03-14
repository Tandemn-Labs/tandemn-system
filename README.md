**Orca**

Batch inference on large models means choosing the right GPU, figuring out tensor and pipeline parallelism, picking a region with available quota, and hoping it all fits in memory before your deadline. Most teams just guess.

Orca handles all of that. Give it a model name, a JSONL file, and a deadline. Its roofline-based placement solver sizes the job automatically — picking the instance type, parallelism configuration, and AWS region — then launches on spot via SkyPilot. Output lands in S3 when it's done.

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

Start the server (control plane):

```bash
python server.py
```

Run a batch job:

```bash
orca deploy Qwen/Qwen2.5-72B-Instruct input.jsonl --slo 4
```

Orca parses the input file, runs the solver, and launches on the cheapest viable spot configuration. No GPU selection required.

Track progress:

```bash
orca progress
```

---

## CLI

```
orca deploy <model> <input>     Run a batch job (solver picks GPU automatically)
orca plan   <model> <input>     Show placement plan without launching
orca progress [job_id]          Monitor job progress
orca status                     List all jobs
orca logs <cluster>             Stream logs from a running cluster
orca clusters                   Show active SkyPilot clusters
```

Key options for `deploy` and `plan`:

```
--slo <hours>           Deadline in hours (default: 4)
--max-output-tokens N   Max tokens per response (default: 1024)
--gpu <type>            Override GPU type (e.g. A100, L40S, H100)
--tp / --pp             Override tensor / pipeline parallelism
--force                 Skip feasibility check and launch anyway
```

---

## How It Works

The placement solver uses a roofline model to estimate throughput and memory requirements for your workload across GPU types and parallelism configurations. It picks the cheapest option that completes within your SLO, with automatic fallback to alternative regions and instance types if the primary launch fails.

Quota usage is tracked in real time across AWS regions — Orca won't try to launch where you have no capacity.

---

## Input Format

Standard OpenAI batch JSONL:

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "placeholder", "messages": [{"role": "user", "content": "Your prompt here"}], "max_tokens": 256}}
```

Local files are uploaded to S3 automatically. S3 URIs are passed through directly.

---

## Contact

- Hetarth — hetarth@tandemn.com
- Mankeerat — mankeerat@tandemn.com

---

## License

MIT (depends on SkyPilot under Apache License 2.0)
