# Tandemn System

Self-hosted inference orchestration for large language models on your GPU infrastructure.

This repository contains the Tandemn System control plane. It may still use the legacy `orca` name in code, environment variables, and internal paths.

## Getting Started

### Prerequisites

- Python 3.12+
- `uv`
- AWS credentials configured for EC2, S3, and service quota access
- An existing S3 bucket for inputs and outputs
- Redis for chunked multi-replica jobs
- HuggingFace token for gated models

### Install the Server

```bash
git clone --recurse-submodules https://github.com/Tandemn-Labs/Tandemn-server.git
cd Tandemn-server
bash setup.sh
```

`setup.sh` installs dependencies, checks cloud access, verifies Redis connectivity, and creates `.env`.

### Install the CLI

```bash
pip install tandemn
```

### Start the Control Plane

The control plane must be reachable by EC2 replicas. Set the public or private URL before starting it.

```bash
python server.py --url https://your-public-url.example.com
```

### Run a Job

```bash
tandemn deploy Qwen/Qwen2.5-7B-Instruct examples/workloads/demo_batch.jsonl --slo 4
```

### Monitor Progress

```bash
tandemn progress
tandemn web
```

## Documentation

Full documentation lives in the [`tandemn-docs`](../tandemn-docs) repository.
