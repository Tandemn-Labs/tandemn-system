# Docker Compose Packaging Plan

## Overview

Package the server for easy user deployment using Docker Compose. Users go from a multi-step
Python/uv/SkyPilot setup to:

```bash
curl -O https://raw.githubusercontent.com/.../docker-compose.yml
curl -O https://raw.githubusercontent.com/.../.env.example
cp .env.example .env   # fill in values
docker compose up -d
```

Release artifacts: Docker image (pushed to GHCR on tag), `docker-compose.yml`, `.env.example`.

---

## Files to create/modify

| File | Action |
|---|---|
| `requirements.txt` | Fix malformed line 32 (split `-r` and `faiss-cpu` onto separate lines) |
| `Dockerfile` | New file |
| `docker-compose.yml` | Add `orca` service, `sky_state` volume, `REDIS_URL` override |
| `.env.example` | New file |
| `.github/workflows/release.yml` | New file |

---

## 1. Dockerfile

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install torch CPU-only first to prevent the GPU wheel (~2.5GB) from being pulled.
# The server never runs GPU inference — torch is only used for tokenizer metadata.
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Copy and install the rest of dependencies (uv won't reinstall torch)
COPY requirements.txt ./
COPY LLM_placement_solver/requirements.txt ./LLM_placement_solver/
RUN uv pip install --system --no-cache -r requirements.txt

# Install cloudflared for --tunnel support
RUN curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared

COPY . .

EXPOSE 26336

CMD ["python", "server.py"]
```

### Key decisions

- **CPU-only torch**: The server uses `torch`/`transformers` only for placement heuristics and
  tokenizer metadata — never for inference. GPU wheel is ~2.5GB; CPU wheel is ~200MB. Installing
  it first with `--index-url` prevents uv from pulling the GPU wheel when processing
  `requirements.txt`.
- **`openssh-client`**: SkyPilot SSHes into launched EC2 instances to deploy vLLM. Required.
- **`cloudflared` baked in**: The `--tunnel` flag looks for `cloudflared` on PATH.
- **Submodule**: `COPY . .` copies the already-initialized `LLM_placement_solver/` directory.
  Users building locally must run `git submodule update --init` first. The CI workflow handles
  this automatically via `submodules: recursive`.

### Bug to fix first

`requirements.txt` line 32 is malformed — `faiss-cpu>=1.7` is concatenated onto the `-r` line:

```
# Current (broken):
-r LLM_placement_solver/requirements.txtfaiss-cpu>=1.7

# Fix:
-r LLM_placement_solver/requirements.txt
faiss-cpu>=1.7
```

---

## 2. docker-compose.yml (updated)

```yaml
services:
  orca:
    image: ghcr.io/YOUR_ORG/orca:latest
    # build: .    # uncomment to build from source
    ports:
      - "26336:26336"
    env_file: .env
    environment:
      # Override the localhost default — Redis is reachable by service name in compose networking
      REDIS_URL: redis://redis:6379/0
    volumes:
      - ${HOME}/.aws:/root/.aws:ro       # AWS credentials for SkyPilot + boto3
      - sky_state:/root/.sky             # SkyPilot cluster state — CRITICAL, see note
      - ./outputs:/app/outputs           # job outputs written to disk
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5
    restart: unless-stopped

  # Placeholder for the future intelligence service
  # intelligence:
  #   image: ghcr.io/YOUR_ORG/orca-intelligence:latest
  #   restart: unless-stopped

volumes:
  redis_data:
  sky_state:
```

### Key decisions

- **`sky_state` named volume (CRITICAL)**: SkyPilot writes all cluster state — including SSH keys
  for EC2 instances — to `~/.sky`. If this is not persisted, container restarts cause orca to
  lose track of every running cluster, orphaning live EC2 instances that continue to run and bill.
  Use a named volume (not a bind-mount) so it survives `docker compose restart` and image upgrades.
  Only destroyed by `docker compose down -v`.

- **`REDIS_URL` in `environment:`, not `.env`**: `config.py` defaults to
  `redis://localhost:6379/0`. Inside compose, `localhost` resolves to the container itself — Redis
  is at `redis://redis:6379/0` (the service name). Setting this in the compose `environment:`
  block means users never need to know about it.

- **`~/.aws` bind-mount**: SkyPilot and boto3 both use standard AWS credential discovery.
  Read-only mount of the host's `~/.aws` is the standard pattern. Alternative: pass
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as env vars in `.env`.

---

## 3. .env.example

```bash
# ─── Required ────────────────────────────────────────────────────────────────

# Public URL where EC2 replicas can reach this server.
# If you don't have a fixed URL, leave this empty and start the server with:
#   docker compose run orca python server.py --tunnel
ORCA_SERVER_URL=https://your-server-url.example.com

# HuggingFace token — required for gated models (Llama, Gemma, Mistral, etc.)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# S3 bucket for job inputs and outputs
S3_UPLOAD_BUCKET=my-orca-bucket

# ─── Optional ────────────────────────────────────────────────────────────────

# API key used by replicas to authenticate with this server.
# Leave empty to disable authentication (not recommended for public URLs).
# ORCA_API_KEY=

# Placement solver: "roofline" (default, no license required) or "user_specified"
# TD_PLACEMENT_SOLVER=roofline

# Optimization target: "cost_first" or "throughput_first"
# TD_PLACEMENT_PRIORITY=cost_first

# ─── Do not set — managed automatically by docker-compose.yml ────────────────
# REDIS_URL  (set to redis://redis:6379/0 by compose)
```

---

## 4. GitHub Actions release workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  docker:
    runs-on: ubuntu-latest
    permissions:
      contents: write      # to create GitHub release assets
      packages: write      # to push to GHCR

    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive    # initializes LLM_placement_solver

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
            ghcr.io/${{ github.repository }}:latest

      - name: Attach install files to release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            docker-compose.yml
            .env.example
```

The `softprops/action-gh-release` step attaches `docker-compose.yml` and `.env.example` as
versioned release assets, so the user curl commands point at stable URLs rather than raw branch HEAD.

---

## 5. Gurobi note

`LLM_placement_solver/requirements.txt` includes `gurobipy`, which is installed into the image
but requires a commercial license to run. Since `TD_PLACEMENT_SOLVER` defaults to `roofline` in
`config.py`, this is dormant for most users — Gurobi is never invoked unless explicitly configured.
Document in `.env.example` that the default requires no license.
