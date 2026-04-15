FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gosu \
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

# Create non-root user and hand off /app ownership
RUN useradd --create-home --uid 1001 --shell /bin/bash orca \
    && chown -R orca:orca /app

COPY --chown=orca:orca . .
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 26336

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "server.py"]
