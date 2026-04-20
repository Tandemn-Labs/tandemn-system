# Orca Docker Compose Release

This bundle is intended for running Orca from the published GHCR image.
Detailed instructions for running can be found on [docs.tandemn.com](https://docs.tandemn.com)

## Files

- `docker-compose.yml`
- `.env.example`

## Start

```bash
cp .env.example .env
docker compose pull
docker compose up -d
```

The service listens on port `26336`.

Before starting, update `.env` with your real values for required settings such as:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `S3_UPLOAD_BUCKET`
- `HF_TOKEN`
- `TD_SERVER_URL`

If you use the placement advisor, also set `ANTHROPIC_API_KEY`.
