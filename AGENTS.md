# Agent Instructions for Tandemn System

This file is for coding agents working on Orca inside `tandemn-system`. Keep it lean: project-wide rules belong here; area-specific details should move into smaller domain guides when they grow.

## Project Identity

- Orca is not a generic developer tool. It is the control-plane/runtime layer for launching, monitoring, and managing model serving/batch jobs.
- Koi is first class. Orca is most useful with Koi placement/advisor flows, but users must still be able to run Orca with AIC data and non-Koi paths.
- The CLI is not part of Orca. Treat it as a separate HTTP client wrapper repo; do not add CLI-only behavior or UX code here unless explicitly requested.

## Code Style

- Prefer small, direct changes over broad abstractions. Avoid Transformer-style over-abstraction when a clear function or small module is enough.
- Keep code understandable for systems engineers operating production infrastructure: explicit control flow beats clever generalization.
- Follow the existing local patterns before inventing a new one. Search for similar launch, reconciliation, health, compiler, and monitoring code before adding new shapes.
- Use comments only for non-obvious operational logic, failure modes, or cross-service contracts. Do not comment what the code already says.
- Good style references: vLLM, SGLang, Ray, NVIDIA Dynamo, and Kubernetes controllers. Transformers has strong quality but is often more abstract than Orca should be.

## Linting, Formatting, and Type Checking

- Use the repo tooling, not ad hoc formatter choices.
- Run before submitting: `ruff check .` and `ruff format --check .`.
- If you changed tests or behavior, run the relevant pytest target. For broad changes, run `pytest`.
- Mypy is currently advisory in CI while the existing type backlog is cleaned up. Do not ignore new type errors casually; explain them if you leave them.
- Install local hooks when working repeatedly in this repo: `pip install pre-commit && pre-commit install`.

## CI/CD Expectations

- GitHub Actions is the source of truth for PR checks.
- The CI workflow runs lint, advisory typecheck, pre-commit hooks, pytest, and a Docker build.
- Release tags matching `v*.*.*` build and publish the Orca image to GHCR.
- Do not add CI steps that require AWS, GPUs, paid cloud resources, Hugging Face gated-model downloads, or Anthropic keys to normal PR CI. Those belong in manual workflows only.
- If a test needs external data or a cluster dependency, make that dependency explicit with a marker, skip guard, fixture, or separate workflow.

## Dependency Policy

- Do not pin to the latest vLLM just because it exists. Upgrade only when it passes Orca's correctness, launch, monitoring, compatibility, and cost constraints.
- Prefer stable, boring dependencies for the control plane. Experimental serving-stack upgrades need focused validation and clear rollback paths.
- Keep dependency changes narrow. If bumping vLLM, Dynamo, Kubernetes, torch, pydantic, or PostgreSQL clients, call out the operational impact and test coverage.

## Cloud and Cluster Boundaries

- Keep per-cloud selectors and catalog behavior isolated. Cross-cloud workload behavior belongs in the Kubernetes compilers and launcher boundaries.
- Use shared storage and model-volume primitives. Do not hardcode local SSD paths or provider-specific assumptions into cross-cloud code.
- Direct provider APIs belong in catalog/resource-map scripts, not workload reconciliation.

## Testing Philosophy

- Use tests to protect business and systems behavior: plan application, launch lifecycle, retries, failure classification, storage semantics, and observability.
- Code -> test -> code is fine. Business-function contract -> test -> code is often better for Orca because regressions are usually operational.
- Avoid overfitting tests to implementation details when the user-visible or Koi-visible contract is what matters.
- For bug fixes, prefer a focused regression test unless the setup would require paid cloud/GPU resources.

## Repository Boundaries

- Do not commit generated profiling results, sweep CSVs, caches, credentials, local `.env` files, or large artifacts unless explicitly requested.
- `tandemn-store` owns canonical models and persistence contracts; update that repository rather than duplicating them here.

## Editing Agent Instruction Files

- Keep `AGENTS.md` under 200 lines.
- Add a rule only if an agent would likely do the wrong thing without it.
- Do not copy tool documentation here; link or rely on repo config.
- If more than five non-obvious rules are about one subsystem, create a domain guide instead of bloating this file.
