# Agent Instructions for Tandemn System

This file is for coding agents working on Orca inside `tandemn-system`. Keep it lean: project-wide rules belong here; area-specific details should move into smaller domain guides when they grow.

## Project Identity

- Orca is not a generic developer tool. It is the control-plane/runtime layer for launching, monitoring, and managing model serving/batch jobs.
- Koi is first class. Orca is most useful with Koi placement/advisor flows, but users must still be able to run Orca with AIC data and non-Koi paths.
- The CLI is not part of Orca. Treat it as a separate HTTP client wrapper repo; do not add CLI-only behavior or UX code here unless explicitly requested.

## Code Style

- Prefer small, direct changes over broad abstractions. Avoid Transformer-style over-abstraction when a clear function or small module is enough.
- Keep code understandable for systems engineers operating production infrastructure: explicit control flow beats clever generalization.
- Follow the existing local patterns before inventing a new one. Search for similar launch, storage, quota, placement, webhook, and monitoring code before adding new shapes.
- Use comments only for non-obvious operational logic, failure modes, or cross-service contracts. Do not comment what the code already says.
- Good style references: vLLM, SGLang, Ray, NVIDIA Dynamo, SkyPilot. Transformers has strong quality but is often more abstract than Orca should be.

## Linting, Formatting, and Type Checking

- Use the repo tooling, not ad hoc formatter choices.
- Run before submitting: `ruff check .` and `ruff format --check .`.
- If you changed tests or behavior, run the relevant pytest target. For broad changes, run `pytest`.
- Mypy is currently advisory in CI while the existing type backlog is cleaned up. Do not ignore new type errors casually; explain them if you leave them.
- Install local hooks when working repeatedly in this repo: `pip install pre-commit && pre-commit install`.

## CI/CD Expectations

- GitHub Actions is the source of truth for PR checks.
- The CI workflow runs lint, advisory typecheck, pre-commit hooks, pytest, the profiling parser unit test, and a Docker build.
- Release tags matching `v*.*.*` build and publish the Orca image to GHCR.
- Do not add CI steps that require AWS, GPUs, paid cloud resources, Hugging Face gated-model downloads, or Anthropic keys to normal PR CI. Those belong in manual workflows only.
- If a test needs external data or a heavy solver dependency, make that dependency explicit with a marker, skip guard, fixture, or separate workflow.

## Dependency Policy

- Do not pin to the latest vLLM just because it exists. Upgrade only when it passes Orca's correctness, launch, monitoring, placement, compatibility, and cost constraints.
- Prefer stable, boring dependencies for the control plane. Experimental serving-stack upgrades need focused validation and clear rollback paths.
- Keep dependency changes narrow. If bumping vLLM, SkyPilot, torch, pydantic, FastAPI, or Redis clients, call out the operational impact and test coverage.

## Cloud, Storage, and Abstraction Boundaries

- Follow SkyPilot's abstraction style for cloud/deployment behavior. Avoid scattered direct AWS boto calls for launch/deployment paths.
- If direct provider APIs are unavoidable, wrap them behind a factory/interface so other clouds can implement the same primitives.
- Storage should use shared primitives. Do not hardcode local SSD paths or AWS-only assumptions in code that should work for other clouds or on-prem.
- For on-prem flows, prefer MinIO-compatible storage behind the same storage abstraction instead of special-casing filesystem paths.
- Keep provider-specific code isolated and searchable. Cross-cloud behavior should be expressed through Orca/SkyPilot/storage interfaces.

## Testing Philosophy

- Use tests to protect business and systems behavior: placement decisions, launch lifecycle, callbacks, retries, failure classification, storage semantics, and observability.
- Code -> test -> code is fine. Business-function contract -> test -> code is often better for Orca because regressions are usually operational.
- Avoid overfitting tests to implementation details when the user-visible or Koi-visible contract is what matters.
- For bug fixes, prefer a focused regression test unless the setup would require paid cloud/GPU resources.

## Repository Boundaries

- Keep profiling research code separate from production Orca code unless intentionally promoting a feature into the runtime path.
- Do not commit generated profiling results, sweep CSVs, caches, credentials, local `.env` files, or large artifacts unless explicitly requested.
- Respect submodules. Do not assume `LLM_placement_solver` internals are editable from this repo unless the task explicitly spans that repo.

## Editing Agent Instruction Files

- Keep `AGENTS.md` under 200 lines.
- Add a rule only if an agent would likely do the wrong thing without it.
- Do not copy tool documentation here; link or rely on repo config.
- If more than five non-obvious rules are about one subsystem, create a domain guide instead of bloating this file.
