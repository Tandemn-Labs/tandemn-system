# vLLM Roofline Profiling

This directory is the active offline profiling stack migrated from `tandemn-profiling/roofline/vllm` into `tandemn-system` so Orca and profiling can evolve in the same repository.

## What Lives Here

- `automatic_benchmark_1.py`: main sweep entrypoint
- `automatic_launch_1.py`: SkyPilot launch + benchmark orchestration
- `benchmark_client.py`: async OpenAI-compatible benchmark client
- `prometheus_parser.py`: parse vLLM `/metrics` output
- `plot_benchmark_results.py`: aggregate result plots
- `plot_timeseries.py`: per-run timeseries plots
- `scripts/build_perfdb.py`: merge run outputs into `perfdb_all.csv`
- `test_prometheus_parser.py`: parser unit tests

## Why It Moved

The immediate goal is to keep the active profiling code close to Orca so we can:

- build a `vLLM` failure bench
- add an Orca `--profiling` execution mode later
- normalize profiling outputs into a shared database/schema for future Koi integration

Generated results, old sweeps, and one-off experiment artifacts were intentionally not migrated in this first pass.

## How To Use It

Run from this directory:

```bash
cd profiling/roofline/vllm
```

Dry-run a small sweep:

```bash
python automatic_benchmark_1.py --gpu L40S --models Qwen/Qwen3-32B --tp-pp 4,2 --io 1024,512
```

Launch it for real:

```bash
python automatic_benchmark_1.py --gpu L40S --models Qwen/Qwen3-32B --tp-pp 4,2 --io 1024,512 --run
```

Run the parser test:

```bash
pytest test_prometheus_parser.py
```
