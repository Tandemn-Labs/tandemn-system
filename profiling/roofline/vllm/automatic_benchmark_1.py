#!/usr/bin/env python3
"""Full benchmark sweep across models × TP/PP × IO lengths × GPU types.

Wrapper around automatic_launch_1.py that generates experiment matrices
and orchestrates the full sweep. Supports dry-run to preview the plan
and selective GPU type filtering for quick tests.

Usage:
    # Dry-run: show full plan without launching
    python automatic_benchmark_1.py

    # Dry-run for specific GPU type
    python automatic_benchmark_1.py --gpu L40S

    # Actually launch on L40S
    python automatic_benchmark_1.py --gpu L40S --run

    # Quick test: single GPU type, single model
    python automatic_benchmark_1.py --gpu A10G --run --models Qwen/Qwen3-32B

    # Generate CSV only (don't launch)
    python automatic_benchmark_1.py --gpu A10G --csv-only
"""

import csv
import sys
import argparse
from pathlib import Path
from itertools import product
from datetime import datetime

from automatic_launch_1 import (
    GPU_CONFIGS,
    load_experiments,
    group_by_cluster_then_io,
    run_cluster_benchmarks,
    check_orphaned_cluster,
    save_results_csv,
)

# ── Sweep configuration ──────────────────────────────────────────────────────

MODELS = [
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-235B-A22B",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",  # tentative
]

TP_PP_CONFIGS = [
    (2, 4),
    (4, 2), (4, 3), (4, 4),
    (8, 1), (8, 2), (8, 3),
]

IO_LENGTHS = [
    (128, 128),      # ratio 1.0  — baseline, max concurrency
    (128, 2048),     # ratio 0.06 — extreme decode (chatbot-like)
    (512, 1024),     # ratio 0.5  — medium decode-heavy
    (1024, 512),     # ratio 2.0  — medium balanced
    (4096, 1024),    # ratio 4.0  — long context, prefill-leaning
    (8192, 256),     # ratio 32.0 — extreme prefill (RAG/summarization)
    (16384, 2048),   # ratio 8.0  — maximum context stress test
]

GPU_TYPES = ["A10G", "L40S", "L4", "A100_40gb", "A100_80gb", "H100"]

# ── Helper functions ──────────────────────────────────────────────────────────


def generate_experiment_csv(output_path, models, tp_pp_configs, io_lengths):
    """Generate CSV with all experiment combinations.

    Returns the number of experiments written.
    """
    fieldnames = ["model", "tensor_degree", "pipeline_degree", "max_input_length", "max_output_length"]
    count = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, (tp, pp), (in_len, out_len) in product(models, tp_pp_configs, io_lengths):
            writer.writerow({
                "model": model,
                "tensor_degree": tp,
                "pipeline_degree": pp,
                "max_input_length": in_len,
                "max_output_length": out_len,
            })
            count += 1
    return count


def get_cluster_info(tp, pp, gpu_type):
    """Get cluster sizing for a TP/PP config (uses same logic as automatic_launch_1)."""
    from automatic_launch_1 import get_cluster_config
    gpu_config = GPU_CONFIGS[gpu_type]
    gpus_per_node, num_nodes = get_cluster_config(tp, pp, gpu_type)
    total_gpus = gpus_per_node * num_nodes

    pricing = gpu_config["pricing"].get(gpus_per_node, {})
    instance_type = pricing.get("instance_type", "unknown")
    price_per_hour = pricing.get("price_per_hour", 0) * num_nodes

    return {
        "gpus_per_node": gpus_per_node,
        "num_nodes": num_nodes,
        "total_gpus": total_gpus,
        "instance_type": instance_type,
        "price_per_hour": price_per_hour,
        "instance_desc": f"{num_nodes}x {instance_type}" if num_nodes > 1 else instance_type,
    }


def render_sweep_plan(models, tp_pp_configs, io_lengths, gpu_types, cloud="aws"):
    """Print a human-readable summary of the sweep plan."""
    total_experiments = len(models) * len(tp_pp_configs) * len(io_lengths)

    print("\n" + "=" * 80)
    print("  BENCHMARK SWEEP PLAN")
    print("=" * 80)
    print(f"\n  Cloud: {cloud.upper()}")

    print(f"\n  Models ({len(models)}):")
    for m in models:
        print(f"    - {m}")

    print(f"\n  TP/PP configs ({len(tp_pp_configs)}):")
    for tp, pp in tp_pp_configs:
        total_gpus = tp * pp
        print(f"    - TP={tp}, PP={pp}  ({total_gpus} GPUs)")

    print(f"\n  Input/Output lengths ({len(io_lengths)}):")
    for in_len, out_len in io_lengths:
        ratio = in_len / out_len if out_len > 0 else 0
        print(f"    - {in_len:>5}in / {out_len:>4}out  (ratio {ratio:.1f})")

    print(f"\n  Experiments per GPU type: {total_experiments}")
    print(f"  Total experiments across all GPU types: {total_experiments * len(gpu_types)}")

    print(f"\n  GPU types ({len(gpu_types)}):")
    for gpu_type in gpu_types:
        gpu_config = GPU_CONFIGS[gpu_type]
        print(f"\n    {gpu_type} ({gpu_config['gpu_generation']}, {gpu_config['gpu_mem_gb']}GB, {gpu_config['interconnect']}):")

        # Group experiments by cluster config to show how many clusters are needed
        clusters = {}
        for (tp, pp), (in_len, out_len) in product(tp_pp_configs, io_lengths):
            info = get_cluster_info(tp, pp, gpu_type)
            key = (info["gpus_per_node"], info["num_nodes"], info["instance_desc"])
            if key not in clusters:
                clusters[key] = {"info": info, "experiments": []}
            clusters[key]["experiments"].append((tp, pp, in_len, out_len))

        total_cost_estimate = 0
        for (gpus, nodes, desc), data in sorted(clusters.items()):
            n_exp = len(data["experiments"]) * len(models)
            price = data["info"]["price_per_hour"]
            # Rough estimate: ~5 min per experiment (model load + warmup + benchmark + cleanup)
            est_hours = n_exp * 5 / 60
            est_cost = price * est_hours
            total_cost_estimate += est_cost
            print(f"      {desc:>25s}  ({gpus} GPU/node × {nodes} node)  "
                  f"{n_exp:>3} experiments  ~${est_cost:.0f} est.")

        print(f"      {'':>25s}  Total: ~${total_cost_estimate:.0f} estimated cost")

    grand_total = 0
    for gpu_type in gpu_types:
        clusters = {}
        for (tp, pp), (in_len, out_len) in product(tp_pp_configs, io_lengths):
            info = get_cluster_info(tp, pp, gpu_type)
            key = (info["gpus_per_node"], info["num_nodes"])
            if key not in clusters:
                clusters[key] = {"info": info, "count": 0}
            clusters[key]["count"] += len(models)
        for data in clusters.values():
            est_hours = data["count"] * 5 / 60
            grand_total += data["info"]["price_per_hour"] * est_hours

    print(f"\n  {'=' * 60}")
    print(f"  GRAND TOTAL: ~${grand_total:.0f} estimated cost (all GPU types)")
    print(f"  (Assumes ~5 min per experiment — actual time depends on model size)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run full benchmark sweep across models × TP/PP × IO lengths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automatic_benchmark_1.py                          # Dry-run: show full plan
  python automatic_benchmark_1.py --gpu L40S               # Dry-run for L40S only
  python automatic_benchmark_1.py --gpu L40S --run         # Launch L40S sweep
  python automatic_benchmark_1.py --gpu A10G --csv-only    # Generate CSV only
  python automatic_benchmark_1.py --gpu A10G --run --models Qwen/Qwen3-32B  # Quick test
        """,
    )
    parser.add_argument(
        "--gpu", "--gpu-type", dest="gpu_types",
        nargs="+", choices=GPU_TYPES, default=None,
        help=f"GPU type(s) to run. Default: all ({', '.join(GPU_TYPES)})",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Actually launch clusters (default: dry-run plan only)",
    )
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Generate experiment CSV(s) and exit without launching",
    )
    parser.add_argument(
        "--s3-models", action="store_true", default=False,
        help="Load models from S3 instead of HuggingFace",
    )
    parser.add_argument(
        "--hf-models", action="store_true", default=False,
        help="Load models from HuggingFace Hub (default when --s3-models not set)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Override model list (default: {', '.join(MODELS)})",
    )
    parser.add_argument(
        "--tp-pp", nargs="+", default=None,
        help="Override TP/PP configs as 'TP,PP' pairs (e.g., --tp-pp 4,2 8,1)",
    )
    parser.add_argument(
        "--io", nargs="+", default=None,
        help="Override IO lengths as 'IN,OUT' pairs (e.g., --io 512,128 2048,512)",
    )
    parser.add_argument(
        "--cloud", dest="cloud",
        choices=["aws", "gcp", "azure"],
        default="aws",
        help="Cloud provider to launch on (default: aws). Ensures all instances stay on one cloud.",
    )
    parser.add_argument(
        "--spot", action="store_true",
        help="Use spot/preemptible instances (cheaper but may be interrupted)",
    )

    args = parser.parse_args()

    # Resolve parameters
    models = args.models or MODELS
    gpu_types = args.gpu_types or GPU_TYPES

    if args.tp_pp:
        tp_pp_configs = []
        for pair in args.tp_pp:
            tp, pp = pair.split(",")
            tp_pp_configs.append((int(tp), int(pp)))
    else:
        tp_pp_configs = TP_PP_CONFIGS

    if args.io:
        io_lengths = []
        for pair in args.io:
            in_len, out_len = pair.split(",")
            io_lengths.append((int(in_len), int(out_len)))
    else:
        io_lengths = IO_LENGTHS

    # Always show the plan
    render_sweep_plan(models, tp_pp_configs, io_lengths, gpu_types, cloud=args.cloud)

    if not args.run and not args.csv_only:
        print("This is a DRY RUN. Add --run to actually launch clusters.")
        print("Add --csv-only to generate experiment CSVs without launching.\n")
        return

    # Check for orphaned clusters
    check_orphaned_cluster()

    # Process each GPU type
    all_results = []
    for gpu_type in gpu_types:
        csv_path = Path(f"sweep_{gpu_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        n_experiments = generate_experiment_csv(csv_path, models, tp_pp_configs, io_lengths)
        print(f"\n📊 Generated {csv_path} with {n_experiments} experiments")

        if args.csv_only:
            continue

        # Load experiments and group them
        experiments = load_experiments(str(csv_path))
        cluster_groups = group_by_cluster_then_io(experiments, gpu_type)

        print(f"\n🚀 Running sweep for GPU type: {gpu_type}")
        print(f"   {len(experiments)} experiments in {len(cluster_groups)} cluster groups")

        try:
            for (gpus, nodes, tp, pp, model), exps in sorted(cluster_groups.items()):
                parent_dir = "results"
                cluster_config = (gpus, nodes)
                results = run_cluster_benchmarks(
                    cluster_config, exps,
                    parent_dir=parent_dir,
                    dry_run=False,
                    gpu_type=gpu_type,
                    s3_models=args.s3_models,
                    cloud=args.cloud,
                    use_spot=args.spot,
                )
                if results:
                    all_results.extend(results)
        except Exception as e:
            print(f"\n❌ Error during {gpu_type} sweep: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        output_csv = f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_results_csv(all_results, output_csv)
        print(f"\n✅ Sweep complete! {len(all_results)} results saved to {output_csv}")

        # Summary
        success = sum(1 for r in all_results if r.get("status") == "success")
        errors = sum(1 for r in all_results if r.get("status") in ("error", "cluster_error"))
        print(f"   Success: {success}, Errors: {errors}")


if __name__ == "__main__":
    main()
