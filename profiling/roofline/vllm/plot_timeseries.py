#!/usr/bin/env python3
"""
Plot GPU and scheduler time-series data from vLLM benchmarks.
Produces publication-quality PDF figures.

Usage:
    python plot_timeseries.py <timeseries_dir_or_file> [--output <output.pdf>]
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import defaultdict
from scipy import interpolate

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
COLORS = {
    'gpu0': '#0072B2',  # Blue
    'gpu1': '#D55E00',  # Orange
    'gpu2': '#009E73',  # Green
    'gpu3': '#CC79A7',  # Pink
    'gpu4': '#F0E442',  # Yellow
    'gpu5': '#56B4E9',  # Light blue
    'gpu6': '#E69F00',  # Amber
    'gpu7': '#8B4513',  # Brown/SaddleBrown
    'avg': '#333333',   # Dark gray for averages
    'running': '#0072B2',
    'waiting': '#D55E00',
    'swapped': '#009E73',
}

# Markers for different nodes
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']


def load_timeseries(path: Path) -> list[dict]:
    """Load timeseries data from file or directory."""
    experiments = []

    if path.is_file():
        with open(path) as f:
            experiments.append(json.load(f))
    elif path.is_dir():
        for json_file in sorted(path.glob('timeseries_*.json')):
            with open(json_file) as f:
                experiments.append(json.load(f))

    return experiments


def extract_gpu_data(gpu_timeseries: list[dict]) -> dict:
    """
    Extract GPU metrics into arrays for plotting.
    
    Handles the case where different GPUs are sampled at different times.
    Each GPU metric gets its own time series (only times when that GPU was sampled).
    """
    if not gpu_timeseries:
        return {}

    # First pass: collect all unique metric keys
    all_keys = set()
    for sample in gpu_timeseries:
        for key in sample.keys():
            if key != 't':
                all_keys.add(key)
    
    # Second pass: extract data per metric (each metric has its own time series)
    data = {}
    for key in all_keys:
        times = []
        values = []
        for sample in gpu_timeseries:
            if key in sample:
                times.append(sample['t'])
                values.append(sample[key])
        if times:  # Only add if we have data
            data[key] = {
                'time': np.array(times),
                'value': np.array(values)
            }
    
    # Also create a unified time array for backward compatibility (all unique times)
    all_times = sorted(set(sample['t'] for sample in gpu_timeseries))
    data['_all_times'] = np.array(all_times)
    
    return data


def extract_scheduler_data(scheduler_timeseries: list[dict]) -> dict:
    """Extract scheduler metrics into arrays for plotting."""
    if not scheduler_timeseries:
        return {}

    data = defaultdict(list)

    for sample in scheduler_timeseries:
        data['time'].append(sample['t'])
        for key, value in sample.items():
            if key != 't':
                data[key].append(value)

    return {k: np.array(v) for k, v in data.items()}


def get_gpu_identifiers(gpu_data: dict) -> list[str]:
    """
    Extract unique GPU identifiers from data keys.

    Handles both formats:
    - Old single-node: gpu0_sm_pct, gpu1_sm_pct -> ['gpu0', 'gpu1']
    - New multi-node: node0_gpu0_sm_pct, node1_gpu0_sm_pct -> ['node0_gpu0', 'node1_gpu0']

    Returns sorted list of GPU identifiers.
    """
    gpu_ids = set()
    for key in gpu_data.keys():
        if key == '_all_times' or key == 'time':
            continue
        # Check for node-prefixed format: node0_gpu0_metric
        if key.startswith('node') and '_gpu' in key:
            parts = key.split('_')
            if len(parts) >= 3:
                gpu_id = f"{parts[0]}_{parts[1]}"  # e.g., "node0_gpu0"
                gpu_ids.add(gpu_id)
        # Check for old format: gpu0_metric
        elif key.startswith('gpu') and '_' in key:
            gpu_id = key.split('_')[0]  # e.g., "gpu0"
            gpu_ids.add(gpu_id)

    # Sort: by node first (if present), then by GPU number
    def sort_key(gpu_id):
        if gpu_id.startswith('node'):
            parts = gpu_id.split('_')
            node_num = int(parts[0][4:])  # Extract number from "node0"
            gpu_num = int(parts[1][3:])   # Extract number from "gpu0"
            return (node_num, gpu_num)
        else:
            return (0, int(gpu_id[3:]))   # Single node: sort by GPU number

    return sorted(gpu_ids, key=sort_key)


def get_num_gpus(gpu_data: dict) -> int:
    """Determine number of GPUs from data keys (legacy compatibility)."""
    return len(get_gpu_identifiers(gpu_data))


def plot_experiment(exp_data: dict, output_path: Path):
    """Create a multi-panel figure for one experiment."""

    gpu_data = extract_gpu_data(exp_data.get('gpu_timeseries', []))
    scheduler_data = extract_scheduler_data(exp_data.get('scheduler_timeseries', []))

    config = exp_data.get('config', {})
    exp_id = exp_data.get('exp_id', 'unknown')
    elapsed = exp_data.get('elapsed_time', 0)
    
    # Get instance_type and gpu_type from results.json in the same directory
    instance_type = None
    gpu_type = None
    try:
        results_json_path = output_path.parent / 'results.json'
        if results_json_path.exists():
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)
                if isinstance(results_data, list) and len(results_data) > 0:
                    exp_data = results_data[0]
                elif isinstance(results_data, dict):
                    exp_data = results_data
                else:
                    exp_data = {}
                
                instance_type = exp_data.get('instance_type')
                gpu_type = exp_data.get('gpu_type')  # Check if gpu_type is directly stored
                
                # If gpu_type not found, infer from instance_type
                if not gpu_type and instance_type:
                    # Map instance family to GPU type
                    instance_str = instance_type.lower()
                    if 'g6e' in instance_str:
                        gpu_type = 'L40S'
                    elif 'g6' in instance_str and 'g6e' not in instance_str:
                        gpu_type = 'L4'
                    elif 'g5' in instance_str:
                        gpu_type = 'A10G'
                    elif 'p4d' in instance_str:
                        gpu_type = 'A100_40gb'  # Could be 40gb or 80gb, default to 40gb
                    elif 'p4de' in instance_str:
                        gpu_type = 'A100_80gb'
    except Exception as e:
        pass

    # Determine what to plot
    has_gpu = bool(gpu_data)
    has_scheduler = bool(scheduler_data)
    has_kv_cache = 'kv_cache_util_pct' in scheduler_data if scheduler_data else False
    has_queues = any(k in scheduler_data for k in ['running', 'waiting', 'swapped']) if scheduler_data else False
    gpu_ids = get_gpu_identifiers(gpu_data) if has_gpu else []
    num_gpus = len(gpu_ids)

    # Calculate number of subplots needed
    # 1. SM Utilization (per GPU)
    # 2. Memory Bandwidth Utilization (per GPU)
    # 3. Memory Usage (GB, per GPU)
    # 4. KV Cache Utilization (if available)
    # 5. Scheduler Queues (if available)

    num_plots = 3  # GPU metrics
    if has_kv_cache:
        num_plots += 1
    if has_queues:
        num_plots += 1

    # Create figure
    fig_height = 2.5 * num_plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, fig_height), sharex=True)

    if num_plots == 1:
        axes = [axes]

    # Get all times to determine time unit and calculate GPU time
    all_times = gpu_data.get('_all_times', np.array([]))
    if len(all_times) == 0 and gpu_data:
        # Fallback: get times from first available metric
        for key, value in gpu_data.items():
            if isinstance(value, dict) and 'time' in value:
                all_times = value['time']
                break
    
    # Calculate GPU monitoring time span
    gpu_time = 0.0
    if len(all_times) > 0:
        gpu_time = all_times[-1] - all_times[0] if len(all_times) > 1 else 0.0

    # Title with experiment configuration
    model_name = config.get('model', 'Unknown').split('/')[-1]
    title_parts = [model_name]
    title_parts.append(f"TP={config.get('tp', '?')}, PP={config.get('pp', '?')}, Input={config.get('max_input_length', '?')}, Output={config.get('max_output_length', '?')}")
    title_parts.append(f"Instance={instance_type}, GPU={gpu_type}")
    title_parts.append(f"Elapsed={elapsed:.1f}s, GPU Time={gpu_time:.1f}s")
    title = "\n".join(title_parts)
    fig.suptitle(title, y=0.995)

    ax_idx = 0
    
    # Convert time to minutes if > 120 seconds
    time_unit = 's'
    time_divisor = 1
    if len(all_times) > 0 and all_times[-1] > 120:
        time_unit = 'min'
        time_divisor = 60

    # --- Plot 1: SM Utilization ---
    ax = axes[ax_idx]
    for idx, gpu_id in enumerate(gpu_ids):
        key = f'{gpu_id}_sm_pct'
        if key in gpu_data and isinstance(gpu_data[key], dict):
            metric_data = gpu_data[key]
            time_arr = metric_data['time']
            values = metric_data['value']
            time_plot = time_arr / time_divisor
            
            # Create readable label: "Node0:GPU0" for node0_gpu0, or "GPU 0" for gpu0
            # Get marker based on node number
            if gpu_id.startswith('node'):
                parts = gpu_id.split('_')
                node_num = int(parts[0][4:])
                gpu_num = int(parts[1][3:])
                label = f"Node{node_num}:GPU{gpu_num}"
                marker = MARKERS[node_num % len(MARKERS)]
            else:
                label = f"GPU {gpu_id[3:]}"
                marker = MARKERS[0]  # Default marker for single-node
            
            ax.plot(time_plot, values,
                   color=COLORS.get(f'gpu{idx % 8}', f'C{idx}'),
                   label=label, alpha=0.8, marker=marker, markersize=3, markevery=max(1, len(time_plot)//50))

    ax.set_ylabel('SM Utilization (%)')
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, framealpha=0.9)
    ax_idx += 1

    # --- Plot 2: Memory Bandwidth Utilization ---
    ax = axes[ax_idx]
    for idx, gpu_id in enumerate(gpu_ids):
        key = f'{gpu_id}_membw_pct'
        if key in gpu_data and isinstance(gpu_data[key], dict):
            metric_data = gpu_data[key]
            time_arr = metric_data['time']
            values = metric_data['value']
            time_plot = time_arr / time_divisor
            
            if gpu_id.startswith('node'):
                parts = gpu_id.split('_')
                node_num = int(parts[0][4:])
                gpu_num = int(parts[1][3:])
                label = f"Node{node_num}:GPU{gpu_num}"
                marker = MARKERS[node_num % len(MARKERS)]
            else:
                label = f"GPU {gpu_id[3:]}"
                marker = MARKERS[0]  # Default marker for single-node
            
            ax.plot(time_plot, values,
                   color=COLORS.get(f'gpu{idx % 8}', f'C{idx}'),
                   label=label, alpha=0.8, marker=marker, markersize=3, markevery=max(1, len(time_plot)//50))

    ax.set_ylabel('Memory BW Util. (%)')
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, framealpha=0.9)
    ax_idx += 1

    # --- Plot 3: Memory Usage (GB) ---
    ax = axes[ax_idx]
    all_mem = []
    for idx, gpu_id in enumerate(gpu_ids):
        key = f'{gpu_id}_mem_gb'
        if key in gpu_data and isinstance(gpu_data[key], dict):
            metric_data = gpu_data[key]
            time_arr = metric_data['time']
            values = metric_data['value']
            time_plot = time_arr / time_divisor
            all_mem.extend(values)
            
            if gpu_id.startswith('node'):
                parts = gpu_id.split('_')
                node_num = int(parts[0][4:])
                gpu_num = int(parts[1][3:])
                label = f"Node{node_num}:GPU{gpu_num}"
                marker = MARKERS[node_num % len(MARKERS)]
            else:
                label = f"GPU {gpu_id[3:]}"
                marker = MARKERS[0]  # Default marker for single-node
            
            ax.plot(time_plot, values,
                   color=COLORS.get(f'gpu{idx % 8}', f'C{idx}'),
                   label=label, alpha=0.8, marker=marker, markersize=3, markevery=max(1, len(time_plot)//50))

    ax.set_ylabel('Memory Usage (GB)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, framealpha=0.9)

    # Set reasonable y-limits for memory
    if all_mem:
        ax.set_ylim(0, max(all_mem) * 1.1)
    ax_idx += 1

    # --- Plot 4: KV Cache Utilization (if available) ---
    if has_kv_cache:
        ax = axes[ax_idx]
        sched_time = scheduler_data.get('time', np.array([]))
        sched_time_plot = sched_time / time_divisor

        ax.plot(sched_time_plot, scheduler_data['kv_cache_util_pct'],
               color='#E69F00', linewidth=1.5, label='KV Cache Util.')
        ax.fill_between(sched_time_plot, 0, scheduler_data['kv_cache_util_pct'],
                       alpha=0.3, color='#E69F00')

        ax.set_ylabel('KV Cache Util. (%)')
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)
        ax_idx += 1

    # --- Plot 5: Scheduler Queues (if available) ---
    if has_queues:
        ax = axes[ax_idx]
        sched_time = scheduler_data.get('time', np.array([]))
        sched_time_plot = sched_time / time_divisor

        if 'running' in scheduler_data:
            ax.plot(sched_time_plot, scheduler_data['running'],
                   color=COLORS['running'], label='Running', linewidth=1.5)
        if 'waiting' in scheduler_data:
            ax.plot(sched_time_plot, scheduler_data['waiting'],
                   color=COLORS['waiting'], label='Waiting', linewidth=1.5)
        if 'swapped' in scheduler_data:
            ax.plot(sched_time_plot, scheduler_data['swapped'],
                   color=COLORS['swapped'], label='Swapped', linewidth=1.5)

        ax.set_ylabel('Queue Depth')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)
        ax_idx += 1

    # Set x-axis label on bottom plot
    axes[-1].set_xlabel(f'Time ({time_unit})')

    # Adjust layout - increase spacing between subplots to accommodate legends on top
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, hspace=0.55)

    # Save figure
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_all_experiments(experiments: list[dict], output_path: Path):
    """Create a combined figure with all experiments, or individual files."""

    if len(experiments) == 1:
        plot_experiment(experiments[0], output_path)
        return

    # Multiple experiments: create one PDF per experiment
    output_dir = output_path.parent
    base_name = output_path.stem

    for exp in experiments:
        exp_id = exp.get('exp_id', 'unknown')
        exp_output = output_dir / f"{base_name}_{exp_id}.pdf"
        plot_experiment(exp, exp_output)

    print(f"\nGenerated {len(experiments)} PDF files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot GPU and scheduler time-series from vLLM benchmarks')
    parser.add_argument('input', type=Path,
                       help='Timeseries JSON file or directory containing timeseries files')
    parser.add_argument('--output', '-o', type=Path, default=None,
                       help='Output PDF path (default: timeseries_plot.pdf)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1

    # Default output path
    if args.output is None:
        if args.input.is_file():
            args.output = args.input.with_suffix('.pdf')
        else:
            args.output = args.input / 'timeseries_plot.pdf'

    # Load data
    experiments = load_timeseries(args.input)

    if not experiments:
        print(f"Error: No timeseries data found in {args.input}")
        return 1

    print(f"Loaded {len(experiments)} experiment(s)")

    # Plot
    plot_all_experiments(experiments, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
