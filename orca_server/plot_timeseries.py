#!/usr/bin/env python3
"""
Generate publication-quality PDF figures from Orca timeseries.csv files.

Usage:
    python -m orca_server.plot_timeseries outputs/.../timeseries.csv
    python -m orca_server.plot_timeseries outputs/.../timeseries.csv --metrics outputs/.../metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette
# ---------------------------------------------------------------------------
COLORS = {
    'gen_throughput': '#C73E1D',
    'prompt_throughput': '#2E8B57',
    'kv_cache': '#E69F00',
    'running': '#0072B2',
    'waiting': '#D55E00',
    'swapped': '#009E73',
    'sm_util': '#0072B2',
    'membw_util': '#D55E00',
    'ttft': '#0072B2',
    'tpot': '#D55E00',
    'e2e': '#009E73',
    'completions': '#2E8B57',
    'preemptions': '#C73E1D',
}


def _apply_style():
    """Apply publication-quality rcParams (call after importing matplotlib)."""
    import matplotlib.pyplot as plt

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


def _safe_float(val, default=float('nan')):
    """Convert a value to float, returning *default* for None / empty / non-numeric."""
    if val is None or val == '' or val == 'None':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _load_csv(path: str) -> list[dict]:
    """Load a CSV file as a list of dicts."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_metrics_summary(path: str) -> dict:
    """Load a metrics.csv (metric,value format) into a dict."""
    result = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get('metric', '')
            val = row.get('value', '')
            result[key] = val
    return result


def _column_has_data(rows: list[dict], col: str) -> bool:
    """Return True if *col* exists and has at least one non-zero, non-NaN value."""
    for row in rows:
        v = _safe_float(row.get(col))
        if not math.isnan(v) and v != 0.0:
            return True
    return False


def plot_timeseries(
    csv_path: str,
    output_path: Optional[str] = None,
    metrics_csv_path: Optional[str] = None,
) -> str:
    """Generate a multi-panel timeseries PDF from timeseries.csv.

    Args:
        csv_path: Path to timeseries.csv.
        output_path: Output PDF path. Defaults to timeseries.pdf in the same
            directory as the CSV.
        metrics_csv_path: Optional path to metrics.csv for a summary text panel.

    Returns:
        The path to the generated PDF.
    """
    # Late imports so the module is importable even without matplotlib.
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "matplotlib and numpy are required for plotting. "
            "Install with: pip install matplotlib numpy"
        )

    _apply_style()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    csv_p = Path(csv_path)
    if output_path is None:
        output_path = str(csv_p.with_name('timeseries.pdf'))

    rows = _load_csv(csv_path)
    if not rows:
        logger.warning("timeseries.csv is empty — skipping plot generation")
        return output_path

    # ------------------------------------------------------------------
    # Parse columns into numpy arrays
    # ------------------------------------------------------------------
    timestamps = np.array([_safe_float(r.get('timestamp', 0)) for r in rows])
    t0 = np.nanmin(timestamps)
    t_rel = timestamps - t0  # seconds since first sample

    # Filter out pre-warmup rows where all interesting metrics are zero
    _interesting = [
        'avg_generation_throughput_toks_per_s',
        'avg_prompt_throughput_toks_per_s',
        'num_requests_running',
        'num_requests_waiting',
    ]
    keep_mask = np.zeros(len(rows), dtype=bool)
    for col in _interesting:
        vals = np.array([_safe_float(r.get(col)) for r in rows])
        keep_mask |= (~np.isnan(vals)) & (vals != 0.0)
    # Always keep all rows if filtering would remove everything
    if np.any(keep_mask):
        idx = np.where(keep_mask)[0]
        rows = [rows[i] for i in idx]
        t_rel = t_rel[idx]

    n = len(rows)
    if n == 0:
        logger.warning("All rows filtered out — skipping plot generation")
        return output_path

    # Decide time unit
    time_unit = 's'
    time_divisor = 1.0
    if t_rel[-1] > 120:
        time_unit = 'min'
        time_divisor = 60.0
    t_plot = t_rel / time_divisor

    # Helper to extract a column as float array
    def col(name):
        return np.array([_safe_float(r.get(name)) for r in rows])

    # ------------------------------------------------------------------
    # Determine which panels have data
    # ------------------------------------------------------------------
    panel_specs = []  # (panel_id, title)

    # 1. Throughput
    gen_tp = col('avg_generation_throughput_toks_per_s')
    prompt_tp = col('avg_prompt_throughput_toks_per_s')
    if np.any(~np.isnan(gen_tp)) or np.any(~np.isnan(prompt_tp)):
        panel_specs.append('throughput')

    # 2. KV Cache
    kv_raw = col('gpu_cache_usage_perc')
    kv_pct = kv_raw * 100.0  # convert 0-1 to 0-100
    if np.any(~np.isnan(kv_raw)):
        panel_specs.append('kv_cache')

    # 3. Scheduler Queues
    running = col('num_requests_running')
    waiting = col('num_requests_waiting')
    swapped = col('num_requests_swapped')
    if np.any(~np.isnan(running)):
        panel_specs.append('scheduler')

    # 4. GPU Utilization
    sm_util = col('gpu_sm_util_pct')
    membw_util = col('gpu_mem_bw_util_pct')
    if np.any(~np.isnan(sm_util)) or np.any(~np.isnan(membw_util)):
        panel_specs.append('gpu_util')

    # 5. TTFT
    ttft_p50 = col('ttft_ms_p50')
    ttft_p95 = col('ttft_ms_p95')
    ttft_p99 = col('ttft_ms_p99')
    if np.any(~np.isnan(ttft_p50)):
        panel_specs.append('ttft')

    # 6. TPOT
    tpot_p50 = col('tpot_ms_p50')
    tpot_p95 = col('tpot_ms_p95')
    tpot_p99 = col('tpot_ms_p99')
    if np.any(~np.isnan(tpot_p50)):
        panel_specs.append('tpot')

    # 7. E2E
    e2e_p50 = col('e2e_ms_p50')
    e2e_p95 = col('e2e_ms_p95')
    e2e_p99 = col('e2e_ms_p99')
    if np.any(~np.isnan(e2e_p50)):
        panel_specs.append('e2e')

    # 8. Completions
    completions = col('request_success_total')
    preemptions = col('num_preemptions_total')
    if np.any(~np.isnan(completions)):
        panel_specs.append('completions')

    # Optional summary panel
    summary = None
    if metrics_csv_path and Path(metrics_csv_path).exists():
        try:
            summary = _load_metrics_summary(metrics_csv_path)
            if summary:
                panel_specs.append('summary')
        except Exception:
            pass

    if not panel_specs:
        logger.warning("No plottable data found — skipping plot generation")
        return output_path

    # ------------------------------------------------------------------
    # Precompute KV-full regions for cross-panel shading
    # ------------------------------------------------------------------
    kv_full_regions = []
    if 'kv_cache' in panel_specs:
        kv_full_mask = kv_pct >= 99.0
        in_region = False
        region_start = None
        for i in range(len(kv_full_mask)):
            if kv_full_mask[i] and not in_region:
                region_start = t_plot[i]
                in_region = True
            elif not kv_full_mask[i] and in_region:
                kv_full_regions.append((region_start, t_plot[i]))
                in_region = False
        if in_region:
            kv_full_regions.append((region_start, t_plot[-1]))

    def shade_kv_full(ax):
        for (a, b) in kv_full_regions:
            ax.axvspan(a, b, alpha=0.10, color='#DC143C', zorder=0)

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    num_panels = len(panel_specs)
    fig_height = 2.8 * num_panels
    fig, axes = plt.subplots(num_panels, 1, figsize=(10, fig_height), sharex=True)
    if num_panels == 1:
        axes = [axes]

    ax_map = {pid: axes[i] for i, pid in enumerate(panel_specs)}

    # ------------------------------------------------------------------
    # Panel 1: Throughput
    # ------------------------------------------------------------------
    if 'throughput' in ax_map:
        ax = ax_map['throughput']
        shade_kv_full(ax)

        valid_gen = ~np.isnan(gen_tp)
        valid_prompt = ~np.isnan(prompt_tp)

        if np.any(valid_gen):
            ax.scatter(t_plot[valid_gen], gen_tp[valid_gen],
                       color=COLORS['gen_throughput'], alpha=0.45, s=18,
                       label='Generation tok/s', zorder=2)
            active = gen_tp[valid_gen & (gen_tp > 0)]
            if len(active) > 0:
                mean_gen = np.mean(active)
                ax.axhline(mean_gen, color=COLORS['gen_throughput'],
                           linestyle='--', alpha=0.5, linewidth=1.0)
                ax.text(t_plot[-1], mean_gen, f'  avg {mean_gen:.0f}',
                        fontsize=8, color=COLORS['gen_throughput'], va='center')

        if np.any(valid_prompt):
            ax.scatter(t_plot[valid_prompt], prompt_tp[valid_prompt],
                       color=COLORS['prompt_throughput'], alpha=0.40, s=15,
                       label='Prompt tok/s', zorder=2)

        ax.set_title('Throughput (tok/s)')
        ax.set_ylabel('Throughput (tok/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 2: KV Cache Utilization
    # ------------------------------------------------------------------
    if 'kv_cache' in ax_map:
        ax = ax_map['kv_cache']
        valid = ~np.isnan(kv_pct)
        if np.any(valid):
            ax.plot(t_plot[valid], kv_pct[valid],
                    color=COLORS['kv_cache'], linewidth=1.5, label='KV Cache Util.')
            ax.fill_between(t_plot[valid], 0, kv_pct[valid],
                            alpha=0.3, color=COLORS['kv_cache'])
        # Shade preemption zones
        for (a, b) in kv_full_regions:
            ax.axvspan(a, b, alpha=0.15, color='#DC143C', zorder=0)
        ax.set_title('KV Cache Utilization (%)')
        ax.set_ylabel('KV Cache Util. (%)')
        ax.set_ylim(-2, 105)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 3: Scheduler Queue Depth
    # ------------------------------------------------------------------
    if 'scheduler' in ax_map:
        ax = ax_map['scheduler']
        shade_kv_full(ax)
        for arr, name, color_key in [
            (running, 'Running', 'running'),
            (waiting, 'Waiting', 'waiting'),
            (swapped, 'Swapped', 'swapped'),
        ]:
            valid = ~np.isnan(arr)
            if np.any(valid):
                ax.plot(t_plot[valid], arr[valid],
                        color=COLORS[color_key], linewidth=1.5, label=name)
        ax.set_title('Scheduler Queue Depth')
        ax.set_ylabel('Queue Depth')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 4: GPU Utilization
    # ------------------------------------------------------------------
    if 'gpu_util' in ax_map:
        ax = ax_map['gpu_util']
        shade_kv_full(ax)
        valid_sm = ~np.isnan(sm_util)
        valid_bw = ~np.isnan(membw_util)
        if np.any(valid_sm):
            ax.scatter(t_plot[valid_sm], sm_util[valid_sm],
                       color=COLORS['sm_util'], alpha=0.45, s=18,
                       label='SM Util %', zorder=2)
        if np.any(valid_bw):
            ax.scatter(t_plot[valid_bw], membw_util[valid_bw],
                       color=COLORS['membw_util'], alpha=0.40, s=15,
                       label='Mem BW Util %', zorder=2)
        ax.set_title('GPU Utilization (%)')
        ax.set_ylabel('Utilization (%)')
        ax.set_ylim(-2, 105)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 5: TTFT
    # ------------------------------------------------------------------
    if 'ttft' in ax_map:
        ax = ax_map['ttft']
        shade_kv_full(ax)
        base_color = COLORS['ttft']
        for arr, style, label in [
            (ttft_p50, '-', 'p50'),
            (ttft_p95, '--', 'p95'),
            (ttft_p99, ':', 'p99'),
        ]:
            valid = ~np.isnan(arr)
            if np.any(valid):
                ax.plot(t_plot[valid], arr[valid],
                        color=base_color, linestyle=style, linewidth=1.5,
                        label=label)
        ax.set_title('Time to First Token (ms)')
        ax.set_ylabel('TTFT (ms)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 6: TPOT
    # ------------------------------------------------------------------
    if 'tpot' in ax_map:
        ax = ax_map['tpot']
        shade_kv_full(ax)
        base_color = COLORS['tpot']
        for arr, style, label in [
            (tpot_p50, '-', 'p50'),
            (tpot_p95, '--', 'p95'),
            (tpot_p99, ':', 'p99'),
        ]:
            valid = ~np.isnan(arr)
            if np.any(valid):
                ax.plot(t_plot[valid], arr[valid],
                        color=base_color, linestyle=style, linewidth=1.5,
                        label=label)
        ax.set_title('Time Per Output Token (ms)')
        ax.set_ylabel('TPOT (ms)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 7: E2E Latency
    # ------------------------------------------------------------------
    if 'e2e' in ax_map:
        ax = ax_map['e2e']
        shade_kv_full(ax)
        base_color = COLORS['e2e']
        for arr, style, label in [
            (e2e_p50, '-', 'p50'),
            (e2e_p95, '--', 'p95'),
            (e2e_p99, ':', 'p99'),
        ]:
            valid = ~np.isnan(arr)
            if np.any(valid):
                ax.plot(t_plot[valid], arr[valid],
                        color=base_color, linestyle=style, linewidth=1.5,
                        label=label)
        ax.set_title('End-to-End Latency (ms)')
        ax.set_ylabel('E2E (ms)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9)

    # ------------------------------------------------------------------
    # Panel 8: Cumulative Completions & Preemptions
    # ------------------------------------------------------------------
    if 'completions' in ax_map:
        ax = ax_map['completions']
        shade_kv_full(ax)
        valid_c = ~np.isnan(completions)
        valid_p = ~np.isnan(preemptions)
        if np.any(valid_c):
            ax.plot(t_plot[valid_c], completions[valid_c],
                    color=COLORS['completions'], linewidth=1.5,
                    label='Completions')
        if np.any(valid_p):
            ax.plot(t_plot[valid_p], preemptions[valid_p],
                    color=COLORS['preemptions'], linewidth=1.5,
                    label='Preemptions')
        ax.set_title('Cumulative Completions & Preemptions')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)

    # ------------------------------------------------------------------
    # Summary text panel
    # ------------------------------------------------------------------
    if 'summary' in ax_map and summary:
        ax = ax_map['summary']
        ax.axis('off')

        def _sv(key, fmt='.1f', fallback='N/A'):
            v = summary.get(key, '')
            if v == '' or v == 'None':
                return fallback
            try:
                return f'{float(v):{fmt}}'
            except (ValueError, TypeError):
                return str(v)

        sections = [
            ('Throughput', [
                ('Total tok/s', _sv('total_tokens_per_sec', '.1f')),
                ('Prefill tok/s', _sv('input_tokens_per_sec', '.1f')),
                ('Decode tok/s', _sv('output_tokens_per_sec', '.1f')),
            ]),
            ('Cost', [
                ('Cost', f'${_sv("cost_for_run_usd", ".4f")}'),
                ('Tok/dollar', _sv('tokens_per_dollar', ',.0f')),
            ]),
            ('Runtime', [
                ('Elapsed', _sv('elapsed_time', '.1f') + 's'),
                ('Requests', _sv('request_success_total', '.0f')),
            ]),
            ('GPU Util (avg)', [
                ('SM util', _sv('avg_sm_util_pct', '.1f') + '%'),
                ('Mem BW util', _sv('avg_mem_bw_util_pct', '.1f') + '%'),
            ]),
        ]

        col_x = [0.02, 0.26, 0.50, 0.74]
        y_top = 0.85
        line_h = 0.20

        for col_idx, (section_title, items) in enumerate(sections):
            cx = col_x[col_idx]
            ax.text(cx, y_top, section_title, fontsize=8, fontweight='bold',
                    color='#1a1a1a', transform=ax.transAxes)
            for i, (label, value) in enumerate(items):
                y = y_top - (i + 1) * line_h
                ax.text(cx + 0.01, y, f'{label}:', fontsize=7,
                        color='#555555', transform=ax.transAxes)
                ax.text(cx + 0.13, y, value, fontsize=7, fontweight='bold',
                        color='#1a1a1a', transform=ax.transAxes)

        ax.set_title('Summary', fontsize=9, fontweight='bold', loc='left', pad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
        ax.set_facecolor('#f8f8f8')

    # ------------------------------------------------------------------
    # Shared x-axis label on the bottom panel
    # ------------------------------------------------------------------
    axes[-1].set_xlabel(f'Time ({time_unit})')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.85)

    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved timeseries plot to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality PDF from Orca timeseries.csv')
    parser.add_argument('csv_path', type=str,
                        help='Path to timeseries.csv')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output PDF path (default: timeseries.pdf next to CSV)')
    parser.add_argument('--metrics', '-m', type=str, default=None,
                        help='Path to metrics.csv for summary panel')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not Path(args.csv_path).exists():
        logger.error(f"File not found: {args.csv_path}")
        return 1

    try:
        out = plot_timeseries(args.csv_path, args.output, args.metrics)
        print(f"Generated: {out}")
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
