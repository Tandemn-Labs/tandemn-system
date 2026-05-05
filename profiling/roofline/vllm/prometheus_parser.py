"""
prometheus_parser.py — Parse Prometheus text exposition format from vLLM /metrics endpoint.

Standalone module, stdlib only (re, collections).
"""

import re
from collections import defaultdict


def parse_prometheus_text(text):
    """Parse Prometheus text exposition format into structured data.

    Returns:
        {
            "counters": {"metric_name": float, ...},
            "gauges": {"metric_name": float, ...},
            "histograms": {
                "metric_base_name": {
                    "sum": float,
                    "count": float,
                    "buckets": [(bound, cumulative_count), ...]  # sorted by bound
                }, ...
            }
        }
    """
    types = {}  # metric_name -> "counter"|"gauge"|"histogram"|"summary"|"untyped"
    result = {"counters": {}, "gauges": {}, "histograms": {}}
    # Temporary accumulator for histogram buckets
    hist_buckets = defaultdict(list)

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("# HELP"):
            continue
        if line.startswith("# TYPE "):
            parts = line.split(None, 3)
            if len(parts) >= 4:
                types[parts[2]] = parts[3]
            continue
        if line.startswith("#"):
            continue

        # Parse: name{labels} value [timestamp]
        # Handle _bucket{le="..."}, _sum, _count suffixes for histograms
        m = re.match(r'^(\S+?)\s+([\d.eE+\-NnAaIiFf]+)', line)
        if not m:
            continue
        full_name = m.group(1)
        try:
            value = float(m.group(2))
        except ValueError:
            continue

        # Check if it's a histogram bucket
        # Handle labels: _bucket{le="0.3"} or _bucket{le="0.3",model_name="..."}
        bucket_match = re.match(r'^(.+)_bucket\{(?:.*,\s*)?le="([^"]+)"(?:,.*)?\}$', full_name)
        if not bucket_match:
            # Try alternate label order: le might not be first
            bucket_match = re.match(r'^(.+)_bucket\{le="([^"]+)".*\}$', full_name)
        if bucket_match:
            base = bucket_match.group(1)
            le_str = bucket_match.group(2)
            le_val = float('inf') if le_str == '+Inf' else float(le_str)
            hist_buckets[base].append((le_val, value))
            # Ensure histogram entry exists
            if base not in result["histograms"]:
                result["histograms"][base] = {"sum": 0.0, "count": 0.0, "buckets": []}
            continue

        # Check histogram _sum / _count
        sum_match = re.match(r'^(.+)_sum(?:\{.*\})?$', full_name)
        if sum_match:
            base = sum_match.group(1)
            if base in types and types[base] == "histogram" or base in hist_buckets:
                if base not in result["histograms"]:
                    result["histograms"][base] = {"sum": 0.0, "count": 0.0, "buckets": []}
                result["histograms"][base]["sum"] = value
                continue

        count_match = re.match(r'^(.+)_count(?:\{.*\})?$', full_name)
        if count_match:
            base = count_match.group(1)
            if base in types and types[base] == "histogram" or base in hist_buckets:
                if base not in result["histograms"]:
                    result["histograms"][base] = {"sum": 0.0, "count": 0.0, "buckets": []}
                result["histograms"][base]["count"] = value
                continue

        # Strip labels for plain metrics
        plain_name = re.sub(r'\{[^}]*\}', '', full_name)

        # Determine type
        mtype = types.get(plain_name, "untyped")
        if mtype == "counter":
            # Use full_name to distinguish label variants, but for simplicity use plain_name
            # If there are labels, use full_name; otherwise plain_name
            key = full_name if '{' in full_name else plain_name
            result["counters"][key] = value
        elif mtype == "gauge":
            key = full_name if '{' in full_name else plain_name
            result["gauges"][key] = value
        else:
            # Untyped or summary — treat as gauge
            key = full_name if '{' in full_name else plain_name
            result["gauges"][key] = value

    # Attach sorted buckets to histograms
    for base, blist in hist_buckets.items():
        if base in result["histograms"]:
            result["histograms"][base]["buckets"] = sorted(blist, key=lambda x: x[0])

    return result


def histogram_quantile(buckets, quantile):
    """Estimate a quantile from sorted (bound, cumulative_count) pairs.

    Uses linear interpolation within the bucket where the quantile falls,
    matching the standard Prometheus histogram_quantile algorithm.

    Args:
        buckets: list of (upper_bound, cumulative_count) sorted by upper_bound
        quantile: float in [0, 1]

    Returns:
        Estimated value at the given quantile, or None if not computable.
    """
    if not buckets:
        return None
    total = buckets[-1][1]  # Last bucket (+Inf) has total count
    if total == 0:
        return None
    target = quantile * total
    prev_bound = 0.0
    prev_count = 0.0
    for bound, count in buckets:
        if bound == float('inf'):
            if count >= target and prev_count < target:
                return prev_bound
            continue
        if count >= target:
            bucket_count = count - prev_count
            if bucket_count == 0:
                return bound
            fraction = (target - prev_count) / bucket_count
            return prev_bound + fraction * (bound - prev_bound)
        prev_bound = bound
        prev_count = count
    return prev_bound


def compute_deltas(warmup_parsed, final_parsed):
    """Compute metric deltas between two /metrics scrapes.

    - Counters: final - warmup
    - Histograms: sum/count/bucket counts subtracted
    - Gauges: final values used as-is

    Returns dict with:
        - counter keys -> float delta
        - histogram keys -> {"sum", "count", "avg", "buckets": [(bound, delta_count)]}
        - "gauges" -> dict of final gauge values
    """
    deltas = {}

    # Counter deltas
    for name in final_parsed["counters"]:
        deltas[name] = final_parsed["counters"][name] - warmup_parsed["counters"].get(name, 0.0)

    # Histogram deltas
    for name in final_parsed["histograms"]:
        fh = final_parsed["histograms"][name]
        wh = warmup_parsed.get("histograms", {}).get(name, {"sum": 0.0, "count": 0.0, "buckets": []})

        delta_sum = fh["sum"] - wh["sum"]
        delta_count = fh["count"] - wh["count"]

        wb_dict = {b: c for b, c in wh["buckets"]}
        delta_buckets = [(b, c - wb_dict.get(b, 0.0)) for b, c in fh["buckets"]]

        deltas[name] = {
            "sum": delta_sum,
            "count": delta_count,
            "avg": delta_sum / delta_count if delta_count > 0 else None,
            "buckets": delta_buckets,
        }

    # Gauges: use final values
    deltas["gauges"] = dict(final_parsed["gauges"])

    return deltas


def _find_metric(deltas, base_name, default=0.0):
    """Find a metric value by base name, ignoring label suffixes.
    
    Handles both 'vllm:prompt_tokens_total' and 
    'vllm:prompt_tokens_total{model_name="..."}' style keys.
    """
    # Exact match first
    if base_name in deltas:
        return deltas[base_name]
    # Search for key starting with base_name followed by { or end
    for key, val in deltas.items():
        stripped = key.split("{")[0] if "{" in key else key
        if stripped == base_name:
            return val
    return default


def extract_throughput_metrics(deltas, wall_clock_elapsed):
    """Compute throughput metrics from counter deltas and wall-clock time.

    Token counts come from vLLM server-side counters:
        vllm:prompt_tokens_total, vllm:generation_tokens_total

    Returns dict with tokens_per_sec_total, tokens_per_sec_prefill, tokens_per_sec_decode.
    """
    prompt_toks = _find_metric(deltas, "vllm:prompt_tokens_total", 0.0)
    gen_toks = _find_metric(deltas, "vllm:generation_tokens_total", 0.0)

    if wall_clock_elapsed <= 0:
        return {
            "tokens_per_sec_total": 0.0,
            "tokens_per_sec_prefill": 0.0,
            "tokens_per_sec_decode": 0.0,
            "total_prompt_tokens": prompt_toks,
            "total_generation_tokens": gen_toks,
        }

    preemptions = _find_metric(deltas, "vllm:num_preemptions_total", 0.0)

    return {
        "tokens_per_sec_total": round((prompt_toks + gen_toks) / wall_clock_elapsed, 2),
        "tokens_per_sec_prefill": round(prompt_toks / wall_clock_elapsed, 2),
        "tokens_per_sec_decode": round(gen_toks / wall_clock_elapsed, 2),
        "total_prompt_tokens": prompt_toks,
        "total_generation_tokens": gen_toks,
        "num_preemptions": int(preemptions),
    }


def extract_latency_percentiles(deltas):
    """Compute latency percentiles from histogram bucket deltas.

    Extracts p50/p95/p99 for:
        - TTFT (vllm:time_to_first_token_seconds)
        - TPOT (vllm:time_per_output_token_seconds)
        - E2E  (vllm:e2e_request_latency_seconds)

    Returns dict with keys like ttft_ms_p50, tpot_ms_p95, e2e_ms_p99, etc.
    """
    result = {}
    metric_map = {
        "ttft": "vllm:time_to_first_token_seconds",
        "tpot": "vllm:time_per_output_token_seconds",
        "e2e": "vllm:e2e_request_latency_seconds",
    }
    quantiles = [(0.5, "p50"), (0.95, "p95"), (0.99, "p99")]

    for prefix, metric_name in metric_map.items():
        hist_delta = _find_metric(deltas, metric_name, None)
        if hist_delta is None or not isinstance(hist_delta, dict):
            for _, label in quantiles:
                result[f"{prefix}_ms_{label}"] = None
            result[f"avg_{prefix}_ms"] = None
            continue

        buckets = hist_delta.get("buckets", [])
        for q, label in quantiles:
            val = histogram_quantile(buckets, q)
            result[f"{prefix}_ms_{label}"] = round(val * 1000, 3) if val is not None else None

        avg = hist_delta.get("avg")
        result[f"avg_{prefix}_ms"] = round(avg * 1000, 3) if avg is not None else None

    return result
