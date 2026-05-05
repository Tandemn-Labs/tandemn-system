#!/usr/bin/env python3
"""Tests for prometheus_parser.py"""

from prometheus_parser import (
    parse_prometheus_text,
    histogram_quantile,
    compute_deltas,
    extract_throughput_metrics,
    extract_latency_percentiles,
)

SAMPLE_METRICS = """\
# HELP vllm:prompt_tokens_total Total prompt tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 50000.0

# HELP vllm:generation_tokens_total Total generation tokens
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 12000.0

# HELP vllm:num_preemptions_total Total preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 3.0

# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.45

# HELP vllm:num_requests_running Running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 10.0

# HELP vllm:e2e_request_latency_seconds End to end latency
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 5
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 15
vllm:e2e_request_latency_seconds_bucket{le="2.0"} 25
vllm:e2e_request_latency_seconds_bucket{le="5.0"} 30
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 30
vllm:e2e_request_latency_seconds_sum 42.5
vllm:e2e_request_latency_seconds_count 30

# HELP vllm:time_to_first_token_seconds TTFT
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 10
vllm:time_to_first_token_seconds_bucket{le="0.5"} 25
vllm:time_to_first_token_seconds_bucket{le="1.0"} 30
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 30
vllm:time_to_first_token_seconds_sum 8.5
vllm:time_to_first_token_seconds_count 30

# HELP vllm:time_per_output_token_seconds TPOT
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.01"} 5
vllm:time_per_output_token_seconds_bucket{le="0.05"} 20
vllm:time_per_output_token_seconds_bucket{le="0.1"} 28
vllm:time_per_output_token_seconds_bucket{le="0.5"} 30
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 30
vllm:time_per_output_token_seconds_sum 1.2
vllm:time_per_output_token_seconds_count 30
"""

WARMUP_METRICS = """\
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 1000.0

# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 200.0

# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 0.0

# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.05

# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 0.0

# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 2
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 4
vllm:e2e_request_latency_seconds_bucket{le="2.0"} 5
vllm:e2e_request_latency_seconds_bucket{le="5.0"} 5
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 5
vllm:e2e_request_latency_seconds_sum 3.2
vllm:e2e_request_latency_seconds_count 5

# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 3
vllm:time_to_first_token_seconds_bucket{le="0.5"} 5
vllm:time_to_first_token_seconds_bucket{le="1.0"} 5
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 5
vllm:time_to_first_token_seconds_sum 1.0
vllm:time_to_first_token_seconds_count 5

# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.01"} 1
vllm:time_per_output_token_seconds_bucket{le="0.05"} 4
vllm:time_per_output_token_seconds_bucket{le="0.1"} 5
vllm:time_per_output_token_seconds_bucket{le="0.5"} 5
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 5
vllm:time_per_output_token_seconds_sum 0.15
vllm:time_per_output_token_seconds_count 5
"""


def test_parse():
    parsed = parse_prometheus_text(SAMPLE_METRICS)
    
    assert parsed["counters"]["vllm:prompt_tokens_total"] == 50000.0, f"Got {parsed['counters']}"
    assert parsed["counters"]["vllm:generation_tokens_total"] == 12000.0
    assert parsed["counters"]["vllm:num_preemptions_total"] == 3.0
    
    assert parsed["gauges"]["vllm:gpu_cache_usage_perc"] == 0.45
    assert parsed["gauges"]["vllm:num_requests_running"] == 10.0
    
    e2e = parsed["histograms"]["vllm:e2e_request_latency_seconds"]
    assert e2e["sum"] == 42.5
    assert e2e["count"] == 30
    assert len(e2e["buckets"]) == 5
    assert e2e["buckets"][0] == (0.5, 5)
    assert e2e["buckets"][-1] == (float('inf'), 30)
    
    print("✅ parse_prometheus_text OK")


def test_histogram_quantile():
    # Simple: 10 items in [0,1], 20 items in [1,2]
    buckets = [(1.0, 10), (2.0, 30), (float('inf'), 30)]
    
    p50 = histogram_quantile(buckets, 0.5)
    # target = 15, falls in bucket (1.0, 2.0), count 10->30, fraction=(15-10)/20=0.25
    # result = 1.0 + 0.25 * 1.0 = 1.25
    assert abs(p50 - 1.25) < 0.001, f"p50={p50}, expected 1.25"
    
    p90 = histogram_quantile(buckets, 0.9)
    # target = 27, bucket (1,2), fraction=(27-10)/20=0.85, result=1.0+0.85=1.85
    assert abs(p90 - 1.85) < 0.001, f"p90={p90}"
    
    # Empty
    assert histogram_quantile([], 0.5) is None
    assert histogram_quantile([(1.0, 0), (float('inf'), 0)], 0.5) is None
    
    print("✅ histogram_quantile OK")


def test_compute_deltas():
    warmup = parse_prometheus_text(WARMUP_METRICS)
    final = parse_prometheus_text(SAMPLE_METRICS)
    deltas = compute_deltas(warmup, final)
    
    assert deltas["vllm:prompt_tokens_total"] == 49000.0
    assert deltas["vllm:generation_tokens_total"] == 11800.0
    assert deltas["vllm:num_preemptions_total"] == 3.0
    
    e2e = deltas["vllm:e2e_request_latency_seconds"]
    assert e2e["count"] == 25
    assert abs(e2e["sum"] - 39.3) < 0.01
    # Bucket deltas: (0.5, 5-2=3), (1.0, 15-4=11), ...
    bd = dict(e2e["buckets"])
    assert bd[0.5] == 3
    assert bd[1.0] == 11
    assert bd[2.0] == 20
    
    assert deltas["gauges"]["vllm:gpu_cache_usage_perc"] == 0.45
    
    print("✅ compute_deltas OK")


def test_throughput():
    warmup = parse_prometheus_text(WARMUP_METRICS)
    final = parse_prometheus_text(SAMPLE_METRICS)
    deltas = compute_deltas(warmup, final)
    
    tp = extract_throughput_metrics(deltas, wall_clock_elapsed=100.0)
    assert tp["total_prompt_tokens"] == 49000.0
    assert tp["total_generation_tokens"] == 11800.0
    assert abs(tp["tokens_per_sec_total"] - 608.0) < 0.1
    assert abs(tp["tokens_per_sec_prefill"] - 490.0) < 0.1
    assert abs(tp["tokens_per_sec_decode"] - 118.0) < 0.1
    
    print("✅ extract_throughput_metrics OK")


def test_latency_percentiles():
    warmup = parse_prometheus_text(WARMUP_METRICS)
    final = parse_prometheus_text(SAMPLE_METRICS)
    deltas = compute_deltas(warmup, final)
    
    lat = extract_latency_percentiles(deltas)
    
    # Check that percentiles are populated and reasonable
    for key in ["ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
                "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
                "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99"]:
        assert lat[key] is not None, f"{key} is None"
        assert lat[key] > 0, f"{key} = {lat[key]}"
    
    # TTFT p50 should be < 500ms (most in 0-0.5s range)
    assert lat["ttft_ms_p50"] < 500, f"ttft_ms_p50={lat['ttft_ms_p50']}"
    # E2E p50 should be around 0.5-1.0s range
    assert 200 < lat["e2e_ms_p50"] < 2000, f"e2e_ms_p50={lat['e2e_ms_p50']}"
    
    print(f"✅ extract_latency_percentiles OK")
    for k, v in sorted(lat.items()):
        print(f"   {k}: {v}")


if __name__ == "__main__":
    test_parse()
    test_histogram_quantile()
    test_compute_deltas()
    test_throughput()
    test_latency_percentiles()
    print("\n🎉 All tests passed!")
