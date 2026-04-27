#!/usr/bin/env python3
"""
benchmark_client.py — Async HTTP benchmark client for vLLM OpenAI-compatible servers.

Sends requests with sustained concurrency control via asyncio.Semaphore.
Supports streaming SSE for client-side TTFT measurement.

Dependencies: aiohttp, datasets, transformers
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="vLLM async benchmark client")
    p.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    p.add_argument("--model", required=True, help="Model name/path")
    p.add_argument("--input-len", type=int, required=True, help="Target input token length")
    p.add_argument("--output-len", type=int, required=True, help="Target output token length")
    p.add_argument("--num-requests", type=int, default=150, help="Total requests to send")
    p.add_argument("--target-concurrency", type=int, default=50, help="Max concurrent in-flight requests")
    p.add_argument("--warmup-requests", type=int, default=5, help="Warmup requests (results discarded)")
    p.add_argument("--output", default="benchmark_results.json", help="Output JSON path")
    p.add_argument("--dataset", default="emozilla/pg19-test", help="HuggingFace dataset for prompts")
    p.add_argument("--request-timeout", type=float, default=600.0, help="Per-request timeout in seconds")
    return p.parse_args(argv)


def compute_percentiles(values, quantiles=(0.5, 0.95, 0.99)):
    """Compute percentiles from a list of values. Returns dict {q: value}."""
    if not values:
        return {q: None for q in quantiles}
    s = sorted(values)
    n = len(s)
    result = {}
    for q in quantiles:
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        result[q] = s[lo] * (1 - frac) + s[hi] * frac
    return result


def load_prompts(dataset_name, model_name, input_len, num_prompts):
    """Load and tokenize prompts from HuggingFace dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading dataset {dataset_name}...")
    ds = load_dataset(dataset_name, split="test", trust_remote_code=True)

    prompts = []
    for row in ds:
        text = row.get("text", "")
        if not text or len(text) < input_len:  # rough char filter
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < input_len:
            continue
        truncated_ids = token_ids[:input_len]
        prompt_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        prompts.append(prompt_text)
        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        print(f"⚠️  Only found {len(prompts)} prompts with >= {input_len} tokens, "
              f"recycling to reach {num_prompts}")
        while len(prompts) < num_prompts:
            prompts.append(prompts[len(prompts) % max(1, len(prompts) - num_prompts + len(prompts))])

    return prompts


async def send_request(session, semaphore, base_url, model, prompt, output_len, request_id, timeout_s):
    """Send a single streaming request and measure TTFT/E2E."""
    import aiohttp

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "temperature": 0.8,
        "ignore_eos": True,
        "stream": True,
    }

    async with semaphore:
        t_start = time.perf_counter()
        ttft = None
        output_tokens = 0
        prompt_tokens = 0

        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with session.post(
                f"{base_url}/v1/completions",
                json=payload,
                timeout=req_timeout,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {
                        "request_id": request_id,
                        "status": "error",
                        "error": f"HTTP {resp.status}: {error_text[:500]}",
                    }

                # Parse SSE stream
                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    try:
                        chunk = json.loads(data_str)
                        # Extract usage from final chunk if present
                        usage = chunk.get("usage")
                        if usage:
                            output_tokens = usage.get("completion_tokens", output_tokens)
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    except json.JSONDecodeError:
                        pass

            t_end = time.perf_counter()
            e2e = t_end - t_start
            tpot = (e2e - ttft) / max(1, output_tokens - 1) if ttft is not None and output_tokens > 1 else None

            return {
                "request_id": request_id,
                "status": "success",
                "ttft_s": ttft,
                "e2e_s": e2e,
                "tpot_s": tpot,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }

        except asyncio.TimeoutError:
            return {"request_id": request_id, "status": "error", "error": "timeout"}
        except Exception as e:
            return {"request_id": request_id, "status": "error", "error": str(e)}


async def run_benchmark(args, prompts):
    """Run the full benchmark: warmup + measured requests."""
    import aiohttp

    semaphore = asyncio.Semaphore(args.target_concurrency)
    connector = aiohttp.TCPConnector(limit=args.target_concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=None)  # per-request timeout handled individually

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup
        if args.warmup_requests > 0:
            print(f"Running {args.warmup_requests} warmup requests...")
            warmup_tasks = [
                send_request(session, semaphore, args.base_url, args.model,
                             prompts[i % len(prompts)], args.output_len, f"warmup_{i}",
                             args.request_timeout)
                for i in range(args.warmup_requests)
            ]
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            print("Warmup done.")

        # Measured run
        print(f"Running {args.num_requests} benchmark requests (concurrency={args.target_concurrency})...")
        t_start = time.perf_counter()

        tasks = [
            send_request(session, semaphore, args.base_url, args.model,
                         prompts[i % len(prompts)], args.output_len, i,
                         args.request_timeout)
            for i in range(args.num_requests)
        ]

        # Track progress so long-running benchmarks don't trigger watchdog
        completed = 0
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % max(1, args.num_requests // 10) == 0 or completed == args.num_requests:
                elapsed = time.perf_counter() - t_start
                print(f"  Progress: {completed}/{args.num_requests} ({elapsed:.0f}s elapsed)")
        wall_clock = time.perf_counter() - t_start

    # Process results
    processed = []
    for r in results:
        if isinstance(r, Exception):
            processed.append({"request_id": -1, "status": "error", "error": str(r)})
        else:
            processed.append(r)

    return processed, wall_clock


def build_output(args, results, wall_clock):
    """Build output JSON from benchmark results."""
    successful = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") != "success"]

    ttft_vals = [r["ttft_s"] for r in successful if r.get("ttft_s") is not None]
    tpot_vals = [r["tpot_s"] for r in successful if r.get("tpot_s") is not None]
    e2e_vals = [r["e2e_s"] for r in successful if r.get("e2e_s") is not None]

    ttft_pcts = compute_percentiles(ttft_vals)
    tpot_pcts = compute_percentiles(tpot_vals)
    e2e_pcts = compute_percentiles(e2e_vals)

    total_prompt = sum(r.get("prompt_tokens", 0) for r in successful)
    total_output = sum(r.get("output_tokens", 0) for r in successful)

    output = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_requests": args.num_requests,
            "target_concurrency": args.target_concurrency,
            "warmup_requests": args.warmup_requests,
        },
        "wall_clock_s": round(wall_clock, 3),
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "num_successful": len(successful),
        "num_errors": len(errors),
        "client_percentiles": {
            "ttft_ms_p50": round(ttft_pcts[0.5] * 1000, 3) if ttft_pcts[0.5] is not None else None,
            "ttft_ms_p95": round(ttft_pcts[0.95] * 1000, 3) if ttft_pcts[0.95] is not None else None,
            "ttft_ms_p99": round(ttft_pcts[0.99] * 1000, 3) if ttft_pcts[0.99] is not None else None,
            "tpot_ms_p50": round(tpot_pcts[0.5] * 1000, 3) if tpot_pcts[0.5] is not None else None,
            "tpot_ms_p95": round(tpot_pcts[0.95] * 1000, 3) if tpot_pcts[0.95] is not None else None,
            "tpot_ms_p99": round(tpot_pcts[0.99] * 1000, 3) if tpot_pcts[0.99] is not None else None,
            "e2e_ms_p50": round(e2e_pcts[0.5] * 1000, 3) if e2e_pcts[0.5] is not None else None,
            "e2e_ms_p95": round(e2e_pcts[0.95] * 1000, 3) if e2e_pcts[0.95] is not None else None,
            "e2e_ms_p99": round(e2e_pcts[0.99] * 1000, 3) if e2e_pcts[0.99] is not None else None,
        },
        "requests": results,
    }
    return output


def main(argv=None):
    args = parse_args(argv)

    # Load prompts
    prompts = load_prompts(args.dataset, args.model, args.input_len,
                           args.num_requests + args.warmup_requests)

    # Run benchmark
    results, wall_clock = asyncio.run(run_benchmark(args, prompts))

    # Build and save output
    output = build_output(args, results, wall_clock)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Benchmark complete: {output['num_successful']}/{args.num_requests} successful")
    print(f"Wall clock: {wall_clock:.1f}s")
    print(f"Results saved to: {args.output}")
    cp = output["client_percentiles"]
    if cp.get("ttft_ms_p50") is not None:
        print(f"TTFT p50/p95/p99: {cp['ttft_ms_p50']:.1f} / {cp['ttft_ms_p95']:.1f} / {cp['ttft_ms_p99']:.1f} ms")
    if cp.get("tpot_ms_p50") is not None:
        print(f"TPOT p50/p95/p99: {cp['tpot_ms_p50']:.1f} / {cp['tpot_ms_p95']:.1f} / {cp['tpot_ms_p99']:.1f} ms")
    if cp.get("e2e_ms_p50") is not None:
        print(f"E2E  p50/p95/p99: {cp['e2e_ms_p50']:.1f} / {cp['e2e_ms_p95']:.1f} / {cp['e2e_ms_p99']:.1f} ms")


if __name__ == "__main__":
    main()
