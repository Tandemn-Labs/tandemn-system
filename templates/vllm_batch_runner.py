#!/usr/bin/env python3
"""
vLLM batch runner with proper distributed cleanup and performance tracking.

This script replaces `vllm run-batch` to ensure proper cleanup of NCCL/TCPStore
before exit, avoiding the noisy shutdown errors.

Key features:
- Uses AsyncLLMEngine when pipeline parallelism (PP>1) is enabled
- Calls cleanup_dist_env_and_memory() before exit
- Properly destroys the distributed process group
- Writes progress to /tmp/vllm_progress.json for external monitoring
- Saves performance metrics to CSV file alongside output
"""

import argparse
import asyncio
import csv
import gc
import json
import sys
import time
import os
from datetime import datetime
from typing import List, Dict, Any

# Note: CUDA is initialized by nvidia-smi in vllm_run script
# Let vLLM handle torch CUDA initialization to avoid conflicts

# Progress file for external monitoring
PROGRESS_FILE = "/tmp/vllm_progress.json"


def write_progress(done: int, total: int, status: str = "running"):
    """Write progress to file for external monitoring."""
    progress = {
        "done": done,
        "total": total,
        "status": status,
        "timestamp": time.time(),
    }
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)
    except Exception:
        pass  # Don't fail on progress write errors


def write_metrics_csv(output_path: str, metrics: Dict[str, Any]):
    """
    Write performance metrics to CSV file in same directory as output.

    Args:
        output_path: Path to output.jsonl file
        metrics: Dict of metric_name -> value
    """
    # Create metrics file path (same dir as output, named metrics.csv)
    output_dir = os.path.dirname(output_path)
    metrics_file = os.path.join(output_dir, "metrics.csv") if output_dir else "metrics.csv"

    try:
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for metric, value in metrics.items():
                # Format floats to reasonable precision
                if isinstance(value, float):
                    value = f"{value:.4f}"
                writer.writerow([metric, value])
        print(f"[BatchRunner] Wrote metrics to {metrics_file}")
    except Exception as e:
        print(f"[BatchRunner] Warning: Failed to write metrics: {e}")


def calculate_percentiles(values: List[int]) -> Dict[str, float]:
    """Calculate p50, p90, p99 percentiles for a list of values."""
    if not values:
        return {"p50": 0, "p90": 0, "p99": 0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        idx = (n - 1) * p / 100
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

    return {
        "p50": percentile(50),
        "p90": percentile(90),
        "p99": percentile(99),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM batch inference with proper cleanup")
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file (OpenAI batch format)")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("-pp", "--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None, help="Max context length. If not set, vLLM auto-detects.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    # Infrastructure configuration
    parser.add_argument("--cloud", default="aws", help="Cloud provider (aws, gcp, azure)")
    parser.add_argument("--instance-type", default="unknown", help="Instance type (e.g., g6e.12xlarge)")
    parser.add_argument("--gpu-name", default="unknown", help="GPU name (e.g., L40S, A100)")
    parser.add_argument("--engine", default="vllm", help="Inference engine (vllm, sglang)")
    parser.add_argument("--quantization", default="none", help="Quantization type (none, fp16, bf16, int8, int4)")
    parser.add_argument("--dtype", default="auto", help="Model dtype (auto, float16, bfloat16, float32)")
    parser.add_argument("--kv-cache-dtype", default="auto", help="KV cache dtype (auto, fp8, fp8_e5m2, etc.)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    parser.add_argument("--port", type=int, default=8001, help="Metrics port (unused, for compatibility)")
    return parser.parse_args()


def load_requests(input_file: str) -> List[Dict[str, Any]]:
    """Load requests from OpenAI batch format JSONL."""
    requests = []
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    requests.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[BatchRunner] Warning: Skipping invalid JSON on line {line_num}: {e}")
    return requests


def format_chat_prompt(messages: List[Dict], tokenizer) -> str:
    """Format chat messages using the model's chat template."""
    try:
        # Use the tokenizer's chat template if available
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: simple concatenation
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt


async def run_async_generation(engine, prompts: List[str], sampling_params) -> List:
    """
    Run batch generation using AsyncLLMEngine.
    Required for pipeline parallelism (PP>1).
    """
    results = {}
    num_prompts = len(prompts)

    async def process_request(request_id: str, prompt: str):
        """Process a single request and store result."""
        async for output in engine.generate(prompt, sampling_params, request_id):
            if output.finished:
                results[request_id] = output
                return

    # Create tasks for all prompts
    tasks = []
    for i, prompt in enumerate(prompts):
        request_id = f"req-{i}"
        tasks.append(asyncio.create_task(process_request(request_id, prompt)))

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Return results in original order
    return [results[f"req-{i}"] for i in range(num_prompts)]


def main():
    args = parse_args()

    # =========================================================================
    # Performance tracking - start
    # =========================================================================
    job_start_time = time.time()
    job_start_timestamp = datetime.now().isoformat()

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Import vLLM components
    from vllm import LLM, SamplingParams

    # Import cleanup function
    try:
        from vllm.distributed import cleanup_dist_env_and_memory
    except ImportError:
        def cleanup_dist_env_and_memory(shutdown_ray=True):
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[BatchRunner] Loading model: {args.model}")
    print(f"[BatchRunner] TP={args.tensor_parallel_size}, PP={args.pipeline_parallel_size}")
    print(f"[BatchRunner] max_model_len={args.max_model_len or 'auto'}, gpu_util={args.gpu_memory_utilization}")

    # Pipeline parallelism requires AsyncLLMEngine
    use_async_engine = args.pipeline_parallel_size > 1
    if use_async_engine:
        print("[BatchRunner] PP>1 detected, using AsyncLLMEngine")

    # Determine executor backend
    # Use Ray when any multi-GPU parallelism is needed.
    # The "mp" (fork) backend crashes on vLLM 0.7.x because CUDA gets
    # initialized during `from vllm import LLM` (platform detection),
    # and forked children cannot re-initialize CUDA.
    if args.pipeline_parallel_size > 1 or args.tensor_parallel_size > 1:
        backend = "ray"
    else:
        backend = "mp"

    # Track model loading time
    model_load_start = time.time()

    # Get model's max supported context length from config
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_max_len = getattr(model_config, 'max_position_embeddings', None)
    print(f"[BatchRunner] Model max_position_embeddings={model_max_len}")

    # Cap max_model_len by model's supported context length
    effective_max_model_len = args.max_model_len
    if args.max_model_len is not None and model_max_len is not None:
        if args.max_model_len > model_max_len:
            print(f"[BatchRunner] WARNING: Requested max_model_len={args.max_model_len} exceeds model limit={model_max_len}, capping to {model_max_len}")
            effective_max_model_len = model_max_len

    # Build engine args - only include max_model_len if explicitly set
    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": True,
        "distributed_executor_backend": backend,
    }
    if effective_max_model_len is not None:
        engine_kwargs["max_model_len"] = effective_max_model_len
        print(f"[BatchRunner] Using max_model_len={effective_max_model_len}")
    else:
        print("[BatchRunner] max_model_len not set, vLLM will auto-detect based on available memory")

    # Initialize vLLM engine
    if use_async_engine:
        # Use AsyncLLMEngine for pipeline parallelism
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(**engine_kwargs)
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        # Get tokenizer from the engine
        tokenizer = llm.engine.tokenizer.tokenizer
    else:
        # Use synchronous LLM for TP-only
        llm = LLM(**engine_kwargs)
        tokenizer = llm.get_tokenizer()

    model_load_time = time.time() - model_load_start
    print(f"[BatchRunner] Model loaded in {model_load_time:.2f}s")

    # Get max_model_len from engine config for prompt length validation
    if use_async_engine:
        max_model_len = llm.engine.model_config.max_model_len
    else:
        max_model_len = llm.llm_engine.model_config.max_model_len
    print(f"[BatchRunner] max_model_len={max_model_len}")

    # Load input requests
    requests = load_requests(args.input)
    print(f"[BatchRunner] Loaded {len(requests)} requests from {args.input}")

    # Prepare prompts and metadata, filtering out prompts that are too long
    prompts = []
    request_metadata = []  # Store custom_id and sampling params per request
    skipped_requests = []  # Track requests that were skipped due to length

    for req in requests:
        custom_id = req.get("custom_id", f"req-{len(prompts) + len(skipped_requests)}")
        body = req.get("body", {})
        messages = body.get("messages", [])

        # Extract sampling parameters from request
        req_max_tokens = body.get("max_tokens", 256)
        req_temperature = body.get("temperature", 0.7)

        # Format prompt using chat template
        prompt = format_chat_prompt(messages, tokenizer)

        # Check prompt length (tokenize to count tokens)
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)

        # Validate: input_len + max_output_tokens must fit within max_model_len
        total_required = prompt_len + req_max_tokens
        if total_required > max_model_len:
            print(f"[BatchRunner] WARNING: Skipping request {custom_id} - "
                  f"input({prompt_len}) + output({req_max_tokens}) = {total_required} > max_model_len({max_model_len})")
            skipped_requests.append({
                "custom_id": custom_id,
                "prompt_len": prompt_len,
                "max_tokens": req_max_tokens,
                "total_required": total_required,
                "max_model_len": max_model_len,
                "error": f"input_len({prompt_len}) + max_tokens({req_max_tokens}) = {total_required} exceeds max_model_len({max_model_len})"
            })
            continue

        prompts.append(prompt)

        # Store sampling parameters per request
        request_metadata.append({
            "custom_id": custom_id,
            "max_tokens": req_max_tokens,
            "temperature": req_temperature,
            "model": body.get("model", args.model),
        })

    print(f"[BatchRunner] {len(prompts)} prompts to process, {len(skipped_requests)} skipped (too long)")

    # Group requests by sampling params for efficient batching
    # For simplicity, use the most common params for all (vLLM handles this efficiently)
    # In production, you might want to group by params

    # Use default sampling params (can be extended to per-request)
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.7,
    )

    # Generate all at once (vLLM handles batching internally)
    print(f"[BatchRunner] Starting batch generation for {len(prompts)} prompts...")
    write_progress(0, len(prompts), "running")
    generation_start_time = time.time()

    if len(prompts) == 0:
        print("[BatchRunner] WARNING: No prompts to process (all skipped due to length)")
        outputs = []
        generation_time = 0
    elif use_async_engine:
        # Use async generation for pipeline parallelism
        outputs = asyncio.run(run_async_generation(llm, prompts, sampling_params))
        generation_time = time.time() - generation_start_time
        print(f"[BatchRunner] Generation complete in {generation_time:.2f}s ({len(prompts)/generation_time:.1f} req/s)")
    else:
        # Use sync generation for TP-only
        outputs = llm.generate(prompts, sampling_params)
        generation_time = time.time() - generation_start_time
        print(f"[BatchRunner] Generation complete in {generation_time:.2f}s ({len(prompts)/generation_time:.1f} req/s)")

    write_progress(len(prompts), len(prompts), "completed")

    # Format results and collect token statistics
    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Per-request token counts for percentile calculations
    prompt_token_counts = []
    completion_token_counts = []

    for i, output in enumerate(outputs):
        meta = request_metadata[i]
        output_text = output.outputs[0].text if output.outputs else ""

        # Count tokens
        prompt_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
        completion_tokens = len(output.outputs[0].token_ids) if output.outputs and hasattr(output.outputs[0], 'token_ids') else 0

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        prompt_token_counts.append(prompt_tokens)
        completion_token_counts.append(completion_tokens)

        result = {
            "id": f"vllm-{output.request_id}" if hasattr(output, 'request_id') else f"vllm-{i}",
            "custom_id": meta["custom_id"],
            "response": {
                "status_code": 200,
                "request_id": f"vllm-batch-{i}",
                "body": {
                    "id": f"chatcmpl-{i}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": meta["model"],
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "finish_reason": output.outputs[0].finish_reason if output.outputs else "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                }
            },
            "error": None,
        }
        results.append(result)

    # Add skipped requests as errors
    for skipped in skipped_requests:
        result = {
            "id": f"vllm-skipped-{skipped['custom_id']}",
            "custom_id": skipped["custom_id"],
            "response": {
                "status_code": 400,
                "request_id": f"vllm-skipped-{skipped['custom_id']}",
                "body": None
            },
            "error": {
                "type": "context_length_exceeded",
                "message": skipped["error"],
                "input_tokens": skipped["prompt_len"],
                "max_output_tokens": skipped["max_tokens"],
                "total_required": skipped["total_required"],
                "max_model_len": skipped["max_model_len"]
            }
        }
        results.append(result)

    # Write output (create directory if needed)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"[BatchRunner] Wrote {len(results)} results to {args.output} ({len(skipped_requests)} skipped)")

    # =========================================================================
    # Performance tracking - calculate and save metrics
    # =========================================================================
    job_end_time = time.time()
    total_runtime = job_end_time - job_start_time
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Calculate percentiles for input/output token lengths
    prompt_percentiles = calculate_percentiles(prompt_token_counts)
    completion_percentiles = calculate_percentiles(completion_token_counts)

    metrics = {
        # === TIMESTAMPS ===
        "job_start_timestamp": job_start_timestamp,
        "job_end_timestamp": datetime.now().isoformat(),

        # === RUNTIME METRICS (seconds) ===
        "total_runtime_sec": total_runtime,
        "model_load_time_sec": model_load_time,
        "generation_time_sec": generation_time,

        # === WORKLOAD METRICS ===
        "num_requests_total": len(results),
        "num_requests_completed": len(results) - len(skipped_requests),
        "num_requests_skipped": len(skipped_requests),
        # Input token statistics (for completed requests only)
        "total_input_tokens": total_prompt_tokens,
        "avg_input_tokens": total_prompt_tokens / len(prompt_token_counts) if prompt_token_counts else 0,
        "p50_input_tokens": prompt_percentiles["p50"],
        "p90_input_tokens": prompt_percentiles["p90"],
        "p99_input_tokens": prompt_percentiles["p99"],
        "min_input_tokens": min(prompt_token_counts) if prompt_token_counts else 0,
        "max_input_tokens": max(prompt_token_counts) if prompt_token_counts else 0,
        # Output token statistics (for completed requests only)
        "total_output_tokens": total_completion_tokens,
        "avg_output_tokens": total_completion_tokens / len(completion_token_counts) if completion_token_counts else 0,
        "p50_output_tokens": completion_percentiles["p50"],
        "p90_output_tokens": completion_percentiles["p90"],
        "p99_output_tokens": completion_percentiles["p99"],
        "min_output_tokens": min(completion_token_counts) if completion_token_counts else 0,
        "max_output_tokens": max(completion_token_counts) if completion_token_counts else 0,
        # Total tokens
        "total_tokens": total_tokens,

        # === THROUGHPUT METRICS ===
        "throughput_requests_per_sec": len(prompt_token_counts) / generation_time if generation_time > 0 else 0,
        "throughput_tokens_per_sec": total_tokens / generation_time if generation_time > 0 else 0,
        "throughput_output_tokens_per_sec": total_completion_tokens / generation_time if generation_time > 0 else 0,

        # === MODEL CONFIGURATION ===
        "model_name": args.model,
        "quantization": args.quantization,

        # === INFRASTRUCTURE CONFIGURATION ===
        "cloud_provider": args.cloud,
        "instance_type": args.instance_type,
        "gpu_name": args.gpu_name,

        # === ENGINE CONFIGURATION ===
        "engine": args.engine,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        # Get actual max_model_len from vLLM (may differ from args if auto-detected)
        "max_model_len": (llm.engine.model_config.max_model_len if use_async_engine else
                         (llm.llm_engine.model_config.max_model_len if hasattr(llm, 'llm_engine') else args.max_model_len)),
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "use_async_engine": use_async_engine,
    }

    # Write metrics to CSV
    write_metrics_csv(args.output, metrics)

    # Print summary
    print(f"\n[BatchRunner] === Performance Summary ===")
    print(f"  Total runtime: {total_runtime:.2f}s")
    print(f"  Model load time: {model_load_time:.2f}s")
    print(f"  Generation time: {generation_time:.2f}s")
    print(f"  Requests: {len(results)}")
    print(f"  Total tokens: {total_tokens:,} (prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
    print(f"  Throughput: {len(results)/generation_time:.2f} req/s, {total_tokens/generation_time:.2f} tok/s")

    # =========================================================================
    # CRITICAL: Proper cleanup to avoid NCCL/TCPStore errors
    # This is the key difference from `vllm run-batch`
    # =========================================================================
    print("[BatchRunner] Cleaning up distributed environment...")

    # Shutdown async engine properly if using PP
    if use_async_engine:
        try:
            # AsyncLLMEngine shutdown - engine already shuts down gracefully
            pass
        except Exception as e:
            print(f"[BatchRunner] Warning: Async engine shutdown: {e}")

    # Delete LLM object first to trigger internal cleanup
    del llm
    del tokenizer

    # Small delay to let async operations settle
    time.sleep(2)

    # Cleanup distributed environment (this calls destroy_process_group internally)
    cleanup_dist_env_and_memory(shutdown_ray=False)

    # Final GPU cleanup
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("[BatchRunner] Cleanup complete. Exiting cleanly.")


if __name__ == "__main__":
    main()
