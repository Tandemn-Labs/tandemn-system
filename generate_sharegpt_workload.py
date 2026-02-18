#!/usr/bin/env python3
"""
Generate batch inference workload from ShareGPT dataset.

This script converts ShareGPT conversations into the OpenAI batch API format
compatible with vLLM batch inference.

Usage:
    python generate_sharegpt_workload.py --num-requests 1000 --avg-input-tokens 512
    python generate_sharegpt_workload.py -n 5000 --avg-input-tokens 256 --tolerance 0.3
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


@dataclass
class WorkloadConfig:
    num_requests: int
    avg_input_tokens: int
    avg_output_tokens: int  # Target for reference (actual output determined by model)
    tolerance: float  # How much variance from avg is acceptable (0.0-1.0)
    max_context_turns: int  # Max conversation turns to include
    seed: int


def count_tokens(text: str, encoding) -> int:
    """Count tokens using tiktoken (cl100k_base encoding, GPT-4/Claude compatible)."""
    return len(encoding.encode(text))


def load_sharegpt(filepath: str) -> List[Dict[str, Any]]:
    """Load ShareGPT JSON file."""
    print(f"Loading ShareGPT data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversations")
    return data


def extract_conversation_samples(
    conversations: List[Dict[str, Any]],
    config: WorkloadConfig,
    encoding
) -> List[Tuple[List[Dict[str, str]], int]]:
    """
    Extract conversation samples that match the target input token length.

    Returns list of (messages, token_count) tuples.
    """
    samples = []
    min_tokens = int(config.avg_input_tokens * (1 - config.tolerance))
    max_tokens = int(config.avg_input_tokens * (1 + config.tolerance))

    print(f"Target token range: {min_tokens} - {max_tokens} (avg: {config.avg_input_tokens})")

    for conv in conversations:
        conv_id = conv.get("id", "unknown")
        turns = conv.get("conversations", [])

        if not turns:
            continue

        # Try different slices of the conversation
        for end_idx in range(1, len(turns) + 1, 2):  # Only end on human turns
            # Get turns up to this point (only human messages as input)
            messages = []
            total_tokens = 0

            for i, turn in enumerate(turns[:end_idx]):
                role = "user" if turn["from"] == "human" else "assistant"
                content = turn["value"].strip()

                if not content:
                    continue

                # For input, we count all messages except the last assistant response
                # (which is what we want the model to generate)
                if role == "user" or i < end_idx - 1:
                    messages.append({"role": role, "content": content})
                    total_tokens += count_tokens(content, encoding)

            # Check if this slice matches our target
            if min_tokens <= total_tokens <= max_tokens:
                # Only keep if last message is from user (we want model to respond)
                if messages and messages[-1]["role"] == "user":
                    samples.append((messages, total_tokens))

            # Also try with just the last human message for shorter samples
            if end_idx >= 1:
                last_human = None
                for turn in reversed(turns[:end_idx]):
                    if turn["from"] == "human":
                        last_human = turn["value"].strip()
                        break

                if last_human:
                    single_tokens = count_tokens(last_human, encoding)
                    if min_tokens <= single_tokens <= max_tokens:
                        samples.append(([{"role": "user", "content": last_human}], single_tokens))

    return samples


def generate_workload(
    sharegpt_data: List[Dict[str, Any]],
    config: WorkloadConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate workload in OpenAI batch API format.

    Returns (requests, stats) tuple.
    """
    random.seed(config.seed)

    # Use cl100k_base encoding (GPT-4, works well for most modern LLMs)
    if HAS_TIKTOKEN:
        encoding = tiktoken.get_encoding("cl100k_base")
    else:
        print("Warning: tiktoken not available, using approximate token counting (~4 chars/token)")
        # Fallback: approximate tokens as ~4 chars per token (typical for English)
        class ApproxEncoding:
            def encode(self, text):
                return [0] * (len(text) // 4 + 1)
        encoding = ApproxEncoding()

    # Extract all valid samples
    print("Extracting conversation samples...")
    samples = extract_conversation_samples(sharegpt_data, config, encoding)
    print(f"Found {len(samples)} valid samples matching token criteria")

    if len(samples) < config.num_requests:
        print(f"Warning: Only found {len(samples)} samples, need {config.num_requests}")
        print("Will sample with replacement...")

    # Sample requests
    if len(samples) >= config.num_requests:
        selected = random.sample(samples, config.num_requests)
    else:
        # Sample with replacement
        selected = random.choices(samples, k=config.num_requests)

    # Convert to batch format
    requests = []
    token_counts = []

    for i, (messages, token_count) in enumerate(selected):
        request = {
            "custom_id": f"sharegpt-req-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": messages
            }
        }
        requests.append(request)
        token_counts.append(token_count)

    # Calculate stats
    stats = {
        "num_requests": len(requests),
        "avg_input_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_input_tokens": min(token_counts) if token_counts else 0,
        "max_input_tokens": max(token_counts) if token_counts else 0,
        "total_input_tokens": sum(token_counts),
        "target_avg_output_tokens": config.avg_output_tokens,
    }

    return requests, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate batch inference workload from ShareGPT dataset"
    )
    parser.add_argument(
        "--input", "-i",
        default="ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to ShareGPT JSON file"
    )
    parser.add_argument(
        "--num-requests", "-n",
        type=int,
        default=1000,
        help="Number of requests to generate"
    )
    parser.add_argument(
        "--avg-input-tokens",
        type=int,
        default=512,
        help="Target average input tokens per request"
    )
    parser.add_argument(
        "--avg-output-tokens",
        type=int,
        default=256,
        help="Target average output tokens (for reference only)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Tolerance for input token variance (0.0-1.0, default 0.5 = +/-50%%)"
    )
    parser.add_argument(
        "--max-context-turns",
        type=int,
        default=10,
        help="Maximum conversation turns to include"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload to S3 (s3://workload/sharegpt-numreq_N-avginputlen_X-avgoutputlen_Y.jsonl)"
    )

    args = parser.parse_args()

    config = WorkloadConfig(
        num_requests=args.num_requests,
        avg_input_tokens=args.avg_input_tokens,
        avg_output_tokens=args.avg_output_tokens,
        tolerance=args.tolerance,
        max_context_turns=args.max_context_turns,
        seed=args.seed,
    )

    # Load ShareGPT data
    sharegpt_data = load_sharegpt(args.input)

    # Generate workload
    print(f"\nGenerating workload with {config.num_requests} requests...")
    print(f"Target avg input tokens: {config.avg_input_tokens}")

    requests, stats = generate_workload(sharegpt_data, config)

    # Generate filename from stats
    filename = f"sharegpt-numreq_{stats['num_requests']}-avginputlen_{int(stats['avg_input_tokens'])}-avgoutputlen_{stats['target_avg_output_tokens']}.jsonl"
    output_path = Path(filename)
    print(f"\nWriting {len(requests)} requests to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')

    # Print stats
    print("\n" + "=" * 60)
    print("WORKLOAD STATISTICS")
    print("=" * 60)
    print(f"  Requests generated:    {stats['num_requests']}")
    print(f"  Avg input tokens:      {stats['avg_input_tokens']:.1f}")
    print(f"  Min input tokens:      {stats['min_input_tokens']}")
    print(f"  Max input tokens:      {stats['max_input_tokens']}")
    print(f"  Total input tokens:    {stats['total_input_tokens']:,}")
    print(f"  Target avg output:     {stats['target_avg_output_tokens']}")
    print("=" * 60)

    # S3 path format
    s3_path = f"s3://tandemn-orca/workload/{filename}"

    # Upload to S3 if requested
    if args.upload_s3:
        import subprocess
        print(f"\nUploading to {s3_path}...")
        result = subprocess.run(
            ["aws", "s3", "cp", str(output_path), s3_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Uploaded successfully!")
        else:
            print(f"Upload failed: {result.stderr}")

    # Print curl.sh example
    print(f"\nExample curl.sh config for this workload:")
    print(f'  "input_file": "{s3_path}",')
    print(f'  "num_lines": {stats["num_requests"]},')
    print(f'  "avg_input_tokens": {int(stats["avg_input_tokens"])},')
    print(f'  "avg_output_tokens": {stats["target_avg_output_tokens"]},')


if __name__ == "__main__":
    main()
