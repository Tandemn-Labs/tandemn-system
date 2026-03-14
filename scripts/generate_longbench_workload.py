#!/usr/bin/env python3
"""
Generate batch inference workload from LongBench dataset.

Downloads LongBench from HuggingFace and converts to OpenAI batch API format
compatible with vLLM batch inference.

Usage:
    python generate_longbench_workload.py --num-requests 1000 --avg-input-tokens 8000
    python generate_longbench_workload.py -n 500 --tasks narrativeqa,hotpotqa
    python generate_longbench_workload.py --list-tasks  # Show available tasks
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# LongBench task categories and datasets
LONGBENCH_TASKS = {
    # Single-Document QA
    "narrativeqa": "Single-Doc QA - Stories",
    "qasper": "Single-Doc QA - Papers",
    "multifieldqa_en": "Single-Doc QA - Multi-field (EN)",
    "multifieldqa_zh": "Single-Doc QA - Multi-field (ZH)",

    # Multi-Document QA
    "hotpotqa": "Multi-Doc QA - Wikipedia",
    "2wikimqa": "Multi-Doc QA - 2Wiki",
    "musique": "Multi-Doc QA - MuSiQue",
    "dureader": "Multi-Doc QA - DuReader (ZH)",

    # Summarization
    "gov_report": "Summarization - Government Reports",
    "qmsum": "Summarization - Meeting Transcripts",
    "multi_news": "Summarization - News Articles",
    "vcsum": "Summarization - VCSUM (ZH)",

    # Few-shot Learning
    "trec": "Few-shot - Question Classification",
    "triviaqa": "Few-shot - Trivia QA",
    "samsum": "Few-shot - Dialogue Summarization",
    "lsht": "Few-shot - LSHT (ZH)",

    # Synthetic Tasks
    "passage_count": "Synthetic - Passage Counting",
    "passage_retrieval_en": "Synthetic - Passage Retrieval (EN)",
    "passage_retrieval_zh": "Synthetic - Passage Retrieval (ZH)",

    # Code
    "lcc": "Code - Line Completion",
    "repobench-p": "Code - Repo-level Completion",
}

# English-only tasks (for default)
ENGLISH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p"
]


@dataclass
class WorkloadConfig:
    num_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    tolerance: float
    tasks: List[str]
    seed: int


def count_tokens_approx(text: str) -> int:
    """Approximate token count (~4 chars per token for English)."""
    return len(text) // 4 + 1


def download_longbench(tasks: List[str], cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Download LongBench datasets from HuggingFace."""
    import urllib.request
    import zipfile
    import os

    cache_dir = cache_dir or ".longbench_cache"
    os.makedirs(cache_dir, exist_ok=True)

    data_dir = os.path.join(cache_dir, "data")
    zip_path = os.path.join(cache_dir, "data.zip")

    # Download and extract zip if not already done
    if not os.path.exists(data_dir):
        url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
        print(f"  Downloading LongBench data.zip (~114MB)...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"  Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cache_dir)
            print(f"  Done!")
        except Exception as e:
            print(f"  Failed to download: {e}")
            raise SystemExit(1)
    else:
        print(f"  Using cached data from {data_dir}")

    # Load requested tasks
    all_data = {}
    for task in tasks:
        jsonl_file = os.path.join(data_dir, f"{task}.jsonl")

        if not os.path.exists(jsonl_file):
            print(f"  {task}: file not found at {jsonl_file}")
            continue

        try:
            items = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            all_data[task] = items
            print(f"  {task}: {len(items)} examples")
        except Exception as e:
            print(f"  {task}: failed to parse - {e}")

    return all_data


def build_prompt_for_task(item: Dict[str, Any], task: str) -> str:
    """Build a prompt from LongBench item based on task type."""
    context = item.get("context", "")
    input_text = item.get("input", "")

    # Different prompt formats based on task category
    if task in ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"]:
        # Single-doc QA
        prompt = f"Read the following document and answer the question.\n\nDocument:\n{context}\n\nQuestion: {input_text}\n\nAnswer:"

    elif task in ["hotpotqa", "2wikimqa", "musique", "dureader"]:
        # Multi-doc QA
        prompt = f"Based on the following documents, answer the question.\n\nDocuments:\n{context}\n\nQuestion: {input_text}\n\nAnswer:"

    elif task in ["gov_report", "qmsum", "multi_news", "vcsum"]:
        # Summarization
        prompt = f"Summarize the following text.\n\nText:\n{context}\n\nSummary:"

    elif task in ["trec", "triviaqa", "samsum", "lsht"]:
        # Few-shot
        prompt = f"{context}\n\n{input_text}"

    elif task in ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"]:
        # Synthetic
        prompt = f"{context}\n\nQuestion: {input_text}\n\nAnswer:"

    elif task in ["lcc", "repobench-p"]:
        # Code completion
        prompt = f"Complete the following code.\n\n{context}\n\n{input_text}"

    else:
        # Default format
        prompt = f"{context}\n\n{input_text}"

    return prompt


def extract_samples(
    data: Dict[str, List[Dict]],
    config: WorkloadConfig
) -> List[Tuple[str, str, int]]:
    """
    Extract samples matching target token length.
    Returns list of (prompt, task_name, token_count) tuples.
    """
    samples = []
    min_tokens = int(config.avg_input_tokens * (1 - config.tolerance))
    max_tokens = int(config.avg_input_tokens * (1 + config.tolerance))

    print(f"Target token range: {min_tokens} - {max_tokens}")

    for task, items in data.items():
        task_samples = 0
        for item in items:
            prompt = build_prompt_for_task(item, task)
            token_count = count_tokens_approx(prompt)

            if min_tokens <= token_count <= max_tokens:
                samples.append((prompt, task, token_count))
                task_samples += 1

        print(f"  {task}: {task_samples} samples in range")

    return samples


def generate_workload(
    data: Dict[str, List[Dict]],
    config: WorkloadConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate workload in OpenAI batch API format."""
    random.seed(config.seed)

    # Extract matching samples
    print("\nExtracting samples matching token criteria...")
    samples = extract_samples(data, config)
    print(f"Total: {len(samples)} valid samples")

    if len(samples) == 0:
        print("Error: No samples found matching criteria. Try adjusting --tolerance or --avg-input-tokens")
        raise SystemExit(1)

    if len(samples) < config.num_requests:
        print(f"Warning: Only {len(samples)} samples, need {config.num_requests}. Sampling with replacement...")

    # Sample requests
    if len(samples) >= config.num_requests:
        selected = random.sample(samples, config.num_requests)
    else:
        selected = random.choices(samples, k=config.num_requests)

    # Convert to batch format
    requests = []
    token_counts = []
    task_counts = {}

    for i, (prompt, task, token_count) in enumerate(selected):
        request = {
            "custom_id": f"longbench-{task}-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        }
        requests.append(request)
        token_counts.append(token_count)
        task_counts[task] = task_counts.get(task, 0) + 1

    stats = {
        "num_requests": len(requests),
        "avg_input_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "min_input_tokens": min(token_counts) if token_counts else 0,
        "max_input_tokens": max(token_counts) if token_counts else 0,
        "total_input_tokens": sum(token_counts),
        "target_avg_output_tokens": config.avg_output_tokens,
        "task_distribution": task_counts,
    }

    return requests, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate batch inference workload from LongBench dataset"
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
        default=8000,
        help="Target average input tokens per request"
    )
    parser.add_argument(
        "--avg-output-tokens",
        type=int,
        default=256,
        help="Target average output tokens (for reference)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Tolerance for token variance (0.0-1.0, default 0.5 = +/-50%%)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks (default: all English tasks)"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloaded datasets"
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload to S3 after generation"
    )

    args = parser.parse_args()

    # List tasks mode
    if args.list_tasks:
        print("\nAvailable LongBench Tasks:")
        print("=" * 60)
        for task, desc in LONGBENCH_TASKS.items():
            lang = "(ZH)" if "_zh" in task or task in ["dureader", "vcsum", "lsht"] else "(EN)"
            print(f"  {task:25s} {lang:5s} {desc}")
        print("\nDefault (English only):")
        print(f"  {','.join(ENGLISH_TASKS)}")
        return

    # Parse tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
        invalid = [t for t in tasks if t not in LONGBENCH_TASKS]
        if invalid:
            print(f"Error: Invalid tasks: {invalid}")
            print("Use --list-tasks to see available tasks")
            return
    else:
        tasks = ENGLISH_TASKS

    config = WorkloadConfig(
        num_requests=args.num_requests,
        avg_input_tokens=args.avg_input_tokens,
        avg_output_tokens=args.avg_output_tokens,
        tolerance=args.tolerance,
        tasks=tasks,
        seed=args.seed,
    )

    # Download data
    print(f"\nDownloading LongBench datasets ({len(tasks)} tasks)...")
    data = download_longbench(tasks, args.cache_dir)

    total_examples = sum(len(v) for v in data.values())
    print(f"Downloaded {total_examples} total examples")

    # Generate workload
    print(f"\nGenerating workload...")
    print(f"  Requests: {config.num_requests}")
    print(f"  Target avg input tokens: {config.avg_input_tokens}")

    requests, stats = generate_workload(data, config)

    # Generate filename
    filename = f"longbench-numreq_{stats['num_requests']}-avginputlen_{int(stats['avg_input_tokens'])}-avgoutputlen_{stats['target_avg_output_tokens']}.jsonl"
    output_dir = Path(__file__).resolve().parent.parent / "examples" / "workloads"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Write output
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
    print("\n  Task distribution:")
    for task, count in sorted(stats['task_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {task}: {count}")
    print("=" * 60)

    # S3 upload
    s3_path = f"s3://tandemn-orca/workload/{filename}"

    if args.upload_s3:
        import subprocess
        print(f"\nUploading to {s3_path}...")
        result = subprocess.run(
            ["aws", "s3", "cp", str(output_path), s3_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Uploaded successfully!")
        else:
            print(f"Upload failed: {result.stderr}")

    # Print curl example
    print(f"\nExample curl.sh config:")
    print(f'  "input_file": "{s3_path}",')
    print(f'  "num_lines": {stats["num_requests"]},')
    print(f'  "avg_input_tokens": {int(stats["avg_input_tokens"])},')
    print(f'  "avg_output_tokens": {stats["target_avg_output_tokens"]},')


if __name__ == "__main__":
    main()
