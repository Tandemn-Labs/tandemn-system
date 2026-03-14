"""
Input file parsing utilities for Orca batch jobs.

Parses JSONL input files (local or S3) to extract prompt statistics:
num_lines, avg_input_tokens, max_input_tokens.
"""

import json
import os
import subprocess
import tempfile


def estimate_tokens(text: str) -> int:
    """Estimate token count using chars/4 approximation."""
    return max(1, len(text) // 4)


def extract_prompt_text(entry: dict) -> str:
    """Extract prompt text from OpenAI batch format entry."""
    try:
        messages = entry.get("body", {}).get("messages", [])
        # Concatenate all message contents
        return " ".join(msg.get("content", "") for msg in messages)
    except (KeyError, TypeError):
        return ""


def parse_input_file_stats(
    input_file: str, model_name: str = None, top_k_tokenize: int = 100
) -> tuple[int, int, int]:
    """
    Parse input file to extract real stats.

    Uses chars/4 for fast average estimation, then tokenizes the top-K longest
    prompts (by character count) with the model's actual tokenizer for an
    accurate max_input_tokens.

    Args:
        input_file: S3 URI (s3://bucket/path) or local path
        model_name: HuggingFace model name for tokenizer (e.g. "Qwen/Qwen2.5-72B-Instruct")
        top_k_tokenize: Number of longest prompts to tokenize (default 100)

    Returns:
        (num_lines, avg_input_tokens, max_input_tokens)
    """
    # Download from S3 if needed
    if input_file.startswith("s3://"):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = tmp.name
        result = subprocess.run(
            ["aws", "s3", "cp", input_file, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {input_file}: {result.stderr}")
        file_path = tmp_path
    else:
        file_path = input_file

    # Parse JSONL: collect prompt texts and chars/4 estimates
    prompt_texts = []
    char_estimates = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt_text = extract_prompt_text(entry)
                prompt_texts.append(prompt_text)
                char_estimates.append(estimate_tokens(prompt_text))
    finally:
        if input_file.startswith("s3://"):
            os.unlink(tmp_path)

    if not prompt_texts:
        raise ValueError(f"No valid entries found in {input_file}")

    num_lines = len(prompt_texts)
    avg_input_tokens = sum(char_estimates) // num_lines

    # Tokenize top-K longest prompts for accurate max_input_tokens
    max_input_tokens = max(char_estimates)  # fallback if tokenizer unavailable
    if model_name and top_k_tokenize > 0:
        try:
            from transformers import AutoTokenizer, logging as hf_logging

            hf_logging.set_verbosity_error()  # Suppress warn msgs about hf key
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Sort indices by character length descending, take top-K
            sorted_indices = sorted(
                range(num_lines), key=lambda i: len(prompt_texts[i]), reverse=True
            )
            top_indices = sorted_indices[:top_k_tokenize]

            max_tokenized = 0
            for idx in top_indices:
                token_count = len(tokenizer.encode(prompt_texts[idx]))
                max_tokenized = max(max_tokenized, token_count)

            print(
                f"[InputParser] Tokenized top-{len(top_indices)} longest prompts: "
                f"max_input={max_tokenized} tokens (chars/4 estimate was {max(char_estimates)})"
            )
            max_input_tokens = max_tokenized
        except Exception as e:
            print(
                f"[InputParser] WARNING: Tokenizer failed ({e}), using chars/4 estimate for max_input_tokens"
            )

    print(
        f"[InputParser] Parsed {num_lines} lines: avg_input={avg_input_tokens}, max_input={max_input_tokens}"
    )

    return num_lines, avg_input_tokens, max_input_tokens
