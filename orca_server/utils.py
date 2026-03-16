"""Shared utilities for Orca server."""
import re
import uuid


def make_job_id(model_name: str | None = None) -> str:
    """Generate a short, readable job ID like 'mo-qwen72b-a3f7'.

    Extracts model family and parameter size from HuggingFace model names.
    Falls back to a short hex ID if model_name is missing.
    """
    short_hash = uuid.uuid4().hex[:4]
    if not model_name:
        return f"mo-{short_hash}"
    name = model_name.split("/")[-1]  # strip org prefix
    # Find parameter size: number+B preceded by - or start (skip version-like A22B)
    # "Qwen2.5-72B-Instruct" → "72B", "Qwen3-235B-A22B" → "235B"
    size_matches = re.findall(r"(?:^|-)(\d+(?:\.\d+)?)[bB]", name)
    size = size_matches[-1].replace(".", "") + "b" if size_matches else ""
    # Model family: leading alpha chars from the name
    family_match = re.match(r"([a-zA-Z]+)", name)
    family = family_match.group(1).lower()[:8] if family_match else "model"
    return f"mo-{family}{size}-{short_hash}"
