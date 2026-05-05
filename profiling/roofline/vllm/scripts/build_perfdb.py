#!/usr/bin/env python3
"""Build a unified performance database CSV from all benchmark results.

Crawls results/*/results.json recursively, deduplicates by
(model, tp, pp, max_input_length, max_output_length, gpu_model),
keeping only the latest entry per key, and writes perfdb_all.csv.

Usage:
    python scripts/build_perfdb.py                          # defaults
    python scripts/build_perfdb.py --results-dir results/   # explicit
    python scripts/build_perfdb.py --output perfdb_all.csv  # explicit
    python scripts/build_perfdb.py --dry-run                # preview only
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

DEDUP_KEY = ("model", "tp", "pp", "max_input_length", "max_output_length", "gpu_model")

# Map from directory path components to gpu_model for legacy results
GPU_MODEL_FROM_PATH = {
    "g6e": "L40S",
    "L40S": "L40S",
    "g6": "L4",
    "L4": "L4",
    "g5": "A10G",
    "A10G": "A10G",
}

# Regex to extract timestamp from directory name (e.g. 20260308_103110)
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def infer_gpu_model(results_json_path: str) -> str:
    """Infer GPU model from the directory path when not present in data."""
    path_str = str(results_json_path)
    for token, gpu in GPU_MODEL_FROM_PATH.items():
        if token in path_str:
            return gpu
    return "unknown"


def extract_dir_timestamp(dirpath: str) -> str:
    """Extract the timestamp string from a result directory name for sorting."""
    m = TIMESTAMP_RE.search(os.path.basename(dirpath))
    return m.group(1) if m else "00000000_000000"


def load_results(results_dir: Path) -> pd.DataFrame:
    """Recursively load all results.json files under results_dir."""
    all_rows = []
    results_files = sorted(results_dir.rglob("results.json"))

    if not results_files:
        print(f"No results.json files found under {results_dir}")
        return pd.DataFrame()

    for rfile in results_files:
        dirpath = str(rfile.parent)
        dir_ts = extract_dir_timestamp(dirpath)

        try:
            with open(rfile) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARN: skipping {rfile}: {e}")
            continue

        if not isinstance(entries, list):
            entries = [entries]

        for entry in entries:
            # Ensure gpu_model exists
            if "gpu_model" not in entry or not entry["gpu_model"]:
                entry["gpu_model"] = infer_gpu_model(rfile)

            # Add metadata for dedup ordering
            entry["_source_dir"] = dirpath
            entry["_dir_timestamp"] = dir_ts

            all_rows.append(entry)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    print(f"  Loaded {len(df)} entries from {len(results_files)} results.json files")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest entry per dedup key, based on directory timestamp."""
    if df.empty:
        return df

    # Ensure dedup key columns exist
    for col in DEDUP_KEY:
        if col not in df.columns:
            df[col] = None

    # Sort so latest timestamp is last, then drop_duplicates keeping last
    df = df.sort_values("_dir_timestamp", ascending=True)
    before = len(df)
    df = df.drop_duplicates(subset=list(DEDUP_KEY), keep="last")
    after = len(df)
    if before != after:
        print(f"  Deduplicated: {before} -> {after} entries ({before - after} duplicates removed)")

    return df


def sort_output(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by model, gpu_model, tp, pp, max_input_length, max_output_length."""
    sort_cols = [c for c in ["model", "gpu_model", "tp", "pp", "max_input_length", "max_output_length"]
                 if c in df.columns]
    return df.sort_values(sort_cols, ascending=True).reset_index(drop=True)


def build_perfdb(results_dir: Path, output_path: Path, dry_run: bool = False):
    """Main entry point: load, merge with existing, deduplicate, write."""
    print(f"Scanning {results_dir} ...")
    new_df = load_results(results_dir)

    if new_df.empty:
        print("No results found. Nothing to do.")
        return

    # Load existing CSV if present
    existing_df = pd.DataFrame()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            print(f"  Loaded existing {output_path}: {len(existing_df)} entries")
        except Exception as e:
            print(f"  WARN: could not read existing {output_path}: {e}")

    # Merge: new data takes precedence (concat then dedup keeps latest)
    if not existing_df.empty:
        # Add a synthetic old timestamp for existing CSV rows without _dir_timestamp
        if "_dir_timestamp" not in existing_df.columns:
            existing_df["_dir_timestamp"] = "00000000_000000"
        if "_source_dir" not in existing_df.columns:
            existing_df["_source_dir"] = ""

        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    deduped = deduplicate(combined)
    deduped = sort_output(deduped)

    # Compute stats
    n_existing = len(existing_df)
    n_final = len(deduped)

    if not existing_df.empty:
        # Figure out which are new vs updated
        existing_keys = set()
        for _, row in existing_df.iterrows():
            key = tuple(row.get(c) for c in DEDUP_KEY)
            existing_keys.add(key)

        n_new = 0
        n_updated = 0
        for _, row in deduped.iterrows():
            key = tuple(row.get(c) for c in DEDUP_KEY)
            if key in existing_keys:
                n_updated += 1
            else:
                n_new += 1
        # "updated" means key existed before but may have new data
        n_updated = n_updated  # keys that existed before and are still present
        n_truly_new = n_new
        print(f"\n  Summary: {n_truly_new} new, {n_updated} existing (kept/updated), {n_final} total entries")
    else:
        print(f"\n  Summary: {n_final} new entries (no existing CSV), {n_final} total")

    # Drop internal columns before writing
    output_df = deduped.drop(columns=["_source_dir", "_dir_timestamp"], errors="ignore")

    if dry_run:
        print(f"\n  [DRY RUN] Would write {len(output_df)} entries to {output_path}")
        print(f"  Columns ({len(output_df.columns)}): {', '.join(output_df.columns[:20])}...")
        # Show breakdown by model/gpu
        if "model" in output_df.columns and "gpu_model" in output_df.columns:
            print("\n  Breakdown by model x gpu_model:")
            for (model, gpu), grp in output_df.groupby(["model", "gpu_model"]):
                statuses = grp["status"].value_counts().to_dict() if "status" in grp.columns else {}
                print(f"    {model} / {gpu}: {len(grp)} entries {statuses}")
    else:
        output_df.to_csv(output_path, index=False)
        print(f"\n  Wrote {len(output_df)} entries to {output_path}")
        # Show breakdown
        if "model" in output_df.columns and "gpu_model" in output_df.columns:
            print("\n  Breakdown by model x gpu_model:")
            for (model, gpu), grp in output_df.groupby(["model", "gpu_model"]):
                statuses = grp["status"].value_counts().to_dict() if "status" in grp.columns else {}
                print(f"    {model} / {gpu}: {len(grp)} entries {statuses}")


def main():
    parser = argparse.ArgumentParser(description="Build unified performance database CSV")
    parser.add_argument("--results-dir", default="results/",
                        help="Root directory containing result subdirectories (default: results/)")
    parser.add_argument("--output", default="perfdb_all.csv",
                        help="Output CSV path (default: perfdb_all.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be written without actually writing")
    args = parser.parse_args()

    # Resolve paths relative to the script's parent directory (roofline/vllm/)
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    build_perfdb(results_dir, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
