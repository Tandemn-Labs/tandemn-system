#!/usr/bin/env python3
"""
Upload HuggingFace models to S3 for fast loading in SkyPilot benchmarks.

This script launches a cheap EC2 instance via SkyPilot (no GPU needed),
downloads the model from HuggingFace, and uploads it to S3. This avoids
the ~2.5 hour download from HF during each benchmark run.

Usage:
    # Dry run (shows what would happen)
    python upload_model_to_s3.py

    # Actually run
    python upload_model_to_s3.py --run

    # Custom model and bucket
    python upload_model_to_s3.py --model meta-llama/Llama-3-8B --bucket my-bucket --run

    # Custom disk size (for very large models)
    python upload_model_to_s3.py --disk-size 500 --run
"""
import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

# Default configuration
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_BUCKET = "tandemn-model-shards"
DEFAULT_S3_PREFIX = "hf-models"
DEFAULT_DISK_SIZE = 300  # GB, needs ~2x model size for download + sync headroom
DEFAULT_REGION = "us-east-1"
CLUSTER_NAME = "model-upload-worker"


def generate_yaml(model: str, bucket: str, s3_prefix: str, disk_size: int, region: str, hf_token: str = "") -> str:
    """Generate SkyPilot YAML for downloading a model and uploading to S3."""
    # Convert HF model name to S3-safe path: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # -> "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    s3_model_path = f"s3://{bucket}/{s3_prefix}/{model}"
    local_model_dir = f"$HOME/models/{model}"

    return textwrap.dedent(f"""\
        name: {CLUSTER_NAME}

        resources:
          cloud: aws
          region: {region}
          cpus: 8+
          memory: 32+
          use_spot: true
          disk_size: {disk_size}

        setup: |
          set -euxo pipefail
          pip install -U huggingface_hub hf_transfer
          python -m huggingface_hub.cli.hf version
          # Install AWS CLI if not present
          if ! command -v aws &> /dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq unzip > /dev/null
            curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
            cd /tmp && unzip -qo awscliv2.zip && sudo ./aws/install
            rm -rf /tmp/awscliv2.zip /tmp/aws
          fi
          aws --version

        run: |
          set -euxo pipefail

          # Enable fast HF downloads (multi-threaded)
          export HF_HUB_ENABLE_HF_TRANSFER=1
          export HF_TOKEN="{hf_token}"

          MODEL="{model}"
          S3_PATH="{s3_model_path}"
          LOCAL_DIR="{local_model_dir}"

          echo "=============================================="
          echo "Model:      $MODEL"
          echo "Local dir:  $LOCAL_DIR"
          echo "S3 path:    $S3_PATH"
          echo "=============================================="

          # Check if model already exists on S3
          EXISTING_FILES=$(aws s3 ls "$S3_PATH/" --recursive 2>/dev/null | wc -l | tr -d ' ' || echo "0")
          if [ "$EXISTING_FILES" -gt "5" ] 2>/dev/null; then
            echo "WARNING: S3 path already has $EXISTING_FILES files!"
            echo "If you want to re-upload, delete the S3 path first:"
            echo "  aws s3 rm $S3_PATH/ --recursive"
            echo ""
            echo "Continuing anyway (will overwrite)..."
          fi

          # Download from HuggingFace (use python -m to avoid PATH issues)
          echo ""
          echo "=== Downloading model from HuggingFace ==="
          START_DL=$(date +%s)

          python -m huggingface_hub.cli.hf download "$MODEL" \\
            --local-dir "$LOCAL_DIR"

          END_DL=$(date +%s)
          DL_SECS=$((END_DL - START_DL))
          DL_SIZE=$(du -sh "$LOCAL_DIR" | cut -f1)
          echo "Download complete: $DL_SIZE in ${{DL_SECS}}s"

          # Upload to S3
          echo ""
          echo "=== Uploading to S3 ==="
          START_UL=$(date +%s)

          aws s3 sync "$LOCAL_DIR" "$S3_PATH/" \\
            --exclude ".git/*" \\
            --exclude ".gitattributes" \\
            --exclude "*.md"

          END_UL=$(date +%s)
          UL_SECS=$((END_UL - START_UL))
          echo "Upload complete in ${{UL_SECS}}s"

          # Verify
          echo ""
          echo "=== Verification ==="
          S3_COUNT=$(aws s3 ls "$S3_PATH/" --recursive | wc -l)
          LOCAL_COUNT=$(find "$LOCAL_DIR" -type f | grep -v '.git' | wc -l)
          echo "Local files: $LOCAL_COUNT"
          echo "S3 files:    $S3_COUNT"
          echo ""
          echo "S3 contents:"
          aws s3 ls "$S3_PATH/" --summarize --human-readable | tail -5

          echo ""
          echo "=============================================="
          echo "DONE!"
          echo "Download: $DL_SIZE in ${{DL_SECS}}s"
          echo "Upload:   ${{UL_SECS}}s"
          echo "S3 path:  $S3_PATH"
          echo "=============================================="
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Upload HuggingFace models to S3 via a temporary SkyPilot instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            Examples:
              python upload_model_to_s3.py                          # Dry run with defaults
              python upload_model_to_s3.py --run                    # Upload default model
              python upload_model_to_s3.py --model meta-llama/Llama-3-8B --run
              python upload_model_to_s3.py --bucket my-bucket --run

            Default model:  {DEFAULT_MODEL}
            Default bucket: {DEFAULT_BUCKET}
            S3 path:        s3://{DEFAULT_BUCKET}/{DEFAULT_S3_PREFIX}/{DEFAULT_MODEL}
        """),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--s3-prefix", default=DEFAULT_S3_PREFIX,
        help=f"S3 key prefix within the bucket (default: {DEFAULT_S3_PREFIX})",
    )
    parser.add_argument(
        "--disk-size", type=int, default=DEFAULT_DISK_SIZE,
        help=f"Disk size in GB for the worker instance (default: {DEFAULT_DISK_SIZE})",
    )
    parser.add_argument(
        "--region", default=DEFAULT_REGION,
        help=f"AWS region (should match your benchmark region) (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--hf-token", default="",
        help="HuggingFace token for authenticated downloads (faster speeds, required for gated models)",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Actually launch the instance (default: dry run)",
    )

    args = parser.parse_args()

    s3_path = f"s3://{args.bucket}/{args.s3_prefix}/{args.model}"

    print(f"Model:      {args.model}")
    print(f"S3 path:    {s3_path}")
    print(f"Region:     {args.region}")
    print(f"Disk size:  {args.disk_size} GB")
    print(f"Instance:   spot (cheapest with 8+ CPUs, 32+ GB RAM)")
    print()

    # Generate YAML
    yaml_content = generate_yaml(
        model=args.model,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        disk_size=args.disk_size,
        region=args.region,
        hf_token=args.hf_token,
    )

    yaml_path = Path("upload_model_to_s3.yaml")
    yaml_path.write_text(yaml_content)
    print(f"Generated: {yaml_path}")

    if not args.run:
        print()
        print("DRY RUN - add --run to actually launch. Generated YAML:")
        print("-" * 60)
        print(yaml_content)
        print("-" * 60)
        return

    # Launch
    print()
    print(f"Launching {CLUSTER_NAME}...")
    print("=" * 60)

    try:
        process = subprocess.Popen(
            ["sky", "launch", "-y", "--retry-until-up", "-c", CLUSTER_NAME, str(yaml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line.rstrip())

        process.wait()
        if process.returncode != 0:
            print(f"\nERROR: sky launch failed with return code {process.returncode}")
            sys.exit(1)

        print()
        print("=" * 60)
        print(f"Model uploaded to: {s3_path}")
        print()
        print("To use in benchmarks, the automatic_launch_1.py will need")
        print("file_mounts added to pull this S3 path to /models/ on each node.")
        print("=" * 60)

    finally:
        # Always tear down the worker instance
        print(f"\nTearing down {CLUSTER_NAME}...")
        subprocess.run(["sky", "down", "-y", CLUSTER_NAME])

    # Clean up YAML
    yaml_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
