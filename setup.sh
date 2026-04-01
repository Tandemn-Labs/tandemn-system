#!/usr/bin/env bash
# Orca — one-liner setup
# Usage: bash setup.sh
set -e

echo ""
echo "  ██████╗ ██████╗  ██████╗ █████╗"
echo " ██╔═══██╗██╔══██╗██╔════╝██╔══██╗"
echo " ██║   ██║██████╔╝██║     ███████║"
echo " ██║   ██║██╔══██╗██║     ██╔══██║"
echo " ╚██████╔╝██║  ██║╚██████╗██║  ██║"
echo "  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝"
echo "  Setup"
echo ""

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 not found. Install Python 3.12+."; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "📦 Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"; exit 1; }

# Venv + deps
echo "📦 Creating virtual environment..."
uv venv .venv --python 3.12 --seed 2>/dev/null || uv venv .venv --seed
source .venv/bin/activate
echo "📦 Installing dependencies..."
uv pip install -r requirements.txt -q

# SkyPilot
echo "☁️  Checking SkyPilot cloud access..."
sky check 2>/dev/null || { echo "⚠️  Run 'sky check' manually to configure cloud access."; }

# Redis check
if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis: running"
else
    echo "⚠️  Redis not running. Start it: docker run -d -p 6379:6379 redis"
fi

# .env
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file..."
    read -p "   S3 bucket name (for uploads/outputs): " S3_BUCKET
    read -p "   HuggingFace token (for gated models, or press Enter to skip): " HF_TOKEN
    cat > .env <<EOF
S3_UPLOAD_BUCKET=${S3_BUCKET:-my-orca-bucket}
HF_TOKEN=${HF_TOKEN}
EOF
    echo "   Written to .env"
else
    echo "✅ .env already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "   Start the server:    source .venv/bin/activate && ./orca server"
echo "   Deploy a job:        ./orca deploy <model> <input.jsonl>"
echo "   Open dashboard:      ./orca web"
echo ""
