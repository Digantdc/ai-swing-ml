#!/usr/bin/env bash
# Daily run: activate venv, fetch latest data, generate picks, open the report.
# Usage: bash daily_run.sh [top_n]

set -e

TOP_N=${1:-5}

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv/ not found. Run 'bash setup.sh' first."
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Check that model is trained
if [ ! -f "output/ranker_model.pkl" ]; then
    echo "Model not yet trained. Training now (one-time, ~3 min)..."
    python train.py
    echo ""
fi

echo "Generating today's top-$TOP_N picks..."
python predict.py --top-n "$TOP_N"

# Open the most recent picks file
LATEST=$(ls -t output/daily_picks_*.md 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo ""
    echo "Opening: $LATEST"
    open "$LATEST"
else
    echo "No picks file produced. Check logs for errors."
fi
