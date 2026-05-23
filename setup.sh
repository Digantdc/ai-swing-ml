#!/usr/bin/env bash
# One-time setup script for ai_swing_ml.
# Run from inside the project folder: bash setup.sh

set -e  # exit on any error

echo "=========================================="
echo "  AI Swing ML — One-time setup"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ from python.org or 'brew install python@3.12'"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PY_VERSION"

PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10 or newer required. Install with 'brew install python@3.12'"
    exit 1
fi
echo ""

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment in .venv/..."
    python3 -m venv .venv
    echo "Done."
else
    echo "[1/4] Virtual environment .venv/ already exists, skipping creation."
fi
echo ""

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[2/4] Activated venv. Python: $(which python3)"
echo ""

# Upgrade pip
echo "[3/4] Upgrading pip and installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo ""

# Run smoke tests
echo "[4/4] Running smoke tests..."
echo ""
PASS_COUNT=0
FAIL_COUNT=0
for testfile in tests/test_features.py tests/test_backtest.py tests/test_regime_and_sizer.py; do
    echo "--- $testfile ---"
    if python3 "$testfile" 2>&1 | tee /tmp/test_out.txt; then
        PASSES=$(grep -c "PASS" /tmp/test_out.txt || true)
        FAILS=$(grep -c "FAIL\|ERROR" /tmp/test_out.txt || true)
        PASS_COUNT=$((PASS_COUNT + PASSES))
        FAIL_COUNT=$((FAIL_COUNT + FAILS))
    fi
    echo ""
done

echo "=========================================="
echo "  Setup complete"
echo "=========================================="
echo "Tests: $PASS_COUNT passed, $FAIL_COUNT failed"
echo ""
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "Some tests failed. Check the output above and contact for help."
    exit 1
fi
echo "Next steps:"
echo "  1. Quick 1-year backtest (3-5 min):"
echo "       source .venv/bin/activate  # if not already active"
echo "       python run_backtest.py --quick"
echo ""
echo "  2. Full 5-year backtest (10-15 min):"
echo "       python run_backtest.py --years 5"
echo ""
echo "  3. Train production model:"
echo "       python train.py"
echo ""
echo "  4. Daily picks (run each evening):"
echo "       python predict.py --top-n 5"
echo ""
echo "  Or use the shortcut: bash daily_run.sh"
