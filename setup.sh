#!/usr/bin/env bash
# =============================================================================
# GOAT Live Bot — Setup Script
# Creates a virtual environment, installs dependencies, and prepares .env file.
# Run from the repo root: ./setup.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== GOAT Live Bot Setup ==="
echo ""

# --- 1. Find Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3 not found. Install it with:"
    echo "  sudo apt install python3 python3-venv python3-full"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "Found: $PYTHON_VERSION"

# --- 2. Create virtual environment ---
if [ -d "venv" ]; then
    echo "Virtual environment already exists at ./venv"
else
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
    echo "Created: ./venv"
fi

# --- 3. Activate and install dependencies ---
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r goat_live/requirements.txt --quiet
echo "Dependencies installed."

# --- 4. Copy .env template ---
if [ ! -f "goat_live/.env" ]; then
    cp goat_live/.env.example goat_live/.env
    echo "Created goat_live/.env from template — edit it with your credentials."
else
    echo "goat_live/.env already exists — skipping."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Edit your credentials:  nano goat_live/.env"
echo "  2. Activate the venv:      source venv/bin/activate"
echo "  3. Run the bot:            python -m goat_live.run"
echo ""
