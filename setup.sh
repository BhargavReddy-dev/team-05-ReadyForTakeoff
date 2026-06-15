#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    # macOS/Linux
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows Git Bash
    source "$VENV_DIR/Scripts/activate"
else
    echo "Could not find a virtualenv activation script in $VENV_DIR" >&2
    exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete."
echo "Run the dashboard with:"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py"
