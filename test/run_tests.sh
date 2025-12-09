
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Using virtualenv Python..."
    "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/test_all.py"
else
    echo "Using system Python..."
    echo "Note: For best results, create a virtualenv first: python3 -m venv .venv && .venv/bin/maturin develop"
    python3 "$SCRIPT_DIR/test_all.py"
fi

