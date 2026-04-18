#!/usr/bin/env bash
# Jemma Discord bot launcher — Linux / macOS
# Usage: bash start_bot.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export PYTHONUNBUFFERED=1

# Pick the venv python if present, else system python3
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif [[ -f ".venv/Scripts/python" ]]; then
    # git-bash / MSYS on Windows
    PYTHON=".venv/Scripts/python"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "[start_bot] Using Python: $PYTHON"
echo "[start_bot] TRIBE v2 pre-warms in ~10 s. Press Ctrl-C to stop."

exec "$PYTHON" -m bot.bot
