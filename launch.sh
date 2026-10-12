#!/usr/bin/env bash
# Launcher for Teams Call Transcriber GUI.
# Activates the project's venv and runs the app.

set -e

PROJECT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/bin/activate"

exec python src/main.py "$@"
