#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/services/api:${PYTHONPATH:-}"
exec python3 -m uvicorn main:app --app-dir "$ROOT/services/api" --host 127.0.0.1 --port 8000 --reload
