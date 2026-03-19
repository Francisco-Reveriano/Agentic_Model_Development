#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKEND_PORT=8000
FRONTEND_PORT=5173
VENV_DIR=".venv"

# ── Activate virtual environment ─────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment ($VENV_DIR) ..."
source "$VENV_DIR/bin/activate"

# ── Install Python dependencies ──────────────────────────────────────
echo "Installing Python dependencies from requirements.txt ..."
pip install -q -r requirements.txt

# ── Install frontend dependencies ────────────────────────────────────
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies ..."
  (cd frontend && npm install)
fi

# ── Kill previous sessions ───────────────────────────────────────────
kill_port() {
  local pids
  pids=$(lsof -ti :"$1" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Killing previous process(es) on port $1 (PIDs: $pids)"
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

# ── Start services ───────────────────────────────────────────────────
echo "Starting backend on port $BACKEND_PORT ..."
uvicorn middleware.main:app --reload --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "Starting frontend on port $FRONTEND_PORT ..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

# ── Cleanup on exit ─────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down ..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── Wait for both ───────────────────────────────────────────────────
wait
