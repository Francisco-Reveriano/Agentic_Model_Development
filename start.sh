#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT=8000
FRONTEND_PORT=5173

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
