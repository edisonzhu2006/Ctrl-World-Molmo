#!/usr/bin/env bash
# Start human_eval app + Cloudflare quick tunnel (*.trycloudflare.com).
# Share the printed URL with annotators (no SSH needed on their side).
#
# Prereq: cloudflared installed and outbound HTTPS allowed from this host.
#   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/bin/cloudflared && chmod +x ~/bin/cloudflared
#
# Usage (from eval_framework/apps/human_eval):
#   bash scripts/share_via_cloudflare.sh
#   bash scripts/share_via_cloudflare.sh --port 18766 --manifest data/manifest_prism.jsonl

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PORT="${HUMAN_EVAL_PORT:-18765}"
APP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      APP_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found in PATH."
  echo "Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  echo "  mkdir -p ~/bin && curl -L -o ~/bin/cloudflared \\"
  echo "    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
  echo "  chmod +x ~/bin/cloudflared && export PATH=\"\$HOME/bin:\$PATH\""
  exit 1
fi

cleanup() {
  if [[ -n "${APP_PID:-}" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting human eval on 127.0.0.1:${PORT} ..."
python app.py --host 127.0.0.1 --port "$PORT" "${APP_ARGS[@]}" &
APP_PID=$!

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo ""
echo "Starting Cloudflare quick tunnel (public URL below)..."
echo "Keep this session open. URL changes each time you restart cloudflared."
echo "Per-rater tracking: append ?user_id=alice to the shared link."
echo ""

cloudflared tunnel --url "http://127.0.0.1:${PORT}"
