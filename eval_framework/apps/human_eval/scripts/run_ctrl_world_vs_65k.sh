#!/usr/bin/env bash
# Human preference labeling: ctrl-world/checkpoint-10000 vs samples_20260521/checkpoint-65000
#
# From repo root:
#   bash eval_framework/apps/human_eval/scripts/run_ctrl_world_vs_65k.sh
#
# On cluster login node, tunnel from laptop:
#   ssh -L 18765:localhost:18765 lh2004@<login-node>
# Then open http://127.0.0.1:18765/?user_id=your_name

set -euo pipefail

# Optional: on Della / Princeton HPC
if command -v module &>/dev/null; then
  module load anaconda3/2025.12 2>/dev/null || true
  conda activate ctrl-world 2>/dev/null || true
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$REPO_ROOT"

EFW="$REPO_ROOT/eval_framework"
APP_DIR="$EFW/apps/human_eval"
MANIFEST="$APP_DIR/data/manifest_ctrl-world-10k_vs_samples65k.jsonl"
DB="$APP_DIR/data/judgments_ctrl-world-10k_vs_samples65k.sqlite"
PORT="${HUMAN_EVAL_PORT:-18765}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest missing. Regenerate with:"
  echo "  bash $APP_DIR/scripts/regenerate_manifest_ctrl_world_vs_65k.sh"
  exit 1
fi

pip install -q -r "$APP_DIR/requirements.txt" 2>/dev/null || true

cd "$APP_DIR"
exec python app.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --root "$EFW" \
  --manifest "$MANIFEST" \
  --db "$DB"
