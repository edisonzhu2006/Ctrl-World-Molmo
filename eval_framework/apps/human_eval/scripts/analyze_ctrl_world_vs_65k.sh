#!/usr/bin/env bash
# Export judgments and compute metric-vs-human confusion matrix (PSNR by default).
#
# Usage (after labeling):
#   bash eval_framework/apps/human_eval/scripts/analyze_ctrl_world_vs_65k.sh
#   bash eval_framework/apps/human_eval/scripts/analyze_ctrl_world_vs_65k.sh --metric-key ssim

set -euo pipefail

if command -v module &>/dev/null; then
  module load anaconda3/2025.12 2>/dev/null || true
  conda activate ctrl-world 2>/dev/null || true
fi

METRIC_KEY="psnr"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --metric-key) METRIC_KEY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$REPO_ROOT"

EFW="$REPO_ROOT/eval_framework"
APP_DIR="$EFW/apps/human_eval"
MANIFEST="$APP_DIR/data/manifest_ctrl-world-10k_vs_samples65k.jsonl"
DB="$APP_DIR/data/judgments_ctrl-world-10k_vs_samples65k.sqlite"
OUT_DIR="$APP_DIR/data/exports/ctrl-world-10k_vs_samples65k"
CSV="$OUT_DIR/judgments_export.csv"

mkdir -p "$OUT_DIR"

pip install -q -r "$APP_DIR/requirements.txt" 2>/dev/null || true

python "$APP_DIR/scripts/export_results.py" \
  --db "$DB" \
  --manifest "$MANIFEST" \
  --root "$EFW" \
  --out "$CSV"

python "$APP_DIR/scripts/evaluate_results.py" \
  --input "$CSV" \
  --out-dir "$OUT_DIR" \
  --metric-col metric_diff

echo ""
echo "=== Outputs ==="
echo "  Judgments CSV:     $CSV"
echo "  Win rates:         $OUT_DIR/human_winrate_by_model.csv"
echo "  Confusion matrix:  $OUT_DIR/human_metric_confusion.csv"
echo "  Metric accuracy:   $OUT_DIR/human_metric_accuracy.json"
echo ""
if [[ -f "$OUT_DIR/human_metric_confusion.csv" ]]; then
  echo "Confusion matrix (rows=metric_pred, cols=human_pref):"
  column -t -s, "$OUT_DIR/human_metric_confusion.csv" 2>/dev/null || cat "$OUT_DIR/human_metric_confusion.csv"
fi
