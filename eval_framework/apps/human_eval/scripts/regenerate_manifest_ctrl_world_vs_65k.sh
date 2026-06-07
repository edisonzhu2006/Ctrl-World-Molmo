#!/usr/bin/env bash
# Build manifest for ctrl-world/checkpoint-10000 vs samples_20260521/checkpoint-65000 (63 pairs).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$REPO_ROOT"

python3 eval_framework/apps/human_eval/scripts/generate_manifest.py \
  --root eval_framework \
  --category prism \
  --trial trial_0 \
  --checkpoints ctrl-world/checkpoint-10000 samples_20260521-181822-8560524/checkpoint-65000 \
  --metrics-csv eval_framework/outputs/results/aggregated/metrics/runs/samples_20260521-181822-8560524/comparisons/checkpoint-65000_vs_ctrl-world-10000/tables/metrics_samples.csv \
  --metric-key psnr \
  --out eval_framework/apps/human_eval/data/manifest_ctrl-world-10k_vs_samples65k.jsonl
