#!/usr/bin/env bash
set -euo pipefail

WANDB_DIR="${1:-/home/lh2004/Ctrl-World_patched/wandb}"

if [[ ! -d "$WANDB_DIR" ]]; then
  echo "W&B directory not found: $WANDB_DIR" >&2
  exit 1
fi

found=0
while IFS= read -r -d '' run_dir; do
  found=1
  echo "Syncing $run_dir"
  wandb sync --append "$run_dir" || true
done < <(find "$WANDB_DIR" -maxdepth 1 -type d -name 'offline-run-*' -print0 | sort -z)

if [[ "$found" -eq 0 ]]; then
  echo "No offline runs found in $WANDB_DIR"
fi
