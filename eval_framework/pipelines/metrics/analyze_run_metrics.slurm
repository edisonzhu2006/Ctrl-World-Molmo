#!/bin/bash
#SBATCH --job-name=metrics_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lh2004/Ctrl-World_patched/eval_framework/pipelines/metrics/%x-%j.out

# Per-run figures + tables under metrics/runs/<RUN_ID>/
# Edit RUN_ID, then: sbatch eval_framework/pipelines/metrics/analyze_run_metrics.slurm

set -eo pipefail

module load anaconda3/2025.12
conda activate ctrl-world

cd /home/lh2004/Ctrl-World_patched

RUN_ID=samples_20260521-181822-8560524
POOL_CSV=eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv
OUT_BASE=eval_framework/outputs/results/aggregated/metrics/runs/${RUN_ID}

python eval_framework/pipelines/metrics/analyze_checkpoint_metrics.py \
  --root eval_framework \
  --samples-csv "$POOL_CSV" \
  --run "$RUN_ID" \
  --out-dir "$OUT_BASE"

echo "=== Outputs ==="
ls -lhR "$OUT_BASE"
