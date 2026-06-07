#!/bin/bash
#SBATCH --job-name=compare_ckpts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lh2004/Ctrl-World_patched/eval_framework/pipelines/metrics/%x-%j.out

# Compare selected checkpoints within one training run.
# Output: metrics/runs/<RUN_ID>/comparisons/<slug>/{figures,tables}/
#
# sbatch eval_framework/pipelines/metrics/compare_two_checkpoints.slurm

set -eo pipefail

module load anaconda3/2025.12
conda activate ctrl-world

cd /home/lh2004/Ctrl-World_patched

RUN_ID=samples_20260521-181822-8560524
COMPARISON=checkpoint-5000_vs_checkpoint-65000
STEPS=(5000 25000 45000 65000)

CKPTS=()
for step in "${STEPS[@]}"; do
  CKPTS+=("${RUN_ID}/checkpoint-${step}")
done

POOL=eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv
OUT_DIR=eval_framework/outputs/results/aggregated/metrics/runs/${RUN_ID}/comparisons/${COMPARISON}

python eval_framework/pipelines/metrics/analyze_checkpoint_metrics.py \
  --root eval_framework \
  --samples-csv "$POOL" \
  --run "$RUN_ID" \
  --checkpoints "${CKPTS[@]}" \
  --comparison "$COMPARISON" \
  --out-dir "$OUT_DIR"

echo "=== Outputs in ${OUT_DIR} ==="
ls -lhR "$OUT_DIR"
