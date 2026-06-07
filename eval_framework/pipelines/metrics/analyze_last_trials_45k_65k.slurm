#!/bin/bash
#SBATCH --job-name=metrics_last_trials
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/lh2004/Ctrl-World_patched/eval_framework/pipelines/metrics/%x-%j.out

# PSNR/SSIM/LPIPS on the latest trial per checkpoint, then comparison figures.
#
# Checkpoints: 45000, 50000, 55000, 60000, 65000
# Output: eval_framework/outputs/results/aggregated/metrics/runs/<RUN_ID>/comparisons/last_trials_45k-65k/
#
# sbatch eval_framework/pipelines/metrics/analyze_last_trials_45k_65k.slurm

set -eo pipefail

module load anaconda3/2025.12
conda activate ctrl-world

cd /home/lh2004/Ctrl-World_patched

RUN_ID=samples_20260521-181822-8560524
STEPS=(45000 50000 55000 60000 65000)
COMPARISON=last_trials_45k-65k
DEVICE="${METRICS_DEVICE:-cuda}"

PRED_BASE="eval_framework/data/predictions/${RUN_ID}"
OUT_DIR="eval_framework/outputs/results/aggregated/metrics/runs/${RUN_ID}/comparisons/${COMPARISON}"
POOL_CSV="${OUT_DIR}/tables/metrics_samples_last_trials.csv"

CKPTS=()
for step in "${STEPS[@]}"; do
  prism_root="${PRED_BASE}/checkpoint-${step}/prism"
  if [[ ! -d "${prism_root}" ]]; then
    echo "ERROR: missing ${prism_root}"
    exit 1
  fi
  trial=$(ls -d "${prism_root}"/trial_* 2>/dev/null | sort -V | tail -1)
  trial_name=$(basename "${trial}")
  n_mp4=$(find "${trial}" -name '*.mp4' | wc -l)
  echo "checkpoint-${step}: using ${trial_name} (${n_mp4} videos)"

  python eval_framework/pipelines/metrics/compute_psnr_ssim_lpips.py \
    --root eval_framework \
    --category prism \
    --run "${RUN_ID}" \
    --steps "${step}" \
    --trial "${trial_name}" \
    --device "${DEVICE}"

  CKPTS+=("${RUN_ID}/checkpoint-${step}")
done

mkdir -p "${OUT_DIR}/tables"
python eval_framework/pipelines/metrics/collect_metrics_subset.py \
  --root eval_framework \
  --ckpt-names "${CKPTS[@]}" \
  --out "${POOL_CSV}"

python eval_framework/pipelines/metrics/analyze_checkpoint_metrics.py \
  --root eval_framework \
  --samples-csv "${POOL_CSV}" \
  --run "${RUN_ID}" \
  --checkpoints "${CKPTS[@]}" \
  --comparison "${COMPARISON}" \
  --out-dir "${OUT_DIR}"

echo "=== Done: ${OUT_DIR} ==="
ls -lh "${OUT_DIR}/figures/" 2>/dev/null || ls -lhR "${OUT_DIR}"
