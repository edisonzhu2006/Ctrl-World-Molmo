#!/bin/bash
#SBATCH --job-name=prism_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lh2004/Ctrl-World_patched/eval_framework/pipelines/metrics/%x-%j.out

set -eo pipefail

module load anaconda3/2025.12
# conda activate scripts reference PS1; avoid nounset (-u) in batch jobs
conda activate ctrl-world

cd /home/lh2004/Ctrl-World_patched

echo "=== Step 1/4: PSNR / SSIM / LPIPS ==="
python eval_framework/pipelines/metrics/compute_psnr_ssim_lpips.py \
  --root eval_framework \
  --category prism \
  --trial trial_0 \
  --device cuda

echo "=== Step 2/4: FID / FVD ==="
python eval_framework/pipelines/metrics/compute_fid_fvd.py \
  --root eval_framework \
  --category prism \
  --trial trial_0 \
  --device cuda

echo "=== Step 3/4: Collect per-checkpoint sample rows (PSNR/SSIM/LPIPS) ==="
python eval_framework/pipelines/metrics/collect_checkpoint_samples.py \
  --root eval_framework \
  --out eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv

echo "=== Step 4/4: Summarize / PSNR plots / FID-FVD bar plot (if fid_fvd_by_checkpoint.csv exists) ==="
python eval_framework/pipelines/metrics/analyze_checkpoint_metrics.py \
  --root eval_framework \
  --samples-csv eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv

echo "=== Done: pool CSVs in eval_framework/outputs/results/aggregated/metrics ==="
for f in \
  eval_framework/outputs/results/aggregated/metrics/metrics_by_checkpoint.csv \
  eval_framework/outputs/results/aggregated/metrics/fid_fvd_by_checkpoint.csv \
  eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv
do
  if [[ -f "$f" ]]; then ls -lh "$f"; else echo "MISSING: $f"; fi
done
echo "=== Overview figures/tables ==="
for f in \
  eval_framework/outputs/results/aggregated/metrics/overview/tables/metrics_summary_per_view.csv \
  eval_framework/outputs/results/aggregated/metrics/overview/tables/metrics_summary_overall.csv \
  eval_framework/outputs/results/aggregated/metrics/overview/figures/boxplots_all_metrics_by_checkpoint.png \
  eval_framework/outputs/results/aggregated/metrics/overview/figures/violins_all_metrics_by_checkpoint.png \
  eval_framework/outputs/results/aggregated/metrics/overview/figures/per_view_all_metrics_mean_std.png \
  eval_framework/outputs/results/aggregated/metrics/overview/figures/fid_fvd_barplot_by_checkpoint.png
do
  if [[ -f "$f" ]]; then ls -lh "$f"; else echo "MISSING: $f"; fi
done
