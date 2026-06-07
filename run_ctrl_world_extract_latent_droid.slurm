#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --job-name=ctrl_extract_latent_droid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%8
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/%x-%A_%a.out

module load anaconda3/2025.12
conda activate ctrl-world

export HF_HOME=/scratch/gpfs/SHAHD/lh2004/cache/huggingface
export TORCH_HOME=/scratch/gpfs/SHAHD/lh2004/cache/torch
export XDG_CACHE_HOME=/scratch/gpfs/SHAHD/lh2004/cache
export WANDB_MODE=offline
export SWANLAB_MODE=disabled

# --------- Update these three paths ----------
DROID_HF_PATH=/scratch/gpfs/SHAHD/lh2004/droid/cadene/droid_1.0.1
DROID_OUTPUT_PATH=/scratch/gpfs/SHAHD/lh2004/droid/ctrl-world/dataset_example/droid_2
SVD_PATH=/scratch/gpfs/SHAHD/lh2004/models/svd
# ---------------------------------------------
NUM_SHARDS=16
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

cd /home/lh2004/Ctrl-World_patched

echo "[extract_latent] start $(date -Is)"
echo "[extract_latent] input=${DROID_HF_PATH}"
echo "[extract_latent] output=${DROID_OUTPUT_PATH}"
echo "[extract_latent] svd=${SVD_PATH}"
echo "[extract_latent] shard=${SHARD_ID}/${NUM_SHARDS}"

# extract_latent.py is GPU-heavy (VAE encode) and runs with batch_size=1, num_workers=0.
# Keep 1 process / 1 GPU; extra GPUs would not be used by current code.
accelerate launch \
  --num_processes 1 \
  dataset_example/extract_latent.py \
  --droid_hf_path "${DROID_HF_PATH}" \
  --droid_output_path "${DROID_OUTPUT_PATH}" \
  --svd_path "${SVD_PATH}" \
  --encode_batch_size 512 \
  --shard_id "${SHARD_ID}" \
  --num_shards "${NUM_SHARDS}"

echo "[extract_latent] end $(date -Is)"
