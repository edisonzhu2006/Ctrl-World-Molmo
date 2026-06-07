#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --job-name=ctrl_extract_latent_molmobot
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

# --------- Update these paths ----------
MOLMOBOT_PATH=/scratch/gpfs/SHAHD/lh2004/molmo2ctrl/output/molmobot_franka_pick_and_place_omni
SVD_PATH=/scratch/gpfs/SHAHD/lh2004/models/svd
# ---------------------------------------
NUM_SHARDS=16
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}

cd /home/lh2004/Ctrl-World_patched

echo "[extract_latent_molmobot] start $(date -Is)"
echo "[extract_latent_molmobot] input=${MOLMOBOT_PATH}"
echo "[extract_latent_molmobot] svd=${SVD_PATH}"
echo "[extract_latent_molmobot] shard=${SHARD_ID}/${NUM_SHARDS}"

# extract_latent_molmobot.py is GPU-heavy (VAE encode) and runs with batch_size=1, num_workers=0.
# Keep 1 process / 1 GPU per array task; the SLURM array parallelizes independent shards.
accelerate launch \
  --num_processes 1 \
  dataset_example/extract_latent_molmobot.py \
  --molmobot_path "${MOLMOBOT_PATH}" \
  --svd_model_path "${SVD_PATH}" \
  --rgb_skip 1 \
  --shard_id "${SHARD_ID}" \
  --num_shards "${NUM_SHARDS}"

echo "[extract_latent_molmobot] end $(date -Is)"