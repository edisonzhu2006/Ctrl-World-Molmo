#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --job-name=ctrl_extract_latent_droid_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/%x-%j.out

module load anaconda3/2025.12
conda activate ctrl-world

export HF_HOME=/scratch/gpfs/SHAHD/lh2004/cache/huggingface
export TORCH_HOME=/scratch/gpfs/SHAHD/lh2004/cache/torch
export XDG_CACHE_HOME=/scratch/gpfs/SHAHD/lh2004/cache
export WANDB_MODE=offline
export SWANLAB_MODE=disabled

DROID_NEW_SETUP_FULL=/home/lh2004/Ctrl-World_patched/dataset_example/droid_new_setup_full
SVD_PATH=/scratch/gpfs/SHAHD/lh2004/models/svd

# One subdir per task; each has annotation/ + videos/ like droid_new_setup
TASK_TYPES=(drawer laptop pickplace stack)

cd /home/lh2004/Ctrl-World_patched

echo "[extract_latent_droid_full] start $(date -Is)"
echo "[extract_latent_droid_full] root=${DROID_NEW_SETUP_FULL}"
echo "[extract_latent_droid_full] tasks=${TASK_TYPES[*]}"
echo "[extract_latent_droid_full] svd=${SVD_PATH}"

for TASK in "${TASK_TYPES[@]}"; do
  DATASET_PATH="${DROID_NEW_SETUP_FULL}/${TASK}"
  echo "[extract_latent_droid_full] task=${TASK} path=${DATASET_PATH}"

  accelerate launch \
    --num_processes 1 \
    dataset_example/extract_latent_molmobot.py \
    --molmobot_path "${DATASET_PATH}" \
    --svd_model_path "${SVD_PATH}" \
    --rgb_skip 1 \
    --shard_id 0 \
    --num_shards 1
done

echo "[extract_latent_droid_full] end $(date -Is)"