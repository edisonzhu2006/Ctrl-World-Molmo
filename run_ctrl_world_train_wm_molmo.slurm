#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --job-name=cw_molmo_scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:8
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/wm_molmo/%x-%j.out

module load anaconda3/2025.12
conda activate ctrl-world

export HF_HOME=/scratch/gpfs/SHAHD/lh2004/cache/huggingface
export TORCH_HOME=/scratch/gpfs/SHAHD/lh2004/cache/torch
export XDG_CACHE_HOME=/scratch/gpfs/SHAHD/lh2004/cache

# DDP: avoid oversubscribing CPU with BLAS threads (8 ranks × many threads).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# train_wm.py uses Accelerate log_with='wandb'; offline avoids login on clusters
export WANDB_MODE=offline
export SWANLAB_MODE=disabled

cd /home/lh2004/Ctrl-World_patched

accelerate launch \
  --main_process_port 29501 \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  scripts/train_wm.py \
  --dataset_root_path /scratch/gpfs/SHAHD/lh2004/molmo2ctrl/output \
  --dataset_meta_info_path /scratch/gpfs/SHAHD/lh2004/Ctrl-World/dataset_meta_info \
  --dataset_names molmobot_franka_pick_and_place_omni \
  --dataset_cfgs molmobot_franka_pick_and_place_omni \
  --num_workers 4 \
  --svd_model_path /scratch/gpfs/SHAHD/lh2004/models/svd \
  --clip_model_path /scratch/gpfs/SHAHD/lh2004/models/clip
