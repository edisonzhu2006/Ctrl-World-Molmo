#!/bin/bash
#SBATCH --job-name=ctrl_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/%x-%j.out

module load anaconda3/2025.12
conda activate ctrl-world

export HF_HOME=/scratch/gpfs/SHAHD/lh2004/cache/huggingface
export TORCH_HOME=/scratch/gpfs/SHAHD/lh2004/cache/torch
export XDG_CACHE_HOME=/scratch/gpfs/SHAHD/lh2004/cache

cd /home/lh2004/Ctrl-World_patched

python scripts/rollout_replay_traj.py \
  --dataset_root_path /scratch/gpfs/SHAHD/lh2004/molmo2ctrl/output \
  --dataset_meta_info_path /scratch/gpfs/SHAHD/lh2004/Ctrl-World/dataset_meta_info \
  --dataset_names molmobot_franka_pick_and_place_omni \
  --data_stat_path /scratch/gpfs/SHAHD/lh2004/Ctrl-World/dataset_meta_info/molmobot_franka_pick_and_place_omni/stat.json \
  --svd_model_path /scratch/gpfs/SHAHD/lh2004/models/svd \
  --clip_model_path /scratch/gpfs/SHAHD/lh2004/models/clip \
  --ckpt_path /scratch/gpfs/SHAHD/lh2004/model_ckpt/molmobot_subset/samples_20260504-105539/checkpoint-50000.pt