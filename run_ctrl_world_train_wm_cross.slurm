#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --job-name=cw-cotrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/wm_cotrain/%x-%j.out

module load anaconda3/2025.12
conda activate ctrl-world

export HF_HOME=/scratch/gpfs/SHAHD/lh2004/cache/huggingface
export TORCH_HOME=/scratch/gpfs/SHAHD/lh2004/cache/torch
export XDG_CACHE_HOME=/scratch/gpfs/SHAHD/lh2004/cache

# train_wm.py uses Accelerate log_with='wandb'; offline avoids login on clusters
export WANDB_MODE=offline
export SWANLAB_MODE=disabled

cd /home/lh2004/Ctrl-World_patched

# Optional: every train minibatch contains both datasets (counts follow dataset_names order).
#   --dataset_mix_batch --dataset_mix_counts 2,2 --dataset_mix_seed 0

accelerate launch \
  --main_process_port 29501 \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  scripts/train_wm.py \
  --dataset_root_path /scratch/gpfs/SHAHD/lh2004 \
  --dataset_meta_info_path /scratch/gpfs/SHAHD/lh2004/Ctrl-World/dataset_meta_info \
  --dataset_names droid+molmobot_franka_pick_and_place_omni \
  --dataset_cfgs droid+molmobot_franka_pick_and_place_omni \
  --dataset_stat_cfgs droid_molmobot_cross \
  --dataset_sampling_mode manual \
  --dataset_sampling_probs 0.1,0.9 \
  --num_workers 4 \
  --svd_model_path /scratch/gpfs/SHAHD/lh2004/models/svd \
  --clip_model_path /scratch/gpfs/SHAHD/lh2004/models/clip \
  --ckpt_path /scratch/gpfs/SHAHD/lh2004/models/ctrl-world/checkpoint-10000.pt

# Other examples:
# --dataset_sampling_mode manual --dataset_sampling_probs 0.3,0.7
# --dataset_sampling_mode prorated_samples
# --dataset_sampling_mode prorated_trajectories

