#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --job-name=ctrl_train_wm_cross_8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:8
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/wm_cotrain/%x-%j.out

module load anaconda3/2025.12
conda activate ctrl-world

export HF_HOME=/scratch/gpfs/SHAHD/lh2004/cache/huggingface
export TORCH_HOME=/scratch/gpfs/SHAHD/lh2004/cache/torch
export XDG_CACHE_HOME=/scratch/gpfs/SHAHD/lh2004/cache

# train_wm.py uses Accelerate log_with='wandb'; offline avoids login on clusters.
export WANDB_MODE=offline
export SWANLAB_MODE=disabled
# H200 has NVLink/NVSwitch — leave P2P enabled (unlike our local workstation
# where we set NCCL_P2P_DISABLE=1). NCCL_DEBUG=WARN is cheap diagnostic.
export NCCL_DEBUG=WARN

cd /home/lh2004/Ctrl-World_patched

# Hyperparameters come from config.py defaults (5e-5 peak LR, 3000-step warmup,
# cosine to 10%, bs=8/rank x 2 accum x 8 ranks = 128 effective batch, bf16,
# torch.compile via TorchDynamoPlugin, DDP static_graph + gradient_as_bucket_view,
# weight_decay=0.01 on AdamW, max_train_steps=100000). Only paths and dataset
# selection are overridden via CLI here. Pass --learning_rate /
# --train_batch_size / etc. explicitly if you want to deviate.
accelerate launch \
  --main_process_port 29501 \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  scripts/train_wm.py \
  --dataset_root_path /scratch/gpfs/SHAHD/lh2004 \
  --dataset_meta_info_path /scratch/gpfs/SHAHD/lh2004/Ctrl-World/dataset_meta_info \
  --dataset_names droid+molmobot_franka_pick_and_place_omni \
  --dataset_cfgs droid+molmobot_franka_pick_and_place_omni \
  --dataset_stat_cfgs droid_molmobot_cross_zshift058 \
  --dataset_sampling_mode manual \
  --dataset_sampling_probs 0.1,0.9 \
  --svd_model_path /scratch/gpfs/SHAHD/lh2004/models/svd \
  --clip_model_path /scratch/gpfs/SHAHD/lh2004/models/clip \
  --ckpt_path /scratch/gpfs/SHAHD/lh2004/models/ctrl-world/checkpoint-10000.pt
