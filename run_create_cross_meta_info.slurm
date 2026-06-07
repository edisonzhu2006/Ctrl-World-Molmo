#!/bin/bash
#SBATCH --job-name=cross_meta_info
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/SHAHD/lh2004/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G

module load anaconda3/2025.12
conda activate ctrl-world

cd /home/lh2004/Ctrl-World_patched

python dataset_meta_info/create_cross_meta_info.py \
  --droid_output_paths \
    /scratch/gpfs/SHAHD/lh2004/droid/ctrl-world/dataset_example/droid \
    /scratch/gpfs/SHAHD/lh2004/molmo2ctrl/output/molmobot_franka_pick_and_place_omni \
  --dataset_names droid molmobot \
  --output_dataset_name /scratch/gpfs/SHAHD/lh2004/Ctrl-World/dataset_meta_info/droid_molmobot_cross \
  --max_workers 8
