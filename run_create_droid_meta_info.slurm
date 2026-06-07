#!/bin/bash
#SBATCH --job-name=droid_meta_info
#SBATCH --output=/scratch/gpfs/SHAHD/lh2004/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/SHAHD/lh2004/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G

module load anaconda3/2025.12
conda activate ctrl-world

cd /home/lh2004/Ctrl-World_patched

python dataset_meta_info/create_cross_meta_info.py \
  --droid_output_paths \
    /scratch/gpfs/SHAHD/lh2004/droid/ctrl-world/dataset_example/droid \
  --dataset_names droid \
  --output_dataset_name droid \
  --max_workers 8
