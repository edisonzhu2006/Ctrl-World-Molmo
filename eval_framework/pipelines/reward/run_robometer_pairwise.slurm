#!/bin/bash
#SBATCH --job-name=rbm_tournament
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=eval_framework/outputs/logs/reward_eval/rbm_tournament_%j.out
#SBATCH --error=eval_framework/outputs/logs/reward_eval/rbm_tournament_%j.err

set -eo pipefail
cd /home/lh2004/Ctrl-World_patched/eval_framework/vendor/robometer
# Use Robometer's env and reduce fragmentation risk
export HF_HOME=/scratch/gpfs/SHAHD/lh2004/hf_cache
export TRANSFORMERS_CACHE=/scratch/gpfs/SHAHD/lh2004/hf_cache/hub
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/SHAHD/lh2004/hf_cache/hub

# force offline so it never tries internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# avoid allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure matplotlib exists for analysis plotting.
python - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("matplotlib") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
PY

EFW=/home/lh2004/Ctrl-World_patched/eval_framework
MANIFEST=${EFW}/outputs/manifests/reward_pairwise_manifest_prism_tournament.jsonl
REWARD_OUT=${EFW}/outputs/results/aggregated/reward

echo "=== Step 1/3: Generate balanced tournament manifest ==="
uv run python ${EFW}/pipelines/reward/generate_prism_tournament_manifest.py \
  --root ${EFW} \
  --category prism \
  --trial trial_0 \
  --checkpoints \
    ctrl-world/checkpoint-10000 \
    samples_20260427-031822/checkpoint-50000 \
    samples_20260504-105539/checkpoint-50000 \
  --out ${MANIFEST}

echo "=== Step 2/3: Run Robometer built-in pairwise preference ==="
uv run python ${EFW}/pipelines/reward/run_robometer_pairwise_builtin.py \
  --root ${EFW} \
  --manifest ${MANIFEST} \
  --model-path robometer/Robometer-4B \
  --fps 4 \
  --max-frames 100 \
  --out ${REWARD_OUT}/reward_pairwise_prism_tournament.jsonl

echo "=== Step 3/3: Analyze tournament outputs ==="
uv run python ${EFW}/pipelines/reward/analyze_pairwise_tournament.py \
  --input ${REWARD_OUT}/reward_pairwise_prism_tournament.jsonl \
  --out-dir ${REWARD_OUT}

echo "=== Done: tournament outputs in ${REWARD_OUT} ==="
