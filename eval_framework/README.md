# Eval Framework

Standalone evaluation workspace for checkpoint rollout evaluation.

This folder is intentionally isolated from the rest of the repository so you can add new evaluation features without modifying training or rollout code.

## Top-level layout (5 folders)

```
eval_framework/
  data/                         # inputs written by rollout
    gt/prism/sample_<i>/view<j>.mp4
    predictions/<run_id>/checkpoint-<step>/prism/trial_<n>/...
  outputs/                      # pipeline artifacts
    results/raw/<run_id>/checkpoint-<step>/*.csv
    results/aggregated/metrics/   # pool CSVs + per-run/comparison figures
      metrics_samples_all_checkpoints.csv
      metrics_by_checkpoint.csv
      fid_fvd_by_checkpoint.csv
      overview/figures/  overview/tables/   # all runs
      runs/<run_id>/figures/  runs/<run_id>/tables/
      runs/<run_id>/comparisons/<slug>/figures/  .../tables/
    results/aggregated/reward/    # Robometer tournament outputs
    manifests/                    # Robometer pairwise JSONL manifests
    logs/reward_eval/             # SLURM stdout/stderr
  pipelines/
    metrics/                      # PSNR/SSIM/LPIPS, FID/FVD, analysis
    reward/                       # Robometer pairwise tournament
  apps/
    human_eval/                   # human preference web app
    dashboard/                    # Streamlit dashboard
  lib/
    layout.py                     # path conventions (import from eval_framework/lib)
    scripts/                      # calibration utilities, one-off migrations
  vendor/
    robometer/                    # vendored Robometer
```

### Path conventions

- **`<run_id>`** — training-run folder name (parent of `ckpts/`, or parent of a flat `.pt`). Resolved by `lib/layout.py` from `--ckpt_path` in rollout.
- **`checkpoint-<step>`** — matches the training checkpoint directory name.
- **`ckpt_name`** in metrics CSVs — **`{run_id}/{checkpoint-<step>}`**.
- **Manifest / CSV relpaths** — relative to `eval_framework/` root, using `data/gt/...` and `data/predictions/...`. Legacy `gt/` and `predictions/` prefixes are still resolved by `layout.resolve_relpath`.
- **`trial_<n>`** — one dir per `rollout_replay_traj.py` invocation; metrics default to `trial_0`.

## Metrics pipeline

### PSNR / SSIM / LPIPS — `pipelines/metrics/compute_psnr_ssim_lpips.py`

Samples are discovered from `data/gt/<category>/`. Predictions under `data/predictions/<run_id>/checkpoint-<step>/`. Optional: `--run`, `--steps`.

### FID / FVD — `pipelines/metrics/compute_fid_fvd.py`

I3D weights: `pipelines/metrics/.cache/i3d_torchscript.pt` (auto-download on first run if absent).

Install:

```bash
pip install scikit-image lpips
pip install -r eval_framework/pipelines/metrics/requirements_fid_fvd.txt
```

### Full pipeline (from repo root)

```bash
python eval_framework/pipelines/metrics/compute_psnr_ssim_lpips.py \
  --root eval_framework --category prism --trial trial_0 --device cuda

python eval_framework/pipelines/metrics/compute_fid_fvd.py \
  --root eval_framework --category prism --trial trial_0 --device cuda

python eval_framework/pipelines/metrics/collect_checkpoint_samples.py \
  --root eval_framework \
  --out eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv

python eval_framework/pipelines/metrics/analyze_checkpoint_metrics.py \
  --root eval_framework \
  --samples-csv eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv
```

Overview plots land in `outputs/results/aggregated/metrics/overview/`. For one training run:

```bash
python eval_framework/pipelines/metrics/analyze_checkpoint_metrics.py \
  --root eval_framework \
  --samples-csv eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv \
  --run samples_20260521-181822-8560524
```

Or `sbatch pipelines/metrics/analyze_run_metrics.slurm`.

Or submit `pipelines/metrics/run_prism_metrics_analysis.slurm`.

## Reward pipeline (Robometer pairwise tournament)

Scripts in `pipelines/reward/`. Submit `pipelines/reward/run_robometer_pairwise.slurm` (runs from `vendor/robometer`).

Outputs: `outputs/results/aggregated/reward/`; logs: `outputs/logs/reward_eval/`.

## Human evaluation

```bash
pip install -r eval_framework/apps/human_eval/requirements.txt
cd eval_framework/apps/human_eval && python app.py
```

Manifests and exports live under `apps/human_eval/data/` (not `outputs/manifests/`).

## Dashboard

```bash
pip install -r eval_framework/apps/dashboard/requirements.txt
streamlit run eval_framework/apps/dashboard/app.py
```

## Calibration / sanity checks

```bash
python eval_framework/lib/scripts/add_gaussian_noise_to_videos.py \
  --root eval_framework \
  --src-subdir data/gt/prism/sample_0 \
  --dst-subdir data/predictions/my_run/checkpoint-15000/prism/trial_0/sample_0 \
  --sigma 15 --seed 42
```
