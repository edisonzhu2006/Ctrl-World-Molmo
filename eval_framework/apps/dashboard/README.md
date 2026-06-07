# Eval Dashboard (v1)

Interactive one-page viewer for:

- Ground-truth videos
- Predicted videos
- Reward curves (`*_rewards.npy`, `*_rewards_success_probs.npy`)
- Binary pairwise preferences (`reward_pairwise_builtin.jsonl` / `reward_pairwise_results.jsonl`)
- PSNR / SSIM / LPIPS per-sample and per-checkpoint summaries

## Run

From repository root:

```bash
streamlit run eval_framework/apps/dashboard/app.py
```

## Data Sources

- Videos:
  - `eval_framework/data/gt/<category>/<sample>/view*.mp4`
  - `eval_framework/data/predictions/<ckpt>/<category>/<trial?>/<sample>/view*.mp4`
- Metrics:
  - `eval_framework/outputs/results/raw/**/psnr_ssim_lpips_samples.csv`
  - `eval_framework/outputs/results/aggregated/metrics/metrics_by_checkpoint.csv`
- Preferences:
  - `eval_framework/outputs/results/aggregated/reward/reward_pairwise_prism_tournament.jsonl`
  - `eval_framework/outputs/results/aggregated/reward/reward_pairwise_builtin.jsonl`
  - `eval_framework/outputs/results/aggregated/reward/reward_pairwise_results.jsonl`

## Notes

- `trial` selector appears when prediction layout includes `trial_*` folders.
- Reward curves appear only if `.npy` files are present next to selected prediction video.
