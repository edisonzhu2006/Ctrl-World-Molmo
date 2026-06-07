# Human Preference Evaluation (qualitative)

Web app for side-by-side human judgments: context (optional), ground truth, and two generated videos (A/B). Model names are hidden; left/right order is shuffled per session to reduce position bias.

## Refined design (minimal)

| Piece | Choice |
|-------|--------|
| Stack | FastAPI + Jinja + vanilla JS (no React) |
| Storage | SQLite (`data/judgments.sqlite`) |
| Manifest | JSONL, paths relative to `eval_framework/` |
| Position bias | Deterministic swap per `(session_id, pair_id)` |
| Duplicate guard | `UNIQUE(session_id, pair_id)` |
| Analysis | `export_results.py` → CSV → `evaluate_results.py` |

## Manifest schema (JSONL, one object per line)

Required fields:

- `pair_id`, `gt_relpath`, `video_a_relpath`, `video_b_relpath`, `model_a`, `model_b`

Optional: `category`, `sample_id`, `view_id`, `action_id`, `context_relpath` (image or video), `metric_a`, `metric_b` (float or `{"psnr": ...}`), `meta`.

Example:

```json
{
  "pair_id": "pair_0000000",
  "category": "prism",
  "sample_id": "sample_0",
  "view_id": "view0",
  "gt_relpath": "gt/prism/sample_0/view0.mp4",
  "model_a": "ctrl-world/checkpoint-10000",
  "model_b": "samples_20260427-031822/checkpoint-50000",
  "video_a_relpath": "predictions/ctrl-world/checkpoint-10000/prism/trial_0/sample_0/view0.mp4",
  "video_b_relpath": "predictions/samples_20260427-031822/checkpoint-50000/prism/trial_0/sample_0/view0.mp4",
  "metric_a": {"psnr": 24.0},
  "metric_b": {"psnr": 26.0}
}
```

## Install & run

From repository root:

```bash
pip install -r eval_framework/apps/human_eval/requirements.txt

cd eval_framework/apps/human_eval
python app.py
```

Default: `http://127.0.0.1:18765/` (admin: `/admin`). Port **8765** is commonly taken on shared cluster nodes.

### Cluster login node (e.g. Della)

Login nodes should bind to localhost only, then tunnel from your laptop:

```bash
# On the cluster
cd eval_framework/apps/human_eval
python app.py --host 127.0.0.1 --port 18765
# If the port is busy:
python app.py --port 0   # prints the chosen port
```

```bash
# On your laptop (replace host/user)
ssh -L 18765:localhost:18765 lh2004@della-gpu.princeton.edu
```

Open `http://127.0.0.1:18765/` in your **local** browser (not on the cluster).

To see what is using a port: `ss -tlnp | grep 18765`

### Share with others via Cloudflare quick tunnel (`*.trycloudflare.com`)

Good for sending a link to annotators without SSH. Cloudflare gives a random public URL each run (e.g. `https://donation-known-freedom-benefit.trycloudflare.com`).

**1. Install `cloudflared` once** (on the machine where the app runs — cluster login node or your laptop):

```bash
mkdir -p ~/bin
curl -L -o ~/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/bin/cloudflared
export PATH="$HOME/bin:$PATH"
```

**2. Start app + tunnel** (keep the terminal open; use `tmux`/`screen` on the cluster):

```bash
cd eval_framework/apps/human_eval
bash scripts/share_via_cloudflare.sh
```

Or pass app options after `--port`:

```bash
bash scripts/share_via_cloudflare.sh --port 18765 --manifest data/manifest_prism.jsonl
```

**3. Copy the `https://….trycloudflare.com` URL** from the `cloudflared` log and share it. Optional per-person id:

`https://….trycloudflare.com/?user_id=alice`

**Caveats**

| Topic | Note |
|-------|------|
| URL lifetime | New random hostname every time you restart `cloudflared` |
| Session | Tunnel dies if the shell/job stops — use `tmux` on HPC |
| Security | Anyone with the link can annotate; no login by default |
| Videos | Large MP4s stream through Cloudflare; may be slow on first load |
| HPC policy | Confirm login nodes allow long jobs + outbound HTTPS |
| Data | Judgments stay in `data/judgments.sqlite` on the server |

Admin/export still works at `https://….trycloudflare.com/admin` (restrict who you share that with).

Custom paths:

```bash
python app.py \
  --root /path/to/eval_framework \
  --manifest /path/to/manifest.jsonl \
  --db /path/to/judgments.sqlite
```

## Generate manifest from checkpoints

```bash
python eval_framework/apps/human_eval/scripts/generate_manifest.py \
  --root eval_framework \
  --category prism \
  --trial trial_0 \
  --checkpoints ctrl-world/checkpoint-10000 samples_20260427-031822/checkpoint-50000 \
  --out eval_framework/apps/human_eval/data/manifest_prism.jsonl
```

Optional metrics from PSNR CSV:

```bash
python eval_framework/apps/human_eval/scripts/generate_manifest.py \
  --root eval_framework \
  --category prism \
  --trial trial_0 \
  --checkpoints ctrl-world/checkpoint-10000 samples_20260427-031822/checkpoint-50000 \
  --metrics-csv eval_framework/outputs/results/aggregated/metrics/metrics_samples_all_checkpoints.csv \
  --metric-key psnr \
  --out eval_framework/apps/human_eval/data/manifest_prism.jsonl
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Annotation UI |
| `GET /admin` | Counts by session / preference |
| `GET /api/next?session_id=` | Next unannotated pair (creates session if needed) |
| `POST /api/submit` | Save judgment JSON body |
| `GET /api/progress?session_id=` | Annotated / total |
| `GET /api/export?format=jsonl\|csv` | Download judgments |
| `GET /media/{relpath}` | Serve video/image under `--root` |

Keyboard: `1` A, `2` B, `3` tie, `4` invalid, `Space` play/pause all.

## Export & evaluate

```bash
python eval_framework/apps/human_eval/scripts/export_results.py \
  --db eval_framework/apps/human_eval/data/judgments.sqlite \
  --manifest eval_framework/apps/human_eval/data/manifest_demo.jsonl \
  --root eval_framework \
  --out eval_framework/apps/human_eval/data/exports/judgments_export.csv

python eval_framework/apps/human_eval/scripts/evaluate_results.py \
  --input eval_framework/apps/human_eval/data/exports/judgments_export.csv \
  --out-dir eval_framework/apps/human_eval/data/exports
```

Outputs: win rate by model, pairwise win matrix, Pearson(metric_diff, human_pref_binary), metric-vs-human confusion / accuracy, optional inter-rater agreement.

## Demo data

`data/manifest_demo.jsonl` uses two prism pairs under `eval_framework/data/gt/` and `eval_framework/data/predictions/` (existing rollout videos).
