import argparse
import json
import sys
from pathlib import Path

import numpy as np

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import resolve_relpath

from robometer.evals.baselines.rbm_model import RBMModel
from robometer.evals.eval_viz_utils import extract_frames
from robometer.data.dataset_types import Trajectory, PreferenceSample


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/lh2004/Ctrl-World_patched/eval_framework"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_framework/outputs/manifests/reward_pairwise_manifest.jsonl"),
    )
    p.add_argument("--model-path", type=str, default="robometer/Robometer-4B")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--max-frames", type=int, default=128)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "eval_framework/outputs/results/aggregated/reward/reward_pairwise_builtin.jsonl"
        ),
    )
    args = p.parse_args()

    model = RBMModel(checkpoint_path=args.model_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as wf:
        for row in read_jsonl(args.manifest):
            left_path = resolve_relpath(args.root, row["left_video_relpath"])
            right_path = resolve_relpath(args.root, row["right_video_relpath"])
            task = row.get("task", "overall video quality")

            left_frames = extract_frames(str(left_path), fps=args.fps, max_frames=args.max_frames)
            right_frames = extract_frames(str(right_path), fps=args.fps, max_frames=args.max_frames)
            if left_frames is None or right_frames is None:
                raise RuntimeError(f"Failed to extract frames for pair_id={row.get('pair_id')}")

            # Built-in binary preference: chosen(left) vs rejected(right)
            sample = PreferenceSample(
                chosen_trajectory=Trajectory(
                    frames=left_frames,
                    frames_shape=tuple(left_frames.shape),
                    task=task,
                    id=f"{row.get('pair_id','pair')}_left",
                    metadata={"subsequence_length": int(left_frames.shape[0])},
                ),
                rejected_trajectory=Trajectory(
                    frames=right_frames,
                    frames_shape=tuple(right_frames.shape),
                    task=task,
                    id=f"{row.get('pair_id','pair')}_right",
                    metadata={"subsequence_length": int(right_frames.shape[0])},
                ),
            )

            result = model.compute_batched_preference([sample])[0]
            # preference_pred: 1.0 means left preferred, 0.0 means right preferred
            pred = "left" if float(result["preference_pred"]) >= 0.5 else "right"

            out_row = {
                **row,
                "prediction_prob_left_over_right": float(result["prediction_prob"]),
                "preference_logit": float(result["preference_logits"]),
                "pred_preference": pred,
                "raw_result": result,
            }
            wf.write(json.dumps(out_row) + "\n")

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()