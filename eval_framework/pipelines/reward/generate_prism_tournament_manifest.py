import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys

_EFW = Path(__file__).resolve().parents[2]
if str(_EFW / "lib") not in sys.path:
    sys.path.insert(0, str(_EFW / "lib"))
from layout import (
    gt_dir,
    gt_video_relpath,
    manifests_dir,
    parse_checkpoint_spec,
    prediction_video_relpath,
    predictions_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate balanced pairwise manifest for prism tournament.")
    p.add_argument("--root", type=Path, default=Path("eval_framework"))
    p.add_argument("--category", type=str, default="prism")
    p.add_argument("--trial", type=str, default="trial_0")
    p.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Checkpoint selectors: run_id/checkpoint-<step> "
            "(e.g. ctrl-world/checkpoint-10000, samples_20260507-010313/checkpoint-20000)."
        ),
    )
    p.add_argument(
        "--task",
        type=str,
        default="overall video quality",
        help="Task/instruction text passed to Robometer preference head.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (default: <root>/outputs/manifests/reward_pairwise_manifest_prism_tournament.jsonl).",
    )
    return p.parse_args()


def list_samples_and_views(gt_category_dir: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for sample_dir in sorted([p for p in gt_category_dir.iterdir() if p.is_dir()]):
        for view_file in sorted(sample_dir.glob("*.mp4")):
            rows.append((sample_dir.name, view_file.stem))
    return rows


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = manifests_dir(args.root) / "reward_pairwise_manifest_prism_tournament.jsonl"
    gt_category_dir = gt_dir(args.root) / args.category
    if not gt_category_dir.exists():
        raise FileNotFoundError(f"GT category dir not found: {gt_category_dir}")

    model_tags: List[str] = []
    for spec in args.checkpoints:
        run_id, checkpoint_dir = parse_checkpoint_spec(spec)
        tag = f"{run_id}/{checkpoint_dir}"
        ckpt_root = predictions_dir(args.root) / run_id / checkpoint_dir / args.category / args.trial
        if not ckpt_root.exists():
            raise FileNotFoundError(f"Checkpoint trial dir not found: {ckpt_root}")
        model_tags.append(tag)

    unit_rows = list_samples_and_views(gt_category_dir)
    if not unit_rows:
        raise RuntimeError(f"No sample/view videos found under {gt_category_dir}")

    anchor_tag = "gt"
    pair_tags: List[Tuple[str, str, str]] = []
    for i in range(len(model_tags)):
        for j in range(i + 1, len(model_tags)):
            pair_tags.append((model_tags[i], model_tags[j], "model_vs_model"))
    for m in model_tags:
        pair_tags.append((m, anchor_tag, "model_vs_gt"))

    out_rows: List[Dict] = []
    pair_counter = 0
    for sample_id, view_id in unit_rows:
        for left_tag, right_tag, pair_type in pair_tags:

            def _model_rel(tag: str) -> str:
                run_id, checkpoint_dir = parse_checkpoint_spec(tag)
                return prediction_video_relpath(
                    run_id, checkpoint_dir, args.category, args.trial, sample_id, view_id
                )

            if right_tag == anchor_tag:
                left_rel = _model_rel(left_tag)
                right_rel = gt_video_relpath(args.category, sample_id, view_id)
            else:
                left_rel = _model_rel(left_tag)
                right_rel = _model_rel(right_tag)

            out_rows.append(
                {
                    "pair_id": f"pair_{pair_counter:07d}",
                    "category": args.category,
                    "sample_id": sample_id,
                    "view_id": view_id,
                    "task": args.task,
                    "order": "forward",
                    "pair_type": pair_type,
                    "left_tag": left_tag,
                    "right_tag": right_tag,
                    "left_video_relpath": left_rel,
                    "right_video_relpath": right_rel,
                    "meta": {"trial": args.trial},
                }
            )
            pair_counter += 1

            out_rows.append(
                {
                    "pair_id": f"pair_{pair_counter:07d}",
                    "category": args.category,
                    "sample_id": sample_id,
                    "view_id": view_id,
                    "task": args.task,
                    "order": "reverse",
                    "pair_type": pair_type,
                    "left_tag": right_tag,
                    "right_tag": left_tag,
                    "left_video_relpath": right_rel,
                    "right_video_relpath": left_rel,
                    "meta": {"trial": args.trial},
                }
            )
            pair_counter += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote tournament manifest: {args.out}")
    print(f"Total pairs: {len(out_rows)}")


if __name__ == "__main__":
    main()
