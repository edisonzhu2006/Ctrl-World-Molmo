"""Compute the cross-dataset normalization ``stat.json`` only.

A stripped-down version of ``create_cross_meta_info.py`` that skips per-window
sample-dict construction, the shuffle, and the multi-GB
``train_sample.json`` / ``val_sample.json`` writes. It produces a ``stat.json``
that is bit-identical to what the full script would write when run on the same
inputs with the same hyperparameters (``sequence_length=8``,
``sequence_interval=1``, ``start_interval=1``).

Walks ``<dataset_root>/annotation/<split>/*.json`` for each dataset root, reads
``ann["states"]``, keeps the first ``end_idx = n_frames - sequence_length//2``
rows per episode (mirroring the parent script's sliding-window mask), then takes
the 1%/99% percentiles per dim over the pooled rows.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm


DEFAULT_DROID = "/scratch/gpfs/SHAHD/lh2004/droid/ctrl-world/dataset_example/droid"
DEFAULT_MOLMO = (
    "/scratch/gpfs/SHAHD/lh2004/molmo2ctrl/output/molmobot_franka_pick_and_place_omni"
)


def load_episode_states(
    ann_path: str,
    sequence_length: int,
    sequence_interval: int,
    start_interval: int,
) -> Optional[np.ndarray]:
    with open(ann_path, "r") as f:
        ann = json.load(f)
    if "states" not in ann or "video_length" not in ann:
        return None
    n_frames = int(ann["video_length"])
    traj_len = int(sequence_length * sequence_interval)
    end_idx = n_frames - int(traj_len * 0.5)
    if end_idx < 1:
        end_idx = 1
    states = np.asarray(ann["states"], dtype=np.float64)
    if states.ndim != 2:
        return None
    idxs = np.arange(0, end_idx, start_interval)
    idxs = idxs[idxs < states.shape[0]]
    if idxs.size == 0:
        return None
    return states[idxs]


def collect_dataset_states(
    data_root: str,
    split: str,
    sequence_length: int,
    sequence_interval: int,
    start_interval: int,
    max_workers: int,
) -> np.ndarray:
    ann_dir = os.path.join(data_root, "annotation", split)
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Missing annotation dir: {ann_dir}")

    ann_files = []
    with os.scandir(ann_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".json"):
                ann_files.append(entry.path)
    if not ann_files:
        raise RuntimeError(f"No .json files under {ann_dir}")

    chunks: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                load_episode_states,
                p,
                sequence_length,
                sequence_interval,
                start_interval,
            )
            for p in ann_files
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=os.path.basename(os.path.abspath(data_root)),
        ):
            arr = fut.result()
            if arr is not None:
                chunks.append(arr)

    if not chunks:
        raise RuntimeError(f"Collected 0 valid state rows from {ann_dir}")
    return np.concatenate(chunks, axis=0)


def normalize_name(path: str) -> str:
    return os.path.basename(os.path.abspath(path.rstrip("/")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--droid_output_paths",
        nargs="+",
        default=[DEFAULT_DROID, DEFAULT_MOLMO],
        help="Dataset roots, each containing annotation/<split>/*.json",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=None,
        help="Optional labels matching --droid_output_paths order.",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default="droid_molmobot_cross",
        help="Output dir; relative under dataset_meta_info/, absolute as-is.",
    )
    parser.add_argument("--annotation_split", type=str, default="train")
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--sequence_interval", type=int, default=1)
    parser.add_argument("--start_interval", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    if args.dataset_names is None or len(args.dataset_names) == 0:
        dataset_names = [normalize_name(p) for p in args.droid_output_paths]
    else:
        if len(args.dataset_names) != len(args.droid_output_paths):
            raise ValueError(
                "len(--dataset_names) must equal len(--droid_output_paths)"
            )
        dataset_names = list(args.dataset_names)

    out_path = Path(args.output_dataset_name)
    out_dir = out_path if out_path.is_absolute() else Path("dataset_meta_info") / out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"split            = {args.annotation_split}")
    print(f"sequence_length  = {args.sequence_length}")
    print(f"sequence_interval= {args.sequence_interval}")
    print(f"start_interval   = {args.start_interval}")
    print(f"max_workers      = {args.max_workers}")
    print(f"output_dir       = {out_dir}")

    all_states: List[np.ndarray] = []
    for data_root, ds_name in zip(args.droid_output_paths, dataset_names):
        print(f"\n[{ds_name}] root={data_root}")
        states = collect_dataset_states(
            data_root,
            args.annotation_split,
            args.sequence_length,
            args.sequence_interval,
            args.start_interval,
            args.max_workers,
        )
        print(f"[{ds_name}] collected {states.shape}")
        all_states.append(states)

    state_all = np.concatenate(all_states, axis=0)
    print(f"\nstate_all.shape = {state_all.shape}")

    state_01 = np.percentile(state_all, 1, axis=0)
    state_99 = np.percentile(state_all, 99, axis=0)

    stat = {
        "state_01": state_01.tolist(),
        "state_99": state_99.tolist(),
        "dataset_names": dataset_names,
        "dataset_roots": list(args.droid_output_paths),
    }
    stat_path = out_dir / "stat.json"
    tmp_path = stat_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(stat, f)
    os.replace(tmp_path, stat_path)

    print(f"\nstate_01 = {stat['state_01']}")
    print(f"state_99 = {stat['state_99']}")
    print(f"wrote {stat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
