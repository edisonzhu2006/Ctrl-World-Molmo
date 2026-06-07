import json
import os
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_and_process_ann_file(data_root, ann_file, sequence_interval=1, start_interval=4, sequence_length=8):
    samples = []
    try:
        with open(f"{data_root}/{ann_file}", "r") as f:
            ann = json.load(f)
    except Exception:
        print(f"skip {data_root}/{ann_file}")
        return samples

    n_frames = ann["video_length"]
    traj_len = int(sequence_length * sequence_interval)
    end_idx = n_frames - int(traj_len * 0.5)
    if end_idx < 1:
        end_idx = 1

    for start_frame in range(0, end_idx, start_interval):
        idx = start_frame
        sample = dict()
        sample["episode_id"] = ann["episode_id"]
        sample["frame_ids"] = [idx]
        sample["states"] = np.array(ann["states"])[idx : idx + 1]
        samples.append(sample)
    return samples


def init_anns(dataset_root, data_dir):
    final_path = f"{dataset_root}/{data_dir}"
    if not os.path.isdir(final_path):
        print(f"skip missing dir: {final_path}")
        return []
    return [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith(".json")]


def init_sequences(data_root, ann_files, sequence_interval, start_interval, sequence_length, max_workers=32):
    samples = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ann_file = {
            executor.submit(
                load_and_process_ann_file,
                data_root,
                ann_file,
                sequence_interval,
                start_interval,
                sequence_length,
            ): ann_file
            for ann_file in ann_files
        }
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples


def normalize_name(path):
    return os.path.basename(os.path.abspath(path.rstrip("/")))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--droid_output_paths",
        nargs="+",
        required=True,
        help="List of dataset roots, each containing annotation/train and annotation/val",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=None,
        help="Optional list of dataset labels matching --droid_output_paths order.",
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        default="cross_dataset",
        help="Output directory. Relative paths are written under dataset_meta_info/; absolute paths are used as-is.",
    )
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--sequence_interval", type=int, default=1)
    parser.add_argument("--start_interval", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_names is not None and len(args.dataset_names) > 0:
        if len(args.dataset_names) != len(args.droid_output_paths):
            raise ValueError("len(--dataset_names) must equal len(--droid_output_paths)")
        dataset_names = args.dataset_names
    else:
        dataset_names = [normalize_name(path) for path in args.droid_output_paths]

    random.seed(args.seed)

    output_dataset_path = Path(args.output_dataset_name)
    if output_dataset_path.is_absolute():
        output_dir = output_dataset_path
    else:
        output_dir = Path("dataset_meta_info") / output_dataset_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build merged metadata for each split.
    for data_type in ["train", "val"]:
        samples_all = []
        ann_files_all = []

        for data_root, dataset_name in zip(args.droid_output_paths, dataset_names):
            ann_dir = f"annotation/{data_type}"
            ann_files = init_anns(data_root, ann_dir)
            ann_files_all.extend([(dataset_name, ann_file) for ann_file in ann_files])

            samples = init_sequences(
                data_root,
                ann_files,
                args.sequence_interval,
                args.start_interval,
                args.sequence_length,
                max_workers=args.max_workers,
            )
            for s in samples:
                s["dataset_name"] = dataset_name
                s["dataset_root"] = data_root
            print(f"{dataset_name} ({data_root}) {data_type}: {len(samples)} samples")
            samples_all.extend(samples)

        # 1% / 99% percentiles from merged TRAIN split only.
        if data_type == "train":
            print("########################### cross state ###########################")
            if len(samples_all) == 0:
                print("[warn] no train samples; skipping stat.json")
            else:
                state_all = []
                for s in samples_all:
                    state = np.asarray(s["states"], dtype=np.float64).squeeze(0)
                    state_all.append(state)
                state_all = np.stack(state_all, axis=0)
                print("state_all.shape", state_all.shape)
                state_01 = np.percentile(state_all, 1, axis=0)
                state_99 = np.percentile(state_all, 99, axis=0)
                stat = {
                    "state_01": state_01.tolist(),
                    "state_99": state_99.tolist(),
                    "dataset_names": dataset_names,
                    "dataset_roots": args.droid_output_paths,
                }
                stat_path = output_dir / "stat.json"
                with open(stat_path, "w") as f:
                    json.dump(stat, f)
                print(f"wrote {stat_path}")

        for sample in samples_all:
            del sample["states"]
        random.shuffle(samples_all)

        print("step_num", data_type, len(samples_all))
        print("traj_num", data_type, len(ann_files_all))
        split_path = output_dir / f"{data_type}_sample.json"
        with open(split_path, "w") as f:
            json.dump(samples_all, f, indent=4)
        print(f"wrote {split_path}")
