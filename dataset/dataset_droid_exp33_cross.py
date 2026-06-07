import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_mix(Dataset):
    def __init__(self, args, mode="val"):
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset structure (expected by __getitem__)
        # dataset_root/<annotation_name>/<mode>/<episode_id>.json
        # dataset_root/... latent/video paths in annotation file

        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []
        self.dataset_debug_rows = []
        # Positional indices (into dataset_names) of datasets actually kept; val
        # may skip datasets with an empty split, so sampling probs are mapped back
        # to these in _build_sampling_prob.
        self.kept_indices = []

        dataset_root_path = args.dataset_root_path
        dataset_names = self._split_arg_list(args.dataset_names)
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = self._split_arg_list(args.dataset_cfgs)
        dataset_stat_cfgs = self._split_arg_list(getattr(args, "dataset_stat_cfgs", None))
        traj_counts = []

        if len(dataset_names) != len(dataset_cfgs):
            raise ValueError(
                f"dataset_names ({len(dataset_names)}) and dataset_cfgs ({len(dataset_cfgs)}) must match"
            )
        if len(dataset_stat_cfgs) == 0:
            dataset_stat_cfgs = list(dataset_cfgs)
        elif len(dataset_stat_cfgs) == 1 and len(dataset_names) > 1:
            dataset_stat_cfgs = dataset_stat_cfgs * len(dataset_names)
        elif len(dataset_stat_cfgs) != len(dataset_names):
            raise ValueError(
                f"dataset_stat_cfgs ({len(dataset_stat_cfgs)}) must be 1 or match dataset_names ({len(dataset_names)})"
            )

        for i, (dataset_name, dataset_cfg, dataset_stat_cfg) in enumerate(zip(dataset_names, dataset_cfgs, dataset_stat_cfgs)):
            data_json_path = f"{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json"
            if not os.path.isfile(data_json_path):
                if mode != "val":
                    raise FileNotFoundError(f"Missing required meta file for train: {data_json_path}")
                samples = []
            else:
                with open(data_json_path, "r") as f:
                    samples = json.load(f)

            if not samples:
                if mode != "val":
                    raise ValueError(f"[train] empty samples for cfg={dataset_cfg} name={dataset_name}")
                print(f"[val] skip (missing or empty) cfg={dataset_cfg} name={dataset_name} path={data_json_path}")
                continue

            # Cross-meta format includes per-sample dataset_root; fall back to classic root/name layout.
            dataset_path = []
            for sample in samples:
                sample_root = sample.get("dataset_root")
                sample_name = sample.get("dataset_name", dataset_name)
                if sample_root:
                    dataset_path.append(sample_root)
                else:
                    dataset_path.append(os.path.join(dataset_root_path, sample_name))
            unique_roots = sorted(set(dataset_path))
            root_preview = unique_roots[:3]

            print(f"ALL dataset, {len(samples)} samples in total for cfg={dataset_cfg}")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))
            self.kept_indices.append(i)

            # Mixture weights proportional to trajectory count.
            traj_n = len({str(s["episode_id"]) for s in samples})
            traj_counts.append(float(traj_n))
            print(f"  trajectories (unique episode_id) cfg={dataset_cfg}: {traj_n}")

            # Normalization priority:
            # 1) explicit stat cfg (supports cross cfg like droid_molmobot_cross)
            # 2) sample cfg stat
            # 2) dataset_name stat
            # 3) global fallback
            stat_candidates = [
                f"{dataset_meta_info_path}/{dataset_stat_cfg}/stat.json",
                f"{dataset_meta_info_path}/{dataset_cfg}/stat.json",
                f"{dataset_meta_info_path}/{dataset_name}/stat.json",
                f"{dataset_meta_info_path}/stat.json",
            ]
            stat_path = None
            for c in stat_candidates:
                if os.path.exists(c):
                    stat_path = c
                    break
            if stat_path is None:
                raise FileNotFoundError(
                    f"No stat.json found for cfg={dataset_cfg}, stat_cfg={dataset_stat_cfg}. Tried: {stat_candidates}"
                )
            print(f"[{mode}] loading stat.json for cfg={dataset_cfg} stat_cfg={dataset_stat_cfg} name={dataset_name}: {stat_path}")
            with open(stat_path, "r") as f:
                data_stat = json.load(f)
            state_p01 = np.array(data_stat["state_01"])[None, :]
            state_p99 = np.array(data_stat["state_99"])[None, :]
            self.norm_all.append((state_p01, state_p99))
            self.dataset_debug_rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_cfg": dataset_cfg,
                    "dataset_stat_cfg": dataset_stat_cfg,
                    "sample_json_path": data_json_path,
                    "stat_json_path": stat_path,
                    "sample_count": len(samples),
                    "traj_count": traj_n,
                    "root_preview": root_preview,
                    "root_count": len(unique_roots),
                }
            )

        if len(self.samples_len) == 0:
            raise ValueError(f"No non-empty datasets available for mode={mode}")

        self.prob = self._build_sampling_prob(traj_counts)
        self.max_id = max(self.samples_len)
        self._mix_index_stride = self.max_id + 1
        self.dataset_mix_batch = bool(getattr(args, "dataset_mix_batch", False)) and mode == "train"
        if self.dataset_mix_batch:
            if len(self.samples_all) != 2:
                raise ValueError(
                    "dataset_mix_batch requires exactly two datasets in dataset_names / dataset_cfgs; "
                    f"got {len(self.samples_all)}"
                )
        print("samples_len:", self.samples_len, "max_id:", self.max_id, "mix_prob:", self.prob)
        self._print_startup_summary(mode)

    def __len__(self):
        return self.max_id

    def _split_arg_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in re.split(r"[+,]", str(value)) if v.strip()]

    def _normalize_prob(self, values, desc):
        prob_arr = np.array(values, dtype=np.float64)
        if (prob_arr < 0).any() or prob_arr.sum() <= 0:
            raise ValueError(f"Invalid {desc}: {values}")
        prob_arr = prob_arr / prob_arr.sum()
        return prob_arr.tolist()

    def _build_sampling_prob(self, traj_counts):
        mode = getattr(self.args, "dataset_sampling_mode", "prorated_trajectories")

        if mode == "manual":
            raw_probs = self._split_arg_list(getattr(self.args, "dataset_sampling_probs", None))
            # raw_probs is positional over the configured datasets. When a dataset
            # is skipped (e.g. an empty val split), select the probs for the ones we
            # kept and renormalize, rather than requiring the caller to vary the arg
            # between train and val.
            if len(raw_probs) == len(self.samples_len):
                selected = raw_probs
            elif self.kept_indices and max(self.kept_indices) < len(raw_probs):
                selected = [raw_probs[i] for i in self.kept_indices]
            else:
                raise ValueError(
                    "For manual dataset sampling, --dataset_sampling_probs must match either the "
                    f"configured or the non-empty dataset count. Got probs={len(raw_probs)}, "
                    f"non-empty datasets={len(self.samples_len)}"
                )
            manual_probs = [float(p) for p in selected]
            prob = self._normalize_prob(manual_probs, "manual dataset_sampling_probs")
            print(f"[dataset mix] sampling mode=manual, normalized probs={prob}")
            return prob

        if mode == "prorated_samples":
            prob = self._normalize_prob(self.samples_len, "prorated_samples")
            print(f"[dataset mix] sampling mode=prorated_samples, based on sample counts={self.samples_len}")
            return prob

        if mode == "prorated_trajectories":
            prob = self._normalize_prob(traj_counts, "prorated_trajectories")
            print(f"[dataset mix] sampling mode=prorated_trajectories, based on trajectory counts={traj_counts}")
            return prob

        raise ValueError(
            "Unknown dataset_sampling_mode. Choose from: "
            "manual, prorated_samples, prorated_trajectories"
        )

    def _print_startup_summary(self, mode):
        sampling_mode = getattr(self.args, "dataset_sampling_mode", "prorated_trajectories")
        manual_probs = getattr(self.args, "dataset_sampling_probs", None)
        print("\n================ Dataset Startup Summary ================")
        print(f"mode={mode}")
        print(f"dataset_meta_info_path={self.args.dataset_meta_info_path}")
        print(f"dataset_root_path={self.args.dataset_root_path}")
        print(f"dataset_sampling_mode={sampling_mode}")
        if manual_probs is not None:
            print(f"dataset_sampling_probs(raw)={manual_probs}")
        print(f"dataset_mix_batch(effective)={self.dataset_mix_batch}")
        print(f"num_datasets={len(self.dataset_debug_rows)}")
        for i, row in enumerate(self.dataset_debug_rows):
            ratio = self.prob[i] if i < len(self.prob) else None
            print(f"[dataset {i}] name={row['dataset_name']}")
            print(f"  dataset_cfg={row['dataset_cfg']}")
            print(f"  dataset_stat_cfg={row['dataset_stat_cfg']}")
            print(f"  sample_json_path={row['sample_json_path']}")
            print(f"  stat_json_path={row['stat_json_path']}")
            print(f"  sample_count={row['sample_count']}  traj_count={row['traj_count']}  ratio={ratio}")
            print(f"  data_root_count={row['root_count']}")
            for root in row["root_preview"]:
                print(f"    data_root_preview={root}")
            if row["root_count"] > len(row["root_preview"]):
                print(f"    ... ({row['root_count'] - len(row['root_preview'])} more roots)")
        print("========================================================\n")

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path, "rb") as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids = [int(frame_id) if frame_id < max_frames else max_frames - 1 for frame_id in frame_ids]
        return video_tensor[frame_ids]

    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir):
        assert cam_id is not None
        assert pre_encode is True
        video_path = label["latent_videos"][cam_id]["latent_video_path"]
        video_path = os.path.join(video_dir, video_path)
        try:
            frames = self._load_latent_video(video_path, frame_ids)
        except Exception:
            video_path = video_path.replace("latent_videos", "latent_videos_svd")
            frames = self._load_latent_video(video_path, frame_ids)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        temp_cam_id = random.choice(self.cam_ids) if cam_id is None else cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode, video_dir=video_dir)
        return frames, temp_cam_id

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def __getitem__(self, index):
        if self.dataset_mix_batch:
            stride = self._mix_index_stride
            dataset_id = int(index) // stride
            local_idx = int(index) % stride
            if dataset_id < 0 or dataset_id >= len(self.samples_all):
                raise IndexError(f"decoded dataset_id={dataset_id} out of range for index={index}")
            samples = self.samples_all[dataset_id]
            local_idx = int(local_idx % len(samples))
            sample = samples[local_idx]
        else:
            dataset_id = int(np.random.choice(len(self.samples_all), p=self.prob))
            samples = self.samples_all[dataset_id]
            local_idx = int(index) % len(samples)
            sample = samples[local_idx]

        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        dataset_debug = self.dataset_debug_rows[dataset_id]
        dataset_dir = dataset_path[local_idx]

        frame_ids = sample["frame_ids"]
        ann_file = f'{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample["episode_id"]}.json'
        with open(ann_file, "r") as f:
            label = json.load(f)

        joint_len = len(label["observation.state.joint_position"]) - 1
        frame_len = np.floor(joint_len / 1) # 1 for 5Hz annotations; 3 for 15Hz annotations
        skip = random.randint(1, 2)
        skip_his = int(skip * 4)
        if random.random() < 0.15:
            skip_his = 0

        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id) * self.args.down_sample

        data = dict()
        data["text"] = label["texts"][0]
        data["dataset_name"] = dataset_debug["dataset_name"]
        data["dataset_cfg"] = dataset_debug["dataset_cfg"]
        data["sample_dataset_name"] = sample.get("dataset_name", dataset_debug["dataset_name"])
        data["dataset_id"] = int(dataset_id)
        data["episode_id"] = str(sample["episode_id"])

        cond_cam_id1 = 0
        cond_cam_id2 = 1
        cond_cam_id3 = 2
        latnt_cond1, _ = self._get_obs(label, rgb_id, cond_cam_id1, pre_encode=True, video_dir=dataset_dir)
        latnt_cond2, _ = self._get_obs(label, rgb_id, cond_cam_id2, pre_encode=True, video_dir=dataset_dir)
        latnt_cond3, _ = self._get_obs(label, rgb_id, cond_cam_id3, pre_encode=True, video_dir=dataset_dir)
        latent = torch.zeros((self.args.num_frames + self.args.num_history, 4, 72, 40), dtype=torch.float32)
        latent[:, :, 0:24] = latnt_cond1
        latent[:, :, 24:48] = latnt_cond2
        latent[:, :, 48:72] = latnt_cond3
        data["latent"] = latent.float()

        cartesian_pose = np.array(label["observation.state.cartesian_position"])[state_id]
        gripper_pose = np.array(label["observation.state.gripper_position"])[state_id][..., np.newaxis]
        action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)
        action = self.normalize_bound(action, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()
        return data
