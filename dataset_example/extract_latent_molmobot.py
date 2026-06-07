import os
import json
import time
import datetime
import mediapy
import numpy as np
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset


class EncodeLatentDataset(Dataset):
    def __init__(
        self,
        molmobot_path,
        svd_model_path,
        device,
        rgb_skip=3,
        shard_id=0,
        num_shards=1,
    ):
        self.path = molmobot_path
        self.rgb_skip = rgb_skip
        self.shard_id = shard_id
        self.num_shards = max(1, num_shards)
        self.vae  = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_model_path, subfolder="vae"
        ).to(device)
        self.vae.eval()

        # Collect all annotation files from train + val
        self.items = []   # list of (split, episode_id, ann_path)
        for split in ("train", "val"):
            ann_dir = os.path.join(molmobot_path, "annotation", split)
            if not os.path.isdir(ann_dir):
                continue
            for fname in sorted(os.listdir(ann_dir)):
                if fname.endswith(".json"):
                    ep_id = int(fname.replace(".json", ""))
                    self.items.append((split, ep_id, os.path.join(ann_dir, fname)))

        print(f"Found {len(self.items)} episodes in {molmobot_path}")

    def __len__(self):
        return len(self.items)

    def shard_len(self):
        return sum((int(ep_id) % self.num_shards) == self.shard_id for _, ep_id, _ in self.items)

    def __getitem__(self, idx):
        split, ep_id, ann_path = self.items[idx]
        if (int(ep_id) % self.num_shards) != self.shard_id:
            return 0

        # Skip if all 3 latent files already exist
        lat_dir = os.path.join(self.path, "latent_videos", split, str(ep_id))
        if all(os.path.exists(os.path.join(lat_dir, f"{c}.pt")) for c in range(3)):
            return 0

        with open(ann_path) as f:
            ann = json.load(f)

        try:
            self._process_episode(ann, split, ep_id)
        except Exception as e:
            print(f"  [error] episode {ep_id}: {e}")

        return 0

    def _process_episode(self, ann, split, ep_id):
        device = self.vae.device

        for cam_id in range(3):
            lat_path = os.path.join(
                self.path, "latent_videos", split, str(ep_id), f"{cam_id}.pt"
            )
            if os.path.exists(lat_path):
                continue

            vid_path = os.path.join(self.path, ann["videos"][cam_id]["video_path"])

            # Videos are already spatially preprocessed by phase 1.
            # Match DROID latent extraction by keeping every rgb_skip-th frame.
            video  = mediapy.read_video(vid_path)                          # (T, H, W, 3) uint8
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1  # [-1, 1]
            frames = frames[::self.rgb_skip]
            frames = frames.to(device)

            # Encode in batches
            latents = []
            with torch.no_grad():
                for i in range(0, len(frames), 64):
                    batch  = frames[i : i + 64]
                    latent = (self.vae.encode(batch)
                                  .latent_dist.sample()
                                  .mul_(self.vae.config.scaling_factor)
                                  .cpu())
                    latents.append(latent)
            latent_all = torch.cat(latents, dim=0)   # (T, 4, 24, 40)

            os.makedirs(os.path.dirname(lat_path), exist_ok=True)
            torch.save(latent_all, lat_path)
            print(f"  saved latent {split}/{ep_id}/{cam_id}.pt  shape={list(latent_all.shape)}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--molmobot_path", type=str,
                        default="dataset_example/molmobot_small")
    parser.add_argument("--svd_model_path", type=str,
                        default="/cephfs/shared/llm/stable-video-diffusion-img2vid")
    parser.add_argument("--rgb_skip", type=int, default=3,
                        help="Temporal stride before VAE encoding (3 = 15 Hz to 5 Hz)")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard id in [0, num_shards).")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--debug", action="store_true", help="Stop after 5 episodes")
    args = parser.parse_args()

    start_ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.time()
    print(f"[extract_latent_molmobot] start_time={start_ts}")

    try:
        accelerator = Accelerator()

        dataset = EncodeLatentDataset(
            molmobot_path=args.molmobot_path,
            svd_model_path=args.svd_model_path,
            device=accelerator.device,
            rgb_skip=args.rgb_skip,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
        loader = accelerator.prepare_data_loader(loader)

        total_steps = dataset.shard_len()
        if args.debug:
            total_steps = min(total_steps, 6)
        if accelerator.is_main_process:
            print(f"[extract_latent_molmobot] shard={args.shard_id}/{args.num_shards} episodes={total_steps}")

        for idx, _ in enumerate(loader):
            if args.debug and idx == 5:
                break
            if idx % 10 == 0 and accelerator.is_main_process:
                print(f"Encoded {idx}/{len(dataset)} episodes")

        if accelerator.is_main_process:
            print("Done.")
    finally:
        end_ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed_s = time.time() - t0
        print(f"[extract_latent_molmobot] end_time={end_ts} elapsed_s={elapsed_s:.1f}")

# accelerate launch dataset_example/extract_latent_molmobot.py \
#     --molmobot_path dataset_example/molmobot_small \
#     --svd_model_path /cephfs/shared/llm/stable-video-diffusion-img2vid
