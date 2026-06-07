import os
import torch
import numpy as np
import json
import mediapy
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset

import pandas as pd
from accelerate import Accelerator
from tqdm.auto import tqdm


class EncodeLatentDataset(Dataset): 
    def __init__(
        self,
        old_path,
        new_path,
        svd_path,
        device,
        size=(192, 320),
        rgb_skip=3,
        encode_batch_size=256,
        shard_id=0,
        num_shards=1,
    ):
        self.old_path = old_path
        self.new_path = new_path
        self.size = size
        self.skip = rgb_skip
        self.encode_batch_size = encode_batch_size
        self.shard_id = shard_id
        self.num_shards = max(1, num_shards)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)

        def load_json_file(file_path):
            data = []
            with open(file_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))  # 使用 json.loads() 解析单行
            return data

        self.data = load_json_file(f'{old_path}/meta/episodes.jsonl')

    def __len__(self):
        return len(self.data)

    def shard_len(self):
        return sum((int(item["episode_index"]) % self.num_shards) == self.shard_id for item in self.data)

    def _is_episode_complete(self, data_type, traj_id):
        video_dir = f"{self.new_path}/videos/{data_type}/{traj_id}"
        latent_dir = f"{self.new_path}/latent_videos/{data_type}/{traj_id}"
        ann_path = f"{self.new_path}/annotation/{data_type}/{traj_id}.json"
        expected_video = [f"{video_dir}/{i}.mp4" for i in range(3)]
        expected_latent = [f"{latent_dir}/{i}.pt" for i in range(3)]
        return all(os.path.exists(p) for p in (expected_video + expected_latent + [ann_path]))

    def _col_to_list(self, df, col_name):
        col = df[col_name].to_numpy()
        try:
            return np.stack(col).tolist()
        except Exception:
            # Fallback for irregular/object columns.
            return [np.asarray(x).tolist() for x in col]

    def _read_video_fast(self, video_path):
        return mediapy.read_video(video_path)

    def __getitem__(self, idx):
        traj_data = self.data[idx]
        instruction = traj_data['tasks'][0]
        traj_id = traj_data['episode_index']
        if (int(traj_id) % self.num_shards) != self.shard_id:
            return 0
        chunk_id = int(traj_id/1000)

        data_type = 'val' if traj_id%100 == 99 else 'train'
        if self._is_episode_complete(data_type, traj_id):
            return 0

        file_path = f'{self.old_path}/data/chunk-{chunk_id:03d}/episode_{traj_id:06d}.parquet'
        df = pd.read_parquet(file_path)
        obs_car = self._col_to_list(df, 'observation.state.cartesian_position')
        obs_joint = self._col_to_list(df, 'observation.state.joint_position')
        obs_gripper = self._col_to_list(df, 'observation.state.gripper_position')
        action_car = self._col_to_list(df, 'action.cartesian_position')
        action_joint = self._col_to_list(df, 'action.joint_position')
        action_gripper = self._col_to_list(df, 'action.gripper_position')
        action_joint_vel = self._col_to_list(df, 'action.joint_velocity')
        success = bool(df['is_episode_successful'].iloc[0])
        video_paths = [
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_1_left/episode_{traj_id:06d}.mp4',
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.exterior_2_left/episode_{traj_id:06d}.mp4',
                    f'{self.old_path}/videos/chunk-{chunk_id:03d}/observation.images.wrist_left/episode_{traj_id:06d}.mp4']
        missing_inputs = [p for p in video_paths if not os.path.exists(p)]
        if missing_inputs:
            print(f"Skipping trajectory {traj_id}, missing input videos: {missing_inputs}")
            return 0
        traj_info = {'success': success,
                     'observation.state.cartesian_position': obs_car,
                     'observation.state.joint_position': obs_joint,
                     'observation.state.gripper_position': obs_gripper,
                     'action.cartesian_position': action_car,
                     'action.joint_position': action_joint,
                     'action.gripper_position': action_gripper,
                     'action.joint_velocity': action_joint_vel,
                    }
        

        # if f"{save_root}/videos/{data_type}/{traj_id}" exist, skip this trajectory
        try:
            self.process_traj(
                video_paths,
                traj_info,
                instruction,
                self.new_path,
                traj_id=traj_id,
                data_type=data_type,
                size=self.size,
                rgb_skip=self.skip,
                device=self.vae.device,
                encode_batch_size=self.encode_batch_size,
            )
        except Exception as e:
            print(f"Error processing trajectory {traj_id}, skipping... ({type(e).__name__}: {e})")
            return 0
    
        return 0


    def process_traj(
        self,
        video_paths,
        traj_info,
        instruction,
        save_root,
        traj_id=0,
        data_type='val',
        size=(192, 320),
        rgb_skip=3,
        device='cuda',
        encode_batch_size=256,
    ):
        for video_id, video_path in enumerate(video_paths):
            # load and resize video and save
            video = self._read_video_fast(video_path)
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0*2-1
            frames = frames[::rgb_skip]  # Skip frames to save memory here!!!
            x = torch.nn.functional.interpolate(frames, size=size, mode='bilinear', align_corners=False)
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1)*255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            os.makedirs(f"{save_root}/videos/{data_type}/{traj_id}", exist_ok=True)
            mediapy.write_video(f"{save_root}/videos/{data_type}/{traj_id}/{video_id}.mp4", resize_video, fps=5)

            # save svd latent
            x = x.to(device)
            with torch.no_grad():
                latents = []
                for i in range(0, len(x), encode_batch_size):
                    batch = x[i : i + encode_batch_size]
                    latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                    # x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            os.makedirs(f"{save_root}/latent_videos/{data_type}/{traj_id}", exist_ok=True)
            torch.save(x, f"{save_root}/latent_videos/{data_type}/{traj_id}/{video_id}.pt")
        
        # record cartesain aligned with video frames
        cartesian_pose = np.array(traj_info['observation.state.cartesian_position'])
        cartesian_gripper = np.array(traj_info['observation.state.gripper_position'])[:,None]
        # print(cartesian_pose.shape, cartesian_gripper.shape)
        cartesian_states = np.concatenate((cartesian_pose, cartesian_gripper),axis=-1)[::rgb_skip].tolist()
        
        info = {
            "texts": [instruction],
            "episode_id": traj_id,
            "success": int(traj_info['success']),
            "video_length": frames.shape[0],
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info['observation.state.cartesian_position']),
            "videos": [
                {"video_path": f"videos/{data_type}/{traj_id}/0.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/1.mp4"},
                {"video_path": f"videos/{data_type}/{traj_id}/2.mp4"}
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{traj_id}/2.pt"}
            ],
            'states': cartesian_states,
            'observation.state.cartesian_position': traj_info['observation.state.cartesian_position'],
            'observation.state.joint_position': traj_info['observation.state.joint_position'],
            'observation.state.gripper_position': traj_info['observation.state.gripper_position'],
            'action.cartesian_position': traj_info['action.cartesian_position'],
            'action.joint_position': traj_info['action.joint_position'],
            'action.gripper_position': traj_info['action.gripper_position'],
            'action.joint_velocity': traj_info['action.joint_velocity'],
            }
        os.makedirs(f"{save_root}/annotation/{data_type}", exist_ok=True)
        with open(f"{save_root}/annotation/{data_type}/{traj_id}.json", "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_hf_path', type=str, default='/cephfs/shared/droid_hf/droid_1.0.1')
    parser.add_argument('--droid_output_path', type=str, default='dataset_example/droid_subset')
    parser.add_argument('--svd_path', type=str, default='/cephfs/shared/llm/stable-video-diffusion-img2vid')
    parser.add_argument(
        '--encode_batch_size',
        type=int,
        default=256,
        help="Frames per VAE encode chunk (192x320). With ~140GB VRAM try 512, 1024, or 2048 "
        "and watch for OOM on longest episodes.",
    )
    parser.add_argument('--shard_id', type=int, default=0, help='Current shard id in [0, num_shards).')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards.')
    # debug
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        old_path=args.droid_hf_path,
        new_path= args.droid_output_path,
        svd_path=args.svd_path,
        device=accelerator.device,
        size=(192, 320),
        rgb_skip=3, #  to downsample 15hz video to 5hz video
        encode_batch_size=args.encode_batch_size,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    tmp_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )
    tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
    total_steps = dataset.shard_len()
    if args.debug:
        total_steps = min(total_steps, 6)
    progress_bar = tqdm(
        range(total_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Episodes")
    for idx, _ in enumerate(tmp_data_loader):
        if idx == 5 and args.debug:
            break
        progress_bar.update(1)
        if idx % 100 == 0 and accelerator.is_main_process:
            progress_bar.set_postfix({"processed": idx})

# accelerate launch dataset_example/extract_latent.py --droid_hf_path /cephfs/shared/droid_hf/droid_1.0.1 --droid_output_path dataset_example/droid_subset --svd_path /cephfs/shared/llm/stable-video-diffusion-img2vid --debug

