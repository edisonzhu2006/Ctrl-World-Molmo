
# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools
import numpy as np


from accelerate import Accelerator
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import json
from decord import VideoReader, cpu
import swanlab
import mediapy
import sys
import re
from scipy.spatial.transform import Rotation as R


def _normalize_checkpoint_state_dict(state_dict):
    """Strip torch.compile / DDP prefixes so keys match eager CrtlWorld."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        while True:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
            if not changed:
                break
        cleaned[new_key] = value
    return cleaned


def _resolve_rollout_weights_path(ckpt_path):
    """
    Resolve a rollout weights file from either:
    - a flat .pt/.pth (original train_wm or patched legacy weights), or
    - a patched training checkpoint directory (checkpoint-XXXX/ with training_state.json
      and/or accelerate model.safetensors / pytorch_model.bin).
    """
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isfile(ckpt_path):
        return ckpt_path

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    meta_path = os.path.join(ckpt_path, "training_state.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            resume_state = json.load(f)
        weights_pt = resume_state.get("weights_pt")
        if weights_pt and os.path.isfile(weights_pt):
            return os.path.abspath(weights_pt)
        global_step = resume_state.get("global_step")
        output_dir = resume_state.get("output_dir")
        if output_dir is not None and global_step is not None:
            candidate = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    dir_name = os.path.basename(ckpt_path.rstrip("/"))
    step_match = re.search(r"checkpoint-(\d+)", dir_name)
    if step_match:
        step_tag = step_match.group(1)
        root = os.path.dirname(ckpt_path)
        for _ in range(4):
            if not root:
                break
            candidate = os.path.join(root, f"checkpoint-{step_tag}.pt")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            root = os.path.dirname(root)

    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = os.path.join(ckpt_path, name)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve rollout weights under {ckpt_path}. "
        "Expected a .pt file, training_state.json with weights_pt, "
        "sibling checkpoint-<step>.pt, or model.safetensors / pytorch_model.bin."
    )


def _read_weights_state_dict(weights_path):
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device="cpu")
    else:
        payload = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and all(isinstance(k, str) for k in payload):
            state_dict = payload
        else:
            raise ValueError(
                f"Unsupported checkpoint format in {weights_path}: "
                f"expected state dict or dict with 'state_dict' key."
            )
    return _normalize_checkpoint_state_dict(state_dict)


def _load_ctrl_world_for_rollout(model, ckpt_path, strict=True):
    weights_path = _resolve_rollout_weights_path(ckpt_path)
    print(f"Loading Ctrl-World weights from: {weights_path}")
    state_dict = _read_weights_state_dict(weights_path)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        missing_preview = incompatible.missing_keys[:8]
        unexpected_preview = incompatible.unexpected_keys[:8]
        raise RuntimeError(
            "Checkpoint keys do not match CrtlWorld. "
            f"missing ({len(incompatible.missing_keys)}): {missing_preview} ... "
            f"unexpected ({len(incompatible.unexpected_keys)}): {unexpected_preview} ..."
        )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Loaded with strict=False: "
            f"{len(incompatible.missing_keys)} missing, "
            f"{len(incompatible.unexpected_keys)} unexpected keys."
        )
    return weights_path


def _default_flat_export_path(ckpt_path, weights_path):
    """Default flat export path: inside the resume ckpt dir when possible."""
    ckpt_path = os.path.abspath(ckpt_path)
    if os.path.isdir(ckpt_path):
        step_match = re.search(r"checkpoint-(\d+)", os.path.basename(ckpt_path.rstrip("/")))
        name = f"checkpoint-{step_match.group(1)}.pt" if step_match else "weights_rollout.pt"
        return os.path.join(ckpt_path, name)
    root, _ext = os.path.splitext(weights_path)
    return f"{root}_rollout_compat.pt"


def _export_flat_ctrl_world_ckpt(model, export_path):
    """Save eager CrtlWorld weights for the original branch (torch.load + load_state_dict)."""
    export_path = os.path.abspath(export_path)
    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
    state_dict = _normalize_checkpoint_state_dict(model.state_dict())
    torch.save(state_dict, export_path)
    print(f"Exported rollout-compatible flat checkpoint to: {export_path}")
    return export_path


def _maybe_export_flat_ckpt(model, ckpt_path, weights_path, export_path=None):
    if export_path is None:
        export_path = _default_flat_export_path(ckpt_path, weights_path)
    if os.path.isfile(export_path):
        print(f"Flat export already exists, skipping: {export_path}")
        return export_path
    return _export_flat_ctrl_world_ckpt(model, export_path)


def _build_rollout_views(pred_chunks, gt_views):
    """
    Build per-view rollout videos from predicted chunks and align GT length.
    Returns:
        pred_views: np.ndarray [num_views, T, H, W, C]
        gt_views_aligned: np.ndarray [num_views, T, H, W, C]
    """
    per_step_views = []
    for step_idx, pred_step_views in enumerate(pred_chunks):
        # pred_step_views shape: [num_views, pred_step, H, W, C]
        if step_idx == len(pred_chunks) - 1:
            per_step_views.append(pred_step_views)
        else:
            per_step_views.append(pred_step_views[:, :-1])
    pred_views = np.concatenate(per_step_views, axis=1)
    rollout_len = pred_views.shape[1]
    gt_views_aligned = np.stack([v[:rollout_len] for v in gt_views], axis=0)
    return pred_views, gt_views_aligned


def _run_and_checkpoint_from_ckpt_path(ckpt_path):
    """Return (run_id, checkpoint_dir) for eval_framework layout."""
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _layout_dir = os.path.join(_repo_root, "eval_framework", "lib")
    if _layout_dir not in sys.path:
        sys.path.insert(0, _layout_dir)
    from layout import run_and_checkpoint_from_ckpt_path

    return run_and_checkpoint_from_ckpt_path(ckpt_path)


def _next_trial_dir(pred_prism_root):
    os.makedirs(pred_prism_root, exist_ok=True)
    trial_ids = []
    for name in os.listdir(pred_prism_root):
        m = re.fullmatch(r"trial_(\d+)", name)
        if m:
            trial_ids.append(int(m.group(1)))
    next_id = 0 if len(trial_ids) == 0 else max(trial_ids) + 1
    trial_dir = os.path.join(pred_prism_root, f"trial_{next_id}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir


def _save_eval_framework_views(sample_id, gt_views, pred_views, fps, run_id, checkpoint_dir, trial_dir):
    """Save GT/Pred per-view videos for eval framework."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_dir = os.path.join(repo_root, "eval_framework", "data", "gt", "prism", sample_id)
    pred_dir = os.path.join(
        repo_root,
        "eval_framework",
        "data",
        "predictions",
        run_id,
        checkpoint_dir,
        "prism",
        trial_dir,
        sample_id,
    )
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    num_views = min(gt_views.shape[0], pred_views.shape[0])
    for view_idx in range(num_views):
        gt_path = os.path.join(gt_dir, f"view{view_idx}.mp4")
        pred_path = os.path.join(pred_dir, f"view{view_idx}.mp4")
        mediapy.write_video(gt_path, gt_views[view_idx], fps=fps)
        mediapy.write_video(pred_path, pred_views[view_idx], fps=fps)
    print(f"Saved eval GT views to {gt_dir}")
    print(f"Saved eval predicted views to {pred_dir}")


class agent():
    def __init__(self,args):
          
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # # load pi policy
        # if 'pi05' in args.policy_type:
        #     config = config_pi.get_config("pi05_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid' 
        # elif 'pi0fast' in args.policy_type:
        #     config = config_pi.get_config("pi0fast_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        # elif 'pi0' in args.policy_type:
        #     config = config_pi.get_config("pi0_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        # else:
        #     raise ValueError(f"Unknown policy type: {args.policy_type}")
        # self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # load ctrl-world model (flat .pt or patched accelerate ckpt dir)
        self.model = CrtlWorld(args)
        weights_path = _load_ctrl_world_for_rollout(self.model, args.val_model_path, strict=True)
        if getattr(args, "export_flat_ckpt_only", False):
            export_path = getattr(args, "export_flat_ckpt_path", None) or None
            _maybe_export_flat_ckpt(
                self.model, args.val_model_path, weights_path, export_path=export_path
            )
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        # Stat.json is only needed for action normalization at inference time;
        # in --export_flat_ckpt_only mode we just convert weights and exit.
        if not getattr(args, "export_flat_ckpt_only", False):
            print(f"loading stat.json from: {args.data_stat_path}")
            with open(f"{args.data_stat_path}", 'r') as f:
                data_stat = json.load(f)
                self.state_p01 = np.array(data_stat['state_01'])[None,:]
                self.state_p99 = np.array(data_stat['state_99'])[None,:]
        

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

    def get_traj_info(self, id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            print(anno.keys())
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # get action and joint pos
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])
        car_action = car_action[frames_ids]
        if 'joints' in anno:
            joint_pos = np.array(anno['joints'])
        else:
            # Build [7 joint position + 1 gripper] on the same state timeline.
            joint_raw = np.array(anno['observation.state.joint_position'])
            gripper_raw = np.array(anno['observation.state.gripper_position'])
            state_length = len(anno['states'])
            raw_length = len(joint_raw)

            if raw_length == state_length * args.down_sample:
                idx = np.arange(0, raw_length, args.down_sample)[:state_length]
            else:
                # Fallback for non-integer ratio datasets.
                idx = np.linspace(0, raw_length - 1, state_length).round().astype(int)

            joint_pos = np.concatenate([joint_raw[idx], gripper_raw[idx, None]], axis=-1)
        joint_pos = joint_pos[frames_ids]

        # get videos (use only 3 regular views; ignore "*_filter")
        video_dict =[]
        video_latent = []
        video_ids = [i for i, v in enumerate(anno['videos']) if '_filter' not in v['video_path']][:3] or list(range(min(3, len(anno['videos']))))
        for id in video_ids:
            video_path = anno['videos'][id]['video_path']
            video_path = f"{val_dataset_dir}/{video_path}"
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video
            device = self.device
            true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video.permute(0,3,1,2).to(device) / 255.0*2-1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
    
            video_latent.append(x)

        
        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)


        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)           
            pipeline = self.model.pipeline
            
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)


        # decode ground truth video
        true_video = torch.stack(video_latent_true, dim=0) # (bsz, 8,32,32)
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

        # decode predicted video
        decoded_video = []
        bsz,frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,x.shape[0],args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # concatenate true videos and video
        videos_cat = np.concatenate([true_video,videos],axis=-3) # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat],axis=-2).astype(np.uint8) 

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)


def _resolve_replay_stat_path(dataset_meta_info_path, dataset_names, explicit_path=None):
    """Find stat.json for replay normalization (matches training lookup order roughly)."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    meta = dataset_meta_info_path
    if not meta:
        raise ValueError("dataset_meta_info_path is required to resolve stat.json")
    names = []
    if dataset_names:
        if "+" in dataset_names:
            names = [p.strip() for p in dataset_names.split("+") if p.strip()]
        else:
            names = [dataset_names.strip()]
    candidates = [os.path.join(meta, n, "stat.json") for n in names]
    candidates.extend(
        [
            os.path.join(meta, "droid_molmobot_cross", "stat.json"),
            os.path.join(meta, "droid_subset", "stat.json"),
            os.path.join(meta, "stat.json"),
        ]
    )
    tried = []
    for c in candidates:
        tried.append(c)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "No stat.json found for replay. Tried:\n  "
        + "\n  ".join(tried)
        + "\nPass --data_stat_path /path/to/stat.json"
    )


if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument(
        '--data_stat_path',
        type=str,
        default=None,
        help='Explicit path to stat.json for action normalization (overrides auto-resolve)',
    )
    parser.add_argument('--task_type', type=str, default='replay')
    parser.add_argument(
        '--export_flat_ckpt_only',
        action='store_true',
        help='Export flat .pt for eval and exit; default mode runs trajectory rollout and saves videos',
    )
    parser.add_argument(
        '--export_flat_ckpt_path',
        type=str,
        default=None,
        help='Explicit export path with --export_flat_ckpt_only (default: <ckpt_dir>/checkpoint-<step>.pt)',
    )
    args_new = parser.parse_args()
    cli_data_stat_path = args_new.data_stat_path

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args
    
    args = merge_args(args, args_new)

    # Format-conversion shortcut: load weights from a patched accelerate ckpt dir,
    # strip torch.compile/DDP prefixes, save flat .pt, exit. No dataset args needed.
    if args.export_flat_ckpt_only:
        print("export_flat_ckpt_only: loading model and exporting checkpoint, skipping rollout")
        agent(args)
        print("Done (no trajectory rollout or videos).")
        raise SystemExit(0)

    if args.task_type == "replay":
        args.val_dataset_dir = os.path.join(args.dataset_root_path, args.dataset_names)
        args.data_stat_path = _resolve_replay_stat_path(
            args.dataset_meta_info_path,
            args.dataset_names,
            explicit_path=cli_data_stat_path,
        )
        print(f"[replay] stat.json: {args.data_stat_path}")
        # Empty val_id -> run every trajectory in annotation/val (set val_id = [] in config).
        if len(args.val_id) == 0:
            ann_val = os.path.join(args.val_dataset_dir, "annotation", "val")
            ids = sorted(
                int(f[:-5])
                for f in os.listdir(ann_val)
                if f.endswith(".json") and f[:-5].isdigit()
            )
            args.val_id = [str(i) for i in ids]
            args.start_idx = [8] * len(args.val_id)
            args.instruction = [""] * len(args.val_id)
            print(f"Using all {len(args.val_id)} val trajectories from {ann_val}")

    # create rollout agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f'rollout with {args.task_type}')

    # Eval-framework prediction layout: pick the trial directory once per run so all
    # sample_<i> dirs in this invocation land under the same trial_<n>.
    run_id, checkpoint_dir = _run_and_checkpoint_from_ckpt_path(args.ckpt_path)
    pred_prism_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "eval_framework", "data", "predictions",
        run_id, checkpoint_dir, "prism",
    )
    print(f"Eval run_id={run_id} checkpoint={checkpoint_dir}")
    trial_path = _next_trial_dir(pred_prism_root)
    trial_name = os.path.basename(trial_path)
    print(f"Eval prediction trial directory: {trial_path}")

    for sample_idx, (val_id_i, text_i, start_idx_i) in enumerate(zip(args.val_id, args.instruction, args.start_idx)):
        # read ground truth trajectory informations
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
        text_i = instruction
        print("text_i:",instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # create buffers and push first frames to history buffer
        predicted_latents = None
        video_to_save = []
        pred_views_to_save = []
        info_to_save = []
        his_cond = []
        his_joint = []
        his_eef = []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        for i in range(Agent.args.num_history*4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)

        # interact loop
        for i in range(interact_num):
            # ground truth video
            start_id = int(i*(pred_step-1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]
            
            # prepare input for policy
            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            if i==0:
                video_first = [v[0] for v in video_dict]
            else:
                video_first = [v[-1] for v in video_dict_pred]
            assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
            assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"
            
            # forward policy
            print("################ policy forward ####################")
            # in the trajectory replay model, we use action recorded in trajetcory
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
            print("cartesian space action", cartesian_pose[0]) # output xyz and gripper for debug
            print("cartesian space action", cartesian_pose[-1]) # output xyz and gripper for debug
            
            print("################ world model forward ################")
            print(f'traj_id:{val_id_i}, interact step: {i}/{interact_num}')
            # retrive history cond and action cond
            history_idx = [0,0,-8,-6,-4,-2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (4, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history+num_frames), 7), f"Expected action_cond shape ({int(num_history+num_frames)}, 7), got {action_cond.shape}"
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 72, 40), got {his_cond_input.shape}"
            # forward world model
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(action_cond, video_latent_true, current_latent, his_cond=his_cond_input,text=text_i if Agent.args.text_cond else None)

            print("################ record information ################")
            # push current step to history buffer
            his_eef.append(cartesian_pose[pred_step-1:pred_step]) #(1,7)
            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            if i == interact_num - 1:
                video_to_save.append(videos_cat)  # save all frames for the last interaction step
                pred_views_to_save.append(video_dict_pred)
            else:
                video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here
                pred_views_to_save.append(video_dict_pred)
                
        
        # save rollout video and info with parameters
        video = np.concatenate(video_to_save, axis=0)
        task_name = args.task_name
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace('\'', '').replace('\"', '')[:30]
        videos_dir = args.val_model_path.split('/')[:-1]
        videos_dir = '/'.join(videos_dir)
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_dir = datetime.date.today().isoformat()  # e.g. 2026-04-10
        filename_video = (
            f"{args.save_dir}/{task_name}/video/{date_dir}/"
            f"time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"
        )
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        mediapy.write_video(filename_video, video, fps=4)
        print(f"Saving video to {filename_video}")

        # Per-view GT and prediction videos for the eval framework metric pipelines.
        pred_views_rollout, gt_views_rollout = _build_rollout_views(pred_views_to_save, video_dict)
        _save_eval_framework_views(
            sample_id=f"sample_{sample_idx}",
            gt_views=gt_views_rollout,
            pred_views=pred_views_rollout,
            fps=args.fps,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            trial_dir=trial_name,
        )
        print("##########################################################################")
        
        print("USING config from:", __file__)
        print("task_type:", args.task_type)
        print("val_dataset_dir:", args.val_dataset_dir)
        print("val_id:", args.val_id)
        print("start_idx:", args.start_idx)


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py
        
        
