# from diffusers import StableVideoDiffusionPipeline
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, TorchDynamoPlugin
import datetime
import os
import time
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
import swanlab
import mediapy
from models.ctrl_world import CrtlWorld
from config import wm_args
import math


def _make_lr_lambda(warmup_steps, total_steps, min_ratio, num_processes):
    """Linear warmup 0→peak over `warmup_steps` optimizer updates, then cosine
    decay peak→peak*min_ratio over the remaining optimizer updates. Returns a
    multiplier on the base LR.

    Horizon is scaled by `num_processes` because AcceleratedScheduler (returned
    from accelerator.prepare()) advances the underlying torch scheduler
    `num_processes` times per external call under default settings
    (split_batches=False, step_scheduler_with_optimizer=True). Without this
    scaling, an 8-rank run hits the cosine floor at step max_train_steps/8.

    This scaling is WRONG if any of these change:
      - split_batches=True on the Accelerator
      - step_scheduler_with_optimizer=False
      - scheduler not passed through accelerator.prepare(...)
    In those cases the underlying scheduler steps once per external call and
    this scaling would stretch the schedule by num_processes instead."""
    schedule_scale = max(1, num_processes)
    warmup_steps_internal = warmup_steps * schedule_scale
    total_steps_internal = total_steps * schedule_scale

    def lr_lambda(step):
        if step < warmup_steps_internal:
            return step / max(1, warmup_steps_internal)
        progress = (step - warmup_steps_internal) / max(
            1, total_steps_internal - warmup_steps_internal,
        )
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * max(0.0, cosine)
    return lr_lambda


RESUME_OVERRIDABLE = frozenset({
    "max_train_steps",
    "checkpointing_steps",
    "validation_steps",
    "ckpt_path",
    "resume",
    "video_num",
    "num_inference_steps",
})


def _wm_args_field_names():
    """wm_args stores many settings as class attrs; they are not always in __dict__."""
    names = set()
    for k, v in wm_args.__dict__.items():
        if k.startswith("_") or callable(v):
            continue
        names.add(k)
    names.update(getattr(wm_args, "__annotations__", {}).keys())
    return names


def args_to_jsonable(args) -> dict:
    out = {}
    skip = {"dtype"}
    for k in set(args.__dict__) | _wm_args_field_names():
        if k in skip:
            continue
        if not hasattr(args, k):
            continue
        v = getattr(args, k)
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
    return out


def resolve_resume_ckpt_dir(ckpt_path: str) -> str:
    ckpt_dir = os.path.abspath(ckpt_path)
    if os.path.isfile(ckpt_dir):
        raise ValueError(
            f"--resume expects a checkpoint directory (with training_state.json), "
            f"not a weight file: {ckpt_dir}"
        )
    meta_path = os.path.join(ckpt_dir, "training_state.json")
    if not os.path.isfile(meta_path):
        raise ValueError(
            f"Missing training_state.json in {ckpt_dir}. "
            "Use --resume only with checkpoints saved by the patched train_wm.py."
        )
    return ckpt_dir


def load_training_state(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "training_state.json")) as f:
        return json.load(f)


def restore_args_from_checkpoint(ckpt_dir: str, cli_overrides: dict):
    resume_state = load_training_state(ckpt_dir)
    base = wm_args()
    saved_args = resume_state["args"]
    for k, v in saved_args.items():
        if k not in RESUME_OVERRIDABLE:
            base.__dict__[k] = v
    for k, v in cli_overrides.items():
        if v is not None and k in RESUME_OVERRIDABLE:
            base.__dict__[k] = v
    resumed_step = int(resume_state["global_step"])
    if base.max_train_steps <= resumed_step:
        raise ValueError(
            f"max_train_steps ({base.max_train_steps}) must be greater than "
            f"resumed global_step ({resumed_step})"
        )
    return base, resume_state


def validate_resume_config(args, resume_state, num_processes: int):
    ckpt_args = resume_state["args"]
    checks = [
        ("dataset_names", ckpt_args.get("dataset_names"), getattr(args, "dataset_names", None)),
        ("train_batch_size", ckpt_args.get("train_batch_size"), args.train_batch_size),
        ("learning_rate", ckpt_args.get("learning_rate"), args.learning_rate),
        ("gradient_accumulation_steps", ckpt_args.get("gradient_accumulation_steps"), args.gradient_accumulation_steps),
    ]
    for name, saved, current in checks:
        if saved is None:
            continue  # older checkpoints may omit class-level wm_args defaults
        if saved != current:
            raise ValueError(f"Resume mismatch on {name}: checkpoint={saved!r} current={current!r}")
    saved_world = resume_state.get("num_processes")
    if saved_world is not None and int(saved_world) != int(num_processes):
        raise ValueError(
            f"Resume world size mismatch: checkpoint={saved_world} current={num_processes}"
        )


def dataloader_state(train_dataloader, epoch: int, step_in_epoch: int) -> dict:
    batch_sampler = getattr(train_dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "state_dict"):
        state = batch_sampler.state_dict()
        state["epoch"] = epoch
        state["step_in_epoch"] = step_in_epoch
        return state
    return {"kind": "distributed", "epoch": epoch, "step_in_epoch": step_in_epoch}


def restore_dataloader_state(train_dataloader, dataloader_state: dict, start_epoch: int, skip_batches: int):
    batch_sampler = getattr(train_dataloader, "batch_sampler", None)
    if dataloader_state.get("kind") == "mixed_domain_batch_sampler":
        if batch_sampler is None or not hasattr(batch_sampler, "load_state_dict"):
            raise ValueError("Checkpoint expects MixedDomainBatchSampler but dataloader has none")
        batch_sampler._epoch_idx = start_epoch
        batch_sampler._resume_batch_idx = skip_batches
        batch_sampler.seed = int(dataloader_state["seed"])
        batch_sampler._consume_resume_offset = True
        return
    sampler = getattr(train_dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(start_epoch)


def save_training_state_json(accelerator, args, payload: dict, ckpt_dir: str):
    if not accelerator.is_main_process:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = dict(payload)
    payload["args"] = args_to_jsonable(args)
    payload["output_dir"] = args.output_dir
    payload["samples_dir_name"] = args.samples_dir_name
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(payload, f, indent=2)


def save_checkpoint(accelerator, model, args, ckpt_root: str, cursor: dict, run_meta: dict, logger):
    global_step = cursor["global_step"]
    ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
    accelerator.save_state(ckpt_dir)
    payload = {
        **cursor,
        "wandb_run_id": run_meta.get("wandb_run_id"),
        "wandb_run_name": run_meta.get("wandb_run_name"),
        "num_processes": accelerator.num_processes,
    }
    save_training_state_json(accelerator, args, payload, ckpt_dir)
    if accelerator.is_main_process:
        weights_path = os.path.join(ckpt_dir, f"weights-only-step-{global_step}.pt")
        torch.save(accelerator.unwrap_model(model).state_dict(), weights_path)
        logger.info(f"Saved full training checkpoint to {ckpt_dir}")


def _make_samples_dir_name() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"samples_{ts}-{job_id}"
    return f"samples_{ts}"


def _gather_timing(accelerator, metrics):
    """Cross-rank reduce of tiny scalar timing metrics. Returns mean and max
    per metric — max is the straggler-rank value, which is what actually
    dictates step wall-clock under DDP (everyone waits for the slowest)."""
    keys = list(metrics.keys())
    vals = torch.tensor([metrics[k] for k in keys], device=accelerator.device, dtype=torch.float32)
    gathered = accelerator.gather(vals.unsqueeze(0))  # [world, num_metrics]
    means = gathered.mean(dim=0).tolist()
    maxes = gathered.max(dim=0).values.tolist()
    out = {}
    for i, k in enumerate(keys):
        out[f"timing/{k}_mean"] = means[i]
        out[f"timing/{k}_max"] = maxes[i]
    return out


def main(args, resume_state=None, resume_ckpt_dir=None):
    logger = get_logger(__name__, log_level="INFO")
    is_resume = resume_state is not None
    run_meta = {
        "wandb_run_id": resume_state.get("wandb_run_id") if is_resume else None,
        "wandb_run_name": resume_state.get("wandb_run_name") if is_resume else None,
    }

    if is_resume:
        args.output_dir = resume_state["output_dir"]
        args.samples_dir_name = resume_state["samples_dir_name"]
    else:
        args.samples_dir_name = _make_samples_dir_name()
    # swanlab.sync_wandb()
    swanlab.sync_wandb(mode=os.environ.get("SWANLAB_MODE", "local"))
    # DDP kwargs: gradient_as_bucket_view ties param.grad memory to allreduce
    # bucket slices (avoids the slow stride-mismatch copy path with compile).
    # static_graph lets DDP specialize after the first iteration; only valid
    # because every step uses the same forward graph and all trainable params
    # receive gradient (CFG dropout is a tensor mask, not a Python branch).
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=True,
        gradient_as_bucket_view=True,
    )
    # Compile the prepared (DDP-wrapped) model. This lets Dynamo's DDPOptimizer
    # break the graph at DDP bucket boundaries so allreduce overlaps backward.
    dynamo_plugin = TorchDynamoPlugin(
        backend="inductor",
        mode="default",  # not 'reduce-overhead' — CUDA graphs conflict with grad-ckpt
        fullgraph=False,
        dynamic=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs],
        dynamo_plugin=dynamo_plugin,
    )

    # model and optimizer
    model = CrtlWorld(args)
    if (not is_resume) and args.ckpt_path is not None:
        print(f"Loading init weights from {args.ckpt_path}!")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        # Legacy .pt files saved before unwrap_model() carry torch.compile
        # ("_orig_mod.") and/or DDP ("module.") wrapper prefixes on every key.
        # Strip them so strict=True still works against an uncompiled CrtlWorld,
        # while leaving clean state_dicts (no prefix) untouched.
        def _strip_known_prefixes(sd, prefixes=("_orig_mod.", "module.")):
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if sd and all(isinstance(k, str) and k.startswith(p) for k in sd):
                        sd = {k[len(p):]: v for k, v in sd.items()}
                        changed = True
            return sd
        state_dict = _strip_known_prefixes(state_dict)
        model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    # LambdaLR horizon is scaled by num_processes inside _make_lr_lambda because
    # this scheduler IS passed through accelerator.prepare() below.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _make_lr_lambda(args.lr_warmup_steps, args.max_train_steps, args.lr_min_ratio, accelerator.num_processes),
    )

    # logs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if is_resume:
            wandb_kwargs = {
                "name": run_meta["wandb_run_name"],
                "id": run_meta["wandb_run_id"],
                "resume": "allow",
            }
        else:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            run_name = f"train_{now}_{args.tag}"
            wandb_kwargs = {"name": run_name}
            run_meta["wandb_run_name"] = run_name
        accelerator.init_trackers(
            args.wandb_project_name,
            config={},
            init_kwargs={"wandb": wandb_kwargs},
        )
        if not is_resume and wandb.run is not None:
            run_meta["wandb_run_id"] = wandb.run.id
            run_meta["wandb_run_name"] = wandb.run.name
        # count parameters num in each part
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters in the unet: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"Number of parameters in the vae: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Number of parameters in the image_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Number of parameters in the text_encoder: {num_params/1000000:.2f}M")
        num_params = sum(p.numel() for p in model.action_encoder.parameters())
        print(f"Number of parameters in the action_encoder: {num_params/1000000:.2f}M")

    # train and val datasets
    from dataset.dataset_droid_exp33_cross import Dataset_mix
    train_dataset = Dataset_mix(args,mode='train')
    val_dataset = Dataset_mix(args,mode='val')
    if getattr(args, "dataset_mix_batch", False):
        from dataset.mixed_domain_batch_sampler import MixedDomainBatchSampler, parse_dataset_mix_counts

        counts = parse_dataset_mix_counts(
            getattr(args, "dataset_mix_counts", "2,2"), args.train_batch_size
        )
        train_batch_sampler = MixedDomainBatchSampler(
            train_dataset,
            args.train_batch_size,
            counts,
            seed=int(getattr(args, "dataset_mix_seed", 0) or 0),
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
        )
        logger.info(
            f"Using MixedDomainBatchSampler: counts={counts} (dataset_names order), "
            f"seed={getattr(args, 'dataset_mix_seed', 0)}"
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )

    # Prepare everything with our accelerator. lr_scheduler MUST be passed
    # through prepare for the num_processes scaling in _make_lr_lambda to be
    # correct — see that function's docstring.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    start_epoch = 0
    skip_batches = 0
    forward_step = 0
    train_loss = 0.0

    if is_resume:
        validate_resume_config(args, resume_state, accelerator.num_processes)
        accelerator.load_state(resume_ckpt_dir)
        global_step = int(resume_state["global_step"])
        start_epoch = int(resume_state["epoch"])
        skip_batches = int(resume_state["step_in_epoch"])
        forward_step = int(resume_state.get("forward_step", global_step))
        train_loss = float(resume_state.get("train_loss_accum", 0.0))
        restore_dataloader_state(
            train_dataloader,
            resume_state.get("dataloader", {}),
            start_epoch,
            skip_batches,
        )
        logger.info(
            f"Resumed from {resume_ckpt_dir}: global_step={global_step}, "
            f"epoch={start_epoch}, skip_batches={skip_batches}, "
            f"max_train_steps={args.max_train_steps}"
        )

    # Training compile is handled by Accelerator's TorchDynamoPlugin (above).
    # The inference path (pipeline.__call__ during validation) holds its own
    # ref to the UNet through pipeline.unet, which Dynamo's outer-model compile
    # does not cover. Compile it separately so val sampling isn't eager — the
    # compiled wrapper shares the same parameter tensors as training (weight
    # updates from training propagate; no duplicated weights, just two graphs).
    inner = accelerator.unwrap_model(model)
    inner.pipeline.unet = torch.compile(
        inner.unet,
        backend="inductor",
        mode="default",
        fullgraph=False,
        dynamic=False,
    )

    ############################ training ##############################
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_train_epochs = math.ceil(args.max_train_steps * args.gradient_accumulation_steps*total_batch_size / len(train_dataloader))
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Train dataloader steps per epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    if is_resume:
        logger.info(f"  Resuming global_step = {global_step}")
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Timing instrumentation. CUDA events are placed on the stream cheaply (~us)
    # every step, but we only pay torch.cuda.synchronize() + cross-rank gather on
    # sampled steps (timing_log_every). Dataloader wait is pure CPU wall time so
    # needs no sync — it's the time from "last batch's compute enqueued" to
    # "next batch returned by iterator" (which under accelerate.prepare includes
    # the H2D copy onto device).
    # The full instrumentation overhead (sync + gather + log + postfix) is
    # measured around each sampled block and reported on the NEXT sample,
    # so Sam can see the cost in wandb. One-sample reporting lag.
    timing_log_every = 100 #50
    _step_wall_t0 = time.perf_counter()
    _data_accum = 0.0
    _data_t0 = time.perf_counter()
    _gpu_step_start = torch.cuda.Event(enable_timing=True)
    _gpu_step_end = torch.cuda.Event(enable_timing=True)
    _gpu_step_start.record()  # mark start of opt step 0
    _prev_instr_overhead_s = 0.0  # carried from previous sample, 0 on first

    for epoch in range(start_epoch, num_train_epochs):
        train_sampler = getattr(train_dataloader, "sampler", None)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        for step_in_epoch, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step_in_epoch < skip_batches:
                _data_t0 = time.perf_counter()
                continue

            _data_accum += time.perf_counter() - _data_t0

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen, _ = model(batch)
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()/ args.gradient_accumulation_steps
                accelerator.backward(loss_gen)
                params_to_clip = model.parameters()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()  # AcceleratedScheduler internally guards on sync_gradients
                optimizer.zero_grad()
                forward_step += 1

            if accelerator.sync_gradients:
                _gpu_step_end.record()
                progress_bar.update(1)
                global_step += 1

                # Sample-log timing every timing_log_every steps. The synchronize
                # here is the only blocking op; on non-sampled steps we just record
                # the next event and continue. We bracket the entire sampled block
                # with _instr_t0 / _instr_overhead_s so the full cost of the
                # instrumentation itself (sync + gather + log + postfix) lands in
                # wandb on the NEXT sample.
                if global_step % timing_log_every == 0:
                    _instr_t0 = time.perf_counter()
                    _sync_t0 = time.perf_counter()
                    torch.cuda.synchronize()
                    _gpu_sync_wait_s = time.perf_counter() - _sync_t0
                    _step_wall_s = time.perf_counter() - _step_wall_t0
                    _gpu_active_s = _gpu_step_start.elapsed_time(_gpu_step_end) / 1000.0
                    metrics = {
                        "step_wall_s": _step_wall_s,
                        "data_wait_s": _data_accum,
                        "gpu_active_s": _gpu_active_s,
                        "gpu_sync_wait_s": _gpu_sync_wait_s,
                        "data_wait_frac": _data_accum / max(_step_wall_s, 1e-9),
                        "gpu_active_frac": _gpu_active_s / max(_step_wall_s, 1e-9),
                        # Previous sample's full overhead (sync + gather + log + postfix).
                        # Reports as 0 on the first sample; captures the real number
                        # from the second sample on.
                        "instr_overhead_s": _prev_instr_overhead_s,
                        "lr": optimizer.param_groups[0]['lr'],
                    }
                    reduced = _gather_timing(accelerator, metrics)
                    accelerator.log(reduced, step=global_step)
                    progress_bar.set_postfix({
                        "data_frac": f"{metrics['data_wait_frac']:.2f}",
                        "gpu_active": f"{metrics['gpu_active_frac']:.2f}",
                        "instr_s": f"{_prev_instr_overhead_s*1000:.0f}ms",
                        "loss": train_loss / 100,
                        "lr": f"{metrics['lr']:.2e}",
                    })
                    _prev_instr_overhead_s = time.perf_counter() - _instr_t0

                # Reset timers / events for the next opt step (do this every step,
                # not just sampled ones — start event must be recorded for elapsed
                # to read correctly next sample).
                _step_wall_t0 = time.perf_counter()
                _data_accum = 0.0
                _gpu_step_start = torch.cuda.Event(enable_timing=True)
                _gpu_step_end = torch.cuda.Event(enable_timing=True)
                _gpu_step_start.record()

                # log loss every 100 steps
                if global_step %100 == 0:
                    accelerator.log({"train_loss": train_loss/100}, step=global_step)
                    train_loss = 0.0
                # save ckpt every checkpointing_steps
                if global_step % args.checkpointing_steps == 0:
                    ckpt_root = os.path.join(args.output_dir, args.samples_dir_name, "ckpts")
                    save_checkpoint(
                        accelerator,
                        model,
                        args,
                        ckpt_root,
                        cursor={
                            "global_step": global_step,
                            "epoch": epoch,
                            "step_in_epoch": step_in_epoch + 1,
                            "forward_step": forward_step,
                            "train_loss_accum": train_loss,
                            "dataloader": dataloader_state(
                                train_dataloader, epoch, step_in_epoch + 1
                            ),
                        },
                        run_meta=run_meta,
                        logger=logger,
                    )
                # generate video every validation_steps
                if global_step % args.validation_steps == 5 and accelerator.is_main_process:
                    model.eval()
                    torch.cuda.synchronize()
                    val_t0 = time.perf_counter()
                    with accelerator.autocast():
                        for id in range(args.video_num):
                            validate_video_generation(model, val_dataset, args,global_step, args.output_dir, id, accelerator)
                    torch.cuda.synchronize()
                    val_dt = time.perf_counter() - val_t0
                    approx_unet_calls = args.video_num * args.num_inference_steps
                    print(f"[val] step={global_step} dt={val_dt:.1f}s videos={args.video_num} "
                          f"steps={args.num_inference_steps} approx_unet_calls={approx_unet_calls}")
                    model.train()
                if global_step >= args.max_train_steps:
                    return

            # Mark start of next batch wait — fires every micro-batch, sync or not.
            _data_t0 = time.perf_counter()



def main_val(args):
    accelerator = Accelerator()
    model = CrtlWorld(args)
    # load form val_model_path
    print("load from val_model_path",args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(model, None, args, 0, 'output', 0, accelerator, load_from_dataset=False)
    
            

def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator, load_from_dataset=True):
    device = accelerator.device
    # unwrap_model strips DDP / OptimizedModule / any other accelerate wrappers
    # regardless of wrap order, so we don't have to special-case num_processes.
    inner = accelerator.unwrap_model(model)
    pipeline = inner.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # sample from val dataset
    batch_id = list(range(0,len(val_dataset),int(len(val_dataset)/videos_row/videos_col)))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)],dim=0).to(device, non_blocking=True)
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    print("image",current_latent.shape, 'action', actions.shape)
    assert current_latent.shape[1:] == (4, 72, 40)
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim)

    # start generate
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = inner.action_encoder(actions, text, inner.tokenizer, inner.text_encoder, args.frame_level_cond)  # (8, 1, 1024)
        print("action_latent",action_latent.shape)

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3*args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
    
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
    video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1) # (B, 8, 4, 32,32)
    video_gt = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
    
    # decode latent
    if video_gt.shape[2] != 3:  
        decoded_video = []
        bsz,frame_num = video_gt.shape[:2]
        video_gt = video_gt.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,video_gt.shape[0],args.decode_chunk_size):
            chunk = video_gt[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        video_gt = torch.cat(decoded_video,dim=0)
        video_gt = video_gt.reshape(bsz,frame_num,*video_gt.shape[1:])
        
        decoded_video = []
        bsz,frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,pred_latents.shape[0],args.decode_chunk_size):
            chunk = pred_latents[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])

    # [0, 1] RGB for residual (same timeline: history from GT, future from pred decode).
    gt01 = (video_gt / 2.0 + 0.5).clamp(0, 1)
    pred01 = (videos / 2.0 + 0.5).clamp(0, 1)
    stitched_pred01 = torch.cat([gt01[:, : args.num_history], pred01], dim=1)
    delta01 = stitched_pred01 - gt01
    # map signed error to [0, 1]: no error -> 0.5 gray; tune gain if errors are too faint/loud
    diff_gain = float(getattr(args, "validation_diff_gain", 4.0))
    diff01 = (delta01 * diff_gain + 0.5).clamp(0, 1)

    video_gt_u8 = (
        (gt01 * 255)
        .to(pipeline.unet.dtype)
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 1, 3, 4, 2)
        .astype(np.uint8)
    )
    pred_u8 = (
        (stitched_pred01 * 255)
        .to(pipeline.unet.dtype)
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 1, 3, 4, 2)
        .astype(np.uint8)
    )
    diff_u8 = (
        (diff01 * 255)
        .to(pipeline.unet.dtype)
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 1, 3, 4, 2)
        .astype(np.uint8)
    )
    # Three rows: ground truth | prediction (GT history + pred future) | signed residual
    videos = np.concatenate([video_gt_u8, pred_u8, diff_u8], axis=-3)
    videos = np.concatenate([video for video in videos], axis=-2).astype(np.uint8)
    
    run_id = getattr(args, "samples_dir_name", None) or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    video_out_dir = os.path.join(videos_dir, run_id, "videos")
    os.makedirs(video_out_dir, exist_ok=True)
    filename = os.path.join(video_out_dir, f"train_steps_{train_steps}_{id}.mp4")
    mediapy.write_video(filename, videos, fps=2)
    return 



if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser, BooleanOptionalAction
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Init weights (.pt) or resume checkpoint dir (checkpoint-XXXX/)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume full training state from --ckpt_path directory')
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    # dataset_names
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--dataset_cfgs', type=str, default=None)

    # Lei: data sampling parameters
    parser.add_argument(
        '--dataset_stat_cfgs',
        type=str,
        default=None,
        help='Stat cfg name(s). Can be one value shared by all datasets, or one per dataset.',
    )
    parser.add_argument(
        '--dataset_sampling_mode',
        type=str,
        default=None,
        choices=['manual', 'prorated_samples', 'prorated_trajectories'],
    )
    parser.add_argument(
        '--dataset_sampling_probs',
        type=str,
        default=None,
        help='Manual dataset sampling weights, e.g. "0.5,0.5" or "0.3+0.7". Used when dataset_sampling_mode=manual.',
    )
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument(
        '--dataset_mix_batch',
        action=BooleanOptionalAction,
        default=None,
        help='Train with MixedDomainBatchSampler so each minibatch includes both datasets (exactly two).',
    )
    parser.add_argument(
        '--dataset_mix_counts',
        type=str,
        default=None,
        help='Two integers summing to train_batch_size, e.g. "2,2" (dataset_names order).',
    )
    parser.add_argument('--dataset_mix_seed', type=int, default=None, help='RNG seed for mixed batches.')

    # Sam: training parameters
    parser.add_argument('--max_train_steps', type=int, default=None)
    parser.add_argument('--validation_steps', type=int, default=None)
    parser.add_argument('--checkpointing_steps', type=int, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--lr_warmup_steps', type=int, default=None)
    parser.add_argument('--lr_min_ratio', type=float, default=None)
    parser.add_argument('--video_num', type=int, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    args_new = parser.parse_args()
    cli = {k: v for k, v in args_new.__dict__.items() if v is not None}

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        # Keep dataset cfg mapping aligned by default when dataset_names is overridden.
        if new_args.dataset_names is not None and new_args.dataset_cfgs is None:
            args.__dict__['dataset_cfgs'] = args.__dict__['dataset_names']
        return args

    if args_new.resume:
        if not args_new.ckpt_path:
            raise SystemExit("--resume requires --ckpt_path pointing at a checkpoint-XXXX/ directory")
        resume_ckpt_dir = resolve_resume_ckpt_dir(args_new.ckpt_path)
        args, resume_state = restore_args_from_checkpoint(resume_ckpt_dir, cli)
        main(args, resume_state=resume_state, resume_ckpt_dir=resume_ckpt_dir)
    else:
        args = merge_args(wm_args(), args_new)
        main(args)

    # CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=offline accelerate launch --main_process_port 29501 train_wm.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info
    # CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29506 unit_test2.py

    # args = Args()
    # from video_dataset.dataset_droid_exp33 import Dataset_mix
    # dataset = Dataset_mix(args,mode='val')
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)
    # model = CrtlWorld(args).to('cuda')
    # # print model parameter num
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    # total_elements = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    # print(f"Total number of learnable parameters: {total_elements}")
    # model.train()
    

    # for batch in dataloader:
    #     print(batch['latent'].shape)
    #     print(batch['text'])
    #     print(batch['action'].shape)

    #     loss,_ = model(batch)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(loss.item())





    # device = 'cuda'
    # video_encoder = VideoEncoder(hidden_size=1024).to(device)
    # # count the parameters of the model
    # num_params = sum(p.numel() for p in video_encoder.parameters())
    # print(f"Number of parameters in the model: {num_params/1000000:.2f}M")
    # vae_latent = torch.randn(8, 1, 4, 32, 32).to(device)
    # clip_latent = torch.randn(8, 20, 512).to(device)
    # current_img = video_encoder(vae_latent, clip_latent)
    # print(current_img.shape)  # (8, 1, 4, 32, 32)


    # pos_emb = get_2d_sincos_pos_embed(1024, 16)
    # print(pos_emb.shape)  # (256, 1024)
    # clip_emb = get_1d_sincos_pos_embed_from_grid(1024, np.arange(20))
    # print(clip_emb.shape)  # (20, 512)
