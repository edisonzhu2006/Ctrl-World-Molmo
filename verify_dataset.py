#!/usr/bin/env python
# ABOUTME: Login-node dataset smoke test — builds train+val Dataset_mix from a .slurm
# ABOUTME: file's dataset args, constructs a DataLoader, and pulls a few real samples.
import argparse
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# Dataset-related launch args we lift from the .slurm so this checks exactly what
# the job will load. Everything else (num_frames, resolution, ...) comes from
# config.wm_args defaults, the same as train_wm.py.
DATASET_ARG_KEYS = [
    "dataset_root_path",
    "dataset_meta_info_path",
    "dataset_names",
    "dataset_cfgs",
    "dataset_stat_cfgs",
    "dataset_sampling_mode",
    "dataset_sampling_probs",
]


def parse_slurm_dataset_args(path):
    with open(path) as f:
        text = f.read()

    def arg(flag):
        m = re.search(r"(?:^|\s)--%s\s+(\S+)" % re.escape(flag), text)
        return m.group(1) if m else None

    return {k: arg(k) for k in DATASET_ARG_KEYS}


def build_args(slurm_path):
    sys.path.insert(0, HERE)
    from config import wm_args

    args = wm_args()
    parsed = parse_slurm_dataset_args(slurm_path)
    for k, v in parsed.items():
        if v is not None:
            setattr(args, k, v)
    # Mirror train_wm.merge_args: default cfg mapping to names when cfgs omitted.
    if getattr(args, "dataset_cfgs", None) is None and getattr(args, "dataset_names", None) is not None:
        args.dataset_cfgs = args.dataset_names
    return args, parsed


def check_mode(args, mode, batch_size, num_batches):
    import torch
    from dataset.dataset_droid_exp33_cross import Dataset_mix

    print(f"\n--- building {mode} dataset ---")
    ds = Dataset_mix(args, mode=mode)
    if len(ds) == 0:
        print(f"  [{mode}] dataset is empty; nothing to sample")
        return

    # num_workers=0 keeps this a light, well-behaved login-node check and gives
    # clean tracebacks if a sample fails to load.
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    it = iter(loader)
    seen = {}
    pulled = 0
    t0 = time.time()
    for b in range(num_batches):
        batch = next(it)
        pulled += len(batch["episode_id"])
        for name in batch["dataset_name"]:
            seen[name] = seen.get(name, 0) + 1
        if b == 0:
            print(f"  batch keys: {sorted(batch.keys())}")
            print(f"  latent: shape={tuple(batch['latent'].shape)} dtype={batch['latent'].dtype}")
            print(f"  action: shape={tuple(batch['action'].shape)} dtype={batch['action'].dtype}")
            print(f"  example text: {batch['text'][0][:80]!r}")
    dt = time.time() - t0
    print(f"  pulled {pulled} samples in {dt:.1f}s ({dt / max(pulled, 1):.2f}s/sample)")
    print(f"  datasets exercised: {seen}")


def main():
    ap = argparse.ArgumentParser(description="Login-node dataset/dataloader smoke test.")
    ap.add_argument("slurm", help="path to the run_*.slurm whose dataset args to verify")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--modes", default="train,val", help="comma-separated subset of train,val")
    a = ap.parse_args()

    args, parsed = build_args(a.slurm)
    print(f"dataset args parsed from {a.slurm}:")
    for k, v in parsed.items():
        print(f"  {k} = {v}")

    ok = True
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        try:
            check_mode(args, mode, a.batch_size, a.num_batches)
        except Exception:
            ok = False
            print(f"\n[{mode}] FAILED:")
            traceback.print_exc()

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
