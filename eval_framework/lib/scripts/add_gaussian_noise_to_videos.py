"""
Add i.i.d. Gaussian noise (in [0, 255] space) to every frame of each MP4 under a source folder.

Use this to build a "medium quality" checkpoint from ground truth (or any reference rollout) so
metrics are not identical to GT.

Theoretical PSNR (clean vs. clean + noise, ignoring clipping and spatial correlation):
  MSE ≈ sigma^2  =>  PSNR ≈ 10 * log10(255^2 / sigma^2) = 20 * log10(255 / sigma)

SSIM and LPIPS do not have a simple closed form from sigma alone; after generating videos, run
`metrics/compute_psnr_ssim_lpips.py` to measure them.

Requires: numpy, ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_r_frame_rate(expr: str) -> float:
    if "/" in expr:
        num, den = expr.split("/", 1)
        return float(num) / float(den)
    return float(expr)


def ffprobe_video_stream(path: Path) -> Tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    w = h = None
    fps_expr = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("width="):
            w = int(line.split("=", 1)[1])
        elif line.startswith("height="):
            h = int(line.split("=", 1)[1])
        elif line.startswith("r_frame_rate="):
            fps_expr = line.split("=", 1)[1]
    if w is None or h is None or not fps_expr:
        raise RuntimeError(f"ffprobe could not read video stream for {path}")
    return w, h, parse_r_frame_rate(fps_expr)


def read_video_rgb_uint8(path: Path) -> Tuple[np.ndarray, float]:
    w, h, fps = ffprobe_video_stream(path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    frame_nbytes = w * h * 3
    if len(raw) % frame_nbytes != 0:
        raise RuntimeError(
            f"Decoded byte length {len(raw)} not divisible by frame size {frame_nbytes} for {path}"
        )
    n = len(raw) // frame_nbytes
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3)
    return frames, fps


def write_video_rgb_uint8(frames: np.ndarray, fps: float, out_path: Path) -> None:
    t, h, w, c = frames.shape
    if c != 3:
        raise ValueError(f"Expected RGB last dim 3, got {frames.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(t):
        proc.stdin.write(np.ascontiguousarray(frames[i], dtype=np.uint8).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) writing {out_path}:\n{err}")


def expected_psnr_gaussian(sigma: float, data_range: float = 255.0) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(10.0 * math.log10((data_range**2) / (sigma**2)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add Gaussian noise to MP4s in a folder tree.")
    p.add_argument("--root", type=Path, default=Path("eval_framework"), help="Eval root (cwd-relative).")
    p.add_argument(
        "--src-subdir",
        type=str,
        required=True,
        help="Source folder relative to root (e.g. data/gt/prism/sample_0).",
    )
    p.add_argument(
        "--dst-subdir",
        type=str,
        required=True,
        help="Output folder relative to root (e.g. data/predictions/my_run/checkpoint-15000/prism/trial_0/sample_0).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=15.0,
        help="Noise std dev in pixel units [0, 255]. Default 15 => PSNR ≈ 20*log10(255/15) ≈ 24.6 dB.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (per-file seed offsets by file index).")
    p.add_argument(
        "--meta-name",
        type=str,
        default="noise_meta.json",
        help="Write a small JSON next to outputs with sigma, seed, theoretical_psnr.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    src_dir = root / args.src_subdir
    dst_dir = root / args.dst_subdir
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(src_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 files in {src_dir}")

    theo = expected_psnr_gaussian(args.sigma)
    meta = {
        "sigma": args.sigma,
        "seed_base": args.seed,
        "theoretical_psnr_db_mse_equals_sigma2": theo,
        "note": "PSNR formula assumes i.i.d. Gaussian noise and ignores uint8 clipping; measured PSNR will be close.",
        "src_subdir": args.src_subdir,
        "dst_subdir": args.dst_subdir,
        "files": [],
    }

    for i, src in enumerate(mp4s):
        frames, fps = read_video_rgb_uint8(src)
        rng = np.random.default_rng(args.seed + i)
        noise = rng.standard_normal(frames.shape, dtype=np.float64) * float(args.sigma)
        noisy = np.clip(frames.astype(np.float64) + noise, 0.0, 255.0).astype(np.uint8)
        out = dst_dir / src.name
        write_video_rgb_uint8(noisy, fps, out)
        meta["files"].append({"name": src.name, "seed": args.seed + i})

    meta_path = dst_dir / args.meta_name
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(mp4s)} noisy videos to {dst_dir}")
    print(f"Theoretical PSNR (sigma={args.sigma}): {theo:.4f} dB")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
