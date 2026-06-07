"""
Fréchet Video Distance (FVD) via the Kinetics I3D TorchScript checkpoint used in StyleGAN-V,
aligned with the reference TensorFlow FVD pipeline (see `universome/fvd-comparison`).

References:
  https://github.com/universome/fvd-comparison/blob/master/compare_models.py
  https://github.com/google-research/google-research/tree/master/frechet_video_distance
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

I3D_DEFAULT_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"


def default_i3d_cache_path() -> Path:
    return Path(__file__).resolve().parent / ".cache" / "i3d_torchscript.pt"


def ensure_i3d_weights(path: Path, url: str = I3D_DEFAULT_URL, verbose: bool = True) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading I3D TorchScript to {path} ...")
    urllib.request.urlretrieve(url, str(path))  # noqa: S310 — fixed official URL
    if verbose:
        print("Done.")


def preprocess_video_cthw(
    video_cthw: torch.Tensor,
    resolution: int = 224,
    sequence_length: Optional[int] = None,
) -> torch.Tensor:
    """Resize shorter side to `resolution`, center-crop square, map [0,1] -> [-1,1].

    video_cthw: float tensor (C, T, H, W) in [0, 1].
    """
    c, t, h, w = video_cthw.shape
    if sequence_length is not None:
        if sequence_length > t:
            raise ValueError(f"sequence_length={sequence_length} exceeds T={t}")
        video_cthw = video_cthw[:, :sequence_length]

    _, t, h, w = video_cthw.shape
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, int(math.ceil(w * scale)))
    else:
        target_size = (int(math.ceil(h * scale)), resolution)
    # 2D resize on H,W only: (C,T,H,W) -> (T,C,H,W) so F.interpolate sees spatial (H,W), not (T,H,W).
    x = video_cthw.permute(1, 0, 2, 3).contiguous()
    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    video_cthw = x.permute(1, 0, 2, 3).contiguous()

    _, _, h2, w2 = video_cthw.shape
    w_start = (w2 - resolution) // 2
    h_start = (h2 - resolution) // 2
    video_cthw = video_cthw[:, :, h_start : h_start + resolution, w_start : w_start + resolution]
    return ((video_cthw - 0.5) * 2.0).contiguous()


def frechet_distance_features(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    """Same Fréchet form as torch-fidelity FID (TTUR-style via eigenvalues of sigma1 @ sigma2)."""
    mu1 = np.mean(feats_fake, axis=0)
    mu2 = np.mean(feats_real, axis=0)
    sigma1 = np.cov(feats_fake, rowvar=False)
    sigma2 = np.cov(feats_real, rowvar=False)

    diff = mu1 - mu2
    tr_covmean = float(
        np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real)
    )
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


@torch.no_grad()
def i3d_features_bcthw(
    videos_bcthw: torch.Tensor,
    i3d: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """videos_bcthw: float32 (B, C, T, H, W) in [0, 1], C=3."""
    if videos_bcthw.dim() != 5:
        raise ValueError("Expected BCTHW")
    n = videos_bcthw.shape[0]
    feats: list[np.ndarray] = []
    kwargs = dict(rescale=False, resize=False, return_features=True)
    for start in range(0, n, batch_size):
        chunk = videos_bcthw[start : start + batch_size]
        processed = torch.stack(
            [preprocess_video_cthw(chunk[i]) for i in range(chunk.shape[0])],
            dim=0,
        ).to(device)
        out = i3d(processed, **kwargs)
        feats.append(out.detach().float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def load_i3d(detector_path: Path, device: torch.device) -> torch.nn.Module:
    ensure_i3d_weights(detector_path)
    m = torch.jit.load(str(detector_path)).eval().to(device)
    return m
