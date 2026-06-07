# ABOUTME: Rerun visualizer for the molmo2ctrl quaternion-convention bug.
# ABOUTME: Shows buggy vs corrected EE triad alongside camera frames + action chunks.

"""Visualize an annotation produced by ``molmo2ctrl/convert_molmobot_to_ctrlworld.py``
and show, for every timestep:

  - The end-effector triad as the buggy converter wrote it (what the model
    sees during training).
  - The end-effector triad recovered to the rotation that was actually in the
    raw H5 file's ``obs/extra/tcp_pose`` (scalar-first).
  - The 3 camera frames the dataloader feeds to the model.
  - A position trail for the next ``chunk_len`` steps — i.e. the action chunk
    the BC/IQL/AWR head predicts at this anchor.

Reading the visualization
-------------------------
The wrist camera is rigidly mounted on the gripper, so its orientation in
the world matches the gripper's. If the buggy triad and the corrected triad
disagree (they will, by ~π for most poses), the one whose Z axis points
"down into the wrist camera's apparent view direction" is the rotation that
matches the camera's evidence — and that's the one the rest of molmospaces
is consistent with.

Usage
-----
SSH-tunnel mode (preferred — open the URL on your laptop)::

    pixi run python tools/viz_molmobot_quat_bug.py \\
        --dataset_root /path/to/molmobot_small \\
        --episode_id 0 \\
        --mode serve --port 9876

Save-to-file mode (review offline with ``rerun foo.rrd``)::

    pixi run python tools/viz_molmobot_quat_bug.py \\
        --dataset_root /path/to/molmobot_small \\
        --episode_id 0 \\
        --mode save --save_path /tmp/quat_bug.rrd
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mediapy
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

# Fixed camera-name → annotation-slot mapping used by Lei's converter
# (molmo2ctrl/convert_molmobot_to_ctrlworld.py:33-37).
CAMERA_NAMES = [
    "droid_shoulder_light_randomization",
    "randomized_zed2_analogue_1",
    "wrist_camera_zed_mini",
]


# ---------------------------------------------------------------------------
# Recovery formula: invert molmo2ctrl/convert_molmobot_to_ctrlworld.py:101
#
# The converter does
#     euler_buggy = R.from_quat(tcp_pose[:, 3:7]).as_euler("xyz")
# without ``scalar_first=True``. The H5 stores a scalar-first quat
# ``[w, x, y, z]``; scipy default is scalar-last so it reads it as
# ``[x', y', z', w']`` — a different unit quaternion ``z + w·i + x·j + y·k``
# — and produces Euler angles for that wrong rotation.
#
# Given only the buggy Euler, we reconstruct the original rotation:
#     R_buggy   = R.from_euler("xyz", euler_buggy)
#     q_xyzw    = R_buggy.as_quat()           # scipy default: scalar-LAST
#     R_correct = R.from_quat(q_xyzw, scalar_first=True)
# (q and -q represent the same rotation, so any sign flip scipy applies
# during canonicalization is harmless.)
# ---------------------------------------------------------------------------


def apply_buggy_quat_misread(q_wxyz: np.ndarray) -> R:
    """Replicate the bug. Pass a scalar-first quat ``[w, x, y, z]`` and get
    back the ``Rotation`` the buggy converter produces.
    """
    return R.from_quat(np.asarray(q_wxyz))


def recover_true_rotation_from_buggy_euler(euler_buggy: np.ndarray) -> R:
    """Given the converter's buggy ``cart[3:6]``, return the ``Rotation``
    that was originally in the raw H5 file's ``obs/extra/tcp_pose``.
    """
    R_buggy = R.from_euler("xyz", np.asarray(euler_buggy))
    q_xyzw = R_buggy.as_quat()
    return R.from_quat(q_xyzw, scalar_first=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episode(annotation_path: Path, dataset_root: Path):
    """Return ``(ann, cart, grip, video_paths)``.

    ``cart`` is ``[T, 6]`` (xyz + buggy Euler-XYZ); ``grip`` is ``[T]``.
    """
    ann = json.loads(annotation_path.read_text())
    cart = np.asarray(ann["observation.state.cartesian_position"], dtype=np.float64)
    grip = np.asarray(ann["observation.state.gripper_position"], dtype=np.float64)
    video_paths = [dataset_root / v["video_path"] for v in ann["videos"]]
    return ann, cart, grip, video_paths


def read_video(path: Path) -> np.ndarray:
    """Decode an mp4 to a uint8 RGB array ``[T, H, W, 3]``."""
    return np.asarray(mediapy.read_video(str(path)), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_episode(
    ann: dict,
    cart: np.ndarray,
    grip: np.ndarray,
    videos: list[np.ndarray],
    chunk_len: int = 16,
) -> None:
    """Log all timesteps of one episode to the active rerun recording."""
    T = cart.shape[0]

    # World coordinate system: Z up, right-handed (matches molmospaces world frame).
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    instruction = ann["texts"][0] if ann.get("texts") else ""
    rr.log(
        "instruction",
        rr.TextDocument(f"episode_id={ann.get('episode_id', '?')}\n{instruction}"),
        static=True,
    )

    # Two interpretations of the stored cart[:, 3:6] Euler:
    #   AS-STORED  — read the Euler at face value (what the model trains on).
    #   SWAPPED    — apply the inverse of the converter's quat-component swap
    #                (the recovery formula). On data produced by the buggy
    #                converter this gives the right rotation; on data produced
    #                by the FIXED converter it gives the wrong rotation.
    # The test for either dataset is the same: which triad's Z-axis tracks
    # the gripper's apparent orientation in the wrist camera?
    R_as_stored = R.from_euler("xyz", cart[:, 3:6])
    R_swapped = R.from_quat(R_as_stored.as_quat(), scalar_first=True)

    AXIS_LEN = 0.08
    AS_STORED_COLORS = np.array(
        [[230, 50, 50], [240, 140, 30], [240, 220, 30]], dtype=np.uint8
    )  # warm: X=red, Y=orange, Z=yellow
    SWAPPED_COLORS = np.array(
        [[40, 100, 230], [40, 200, 230], [220, 60, 220]], dtype=np.uint8
    )  # cool: X=blue, Y=cyan, Z=magenta

    for t in range(T):
        rr.set_time("step", sequence=t)

        pos = cart[t, :3]
        Ra = R_as_stored[t].as_matrix()
        Rs = R_swapped[t].as_matrix()

        # Triad A: rotation as the dataset stores it. The columns of the
        # rotation matrix are the world-frame coordinate axis directions;
        # row-major numpy makes the columns ``Ra.T``.
        rr.log(
            "world/ee/as_stored",
            rr.Arrows3D(
                origins=np.tile(pos, (3, 1)),
                vectors=Ra.T * AXIS_LEN,
                colors=AS_STORED_COLORS,
                labels=["STORED x", "STORED y", "STORED z"],
            ),
        )

        # Triad B: same Euler reinterpreted under the opposite scalar
        # convention. Useful for spot-checking that the converter produced
        # data in the rotation frame the rest of molmospaces expects.
        rr.log(
            "world/ee/swapped",
            rr.Arrows3D(
                origins=np.tile(pos, (3, 1)),
                vectors=Rs.T * AXIS_LEN,
                colors=SWAPPED_COLORS,
                labels=["SWAP x", "SWAP y", "SWAP z"],
            ),
        )

        # Action-chunk position trail: the next chunk_len positions the
        # state will visit. Matches what the BC/IQL chunked head would
        # predict at anchor t (modulo normalization).
        end = min(t + chunk_len + 1, T)
        chunk_xyz = cart[t:end, :3]
        if chunk_xyz.shape[0] >= 2:
            rr.log(
                "world/action_chunk",
                rr.LineStrips3D(
                    [chunk_xyz],
                    colors=[(255, 200, 0)],
                ),
            )
        else:
            rr.log("world/action_chunk", rr.Clear(recursive=False))

        # Gripper scalar over time.
        rr.log("gripper/value", rr.Scalars(float(grip[t])))

        # Cameras as 2D panels.
        for cam_name, vid in zip(CAMERA_NAMES, videos):
            if t < vid.shape[0]:
                rr.log(f"cameras/{cam_name}", rr.Image(vid[t]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, required=True,
                   help="Root of converted molmobot dataset (the dir containing annotation/ and videos/)")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--episode_id", type=int, required=True)
    p.add_argument("--chunk_len", type=int, default=16,
                   help="Action-chunk length (matches BC/IQL chunked_chunk_length default)")
    p.add_argument("--mode", choices=["serve", "save"], default="serve",
                   help="serve: rerun web viewer + gRPC data server (SSH-tunnel-friendly); "
                        "save: write a .rrd file you can scp + open with `rerun foo.rrd`")
    p.add_argument("--web_port", type=int, default=9090,
                   help="HTTP port serving the rerun web viewer HTML/WASM (--mode serve). "
                        "Forward this to your laptop with: ssh -L 9090:localhost:9090 ...")
    p.add_argument("--grpc_port", type=int, default=9876,
                   help="gRPC port the web viewer connects back to for data (--mode serve). "
                        "Also forward this with: ssh -L 9876:localhost:9876 ...")
    p.add_argument("--save_path", type=Path, default=Path("/tmp/molmobot_quat_bug.rrd"),
                   help="Path for the .rrd file (--mode save)")
    args = p.parse_args()

    ann_path = args.dataset_root / "annotation" / args.split / f"{args.episode_id}.json"
    if not ann_path.is_file():
        raise SystemExit(f"Annotation not found: {ann_path}")

    ann, cart, grip, video_paths = load_episode(ann_path, args.dataset_root)
    videos = [read_video(vp) for vp in video_paths]

    print(f"Loaded episode {args.episode_id} ({args.split}): "
          f"T={cart.shape[0]}, instruction={ann.get('texts', [''])[0]!r}")
    for cam_name, vp, vid in zip(CAMERA_NAMES, video_paths, videos):
        print(f"  {cam_name}: {vid.shape} from {vp}")

    rr.init(f"molmobot_quat_bug_traj_{args.episode_id}", spawn=False)

    if args.mode == "serve":
        # Split since rerun 0.21+: serve_grpc holds the data buffer, serve_web_viewer
        # serves the HTML/WASM client. Browser connects to grpc_port for data.
        server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=server_uri)

    log_episode(ann, cart, grip, videos, chunk_len=args.chunk_len)

    if args.mode == "save":
        rr.save(str(args.save_path))
        print(f"Saved {args.save_path}. View with: rerun {args.save_path}")
        return

    # Without the ?url= query param the viewer shows the rerun examples landing
    # page; the param tells it to auto-connect to our gRPC server.
    viewer_url = f"http://localhost:{args.web_port}/?url=rerun+http://localhost:{args.grpc_port}/proxy"
    print(
        f"Serving:\n"
        f"  gRPC server {server_uri}\n"
        f"  web viewer  http://localhost:{args.web_port}\n"
        f"\n"
        f"SSH tunnel both ports to your laptop:\n"
        f"  ssh -L {args.web_port}:localhost:{args.web_port} -L {args.grpc_port}:localhost:{args.grpc_port} <host>\n"
        f"\n"
        f"Then open in your laptop browser:\n"
        f"  {viewer_url}\n"
        f"\n"
        f"Ctrl-C to stop."
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping.")


if __name__ == "__main__":
    main()
