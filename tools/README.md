# tools/

## `viz_molmobot_quat_bug.py`

Rerun-based 3D visualizer that overlays two interpretations of the stored
`cart[:, 3:6]` Euler angles (as-stored vs. swapped under the opposite
scalar-quaternion convention) against the 3 camera frames so you can
verify by eye whether a converted MolmoBot dataset is in the canonical
molmospaces rotation frame.

This is the same script that was used to surface the original
[molmo2ctrl quaternion-convention bug](https://github.com/LeiHHHuang/molmo2ctrl)

(see commit `Fix scalar-first quaternion misread in tcp_pose conversion`
on that repo). Running it on the bundled `dataset_example/molmobot_small/`
should now show the **STORED** triad matching the wrist-camera evidence
on every episode — confirming the repair landed correctly.

### Setup

`rerun-sdk` is in `requirements.txt`. The visualizer also needs `ffmpeg`
on the system path (used by `mediapy` for mp4 decode). Install via
`apt install ffmpeg` (or your conda env equivalent).

### Usage

Save mode (write a `.rrd` you can scp + open with the native rerun viewer):

```bash
python tools/viz_molmobot_quat_bug.py \
    --dataset_root dataset_example/molmobot_small \
    --episode_id 23 \
    --split train \
    --mode save --save_path /tmp/foo.rrd
```

Serve mode (HTTP server on `9090` for the web viewer, gRPC on `9876` for
the data buffer; SSH-tunnel both to your laptop):

```bash
python tools/viz_molmobot_quat_bug.py \
    --dataset_root dataset_example/molmobot_small \
    --episode_id 23 --split train --mode serve

# from your laptop
ssh -L 9090:localhost:9090 -L 9876:localhost:9876 <host>
# then open
http://localhost:9090/?url=rerun+http://localhost:9876/proxy
```

Without the `?url=…` query param the viewer shows the rerun examples
landing page; the param tells it to auto-connect to our gRPC server.

### What you'll see

Two coordinate triads at the EE position:

- **STORED** (warm — red/orange/yellow): the rotation as the dataset stores it.
- **SWAPPED** (cool — blue/cyan/magenta): same Euler reinterpreted under
  the opposite scalar convention.

Plus a yellow `LineStrips3D` for the next 16 steps' position trail (the
action chunk a chunked BC/IQL head would predict at this anchor) and the
3 camera frames as 2D image panels.

The test on either dataset is the same: which triad's Z-axis tracks the
gripper's apparent orientation in the **wrist camera** image?

- On a buggy (pre-repair) MolmoBot dataset: SWAPPED matches.
- On a fixed dataset: STORED matches.

For dramatic effect, episodes with high rotation variation in the
bundled example data:

| split | id | T   | instruction                            |
|-------|----|-----|----------------------------------------|
| train | 23 | 116 | Pick up the fungus model               |
| train | 11 | 100 | Pick up the brown textured mushroom    |
| val   | 164| 68  | Pick up the detailed rusty white truck |

## `test_molmobot_quat_bug.py`

20-test pytest module pinning the recovery formula used internally by
the visualizer (and the upstream `repair_quat_convention.py`) to <1e-6
angular error on synthetic round-trips: identity, principal-axis
rotations, realistic Franka gripper-down poses, near-gimbal-lock cases,
and 256 random unit quaternions.

```bash
pytest tools/test_molmobot_quat_bug.py -v
```
