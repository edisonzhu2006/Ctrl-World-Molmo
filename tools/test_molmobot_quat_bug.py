# ABOUTME: Tests the recovery formula for the molmo2ctrl quaternion-convention bug.
# ABOUTME: Round-trips synthetic rotations through the bug + recovery to <1e-6 angular error.

"""Pytest suite anchoring ``recover_true_rotation_from_buggy_euler``.

We don't have raw H5 access, so the recovery formula has to be provable
from synthetic data alone. For each test rotation R_true:

    1. q_wxyz = R_true.as_quat(scalar_first=True)         — molmospaces convention
    2. R_buggy = apply_buggy_quat_misread(q_wxyz)          — converter line 101
    3. euler_buggy = R_buggy.as_euler("xyz")               — what gets stored in JSON
    4. R_recovered = recover_true_rotation_from_buggy_euler(euler_buggy)
    5. assert angular_error(R_true, R_recovered) < 1e-6

If this round-trip ever drifts above tolerance the visualizer's "corrected"
triad is no longer trustworthy, and the bug-prove flow falls apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Allow ``pytest tools/test_molmobot_quat_bug.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.viz_molmobot_quat_bug import (  # noqa: E402
    apply_buggy_quat_misread,
    recover_true_rotation_from_buggy_euler,
)


def angular_error(R_a: R, R_b: R) -> float:
    """Geodesic distance on SO(3), in radians."""
    return float((R_a.inv() * R_b).magnitude())


def _round_trip(R_true: R) -> float:
    q_wxyz = R_true.as_quat(scalar_first=True)
    R_buggy = apply_buggy_quat_misread(q_wxyz)
    euler_buggy = R_buggy.as_euler("xyz")
    R_recovered = recover_true_rotation_from_buggy_euler(euler_buggy)
    return angular_error(R_true, R_recovered)


def test_identity_round_trips():
    assert _round_trip(R.identity()) < 1e-6


@pytest.mark.parametrize(
    "axis,angle_deg",
    [
        ("x", 90),  ("y", 90),  ("z", 90),
        ("x", 180), ("y", 180), ("z", 180),
        ("x", 45),  ("y", -30), ("z", 137),
        ("x", -90), ("y", 60),  ("z", -90),
    ],
)
def test_principal_axis_rotations_round_trip(axis: str, angle_deg: float):
    R_true = R.from_euler(axis, angle_deg, degrees=True)
    assert _round_trip(R_true) < 1e-6


@pytest.mark.parametrize(
    "euler_deg",
    [
        (180, 0, 0),     # gripper-down primary case (typical Franka pick pose)
        (180, 0, 180),   # gripper-down with yaw flip
        (170, 13, 170),  # close to what the molmobot_small data shows
        (45, 45, 45),    # generic
        (-90, 90, 30),   # near gimbal lock — pitch ≈ +π/2
    ],
)
def test_realistic_franka_poses_round_trip(euler_deg):
    R_true = R.from_euler("xyz", euler_deg, degrees=True)
    assert _round_trip(R_true) < 1e-6


def test_random_unit_quats_round_trip():
    rng = np.random.default_rng(42)
    n = 256
    R_true_batch = R.random(n, random_state=rng)
    max_err = 0.0
    for i in range(n):
        max_err = max(max_err, _round_trip(R_true_batch[i]))
    # Slightly looser tolerance for batch with random near-singular cases.
    assert max_err < 1e-5, f"max angular error over {n} random rotations: {max_err:.3e}"


def test_bug_actually_changes_rotation_for_typical_pose():
    """Sanity-check that the bug produces a meaningfully different rotation
    for a typical Franka EE pose. If this test ever passes with ~0 error,
    the bug has been fixed (or stopped existing) and the visualizer is
    showing nothing interesting — worth investigating.
    """
    R_true = R.from_euler("xyz", [180, 0, 0], degrees=True)  # gripper down
    q_wxyz = R_true.as_quat(scalar_first=True)
    R_buggy = apply_buggy_quat_misread(q_wxyz)
    err = angular_error(R_true, R_buggy)
    assert err > 0.5, (
        f"buggy rotation differs from true by only {err:.4f} rad — "
        "the bug should produce ≳π for typical poses"
    )
