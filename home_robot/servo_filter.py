"""Robust target aggregation for the pick visual-servo loop — the pure,
ROS-free core behind pick_place_node.py's _servo() and _hand_eye_correct().

The D435 is body-mounted (eye-to-hand): it can't see the gripper, so the
"servo" can't close a loop on the gripper→object error. What it *can* do is
watch the object over several frames and settle on a stable grasp point instead
of trusting one noisy snapshot — RealSense depth in particular is jittery, and a
single bad z means the gripper descends into the table or grasps air.

This module does that settling: collect the per-frame arm-frame estimates and
reduce them to a robust point (component-wise median, which shrugs off a lone
outlier frame that a mean would smear in), plus a convergence test on the
spread of recent samples. `hand_eye_correction` is the separate closed loop
added once a wrist-mounted AprilTag (`gripper_tag`) gave the camera something
on the gripper itself to watch — see that function's docstring. Dependency-free
so both unit-test without a robot — see tests/test_servo_filter.py.
"""

import math


def median(values: list[float]) -> float:
    """Median of a non-empty list (outlier-robust unlike the mean)."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def median_point(samples: list[tuple]) -> tuple | None:
    """Component-wise median of (x, y, z) samples. None if empty."""
    if not samples:
        return None
    return (median([p[0] for p in samples]),
            median([p[1] for p in samples]),
            median([p[2] for p in samples]))


def spread(samples: list[tuple]) -> float:
    """Largest XY distance of any sample from the sample-set median — a measure
    of how settled the estimate is. 0 for fewer than two samples."""
    if len(samples) < 2:
        return 0.0
    mx, my, _ = median_point(samples)
    return max(math.hypot(p[0] - mx, p[1] - my) for p in samples)


def converged(samples: list[tuple], tolerance: float, min_samples: int = 2) -> bool:
    """True once at least `min_samples` frames agree to within `tolerance` (m)
    in XY — i.e. the target estimate has stopped wandering."""
    if len(samples) < min_samples:
        return False
    return spread(samples[-min_samples:]) <= tolerance


def hand_eye_correction(cmd_xy: tuple, target_xy: tuple, actual_xy: tuple,
                        tolerance: float) -> tuple:
    """One proportional step of pick_place_node's wrist-tag hand-eye loop.

    `actual_xy` is where the wrist AprilTag says the gripper really is after
    moving to `cmd_xy` — which, unlike the object-detection servo above, can
    disagree with `cmd_xy` itself: that gap IS the base_link->arm_base
    calibration bias the module docstring says the camera can't otherwise see
    (it never looks at the gripper). Nudge the next command by the observed
    error and let it settle over a few iterations rather than trusting one
    noisy tag frame.

    Returns (next_cmd_xy, converged)."""
    err_x = target_xy[0] - actual_xy[0]
    err_y = target_xy[1] - actual_xy[1]
    if math.hypot(err_x, err_y) <= tolerance:
        return cmd_xy, True
    return (cmd_xy[0] + err_x, cmd_xy[1] + err_y), False
