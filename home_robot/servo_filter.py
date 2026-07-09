"""Robust target aggregation for the pick visual-servo loop — the pure,
ROS-free core behind pick_place_node.py's _servo().

The D435 is body-mounted (eye-to-hand): it can't see the gripper, so the
"servo" can't close a loop on the gripper→object error. What it *can* do is
watch the object over several frames and settle on a stable grasp point instead
of trusting one noisy snapshot — RealSense depth in particular is jittery, and a
single bad z means the gripper descends into the table or grasps air.

This module does that settling: collect the per-frame arm-frame estimates and
reduce them to a robust point (component-wise median, which shrugs off a lone
outlier frame that a mean would smear in), plus a convergence test on the
spread of recent samples. Dependency-free so it unit-tests without a robot —
see tests/test_servo_filter.py.
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
