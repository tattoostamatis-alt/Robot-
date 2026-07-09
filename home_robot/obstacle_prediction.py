"""Constant-velocity obstacle prediction — the pure, ROS-free core behind
obstacle_prediction_node.py.

The SORT tracker (tracker_node.py) reports each object's velocity in *pixels per
detection cycle* — a pixel-space Kalman state, with no notion of metres or
wall-clock seconds. To inflate the local costmap *ahead* of a moving person we
need that motion in the world: metres per second, projected over a time horizon.

This module does exactly that conversion and the straight-line projection, kept
dependency-free (no ROS, no numpy) so the maths can be unit-tested without a
camera or a running graph — see tests/test_obstacle_prediction.py. The ROS node
supplies the live camera intrinsics, the measured detection-cycle dt, and the
camera→base_link TF.
"""

import math


def camera_velocity_mps(vel_px: float, vel_py: float, z_cam: float,
                        fx: float, fy: float, dt: float) -> tuple[float, float]:
    """Convert a pixel-space velocity (pixels per detection cycle) into a
    metric velocity (m/s) in the camera frame, at object depth `z_cam`.

    A displacement of `vel_px` pixels at depth z subtends `vel_px * z / fx`
    metres (pinhole model); dividing by the cycle time dt turns "per cycle"
    into "per second". Returns (0, 0) for a non-positive dt or depth so a
    missing/garbage timing can never blow up the projection.
    """
    if dt <= 0.0 or z_cam <= 0.0 or fx <= 0.0 or fy <= 0.0:
        return 0.0, 0.0
    vx = vel_px * z_cam / fx / dt
    vy = vel_py * z_cam / fy / dt
    return vx, vy


def should_predict(vx_ms: float, vy_ms: float, track_hits: int,
                   min_speed: float, min_hits: int) -> bool:
    """Whether a track is worth projecting: moving faster than `min_speed`
    (m/s) AND seen enough times that its velocity estimate has settled.

    SORT initialises velocity with huge covariance, so a one- or two-frame-old
    track has a wild, meaningless velocity; requiring `min_hits` observations
    first stops us painting phantom obstacles from that startup transient.
    """
    if track_hits < min_hits:
        return False
    return math.hypot(vx_ms, vy_ms) >= min_speed


def project_track(x_cam: float, y_cam: float, z_cam: float,
                  vx_ms: float, vy_ms: float,
                  horizon: float, steps: int) -> list[tuple]:
    """Straight-line (constant-velocity, constant-depth) forecast of an object's
    camera-frame position at `steps` evenly spaced instants out to `horizon`
    seconds. Returns a list of (x, y, z) points to mark in the costmap.
    """
    pts = []
    steps = max(1, int(steps))
    for step in range(1, steps + 1):
        t = horizon * step / steps
        pts.append((x_cam + vx_ms * t, y_cam + vy_ms * t, z_cam))
    return pts


def clamp_dt(dt: float, lo: float = 0.02, hi: float = 1.0) -> float:
    """Keep the measured detection-cycle dt in a sane band. A hiccup (paused
    detector, first message) must not produce dt≈0 (→ infinite velocity) or a
    multi-second dt (→ a stale track flung across the room)."""
    if dt <= 0.0 or dt != dt:   # non-positive or NaN
        return hi
    return min(hi, max(lo, dt))
