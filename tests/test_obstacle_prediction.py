"""Tests for the constant-velocity obstacle prediction core
(see home_robot/obstacle_prediction.py).

Robot-free unit tests of the pixel→metric velocity conversion, the
predict-worthiness gate, and the straight-line projection.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_obstacle_prediction.py -q
"""
import math

from home_robot.obstacle_prediction import (
    camera_velocity_mps, should_predict, project_track, clamp_dt,
)


# ── px/cycle → m/s conversion ────────────────────────────────────────────

def test_velocity_scales_with_depth():
    # Same pixel velocity is a larger real velocity when the object is farther.
    near = camera_velocity_mps(10.0, 0.0, z_cam=1.0, fx=600, fy=600, dt=0.1)
    far  = camera_velocity_mps(10.0, 0.0, z_cam=4.0, fx=600, fy=600, dt=0.1)
    assert far[0] > near[0]


def test_velocity_independent_of_frame_rate():
    # 10 px over a 0.05 s cycle is the same m/s as 20 px over a 0.10 s cycle.
    a = camera_velocity_mps(10.0, 0.0, 2.0, 600, 600, dt=0.05)
    b = camera_velocity_mps(20.0, 0.0, 2.0, 600, 600, dt=0.10)
    assert math.isclose(a[0], b[0], rel_tol=1e-9)


def test_velocity_guards_bad_inputs():
    assert camera_velocity_mps(10, 0, 2.0, 600, 600, dt=0.0) == (0.0, 0.0)
    assert camera_velocity_mps(10, 0, -1.0, 600, 600, dt=0.1) == (0.0, 0.0)


# ── predict-worthiness gate ──────────────────────────────────────────────

def test_slow_object_not_predicted():
    assert not should_predict(0.05, 0.0, track_hits=10, min_speed=0.15, min_hits=3)


def test_fast_object_predicted():
    assert should_predict(0.5, 0.0, track_hits=10, min_speed=0.15, min_hits=3)


def test_young_track_not_predicted_even_if_fast():
    # A brand-new track has a wild SORT velocity — don't paint obstacles from it.
    assert not should_predict(2.0, 0.0, track_hits=1, min_speed=0.15, min_hits=3)


# ── straight-line projection ─────────────────────────────────────────────

def test_projection_walks_forward_in_time():
    pts = project_track(0.0, 0.0, 2.0, vx_ms=1.0, vy_ms=0.0, horizon=2.0, steps=4)
    assert len(pts) == 4
    xs = [p[0] for p in pts]
    assert xs == sorted(xs)                 # monotonically advancing
    assert math.isclose(xs[-1], 2.0)        # last point = v * horizon
    assert all(math.isclose(p[2], 2.0) for p in pts)   # constant depth


def test_projection_min_one_step():
    assert len(project_track(0, 0, 1, 1, 0, horizon=1.0, steps=0)) == 1


# ── dt clamping ──────────────────────────────────────────────────────────

def test_clamp_dt_bounds():
    assert clamp_dt(0.0) == 1.0            # non-positive → upper bound (safe)
    assert clamp_dt(5.0) == 1.0            # too large → clamped
    assert clamp_dt(0.001) == 0.02         # too small → clamped (no infinite v)
    assert clamp_dt(0.1) == 0.1            # in-band passes through
    assert clamp_dt(float('nan')) == 1.0   # NaN → safe
