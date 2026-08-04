"""Unit tests for the pointing-gesture geometry (home_robot/gesture.py)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.gesture import (  # noqa: E402
    GestureStabilizer, arm_straightness, depths_are_consistent,
    floor_intersection, is_pointing,
)


# ── arm_straightness / is_pointing ────────────────────────────────────────────

def test_straight_arm_is_180_degrees():
    # shoulder --- elbow --- wrist, colinear
    assert arm_straightness((0, 0), (50, 0), (100, 0)) == pytest.approx(180.0)


def test_folded_arm_is_near_zero():
    # wrist doubled back to the shoulder
    assert arm_straightness((0, 0), (50, 0), (1, 0)) < 10.0


def test_right_angle_elbow():
    assert arm_straightness((0, 0), (50, 0), (50, 50)) == pytest.approx(90.0)


def test_pointing_accepts_extended_arm():
    assert is_pointing((0, 0), (50, 5), (100, 10))


def test_pointing_rejects_bent_arm():
    # A "talking" gesture: elbow at a right angle.
    assert not is_pointing((0, 0), (50, 0), (50, 50))


def test_pointing_rejects_tiny_skeleton():
    # Straight, but only a few pixels long — a distant/occluded person.
    assert not is_pointing((0, 0), (10, 0), (20, 0))


def test_degenerate_keypoints_do_not_crash():
    # All three keypoints identical: undefined angle, must not raise.
    assert arm_straightness((7, 7), (7, 7), (7, 7)) == 0.0
    assert not is_pointing((7, 7), (7, 7), (7, 7))


# ── depth consistency ─────────────────────────────────────────────────────────

def test_depths_consistent_for_a_real_arm():
    assert depths_are_consistent(2.0, 1.6)


def test_wrist_depth_landing_on_the_far_wall_is_rejected():
    # The classic failure: hand at 2 m, depth patch caught the wall at 5 m.
    assert not depths_are_consistent(2.0, 5.0)


def test_zero_depth_is_rejected():
    # RealSense reports 0 for "no reading", which must never become a point.
    assert not depths_are_consistent(0.0, 2.0)
    assert not depths_are_consistent(2.0, 0.0)


def test_out_of_range_depth_is_rejected():
    assert not depths_are_consistent(2.0, 9.0)


# ── floor_intersection ────────────────────────────────────────────────────────

def test_downward_point_hits_the_floor_in_front():
    # Shoulder at 1.4 m, wrist at 1.0 m, both on the +x axis: the ray descends
    # 0.4 m per 0.5 m of x, so from the wrist it needs 1.0/0.4 * 0.5 = 1.25 m.
    hit = floor_intersection((0.0, 0.0, 1.4), (0.5, 0.0, 1.0))
    assert hit is not None
    x, y = hit
    assert x == pytest.approx(1.75)
    assert y == pytest.approx(0.0)


def test_level_arm_never_reaches_the_floor():
    assert floor_intersection((0.0, 0.0, 1.2), (0.5, 0.0, 1.2)) is None


def test_upward_point_returns_none():
    # Pointing at a shelf or the ceiling is not a floor destination.
    assert floor_intersection((0.0, 0.0, 1.2), (0.5, 0.0, 1.5)) is None


def test_far_target_is_rejected():
    # Nearly level: the intersection lands implausibly far away.
    assert floor_intersection((0.0, 0.0, 1.4), (0.5, 0.0, 1.399)) is None


def test_straight_down_is_rejected_as_too_close():
    # Pointing at one's own feet is not a navigation goal.
    assert floor_intersection((0.0, 0.0, 1.4), (0.0, 0.0, 1.0)) is None


def test_diagonal_point_keeps_its_bearing():
    # Equal x and y components stay on the 45-degree line.
    hit = floor_intersection((0.0, 0.0, 1.4), (0.5, 0.5, 1.0))
    assert hit is not None
    x, y = hit
    assert x == pytest.approx(y)


def test_respects_a_raised_floor_plane():
    # floor_z is a parameter, so a non-zero map floor still works.
    hit = floor_intersection((0.0, 0.0, 2.4), (0.5, 0.0, 2.0), floor_z=1.0)
    assert hit is not None
    assert hit[0] == pytest.approx(1.75)


# ── GestureStabilizer ─────────────────────────────────────────────────────────

def _confirm(stab, x, y, n):
    """Feed n identical frames, returning whatever the stabilizer emitted."""
    out = [stab.update((x, y)) for _ in range(n)]
    return [o for o in out if o is not None]


def test_confirms_only_after_enough_agreeing_frames():
    s = GestureStabilizer(confirm_hits=5, radius=0.6)
    for _ in range(4):
        assert s.update((2.0, 1.0)) is None
    assert s.update((2.0, 1.0)) == pytest.approx((2.0, 1.0))


def test_single_frame_never_confirms():
    # One frame of bad depth must not send the robot anywhere.
    s = GestureStabilizer(confirm_hits=5)
    assert s.update((2.0, 1.0)) is None


def test_scattered_points_never_confirm():
    # A hand swinging past: every frame lands somewhere different.
    s = GestureStabilizer(confirm_hits=3, radius=0.5)
    fired = [s.update((float(i) * 2.0, 0.0)) for i in range(10)]
    assert all(f is None for f in fired)


def test_confirms_once_and_does_not_repeat_while_held():
    # Holding the point must not fire a goal every frame.
    s = GestureStabilizer(confirm_hits=3, radius=0.6)
    fired = _confirm(s, 2.0, 1.0, 20)
    assert len(fired) == 1


def test_aiming_somewhere_new_can_confirm_again():
    s = GestureStabilizer(confirm_hits=3, radius=0.5)
    assert len(_confirm(s, 2.0, 1.0, 5)) == 1
    # Swing to a clearly different spot and hold it.
    assert len(_confirm(s, 8.0, 4.0, 5)) == 1


def test_brief_dropout_does_not_lose_progress():
    s = GestureStabilizer(confirm_hits=4, radius=0.6, miss_tolerance=3)
    s.update((2.0, 1.0))
    s.update((2.0, 1.0))
    s.update(None)          # keypoint flickered out
    s.update((2.0, 1.0))
    assert s.update((2.0, 1.0)) == pytest.approx((2.0, 1.0))


def test_long_dropout_discards_the_candidate():
    s = GestureStabilizer(confirm_hits=3, radius=0.6, miss_tolerance=2)
    s.update((2.0, 1.0))
    s.update((2.0, 1.0))
    for _ in range(3):
        s.update(None)      # arm dropped
    assert s.candidate is None
    assert s.update((2.0, 1.0)) is None   # starts from scratch


def test_confirmed_point_is_the_cluster_mean_not_the_last_frame():
    # Jitter around (2, 1); the emitted point should sit in the middle rather
    # than inherit whichever frame happened to trip the threshold.
    s = GestureStabilizer(confirm_hits=3, radius=1.0)
    s.update((1.8, 1.0))
    s.update((2.0, 1.0))
    hit = s.update((2.2, 1.0))
    assert hit is not None
    assert hit[0] == pytest.approx(2.0)


def test_progress_reports_fraction_towards_confirmation():
    s = GestureStabilizer(confirm_hits=4, radius=0.6)
    assert s.progress == 0.0
    s.update((2.0, 1.0))
    assert s.progress == pytest.approx(0.25)
    s.update((2.0, 1.0))
    assert s.progress == pytest.approx(0.5)


def test_reset_clears_everything():
    s = GestureStabilizer(confirm_hits=3)
    s.update((2.0, 1.0))
    s.reset()
    assert s.candidate is None
    assert s.progress == 0.0


# ── end-to-end geometry ───────────────────────────────────────────────────────

def test_pipeline_from_skeleton_to_confirmed_floor_point():
    """A person standing at the origin pointing down-and-forward for 5 frames
    yields one confirmed goal, and it is in front of them."""
    s = GestureStabilizer(confirm_hits=5, radius=0.6)
    fired = None
    for _ in range(5):
        assert is_pointing((100, 200), (150, 220), (200, 240))
        assert depths_are_consistent(2.2, 1.9)
        hit = floor_intersection((0.0, 0.0, 1.45), (0.45, 0.0, 1.05))
        assert hit is not None
        fired = s.update(hit) or fired
    assert fired is not None
    assert fired[0] > 0.5          # in front of the person
    assert math.isclose(fired[1], 0.0, abs_tol=1e-6)
