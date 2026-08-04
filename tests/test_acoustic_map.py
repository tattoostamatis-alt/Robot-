"""Unit tests for the acoustic map (home_robot/acoustic_map.py)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.acoustic_map import (  # noqa: E402
    SoundSourceMap, bearing_to_world, ray_points,
)

D = math.radians


# ── the angle convention ──────────────────────────────────────────────────────

def test_zero_bearing_is_straight_ahead():
    assert bearing_to_world(D(30), 0) == pytest.approx(D(30))


def test_doa_and_yaw_ADD():
    """‼️ The XVF3800's 0 is the robot's nose and grows counter-clockwise, the
    same sense as ROS yaw — so they add. The compass subtracts, because compass
    bearings grow clockwise. Same trap, opposite answer."""
    assert bearing_to_world(0.0, 90) == pytest.approx(D(90))
    assert bearing_to_world(D(90), 90) == pytest.approx(D(180))


def test_result_is_wrapped():
    for yaw in range(-360, 361, 45):
        for doa in range(0, 360, 45):
            a = bearing_to_world(D(yaw), doa)
            assert -math.pi - 1e-9 <= a <= math.pi + 1e-9


def test_a_sound_behind_the_robot():
    assert abs(bearing_to_world(0.0, 180)) == pytest.approx(math.pi)


# ── ray casting ───────────────────────────────────────────────────────────────

def test_ray_starts_clear_of_the_robot():
    """The robot's own body and the mic mount sit at the origin; a hotspot on
    the cell the robot is standing in would always land on the dock."""
    pts = ray_points(0, 0, 0.0)
    assert min(math.hypot(x, y) for x, y in pts) >= 0.29


def test_ray_follows_the_angle():
    pts = ray_points(0, 0, D(90), max_range=2.0)
    assert all(abs(x) < 1e-9 for x, _ in pts)
    assert all(y > 0 for _, y in pts)


def test_ray_respects_max_range():
    pts = ray_points(0, 0, 0.0, max_range=2.0)
    assert max(x for x, _ in pts) <= 2.0


def test_ray_starts_from_the_robot_position():
    pts = ray_points(5.0, -3.0, 0.0, max_range=1.0)
    assert pts[0][0] > 5.0
    assert pts[0][1] == pytest.approx(-3.0)


# ── accumulation ──────────────────────────────────────────────────────────────

def test_one_bearing_is_not_a_location():
    """‼️ A single ray makes every cell along it equally likely. Returning its
    far end as 'where the doorbell is' would be an invention."""
    m = SoundSourceMap()
    m.observe('doorbell', 0, 0, 0.0, 0)
    assert m.hotspots('doorbell') == []


def test_two_crossing_bearings_find_the_source():
    # A source at (3, 0): seen straight ahead from the origin, and at 90 deg
    # from (3, -3) where the robot faces +x.
    m = SoundSourceMap()
    m.observe('doorbell', 0.0, 0.0, 0.0, 0)
    m.observe('doorbell', 3.0, -3.0, 0.0, 90)
    spots = m.hotspots('doorbell', top=1)
    assert spots
    x, y, _ = spots[0]
    assert math.dist((x, y), (3.0, 0.0)) < 0.6


def test_repeated_bearings_from_one_spot_still_do_not_localize():
    m = SoundSourceMap()
    for _ in range(20):
        m.observe('tap', 0.0, 0.0, 0.0, 45)
    spots = m.hotspots('tap', top=1)
    # It will report something — twenty rays is evidence of a DIRECTION — but
    # the peak must sit on that ray, not somewhere invented.
    assert spots
    x, y, _ = spots[0]
    assert abs(math.atan2(y, x) - D(45)) < 0.15


def test_nearer_cells_weigh_more_than_far_ones():
    """A few degrees of bearing error is centimetres nearby and metres far
    away; without the decay the far cells win by sheer count."""
    m = SoundSourceMap()
    m.observe('x', 0, 0, 0.0, 0)
    near = m._grid[m._key('x', 0.5, 0.0)]
    far = m._grid[m._key('x', 7.0, 0.0)]
    assert near > far


def test_kinds_are_kept_apart():
    m = SoundSourceMap()
    m.observe('doorbell', 0, 0, 0.0, 0)
    m.observe('doorbell', 3, -3, 0.0, 90)
    m.observe('water', 0, 0, 0.0, 180)
    m.observe('water', -3, 3, 0.0, 270)
    assert set(m.kinds()) == {'doorbell', 'water'}
    db = m.hotspots('doorbell', top=1)[0]
    wt = m.hotspots('water', top=1)[0]
    assert math.dist(db[:2], wt[:2]) > 2.0


def test_observation_counts():
    m = SoundSourceMap()
    for _ in range(4):
        m.observe('doorbell', 0, 0, 0.0, 0)
    assert m.observations('doorbell') == 4
    assert m.observations('never') == 0


def test_missing_bearing_is_ignored():
    m = SoundSourceMap()
    assert m.observe('doorbell', 0, 0, 0.0, None) == []
    assert m.observations('doorbell') == 0


def test_blank_kind_is_ignored():
    m = SoundSourceMap()
    assert m.observe('', 0, 0, 0.0, 0) == []


def test_hotspots_are_not_the_same_blob_three_times():
    m = SoundSourceMap()
    for i in range(6):
        m.observe('doorbell', 0.0, i * 0.05, 0.0, 0)
    spots = m.hotspots('doorbell', top=3)
    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            assert math.dist(spots[i][:2], spots[j][:2]) >= 4 * m.cell - 1e-9


def test_prune_drops_the_noise():
    m = SoundSourceMap()
    m.observe('x', 0, 0, 0.0, 0)
    before = len(m._grid)
    m.prune(floor=0.9)
    assert len(m._grid) < before


# ── persistence ───────────────────────────────────────────────────────────────

def test_round_trip():
    m = SoundSourceMap()
    m.observe('doorbell', 0.0, 0.0, 0.0, 0)
    m.observe('doorbell', 3.0, -3.0, 0.0, 90)
    restored = SoundSourceMap.from_dict(m.to_dict())
    assert restored.observations('doorbell') == 2
    assert restored.hotspots('doorbell', top=1) == m.hotspots('doorbell', top=1)


def test_from_dict_tolerates_empty():
    assert SoundSourceMap.from_dict({}).kinds() == []
    assert SoundSourceMap.from_dict(None).kinds() == []


def test_from_dict_skips_corrupt_keys():
    m = SoundSourceMap.from_dict({'grid': {'bad-key': 1.0, 'x|1|2': 3.0},
                                  'counts': {'x': 5}})
    assert m.observations('x') == 5
