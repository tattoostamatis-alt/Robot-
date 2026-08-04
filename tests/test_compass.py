"""Unit tests for the map-referenced compass (home_robot/compass.py)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.compass import (  # noqa: E402
    CARDINALS_EL, bearing_from_yaw, cardinal_el, cardinal_full_el,
    normalize_deg, offset_from_known_bearing, yaw_for_bearing,
)

D = math.radians


# ── normalize_deg ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('given,expected', [
    (0, 0), (359, 359), (360, 0), (361, 1), (-1, 359), (-90, 270), (720, 0),
])
def test_normalize(given, expected):
    assert normalize_deg(given) == pytest.approx(expected)


# ── bearing_from_yaw ──────────────────────────────────────────────────────────

def test_facing_the_calibrated_direction_reads_north():
    assert bearing_from_yaw(D(37), north_offset_rad=D(37)) == pytest.approx(0)


def test_turning_left_goes_west_not_east():
    """‼️ The sign trap. ROS yaw grows counter-clockwise, compass bearings grow
    clockwise, so a LEFT turn from north must read west (270), not east (90)."""
    assert bearing_from_yaw(D(90), north_offset_rad=0.0) == pytest.approx(270)


def test_turning_right_goes_east():
    assert bearing_from_yaw(D(-90), north_offset_rad=0.0) == pytest.approx(90)


def test_backwards_is_south():
    assert bearing_from_yaw(D(180), north_offset_rad=0.0) == pytest.approx(180)


def test_offset_shifts_every_reading():
    # With north at map-yaw 90, facing along +x (yaw 0) is due east.
    assert bearing_from_yaw(0.0, north_offset_rad=D(90)) == pytest.approx(90)


def test_result_is_always_in_range():
    for yaw in range(-720, 721, 17):
        b = bearing_from_yaw(D(yaw), north_offset_rad=D(33))
        assert 0.0 <= b < 360.0


# ── yaw_for_bearing round trip ────────────────────────────────────────────────

@pytest.mark.parametrize('bearing', [0, 45, 90, 180, 270, 359])
def test_round_trip(bearing):
    off = D(23)
    yaw = yaw_for_bearing(bearing, north_offset_rad=off)
    assert bearing_from_yaw(yaw, north_offset_rad=off) == pytest.approx(bearing, abs=1e-6)


def test_yaw_for_bearing_stays_in_pi_range():
    for bearing in range(0, 360, 13):
        yaw = yaw_for_bearing(bearing, north_offset_rad=D(200))
        assert -math.pi - 1e-9 <= yaw <= math.pi + 1e-9


# ── calibration ───────────────────────────────────────────────────────────────

def test_calibrating_while_facing_north_stores_the_current_yaw():
    """The common case: point the robot north, press calibrate."""
    assert offset_from_known_bearing(D(41), 0) == pytest.approx(D(41))


def test_calibrating_while_facing_a_known_other_bearing():
    # Standing due east (90) at map yaw 0 means north is at map yaw +90.
    off = offset_from_known_bearing(0.0, 90)
    assert off == pytest.approx(D(90))
    assert bearing_from_yaw(0.0, off) == pytest.approx(90)


def test_calibration_then_reading_is_consistent():
    yaw_now = D(-118)
    off = offset_from_known_bearing(yaw_now, 0)
    assert bearing_from_yaw(yaw_now, off) == pytest.approx(0, abs=1e-9)


def test_offset_is_wrapped():
    off = offset_from_known_bearing(D(170), 270)
    assert -math.pi - 1e-9 <= off <= math.pi + 1e-9


# ── cardinals ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('bearing,short', [
    (0, 'Β'), (45, 'ΒΑ'), (90, 'Α'), (135, 'ΝΑ'),
    (180, 'Ν'), (225, 'ΝΔ'), (270, 'Δ'), (315, 'ΒΔ'),
])
def test_cardinal_points(bearing, short):
    assert cardinal_el(bearing) == short


def test_cardinal_rounds_to_the_nearest_point():
    assert cardinal_el(20) == 'Β'        # nearer 0 than 45
    assert cardinal_el(25) == 'ΒΑ'       # nearer 45
    assert cardinal_el(350) == 'Β'       # wraps back to north


def test_cardinal_wraps_past_360():
    assert cardinal_el(361) == 'Β'
    assert cardinal_el(-90) == 'Δ'


def test_every_cardinal_has_a_full_name():
    for i, short in enumerate(CARDINALS_EL):
        full = cardinal_full_el(i * 45)
        assert full and full != short


def test_full_names_are_greek_and_specific():
    assert cardinal_full_el(0) == 'Βορράς'
    assert cardinal_full_el(180) == 'Νότος'
    assert cardinal_full_el(225) == 'Νοτιοδυτικά'


def test_north_and_south_are_opposite():
    """The user asked for north AND south; they must be 180 apart."""
    off = D(17)
    north_yaw = yaw_for_bearing(0, off)
    south_yaw = yaw_for_bearing(180, off)
    delta = abs(math.degrees(math.atan2(math.sin(north_yaw - south_yaw),
                                        math.cos(north_yaw - south_yaw))))
    assert delta == pytest.approx(180, abs=1e-6)
