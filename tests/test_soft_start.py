"""Tests for the drive soft-start ramp (roomba_driver._ramp_step).

The wheel ramp used one rate for everything. A single rate can be gentle off
the mark or quick to cruising speed, not both: 2026-07-28 raised it 600 -> 1200
because 600 "felt sluggish", which brought back the lurch on start. Soft start
splits the two — a gentler rate only while pulling away from a standstill.

The property that actually matters here is the one that is easy to break by
accident: **braking must never get slower**. A softer stop is a longer stop,
and this robot stops for bumpers and e-stops. So every non-accelerating case is
pinned to the full rate.

Robot-free: _ramp_step is pure arithmetic over three parameters, exercised on a
bare object rather than a live node, so no serial port or rclpy context.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_soft_start.py -q
"""
import importlib.util
import os
import re
import types

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = f'{PKG}/home_robot/nodes/roomba_driver.py'

DT = 0.05           # the driver's 20 Hz control tick
MAX_ACCEL = 1200.0
SOFT_ACCEL = 400.0
SOFT_SPEED = 120.0


def _ramp():
    """_ramp_step bound to a stand-in carrying just the three parameters.

    Importing roomba_driver would pull in pyserial and the Create OI wrapper,
    so lift the method out of the source instead — it only touches self.* for
    those parameters.
    """
    src = open(DRIVER).read()
    match = re.search(r'\n    def _ramp_step\(self.*?\n(?=    def )', src, re.S)
    assert match, '_ramp_step not found — was it renamed?'
    ns = {}
    exec('class _Stub:\n' + match.group(0), ns)
    stub = ns['_Stub']()
    stub.max_accel = MAX_ACCEL
    stub.soft_start_accel = SOFT_ACCEL
    stub.soft_start_speed = SOFT_SPEED
    return stub


@pytest.fixture
def ramp():
    return _ramp()


SOFT = SOFT_ACCEL * DT
FULL = MAX_ACCEL * DT


# ── Soft start: accelerating from rest ────────────────────────────────

def test_pulling_away_from_standstill_is_gentle(ramp):
    assert ramp._ramp_step(0.0, 500.0, DT) == pytest.approx(SOFT)


def test_still_gentle_just_below_the_threshold(ramp):
    assert ramp._ramp_step(SOFT_SPEED - 1, 500.0, DT) == pytest.approx(SOFT)


def test_full_rate_once_moving(ramp):
    """Past the threshold the robot must not feel sluggish — that was the 600
    mm/s² complaint that raised the rate in the first place."""
    assert ramp._ramp_step(SOFT_SPEED + 1, 500.0, DT) == pytest.approx(FULL)
    assert ramp._ramp_step(300.0, 500.0, DT) == pytest.approx(FULL)


def test_soft_start_applies_in_reverse_too(ramp):
    assert ramp._ramp_step(0.0, -500.0, DT) == pytest.approx(SOFT)
    assert ramp._ramp_step(-50.0, -500.0, DT) == pytest.approx(SOFT)


def test_turning_from_rest_is_gentle(ramp):
    """A spin starts both wheels from zero in opposite directions, so the same
    soft start covers 'starts turning too abruptly'."""
    assert ramp._ramp_step(0.0, 294.0, DT) == pytest.approx(SOFT)    # left wheel
    assert ramp._ramp_step(0.0, -294.0, DT) == pytest.approx(SOFT)   # right wheel


# ── Braking must never be softened ────────────────────────────────────

def test_braking_from_low_speed_gets_full_rate(ramp):
    """The dangerous case: slow and stopping is exactly where a soft rate would
    silently lengthen the stopping distance."""
    assert ramp._ramp_step(50.0, 0.0, DT) == pytest.approx(FULL)
    assert ramp._ramp_step(-50.0, 0.0, DT) == pytest.approx(FULL)


def test_braking_from_high_speed_gets_full_rate(ramp):
    assert ramp._ramp_step(500.0, 0.0, DT) == pytest.approx(FULL)


def test_easing_off_without_stopping_gets_full_rate(ramp):
    assert ramp._ramp_step(500.0, 200.0, DT) == pytest.approx(FULL)


def test_reversing_direction_gets_full_rate(ramp):
    """Crossing through zero is a brake first; it must not be slowed down even
    though |target| > |current| and |current| is small."""
    assert ramp._ramp_step(50.0, -500.0, DT) == pytest.approx(FULL)
    assert ramp._ramp_step(-50.0, 500.0, DT) == pytest.approx(FULL)


def test_holding_speed_gets_full_rate(ramp):
    assert ramp._ramp_step(300.0, 300.0, DT) == pytest.approx(FULL)


# ── Resulting feel ────────────────────────────────────────────────────

def _time_to(ramp, target, from_speed=0.0):
    """Seconds of 20 Hz ticks to reach target from a standstill."""
    cur, t = from_speed, 0.0
    for _ in range(400):
        step = ramp._ramp_step(cur, target, DT)
        cur += max(-step, min(step, target - cur))
        t += DT
        if abs(cur - target) < 1e-6:
            return t
    raise AssertionError('never reached target')


def test_departure_is_softer_but_top_speed_is_not_delayed_much(ramp):
    """Both halves of the request: softer start, still not sluggish overall."""
    soft = _time_to(ramp, 500.0)
    ramp.soft_start_accel = MAX_ACCEL       # what it was before soft start
    hard = _time_to(ramp, 500.0)
    assert soft > hard, 'soft start must actually slow the departure'
    # 0.3 s at 400 mm/s² covers the first 120 mm/s; the rest is unchanged, so
    # full speed arrives only ~0.2 s later than before.
    assert soft - hard < 0.25, f'soft start delayed full speed by {soft - hard:.2f}s'


def test_stopping_time_is_unchanged_by_soft_start(ramp):
    """Regression guard on the safety property, end to end."""
    def stop_time(node):
        cur, t = 500.0, 0.0
        while abs(cur) > 1e-6:
            step = node._ramp_step(cur, 0.0, DT)
            cur += max(-step, min(step, 0.0 - cur))
            t += DT
        return t

    with_soft = stop_time(ramp)
    ramp.soft_start_accel = MAX_ACCEL
    without = stop_time(ramp)
    assert with_soft == pytest.approx(without)


# ── Tunability ────────────────────────────────────────────────────────

def test_parameters_are_live_tunable():
    """Retuning the feel happens with the controller in hand, not by relaunch."""
    src = open(DRIVER).read()
    callback = src[src.index('def _on_set_parameters'):]
    callback = callback[:callback.index('\n    def ')]
    for name in ('soft_start_accel', 'soft_start_speed'):
        assert f"p.name == '{name}'" in callback, f'{name} not handled live'
        assert f'self.declare_parameter(\'{name}\'' in src


def test_zero_threshold_disables_soft_start(ramp):
    """Escape hatch: soft_start_speed 0 restores the old single-rate behaviour."""
    ramp.soft_start_speed = 0.0
    assert ramp._ramp_step(0.0, 500.0, DT) == pytest.approx(FULL)
