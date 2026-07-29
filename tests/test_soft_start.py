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


# ── Turn-rate slew limit (_ramp_angular) ──────────────────────────────

ANG_ACCEL = 2.5     # rad/s^2
TOP_YAW = 1.20      # what teleop_twist_joy_ps5.yaml now commands


def _angular(monotonic):
    """_ramp_angular bound to a stub, with an injected clock.

    Its dt comes from time.monotonic() between cmd_vel messages, so the module
    the method executes in gets a fake `time` — no sleeping in tests.
    """
    src = open(DRIVER).read()
    match = re.search(r'\n    def _ramp_angular\(self.*?\n(?=    def )', src, re.S)
    assert match, '_ramp_angular not found — was it renamed?'
    ns = {'time': types.SimpleNamespace(monotonic=monotonic)}
    exec('class _Stub:\n' + match.group(0), ns)
    stub = ns['_Stub']()
    stub.max_angular_accel = ANG_ACCEL
    stub._cur_angular = 0.0
    stub._last_angular_time = monotonic()
    stub._stopped = False
    return stub


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def tick(self, dt=DT):
        self.t += dt


@pytest.fixture
def yaw():
    clock = _Clock()
    stub = _angular(clock)
    stub._clock = clock
    return stub


def _hold_yaw(yaw, requested, seconds):
    """Feed one cmd_vel per 20 Hz tick for `seconds`; return the eased rate."""
    out = yaw._cur_angular
    for _ in range(int(round(seconds / DT))):
        yaw._clock.tick()
        out = yaw._ramp_angular(requested)
    return out


def test_a_turn_does_not_snap_to_full_rate(yaw):
    """One command at full yaw must not become full yaw on the same tick."""
    yaw._clock.tick()
    first = yaw._ramp_angular(TOP_YAW)
    assert first == pytest.approx(ANG_ACCEL * DT)
    assert first < TOP_YAW


def test_turn_reaches_the_commanded_rate(yaw):
    """Gentler must not mean never gets there."""
    assert _hold_yaw(yaw, TOP_YAW, 1.0) == pytest.approx(TOP_YAW)


def test_time_to_full_turn_rate_is_about_half_a_second(yaw):
    """1.20 rad/s at 2.5 rad/s² ≈ 0.48 s — eased, not sluggish."""
    assert _hold_yaw(yaw, TOP_YAW, 0.30) < TOP_YAW
    assert _hold_yaw(yaw, TOP_YAW, 0.30) == pytest.approx(TOP_YAW, abs=0.05)


def test_easing_works_in_both_directions(yaw):
    assert _hold_yaw(yaw, -TOP_YAW, 1.0) == pytest.approx(-TOP_YAW)


def test_reversing_a_turn_passes_through_zero_gradually(yaw):
    _hold_yaw(yaw, TOP_YAW, 1.0)
    # Flipping the stick must not slam from +1.2 to -1.2 in one tick.
    yaw._clock.tick()
    after = yaw._ramp_angular(-TOP_YAW)
    assert after == pytest.approx(TOP_YAW - ANG_ACCEL * DT)
    assert after > 0, 'a reversal should ease down, not jump across zero'


def test_releasing_the_stick_eases_the_turn_out(yaw):
    _hold_yaw(yaw, TOP_YAW, 1.0)
    assert _hold_yaw(yaw, 0.0, 0.1) < TOP_YAW
    assert _hold_yaw(yaw, 0.0, 1.0) == pytest.approx(0.0)


def test_a_gap_longer_than_the_watchdog_starts_fresh(yaw):
    """The wheels were zeroed during the gap, so resuming from the old rate
    would fight the new command rather than easing it in."""
    _hold_yaw(yaw, TOP_YAW, 1.0)
    yaw._clock.tick(0.5)                      # > 0.25 s watchdog timeout
    resumed = yaw._ramp_angular(TOP_YAW)
    assert resumed == pytest.approx(0.0), 'stale rate must be dropped'


def test_short_gaps_do_not_reset(yaw):
    """Normal 20 Hz jitter must not keep restarting the ramp."""
    _hold_yaw(yaw, TOP_YAW, 0.2)
    before = yaw._cur_angular
    yaw._clock.tick(0.15)                     # under the timeout
    assert yaw._ramp_angular(TOP_YAW) > before


def test_zero_disables_the_limiter(yaw):
    """Escape hatch, matching soft_start_speed: 0."""
    yaw.max_angular_accel = 0.0
    yaw._clock.tick()
    assert yaw._ramp_angular(TOP_YAW) == pytest.approx(TOP_YAW)


def test_teleop_yaw_scale_is_no_longer_flat_out():
    """The other half of "not so fast": the commanded ceiling came down too."""
    import re as _re
    cfg = open(f'{PKG}/config/teleop_twist_joy_ps5.yaml').read()
    block = cfg[cfg.index('scale_angular:'):]
    yaw_value = float(_re.search(r'yaw:\s*([0-9.]+)', block).group(1))
    assert 0.35 < yaw_value <= 1.5, f'yaw scale {yaw_value} outside the sane band'
    # ‼️ the 879 cannot rotate below ~0.31 rad/s at all.
    assert yaw_value > 0.35


# ── Tunability ────────────────────────────────────────────────────────

def test_parameters_are_live_tunable():
    """Retuning the feel happens with the controller in hand, not by relaunch."""
    src = open(DRIVER).read()
    callback = src[src.index('def _on_set_parameters'):]
    callback = callback[:callback.index('\n    def ')]
    for name in ('soft_start_accel', 'soft_start_speed', 'max_angular_accel'):
        assert f"p.name == '{name}'" in callback, f'{name} not handled live'
        assert f'self.declare_parameter(\'{name}\'' in src


def test_zero_threshold_disables_soft_start(ramp):
    """Escape hatch: soft_start_speed 0 restores the old single-rate behaviour."""
    ramp.soft_start_speed = 0.0
    assert ramp._ramp_step(0.0, 500.0, DT) == pytest.approx(FULL)
