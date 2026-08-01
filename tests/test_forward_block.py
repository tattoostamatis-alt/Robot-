"""Tests for the bumper/cliff forward block (roomba_driver._is_forward/_strip_forward).

The guard used to read `target_left > 0 and target_right > 0`, which is not the
question it meant to ask. The robot's forward speed is the AVERAGE of the two
wheels; requiring BOTH to be positive misses every arc. With this chassis'
wheel_base, 0.10 m/s commanded with 1.0 rad/s of turn comes out left=-9,
right=+216 — the full 103 mm/s forward, one wheel negative — so the robot drove
on at commanded speed with the bumper pressed or a cliff under the nose. In
`oi_mode='full'` (the default since 2026-07-31) the Roomba's own cliff abort is
switched off, so this block is the only thing between the robot and a step.

The arc cases below are therefore the point of this file; the straight-line ones
only pin down what already worked.

Robot-free: both methods are pure arithmetic over the wheel pair, so they run on
a bare class with no serial port and no rclpy context.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_forward_block.py -q
"""
import os
import re

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = f'{PKG}/home_robot/nodes/roomba_driver.py'

# The 879's measured calibration, so the cases below are the real commands.
WHEEL_BASE = 0.218405
RIGHT_TRIM = 1.033
EPS = 20.0          # _FORWARD_EPS_MM_S


def _wheels(linear_ms, angular):
    """cmd_vel -> (left, right) target speeds, mirroring _cmd_vel_cb."""
    linear = linear_ms * 1000
    right = (linear + angular * WHEEL_BASE * 500) * RIGHT_TRIM
    left = linear - angular * WHEEL_BASE * 500
    return left, right


@pytest.fixture
def guard():
    """_is_forward + _strip_forward lifted onto a bare class.

    Importing roomba_driver would pull in pyserial and the Create OI wrapper,
    so extract the methods from source instead — neither touches self beyond
    the epsilon constant.
    """
    src = open(DRIVER).read()
    body = ''
    for name in ('_is_forward', '_strip_forward'):
        match = re.search(rf'\n    @(?:class|static)method\n    def {name}\(.*?\n(?=    [@\w])',
                          src, re.S)
        assert match, f'{name} not found — was it renamed?'
        body += match.group(0)
    eps = re.search(r'_FORWARD_EPS_MM_S = ([\d.]+)', src)
    assert eps, '_FORWARD_EPS_MM_S not found'
    assert float(eps.group(1)) == EPS, 'epsilon changed — update this test'
    ns = {}
    exec(f'class _Stub:\n    _FORWARD_EPS_MM_S = {EPS}\n' + body, ns)
    return ns['_Stub']()


# ── The regression: arcs must count as forward ────────────────────────

@pytest.mark.parametrize('linear,angular', [
    (0.10, 0.92),    # exactly where the left wheel crosses zero
    (0.10, 1.00),
    (0.10, 1.20),    # the teleop yaw ceiling
    (0.05, 0.60),
    (0.10, -1.00),   # and the same arc the other way
    (0.15, -1.50),
])
def test_forward_arcs_are_blocked_even_with_one_wheel_negative(guard, linear, angular):
    """‼️ The whole reason this file exists.

    Every one of these has a negative wheel, so the old per-wheel test let it
    straight through — at full commanded forward speed, into the obstacle the
    bumper was already reporting.
    """
    left, right = _wheels(linear, angular)
    assert min(left, right) < 0, 'case is pointless unless a wheel is negative'
    assert (left + right) / 2 > 50, 'case is pointless unless it really moves forward'
    assert guard._is_forward(left, right), f'arc ({linear}, {angular}) not seen as forward'


def test_the_old_per_wheel_test_would_have_missed_these(guard):
    """Nails the specific defect, so a revert cannot pass silently."""
    left, right = _wheels(0.10, 1.0)
    assert not (left > 0 and right > 0), 'the old condition'
    assert guard._is_forward(left, right), 'the fixed condition'


# ── Straight lines: what already worked, kept working ─────────────────

def test_straight_forward_is_blocked(guard):
    assert guard._is_forward(*_wheels(0.10, 0.0))
    assert guard._is_forward(*_wheels(0.03, 0.0))     # the mapping crawl


def test_straight_reverse_is_never_blocked(guard):
    """Reverse is how the robot backs off a bumper or away from an edge."""
    assert not guard._is_forward(*_wheels(-0.10, 0.0))
    assert not guard._is_forward(*_wheels(-0.05, 0.0))


def test_reverse_arcs_are_never_blocked(guard):
    """Backing away while turning must stay available too."""
    assert not guard._is_forward(*_wheels(-0.10, 1.0))
    assert not guard._is_forward(*_wheels(-0.10, -1.0))


def test_stationary_is_not_forward(guard):
    assert not guard._is_forward(0.0, 0.0)


# ── Turning on the spot must survive the right_trim asymmetry ─────────

@pytest.mark.parametrize('angular', [0.35, 0.6, 1.0, 1.2, -0.35, -1.2])
def test_spinning_in_place_is_not_treated_as_forward(guard, angular):
    """right_trim multiplies the right wheel only, so a pure spin averages a few
    mm/s positive. Without the epsilon the guard would block turning — the one
    manoeuvre that gets the robot off whatever it hit."""
    left, right = _wheels(0.0, angular)
    assert 0 < abs((left + right) / 2) < EPS, 'trim artefact should be small but real'
    assert not guard._is_forward(left, right)


def test_epsilon_is_far_below_any_deliberate_speed(guard):
    """The escape hatch must not become a hole: 20 mm/s is under the slowest
    speed anything actually commands (30 mm/s while mapping)."""
    assert guard._is_forward(*_wheels(0.03, 0.0))


# ── Stripping the forward component keeps the turn ────────────────────

def test_strip_removes_all_forward_motion(guard):
    for linear, angular in [(0.10, 0.0), (0.10, 1.0), (0.15, 1.2), (0.03, 0.5)]:
        left, right = guard._strip_forward(*_wheels(linear, angular))
        assert (left + right) / 2 == pytest.approx(0.0, abs=1e-9)


def test_strip_preserves_the_turn_exactly(guard):
    """Blocked must mean "cannot go forward", not "cannot move" — otherwise a
    robot arcing into a doorway frame is pinned there with no way to turn out."""
    for angular in (0.5, 1.0, 1.2, -1.0):
        left, right = _wheels(0.10, angular)
        sl, sr = guard._strip_forward(left, right)
        assert (sr - sl) == pytest.approx(right - left), 'differential changed'


def test_a_blocked_arc_still_turns_the_right_way(guard):
    """Sign check: a left turn must stay a left turn after stripping."""
    sl, sr = guard._strip_forward(*_wheels(0.10, 1.0))    # +z = left/CCW
    assert sr > sl, 'left turn should keep the right wheel ahead'
    sl, sr = guard._strip_forward(*_wheels(0.10, -1.0))
    assert sl > sr, 'right turn should keep the left wheel ahead'


def test_blocked_straight_line_barely_moves(guard):
    """A pure forward command has no turn to preserve, so what is left over is
    only the trim asymmetry — it must not amount to a usable speed."""
    left, right = guard._strip_forward(*_wheels(0.10, 0.0))
    assert abs(left) < 5 and abs(right) < 5


# ── End to end through the real _motor_control guard ──────────────────

@pytest.fixture
def motor():
    """The real _motor_control, on a stub with the ramp made transparent.

    Testing _is_forward alone is not enough: the original defect was an
    open-coded condition inside _motor_control, and helpers can sit there
    correct and unused (verified — reinstating the old condition leaves every
    unit test on this page green). So drive the actual method and read what it
    hands to drive_direct.

    max_accel is set absurdly high so the per-wheel ramp reaches the target in
    one tick and what comes out is exactly what the guards let through.
    """
    src = open(DRIVER).read()
    body = ''
    for name in ('_is_forward', '_strip_forward', '_ramp_step', '_drive',
                 '_on_write_failure', '_sensors_fresh', '_motor_control'):
        match = re.search(
            rf'\n    (?:@(?:class|static)method\n    )?def {name}\(.*?\n(?=    (?:def |@))',
            src, re.S)
        assert match, f'{name} not found — was it renamed?'
        body += match.group(0)

    sent = []
    ns = {'time': __import__('time')}
    exec(f'class _Stub:\n    _FORWARD_EPS_MM_S = {EPS}\n' + body, ns)
    stub = ns['_Stub']()
    stub.bot = type('B', (), {'drive_direct': lambda _s, r, l: sent.append((l, r))})()
    stub.get_logger = lambda: type('L', (), {
        'warn': lambda *a, **k: None, 'info': lambda *a, **k: None})()
    stub.max_accel = stub.soft_start_accel = 1e9      # ramp out of the way
    stub.soft_start_speed = 0.0
    stub._estop = stub._docking = stub._idle = False
    stub._stopped = False
    stub._cur_left = stub._cur_right = 0.0
    stub._last_control_time = 0.0
    stub._last_cmd_time = float('inf')                # never stale
    stub.bump_stop = stub.cliff_stop = stub.wheel_drop_stop = True
    stub.bump_block_s = stub.cliff_block_s = 1.0
    stub._last_bump_time = stub._last_cliff_time = -1e9   # nothing triggered
    stub._bump_warned = stub._cliff_warned = False
    stub._wheel_drop = stub._wheel_drop_warned = False
    # Serial-link state (2026-08-01): every wheel write now goes through
    # _drive(), and stale bumper/cliff readings block forward motion. A healthy
    # link is the default here so the guard under test stays the bumper/cliff.
    stub.sensor_stale_s = 1.0
    stub._last_ok_time = __import__('time').monotonic()
    stub._stale_warned = False
    stub._write_failures = 0
    stub._last_write_ok = 0.0
    stub._cur_angular = 0.0
    stub._connect_oi = lambda: None
    stub.sent = sent
    return stub


def _tick(motor, left, right):
    """One control tick with these wheel targets; returns what reached the motors."""
    motor._target_left, motor._target_right = left, right
    motor._last_cmd_time = __import__('time').monotonic()
    motor._motor_control()
    return motor.sent[-1]


def test_motor_control_blocks_a_forward_arc_on_the_bumper(motor):
    """‼️ The end-to-end regression. 0.10 m/s + 1.0 rad/s, bumper pressed."""
    left, right = _wheels(0.10, 1.0)
    motor._last_bump_time = __import__('time').monotonic()
    out_left, out_right = _tick(motor, left, right)
    assert (out_left + out_right) / 2 <= EPS, \
        f'robot still moving forward at {(out_left + out_right) / 2:.0f} mm/s on the bumper'


def test_motor_control_blocks_a_forward_arc_on_a_cliff(motor):
    """Same, for the guard that stops the robot going down a step."""
    left, right = _wheels(0.10, 1.2)
    motor._last_cliff_time = __import__('time').monotonic()
    out_left, out_right = _tick(motor, left, right)
    assert (out_left + out_right) / 2 <= EPS


def test_motor_control_lets_the_blocked_robot_turn_away(motor):
    """Blocked must not mean immobilised, or the robot is pinned on the obstacle."""
    left, right = _wheels(0.10, 1.0)
    motor._last_bump_time = __import__('time').monotonic()
    out_left, out_right = _tick(motor, left, right)
    assert abs(out_right - out_left) == pytest.approx(abs(right - left), abs=2)


def test_motor_control_lets_reverse_through_on_the_bumper(motor):
    """Backing off is the whole recovery — it must survive both guards."""
    left, right = _wheels(-0.10, 0.0)
    motor._last_bump_time = __import__('time').monotonic()
    out_left, out_right = _tick(motor, left, right)
    assert (out_left + out_right) / 2 < -50


def test_motor_control_lets_a_spin_through_on_the_bumper(motor):
    """Turning on the spot must not be swallowed by the trim artefact."""
    left, right = _wheels(0.0, 1.0)
    motor._last_bump_time = __import__('time').monotonic()
    out_left, out_right = _tick(motor, left, right)
    assert abs(out_right - out_left) == pytest.approx(abs(right - left), abs=2)


def test_motor_control_passes_straight_forward_when_nothing_is_triggered(motor):
    """Sanity: the guards must not be blocking all the time."""
    left, right = _wheels(0.10, 0.0)
    out_left, out_right = _tick(motor, left, right)
    assert (out_left + out_right) / 2 > 50


# ── The guards are actually wired to these methods ────────────────────

def test_every_forward_guard_uses_the_shared_test():
    """All of them must go through _is_forward — the bug was one open-coded
    condition duplicated in two places, and a partial fix is the likely relapse.

    Three guards since 2026-08-01: bumper, cliff, and stale sensor readings.
    """
    src = open(DRIVER).read()
    control = src[src.index('def _motor_control'):]
    control = control[:control.index('\n    def _check_idle')]
    assert control.count('self._is_forward(target_left, target_right)') == 3
    assert control.count('self._strip_forward(target_left, target_right)') == 3
    # ...and none of them open-codes the per-wheel test again. Checked against
    # the method body only: the docstrings quote the old condition on purpose.
    assert 'target_left > 0 and target_right > 0' not in control, \
        'the old per-wheel test is back'


def test_wheel_drop_still_stops_everything():
    """Unlike bump/cliff, a robot off the floor gets no motion at all — make
    sure it was not swept into the forward-only treatment."""
    src = open(DRIVER).read()
    control = src[src.index('def _motor_control'):]
    drop = control[control.index('if self.wheel_drop_stop'):]
    drop = drop[:drop.index('elif self._wheel_drop_warned')]
    assert 'target_left = target_right = 0.0' in drop
    assert '_strip_forward' not in drop
