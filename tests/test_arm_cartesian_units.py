"""T:104 takes millimetres; everything upstream of it is in metres.

This is the bug that made every grasp close on empty air. pick_place_node
computes its target in `arm_base` metres (tf2, MoveIt and the drop_* params
are all SI) and used to hand those numbers to the firmware unchanged, so a
grasp at (0.478, 0.003, 0.102) m was sent as 0.478 mm — half a millimetre from
the arm's own origin.

Measured on the real arm 2026-08-19: T:105 feedback in the init pose reports
x=352.3 y=-2.2 z=199.3, i.e. millimetres, and one metre-valued T:104 wedged the
ESP32 hard enough that only a power cycle of the arm board brought it back.
That is why the conversion has a floor guard as well as a scale factor.

_cartesian is exercised unbound so the arithmetic stays testable without rclpy
or a serial port.
"""

import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        pass

    def warn(self, msg, **kw):
        pass


def _arm(speed=0):
    """A stand-in for the node: records what would go out over serial."""
    sent = []
    logger = _Logger()
    return types.SimpleNamespace(
        sent=sent,
        logger=logger,
        arm_speed=speed,
        movement_settle_time=0.0,
        _raw=sent.append,
        get_logger=lambda: logger,
    )


def _cartesian(node, x, y, z, **kw):
    from home_robot.nodes.pick_place_node import PickPlaceNode
    # settle=0 everywhere: the real method sleeps, and no test wants that.
    kw.setdefault('settle', 0.0)
    return PickPlaceNode._cartesian(node, x, y, z, **kw)


# ── the reported case ─────────────────────────────────────────────────────────

def test_the_grasp_target_that_used_to_close_on_air_goes_out_in_millimetres():
    """The live target from 2026-08-19: 0.478 m forward became 0.478 mm."""
    node = _arm()
    _cartesian(node, 0.478, 0.003, 0.102)
    cmd = node.sent[0]
    assert cmd['T'] == 104
    assert math.isclose(cmd['x'], 478.0)
    assert math.isclose(cmd['y'], 3.0)
    assert math.isclose(cmd['z'], 102.0)


def test_the_drop_pose_defaults_are_converted_too():
    """drop_x/drop_y/drop_z are metres (0.15, -0.15, 0.10). Unconverted they
    are a fraction of a millimetre from the origin — the place-half of
    pick-and-place hung the firmware for exactly the same reason the grasp
    half did."""
    node = _arm()
    _cartesian(node, 0.15, -0.15, 0.10)
    cmd = node.sent[0]
    assert (cmd['x'], cmd['y'], cmd['z']) == (150.0, -150.0, 100.0)


# ── the guard ────────────────────────────────────────────────────────────────

def test_a_target_inside_the_base_is_refused_not_forwarded():
    """‼️ Recovering a wedged ESP32 costs a physical power cycle of the arm
    board — no amount of USB re-enumeration does it (verified 2026-08-19). A
    degenerate target must never reach the firmware."""
    node = _arm()
    _cartesian(node, 0.0004, 0.0, 0.0001)
    assert node.sent == []
    assert node.logger.errors, 'a refusal should say why'


def test_the_guard_measures_radius_not_a_single_axis():
    """A target can be tiny on every axis yet fine on one — and vice versa.
    (0, 0, 0.06) m is 60 mm straight up: outside the floor, so allowed."""
    node = _arm()
    _cartesian(node, 0.0, 0.0, 0.06)
    assert node.sent, 'a reachable point 60 mm up must not be refused'


def test_a_normal_reach_is_never_caught_by_the_guard():
    """Anything the max_reach check already let through is far outside the
    guard's radius — the guard exists for unit slips, not for real targets."""
    node = _arm()
    for x in (0.20, 0.35, 0.52):
        node.sent.clear()
        _cartesian(node, x, 0.0, 0.10)
        assert node.sent, f'{x} m should be sent'


# ── the rest of the payload ──────────────────────────────────────────────────

def test_roll_stays_in_radians():
    """Only x/y/z are lengths. 'r' is an angle and must NOT be scaled."""
    node = _arm()
    _cartesian(node, 0.30, 0.0, 0.10, roll=1.2)
    assert math.isclose(node.sent[0]['r'], 1.2)


def test_speed_is_passed_through_untouched():
    node = _arm(speed=250)
    _cartesian(node, 0.30, 0.0, 0.10)
    assert node.sent[0]['spd'] == 250
