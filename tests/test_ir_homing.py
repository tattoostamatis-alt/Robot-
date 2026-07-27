"""Tests for the closed-loop IR dock homing (see home_robot/ir_homing.py).

Robot-free: step() is pure, so every approach geometry is expressible as a
sequence of receiver bytes. The cases that matter are the ones measured on
hardware 2026-07-26, when the built-in Seek Dock failed — 172/172/172 before
the attempt (centred), 161/0/0 after (force field only, both directional
receivers blind), which is the state it never escaped.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_ir_homing.py -q
"""
import os

import pytest

from home_robot.ir_homing import (
    DOCK_CODES, HOMING, LOST, SEARCHING, SEATED, IrHoming, decode,
)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = f'{PKG}/home_robot/nodes/roomba_driver.py'

CENTRE = 172        # Red + Green
GREEN = 164
RED = 168
FORCE_FIELD = 161
BOTH_PLUS_FF = 173
VIRTUAL_WALL = 162  # NOT the dock


# ── decode ────────────────────────────────────────────────────────────
def test_decodes_the_measured_centre_reading():
    assert decode(CENTRE) == (False, True, True)


def test_decodes_force_field_alone():
    assert decode(FORCE_FIELD) == (True, False, False)


def test_decodes_all_three():
    assert decode(BOTH_PLUS_FF) == (True, True, True)


@pytest.mark.parametrize('code', sorted(DOCK_CODES))
def test_every_dock_code_decodes(code):
    assert decode(code) is not None


def test_virtual_wall_is_not_the_dock():
    """162 shares the 0xA0 prefix; masking instead of matching would have let
    it through as a dock sighting with no buoys, i.e. a phantom force field."""
    assert decode(VIRTUAL_WALL) is None


@pytest.mark.parametrize('byte', [None, 0, 1, 255])
def test_non_dock_bytes_decode_to_none(byte):
    assert decode(byte) is None


# ── steering ──────────────────────────────────────────────────────────
def test_centred_drives_straight():
    h = IrHoming()
    left, right, status = h.step(CENTRE, CENTRE, CENTRE, moved=True, now=0.0)
    assert status == HOMING
    assert left == right > 0


def test_beam_on_the_left_receiver_steers_left():
    """Directional geometry, independent of buoy colour: only the left
    receiver sees the station, so the station is to our left."""
    h = IrHoming()
    left, right, status = h.step(None, GREEN, None, moved=True, now=0.0)
    assert status == HOMING
    assert left < right
    assert left > 0            # an arc, never a pivot — see the rotation floor


def test_beam_on_the_right_receiver_steers_right():
    h = IrHoming()
    left, right, status = h.step(None, None, GREEN, moved=True, now=0.0)
    assert status == HOMING
    assert right < left
    assert right > 0


def test_directional_receivers_outrank_buoy_colour():
    """Both buoy colours would say one thing; the receiver that actually sees
    the beam says another. Geometry wins, so swap_buoys cannot break this."""
    a = IrHoming(swap_buoys=False).step(None, GREEN, None, moved=True, now=0.0)
    b = IrHoming(swap_buoys=True).step(None, GREEN, None, moved=True, now=0.0)
    assert a == b
    assert a[0] < a[1]


def test_single_buoy_on_omni_only_steers_hard():
    h = IrHoming()
    left, right, _ = h.step(GREEN, None, None, moved=True, now=0.0)
    assert abs(left - right) == 2 * h.hard_steer_mm_s


def test_swap_buoys_reverses_the_colour_rule():
    plain = IrHoming(swap_buoys=False).step(GREEN, None, None, moved=True, now=0.0)
    swapped = IrHoming(swap_buoys=True).step(GREEN, None, None, moved=True, now=0.0)
    assert plain[0] < plain[1]
    assert swapped[0] > swapped[1]


def test_red_and_green_steer_opposite_ways():
    g = IrHoming().step(GREEN, None, None, moved=True, now=0.0)
    r = IrHoming().step(RED, None, None, moved=True, now=0.0)
    assert (g[0] - g[1]) == -(r[0] - r[1])


# ── the failure the built-in seek got stuck in ────────────────────────
def test_force_field_only_backs_out():
    """The measured dead end: omni 161, both directional receivers 0. Grinding
    forward here is what the built-in behaviour did for 120 s."""
    h = IrHoming()
    left, right, status = h.step(FORCE_FIELD, None, None, moved=True, now=0.0)
    assert status == SEARCHING
    assert left < 0 and right < 0


def test_backup_runs_to_completion_before_re_deciding():
    h = IrHoming()
    h.step(FORCE_FIELD, None, None, moved=True, now=0.0)
    left, right, status = h.step(FORCE_FIELD, None, None, moved=True, now=0.5)
    assert status == SEARCHING and left < 0
    # ...and ends, rather than reversing forever
    _, _, after = h.step(CENTRE, CENTRE, CENTRE, moved=True, now=2.0)
    assert after == HOMING


# ── searching and giving up ───────────────────────────────────────────
def test_no_signal_sweeps_in_place():
    h = IrHoming()
    left, right, status = h.step(None, None, None, moved=True, now=0.0)
    assert status == SEARCHING
    assert left == -right != 0


def test_sweep_clears_the_rotation_floor():
    """Below ~34 mm/s per wheel this base does not turn at all, so a sweep
    slower than that would look like a hang."""
    h = IrHoming()
    left, right, _ = h.step(None, None, None, moved=True, now=0.0)
    assert abs(left) >= 34.0 and abs(right) >= 34.0


def test_gives_up_after_the_search_timeout():
    h = IrHoming(search_timeout_s=5.0)
    h.step(None, None, None, moved=True, now=0.0)
    _, _, status = h.step(None, None, None, moved=True, now=6.0)
    assert status == LOST


def test_reacquiring_resets_the_give_up_clock():
    h = IrHoming(search_timeout_s=5.0)
    h.step(None, None, None, moved=True, now=0.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=True, now=3.0)
    _, _, status = h.step(None, None, None, moved=True, now=7.0)
    assert status == SEARCHING       # not LOST: the clock restarted at 7.0


# ── seating ───────────────────────────────────────────────────────────
def test_stalled_while_centred_is_seated():
    h = IrHoming(stall_s=1.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=False, now=0.0)
    _, _, status = h.step(CENTRE, CENTRE, CENTRE, moved=False, now=1.5)
    assert status == SEATED


def test_seated_stops_the_wheels():
    h = IrHoming(stall_s=1.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=False, now=0.0)
    left, right, _ = h.step(CENTRE, CENTRE, CENTRE, moved=False, now=1.5)
    assert (left, right) == (0.0, 0.0)


def test_brief_stall_is_not_seating():
    h = IrHoming(stall_s=1.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=False, now=0.0)
    _, _, status = h.step(CENTRE, CENTRE, CENTRE, moved=False, now=0.5)
    assert status == HOMING


def test_movement_resets_the_stall_clock():
    h = IrHoming(stall_s=1.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=False, now=0.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=True, now=0.5)
    _, _, status = h.step(CENTRE, CENTRE, CENTRE, moved=False, now=1.2)
    assert status == HOMING


def test_stall_off_centre_is_not_seating():
    """This is the whole point of requiring both buoys: a robot wedged against
    a chair leg while glimpsing one buoy is stuck, not docked. The old
    criterion — stationary and seeing any IR — called that "docked"."""
    h = IrHoming(stall_s=1.0)
    h.step(GREEN, None, None, moved=False, now=0.0)
    _, _, status = h.step(GREEN, None, None, moved=False, now=5.0)
    assert status != SEATED


def test_stall_with_no_signal_is_not_seating():
    h = IrHoming(stall_s=1.0)
    h.step(None, None, None, moved=False, now=0.0)
    _, _, status = h.step(None, None, None, moved=False, now=5.0)
    assert status == SEARCHING


def test_force_field_plus_both_buoys_still_seats():
    """173 is the reading right at the station — force field does not
    disqualify a centred stall."""
    h = IrHoming(stall_s=1.0)
    h.step(BOTH_PLUS_FF, BOTH_PLUS_FF, BOTH_PLUS_FF, moved=False, now=0.0)
    _, _, status = h.step(BOTH_PLUS_FF, BOTH_PLUS_FF, BOTH_PLUS_FF,
                          moved=False, now=1.5)
    assert status == SEATED


def test_split_buoys_across_receivers_count_as_centred():
    """Left receiver on one buoy, right on the other: the robot straddles the
    corridor centre even though no single receiver sees both."""
    h = IrHoming()
    left, right, status = h.step(None, GREEN, RED, moved=True, now=0.0)
    assert status == HOMING
    assert left == right


# ── reset ─────────────────────────────────────────────────────────────
def test_reset_clears_a_pending_stall():
    h = IrHoming(stall_s=1.0)
    h.step(CENTRE, CENTRE, CENTRE, moved=False, now=0.0)
    h.reset()
    _, _, status = h.step(CENTRE, CENTRE, CENTRE, moved=False, now=1.5)
    assert status == HOMING


# ── driver wiring (static, no ROS) ────────────────────────────────────
@pytest.fixture(scope='module')
def driver_src():
    with open(DRIVER) as f:
        return f.read()


def test_driver_runs_homing_on_a_timer(driver_src):
    assert '_ir_homing_step' in driver_src
    assert 'self.create_timer(0.1,  self._ir_homing_step)' in driver_src


def test_driver_enters_full_mode_before_homing(driver_src):
    """We steer, so the OI must be in Full — the opposite of the 143 path,
    which drops to Passive. Homing in Passive is silently ignored wheels."""
    cb = driver_src.split('def _dock_cb')[1].split('def _dock_seated')[0]
    connect = cb.index('self._connect_oi()')
    build = cb.index('self._homing = IrHoming(')
    assert connect < build


def test_homing_bypasses_the_loose_seated_check(driver_src):
    """_dock_seated ("stationary and sees any IR") passes for a robot stuck
    near the base. While homing owns the approach it must not be consulted."""
    chk = driver_src.split('def _check_docking')[1].split('def _safe_get_ir')[0]
    guard = chk.index('if self._homing is not None:')
    loose = chk.index('self._dock_seated()')
    assert guard < loose


def test_ir_read_uses_one_query_list(driver_src):
    """Three separate reads cost ~150ms and would starve the 20Hz odometry
    timer sharing this port."""
    assert 'self.bot.SCI.write(149, (3, 17, 52, 53))' in driver_src


def test_driver_stops_the_wheels_when_seated_or_lost(driver_src):
    step = driver_src.split('def _ir_homing_step')[1].split('def _safe_get_encoders')[0]
    for status in ('ir_homing.SEATED', 'ir_homing.LOST'):
        branch = step.split(f'if status == {status}:')[1][:200]
        assert 'drive_direct(0, 0)' in branch


def test_swap_buoys_is_exposed_as_a_parameter(driver_src):
    """A wrong guess steers away from the dock; fixing it must not need a
    rebuild."""
    assert "declare_parameter('ir_swap_buoys'" in driver_src
    assert "self.get_parameter('ir_swap_buoys').value" in driver_src
