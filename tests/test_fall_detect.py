"""Fall detection geometry + hysteresis (home_robot/fall_detect.py).

Most of this file is about NOT crying wolf. A safety monitor that fires when
someone sits on the floor with the kids gets muted within a day, and a muted
monitor is worse than none — so the false-positive cases are tested at least as
hard as the true one.

Robot-free: skeletons are built by hand.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_fall_detect.py -q
"""
import pytest

from home_robot.fall_detect import (
    FallMonitor, box_is_wide, looks_fallen, torso_angle_deg,
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
)

FRAME_H = 480
FRAME_W = 640


def _person(sh_xy, hip_xy, box, conf=0.9):
    """A COCO-17 skeleton with only shoulders and hips filled in."""
    kps = [{'name': f'kp{i}', 'x': 0, 'y': 0, 'v': 0.0} for i in range(17)]
    for idx, (x, y) in ((L_SHOULDER, sh_xy), (R_SHOULDER, sh_xy),
                        (L_HIP, hip_xy), (R_HIP, hip_xy)):
        kps[idx] = {'name': f'kp{idx}', 'x': x, 'y': y, 'v': conf}
    x1, y1, x2, y2 = box
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': 0.9,
            'keypoints': kps}


def _standing(cy=240):
    """Upright: hips directly below shoulders, tall narrow box."""
    return _person((320, cy - 60), (320, cy + 40), (290, cy - 110, 350, cy + 130))


def _lying_on_floor():
    """Horizontal torso, low in the frame, wide box."""
    return _person((250, 430), (390, 435), (230, 400, 410, 460))


# ── geometry ────────────────────────────────────────────────────────────────

def test_an_upright_torso_reads_as_vertical():
    assert torso_angle_deg(_standing()) < 15


def test_a_horizontal_torso_reads_as_flat():
    assert torso_angle_deg(_lying_on_floor()) > 80


def test_missing_keypoints_give_no_angle():
    """Occluded hips must return None, not a confident wrong answer."""
    p = _standing()
    for i in (L_HIP, R_HIP):
        p['keypoints'][i]['v'] = 0.0
    assert torso_angle_deg(p) is None


def test_a_wide_box_is_recognised():
    assert box_is_wide(_lying_on_floor())
    assert not box_is_wide(_standing())


# ── the true positive ───────────────────────────────────────────────────────

def test_a_person_on_the_floor_is_flagged():
    assert looks_fallen(_lying_on_floor(), FRAME_H)


def test_a_fallen_person_with_occluded_torso_still_counts():
    """Down behind furniture is exactly when the hips are hidden — the wide
    box carries the decision on its own."""
    p = _lying_on_floor()
    for i in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER):
        p['keypoints'][i]['v'] = 0.0
    assert looks_fallen(p, FRAME_H)


# ── the false positives that would get this muted ───────────────────────────

def test_standing_is_not_a_fall():
    assert not looks_fallen(_standing(), FRAME_H)


def test_sitting_on_the_floor_is_not_a_fall():
    """Low in the frame, but the torso is still upright."""
    p = _person((320, 380), (320, 450), (295, 350, 350, 470))
    assert not looks_fallen(p, FRAME_H)


def test_lying_on_the_sofa_is_not_a_fall():
    """Horizontal, but at furniture height rather than on the floor."""
    p = _person((250, 210), (390, 215), (230, 185, 410, 245))
    assert not looks_fallen(p, FRAME_H)


def test_bending_over_is_not_a_fall_while_upright_enough():
    """Leaning to pick something up tilts the torso without laying it flat."""
    p = _person((300, 300), (340, 390), (280, 270, 370, 430))
    assert torso_angle_deg(p) < 55
    assert not looks_fallen(p, FRAME_H)


def test_nothing_is_flagged_without_a_frame_height():
    assert not looks_fallen(_lying_on_floor(), 0)
    assert not looks_fallen(None, FRAME_H)


# ── hysteresis: the part that decides whether this is usable ────────────────

class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_a_brief_horizontal_moment_never_alerts():
    """Tying a shoelace, reaching under the sofa, a bad frame."""
    clk = Clock()
    m = FallMonitor(hold_s=6.0, clock=clk)
    for _ in range(5):
        clk.t += 1.0
        assert not m.update(True)      # still inside the hold window
    clk.t += 1.0
    m.update(False)                    # got back up
    clk.t += 100.0
    assert not m.update(False)
    assert not m.alerted


def test_staying_down_alerts_once():
    clk = Clock()
    m = FallMonitor(hold_s=6.0, clock=clk)
    fired = []
    for _ in range(20):
        clk.t += 1.0
        fired.append(m.update(True))
    assert sum(fired) == 1, (
        'a person who stays down must alert exactly once, not on every frame '
        'for as long as they lie there')
    assert m.alerted


def test_the_alert_only_rearms_after_a_sustained_all_clear():
    """One occluded frame must not re-arm and immediately re-alert."""
    clk = Clock()
    m = FallMonitor(hold_s=2.0, clear_s=10.0, clock=clk)
    # First frame only starts the clock — nothing can be known about how long
    # someone has been down from a single sighting.
    m.update(True)
    clk.t += 3.0
    assert m.update(True)              # alert fires
    clk.t += 1.0
    m.update(False)                    # brief blip
    clk.t += 1.0
    m.update(True)
    assert m.alerted, 'the alert was cleared by a one-second blip'

    # A real, sustained recovery does clear it.
    for _ in range(12):
        clk.t += 1.0
        m.update(False)
    assert not m.alerted

    # And it can fire again for a genuine second fall (first frame arms, the
    # one after the hold window fires).
    m.update(True)
    clk.t += 3.0
    assert m.update(True)


def test_a_gap_in_detection_restarts_the_countdown():
    """Person out of frame mid-countdown: do not credit the elapsed time."""
    clk = Clock()
    m = FallMonitor(hold_s=6.0, clock=clk)
    clk.t += 5.0
    m.update(True)                     # 5s in, not yet fired
    clk.t += 1.0
    m.update(False)                    # lost them
    clk.t += 1.0
    assert not m.update(True), 'the countdown was not restarted'
