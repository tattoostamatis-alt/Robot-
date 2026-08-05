"""Tests for the wheel slip / stall core (see home_robot/slip_detect.py).

Robot-free unit tests of the two comparisons and — mostly — of when the
detector must REFUSE to judge. The failure that matters for this feature is
not a missed slip, it is a robot that announces it is stuck while driving
normally, so most of these pin down the "not enough evidence" cases.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_slip_detect.py -q
"""

from home_robot.slip_detect import SlipDetector


def feed(det, seconds=2.0, hz_wheel=20, hz_gyro=6, vx=0.0, wheel_wz=0.0,
         gyro_wz=None, t0=100.0):
    """Drive the detector with steady samples at the real stream rates."""
    if gyro_wz is None:
        gyro_wz = wheel_wz
    for i in range(int(seconds * hz_wheel)):
        det.feed_wheel(t0 + i / hz_wheel, vx, wheel_wz)
    for i in range(int(seconds * hz_gyro)):
        det.feed_gyro(t0 + i / hz_gyro, gyro_wz)
    return t0 + seconds


# ── rotation ─────────────────────────────────────────────────────────────

def test_wheels_turn_but_gyro_flat_is_a_slip():
    det = SlipDetector()
    feed(det, wheel_wz=0.6, gyro_wz=0.0)
    slipping, detail = det.check_rotation()
    assert slipping is True
    assert detail['wheel_wz'] > 0.5 and detail['gyro_wz'] == 0.0


def test_agreeing_sensors_are_not_a_slip():
    det = SlipDetector()
    feed(det, wheel_wz=0.6, gyro_wz=0.55)
    slipping, _ = det.check_rotation()
    assert slipping is False


def test_calibration_error_is_not_a_slip():
    # wheel_base calibration alone puts the two ~20% apart on a good floor;
    # that must stay well clear of the threshold.
    det = SlipDetector()
    feed(det, wheel_wz=0.6, gyro_wz=0.48)
    slipping, _ = det.check_rotation()
    assert slipping is False


def test_slow_turn_is_not_judged():
    # ‼️ The BNO085 quantises small rates to exact 0.0, so a slow turn looks
    # identical to a stuck robot. It must return None, not True.
    det = SlipDetector()
    feed(det, wheel_wz=0.10, gyro_wz=0.0)
    slipping, detail = det.check_rotation()
    assert slipping is None
    assert 'slowly' in detail['reason']


def test_standing_still_is_not_judged():
    det = SlipDetector()
    feed(det, wheel_wz=0.0, gyro_wz=0.0)
    assert det.check_rotation()[0] is None


def test_no_gyro_at_all_is_not_judged():
    # A dead IMU is SILENT. Reporting "stuck" because no gyro samples arrived
    # would turn every IMU failure into the robot complaining about rugs.
    det = SlipDetector()
    for i in range(40):
        det.feed_wheel(100.0 + i / 20, 0.0, 0.6)
    assert det.check_rotation()[0] is None


def test_window_forgets_old_samples():
    det = SlipDetector(window=1.0)
    end = feed(det, seconds=2.0, wheel_wz=0.6, gyro_wz=0.0)   # slipping
    assert det.check_rotation()[0] is True
    # Now it starts turning properly again; the old bad samples must age out.
    feed(det, seconds=2.0, wheel_wz=0.6, gyro_wz=0.6, t0=end)
    assert det.check_rotation()[0] is False


# ── linear ───────────────────────────────────────────────────────────────

def scan_run(det, t0, seconds, hz, start_range, closing_mps):
    for i in range(int(seconds * hz)):
        t = t0 + i / hz
        r = start_range - closing_mps * (t - t0)
        det.feed_scan(t, max(0.05, r))
    return t0 + seconds


def test_driving_at_a_wall_that_does_not_get_closer_is_a_stall():
    det = SlipDetector()
    feed(det, seconds=2.0, vx=0.20, wheel_wz=0.0)
    scan_run(det, 100.0, 2.0, 10, start_range=1.5, closing_mps=0.0)
    slipping, detail = det.check_linear()
    assert slipping is True
    assert detail['closed_m'] == 0.0


def test_driving_at_a_wall_that_does_get_closer_is_fine():
    det = SlipDetector()
    feed(det, seconds=2.0, vx=0.20, wheel_wz=0.0)
    scan_run(det, 100.0, 2.0, 10, start_range=1.5, closing_mps=0.20)
    assert det.check_linear()[0] is False


def test_open_room_is_not_judged():
    # Nothing within max_range ahead: the check has no evidence either way and
    # must not call an open corridor a stall.
    det = SlipDetector()
    feed(det, seconds=2.0, vx=0.20, wheel_wz=0.0)
    for i in range(20):
        det.feed_scan(100.0 + i / 10, None)
    slipping, detail = det.check_linear()
    assert slipping is None
    assert 'nothing ahead' in detail['reason']


def test_far_wall_is_not_judged():
    det = SlipDetector(max_range=3.0)
    feed(det, seconds=2.0, vx=0.20, wheel_wz=0.0)
    scan_run(det, 100.0, 2.0, 10, start_range=6.0, closing_mps=0.0)
    assert det.check_linear()[0] is None


def test_turning_invalidates_the_range_check():
    # A turn sweeps the forward sector across the room, so range deltas stop
    # measuring forward progress.
    det = SlipDetector()
    feed(det, seconds=2.0, vx=0.20, wheel_wz=0.5)
    scan_run(det, 100.0, 2.0, 10, start_range=1.5, closing_mps=0.0)
    slipping, detail = det.check_linear()
    assert slipping is None
    assert detail['reason'] == 'turning'


def test_stationary_robot_is_not_judged():
    det = SlipDetector()
    feed(det, seconds=2.0, vx=0.0, wheel_wz=0.0)
    scan_run(det, 100.0, 2.0, 10, start_range=1.5, closing_mps=0.0)
    assert det.check_linear()[0] is None


def test_reset_clears_everything():
    det = SlipDetector()
    feed(det, wheel_wz=0.6, gyro_wz=0.0)
    assert det.check_rotation()[0] is True
    det.reset()
    assert det.check_rotation()[0] is None
