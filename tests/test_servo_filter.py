"""Tests for the robust servo target aggregation (see home_robot/servo_filter.py).

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_servo_filter.py -q
"""
from home_robot.servo_filter import (median, median_point, spread, converged,
                                     hand_eye_correction)


def test_median_odd_and_even():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_point_rejects_a_single_outlier():
    # Four good frames near (1,1,0.5) and one wild outlier — median ignores it,
    # a mean would be dragged toward the outlier.
    samples = [(1.0, 1.0, 0.50), (1.01, 0.99, 0.51), (0.99, 1.0, 0.49),
               (1.0, 1.01, 0.50), (5.0, 5.0, 3.0)]
    mx, my, mz = median_point(samples)
    assert abs(mx - 1.0) < 0.05
    assert abs(my - 1.0) < 0.05
    assert abs(mz - 0.5) < 0.05


def test_median_point_empty_is_none():
    assert median_point([]) is None


def test_spread_measures_disagreement():
    tight = [(1.0, 1.0, 0.5), (1.005, 1.0, 0.5)]
    loose = [(1.0, 1.0, 0.5), (1.2, 1.0, 0.5)]
    assert spread(tight) < spread(loose)
    assert spread([(1.0, 1.0, 0.5)]) == 0.0     # single sample = settled


def test_converged_needs_agreement():
    settled = [(1.0, 1.0, 0.5), (1.005, 1.002, 0.5)]
    assert converged(settled, tolerance=0.015)
    wandering = [(1.0, 1.0, 0.5), (1.2, 1.0, 0.5)]
    assert not converged(wandering, tolerance=0.015)


def test_converged_needs_min_samples():
    assert not converged([(1.0, 1.0, 0.5)], tolerance=0.015)  # one frame isn't agreement


def test_converged_only_checks_recent_window():
    # An early outlier then two tight frames → converged on the recent window.
    samples = [(5.0, 5.0, 3.0), (1.0, 1.0, 0.5), (1.004, 1.0, 0.5)]
    assert converged(samples, tolerance=0.015, min_samples=2)


# ── hand_eye_correction ────────────────────────────────────────────────────

def test_hand_eye_already_there_converges_immediately():
    cmd, done = hand_eye_correction((0.3, 0.1), target_xy=(0.3, 0.1),
                                    actual_xy=(0.3, 0.1), tolerance=0.01)
    assert done
    assert cmd == (0.3, 0.1)


def test_hand_eye_nudges_by_the_observed_error():
    # Commanded (0.30, 0.10) but the wrist tag says the gripper landed 20mm
    # short in x — a systematic bias the object-detection servo never sees.
    # The next command overshoots by that same 20mm to cancel it out.
    cmd, done = hand_eye_correction((0.30, 0.10), target_xy=(0.30, 0.10),
                                    actual_xy=(0.28, 0.10), tolerance=0.005)
    assert not done
    assert abs(cmd[0] - 0.32) < 1e-9
    assert abs(cmd[1] - 0.10) < 1e-9


def test_hand_eye_within_tolerance_stops_correcting():
    cmd, done = hand_eye_correction((0.30, 0.10), target_xy=(0.30, 0.10),
                                    actual_xy=(0.302, 0.099), tolerance=0.005)
    assert done
    assert cmd == (0.30, 0.10)      # unchanged — no further move needed


def test_hand_eye_repeated_steps_converge_on_a_constant_bias():
    # A constant offset (actual = cmd - bias) should cancel out in one step: a
    # `cmd` that produces `actual` == target, not necessarily cmd == target —
    # `cmd` is what gets sent to the arm, `actual` is what the tag then reads.
    bias = (0.015, -0.02)
    cmd = (0.30, 0.10)
    target = (0.30, 0.10)
    actual = (cmd[0] - bias[0], cmd[1] - bias[1])
    for _ in range(5):
        cmd, done = hand_eye_correction(cmd, target, actual, tolerance=0.001)
        actual = (cmd[0] - bias[0], cmd[1] - bias[1])
        if done:
            break
    assert done
    assert abs(actual[0] - target[0]) < 0.001
    assert abs(actual[1] - target[1]) < 0.001
