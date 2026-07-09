"""Tests for the robust servo target aggregation (see home_robot/servo_filter.py).

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_servo_filter.py -q
"""
from home_robot.servo_filter import median, median_point, spread, converged


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
