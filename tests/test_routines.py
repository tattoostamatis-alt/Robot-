"""Unit tests for routine learning (home_robot/routines.py)."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.routines import (  # noqa: E402
    RoutineLearner, circular_mean_hours, hours_between,
)

DAY = 86400.0


def at(day_offset, hour, minute=0):
    """Epoch for a wall-clock time, day 0 = Monday 2026-08-03."""
    base = time.mktime((2026, 8, 3, 0, 0, 0, 0, 0, -1))     # a Monday
    return base + day_offset * DAY + hour * 3600 + minute * 60


# ── circular time ─────────────────────────────────────────────────────────────

def test_hours_between_is_circular():
    """‼️ 23:50 and 00:10 are 20 minutes apart, not 23h40."""
    assert hours_between(23.833, 0.167) == pytest.approx(0.334, abs=1e-3)
    assert hours_between(1, 2) == pytest.approx(1)
    assert hours_between(0, 12) == pytest.approx(12)


def test_hours_between_is_symmetric():
    assert hours_between(3, 21) == hours_between(21, 3)


def test_circular_mean_of_a_tight_cluster():
    mean, spread = circular_mean_hours([8.0, 8.2, 7.9, 8.1])
    assert mean == pytest.approx(8.05, abs=0.1)
    assert spread < 0.5


def test_circular_mean_across_midnight():
    """Linear averaging would put these at noon — the exact opposite."""
    mean, _ = circular_mean_hours([23.5, 0.5])
    assert mean == pytest.approx(0.0, abs=0.1) or mean == pytest.approx(24.0, abs=0.1)


def test_scattered_times_have_a_large_spread():
    _, spread = circular_mean_hours([2.0, 9.0, 15.0, 21.0])
    assert spread > 3.0


def test_empty_hours_do_not_crash():
    mean, spread = circular_mean_hours([])
    assert spread == 12.0


# ── learning ──────────────────────────────────────────────────────────────────

def _teach(learner, key, hour, days=range(8)):
    for d in days:
        learner.observe(key, at(d, hour))
    return learner


def test_nothing_is_a_routine_at_first():
    r = RoutineLearner()
    r.observe('kouzina', at(0, 8))
    assert r.established() == []


def test_a_daily_habit_becomes_established():
    r = _teach(RoutineLearner(), 'kouzina', 8)
    assert 'kouzina' in r.established()


def test_four_times_in_one_day_is_not_a_habit():
    """‼️ Span matters. Someone in the kitchen four times on Monday has not
    taught the robot anything about Tuesday."""
    r = RoutineLearner()
    for h in (8, 12, 18, 21):
        r.observe('kouzina', at(0, h))
    assert r.established() == []


def test_repeats_within_a_day_count_once():
    r = RoutineLearner()
    for _ in range(10):
        assert r.observe('kouzina', at(0, 8)) in (True, False)
    assert r.describe('kouzina')[2] == 1        # one occurrence, not ten


def test_scattered_times_never_establish():
    # Present every day, but at a wildly different hour each time.
    r = RoutineLearner()
    for d, h in enumerate((3, 11, 17, 22, 6, 14, 20, 1)):
        r.observe('random', at(d, h))
    assert 'random' not in r.established()


def test_describe_reports_the_usual_hour():
    r = _teach(RoutineLearner(), 'max_home', 18)
    mean, spread, count, days = r.describe('max_home')
    assert mean == pytest.approx(18, abs=0.2)
    assert spread < 1.0
    assert count == 8


def test_describe_unknown_key_is_none():
    assert RoutineLearner().describe('nope') is None


# ── noticing the absence ──────────────────────────────────────────────────────

def test_overdue_when_the_window_has_passed():
    r = _teach(RoutineLearner(), 'max_home', 18)
    late = r.overdue(now=at(8, 21))              # 21:00, three hours late
    assert late and late[0][0] == 'max_home'
    assert late[0][1] == pytest.approx(18, abs=0.3)


def test_not_overdue_before_the_usual_time():
    """‼️ Saying "Max is not home" at 17:00 when he arrives at 18:00 is noise."""
    r = _teach(RoutineLearner(), 'max_home', 18)
    assert r.overdue(now=at(8, 17)) == []


def test_not_overdue_inside_the_grace_period():
    r = _teach(RoutineLearner(), 'max_home', 18)
    assert r.overdue(now=at(8, 18, 30)) == []


def test_not_overdue_once_it_has_happened():
    r = _teach(RoutineLearner(), 'max_home', 18)
    r.observe('max_home', at(8, 18))             # happened today
    assert r.overdue(now=at(8, 22)) == []


def test_not_overdue_on_a_day_it_never_happens():
    # Weekdays only: Monday(0) to Friday(4).
    r = RoutineLearner(min_occurrences=4, min_span_days=7)
    for week in range(2):
        for d in range(5):
            r.observe('school', at(week * 7 + d, 8))
    # Day 12 is a Saturday.
    assert time.localtime(at(12, 20)).tm_wday == 5
    assert r.overdue(now=at(12, 20)) == []


def test_unestablished_routines_are_never_overdue():
    r = RoutineLearner()
    r.observe('once', at(0, 8))
    assert r.overdue(now=at(1, 23)) == []


def test_most_overdue_comes_first():
    r = RoutineLearner()
    for d in range(8):
        r.observe('early', at(d, 9))
        r.observe('late', at(d, 19))
    out = r.overdue(now=at(8, 22))
    assert [k for k, _, _ in out][0] == 'early'


# ── unusual timing ────────────────────────────────────────────────────────────

def test_unusual_time_is_flagged():
    r = _teach(RoutineLearner(), 'kouzina', 8)
    assert r.is_unusual_time('kouzina', at(8, 3))       # 03:00, way off


def test_usual_time_is_not_flagged():
    r = _teach(RoutineLearner(), 'kouzina', 8)
    assert not r.is_unusual_time('kouzina', at(8, 8, 20))


def test_unusual_time_needs_an_established_routine():
    r = RoutineLearner()
    r.observe('kouzina', at(0, 8))
    assert not r.is_unusual_time('kouzina', at(1, 3))


# ── persistence and ageing ────────────────────────────────────────────────────

def test_round_trip():
    r = _teach(RoutineLearner(), 'max_home', 18)
    restored = RoutineLearner.from_dict(r.to_dict())
    assert restored.established() == r.established()
    assert restored.describe('max_home')[0] == pytest.approx(
        r.describe('max_home')[0])


def test_from_dict_tolerates_empty():
    assert RoutineLearner.from_dict({}).established() == []


def test_old_occurrences_are_forgotten():
    """A household's rhythm changes; a routine learned in March must not still
    be asserted in July."""
    r = RoutineLearner(max_age_days=30)
    for d in range(8):
        r.observe('old', at(d, 8))
    assert 'old' in r.established()
    r.prune(now=at(200, 12))
    assert r.established() == []
