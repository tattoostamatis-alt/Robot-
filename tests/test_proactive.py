"""Unit tests for proactive observation (home_robot/proactive.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.proactive import (  # noqa: E402
    Baseline, Observation, ObservationGate, find_observations,
)

HOUR = 3600.0


def _settle(baseline, label, x, y, room, t0=1000.0, visits=3):
    """Teach the baseline a home across `visits` well-separated visits."""
    for i in range(visits):
        baseline.observe(label, x, y, room, now=t0 + i * HOUR)
    return baseline


# ── Baseline ──────────────────────────────────────────────────────────────────

def test_one_sighting_is_not_a_baseline():
    b = Baseline()
    b.observe('cup', 1.0, 2.0, 'kouzina', now=1000.0)
    assert b.home_of('cup') is None


def test_settles_after_three_separate_visits():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    home = b.home_of('cup')
    assert home is not None
    assert home[2] == 'kouzina'


def test_repeated_sightings_in_one_visit_do_not_settle():
    # The robot parked in a room staring at a cup for a minute must not
    # conclude the cup lives there.
    b = Baseline()
    for i in range(20):
        b.observe('cup', 1.0, 2.0, 'kouzina', now=1000.0 + i)
    assert b.home_of('cup') is None


def test_visits_must_be_separated_by_the_gap():
    b = Baseline(visits_required=3, visit_gap=1200.0)
    b.observe('cup', 1.0, 2.0, 'kouzina', now=1000.0)
    b.observe('cup', 1.0, 2.0, 'kouzina', now=1000.0 + 1199)   # same visit
    b.observe('cup', 1.0, 2.0, 'kouzina', now=1000.0 + 1300)   # new visit
    assert b.home_of('cup') is None                            # only 2 visits
    b.observe('cup', 1.0, 2.0, 'kouzina', now=1000.0 + 5000)
    assert b.home_of('cup') is not None


def test_baseline_position_tracks_slow_drift():
    b = Baseline()
    for i in range(3):
        b.observe('chair', 1.0, 1.0, 'saloni', now=1000.0 + i * HOUR)
    before = b.home_of('chair')[0]
    for i in range(3, 12):
        b.observe('chair', 2.0, 1.0, 'saloni', now=1000.0 + i * HOUR)
    after = b.home_of('chair')[0]
    assert after > before          # moved towards the new position
    assert after <= 2.0


def test_known_labels_lists_only_settled_ones():
    b = Baseline()
    _settle(b, 'cup', 1.0, 2.0, 'kouzina')
    b.observe('knife', 5.0, 5.0, 'kouzina', now=1000.0)
    assert b.known_labels() == ['cup']


def test_baseline_survives_a_round_trip():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    restored = Baseline.from_dict(b.to_dict())
    assert restored.home_of('cup') == b.home_of('cup')


def test_from_dict_tolerates_empty_state():
    # First boot, or a truncated state file.
    assert Baseline.from_dict({}).known_labels() == []


# ── find_observations ─────────────────────────────────────────────────────────

def test_object_in_its_usual_place_is_not_news():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    obs = find_observations(b, [{'label': 'cup', 'x': 1.1, 'y': 2.1,
                                 'room': 'kouzina'}])
    assert obs == []


def test_small_shift_is_not_news():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    obs = find_observations(b, [{'label': 'cup', 'x': 1.4, 'y': 2.0,
                                 'room': 'kouzina'}])
    assert obs == []


def test_object_in_another_room_is_reported():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    obs = find_observations(b, [{'label': 'cup', 'x': 8.0, 'y': 9.0,
                                 'room': 'saloni'}])
    assert len(obs) == 1
    assert obs[0].kind == 'moved_room'
    assert obs[0].from_room == 'kouzina'
    assert obs[0].room == 'saloni'


def test_big_move_inside_the_same_room_is_reported_more_weakly():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    obs = find_observations(b, [{'label': 'cup', 'x': 6.0, 'y': 2.0,
                                 'room': 'kouzina'}])
    assert len(obs) == 1
    assert obs[0].kind == 'moved'
    assert obs[0].importance < 0.9


def test_unlearned_object_produces_nothing():
    # A fresh map must not make the robot chatter about everything it sees.
    b = Baseline()
    obs = find_observations(b, [{'label': 'cup', 'x': 1.0, 'y': 2.0,
                                 'room': 'saloni'}])
    assert obs == []


def test_room_changes_outrank_in_room_moves():
    b = Baseline()
    _settle(b, 'cup', 1.0, 2.0, 'kouzina')
    _settle(b, 'book', 1.0, 2.0, 'saloni')
    obs = find_observations(b, [
        {'label': 'book', 'x': 7.0, 'y': 2.0, 'room': 'saloni'},   # moved
        {'label': 'cup',  'x': 8.0, 'y': 9.0, 'room': 'toualeta'}, # moved_room
    ])
    assert [o.kind for o in obs] == ['moved_room', 'moved']


def test_malformed_instances_are_skipped():
    b = _settle(Baseline(), 'cup', 1.0, 2.0, 'kouzina')
    obs = find_observations(b, [{}, {'label': 'cup'}, {'x': 1, 'y': 2}])
    assert obs == []


# ── ObservationGate ───────────────────────────────────────────────────────────

def _obs(label='cup', room='saloni'):
    return Observation('moved_room', label, room=room, from_room='kouzina')


def test_says_nothing_to_an_empty_room():
    g = ObservationGate()
    assert not g.allow(_obs(), now=1000.0, person_present=False)


def test_allows_a_first_observation_with_someone_present():
    g = ObservationGate()
    assert g.allow(_obs(), now=1000.0, person_present=True)


def test_enforces_a_gap_between_observations():
    g = ObservationGate(min_interval=180.0)
    o = _obs()
    assert g.allow(o, now=1000.0)
    g.record(o, now=1000.0)
    assert not g.allow(_obs('book'), now=1100.0)     # 100s later, too soon
    assert g.allow(_obs('book'), now=1200.0)


def test_never_repeats_the_same_observation_within_cooldown():
    g = ObservationGate(min_interval=0.0, repeat_cooldown=7200.0)
    o = _obs()
    g.record(o, now=1000.0)
    assert not g.allow(_obs(), now=1000.0 + 3600)    # same key, still cooling
    assert g.allow(_obs(), now=1000.0 + 7300)


def test_a_different_observation_is_not_blocked_by_the_cooldown():
    g = ObservationGate(min_interval=0.0)
    g.record(_obs('cup'), now=1000.0)
    assert g.allow(_obs('book'), now=1001.0)


def test_hourly_cap():
    g = ObservationGate(min_interval=0.0, max_per_hour=3, repeat_cooldown=0.0)
    for i in range(3):
        o = _obs(f'thing{i}')
        assert g.allow(o, now=1000.0 + i)
        g.record(o, now=1000.0 + i)
    assert not g.allow(_obs('extra'), now=1010.0)


def test_hourly_cap_rolls_off():
    g = ObservationGate(min_interval=0.0, max_per_hour=2, repeat_cooldown=0.0)
    for i in range(2):
        o = _obs(f'thing{i}')
        g.record(o, now=1000.0 + i)
    assert not g.allow(_obs('extra'), now=1500.0)
    assert g.allow(_obs('extra'), now=1000.0 + HOUR + 10)


def test_quiet_hours_block_speech():
    g = ObservationGate(quiet_hours=(23, 8))
    assert not g.allow(_obs(), now=1000.0, hour=2)      # 02:00
    assert not g.allow(_obs(), now=1000.0, hour=23)
    assert g.allow(_obs(), now=1000.0, hour=12)


def test_quiet_hours_boundaries():
    g = ObservationGate(quiet_hours=(23, 8))
    assert not g.allow(_obs(), now=1000.0, hour=7)      # last quiet hour
    assert g.allow(_obs(), now=1000.0, hour=8)          # awake again


def test_quiet_hours_can_be_disabled():
    g = ObservationGate(quiet_hours=(0, 0))
    assert g.allow(_obs(), now=1000.0, hour=3)


def test_record_is_required_for_rate_limiting_to_apply():
    # allow() alone must not consume budget: a caller that decided not to speak
    # (TTS busy, say) should not be penalised.
    g = ObservationGate(min_interval=180.0)
    assert g.allow(_obs(), now=1000.0)
    assert g.allow(_obs(), now=1001.0)
