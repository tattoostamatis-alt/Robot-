"""Sorting plans (home_robot/sort_planner.py) + the shipped rules file.

Sorting is fetch in a loop with a rule about where things belong. The parts
worth testing are the ones that decide whether a multi-minute run ends
sensibly: what is in scope, in what order, and what happens when a grasp keeps
missing.

Robot-free.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_sort_planner.py -q
"""
import os

import pytest
import yaml

from home_robot.sort_planner import (
    MAX_ATTEMPTS, SortProgress, SortRules, plan_order,
)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_FILE = os.path.join(PKG, 'config', 'sort_rules.yaml')

_MAP = {
    'kouzina': ['cup', 'bottle'],
    'saloni': ['teddy bear', 'sports ball'],
}


def _rules(default=None):
    return SortRules(mapping=_MAP, default=default)


def _obj(label, x, y):
    return {'label': label, 'x': x, 'y': y}


# ── rules ───────────────────────────────────────────────────────────────────

def test_a_label_maps_to_its_destination():
    r = _rules()
    assert r.destination('cup') == 'kouzina'
    assert r.destination('teddy bear') == 'saloni'


def test_lookup_is_case_and_space_insensitive():
    """Labels arrive from YOLO and from the LLM; neither is careful."""
    r = _rules()
    assert r.destination('  CUP ') == 'kouzina'


def test_an_unknown_label_has_no_destination_by_default():
    """‼️ A robot that carries off something it was never told about is worse
    than one that leaves the floor untidy."""
    assert _rules().destination('chainsaw') is None


def test_a_default_can_catch_everything_else():
    assert _rules(default='saloni').destination('chainsaw') == 'saloni'


def test_a_label_in_two_rooms_is_reported_not_silently_overwritten():
    r = SortRules(mapping={'kouzina': ['cup'], 'saloni': ['cup']})
    assert r.duplicates, 'a duplicated label must be surfaced as a config error'
    assert r.destination('cup') == 'kouzina', 'first rule should win, not last'


# ── ordering ────────────────────────────────────────────────────────────────

def test_nearest_object_comes_first():
    objs = [_obj('cup', 5, 0), _obj('bottle', 1, 0), _obj('teddy bear', 3, 0)]
    order = plan_order(objs, (0, 0), _rules())
    assert [e['label'] for e in order] == ['bottle', 'teddy bear', 'cup']


def test_unknown_objects_are_skipped():
    order = plan_order([_obj('chainsaw', 1, 0), _obj('cup', 2, 0)],
                       (0, 0), _rules())
    assert [e['label'] for e in order] == ['cup']


def test_a_group_narrows_to_one_destination():
    """"μάζεψε τα παιχνίδια" must not also carry the cups to the kitchen."""
    objs = [_obj('cup', 1, 0), _obj('teddy bear', 2, 0)]
    order = plan_order(objs, (0, 0), _rules(), group='saloni')
    assert [e['label'] for e in order] == ['teddy bear']


def test_objects_across_the_house_are_left_for_later():
    order = plan_order([_obj('cup', 50, 50)], (0, 0), _rules(), max_reach=6.0)
    assert order == []


def test_malformed_entries_do_not_break_the_plan():
    """object_memory is machine-written but has been wrong before."""
    objs = [{'label': 'cup'},                    # no position
            {'x': 1, 'y': 1},                    # no label
            {'label': 'cup', 'x': 'nope', 'y': 0},
            _obj('bottle', 1, 0)]
    assert [e['label'] for e in plan_order(objs, (0, 0), _rules())] == ['bottle']


def test_the_order_is_deterministic_for_equidistant_objects():
    a = plan_order([_obj('cup', 1, 0), _obj('bottle', 1, 0)], (0, 0), _rules())
    b = plan_order([_obj('bottle', 1, 0), _obj('cup', 1, 0)], (0, 0), _rules())
    assert [e['label'] for e in a] == [e['label'] for e in b]


def test_each_entry_carries_its_destination():
    order = plan_order([_obj('cup', 1, 0)], (0, 0), _rules())
    assert order[0]['destination'] == 'kouzina'


# ── progress: what happens when it goes wrong ───────────────────────────────

def test_a_completed_object_is_not_attempted_again():
    p = SortProgress()
    order = plan_order([_obj('cup', 1, 0), _obj('bottle', 2, 0)], (0, 0), _rules())
    first = p.next_target(order)
    p.record(first, True)
    assert p.next_target(order)['label'] != first['label']


def test_an_object_that_keeps_failing_is_abandoned():
    """‼️ Otherwise the robot loops on the one thing it cannot grasp while the
    rest of the floor stays untidied."""
    p = SortProgress(max_attempts=2)
    order = plan_order([_obj('cup', 1, 0)], (0, 0), _rules())
    for _ in range(MAX_ATTEMPTS):
        t = p.next_target(order)
        assert t is not None
        p.record(t, False)
    assert p.next_target(order) is None
    assert len(p.failed) == 1


def test_one_failure_still_gets_a_retry():
    """A bad grasp angle or an unsettled servo deserves a second try."""
    p = SortProgress(max_attempts=2)
    order = plan_order([_obj('cup', 1, 0)], (0, 0), _rules())
    p.record(p.next_target(order), False)
    assert p.next_target(order) is not None


def test_two_cups_in_different_rooms_are_two_jobs():
    p = SortProgress()
    order = plan_order([_obj('cup', 1, 0), _obj('cup', 5, 0)], (0, 0), _rules())
    p.record(order[0], True)
    nxt = p.next_target(order)
    assert nxt is not None and nxt['x'] == 5


def test_jitter_does_not_turn_one_cup_into_three():
    """object_memory positions wobble by centimetres between sightings."""
    p = SortProgress()
    a = {'label': 'cup', 'x': 1.00, 'y': 0.0, 'destination': 'kouzina'}
    b = {'label': 'cup', 'x': 1.02, 'y': 0.0, 'destination': 'kouzina'}
    p.record(a, True)
    assert p.next_target([b]) is None


# ── the spoken summary ──────────────────────────────────────────────────────

def test_the_summary_never_overclaims():
    p = SortProgress()
    assert 'Δεν βρήκα' in p.summary_el()

    p.record({'label': 'cup', 'x': 1, 'y': 0}, True)
    assert 'Μάζεψα 1' in p.summary_el()

    p2 = SortProgress(max_attempts=1)
    p2.record({'label': 'cup', 'x': 1, 'y': 0}, False)
    assert 'Δεν κατάφερα' in p2.summary_el()

    p2.record({'label': 'bottle', 'x': 2, 'y': 0}, True)
    s = p2.summary_el()
    assert 'Μάζεψα 1' in s and 'δεν' in s, 'a partial run must say both halves'


# ── the shipped rules file ──────────────────────────────────────────────────

def test_the_rules_file_parses_and_has_no_duplicates():
    data = yaml.safe_load(open(RULES_FILE))
    rules = SortRules.from_yaml(data)
    assert not rules.duplicates, f'duplicated labels: {rules.duplicates}'
    assert rules.known_labels, 'the shipped rules file is empty'


def test_every_destination_is_a_taught_location():
    """A destination the robot cannot navigate to makes the mission fail at
    the last step, after it has already picked the object up."""
    data = yaml.safe_load(open(RULES_FILE))
    locations = yaml.safe_load(open(os.path.join(PKG, 'config', 'locations.yaml')))
    for dest in (data.get('rules') or {}):
        assert dest in locations, (
            f'sort_rules.yaml sends things to {dest!r}, which is not in '
            'locations.yaml — the robot cannot drive there')


def test_the_default_is_off_in_the_shipped_file():
    data = yaml.safe_load(open(RULES_FILE))
    assert data.get('default') is None, (
        'a catch-all default makes the robot carry off objects it was never '
        'told about; leave it null unless deliberately changed')
