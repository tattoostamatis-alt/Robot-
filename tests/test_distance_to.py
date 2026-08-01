"""Tests for the `distance_to` voice tool (llm_bridge_node._distance_to).

Built 2026-08-01: the measuring code had existed since 383165a but was never
wired into TOOLS, so "πόση απόσταση έχει;" answered "δεν ξέρω".

The method is lifted out of the node rather than imported, because importing
llm_bridge_node pulls in rclpy, ollama and cv2 — none of which a unit test
should need. Same trick as tests/test_tool_router.py uses for TOOLS.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_distance_to.py -q
"""
import math
import re
import time
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'llm_bridge_node.py').read_text()

_ns = {'math': math, 'time': time}
_body = re.search(r'^    DETECTION_MAX_AGE = .*?^    def _distance_to.*?'
                  r'(?=\n\n    def |\n\nclass |\Z)', _SRC, re.M | re.S)
assert _body, 'could not extract _distance_to from llm_bridge_node.py'
exec(compile('class Bridge:\n' + _body.group(0),  # noqa: S102 - test extraction
             'llm_bridge_distance', 'exec'), _ns)
Bridge = _ns['Bridge']


def bridge(objects, age=0.0):
    b = Bridge()
    b._latest_objects = objects
    b._objects_stamp = None if objects is None else time.time() - age
    return b


def person(distance, x=0.0, label='person'):
    """One detected_objects entry as object_detector.py publishes it."""
    return {'label': label, 'conf': 0.9, 'x': x, 'y': 0.0, 'z': distance,
            'clutter': False, 'box_distance': distance}


# ── the happy path ───────────────────────────────────────────────────────────

def test_reports_the_distance_of_a_seen_person():
    r = bridge([person(1.66)])._distance_to('person')
    assert r['status'] == 'ok' and r['seen'] is True
    assert r['distance_m'] == 1.66


def test_defaults_are_reported_in_metres_rounded_to_cm():
    r = bridge([person(1.6649)])._distance_to('person')
    assert r['distance_m'] == 1.66


def test_the_nearest_match_wins():
    r = bridge([person(3.2), person(1.1), person(2.0)])._distance_to('person')
    assert r['distance_m'] == 1.1
    assert r['count'] == 3


def test_matching_is_case_insensitive_and_trimmed():
    r = bridge([person(2.0, label='Person')])._distance_to('  PERSON ')
    assert r['seen'] is True


def test_any_coco_label_works_not_just_people():
    r = bridge([person(0.8, label='cup')])._distance_to('cup')
    assert r['seen'] is True and r['distance_m'] == 0.8


# ── bearing ──────────────────────────────────────────────────────────────────
# Camera optical frame is +x right, +z forward.

def test_an_object_to_the_right_reads_as_δεξιά():
    r = bridge([person(2.0, x=1.0)])._distance_to('person')
    assert r['side'] == 'δεξιά'
    assert r['bearing_deg'] < 0


def test_an_object_to_the_left_reads_as_αριστερά():
    r = bridge([person(2.0, x=-1.0)])._distance_to('person')
    assert r['side'] == 'αριστερά'
    assert r['bearing_deg'] > 0


def test_straight_ahead_reads_as_μπροστά():
    r = bridge([person(2.0, x=0.0)])._distance_to('person')
    assert r['side'] == 'μπροστά'
    assert r['bearing_deg'] == 0


# ── the ways it must NOT lie ─────────────────────────────────────────────────

def test_nothing_seen_is_reported_as_not_seen_not_as_zero():
    r = bridge([person(1.5, label='chair')])._distance_to('person')
    assert r['status'] == 'ok' and r['seen'] is False
    assert 'distance_m' not in r


def test_no_detections_at_all_is_an_error_naming_the_detector():
    r = bridge(None)._distance_to('person')
    assert r['status'] == 'error'
    assert 'object_detector' in r['reason']


def test_stale_detections_are_refused_rather_than_answered():
    """A frozen camera used to be indistinguishable from a live one. Answering
    off a 30 s old frame invents a person who has already left the room."""
    r = bridge([person(1.5)], age=30.0)._distance_to('person')
    assert r['status'] == 'error'
    assert '30s' in r['reason']


def test_fresh_detections_just_inside_the_window_still_answer():
    r = bridge([person(1.5)], age=3.0)._distance_to('person')
    assert r['status'] == 'ok' and r['seen'] is True


def test_an_entry_with_no_usable_depth_is_skipped():
    """box_distance is None when every pixel in the box returned 0 (too shiny,
    too close). Falling back to z is fine; reporting None as a distance is not."""
    blind = {'label': 'person', 'x': 0.0, 'z': None, 'box_distance': None}
    assert bridge([blind])._distance_to('person')['seen'] is False


def test_z_is_the_fallback_when_box_distance_is_missing():
    entry = {'label': 'person', 'x': 0.0, 'z': 2.4, 'box_distance': None}
    r = bridge([entry])._distance_to('person')
    assert r['seen'] is True and r['distance_m'] == 2.4


@pytest.mark.parametrize('label', ['', None])
def test_an_empty_label_does_not_crash(label):
    r = bridge([person(1.5)])._distance_to(label)
    assert r['status'] in ('ok', 'error')
