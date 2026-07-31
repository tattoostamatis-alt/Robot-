"""The room vocabulary must follow the map, not the other way round.

task_planner_node used to carry ROOM_ORDER = ['living_room', 'bedroom',
'kitchen'] — names no map has had for a long time. "τακτοποίησε το σπίτι"
therefore visited nothing and said "Δεν ξέρω πού βρίσκεται το δωμάτιο" three
times, and every remap could break it again silently. These tests fail the
moment a hardcoded list drifts from config/locations.yaml.
"""

import os
import re

import yaml

from home_robot.status_query import (NON_ROOMS, ROOM_LOCATIVE_EL,
                                     ROOM_NAMES_EL, format_location, room_el,
                                     room_locative_el, rooms_from_locations)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _locations():
    with open(os.path.join(_ROOT, 'config', 'locations.yaml')) as f:
        return yaml.safe_load(f)


def test_every_room_on_the_map_has_a_greek_name():
    for room in rooms_from_locations(_locations()):
        assert room in ROOM_NAMES_EL, f'{room} has no Greek name'
        assert room in ROOM_LOCATIVE_EL, f'{room} has no locative form'


def test_no_greek_name_for_a_room_the_map_no_longer_has():
    rooms = set(rooms_from_locations(_locations()))
    assert set(ROOM_NAMES_EL) == rooms
    assert set(ROOM_LOCATIVE_EL) == rooms


def test_the_dock_is_not_a_room_to_visit():
    rooms = rooms_from_locations(_locations())
    assert 'dock' not in rooms and 'dock_staging' not in rooms
    assert NON_ROOMS == {'dock', 'dock_staging'}


def test_room_order_follows_the_yaml_order():
    locations = {'saloni': {}, 'dock': {}, 'kouzina': {}}
    assert rooms_from_locations(locations) == ['saloni', 'kouzina']


def test_missing_locations_file_yields_no_rooms():
    assert rooms_from_locations({}) == []
    assert rooms_from_locations(None) == []


def test_greeklish_keys_never_reach_speech():
    # The TTS reads latin text as letter-salad, which is what "Βρίσκομαι στο
    # domatio tou mbamba" sounded like live.
    for room in rooms_from_locations(_locations()):
        assert not re.search(r'[a-zA-Z]', room_el(room))
        assert not re.search(r'[a-zA-Z]', room_locative_el(room))
        assert not re.search(r'[a-zA-Z]', format_location(room))


def test_articles_are_inflected_not_glued_to_one_gender():
    assert room_locative_el('kouzina') == 'στην κουζίνα'   # feminine
    assert room_locative_el('saloni') == 'στο σαλόνι'      # neuter
    assert room_locative_el('diadromos') == 'στον διάδρομο'  # masculine


def test_unknown_room_still_answers():
    # A freshly taught room should get a clumsy answer, never a crash.
    assert room_locative_el('neo domatio') == 'στο neo domatio'
    assert room_el('neo domatio') == 'neo domatio'


def test_the_llm_is_offered_exactly_the_rooms_the_map_has():
    src = open(os.path.join(_ROOT, 'home_robot', 'nodes',
                            'llm_bridge_node.py')).read()
    ns = {'ROOM_NAMES_EL': ROOM_NAMES_EL}
    exec(compile(re.search(r'^_ROOMS = .*$', src, re.M).group(0),  # noqa: S102
                 'rooms', 'exec'), ns)
    # Order is the dict's and does not matter to the model; membership does.
    assert set(ns['_ROOMS']) == set(rooms_from_locations(_locations()))


def test_the_system_prompt_names_the_rooms_of_the_map():
    src = open(os.path.join(_ROOT, 'home_robot', 'nodes',
                            'llm_bridge_node.py')).read()
    ns = {'ROOM_NAMES_EL': ROOM_NAMES_EL}
    snippet = re.search(r'^SYSTEM_PROMPT = """.*?ROOM_NAMES_EL\.items\(\)\)\)',
                        src, re.S | re.M).group(0)
    exec(compile(snippet, 'prompt', 'exec'), ns)  # noqa: S102
    prompt = ns['SYSTEM_PROMPT']
    assert '{rooms}' not in prompt, 'placeholder never got substituted'
    for room in rooms_from_locations(_locations()):
        assert f'{room}={room_el(room)}' in prompt


def test_planner_and_executor_hold_no_hardcoded_room_list():
    for rel in ('home_robot/nodes/task_planner_node.py',
                'home_robot/nodes/mission_executor_node.py'):
        src = open(os.path.join(_ROOT, rel)).read()
        for stale in ('living_room', 'bedroom', "'kitchen'"):
            assert stale not in src, f'{rel} still references {stale}'
        assert 'rooms_from_locations' in src
