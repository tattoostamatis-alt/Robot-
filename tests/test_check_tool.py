"""The `check` tool — "πήγαινε να δεις αν έκλεισα το παράθυρο".

The mission behind it (`check:<room>:<question>` in mission_executor: drive to
the room, ask the VLM, come back, say the answer) has existed since the mission
executor was written. It had no LLM tool, so the only way to trigger it was to
publish to mission/start by hand — a whole feature reachable by nobody. Same
shape as doa_node never being launched.

Robot-free: the tool table and the router are plain data.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_check_tool.py -q
"""
import ast
import os
import re

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(PKG, 'home_robot', 'nodes', 'llm_bridge_node.py')
MISSION = os.path.join(PKG, 'home_robot', 'nodes', 'mission_executor_node.py')

import sys
sys.path.insert(0, PKG)
from home_robot.tool_router import select_tools          # noqa: E402


def _tools():
    """The TOOLS table, lifted out of the node without importing rclpy.

    Not literal_eval: the table references _ROOMS/_APPS, so the assignments it
    depends on are executed too. Same extractor the benches use, so there is
    one way of doing this rather than two that can drift.
    """
    sys.path.insert(0, os.path.join(PKG, 'scripts'))
    from tool_ab import extract
    return extract({'SYSTEM_PROMPT', '_ROOMS', '_APPS', 'TOOLS'})['TOOLS']


def _tool(name):
    for t in _tools():
        if t['function']['name'] == name:
            return t['function']
    return None


# ── the tool exists and is shaped right ─────────────────────────────────────

def test_the_check_tool_is_declared():
    fn = _tool('check')
    assert fn, ('no `check` tool — the check mission stays unreachable by '
                'voice, which is how it sat unused since it was written')


def test_it_takes_a_room_and_an_optional_question():
    fn = _tool('check')
    props = fn['parameters']['properties']
    assert 'room' in props and 'question' in props
    assert fn['parameters'].get('required') == ['room'], (
        'room must be required and question optional — "πήγαινε να δεις στην '
        'κουζίνα" is a complete instruction on its own')


def test_the_description_stays_short():
    """max_prefill_len is 4096 on the NPU; tool text is re-read every call."""
    fn = _tool('check')
    assert len(fn['description']) < 220, 'description is too long for the prefill'


def test_it_is_distinguished_from_looking_at_the_current_room():
    """Otherwise "τι βλέπεις;" drives the robot to another room and back."""
    fn = _tool('check')
    d = fn['description']
    assert 'ΟΧΙ' in d or 'ΑΛΛΟ' in d, (
        'nothing in the description stops this being chosen for "τι βλέπεις;"')


# ── dispatch ────────────────────────────────────────────────────────────────

def test_dispatch_publishes_the_mission_string():
    src = open(BRIDGE).read()
    m = re.search(r"elif name == 'check':(.*?)elif name ==", src, re.S)
    assert m, 'check is not dispatched'
    body = m.group(1)
    assert "f'check:{room}:{question}'" in body, \
        'the mission string does not match what mission_executor parses'
    assert 'self.locations' in body, \
        'an unknown room must be rejected before a mission is started'


def test_a_colon_in_the_question_cannot_break_the_mission_string():
    """mission_executor splits on ':' — a question containing one would be
    cut in half and the robot would ask something different from what it was
    asked."""
    src = open(BRIDGE).read()
    m = re.search(r"elif name == 'check':(.*?)elif name ==", src, re.S)
    assert "replace(':'" in m.group(1), 'the question is not sanitised'


def test_the_mission_side_still_parses_that_format():
    """Both ends of the string, checked against each other."""
    src = open(MISSION).read()
    assert re.search(r"cmd\.startswith\('check:'\)", src), \
        'mission_executor no longer accepts check:'
    assert re.search(r"partition\(':'\)", src), \
        'the room/question split changed shape'


# ── routing (FLM path only sends the tools the router picks) ─────────────────

_ALL = [{'function': {'name': n}} for n in
        ('check', 'goto', 'tidy', 'move', 'system_status', 'follow', 'remember')]


@pytest.mark.parametrize('utterance', [
    'πήγαινε να δεις αν έκλεισα το παράθυρο στην κουζίνα',
    'τσέκαρε αν είναι κλειστή η πόρτα',
    'έλεγξε το σαλόνι',
    'ξέχασα ανοιχτό το φως στο δωμάτιο του Μαξ;',
    'δες αν είναι ανοιχτό το παράθυρο',
])
def test_check_phrasings_reach_the_model(utterance):
    names = {t['function']['name'] for t in select_tools(utterance, _ALL)}
    assert 'check' in names, (
        f'{utterance!r} routes to {sorted(names)} — the check tool never '
        'reaches the model, so the robot cannot act on it')


@pytest.mark.parametrize('utterance', [
    'τι ώρα είναι;',
    'γεια σου Μαξ',
])
def test_conversation_does_not_pull_in_the_check_tool(utterance):
    names = {t['function']['name'] for t in select_tools(utterance, _ALL)}
    assert 'check' not in names, f'{utterance!r} should not route to check'
