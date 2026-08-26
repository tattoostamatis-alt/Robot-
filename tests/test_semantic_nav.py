"""Tests for goto_object — driving to a THING instead of a taught place.

"Πήγαινε στην καρέκλα" used to be unanswerable. `goto` takes an enum of rooms
and waypoints someone taught by hand, and `find:<label>` walks every room in
turn looking through the camera — while object_memory_node had been recording
the map-frame position of every detected object the whole time and nothing on
the voice path could reach it. `fetch` was the only consumer, and it goes on to
grasp the thing, which is not what "go and stand next to it" means.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_semantic_nav.py -q
"""
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MISSION = (_ROOT / 'home_robot' / 'nodes' / 'mission_executor_node.py').read_text()
_BRIDGE = (_ROOT / 'home_robot' / 'nodes' / 'llm_bridge_node.py').read_text()

sys.path.insert(0, str(_ROOT))
from home_robot.tool_router import select_tools  # noqa: E402


def _tools(names):
    return [{'type': 'function', 'function': {'name': n}} for n in names]


ALL = _tools(['tidy', 'goto', 'goto_object', 'dock', 'move', 'patrol',
              'follow', 'fetch', 'pick', 'remember', 'distance_to'])


# ── the voice path reaches it ───────────────────────────────────────────────

def test_the_tool_exists_and_takes_an_object():
    m = re.search(r"'name': 'goto_object',(.*?)\}\},", _BRIDGE, re.S)
    assert m, 'the goto_object tool is gone'
    assert "'object'" in m.group(1), 'goto_object does not take an object'


def test_going_somewhere_offers_both_kinds_of_destination():
    """Only the model can tell "στο σαλόνι" (a room) from "στην καρέκλα" (a
    thing), so the router has to put both tools in front of it."""
    for phrase in ('Πήγαινε στην καρέκλα', 'Πήγαινε στο σαλόνι',
                   'πηγαινε στην τηλεοραση'):
        names = [t['function']['name'] for t in select_tools(phrase, ALL)]
        assert 'goto_object' in names, f'{phrase!r} cannot reach goto_object'
        assert 'goto' in names, f'{phrase!r} lost plain goto'


def test_the_router_still_keeps_navigation_narrow():
    """The TOOLS block is the prefill cost — the reason tool_router exists at
    all. Adding goto_object must not drag the whole set back in."""
    names = select_tools('Πήγαινε στην καρέκλα', ALL)
    assert len(names) <= 5, f'navigation now sends {len(names)} tools'


def test_the_handler_starts_the_mission():
    m = re.search(r"elif name == 'goto_object':(.*?)\n\n", _BRIDGE, re.S)
    assert m, 'the goto_object handler is gone'
    assert "goto_object:{obj}" in m.group(1), \
        'the handler does not publish a goto_object mission'
    assert 'no object specified' in m.group(1), \
        'an empty object is not rejected'


def test_the_mission_is_dispatched():
    assert re.search(r"cmd\.startswith\('goto_object:'\)", _MISSION), \
        'mission_executor does not dispatch goto_object'
    # 'fetch:' is a prefix of nothing here, but 'goto_object:' must be tested
    # BEFORE any shorter prefix that could swallow it.
    assert _MISSION.index("startswith('goto_object:')") > _MISSION.index("startswith('fetch:')")


# ── what it does when it gets there ─────────────────────────────────────────

def _mission_body():
    m = re.search(r'def _mission_goto_object\(self, label.*?(?=\n    # ──)', _MISSION, re.S)
    assert m, '_mission_goto_object is gone'
    return m.group(0)


def test_it_asks_memory_before_walking_the_flat():
    """The whole point: object memory already knows, so a remembered object is
    one navigation rather than a tour of every room."""
    body = _mission_body()
    assert '_resolve_location' in body, \
        'the mission does not consult object memory — it would search blindly'


def test_it_parks_short_of_the_object():
    """Driving at the remembered point puts the object under the bumper."""
    body = _mission_body()
    assert 'approach_pose' in body, 'it navigates straight at the object'
    assert '_navigate_to_xy' in body, 'it does not navigate to the approach pose'
    assert 'dist=self._fetch_approach_dist' in body, \
        'the approach distance is no longer parameterized'


def test_the_default_approach_distance_is_400mm():
    assert "self.declare_parameter('fetch_approach_dist', 0.4)" in _MISSION, \
        'goto_object/fetch no longer default to a 0.4 m stand-off'


def test_it_does_not_try_to_grasp():
    """This is 'go and stand next to it'. Picking is fetch's job, and doing it
    here would be a surprise with an arm attached."""
    body = _mission_body()
    for grabby in ('_do_pick', '_do_place', 'pick_command'):
        assert grabby not in body, f'goto_object calls {grabby}'


def test_a_cancel_is_honoured_before_it_sets_off():
    body = _mission_body()
    assert '_cancel_flag' in body, 'the mission cannot be cancelled'


def test_failure_is_reported_not_swallowed():
    body = _mission_body()
    assert body.count('State.FAILED') >= 2, \
        'not knowing where it is, and not getting there, must both be reported'
