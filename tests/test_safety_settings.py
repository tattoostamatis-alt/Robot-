"""The safety tab's knobs must point at parameters that exist and are honoured.

A slider that writes a parameter nobody reads is worse than no slider: the page
says the robot now stops at 0.8 m, the robot still stops at 0.5 m, and the
person testing it believes the page. Three ways that happens, one test each:

  * the parameter name is wrong or was renamed -> nothing to write to;
  * the owning node reads the parameter once in __init__ and has no
    set-parameters callback -> the write succeeds and changes nothing;
  * the number shown as a fixed limit drifts from what nav2_params.yaml says.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_safety_settings.py -q
"""
import json
import re
from pathlib import Path

import pytest

from home_robot import safety_settings as ss

_ROOT = Path(__file__).resolve().parents[1]
_NODES = _ROOT / 'home_robot' / 'nodes'
_OBSTACLE = (_NODES / 'obstacle_safety_node.py').read_text()
_ROOMBA = (_NODES / 'roomba_driver.py').read_text()
_DASH = (_NODES / 'web_dashboard_node.py').read_text()
_NAV2 = (_ROOT / 'config' / 'nav2_params.yaml').read_text()


# ── the knobs point at real parameters ───────────────────────────────────────

_OWN_SOURCE = {
    '/obstacle_safety_node': _OBSTACLE,
    '/roomba_driver': _ROOMBA,
}


def test_every_target_parameter_is_declared_by_its_node():
    """Our own nodes must declare what we write. The Nav2 costmaps are not in
    this repo, so they are checked against nav2_params.yaml instead (below)."""
    for spec in ss.SPECS:
        for node, param in spec.targets:
            src = _OWN_SOURCE.get(node)
            if src is None:
                continue
            assert f"declare_parameter('{param}'" in src, \
                f'{node} does not declare {param!r}'


def test_costmap_parameters_exist_in_nav2_params():
    for spec in ss.SPECS:
        for node, param in spec.targets:
            if 'costmap' not in node:
                continue
            leaf = param.split('.')[-1]
            assert re.search(rf'^\s*{leaf}:', _NAV2, re.M), \
                f'{leaf} is not in nav2_params.yaml — did the layer get renamed?'


def test_inflation_is_written_to_both_costmaps():
    """‼️ Writing only one leaves the planner and the controller disagreeing
    about where the walls are: a path gets planned that is then refused."""
    for key in ('inflation_radius', 'cost_scaling'):
        nodes = {node for node, _ in ss.BY_KEY[key].targets}
        assert nodes == {'/local_costmap/local_costmap',
                         '/global_costmap/global_costmap'}, key


# ── the owning nodes actually apply a runtime change ─────────────────────────

def test_obstacle_safety_node_has_a_parameter_callback():
    assert 'add_on_set_parameters_callback' in _OBSTACLE, (
        'obstacle_safety_node reads its distances once at startup again — the '
        'slider would report success and change nothing')


@pytest.mark.parametrize('param', ['safety_distance', 'center_fraction',
                                   'detection_timeout'])
def test_obstacle_safety_callback_covers_every_knob(param):
    assert f"'{param}'" in _OBSTACLE.split('_LIMITS = ', 1)[1].split('}', 1)[0], \
        f'{param} is not in obstacle_safety_node._LIMITS, so it is ignored'


@pytest.mark.parametrize('param', ['bump_stop', 'cliff_stop', 'wheel_drop_stop',
                                   'bump_block_s', 'cliff_block_s'])
def test_roomba_driver_callback_covers_the_contact_sensors(param):
    body = _ROOMBA.split('_SAFETY_FLAGS', 1)[1].split('def _drive', 1)[0]
    assert f"'{param}'" in body, \
        f'{param} is not handled in roomba_driver._on_set_parameters'


def test_turning_a_contact_sensor_off_is_logged_loudly():
    """FULL mode means these flags are the only bumper the robot has. Whoever
    turned one off must be findable in the log afterwards."""
    body = _ROOMBA.split('_SAFETY_FLAGS', 1)[1].split('def _drive', 1)[0]
    assert 'logger().warn' in body, \
        'switching a contact guard off no longer produces a warning'


# ── the fixed limits shown on the page match the file they come from ─────────

def test_info_only_values_match_nav2_params():
    stop = ss.INFO_ONLY[0]
    assert stop['key'] == 'stop_skirt'
    # The Stop zone is a circle now (2026-08-06), written as a 16-point
    # approximation, so the radius appears as the point on the +x axis rather
    # than as a box corner. Checking the radius itself also means the panel
    # cannot quietly go on quoting the old 0.22 box.
    assert f"[{stop['value']:.4f}, 0.0000]" in _NAV2, \
        ('the Stop polygon in nav2_params.yaml no longer has this radius — '
         'if the zone changed shape, update INFO_ONLY to match')
    tbc = ss.INFO_ONLY[1]
    assert re.search(rf'time_before_collision:\s*{tbc["value"]}\b', _NAV2), \
        'time_before_collision drifted from nav2_params.yaml'


def test_collision_monitor_is_not_offered_as_a_live_knob():
    """Nav2 reads polygon parameters in on_configure only. A slider for one
    would write successfully and do nothing, which is the failure this whole
    file exists to prevent."""
    for spec in ss.SPECS:
        for node, _ in spec.targets:
            assert 'collision_monitor' not in node


def test_defaults_match_what_the_nodes_ship_with():
    """Otherwise the first tick of the apply loop silently retunes the robot."""
    for spec in ss.SPECS:
        for node, param in spec.targets:
            src = _OWN_SOURCE.get(node)
            if src is None:
                continue
            m = re.search(rf"declare_parameter\('{param}',\s*([^)]+)\)", src)
            assert m, param
            shipped = m.group(1).split('#')[0].strip()
            if spec.kind == 'bool':
                assert shipped == str(spec.default), (param, shipped)
            else:
                assert float(shipped) == pytest.approx(spec.default), \
                    (param, shipped)


def test_inflation_default_matches_the_costmaps():
    radii = re.findall(r'^\s*inflation_radius:\s*([\d.]+)', _NAV2, re.M)
    assert radii, 'no inflation_radius in nav2_params.yaml'
    assert {float(r) for r in radii} == {ss.BY_KEY['inflation_radius'].default}


# ── clamping: the websocket is not a trusted input ───────────────────────────

@pytest.mark.parametrize('value,expected', [
    (0.8, 0.8),
    (99.0, 1.5),        # above hi
    (-5.0, 0.2),        # below lo
    (0.2, 0.2),
])
def test_a_double_is_pulled_into_range(value, expected):
    assert ss.clamp('stop_distance', value) == pytest.approx(expected)


@pytest.mark.parametrize('value', ['0.8', None, [], {}, True, float('nan')])
def test_a_non_number_is_rejected_rather_than_coerced(value):
    assert ss.clamp('stop_distance', value) is None


def test_an_unknown_key_is_rejected():
    assert ss.clamp('drive_into_walls', 1.0) is None


def test_a_bool_knob_takes_a_bool():
    assert ss.clamp('bump_stop', False) is False
    assert ss.clamp('bump_stop', True) is True


def test_a_bool_knob_rejects_text():
    assert ss.clamp('bump_stop', 'false') is None


def test_infinity_is_clamped_not_stored():
    assert ss.clamp('stop_distance', float('inf')) == pytest.approx(1.5)


# ── persistence ──────────────────────────────────────────────────────────────

@pytest.fixture
def state(tmp_path, monkeypatch):
    path = tmp_path / 'safety_settings.json'
    monkeypatch.setattr(ss, 'STATE_PATH', str(path))
    return path


def test_a_missing_file_gives_the_defaults(state):
    assert ss.load() == ss.defaults()


def test_a_saved_value_comes_back(state):
    ss.save({**ss.defaults(), 'stop_distance': 0.75})
    assert ss.load()['stop_distance'] == pytest.approx(0.75)


def test_a_corrupt_file_gives_the_defaults_instead_of_throwing(state):
    state.write_text('{"stop_distance": 0.7')      # truncated by a power cut
    assert ss.load() == ss.defaults()


def test_a_file_that_is_not_an_object_is_ignored(state):
    state.write_text('[1, 2, 3]')
    assert ss.load() == ss.defaults()


def test_an_out_of_range_file_cannot_disable_a_guard(state):
    """‼️ The file is on disk and editable. A hand-edited 0.0 stopping distance
    must come back as the minimum, not as "never stop"."""
    state.write_text(json.dumps({'stop_distance': 0.0, 'center_width': 900}))
    loaded = ss.load()
    assert loaded['stop_distance'] == pytest.approx(ss.BY_KEY['stop_distance'].lo)
    assert loaded['center_width'] == pytest.approx(ss.BY_KEY['center_width'].hi)


def test_unknown_keys_in_the_file_are_dropped(state):
    state.write_text(json.dumps({'stop_distance': 0.6, 'nonsense': 1}))
    assert 'nonsense' not in ss.load()


def test_a_partial_file_is_filled_in(state):
    state.write_text(json.dumps({'stop_distance': 0.6}))
    loaded = ss.load()
    assert loaded['stop_distance'] == pytest.approx(0.6)
    assert loaded['bump_stop'] is True


def test_save_is_atomic(state, monkeypatch):
    """os.replace, not a plain write — a crash mid-save must leave either the
    old file or the new one, never half of one."""
    src = (_ROOT / 'home_robot' / 'safety_settings.py').read_text()
    assert 'os.replace(' in src


# ── regrouping for the service calls ─────────────────────────────────────────

def test_targets_groups_by_node():
    grouped = ss.targets({'inflation_radius': 0.25, 'bump_stop': False})
    assert grouped['/roomba_driver'] == {'bump_stop': False}
    assert grouped['/local_costmap/local_costmap'] == \
        {'inflation_layer.inflation_radius': 0.25}
    assert grouped['/global_costmap/global_costmap'] == \
        {'inflation_layer.inflation_radius': 0.25}


def test_targets_drops_unknown_keys():
    assert ss.targets({'nonsense': 1.0}) == {}


def test_from_ros_maps_back_and_clamps():
    key, value = ss.from_ros('/obstacle_safety_node', 'safety_distance', 9.0)
    assert key == 'stop_distance'
    assert value == pytest.approx(1.5)


def test_from_ros_ignores_a_parameter_we_do_not_own():
    assert ss.from_ros('/roomba_driver', 'right_trim', 1.0) == (None, None)


# ── the dashboard wiring ─────────────────────────────────────────────────────

def test_the_dashboard_reapplies_settings_when_a_node_comes_back():
    """‼️ The dashboard usually starts before Nav2 finishes configuring, and
    roomba_driver restarts itself after a serial drop. A one-shot apply at boot
    would leave those nodes on their launch defaults with the page showing the
    user's numbers."""
    body = _DASH.split('def _safety_tick', 1)[1].split('def _safety_read', 1)[0]
    assert '_safety_applied.discard' in body, \
        'a node that goes away is never re-applied to when it returns'


def test_the_panel_shows_what_the_nodes_report_not_what_was_asked():
    body = _DASH.split('def _safety_payload', 1)[1].split('def _safety_apply', 1)[0]
    assert "'live'" in body and "'set'" in body, \
        'the payload no longer distinguishes the asked-for value from the live one'


def test_a_knob_with_no_node_answering_reports_live_none():
    body = _DASH.split('def _safety_payload', 1)[1].split('def _safety_apply', 1)[0]
    assert 'len(answered) == len(live)' in body, \
        'a half-applied knob (one costmap up, one not) would report as applied'
