"""Tests for desktop-session access (home_robot/desktop.py).

Robot-free and display-free: nothing here launches an app or needs a GNOME
session, so it runs the same on the robot and in CI.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_desktop.py -q
"""
import ast
import os

from home_robot import desktop

NODE_SRC = os.path.join(os.path.dirname(__file__), os.pardir,
                        'home_robot', 'nodes', 'llm_bridge_node.py')


def _node_const(name):
    tree = ast.parse(open(NODE_SRC, encoding='utf-8').read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and getattr(node.targets[0], 'id', None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not found in llm_bridge_node.py')


def test_apps_enum_matches_the_launcher():
    """The tool's enum is a literal (scripts/tool_ab.py execs that assignment
    alone, so it cannot reference desktop.py) — this is what keeps it honest."""
    assert _node_const('_APPS') == sorted(desktop.KNOWN_APPS)


def test_every_known_app_maps_to_a_desktop_entry():
    for name, entry in desktop.KNOWN_APPS.items():
        assert entry.endswith('.desktop'), name


def test_installed_apps_are_a_subset_of_known():
    assert set(desktop.installed_apps()) <= set(desktop.KNOWN_APPS)


def test_unknown_app_is_refused_and_lists_alternatives():
    r = desktop.open_app('spotify')
    assert r['status'] == 'error'
    assert 'unknown app' in r['reason']
    assert r['available'] == desktop.installed_apps()


def test_exec_command_strips_field_codes(tmp_path):
    p = tmp_path / 'x.desktop'
    p.write_text('[Desktop Entry]\nName=X\nExec=someapp --flag %U\n')
    assert desktop.exec_command(str(p)) == ['someapp', '--flag']


def test_comm_lookup_is_truncated_to_15_chars():
    """`ps -o comm=` truncates at 15, so an exact compare silently never
    matches: a visibly open gnome-calculator listed as nothing until this."""
    assert desktop._COMM['gnome-calculato'] == 'calculator'
    assert all(len(k) <= 15 for k in desktop._COMM)


def test_list_windows_reports_partial():
    """Native Wayland windows cannot be enumerated (GNOME refuses
    Introspect.GetWindows), so callers must never read this as exhaustive."""
    r = desktop.list_windows()
    assert r['status'] in ('ok', 'error')
    if r['status'] == 'ok':
        assert r['partial'] is True
        assert r['count'] == len(r['windows'])
