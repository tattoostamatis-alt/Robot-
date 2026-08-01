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


# ── close_app ────────────────────────────────────────────────────────────────
# Added 2026-08-01: the user asked for "κλείσε όλα" by voice and nothing
# existed. The two properties worth defending are that it never escalates to
# SIGKILL (unsaved work) and never signals its own ancestors (killing the
# terminal that owns the robot launch).

def test_close_app_refuses_unknown_and_lists_alternatives():
    r = desktop.close_app('spotify')
    assert r['status'] == 'error'
    assert 'spotify' in r['reason']
    assert r['available'] == desktop.installed_apps()


def test_close_app_enum_includes_all():
    """The tool's enum must offer 'all' or "κλείσε όλα" cannot be expressed."""
    apps_enum = _node_const('_APPS')
    src = open(NODE_SRC, encoding='utf-8').read()
    assert "'app': {'type': 'string', 'enum': _APPS + ['all']}" in src
    assert 'all' not in apps_enum        # 'all' is not a launchable app


def test_ancestors_includes_the_parent_process():
    assert os.getppid() in desktop._ancestors()


def test_an_ancestor_is_never_a_target(monkeypatch):
    """The regression this exists for: 'terminal' is closable, and the robot's
    own launch is often a child of one. Signalling it kills the stack that is
    answering."""
    parent = os.getppid()

    class FakePs:
        stdout = f'{parent} gnome-terminal-\n999999 gnome-terminal-\n'

    monkeypatch.setattr(desktop.subprocess, 'run', lambda *a, **k: FakePs())
    pids = desktop._app_pids('terminal', desktop._ancestors())
    assert parent not in pids
    assert 999999 in pids


def test_close_app_actually_terminates_a_live_process(monkeypatch):
    """End to end against a real process: SIGTERM, and it is gone."""
    import subprocess as sp
    proc = sp.Popen(['sleep', '30'])
    monkeypatch.setattr(desktop, '_app_pids',
                        lambda app, skip: [proc.pid] if app == 'calculator' else [])
    try:
        r = desktop.close_app('calculator', grace=5.0)
        assert r['status'] == 'ok'
        assert r['closed'] == ['calculator'], r
        assert r['still_open'] == []
        assert proc.poll() is not None      # really dead, not just reported
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_a_survivor_is_reported_not_sigkilled(monkeypatch):
    """An app holding a "save your work?" dialog stays open and is reported as
    such. Escalating to -9 here would throw the user's work away."""
    signals = []
    monkeypatch.setattr(desktop, '_app_pids', lambda app, skip: [4242])
    monkeypatch.setattr(desktop.os, 'kill',
                        lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(desktop, '_alive', lambda pid: True)

    r = desktop.close_app('text_editor', grace=0.5)
    assert r['still_open'] == ['text_editor']
    assert r['closed'] == []
    assert signals == [desktop.signal.SIGTERM]
    assert desktop.signal.SIGKILL not in signals


def test_closing_something_not_running_is_not_an_error(monkeypatch):
    monkeypatch.setattr(desktop, '_app_pids', lambda app, skip: [])
    r = desktop.close_app('all')
    assert r['status'] == 'ok'
    assert r['closed'] == [] and r['still_open'] == []


def test_a_reaped_zombie_counts_as_closed():
    """Regression: apps opened by open_app() are our own children, so SIGTERM
    leaves a zombie — and signal 0 to a zombie succeeds. _alive() said True
    forever and close_app reported a closed app as 'still_open'."""
    import subprocess as sp
    proc = sp.Popen(['sleep', '30'])
    proc.terminate()
    proc.wait()                       # now a reaped/zombie pid
    assert desktop._alive(proc.pid) is False
