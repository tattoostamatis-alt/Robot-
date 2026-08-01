"""Regression tests for the second 2026-08-01 audit pass — motion that would
not stop, and state that grew without bound.

The theme is the same as the first pass: everything here was silent.

  * task_planner_node had NO cancellation at all. llm_bridge's emergency stop
    publishes patrol_command=False and _on_patrol dropped it ("if not msg.data:
    return"), so a spoken "σταμάτα" halted the robot for a fraction of a second
    on the zeroed cmd_vel and the patrol then drove off again on its next Nav2
    goal. Its nav timeout also returned without cancelling the goal, so "λήξη
    χρόνου πλοήγησης" was a lie — Nav2 kept driving to that room.

  * doa_node turned the base toward whoever said the wake word with no check
    for whether a navigator was already driving, putting two publishers on
    cmd_vel at once.

  * explore_manager_node signalled only the `ros2 run` parent, never waited,
    and dropped the handle — orphaned explore node, zombie child, and a second
    explore on the next start.

Robot-free: ROS is stubbed, no hardware, no executor.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_cancel_regressions.py -q
"""
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The ROS stubs live next door; tests/ is not a package, so import by name off
# the path above rather than as tests.<module>.
from test_safety_regressions import (        # noqa: E402
    _BaseNode, _load, _mod, _ns, _std_msgs, _twist)


# ── task_planner_node ───────────────────────────────────────────────────────

@pytest.fixture
def planner(tmp_path):
    locs = tmp_path / 'config'
    locs.mkdir()
    (locs / 'locations.yaml').write_text(
        'saloni: {x: 1.0, y: 0.0, yaw: 0.0}\n'
        'kouzina: {x: 2.0, y: 0.0, yaw: 0.0}\n')
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'rclpy.action': _mod('rclpy.action',
                             ActionClient=lambda *a, **k: _ns(
                                 wait_for_server=lambda timeout_sec=0.0: False,
                                 send_goal_async=lambda *_: None)),
        'action_msgs': _mod('action_msgs'),
        'action_msgs.msg': _mod('action_msgs.msg',
                                GoalStatus=_ns(STATUS_SUCCEEDED=4,
                                               STATUS_CANCELED=5)),
        'ament_index_python': _mod('ament_index_python'),
        'ament_index_python.packages': _mod(
            'ament_index_python.packages',
            get_package_share_directory=lambda _: str(tmp_path)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod(
            'geometry_msgs.msg',
            Quaternion=lambda x=0.0, y=0.0, z=0.0, w=1.0: _ns(x=x, y=y, z=z, w=w)),
        'nav2_msgs': _mod('nav2_msgs'),
        'nav2_msgs.action': _mod(
            'nav2_msgs.action',
            NavigateToPose=_ns(Goal=lambda: _ns(
                pose=_ns(header=_ns(frame_id='', stamp=None),
                         pose=_ns(position=_ns(x=0.0, y=0.0, z=0.0),
                                  orientation=None))))),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _std_msgs(),
    }
    mod = _load(f'{PKG}/home_robot/nodes/task_planner_node.py', mods)
    return mod, mod.TaskPlannerNode()


def test_patrol_false_cancels_a_running_task(planner):
    """‼️ THE BUG: this is the signal llm_bridge's emergency stop sends, and it
    was discarded, so "σταμάτα" could not stop a patrol."""
    mod, node = planner
    node._busy.acquire()                 # pretend a task is running
    node._on_patrol(_ns(data=False))
    assert node._cancel.is_set()


def test_spoken_stop_cancels_a_running_task(planner):
    """Works even when llm_bridge is down and never relays the cancel."""
    mod, node = planner
    node._busy.acquire()
    node._on_speech_text(_ns(data='σταμάτα'))
    assert node._cancel.is_set()


def test_spoken_stop_is_ignored_when_idle(planner):
    """Must not arm the flag for the NEXT task."""
    mod, node = planner
    node._on_speech_text(_ns(data='σταμάτα'))
    assert not node._cancel.is_set()


def test_tidy_cancel_payload_cancels(planner):
    mod, node = planner
    node._busy.acquire()
    node._on_tidy(_ns(data='{"cancel": true}'))
    assert node._cancel.is_set()


def test_patrol_stops_visiting_rooms_once_cancelled(planner):
    """The loop must actually break, not just set a flag."""
    mod, node = planner
    visited = []
    node._visit_and_report = lambda room: visited.append(room)
    node._cancel.set()
    node._patrol_run(['saloni', 'kouzina'])
    assert visited == []
    spoken = [m.data for m in node.pubs['speech_response'].sent]
    assert any('Σταμάτησα' in s for s in spoken)


def test_nav_timeout_cancels_the_goal(planner):
    """‼️ THE BUG: the timeout abandoned the future but left the goal ACTIVE,
    so Nav2 kept driving to that room while the planner moved on."""
    mod, node = planner
    node.nav_timeout = 0.3
    cancelled = []
    gh = _ns(accepted=True,
             get_result_async=lambda: _ns(done=lambda: False, result=lambda: None),
             cancel_goal_async=lambda: cancelled.append(True))
    node.nav_client = _ns(wait_for_server=lambda timeout_sec=0.0: True,
                          send_goal_async=lambda _g: 'fut')
    node._await_future = lambda fut, t: gh

    ok, reason = node._navigate({'x': 1.0, 'y': 0.0, 'yaw': 0.0})
    assert ok is False
    assert cancelled, 'the goal was abandoned without being cancelled'


def test_cancel_during_navigation_cancels_the_goal(planner):
    mod, node = planner
    cancelled = []
    gh = _ns(accepted=True,
             get_result_async=lambda: _ns(done=lambda: False, result=lambda: None),
             cancel_goal_async=lambda: cancelled.append(True))
    node.nav_client = _ns(wait_for_server=lambda timeout_sec=0.0: True,
                          send_goal_async=lambda _g: 'fut')
    node._await_future = lambda fut, t: gh
    node._cancel.set()

    ok, reason = node._navigate({'x': 1.0, 'y': 0.0, 'yaw': 0.0})
    assert (ok, reason) == (False, 'ακυρώθηκε')
    assert cancelled


def test_stale_detections_are_not_reported_as_this_rooms_clutter(planner):
    """A detector that went quiet left the PREVIOUS room's objects in place."""
    import time as _t
    mod, node = planner
    node.detect_wait = 0.0
    node._latest_objects = [{'label': 'κούπα', 'clutter': True}]
    node._objects_rx = _t.monotonic() - 60.0        # ancient
    assert node._check_clutter() == []


def test_fresh_detections_are_still_reported(planner):
    import time as _t
    mod, node = planner
    node.detect_wait = 0.05
    node._latest_objects = [{'label': 'κούπα', 'clutter': True}]
    node._objects_rx = _t.monotonic()
    assert node._check_clutter() == ['κούπα']


# ── doa_node ────────────────────────────────────────────────────────────────

@pytest.fixture
def doa(monkeypatch):
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg', Twist=_twist()),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _std_msgs(),
        'usb': _mod('usb'), 'usb.core': _mod('usb.core', find=lambda **k: None),
        'usb.util': _mod('usb.util'),
    }
    mod = _load(f'{PKG}/home_robot/nodes/doa_node.py', mods)
    # The constructor starts a USB poll thread. Neutralise it ONLY for the
    # construction: threading.Timer is a Thread subclass that calls the module
    # global by name, so leaving Thread patched breaks the listen timer.
    with monkeypatch.context() as m:
        m.setattr(mod.threading, 'Thread',
                  lambda *a, **k: _ns(start=lambda: None))
        node = mod.DoaNode()
    return mod, node


def _record_turns(node):
    """Replace the rotation with a recorder; the real Thread still runs it."""
    turns = []
    node._rotate_toward = lambda angle: turns.append(angle)
    return turns


def test_wake_word_does_not_turn_the_base_while_navigating(doa):
    """‼️ THE BUG: the only guard was "not already rotating", so a wake word
    mid-goto put this node and Nav2's controller on cmd_vel simultaneously."""
    import time as _t
    mod, node = doa
    turns = _record_turns(node)
    node._on_drive_cmd(_ns(linear=_ns(x=0.2, y=0.0, z=0.0),
                           angular=_ns(x=0.0, y=0.0, z=0.0)))
    node._last_angle = 90.0
    node._on_wake_word(_ns(data='hey robot'))
    _t.sleep(0.15)

    assert turns == [], 'the base was turned while a navigator was driving'


def test_wake_word_turns_the_base_when_idle(doa):
    """The feature still works when nothing else owns the base."""
    import time as _t
    mod, node = doa
    turns = _record_turns(node)
    node._last_angle = 90.0
    node._on_wake_word(_ns(data='hey robot'))
    _t.sleep(0.15)

    assert turns, 'the wake-word turn stopped working entirely'


def test_zero_drive_commands_do_not_count_as_busy(doa):
    """A stopped robot still publishes zeros; that must not block the turn."""
    mod, node = doa
    node._on_drive_cmd(_ns(linear=_ns(x=0.0, y=0.0, z=0.0),
                           angular=_ns(x=0.0, y=0.0, z=0.0)))
    assert node._base_is_busy() is False


# ── explore_manager_node ────────────────────────────────────────────────────

@pytest.fixture
def explore():
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _std_msgs(),
        'ament_index_python': _mod('ament_index_python'),
        'ament_index_python.packages': _mod(
            'ament_index_python.packages',
            get_package_share_directory=lambda _: '/tmp'),
    }
    mod = _load(f'{PKG}/home_robot/nodes/explore_manager_node.py', mods)
    return mod, mod.ExploreManager()


class _FakeProc:
    """A live child. `ignores_sigterm` makes the FIRST wait() time out, so the
    node has to escalate; later waits succeed, as a killed process would."""

    def __init__(self, ignores_sigterm=False):
        self.pid = 4242
        self._alive = True
        self.ignores_sigterm = ignores_sigterm
        self.signals = []
        self.waits = 0

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self.waits += 1
        if self.ignores_sigterm and self.waits == 1:
            import subprocess
            raise subprocess.TimeoutExpired('explore', timeout or 0)
        self._alive = False
        return 0

    def terminate(self):
        self.signals.append('TERM')

    def kill(self):
        self.signals.append('KILL')
        self._alive = False


def test_stop_signals_the_whole_process_group(explore, monkeypatch):
    """‼️ THE BUG: only the `ros2 run` parent was signalled, so the explore node
    it spawned kept running — and kept driving."""
    mod, node = explore
    killed = []
    monkeypatch.setattr(mod.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(mod.os, 'killpg',
                        lambda pgid, sig: killed.append((pgid, sig)))
    proc = _FakeProc()                     # exits promptly on SIGTERM
    node._proc = proc
    node._stop()

    assert killed and killed[0] == (4242, mod.signal.SIGTERM)
    assert proc.waits, 'the child was never reaped (zombie)'


def test_stop_escalates_to_sigkill(explore, monkeypatch):
    mod, node = explore
    killed = []
    monkeypatch.setattr(mod.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(mod.os, 'killpg',
                        lambda pgid, sig: killed.append((pgid, sig)))
    node._proc = _FakeProc(ignores_sigterm=True)   # never exits
    node._stop()

    sent = [sig for _, sig in killed]
    assert mod.signal.SIGTERM in sent and mod.signal.SIGKILL in sent


def test_stop_clears_the_handle_so_a_restart_is_possible(explore, monkeypatch):
    mod, node = explore
    monkeypatch.setattr(mod.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(mod.os, 'killpg', lambda pgid, sig: None)
    node._proc = _FakeProc()
    node._stop()
    assert node._proc is None


def test_stop_on_nothing_running_is_harmless(explore):
    mod, node = explore
    node._proc = None
    node._stop()          # must not raise
    assert node._proc is None
