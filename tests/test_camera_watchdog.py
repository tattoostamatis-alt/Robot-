"""Tests for the RealSense watchdog (home_robot/nodes/camera_watchdog_node.py).

Written 2026-08-01 after the driver segfaulted (exit -11) during a bringup and
the only visible symptom was object_detector logging "waiting for camera
topics" — the robot was blind and every other reading pointed elsewhere.

Robot-free: ROS and subprocess are stubbed.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_camera_watchdog.py -q
"""
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_safety_regressions import _BaseNode, _load, _mod, _ns  # noqa: E402


@pytest.fixture
def cam(monkeypatch):
    launched = []

    class FakePopen:
        def __init__(self, args, **kw):
            launched.append((args, kw))

    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'rclpy.qos': _mod('rclpy.qos',
                          QoSProfile=lambda **k: _ns(**k),
                          QoSReliabilityPolicy=_ns(BEST_EFFORT='best_effort',
                                                   RELIABLE='reliable')),
        'sensor_msgs': _mod('sensor_msgs'),
        'sensor_msgs.msg': _mod('sensor_msgs.msg', Image=object),
    }
    mod = _load(f'{PKG}/home_robot/nodes/camera_watchdog_node.py', mods)
    monkeypatch.setattr(mod.subprocess, 'Popen', FakePopen)
    node = mod.CameraWatchdog()
    return mod, node, launched


def _age(node, seconds):
    """Pretend the last frame arrived `seconds` ago."""
    import time
    node._last_frame = time.monotonic() - seconds


# ── it must not fire before the camera has ever worked ───────────────────────

def test_a_camera_still_starting_is_not_restarted(cam):
    """A cold start plus USB enumeration is genuinely slow. Restarting a camera
    that is still coming up is how a watchdog turns a slow boot into a loop."""
    mod, node, launched = cam
    assert node._last_frame is None
    for _ in range(30):
        node._check()
    assert launched == []


def test_a_camera_that_never_streams_is_restarted(cam):
    """‼️ THE BUG, caught on the very next bringup after shipping this node:
    the driver segfaults DURING startup (exit -11), before publishing anything.
    The first version refused to arm until it had seen a frame, so it sat there
    doing nothing through exactly the failure it was written for."""
    import time
    mod, node, launched = cam
    node._started_at = time.monotonic() - 300.0
    assert node._last_frame is None

    node._check()

    assert len(launched) == 1, 'a camera that never started was left dead' 


def test_a_streaming_camera_is_left_alone(cam):
    mod, node, launched = cam
    node._on_frame(None)
    node._check()
    assert launched == []


# ── the case it exists for ───────────────────────────────────────────────────

def test_a_dead_camera_is_restarted(cam):
    mod, node, launched = cam
    node._on_frame(None)
    _age(node, 60.0)

    node._check()

    assert len(launched) == 1
    args = launched[0][0]
    assert args[:2] == ['ros2', 'launch']
    assert 'realsense2_camera' in args


def test_the_restart_keeps_the_perception_arguments(cam):
    """Relaunching without align_depth/pointcloud would bring the camera back
    while leaving the object detector and the 3D tab dead — the confusing
    half-recovery this is meant to prevent."""
    mod, node, launched = cam
    node._on_frame(None)
    _age(node, 60.0)
    node._check()

    args = ' '.join(launched[0][0])
    assert 'align_depth.enable:=true' in args
    assert 'pointcloud.enable:=true' in args


def test_it_survives_the_relaunch_being_killed(cam):
    """start_new_session, or the restart dies with whatever killed the node."""
    mod, node, launched = cam
    node._on_frame(None)
    _age(node, 60.0)
    node._check()
    assert launched[0][1].get('start_new_session') is True


# ── it must not become the problem ───────────────────────────────────────────

def test_a_restart_is_given_time_to_come_up(cam):
    """The driver needs ~10 s to stream. Without the grace period the watchdog
    fires again while its own restart is still starting."""
    mod, node, launched = cam
    node._on_frame(None)
    _age(node, 60.0)
    node._check()
    for _ in range(20):          # the camera is still silent, still starting
        _age(node, 60.0)
        node._check()
    assert len(launched) == 1, 'restart storm'


def test_restarts_are_capped(cam):
    """An unplugged camera cannot be fixed by relaunching it forever, and the
    log noise would bury the reason."""
    import time
    mod, node, launched = cam
    node._on_frame(None)
    for _ in range(10):
        _age(node, 60.0)
        node._blocked_until = time.monotonic() - 1     # grace already elapsed
        node._check()
    assert len(launched) == node.max_restarts


def test_restart_can_be_switched_off(cam):
    mod, node, launched = cam
    node.enable_restart = False
    node._on_frame(None)
    _age(node, 60.0)

    node._check()

    assert launched == []


def test_recovery_is_announced(cam):
    """"Camera is back" is what tells you the restart worked, rather than
    leaving the last word in the log as an error."""
    mod, node, launched = cam
    node._on_frame(None)
    _age(node, 60.0)
    node._check()
    assert node._reported_dead is True

    node._on_frame(None)                 # frames return

    assert node._reported_dead is False


def test_it_subscribes_best_effort(cam):
    """Images are BEST_EFFORT. A RELIABLE subscription matches nothing, and the
    watchdog would declare a perfectly healthy camera dead every single time."""
    mod, node, _ = cam
    src = open(f'{PKG}/home_robot/nodes/camera_watchdog_node.py').read()
    assert 'BEST_EFFORT' in src


# ── depth dies on its own while colour keeps streaming (2026-08-04) ──────────
# Measured live: colour ran at 30 Hz, /camera/camera/depth/image_rect_raw
# published nothing at all, the pointcloud topic had a publisher and zero
# messages, and the 3D tab sat empty. The watchdog watched colour only, so it
# reported a healthy camera the whole time.

def _depth_cli(node):
    """Give the node a param client that records the calls it makes."""
    calls = []
    node._depth_cli = _ns(service_is_ready=lambda: True,
                          call_async=lambda req: calls.append(req))
    return calls


def _depth_age(node, seconds):
    import time
    node._last_depth = time.monotonic() - seconds


def test_depth_stalling_alone_is_noticed(cam):
    """‼️ THE BUG. Colour is fine, so _check() is happy; only depth is gone."""
    mod, node, launched = cam
    node._on_frame(None)
    node._on_depth(None)
    _depth_age(node, 60.0)
    _depth_cli(node)

    node._check()                        # colour path: sees nothing wrong
    node._check_depth()

    assert launched == [], 'depth is repaired in place, not by a relaunch'
    assert node._depth_reported is True


def test_depth_is_toggled_off_then_back_on(cam):
    """The fix that worked by hand: enable_depth false, then true. A relaunch
    would drop the colour stream that object detection is riding on."""
    import time
    mod, node, launched = cam
    node._on_frame(None)
    node._on_depth(None)
    _depth_age(node, 60.0)
    calls = _depth_cli(node)

    node._check_depth()
    assert len(calls) == 1
    assert calls[0].parameters[0].name == 'enable_depth'
    assert calls[0].parameters[0].value.bool_value is False

    node._depth_on_at = time.monotonic() - 0.1     # the 2 s wait elapses
    node._check_depth()

    assert len(calls) == 2
    assert calls[1].parameters[0].value.bool_value is True


def test_healthy_depth_is_left_alone(cam):
    mod, node, launched = cam
    node._on_frame(None)
    node._on_depth(None)
    calls = _depth_cli(node)

    node._check_depth()

    assert calls == []


def test_a_dead_camera_is_not_also_depth_repaired(cam):
    """When the whole camera is down the relaunch owns the recovery. Both
    firing at once would toggle a parameter on a driver that is being replaced."""
    mod, node, launched = cam
    node._on_frame(None)
    node._on_depth(None)
    _age(node, 60.0)
    _depth_age(node, 60.0)
    calls = _depth_cli(node)

    node._check_depth()

    assert calls == [], 'depth recovery ran while the camera itself was dead'


def test_depth_recovery_is_rate_limited(cam):
    """Toggling every second while the driver is coming back up would keep it
    from ever coming back up."""
    mod, node, launched = cam
    node._on_frame(None)
    node._on_depth(None)
    _depth_age(node, 60.0)
    calls = _depth_cli(node)

    node._check_depth()
    node._depth_on_at = 0.0              # ignore the scheduled turn-on
    node._check_depth()
    node._check_depth()

    assert len(calls) == 1


def test_depth_recovery_can_be_switched_off(cam):
    mod, node, launched = cam
    node.depth_recover = False
    node._on_frame(None)
    node._on_depth(None)
    _depth_age(node, 60.0)
    calls = _depth_cli(node)

    node._check_depth()

    assert calls == []
    assert node._depth_reported is True, 'it must still say so in the log'


def test_depth_returning_is_announced(cam):
    mod, node, launched = cam
    node._on_frame(None)
    node._on_depth(None)
    _depth_age(node, 60.0)
    _depth_cli(node)
    node._check_depth()
    assert node._depth_reported is True

    node._on_depth(None)

    assert node._depth_reported is False


def test_it_subscribes_to_the_raw_depth_topic(cam):
    """aligned_depth_to_color is turned on and off at runtime by the 3D map tab,
    so watching it would read a deliberate change as a failure."""
    mod, node, _ = cam
    topics = [t for t, _cb in node.subs]
    assert '/camera/camera/depth/image_rect_raw' in topics
