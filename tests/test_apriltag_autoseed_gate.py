"""Tests for the AprilTag auto-seed gate (apriltag_relocalizer.py).

Why this exists: auto_seed re-published /initialpose every 8 s for as long as
the tag stayed in view, with nothing checking whether the pose was already
right. Measured in the flat 2026-07-31 with the robot PARKED in front of the
tag, consecutive fixes wandered over 16 deg of yaw — and every one of them
reset AMCL's particle filter, so the laser scan visibly swung off the walls on
a timer. It also fought the localization watchdog: the watchdog snapped the
scan onto the walls, the tag yanked it back, once per cooldown, forever.

So a periodic re-lock now has to earn its disruption:

  * below the thresholds the tag agrees with the pose -> stay quiet
  * beyond them (robot actually carried/lost) -> seed, that is the whole point
  * an explicit ~/relocalize IGNORES the gate — asking for it means apply it
  * with no current pose to compare against, seed rather than skip
  * yaw alone is enough to fire: a robot rotated in place sits at 0 m shift
  * the interval restarts even when the gate blocks, or a parked robot
    re-evaluates on every single detection frame instead of every 8 s

Robot-free: numpy is real, ROS is stubbed, no camera or tag hardware.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_apriltag_autoseed_gate.py -q
"""
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = f'{PKG}/home_robot/nodes/apriltag_relocalizer.py'

STUB_MODULES = ('rclpy', 'rclpy.node', 'rclpy.time', 'rclpy.duration',
                'geometry_msgs', 'geometry_msgs.msg',
                'apriltag_msgs', 'apriltag_msgs.msg',
                'std_srvs', 'std_srvs.srv', 'tf2_ros')


def _install_stubs():
    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def _cov_msg(**kw):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=None, frame_id=''),
            pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)),
                covariance=[0.0] * 36))

    class _Node:
        def __init__(self, name):
            pass

    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_Node),
        'rclpy.time': _mod('rclpy.time', Time=lambda *a, **k: None),
        'rclpy.duration': _mod('rclpy.duration',
                               Duration=lambda **k: types.SimpleNamespace(**k)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg',
                                  PoseWithCovarianceStamped=_cov_msg,
                                  TransformStamped=object),
        'apriltag_msgs': _mod('apriltag_msgs'),
        'apriltag_msgs.msg': _mod('apriltag_msgs.msg',
                                  AprilTagDetectionArray=object),
        'std_srvs': _mod('std_srvs'),
        'std_srvs.srv': _mod('std_srvs.srv', Empty=object),
        'tf2_ros': _mod('tf2_ros', Buffer=lambda: None,
                        TransformListener=lambda *a, **k: None),
    }
    sys.modules.update(mods)


@pytest.fixture(autouse=True)
def _isolate_stubs():
    saved = {n: sys.modules.get(n) for n in STUB_MODULES}
    yield
    for n, m in saved.items():
        if m is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = m


def _load():
    _install_stubs()
    spec = importlib.util.spec_from_file_location('apriltag_relocalizer', NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tf_at(x, y, yaw):
    """A TransformStamped-shaped stub for map->base_link."""
    return types.SimpleNamespace(transform=types.SimpleNamespace(
        translation=types.SimpleNamespace(x=x, y=y, z=0.0),
        rotation=types.SimpleNamespace(x=0.0, y=0.0,
                                       z=math.sin(yaw / 2), w=math.cos(yaw / 2))))


def _mat_at(x, y, yaw):
    """A 4x4 map->base matrix, as the tag composition produces."""
    m = np.eye(4)
    m[0, 0] = math.cos(yaw); m[0, 1] = -math.sin(yaw)
    m[1, 0] = math.sin(yaw); m[1, 1] = math.cos(yaw)
    m[0, 3] = x
    m[1, 3] = y
    return m


@pytest.fixture
def node():
    mod = _load()
    n = mod.AprilTagRelocalizer.__new__(mod.AprilTagRelocalizer)
    n.tag_frame, n.base_frame, n.map_frame = 'saloni_tag', 'base_link', 'map'
    n.tag_id = 0
    n.auto_seed = True
    n.auto_seed_interval = 8.0
    n.auto_seed_min_shift = 0.25
    n.auto_seed_min_yaw = 8.0
    n.map_tag = np.eye(4)
    n.seeded = False
    n.force_once = False
    n._last_seed = 0.0
    logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                   warn=lambda *a, **k: None,
                                   debug=lambda *a, **k: None,
                                   error=lambda *a, **k: None)
    n.get_logger = lambda: logger
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(to_msg=lambda: None, nanoseconds=0))
    n.published = []
    n.pub = types.SimpleNamespace(publish=lambda m: n.published.append(m))
    n._mod = mod
    return n


def _wire(node, tag_says, robot_thinks):
    """Tag composition yields tag_says; TF says the robot is at robot_thinks."""
    tx, ty, tyaw = tag_says

    def lookup(target, source):
        if target == node.map_frame and source == node.base_frame:
            if robot_thinks is None:
                raise RuntimeError('no map->base_link yet')
            return _tf_at(*robot_thinks)
        return _tf_at(0.0, 0.0, 0.0)      # tag->base, unused: map_tag is identity

    node._lookup = lookup
    node._mod.tf_to_mat = lambda t: _mat_at(tx, ty, tyaw)


# ── the gate ──────────────────────────────────────────────────────────

def test_it_stays_quiet_when_the_tag_agrees_with_the_pose(node):
    """The bug: a parked, correctly-localized robot got its filter reset every
    8 s, injecting the tag's own jitter into a pose that was already right."""
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=(2.05, 1.03, 0.52))
    assert node._publish_initialpose(forced=False) is False
    assert node.published == []


def test_a_real_displacement_still_seeds(node):
    """Carried to another room — this is what auto-seed exists for."""
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=(4.0, 3.0, 0.5))
    assert node._publish_initialpose(forced=False) is True
    assert len(node.published) == 1


def test_a_yaw_error_alone_is_enough_to_seed(node):
    """A robot rotated in place has zero shift, and a wrong heading throws the
    scan off the walls harder than a wrong position does."""
    _wire(node, tag_says=(2.0, 1.0, 0.9), robot_thinks=(2.0, 1.0, 0.5))
    assert node._publish_initialpose(forced=False) is True


def test_an_explicit_relocalize_ignores_the_gate(node):
    """~/relocalize means 'apply the tag now' — it must not be second-guessed."""
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=(2.0, 1.0, 0.5))
    assert node._publish_initialpose(forced=True) is True
    assert len(node.published) == 1


def test_no_current_pose_means_seed(node):
    """Nothing to compare against (no map->base_link yet) must not be read as
    'agrees' — that would leave a fresh boot unlocalized."""
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=None)
    assert node._publish_initialpose(forced=False) is True


def test_thresholds_are_honoured_from_parameters(node):
    """Tightening the threshold must make a borderline case fire."""
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=(2.15, 1.0, 0.5))
    assert node._publish_initialpose(forced=False) is False
    node.auto_seed_min_shift = 0.05
    assert node._publish_initialpose(forced=False) is True


# ── the surrounding fire/interval logic ───────────────────────────────

class _Detections:
    def __init__(self, ids):
        self.detections = [types.SimpleNamespace(id=i) for i in ids]


def test_a_blocked_seed_still_restarts_the_interval(node):
    """Otherwise a parked robot re-runs the gate on every detection frame
    (~30 Hz) instead of once per auto_seed_interval."""
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=(2.0, 1.0, 0.5))
    node.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=100e9))
    node._on_detections(_Detections([0]))
    assert node.published == []
    assert node._last_seed == pytest.approx(100.0)


def test_a_tag_that_is_not_ours_is_ignored(node):
    _wire(node, tag_says=(2.0, 1.0, 0.5), robot_thinks=(9.0, 9.0, 0.5))
    node._on_detections(_Detections([7]))
    assert node.published == []
