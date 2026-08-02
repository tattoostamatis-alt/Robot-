"""Regression tests for the fourth 2026-08-01 audit pass — a dead feature, a
lying topic, duplicated work, and state files that a power cut could shred.

  * person_follower_node subscribed to /camera/depth/image_rect_raw. Every other
    node in the package uses /camera/CAMERA/... because realsense2_camera nests
    the camera name under the camera namespace, and bringup gives this node no
    remapping — so it listened to a topic nobody publishes. "ακολούθησέ με"
    logged "Following activated", never received a frame, never published a
    Twist, and timed out 30 s later. On by default, dead since it was written.
    It also never turned: the DoA column only chose which depth strip to
    MEASURE, so the robot drove dead ahead on a distance read from off to one
    side.

  * tracker_node published tracks that no detection had confirmed this frame,
    at their last-seen 3D position. Everything downstream reads that topic as
    "what the robot can see right now" — semantic_costmap marks obstacles from
    it, fetch picks its grasp target from it.

  * semantic_costmap / situational_awareness / mission_executor each wired BOTH
    detected_objects and tracked_objects to one callback while their comments
    claimed a preference, so with perception up every cycle was processed twice.

  * pose_saver, apriltag calibration, speaker profiles and locations.yaml were
    all truncate-then-write on a machine that has lost power 8 times in 3 days.

Robot-free: ROS is stubbed.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_perception_regressions.py -q
"""
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_safety_regressions import (        # noqa: E402
    _BaseNode, _load, _mod, _ns, _twist)


# ── person_follower_node ────────────────────────────────────────────────────

@pytest.fixture
def follower():
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg',
                             Float32=lambda data=0.0: _ns(data=data),
                             Bool=lambda data=False: _ns(data=data),
                             String=lambda data='': _ns(data=data)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg', Twist=_twist()),
    }
    mod = _load(f'{PKG}/home_robot/nodes/person_follower_node.py', mods)
    return mod, mod.PersonFollowerNode()


def test_follower_listens_to_a_topic_that_exists(follower):
    """‼️ THE BUG: one namespace level short, so the node heard nothing at all
    and the whole follow feature was silently dead.

    The depth subscription is gone entirely as of 2026-08-01 — following now
    reads object_detector's detections, which carry bearing AND distance for a
    box actually labelled 'person'. Kept as a regression test on the wiring:
    the failure mode was subscribing to something nobody publishes, and
    `detected_objects` is the topic that must be there now.
    """
    mod, node = follower
    topics = [t for t, _cb in node.subs]
    assert 'detected_objects' in topics
    assert not any('depth' in t for t in topics), 'the dead depth path is back'


def test_follower_turns_toward_a_speaker_on_the_right(follower):
    """‼️ THE BUG: _control only ever set linear.x, so the robot drove straight
    ahead on a distance measured 40° off to the side. Asserted on the published
    Twist, not on the helper — the helper existing proves nothing if _control
    never calls it."""
    mod, node = follower
    node._control(5.0, mod._CENTER_COL + 200)
    out = node.pubs['/cmd_vel_safe'].sent[-1]
    assert out.angular.z < 0, 'the robot did not turn toward the speaker'


def test_follower_turns_toward_a_speaker_on_the_left(follower):
    mod, node = follower
    node._control(5.0, mod._CENTER_COL - 200)
    out = node.pubs['/cmd_vel_safe'].sent[-1]
    assert out.angular.z > 0


def test_follower_does_not_chase_tiny_errors(follower):
    """Deadband: a speaker roughly ahead should not set the robot spinning."""
    mod, node = follower
    assert node._turn_for(mod._CENTER_COL) == 0.0
    assert node._turn_for(mod._CENTER_COL + node._turn_dead - 1) == 0.0


def test_follower_never_commands_below_the_rotation_floor(follower):
    """This chassis cannot rotate below ~0.31 rad/s — a smaller command just
    stalls the wheels, so it must be either a real turn or none."""
    mod, node = follower
    for offset in range(node._turn_dead + 1, 320, 7):
        for wz in (node._turn_for(mod._CENTER_COL + offset),
                   node._turn_for(mod._CENTER_COL - offset)):
            assert abs(wz) >= node._turn_min - 1e-9, f'{wz} stalls the wheels'
            assert abs(wz) <= node._turn_max + 1e-9, f'{wz} is too fast'


def test_follower_still_drives_forward_when_far(follower):
    mod, node = follower
    node._control(5.0, mod._CENTER_COL)
    out = node.pubs['/cmd_vel_safe'].sent[-1]
    assert out.linear.x == pytest.approx(node._speed)


def test_follower_stops_when_too_close(follower):
    mod, node = follower
    node._control(0.2, mod._CENTER_COL)
    out = node.pubs['/cmd_vel_safe'].sent[-1]
    assert out.linear.x == 0.0


# ── tracker_node ────────────────────────────────────────────────────────────

@pytest.fixture
def tracker():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg', String=lambda data='': _ns(data=data)),
        'numpy': np,
        'scipy': _mod('scipy'),
        'scipy.optimize': _mod('scipy.optimize',
                               linear_sum_assignment=linear_sum_assignment),
    }
    mod = _load(f'{PKG}/home_robot/nodes/tracker_node.py', mods)
    return mod, mod.TrackerNode()


def _feed(node, dets):
    import json
    node._cb(_ns(data=json.dumps(dets)))
    import json as _j
    return _j.loads(node.pubs['tracked_objects'].sent[-1].data)


_CUP = {'label': 'cup', 'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200, 'z': 1.0}


def test_a_visible_object_is_published(tracker):
    mod, node = tracker
    out = _feed(node, [_CUP])
    assert [o['label'] for o in out] == ['cup']


def test_an_object_that_vanished_is_not_reported_as_visible(tracker):
    """‼️ THE BUG: the track survives occlusion by design, but publishing it
    meanwhile told every consumer the object was still there — at its stale 3D
    position. semantic_costmap marks obstacles from this, fetch grasps from it."""
    mod, node = tracker
    _feed(node, [_CUP])
    # A NON-empty frame that does not contain the cup: the empty-list branch
    # returns early, so it would not exercise the publish path at all.
    other = {'label': 'chair', 'x1': 400, 'y1': 400, 'x2': 500, 'y2': 500, 'z': 2.0}
    out = _feed(node, [other])
    labels = [o['label'] for o in out]
    assert 'cup' not in labels, 'a vanished object was still reported as visible'
    assert labels == ['chair']


def test_the_track_survives_a_gap_and_comes_back(tracker):
    """Occlusion tolerance must be intact — same track id, not a new one."""
    mod, node = tracker
    first = _feed(node, [_CUP])
    other = {'label': 'chair', 'x1': 400, 'y1': 400, 'x2': 500, 'y2': 500, 'z': 2.0}
    _feed(node, [other])                     # one frame without the cup
    again = _feed(node, [_CUP])
    assert again and again[0]['track_id'] == first[0]['track_id']


# ── detected/tracked preference ─────────────────────────────────────────────

@pytest.mark.parametrize('path,cls', [
    ('home_robot/nodes/semantic_costmap_node.py', None),
    ('home_robot/nodes/situational_awareness_node.py', None),
    ('home_robot/nodes/mission_executor_node.py', None),
])
def test_raw_detections_are_ignored_while_the_tracker_is_live(path, cls):
    """‼️ THE BUG: both topics went to one callback with no preference, so every
    detection cycle was processed twice with perception up."""
    src = open(f'{PKG}/{path}').read()
    assert "'tracked_objects',  self._on_tracked" in src.replace('   ', '  ')  \
        or '_on_tracked' in src, f'{path} still shares one callback'
    assert '_on_detected' in src
    # the fallback must actually gate on the tracker being live
    body = src.split('def _on_detected')[1].split('def ')[0]
    assert '_tracked_rx' in body and 'return' in body


# ── atomic state writes ─────────────────────────────────────────────────────

@pytest.mark.parametrize('path,func', [
    ('home_robot/nodes/record_location.py', '_save_locations'),
    ('home_robot/nodes/pose_saver_node.py', '_save'),
    ('home_robot/nodes/diarization_node.py', '_save_profiles'),
])
def test_state_files_are_written_atomically(path, func):
    """This machine has lost power 8 times in 3 days; a truncate-then-write
    leaves a half-file that costs re-teaching every room."""
    src = open(f'{PKG}/{path}').read()
    body = src.split(f'def {func}')[1].split('\ndef ')[0]
    assert 'os.replace' in body, f'{path}:{func} is not atomic'
    assert 'fsync' in body, f'{path}:{func} does not fsync before rename'


def test_apriltag_calibration_is_written_atomically():
    src = open(f'{PKG}/home_robot/nodes/apriltag_relocalizer.py').read()
    assert 'os.replace(tmp, self.calib_file)' in src
    assert 'os.fsync' in src
