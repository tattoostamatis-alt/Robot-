"""Tests for the fetch mission's pure helpers (home_robot/fetch_planner.py).

Robot-free: geometry of the approach pose, memory-vs-live target resolution,
and nearest-detection selection, plus a static wiring check that the mission
node and LLM tool speak the fetch:<label> contract.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_fetch_planner.py -q
"""
import math
import os

from home_robot.fetch_planner import (approach_pose, memory_target,
                                      nearest_detection, homing_twist)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = f'{PKG}/home_robot/nodes'


# ── approach_pose ──────────────────────────────────────────────────────

def test_approach_stops_short_and_faces_object():
    # robot at origin, object 2 m ahead on +x; approach 0.4 m short of it.
    ax, ay, yaw = approach_pose((0.0, 0.0), (2.0, 0.0), dist=0.4)
    assert math.isclose(ax, 1.6, abs_tol=1e-6)
    assert math.isclose(ay, 0.0, abs_tol=1e-6)
    # standing at x=1.6 the object is toward +x → yaw ≈ 0
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)


def test_approach_distance_is_respected_off_axis():
    ax, ay, yaw = approach_pose((0.0, 0.0), (3.0, 4.0), dist=1.0)
    # park exactly 1 m from the object
    assert math.isclose(math.hypot(3.0 - ax, 4.0 - ay), 1.0, abs_tol=1e-6)
    # facing the object from the approach point
    assert math.isclose(yaw, math.atan2(4.0 - ay, 3.0 - ax), abs_tol=1e-6)


def test_approach_degenerate_robot_on_object():
    ax, ay, yaw = approach_pose((1.0, 1.0), (1.0, 1.0), dist=0.5)
    # falls back to approaching from -x → parks at x-0.5, faces +x
    assert math.isclose(ax, 0.5, abs_tol=1e-6)
    assert math.isclose(ay, 1.0, abs_tol=1e-6)
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)


# ── memory_target ──────────────────────────────────────────────────────

def test_memory_target_fresh_is_used():
    ans = {'x': 1.0, 'y': 2.0, 'room': 'kouzina', 'last_seen': 100.0}
    t = memory_target(ans, now=200.0, max_age=300.0)
    assert t == {'x': 1.0, 'y': 2.0, 'room': 'kouzina', 'last_seen': 100.0}


def test_memory_target_stale_is_rejected():
    ans = {'x': 1.0, 'y': 2.0, 'room': 'kouzina', 'last_seen': 100.0}
    assert memory_target(ans, now=1000.0, max_age=300.0) is None


def test_memory_target_empty_or_missing_xy_is_none():
    assert memory_target({}, now=0.0, max_age=300.0) is None
    assert memory_target({'room': 'kouzina'}, now=0.0, max_age=300.0) is None


def test_memory_target_max_age_zero_disables_staleness():
    ans = {'x': 1.0, 'y': 2.0, 'last_seen': 0.0}
    assert memory_target(ans, now=1e9, max_age=0.0) is not None


# ── nearest_detection ──────────────────────────────────────────────────

def test_nearest_detection_picks_closest_matching_label():
    objs = [
        {'label': 'cup', 'z': 2.0},
        {'label': 'cup', 'z': 0.8},   # closest cup
        {'label': 'bottle', 'z': 0.3},
    ]
    hit = nearest_detection(objs, 'cup', max_z=3.0)
    assert hit['z'] == 0.8


def test_nearest_detection_filters_range_and_label():
    objs = [{'label': 'cup', 'z': 5.0}, {'label': 'cup', 'z': 0.05}]
    assert nearest_detection(objs, 'cup', max_z=3.0) is None   # too far / too near
    assert nearest_detection([], 'cup', max_z=3.0) is None
    assert nearest_detection([{'label': 'bottle', 'z': 1.0}], 'cup', max_z=3.0) is None


# ── homing_twist (follow delivery) ─────────────────────────────────────

def test_homing_turns_right_toward_person_on_the_right():
    lin, ang, arrived = homing_twist({'x': 1.0, 'z': 2.0}, deliver_distance=0.8)
    assert ang < 0            # person on the right → turn clockwise (negative)
    assert lin == 0.0         # don't drive forward until centred
    assert not arrived


def test_homing_drives_forward_when_centred_and_far():
    lin, ang, arrived = homing_twist({'x': 0.0, 'z': 2.0}, deliver_distance=0.8)
    assert ang == 0.0
    assert lin > 0.0
    assert not arrived


def test_homing_arrived_when_centred_and_close():
    lin, ang, arrived = homing_twist({'x': 0.0, 'z': 0.6}, deliver_distance=0.8)
    assert arrived
    assert lin == 0.0 and ang == 0.0


def test_homing_angular_is_clamped():
    lin, ang, _ = homing_twist({'x': 100.0, 'z': 0.1}, max_ang=0.5)
    assert abs(ang) <= 0.5


def test_homing_no_range_holds():
    assert homing_twist({'x': 0.5, 'z': 0.0}) == (0.0, 0.0, False)


# ── static wiring contract ─────────────────────────────────────────────

def test_mission_node_handles_fetch():
    src = open(f'{NODES}/mission_executor_node.py').read()
    assert "fetch:" in src
    assert "'object_memory/query'" in src and "'object_memory/answer'" in src
    assert "'pick_command'" in src


def test_llm_bridge_exposes_fetch_tool():
    src = open(f'{NODES}/llm_bridge_node.py').read()
    assert "'fetch'" in src or '"fetch"' in src
    assert 'fetch:' in src        # dispatch publishes mission/start fetch:<label>
