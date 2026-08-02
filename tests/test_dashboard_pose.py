"""Tests for how the dashboard learns where the robot is (web_dashboard_node).

2026-08-02, found by opening the page in a real browser against a live stack:
the map tab drew the map and nothing else — no robot marker, no laser dots, and
X/Y/Γωνία all '—' — while `tf2_echo map base_link` answered instantly. The tab
looked like a localization failure on a perfectly localized robot.

/amcl_pose is not a position feed. AMCL publishes it only when it revises its
estimate, and it revises only while the robot drives; a robot standing still
(the state it is in whenever you first open the dashboard) publishes nothing at
all. room_markers_node hit this exact trap and grew a TF poll for it (fc6f1cb);
the dashboard was still subscribed to the topic alone.

Source text is asserted on rather than imported: importing the node pulls in
rclpy, cv2, uvicorn and FastAPI. Same approach as tests/test_dashboard_auth.py.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_pose.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()


def _method(name):
    m = re.search(r'\n    def %s\(self.*?(?=\n    def )' % name, _SRC, re.S)
    assert m, f'{name}() is gone or changed shape'
    return m.group(0)


# ── the poll exists and is wired ────────────────────────────────────────────

def test_the_dashboard_polls_tf_for_the_pose():
    """Without this, a standing robot never reaches the map tab."""
    assert '_poll_tf_pose' in _SRC, \
        'the TF pose poll is gone — the map tab goes blank on a still robot'


def test_the_poll_is_on_a_timer():
    assert re.search(r'create_timer\(\s*[\d.]+\s*,\s*self\._poll_tf_pose\)', _SRC), \
        'the TF poll must run on a timer, not only on a message'


def test_the_poll_is_fast_enough_to_track_a_driving_robot():
    m = re.search(r'create_timer\(\s*([\d.]+)\s*,\s*self\._poll_tf_pose\)', _SRC)
    assert m, 'the TF poll timer is gone'
    assert float(m.group(1)) <= 1.0, \
        'a slower poll makes the marker lag visibly while Nav2 drives'


def test_the_node_owns_a_tf_buffer_and_listener():
    assert 'tf2_ros.Buffer()' in _SRC and 'tf2_ros.TransformListener' in _SRC, \
        'the poll needs a TF buffer fed by a listener'
    assert re.search(r'^import tf2_ros$', _SRC, re.M), 'tf2_ros is not imported'


# ── what the poll looks up ──────────────────────────────────────────────────

def test_the_poll_looks_up_map_to_base_link():
    body = _method('_poll_tf_pose')
    assert re.search(r"lookup_transform\(\s*\n?\s*'map',\s*'base_link'", body), \
        'the pose must come from map->base_link, the same estimate AMCL publishes'


def test_the_poll_asks_for_the_latest_transform():
    body = _method('_poll_tf_pose')
    assert 'rclpy.time.Time()' in body, \
        'a stamped lookup would raise ExtrapolationException on every tick'


def test_an_unlocalized_robot_draws_no_robot():
    """Before AMCL seeds, there is no map->base_link and that is not an error."""
    body = _method('_poll_tf_pose')
    assert 'tf2_ros.LookupException' in body
    assert 'tf2_ros.ConnectivityException' in body
    assert 'tf2_ros.ExtrapolationException' in body
    assert re.search(r'except .*?:\s*\n\s*return', body, re.S), \
        'a missing transform must return quietly, not broadcast a bogus pose'


# ── both sources produce the same message ───────────────────────────────────

def test_the_topic_and_the_poll_share_one_publisher():
    """Two hand-rolled broadcast dicts would drift; they did not, keep it so."""
    assert 'self._publish_pose(' in _method('_cb_pose'), \
        '_cb_pose must funnel through the shared publisher'
    assert 'self._publish_pose(' in _method('_poll_tf_pose'), \
        '_poll_tf_pose must funnel through the shared publisher'


def test_the_shared_publisher_sends_a_pose_message():
    body = _method('_publish_pose')
    assert "'type': 'pose'" in body
    for key in ("'x'", "'y'", "'yaw'"):
        assert key in body, f'the pose message lost {key}'


def test_the_yaw_is_derived_the_same_way_from_both_sources():
    """Quaternion -> yaw, the flat-robot shortcut. If one side ever switches to
    a full conversion the marker points two different ways depending on which
    source last fired."""
    calls = re.findall(r'2\.0 \* math\.atan2\(([\w.]+)\.z, ([\w.]+)\.w\)', _SRC)
    assert len(calls) >= 2, \
        'the topic path and the TF path no longer compute yaw identically'
    for z, w in calls:
        assert z == w, f'yaw is read off two different objects: {z}.z / {w}.w'


def test_amcl_pose_is_still_subscribed():
    """The poll is a floor, not a replacement: the topic still fires the instant
    AMCL corrects, which is faster than waiting for the next tick."""
    assert "'/amcl_pose', self._cb_pose" in _SRC, \
        'dropping the topic makes the marker lag every AMCL correction'
