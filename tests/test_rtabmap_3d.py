"""Tests for the RTAB-Map 3D mapping tab (rtabmap.launch.py + the dashboard).

Added 2026-08-02. This feature runs a second SLAM system against the live robot
while AMCL and Nav2 keep working, which is exactly the arrangement that has
gone wrong here before, so most of these guard blast radius rather than
features. Three of the four were written after the thing they check had already
bitten during development:

  1. publish_tf must stay false. AMCL owns map->odom. A TF frame with two
     parents is what the orphaned Gazebo node did to odom->base_link, and it
     cost a long session chasing a "localization bug" that was not in the
     localization code.

  2. The node must stay in the `rtabmap` namespace. Run bare — which is the
     obvious way to write the launch file — it publishes /map, the SAME
     nav_msgs/OccupancyGrid topic map_server owns and AMCL and both costmaps
     read. `ros2 topic info -v /map` grew an rtabmap publisher within seconds
     of the first test run. It also takes /global_path, /local_path, /goal_out
     and /grid_prob_map, all Nav2 names. Nothing errors; the live map just
     quietly acquires a rival writer.

  3. Depth must come from aligned_depth_to_color. The raw depth image is in the
     depth sensor's frame, and RTAB-Map wants it registered to the colour image.

  4. The dashboard's pause/resume must address /rtabmap/rtabmap/… — namespace
     AND node name. /rtabmap/pause showed up in `ros2 service list` too and is
     the natural guess, but it was a stale graph entry from an un-namespaced
     run: the call succeeded, returned cleanly, and did nothing. Measured with
     the frame counter — it kept climbing at +24 per 10 s through a "pause",
     against +0 once the path was right.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_rtabmap_3d.py -q
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LAUNCH = (_ROOT / 'launch' / 'rtabmap.launch.py').read_text()
_DASH = (_ROOT / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()
_GUI = (_ROOT / 'scripts' / 'gui_session.sh').read_text()


# ── rule 1: never a second publisher of map->odom ───────────────────────────

def test_publish_tf_is_false():
    assert re.search(r"'publish_tf':\s*False", _LAUNCH), \
        "publish_tf is not pinned False — RTAB-Map would fight AMCL for map->odom"
    assert not re.search(r"'publish_tf':\s*True", _LAUNCH)


def test_publish_tf_is_not_exposed_as_a_launch_argument():
    """A DeclareLaunchArgument for it would make the dangerous setting one
    command-line flag away, and the tab passes user-supplied args nowhere near
    this — so there is no reason for the knob to exist."""
    args = re.findall(r"DeclareLaunchArgument\(\s*'([^']+)'", _LAUNCH)
    assert 'publish_tf' not in args and 'publish_tf_map' not in args


def test_the_reason_is_written_down_next_to_the_setting():
    assert 'AMCL' in _LAUNCH and 'map -> odom' in _LAUNCH, \
        'the header no longer explains why publish_tf must stay false'


# ── rule 2: never a second publisher of /map ────────────────────────────────

def test_both_nodes_run_in_the_rtabmap_namespace():
    """Bare, this node publishes /map — map_server's topic."""
    nodes = re.findall(r"Node\((.*?)\n        \)", _LAUNCH, re.S)
    assert nodes, 'no Node() blocks found — did the launch file change shape?'
    for n in nodes:
        pkg = re.search(r"package='([^']+)'", n)
        assert "namespace='rtabmap'" in n, \
            f"{pkg.group(1) if pkg else '?'} is not namespaced — it would " \
            "publish /map, /global_path and /goal_out into the live graph"


def test_the_subscriptions_are_absolute_so_the_namespace_cannot_hide_them():
    """Namespacing the node must not also namespace what it listens to, or it
    would sit waiting on /rtabmap/camera/... forever."""
    for topic in ('/camera/camera/color/image_raw', '/scan', '/odom'):
        assert f"'{topic}'" in _LAUNCH, f'{topic} is no longer remapped absolutely'


# ── rule 3: registered depth ────────────────────────────────────────────────

def test_depth_comes_from_the_aligned_stream():
    m = re.search(r"\('depth/image',\s*'([^']+)'\)", _LAUNCH)
    assert m, 'the depth remap is gone'
    assert 'aligned_depth_to_color' in m.group(1), (
        f'depth is taken from {m.group(1)} — unregistered depth puts every '
        'point in the wrong place')


def test_the_session_turns_aligned_depth_on_and_back_off():
    """The lean localize.launch.py camera starts with align_depth off, and
    leaving it on afterwards is a second 23 Hz stream nobody would look for."""
    assert 'align_depth.enable true' in _GUI, \
        'the rtabmap session never enables aligned depth — the mapper gets nothing'
    assert 'align_depth.enable false' in _GUI, \
        'aligned depth is never restored when the session stops'


def test_the_restore_subshell_escapes_set_u():
    """This file runs under `set -u`, and ROS's setup.bash reads unbound
    variables — so sourcing it inside a subshell aborts before ros2 runs. With
    stderr to /dev/null that failed silently: `stop` said "stopped" and left
    aligned depth on. Only caught by checking the parameter afterwards."""
    stop = _GUI.split('  stop)', 1)[1].split('  status)', 1)[0]
    block = stop.split('align_depth.enable false', 1)[0]
    assert 'set +u' in block, (
        'the align_depth restore sources ROS under set -u — it will abort on '
        'AMENT_TRACE_SETUP_FILES before reaching ros2, and silently')


# ── rule 4: the service path that silently did nothing ──────────────────────

def test_the_dashboard_calls_the_nodes_own_services():
    m = re.search(r"create_client\(Empty,\s*f?'(/rtabmap/[^']*)\{cmd\}'\)", _DASH)
    assert m, 'the rtabmap service client is gone'
    assert m.group(1) == '/rtabmap/rtabmap/', (
        f"calling {m.group(1)}{{cmd}} — that is the stale un-namespaced path; "
        'it returns success and does nothing')


def test_only_the_three_safe_commands_are_accepted():
    """The websocket is reachable by anything holding the token; `reset` wipes
    the map without confirmation and is deliberately not wired up."""
    m = re.search(r"if cmd in \(([^)]*)\)", _DASH)
    assert m, 'the rtabmap command whitelist is gone'
    assert set(re.findall(r"'([^']+)'", m.group(1))) == {
        'pause', 'resume', 'trigger_new_map'}


# ── the tab plumbing ────────────────────────────────────────────────────────

def test_the_display_is_not_one_another_app_already_owns():
    ports = re.search(r"VNC_PORTS = \{([^}]*)\}", _DASH).group(1)
    got = dict(re.findall(r"'(\w+)':\s*(\d+)", ports))
    assert got.get('rtabmap') == '5905', 'rtabmap is not on 5905'
    assert len(set(got.values())) == len(got), f'two apps share a VNC port: {got}'
    # :2 is the RViz session `robot max` shares with the phone; landing on it
    # would mean two RViz instances, which deactivates AMCL.
    assert got['rtabmap'] != got['rviz']


def test_gui_session_knows_the_app():
    assert re.search(r"rtabmap\)\s*DISP=5", _GUI), \
        'gui_session.sh has no rtabmap display mapping'
    assert 'rtabmap:5' in _GUI, 'rtabmap is missing from the status listing'


def test_the_window_title_grep_keeps_the_hyphen():
    """The window is "RTAB-Map* [ROS]". grep -i rtabmap does NOT match that —
    the hyphen — so the maximise never fired and the tab showed the GUI
    floating on black inside a 1600x900 display."""
    m = re.search(r'grep -q\w* "([^"]*RTAB[^"]*)"', _GUI, re.I)
    assert m, 'the rtabmap window-title grep is gone'
    assert 'RTAB-Map' in m.group(1), \
        f'grepping for {m.group(1)!r} will not match "RTAB-Map* [ROS]"'


def test_it_stays_on_the_real_robots_domain():
    """Unlike the Gazebo tab, the whole point is the house the robot is in — so
    it must NOT inherit the simulation's isolating domain."""
    block = _GUI.split('rtabmap)', 1)[1].split('gazebo)', 1)[0]
    assert 'ROS_DOMAIN_ID' not in block, \
        'the rtabmap session overrides ROS_DOMAIN_ID — it would see no sensors'
