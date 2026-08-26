"""The PS5 pad must actually exist during a mapping run.

2026-08-06: "when I go to make a new map it didn't work from the PS5
controller, only from the web buttons — forward/back/left/right". Both halves
of that sentence are the diagnosis. The dashboard's arrow buttons publish a
twist from inside web_dashboard_node, so they were never going to notice; the
pad needs a joy_node, and there wasn't one.

`map_session.sh new` starts the mapping stack with

    ros2 launch home_robot bringup.launch.py use_slam:=true use_dashboard:=true

and bringup declared `use_joy` with default_value='false'. Nothing in that
command line turns it on, so joy_node and teleop_twist_joy_node were both
gated off for the entire run. No error appeared anywhere — a pad that is
never read looks exactly like a pad that is switched off.

Two more traps sat behind that one, each of which would have kept the pad dead
after the flag was fixed. localize.launch.py had already hit and documented
both; bringup and slam_only had not:

  * `dev` is NOT a parameter of Jazzy's joy_node. It is the ROS 1 / joy_linux
    spelling. Jazzy takes device_id (int) or device_name (string) and ignores
    unknown parameters in silence, so `{'dev': '/dev/input/js0'}` bought
    nothing and left device_id at its default 0 — whichever pad enumerated
    first. Confirmed against the live node:

        $ ros2 param list /joy_node
        autorepeat_rate, coalesce_interval_ms, deadzone,
        device_id, device_name, sticky_buttons, use_sim_time

    — no `dev`.
  * autorepeat_rate was unset (default 0.0), so joy publishes only when an
    axis CHANGES. Holding the stick at a steady deflection sends nothing, and
    roomba_driver's watchdog stops the base a moment later. The robot would
    have crawled in twitches.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_joy_mapping_run.py -q
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRINGUP = (ROOT / 'launch' / 'bringup.launch.py').read_text()
SLAM_ONLY = (ROOT / 'launch' / 'slam_only.launch.py').read_text()
LOCALIZE = (ROOT / 'launch' / 'localize.launch.py').read_text()
MAP_SESSION = (ROOT / 'scripts' / 'map_session.sh').read_text()

# The launch files that are expected to drive the base from the pad.
DRIVES_FROM_PAD = {'bringup.launch.py': BRINGUP, 'slam_only.launch.py': SLAM_ONLY,
                   'localize.launch.py': LOCALIZE}


def _joy_node_block(src: str) -> str:
    """The joy_node Node(...) call, from package='joy' to its closing paren.

    Balanced-paren scan rather than a closing-indent match: localize nests its
    copy inside actions.append(Node(...)), so the call closes at a different
    depth from bringup's plain `node = Node(...)`.
    """
    i = src.index("package='joy'")
    start = src.rindex('Node(', 0, i)
    depth, j = 0, start + len('Node')
    while True:
        if src[j] == '(':
            depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        j += 1


def test_new_mapping_run_asks_for_the_pad():
    """A hot mapping swap preserves the already-running joy stack."""
    m = re.search(r'^\s*new\)(.*?)^\s*;;', MAP_SESSION, re.S | re.M)
    assert m, 'the `new` branch of map_session.sh is gone or changed shape'
    branch = m.group(1)
    assert 'switch_map_mode.sh' in MAP_SESSION or 'SWITCH_MODE_SH' in branch
    assert ' slam ' in branch or ' slam' in branch


def test_bringup_enables_joy_by_default():
    """bringup on its own IS the mapping launch, so the pad is on by default."""
    m = re.search(r"DeclareLaunchArgument\('use_joy',\s*default_value='(\w+)'", BRINGUP)
    assert m, "bringup no longer declares use_joy"
    assert m.group(1) == 'true', (
        'bringup defaults to use_slam:=true, i.e. a mapping run; a false '
        'use_joy default is what killed the controller in the first place')


def test_localize_still_switches_bringups_joy_off():
    """The guard that stops use_joy:=true from spawning a SECOND joy_node.

    localize starts its own joy_node/teleop pair (it needs different
    remappings), and passes use_joy:='false' into the bringup include. If that
    ever goes away, the default flipped above yields two joy_node processes
    fighting over one device.
    """
    assert re.search(r"'use_joy':\s*'false'", LOCALIZE), (
        'localize must keep forcing bringup use_joy off, or the new default '
        'gives two joy_nodes on one pad')


def test_no_launch_file_uses_the_ros1_dev_parameter():
    """`dev` is joy_linux/ROS 1. Jazzy's joy_node ignores it silently."""
    for name, src in DRIVES_FROM_PAD.items():
        block = _joy_node_block(src)
        assert "'dev'" not in block, (
            f"{name}: joy_node has no `dev` parameter on Jazzy — it is ignored "
            f"without warning and the node falls back to device_id 0")


def test_pad_is_selected_by_name_not_index():
    """device_id 0 is 'whichever pad came up first', which is not a promise."""
    for name, src in DRIVES_FROM_PAD.items():
        block = _joy_node_block(src)
        assert 'device_name' in block, (
            f'{name}: select the DualSense by name — device_id depends on which '
            f'/dev/input/jsN it landed on')
        assert 'DualSense' in src, f'{name}: the pad name is gone'


def test_held_stick_keeps_publishing():
    """autorepeat_rate, or a held stick sends nothing and the base stops."""
    for name, src in DRIVES_FROM_PAD.items():
        block = _joy_node_block(src)
        m = re.search(r"'autorepeat_rate':\s*([\d.]+)", block)
        assert m, (
            f'{name}: without autorepeat_rate joy_node publishes only on axis '
            f'CHANGE, so holding the stick lets roomba_driver time out')
        assert float(m.group(1)) >= 10.0, (
            f'{name}: {m.group(1)} Hz is slower than roomba_driver expects')
