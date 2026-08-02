"""Tests for driving the robot from the web dashboard (web_dashboard_node).

2026-08-02, reported: "τα πλήκτρα από το web να κουνήσω το ρομπότ δεν δουλεύουν".
Two independent bugs, either of which alone was enough to make the D-pad inert:

 1. The dashboard published to /cmd_vel. Nothing on this graph subscribes to it.
    roomba_driver listens on /cmd_vel_safe alone, and Nav2 reaches that topic by
    its own chain (cmd_vel_nav -> velocity_smoother -> cmd_vel_smoothed ->
    collision_monitor -> cmd_vel_safe). `ros2 topic info -v /cmd_vel` showed four
    publishers and zero subscribers — every press went into a black hole. The
    PS5 teleop had always been wired correctly (localize.launch.py remaps its
    cmd_vel -> cmd_vel_safe), which is why the joystick worked and the web did not.

 2. The turn buttons sent 0.10 rad/s. The 879 has a ~0.31 rad/s rotation floor —
    below it the wheels do not move at all. So even on the right topic, ◄ and ►
    would have done nothing, and the same mistake had already been made once on
    the joystick (angular scale 0.06). teleop_twist_joy_ps5.yaml carries the
    same warning in a comment: NEVER below ~0.35.

Plus: there were no keyboard bindings at all, on a page whose D-pad is the
obvious thing to drive with arrow keys.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_drive.py -q
"""
import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

# The 879's measured rotation floor, from project memory and confirmed against
# teleop_twist_joy_ps5.yaml's own comment.
ROTATION_FLOOR = 0.31


# ── bug 1: the topic nothing listened to ────────────────────────────────────

def test_the_drive_publisher_reaches_the_roomba():
    m = re.search(r"_vel_pub\s*=\s*self\.create_publisher\(\s*Twist,\s*'([^']+)'", _SRC)
    assert m, 'the drive publisher is gone or changed shape'
    assert m.group(1) == '/cmd_vel_safe', (
        f'publishing to {m.group(1)!r}: roomba_driver subscribes to /cmd_vel_safe '
        'only, so anything else is silently discarded')


def test_the_estop_still_zeroes_the_same_topic_it_drives():
    """An e-stop that zeroes a different topic than the D-pad drives is worse
    than none — it would report stopped while the wheels kept the last command."""
    dispatch = re.search(r"def dispatch\(self.*?(?=\n    # ──|\nclass |\Z)", _SRC, re.S)
    assert dispatch, 'dispatch() is gone'
    body = dispatch.group(0)
    assert body.count('self._vel_pub.publish(Twist())') >= 2, \
        'stop and estop must both zero the drive publisher'


# ── bug 2: a turn rate the wheels cannot execute ────────────────────────────

def _consts():
    m = re.search(r'const LIN = ([\d.]+), ANG = ([\d.]+);', _SRC)
    assert m, 'the LIN/ANG constants are gone'
    return float(m.group(1)), float(m.group(2))


def test_the_turn_rate_clears_the_rotation_floor():
    _, ang = _consts()
    assert ang > ROTATION_FLOOR, (
        f'ANG={ang} rad/s is at or below the 879\'s {ROTATION_FLOOR} rad/s '
        'rotation floor — the wheels will not move and the turn buttons look dead')
    # Some margin, not just a hair over: the floor is approximate and the driver
    # ramps into it through max_angular_accel.
    assert ang >= 0.35, f'ANG={ang} leaves no margin above the floor'


def test_the_drive_speed_stays_gentle_indoors():
    """Nothing here should turn into a flat-out sprint by accident: this is a
    phone D-pad with no analogue control and a network in the loop."""
    lin, ang = _consts()
    assert 0 < lin <= 0.25, f'LIN={lin} m/s is too fast for tap-to-nudge driving'
    assert ang <= 1.20, f'ANG={ang} exceeds the joystick\'s own 1.20 rad/s'


# ── the keyboard ────────────────────────────────────────────────────────────

def test_arrow_keys_and_wasd_are_bound():
    m = re.search(r'const KEYS = \{(.*?)\};', _SRC, re.S)
    assert m, 'no keyboard bindings — the arrow keys do nothing'
    keys = m.group(1)
    for k in ('ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd'):
        assert re.search(r'\b%s\s*:' % re.escape(k), keys), f'{k} is not bound'


def test_holding_a_key_does_not_stack_timers():
    """keydown auto-repeats while held; restarting the 100 ms send timer on each
    repeat would leak an interval that keeps driving after keyup."""
    assert re.search(r'e\.repeat\s*&&\s*keyHeld\s*===\s*e\.key', _SRC), \
        'auto-repeat is not filtered, so held keys stack drive timers'


def test_typing_in_a_field_does_not_drive_the_robot():
    """The chat box is on the same page. Typing "sad" must not be a drive command."""
    assert re.search(r"tagName\s*===\s*'INPUT'", _SRC), \
        'keydown does not exempt text inputs — typing would drive the robot'


def test_losing_the_tab_releases_the_keys():
    """keyup never arrives if the tab is switched away mid-press."""
    assert "window.addEventListener('blur'" in _SRC, 'blur does not release the keys'
    assert 'visibilitychange' in _SRC, 'a hidden tab does not release the keys'


def test_there_is_a_keyboard_panic_stop():
    assert re.search(r"e\.key === ' '", _SRC), 'space is not bound to stop'


# ── what still protects the robot on this path ──────────────────────────────

def test_the_bypass_of_collision_monitor_is_documented():
    """cmd_vel_safe skips collision_monitor, exactly as the joystick does. That
    is a deliberate trade, so the reason and the remaining guards must be stated
    where the next person changes this line."""
    m = re.search(r'((?:\n\s*#[^\n]*)+)\n\s*self\._vel_pub\s*=', _SRC)
    assert m, 'the drive publisher lost its explanatory comment'
    note = m.group(1)
    assert 'collision_monitor' in note, 'the bypass is not called out'
    assert 'watchdog' in note, 'the 0.25 s stale-command watchdog is not mentioned'


def test_the_browser_repeats_the_command_faster_than_the_watchdog():
    """roomba_driver zeroes the wheels 0.25 s after the last command. The repeat
    has to be comfortably under that or a held button stutters."""
    m = re.search(r"setInterval\(\(\)=>send\(\{type:'cmd_vel',vx,wz\}\),(\d+)\)", _SRC)
    assert m, 'the drive repeat timer is gone'
    period_s = int(m.group(1)) / 1000
    assert period_s <= 0.125, (
        f'{period_s}s between repeats against a 0.25s watchdog leaves no margin '
        'for a slow phone link')
