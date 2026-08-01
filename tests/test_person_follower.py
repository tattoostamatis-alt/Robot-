"""Tests for person_follower_node's control loop.

The bug these exist for: the loop was OPEN. `_doa_col` was written only by the
wake-angle callback, so during a follow the bearing error was a constant —
`_turn_for` returned the same non-zero rate on every frame and the robot spun on
the spot for the whole 30 s, reading depth at a column it had long since turned
away from. Nothing in the loop looked at where the person actually was, so
nothing could make the error shrink.

So the property under test is not "does it turn" but **does turning reduce the
error and then stop** — plus the other half of the same mistake: never steer on
a bearing nobody has confirmed recently.

Robot-free: `_pick_person` is a static method and `_turn_for`/`_control` touch
only the tuning parameters, so they run on a bare class with no rclpy context
and no camera. `_control`'s Twist is captured through a stub publisher.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_person_follower.py -q
"""
import json
import os
import re
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = f'{PKG}/home_robot/nodes/person_follower_node.py'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_safety_regressions import _BaseNode, _load, _mod, _ns, _twist  # noqa: E402

IMG_W = 640
CENTRE = IMG_W / 2

TURN_GAIN = 0.0025
TURN_DEAD = 40
TURN_MIN = 0.35
TURN_MAX = 0.60
DIST_MIN = 0.8
DIST_MAX = 1.5
SPEED = 0.15


class _Twist:
    """Stands in for geometry_msgs/Twist — _control only sets linear.x/angular.z."""

    def __init__(self):
        self.linear = type('v', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
        self.angular = type('v', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


def _person(centre, distance, img_w=IMG_W, width=80, label='person'):
    """One detection in object_detector.py's published shape."""
    return {'label': label, 'x1': centre - width / 2, 'x2': centre + width / 2,
            'img_w': img_w, 'img_h': 480, 'box_distance': distance}


@pytest.fixture
def node():
    """_pick_person/_turn_for/_control lifted onto a bare class.

    Importing the module needs rclpy and a camera stack; the methods themselves
    need neither, so extract them from source (same approach as
    test_soft_start.py).
    """
    src = open(NODE).read()
    body = ''
    for name in ('_pick_person', '_turn_for', '_control'):
        match = re.search(rf'\n    (?:@staticmethod\n    )?def {name}\(.*?\n(?=    (?:[@#]|def |-))',
                          src, re.S)
        assert match, f'{name} not found — was it renamed?'
        body += match.group(0)
    ns = {'Twist': _Twist, '_IMAGE_WIDTH': IMG_W, 'math': __import__('math')}
    exec('class _Stub:\n' + body, ns)
    stub = ns['_Stub']()
    stub._turn_gain, stub._turn_dead = TURN_GAIN, TURN_DEAD
    stub._turn_min, stub._turn_max = TURN_MIN, TURN_MAX
    stub._dist_min, stub._dist_max, stub._speed = DIST_MIN, DIST_MAX, SPEED
    stub._cmd_pub = _Pub()
    stub.get_logger = lambda: type('L', (), {
        'debug': lambda *a, **k: None, 'info': lambda *a, **k: None,
        'warn': lambda *a, **k: None})()
    return stub


# ── End to end: the real node, the real callback ──────────────────────

@pytest.fixture
def live():
    """The whole node with ROS stubbed, so tests drive _detections_cb itself.

    Unit-testing the helpers is not enough here: `_turn_for` was never wrong on
    its own, the loop around it was. Verified by reinstating the original bug —
    feeding `_control` the seeded column instead of the detected one leaves
    every helper-level test on this page green, and is caught only by
    test_the_robot_converges_on_the_person_and_stops_turning below.
    """
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None, spin=lambda *a, **k: None,
                      ok=lambda: True, shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg',
                             Float32=lambda data=0.0: _ns(data=data),
                             Bool=lambda data=False: _ns(data=data),
                             String=lambda data='': _ns(data=data)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg', Twist=_twist()),
    }
    mod = _load(NODE, mods)
    node = mod.PersonFollowerNode()
    node._follow_cmd_cb(_ns(data=True))          # "ακολούθησέ με"
    return node


def _frame(node, objects):
    """Deliver one detection frame; return the Twist published, or None."""
    before = len(node.pubs['/cmd_vel'].sent)
    node._detections_cb(_ns(data=json.dumps(objects)))
    sent = node.pubs['/cmd_vel'].sent
    return sent[-1] if len(sent) > before else None


PX_PER_RAD = IMG_W / 1.52          # D435 87° HFOV across the frame


def test_the_robot_converges_on_the_person_and_stops_turning(live):
    """‼️ THE REGRESSION, end to end.

    The person stands still off to one side; each frame the robot turns and the
    detector reports where they now appear. The old node span for the full 30 s
    because this column never moved — here it must reach centre and the turn
    must go to zero.
    """
    col = 600.0
    rates = []
    for _ in range(60):
        twist = _frame(live, [_person(col, 1.2)])
        assert twist is not None, 'node stopped steering while the person was visible'
        rates.append(twist.angular.z)
        if twist.angular.z == 0.0:
            break
        col += twist.angular.z * 0.1 * PX_PER_RAD
    assert rates[-1] == 0.0, f'still turning after {len(rates)} frames — loop is open'
    assert abs(col - CENTRE) <= TURN_DEAD


def test_losing_the_person_stops_the_robot(live):
    """A lost target used to keep the last bearing, and therefore the last turn
    rate, indefinitely."""
    _frame(live, [_person(600, 1.2)])
    live._last_seen -= 10.0                       # person out of view, timeout passed
    twist = _frame(live, [])
    assert twist is not None, 'no stop published when the person vanished'
    assert twist.angular.z == 0.0 and twist.linear.x == 0.0


def test_an_empty_frame_never_produces_motion(live):
    """Stronger version of the above: with nothing detected, at no point may a
    non-zero command be published, however long the follow runs."""
    for _ in range(30):
        twist = _frame(live, [])
        if twist is not None:
            assert twist.angular.z == 0.0 and twist.linear.x == 0.0
        live._last_seen -= 0.5


def test_a_person_walking_away_is_followed(live):
    """The forward axis closes too: distance shrinks, and the robot stops
    driving once in range."""
    twist = _frame(live, [_person(CENTRE, 3.0)])
    assert twist.linear.x > 0, 'did not set off after a distant person'
    twist = _frame(live, [_person(CENTRE, (DIST_MIN + DIST_MAX) / 2)])
    assert twist.linear.x == 0.0, 'did not stop once in range'


def test_following_expires_and_stops(live):
    """The timeout must leave the robot stopped, not mid-command."""
    live._active_until -= 1000
    _frame(live, [_person(600, 1.2)])
    twist = live.pubs['/cmd_vel'].sent[-1]
    assert twist.angular.z == 0.0 and twist.linear.x == 0.0
    assert live._active is False


def test_nothing_is_published_before_a_follow_command():
    """The node must be inert until the user actually asks."""
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None, spin=lambda *a, **k: None,
                      ok=lambda: True, shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _mod('std_msgs.msg',
                             Float32=lambda data=0.0: _ns(data=data),
                             Bool=lambda data=False: _ns(data=data),
                             String=lambda data='': _ns(data=data)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg', Twist=_twist()),
    }
    node = _load(NODE, mods).PersonFollowerNode()
    node._detections_cb(_ns(data=json.dumps([_person(600, 1.2)])))
    assert node.pubs['/cmd_vel'].sent == []


# ── ‼️ The regression: the loop must actually close ────────────────────

def test_turning_toward_the_person_reduces_the_error(node):
    """The property the old version could not satisfy at any gain.

    Simulates the real loop: the robot turns, the person's pixel column moves
    toward centre as a result, and the next frame reports the new column.
    """
    col = 620.0                      # person hard right of frame
    seen = [col]
    for _ in range(40):
        wz = node._turn_for(col, IMG_W)
        if wz == 0.0:
            break
        # Turning right (-z) pans the camera right, so a right-of-centre
        # target's column falls toward centre. ~0.1 s per detection frame,
        # 87° HFOV = 1.52 rad across IMG_W pixels.
        col += wz * 0.1 * (IMG_W / 1.52)
        seen.append(col)
    assert abs(seen[-1] - CENTRE) < abs(seen[0] - CENTRE), 'error did not shrink'
    assert abs(seen[-1] - CENTRE) <= TURN_DEAD, f'never centred, ended at {seen[-1]:.0f}'


def test_the_robot_stops_turning_once_centred(node):
    """The old version never stopped, because the error never changed."""
    assert node._turn_for(CENTRE, IMG_W) == 0.0
    assert node._turn_for(CENTRE + TURN_DEAD, IMG_W) == 0.0
    assert node._turn_for(CENTRE - TURN_DEAD, IMG_W) == 0.0


def test_the_controller_is_fed_the_detected_bearing_not_the_seed():
    """Structural companion to the convergence test above, naming the defect.

    `_control` must be handed the column the DETECTOR just reported. Handing it
    the seeded/tracked column instead IS the original bug, and it is a one-word
    edit away at all times.
    """
    src = open(NODE).read()
    detections = src[src.index('def _detections_cb'):]
    detections = detections[:detections.index('\n    # ---')]
    assert 'self._control(distance_m, centre, img_w)' in detections, \
        'the controller is not fed the detected bearing — the loop is open again'
    assert 'self._target_col  = centre' in detections, \
        'the tracked bearing is not updated, so target continuity is lost'
    # And the wake angle must only seed, never steer.
    doa = src[src.index('def _doa_cb'):]
    doa = doa[:doa.index('\n    # ---')]
    assert '_target_col' not in doa, 'the wake angle is steering again'


# ── Never steer on a bearing nobody confirmed ─────────────────────────

def test_losing_the_person_stops_instead_of_spinning():
    """The other half of the bug: a lost target used to keep the last bearing
    and therefore the last turn rate, forever."""
    src = open(NODE).read()
    lost = src[src.index('found = self._pick_person'):]
    lost = lost[:lost.index('centre, distance_m, img_w = found')]
    assert '_publish_stop()' in lost, 'no stop when the person is out of view'
    assert '_control' not in lost, 'still driving without a sighting'


def test_no_person_in_frame_is_not_followed(node):
    """Other labels must not be mistaken for the person."""
    objects = [_person(300, 1.2, label='chair'), _person(500, 2.0, label='tv')]
    assert node._pick_person(objects, CENTRE) is None
    assert node._pick_person([], CENTRE) is None


def test_person_without_a_distance_is_skipped(node):
    """A bearing with no distance is half a control input — turning toward
    someone while guessing their range is how you drive into them."""
    objects = [_person(300, None)]
    assert node._pick_person(objects, CENTRE) is None


def test_person_with_a_broken_box_is_skipped(node):
    objects = [{'label': 'person', 'box_distance': 1.2, 'img_w': IMG_W}]
    assert node._pick_person(objects, CENTRE) is None


# ── Picking who to follow ─────────────────────────────────────────────

def test_picks_the_person_nearest_the_seeded_direction(node):
    """The DoA wake angle decides who to lock onto on the first frame."""
    objects = [_person(120, 2.0), _person(560, 1.2)]
    centre, distance, img_w = node._pick_person(objects, 600)
    assert centre == 560 and distance == 1.2 and img_w == IMG_W


def test_keeps_following_the_same_person_when_another_walks_past(node):
    """Continuity, not nearest-to-camera: someone crossing in front must not
    steal the follow."""
    tracked = _person(200, 2.5)          # who we are following, far away
    passerby = _person(430, 0.9)         # closer, but not our person
    centre, distance, _ = node._pick_person([tracked, passerby], target_col=210)
    assert centre == 200, 'switched target to whoever was closest to the camera'
    assert distance == 2.5


def test_follows_the_person_across_frames_as_they_move(node):
    """The tracked column follows the person rather than snapping to centre."""
    target = CENTRE
    for actual in (360, 420, 480, 520):
        centre, _, _ = node._pick_person([_person(actual, 1.2)], target)
        assert centre == actual
        target = centre


# ── Distance control ──────────────────────────────────────────────────

def test_too_far_drives_forward(node):
    node._control(DIST_MAX + 0.5, CENTRE, IMG_W)
    assert node._cmd_pub.sent[-1].linear.x == pytest.approx(SPEED)


def test_too_close_does_not_drive_forward(node):
    node._control(DIST_MIN - 0.3, CENTRE, IMG_W)
    assert node._cmd_pub.sent[-1].linear.x == 0.0


def test_in_range_holds(node):
    node._control((DIST_MIN + DIST_MAX) / 2, CENTRE, IMG_W)
    assert node._cmd_pub.sent[-1].linear.x == 0.0


def test_turns_while_closing_the_distance(node):
    """Both axes at once — the old node only ever set linear.x."""
    node._control(3.0, 620, IMG_W)
    twist = node._cmd_pub.sent[-1]
    assert twist.linear.x == pytest.approx(SPEED)
    assert twist.angular.z != 0.0


def test_turns_even_when_holding_distance(node):
    """A person at the right distance but off to one side still needs aiming at,
    otherwise the robot holds range on someone it is not facing."""
    node._control((DIST_MIN + DIST_MAX) / 2, 620, IMG_W)
    twist = node._cmd_pub.sent[-1]
    assert twist.linear.x == 0.0
    assert twist.angular.z != 0.0


# ── Turn direction and the chassis' rotation floor ────────────────────

def test_turn_direction_follows_rep_103(node):
    """Person on the right -> clockwise -> negative z."""
    assert node._turn_for(CENTRE + 200, IMG_W) < 0
    assert node._turn_for(CENTRE - 200, IMG_W) > 0


def test_turns_are_never_below_the_rotation_floor(node):
    """‼️ This chassis cannot rotate below ~0.31 rad/s — a smaller command just
    stalls the wheels, which would look exactly like the old spin bug (robot
    'turning' but the error never moving)."""
    for col in range(0, IMG_W, 7):
        wz = node._turn_for(col, IMG_W)
        assert wz == 0.0 or abs(wz) >= TURN_MIN


def test_turns_are_capped(node):
    for col in (0, IMG_W, -500, 2000):
        assert abs(node._turn_for(col, IMG_W)) <= TURN_MAX


def test_centre_comes_from_the_frame_not_a_hardcoded_width(node):
    """object_detector reports img_w per frame; assuming 640 would put 'centre'
    off to one side at any other resolution and bias every turn one way."""
    assert node._turn_for(160, 320) == 0.0        # centred in a 320-wide frame
    assert node._turn_for(160, IMG_W) > 0         # left of centre in a 640 one
