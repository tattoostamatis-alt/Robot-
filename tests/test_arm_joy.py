"""Tests for the PS5 right-stick arm jog (home_robot/nodes/arm_joy_node.py).

The node turns a velocity-ish stick deflection into the *position* stream that
arm_driver's T:102 needs, and that conversion has two ways to go quietly wrong:

  * Integrator windup — clamping only in the driver would let the internal
    target run past the joint limit while the stick is held, so the arm would
    ignore the first part of the way back.
  * A stale /joy message — if the controller drops while the stick is pushed,
    integrating on that last sample drives the arm into its limit by itself.

Both are tested here, plus the static wiring (topics/params) that ties this
node to arm_driver.py and to the teleop button map. Robot-free: the ROS Node
base class is stubbed, so no rclpy context, serial port, or hardware.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_arm_joy.py -q
"""
import importlib.util
import os
import sys
import types

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = f'{PKG}/home_robot/nodes/arm_joy_node.py'
TELEOP_YAML = f'{PKG}/config/teleop_twist_joy_ps5.yaml'
ARM_DRIVER = f'{PKG}/home_robot/nodes/arm_driver.py'

# The RoArm limits, restated rather than imported, so that a regression in the
# node's copy is caught instead of silently agreeing with itself.
BASE_LIMIT = (-3.14, 3.14)
SHOULDER_LIMIT = (-1.57, 1.57)
HAND_LIMIT = (1.08, 3.14)


# ── ROS stubs ─────────────────────────────────────────────────────────

class _FakeClock:
    """Nanosecond clock we advance by hand, shaped like rclpy's."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return _FakeTime(self.t)

    def advance(self, seconds):
        self.t += seconds


class _FakeTime:
    def __init__(self, seconds):
        self.seconds = seconds

    def __sub__(self, other):
        return types.SimpleNamespace(nanoseconds=(self.seconds - other.seconds) * 1e9)

    def to_msg(self):
        return None


class _Recorder:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def _install_ros_stubs():
    """Minimal rclpy/std_msgs/sensor_msgs so the node module imports."""
    if 'rclpy' in sys.modules and getattr(sys.modules['rclpy'], '_arm_joy_stub', False):
        return

    class _Node:
        def __init__(self, name):
            self._params = {}
            self._clock = _FakeClock()
            self.timers = []

        def declare_parameter(self, name, value):
            self._params[name] = value

        def get_parameter(self, name):
            return types.SimpleNamespace(value=self._params[name])

        def create_publisher(self, *_a, **_k):
            return _Recorder()

        def create_subscription(self, *_a, **_k):
            return None

        def create_timer(self, period, cb):
            self.timers.append((period, cb))

        def get_clock(self):
            return self._clock

        def get_logger(self):
            return types.SimpleNamespace(
                info=lambda *_a, **_k: None,
                warn=lambda *_a, **_k: None,
                error=lambda *_a, **_k: None)

    rclpy = types.ModuleType('rclpy')
    rclpy._arm_joy_stub = True
    rclpy.init = lambda *_a, **_k: None
    rclpy.spin = lambda *_a, **_k: None
    rclpy.try_shutdown = lambda *_a, **_k: None
    rclpy_node = types.ModuleType('rclpy.node')
    rclpy_node.Node = _Node
    rclpy_qos = types.ModuleType('rclpy.qos')
    rclpy_qos.QoSProfile = lambda **kw: types.SimpleNamespace(**kw)
    rclpy_qos.QoSDurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1)
    rclpy_qos.QoSHistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)

    def _msg(**defaults):
        def make(**kw):
            return types.SimpleNamespace(**{**defaults, **kw})
        return make

    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs_msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg.Joy = _msg(axes=[], buttons=[])
    sensor_msgs_msg.JointState = _msg(header=types.SimpleNamespace(stamp=None),
                                      name=[], position=[])
    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.Bool = _msg(data=False)
    std_msgs_msg.Float32 = _msg(data=0.0)

    for name, mod in [('rclpy', rclpy), ('rclpy.node', rclpy_node),
                      ('rclpy.qos', rclpy_qos),
                      ('sensor_msgs', sensor_msgs), ('sensor_msgs.msg', sensor_msgs_msg),
                      ('std_msgs', std_msgs), ('std_msgs.msg', std_msgs_msg)]:
        sys.modules[name] = mod


def _load():
    _install_ros_stubs()
    spec = importlib.util.spec_from_file_location('arm_joy_node', NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def node():
    mod = _load()
    n = mod.ArmJoy()
    # Seed the integrator the way arm/joint_states would, at a neutral pose.
    n._target = {'base': 0.0, 'shoulder': 0.0}
    n._grip = 2.0
    n._mod = mod
    return n


def _joy_msg():
    """A DualSense /joy message with everything at rest."""
    return types.SimpleNamespace(axes=[0.0] * 8, buttons=[0] * 13)


def _push(node, base=0.0, shoulder=0.0, grip_dir=0):
    """Pretend a /joy message arrived with these (already-deadzoned) values."""
    node._axes = (base, shoulder)
    node._grip_dir = grip_dir
    node._last_joy = node.get_clock().now()


def _run(node, ticks, fresh=True):
    """Tick the integrator. `fresh` keeps /joy arriving, as it does at ~50 Hz
    while a stick is held; pass False to simulate the controller going away."""
    for _ in range(ticks):
        if fresh:
            node._last_joy = node.get_clock().now()
        node._tick()
        node.get_clock().advance(node._dt)


# ── Deadzone ──────────────────────────────────────────────────────────

def test_deadzone_swallows_resting_stick(node):
    assert node._deadzone(0.0) == 0.0
    assert node._deadzone(0.11) == 0.0
    assert node._deadzone(-0.11) == 0.0


def test_deadzone_starts_from_zero_not_a_jump(node):
    """Just past the deadzone must mean *slow*, not an instant 12% lurch."""
    assert node._deadzone(0.13) == pytest.approx(0.0114, abs=1e-3)
    assert node._deadzone(1.0) == pytest.approx(1.0)
    assert node._deadzone(-1.0) == pytest.approx(-1.0)


# ── Integration ───────────────────────────────────────────────────────

def test_stick_moves_the_target_at_the_configured_rate(node):
    _push(node, base=1.0)
    _run(node, 20)          # 20 ticks @ 20 Hz = 1.0 s
    # scale_base is 0.60 rad/s, so ~0.60 rad after a second.
    assert node._target['base'] == pytest.approx(0.60, abs=0.02)
    assert node._target['shoulder'] == 0.0


def test_publishes_joint_cmd_for_arm_driver(node):
    _push(node, shoulder=1.0)
    _run(node, 5)
    msgs = node.joint_pub.msgs
    assert msgs, 'stick deflection published nothing'
    assert msgs[-1].name == ['base', 'shoulder']
    assert len(msgs[-1].position) == 2


def test_released_stick_stops_commanding(node):
    _push(node, base=1.0)
    _run(node, 5)
    _push(node)                       # let go
    _run(node, 10)
    before = len(node.joint_pub.msgs)
    _run(node, 10)
    # One settling command after release is fine; a stream is not.
    assert len(node.joint_pub.msgs) == before
    assert node._target['base'] == pytest.approx(0.15, abs=0.02)  # holds position


def test_target_clamps_at_joint_limit_no_windup(node):
    """Held past the limit, the target must stop at it — not wind up beyond."""
    _push(node, shoulder=1.0)
    _run(node, 200)                   # 10 s, far past shoulder's 1.57 rad
    assert node._target['shoulder'] == pytest.approx(SHOULDER_LIMIT[1])
    # Reversing must move immediately, which windup would delay.
    _push(node, shoulder=-1.0)
    _run(node, 2)
    assert node._target['shoulder'] < SHOULDER_LIMIT[1]


def test_base_clamps_both_directions(node):
    _push(node, base=-1.0)
    _run(node, 400)
    assert node._target['base'] == pytest.approx(BASE_LIMIT[0])


# ── Safety ────────────────────────────────────────────────────────────

def test_stale_joy_stops_the_arm(node):
    """A dropped controller must not keep integrating its last deflection."""
    _push(node, base=1.0)
    _run(node, 5)
    held = node._target['base']
    node.get_clock().advance(1.0)     # > joy_timeout, no new /joy
    _run(node, 20, fresh=False)
    assert node._target['base'] == pytest.approx(held)


def test_estop_freezes_the_arm(node):
    mod = node._mod
    node._estop_cb(types.SimpleNamespace(data=True))
    _push(node, base=1.0)
    _run(node, 20)
    assert node._target['base'] == 0.0
    assert not node.joint_pub.msgs
    node._estop_cb(types.SimpleNamespace(data=False))
    _push(node, base=1.0)
    _run(node, 5)
    assert node._target['base'] > 0.0


def test_refuses_to_jog_before_feedback(node):
    """T:102 sets every joint at once — jogging from a guessed pose snaps it."""
    node._target = None
    _push(node, base=1.0)
    _run(node, 20)
    assert not node.joint_pub.msgs


# ── Gripper ───────────────────────────────────────────────────────────

def test_open_button_decreases_hand_angle(node):
    """Waveshare T:106: decreasing "cmd" opens the gripper."""
    _push(node, grip_dir=-1)
    _run(node, 5)
    assert node._grip < 2.0
    assert node.grip_pub.msgs[-1].data == pytest.approx(node._grip)


def test_close_button_increases_hand_angle(node):
    _push(node, grip_dir=1)
    _run(node, 5)
    assert node._grip > 2.0


def test_gripper_clamps_to_hand_limits(node):
    _push(node, grip_dir=-1)
    _run(node, 200)
    assert node._grip == pytest.approx(HAND_LIMIT[0])
    _push(node, grip_dir=1)
    _run(node, 200)
    assert node._grip == pytest.approx(HAND_LIMIT[1])


def test_both_gripper_buttons_is_a_no_op(node):
    msg = _joy_msg()
    msg.buttons[node.btn_open] = 1
    msg.buttons[node.btn_close] = 1
    node._joy_cb(msg)
    assert node._grip_dir == 0


# ── R1 / drive dead-man interlock ─────────────────────────────────────

def test_gripper_ignores_r1_while_driving(node):
    """R1 is teleop's dead-man: driving must not crank the gripper open."""
    msg = _joy_msg()
    msg.buttons[node.btn_open] = 1      # R1 held — as it is for every drive
    msg.axes[1] = 0.9                   # left stick pushed forward
    node._joy_cb(msg)
    assert node._grip_dir == 0
    _run(node, 20)
    assert node._grip == 2.0            # gripper never moved
    assert not node.grip_pub.msgs


def test_gripper_works_once_the_drive_stick_is_centred(node):
    msg = _joy_msg()
    msg.buttons[node.btn_open] = 1
    node._joy_cb(msg)                   # left stick at rest
    assert node._grip_dir == -1
    _run(node, 5)
    assert node._grip < 2.0


def test_arm_jog_is_never_locked_out_by_driving(node):
    """Only the gripper is interlocked — the right stick must stay live."""
    msg = _joy_msg()
    msg.axes[1] = 0.9                   # driving
    msg.axes[node.axis_base] = 1.0      # and swinging the arm
    node._joy_cb(msg)
    _run(node, 5)
    assert node._target['base'] > 0.0


def test_lockout_can_be_disabled(node):
    node.drive_lockout = False
    msg = _joy_msg()
    msg.buttons[node.btn_open] = 1
    msg.axes[1] = 0.9
    node._joy_cb(msg)
    assert node._grip_dir == -1


# ── Static wiring ─────────────────────────────────────────────────────

def test_topics_match_arm_driver():
    """The jog is useless if it publishes where the driver isn't listening."""
    node_src = open(NODE).read()
    driver_src = open(ARM_DRIVER).read()
    for topic in ("'arm/joint_cmd'", "'arm/gripper_cmd'", "'arm/joint_states'"):
        assert topic in node_src, f'{topic} missing from arm_joy_node'
        assert topic in driver_src, f'{topic} missing from arm_driver'


def test_joint_limits_agree_with_the_driver():
    mod = _load()
    driver_src = open(ARM_DRIVER).read()
    assert mod.JOINT_LIMITS['base'] == BASE_LIMIT
    assert mod.JOINT_LIMITS['shoulder'] == SHOULDER_LIMIT
    assert mod.JOINT_LIMITS['hand'] == HAND_LIMIT
    # And the driver still agrees, so the duplicated table can't drift.
    for lo, hi in (BASE_LIMIT, SHOULDER_LIMIT, HAND_LIMIT):
        assert f'{lo}, {hi}' in driver_src


def test_gripper_shares_r1_with_the_drive_deadman(node):
    """Pins down *why* the lockout exists: teleop holds R1 for every drive."""
    teleop = open(TELEOP_YAML).read()
    assert 'enable_button: 5' in teleop
    assert 'require_enable_button: true' in teleop
    assert node.btn_open == 5, 'gripper open is no longer on R1 — lockout may be moot'
