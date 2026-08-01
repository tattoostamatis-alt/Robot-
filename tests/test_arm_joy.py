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
ELBOW_LIMIT = (0.0, 3.14)
WRIST_LIMIT = (-1.57, 1.57)
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

        def add_on_set_parameters_callback(self, cb):
            self.param_cb = cb

        def set_parameters(self, params):
            """Route through the node's callback, as rclpy does, then commit."""
            result = self.param_cb(params)
            if getattr(result, 'successful', False):
                for p in params:
                    self._params[p.name] = p.value
            return result

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
    rcl_interfaces = types.ModuleType('rcl_interfaces')
    rcl_interfaces_msg = types.ModuleType('rcl_interfaces.msg')
    rcl_interfaces_msg.SetParametersResult = lambda **kw: types.SimpleNamespace(**kw)
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
                      ('rcl_interfaces', rcl_interfaces),
                      ('rcl_interfaces.msg', rcl_interfaces_msg),
                      ('sensor_msgs', sensor_msgs), ('sensor_msgs.msg', sensor_msgs_msg),
                      ('std_msgs', std_msgs), ('std_msgs.msg', std_msgs_msg)]:
        sys.modules[name] = mod


STUB_MODULES = ('rclpy', 'rclpy.node', 'rclpy.qos',
                'rcl_interfaces', 'rcl_interfaces.msg',
                'sensor_msgs', 'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg')


@pytest.fixture(autouse=True)
def _isolate_ros_stubs():
    """Put sys.modules back afterwards.

    The stubs are installed globally, and leaving them there breaks any later
    test in the run that imports the real rclpy — test_smoke's launch-file
    parsing died with "'rclpy' is not a package" until this existed.
    """
    saved = {name: sys.modules.get(name) for name in STUB_MODULES}
    yield
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _load():
    _install_ros_stubs()
    spec = importlib.util.spec_from_file_location('arm_joy_node', NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Mid-range starting pose, so a test can jog either way without hitting a
# limit. elbow's range is 0..3.14, hence 1.57 rather than 0.
NEUTRAL = {'base': 0.0, 'shoulder': 0.0, 'elbow': 1.57, 'wrist': 0.0}


@pytest.fixture
def node():
    mod = _load()
    n = mod.ArmJoy()
    # Seed the integrator the way arm/joint_states would.
    n._target = {joint: NEUTRAL[joint] for joint, _, _ in n.jog}
    n._grip = 2.0
    n._mod = mod
    return n


def _joy_msg():
    """A DualSense /joy message with everything at rest."""
    return types.SimpleNamespace(axes=[0.0] * 8, buttons=[0] * 13)


def _axis_of(node, joint):
    """The /joy axis index bound to a joint."""
    return next(ax for j, ax, _ in node.jog if j == joint)


def _push(node, grip_dir=0, **axes):
    """Pretend a /joy message arrived with these (already-deadzoned) values.

    Keyword args are joint names, e.g. _push(node, base=1.0, elbow=-1.0).
    """
    unknown = set(axes) - {joint for joint, _, _ in node.jog}
    assert not unknown, f'not a jogged joint: {unknown}'
    node._axes = {joint: axes.get(joint, 0.0) for joint, _, _ in node.jog}
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

def _scale_of(node, joint):
    """The configured rad/s for a joint — read, not hardcoded, so retuning the
    defaults doesn't silently invalidate every rate assertion below."""
    return next(sc for j, _, sc in node.jog if j == joint)


def test_stick_moves_the_target_at_the_configured_rate(node):
    _push(node, base=1.0)
    _run(node, 20)          # 20 ticks @ 20 Hz = 1.0 s
    # One second at full deflection covers exactly one scale's worth of travel.
    assert node._target['base'] == pytest.approx(_scale_of(node, 'base'), abs=0.02)
    assert node._target['shoulder'] == 0.0


def test_publishes_joint_cmd_for_arm_driver(node):
    _push(node, shoulder=1.0)
    _run(node, 5)
    msgs = node.joint_pub.msgs
    assert msgs, 'stick deflection published nothing'
    # Every jogged joint goes in each message: T:102 sets them all at once, so
    # omitting one would let the driver fill it from lagging feedback.
    assert msgs[-1].name == [joint for joint, _, _ in node.jog]
    assert len(msgs[-1].position) == len(node.jog)
    assert 'elbow' in msgs[-1].name, 'reach joint must be commandable'


def test_released_stick_stops_commanding(node):
    _push(node, base=1.0)
    _run(node, 5)
    _push(node)                       # let go
    _run(node, 10)
    before = len(node.joint_pub.msgs)
    _run(node, 10)
    # One settling command after release is fine; a stream is not.
    assert len(node.joint_pub.msgs) == before
    assert node._target['base'] == pytest.approx(_scale_of(node, 'base') * 0.25,
                                                 abs=0.02)  # holds position


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


# ── Reach (the joint the first version forgot) ─────────────────────────

def test_dpad_extends_the_elbow(node):
    """Reaching forward is the elbow; without it the arm cannot extend."""
    assert 'elbow' in dict((j, ax) for j, ax, _ in node.jog)
    _push(node, elbow=1.0)
    _run(node, 20)
    assert node._target['elbow'] == pytest.approx(1.57 + _scale_of(node, 'elbow'),
                                                  abs=0.02)


def test_elbow_and_wrist_are_on_the_dpad_not_the_sticks(node):
    """The sticks are taken (drive + base/shoulder), so these must be D-pad."""
    dpad = (6, 7)
    assert _axis_of(node, 'elbow') in dpad
    assert _axis_of(node, 'wrist') in dpad


def test_elbow_clamps_at_its_asymmetric_limits(node):
    """elbow is 0..3.14, not centred on zero like the others."""
    _push(node, elbow=-1.0)
    _run(node, 400)
    assert node._target['elbow'] == pytest.approx(ELBOW_LIMIT[0])
    _push(node, elbow=1.0)
    _run(node, 400)
    assert node._target['elbow'] == pytest.approx(ELBOW_LIMIT[1])


def test_wrist_clamps(node):
    _push(node, wrist=1.0)
    _run(node, 400)
    assert node._target['wrist'] == pytest.approx(WRIST_LIMIT[1])


def test_joints_jog_independently(node):
    """Pushing the D-pad must not disturb base/shoulder, and vice versa."""
    _push(node, elbow=1.0)
    _run(node, 10)
    assert node._target['base'] == 0.0
    assert node._target['shoulder'] == 0.0
    assert node._target['elbow'] > 1.57


def test_an_axis_of_minus_one_unbinds_a_joint():
    """Escape hatch: axis_wrist: -1 drops the wrist without a code change.

    Exercises the real __init__ by overriding the parameter as a launch file
    would, rather than recomputing the binding list here.
    """
    mod = _load()

    class Unbound(mod.ArmJoy):
        def declare_parameter(self, name, value):
            super().declare_parameter(name, -1 if name == 'axis_wrist' else value)

    bound = [joint for joint, _, _ in Unbound().jog]
    assert 'wrist' not in bound
    assert 'elbow' in bound, 'unbinding one joint must not drop the others'


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
    msg.axes[_axis_of(node, 'base')] = 1.0      # and swinging the arm
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


# ── Live retuning ─────────────────────────────────────────────────────

def _param(name, value):
    return types.SimpleNamespace(name=name, value=value)


def test_speed_change_applies_without_a_restart(node):
    """Documented as live-tunable, so it must actually take effect live."""
    node.set_parameters([_param('scale_base', 2.0)])
    _push(node, base=1.0)
    _run(node, 20)                      # 1 s
    assert node._target['base'] == pytest.approx(2.0, abs=0.05)


def test_flipping_a_scale_sign_reverses_that_joint(node):
    node.set_parameters([_param('scale_shoulder', -1.0)])
    _push(node, shoulder=1.0)
    _run(node, 10)
    assert node._target['shoulder'] < 0.0, 'negated scale must reverse direction'


def test_retuning_one_joint_leaves_the_others_alone(node):
    before = {joint: sc for joint, _, sc in node.jog}
    node.set_parameters([_param('scale_base', 2.0)])
    after = {joint: sc for joint, _, sc in node.jog}
    assert after['base'] == 2.0
    for joint in ('shoulder', 'elbow', 'wrist'):
        assert after[joint] == before[joint]


def test_unbinding_an_axis_live_drops_its_target(node):
    """A stale target for an unbound joint would keep being published."""
    node.set_parameters([_param('axis_wrist', -1)])
    assert 'wrist' not in [j for j, _, _ in node.jog]
    assert 'wrist' not in node._target
    assert 'wrist' not in node._axes
    _push(node, base=1.0)
    _run(node, 5)
    assert 'wrist' not in node.joint_pub.msgs[-1].name


def test_unrelated_parameter_change_is_accepted(node):
    result = node.set_parameters([_param('joy_timeout', 1.0)])
    assert result.successful


# ── max_lead leash ────────────────────────────────────────────────────

def _feedback(**pos):
    full = {'base': 0.0, 'shoulder': 0.0, 'elbow': 1.57, 'wrist': 0.0, 'hand': 2.0}
    full.update(pos)
    return types.SimpleNamespace(name=list(full), position=list(full.values()))


def test_target_cannot_sprint_past_a_lagging_servo(node):
    """Fast scales must not let the target run away from a stalled joint."""
    _push(node, base=1.0)
    _run(node, 60)                      # 3 s at 1.2 rad/s ≈ 3.6 rad of demand
    # The arm reports it never left the start: the leash must reel the target in.
    node._state_cb(_feedback(base=0.0))
    assert abs(node._target['base']) <= node.max_lead + 1e-6


def test_leash_does_not_slow_a_servo_that_keeps_up(node):
    """The whole point of open-loop: keeping up must not cost any speed."""
    for _ in range(4):
        _push(node, base=1.0)
        _run(node, 5)
        node._state_cb(_feedback(base=node._target['base']))   # arm tracks
    assert node._target['base'] == pytest.approx(_scale_of(node, 'base'), abs=0.05)


def test_leash_reels_in_from_either_direction(node):
    _push(node, shoulder=-1.0)
    _run(node, 60)
    node._state_cb(_feedback(shoulder=0.0))
    assert node._target['shoulder'] >= -node.max_lead - 1e-6


def test_leash_respects_joint_limits(node):
    """Reeling in must not park the target outside the joint's range."""
    _push(node, elbow=-1.0)
    _run(node, 200)
    node._state_cb(_feedback(elbow=0.0))
    lo, hi = ELBOW_LIMIT
    assert lo <= node._target['elbow'] <= hi


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
