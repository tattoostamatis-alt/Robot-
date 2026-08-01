"""Regression tests for the 2026-08-01 audit fixes.

Each test here pins down a bug that was live in the tree and silent — none of
them announced itself in a log, and all three survived a full green test suite,
which is exactly why they get tests now:

  * obstacle_safety_node failed OPEN when object_detector never published at
    all (RealSense dead at bringup, use_camera:=false, YOLO still loading). The
    documented fail-safe only ever covered "published, then stopped", so on the
    boots where the robot is blind it also drove forward unguarded.

  * recovery_manager_node left _status at FAILED forever after one exhausted
    recovery. Both entry points require IDLE, so the FIRST time the robot
    failed to free itself, stuck detection switched off for the rest of the
    session — no log line, no symptom.

  * mission_executor_node tested for IDLE and claimed the slot in two separate
    steps, so two mission/start messages arriving back to back both passed and
    two threads drove the robot. The paired risk is the opposite one: now that
    the slot is claimed up front, a mission that bails out early must still
    release it.

Robot-free: ROS is stubbed, no hardware, no executor.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_safety_regressions.py -q
"""
import importlib.util
import os
import sys
import types

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── shared ROS stubs ────────────────────────────────────────────────────────

def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _ns(**kw):
    return types.SimpleNamespace(**kw)


class _Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class _BaseNode:
    """Enough rclpy.node.Node for a constructor to run: params, pubs, subs,
    timers and a logger. Timers and subscriptions are recorded, never fired —
    the tests call the callbacks directly so timing is explicit."""

    def __init__(self, name):
        self._params = {}
        self.pubs = {}
        self.timers = []
        self.subs = []

    def declare_parameter(self, name, value):
        self._params[name] = value

    def get_parameter(self, name):
        return _ns(value=self._params[name])

    def create_publisher(self, msg_type, topic, qos):
        p = _Publisher()
        self.pubs[topic] = p
        return p

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs.append((topic, cb))
        return None

    def create_client(self, *a, **k):
        return _ns(wait_for_service=lambda timeout_sec=0.0: False,
                   call_async=lambda *_: None)

    def create_timer(self, period, cb):
        self.timers.append((period, cb))
        return None

    def create_service(self, *a, **k):
        return None

    def get_clock(self):
        return _ns(now=lambda: _ns(nanoseconds=self._clock_ns(),
                                   to_msg=lambda: None))

    def _clock_ns(self):
        return 0

    def get_logger(self):
        noop = lambda *a, **k: None            # noqa: E731
        return _ns(info=noop, warn=noop, error=noop, debug=noop,
                   exception=noop)


def _std_msgs():
    return _mod('std_msgs.msg',
                String=lambda data='': _ns(data=data),
                Bool=lambda data=False: _ns(data=data),
                Float32=lambda data=0.0: _ns(data=data))


def _twist():
    return lambda: _ns(linear=_ns(x=0.0, y=0.0, z=0.0),
                       angular=_ns(x=0.0, y=0.0, z=0.0))


def _load(path, modules):
    """Import a node module with `modules` installed in sys.modules, then put
    sys.modules back exactly as it was."""
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location('_uut', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ── obstacle_safety_node ────────────────────────────────────────────────────

@pytest.fixture
def safety():
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      shutdown=lambda *a, **k: None,
                      executors=_ns(ExternalShutdownException=RuntimeError)),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg', Twist=_twist()),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _std_msgs(),
    }
    mod = _load(f'{PKG}/home_robot/nodes/obstacle_safety_node.py', mods)
    node = mod.ObstacleSafetyNode()
    return node


def _drive_forward(node, speed=0.2):
    """Feed one forward cmd_vel and return what reached cmd_vel_safe."""
    msg = _ns(linear=_ns(x=speed, y=0.0, z=0.0),
              angular=_ns(x=0.0, y=0.0, z=0.0))
    node._cmd_vel_cb(msg)
    return node.pubs['cmd_vel_safe'].sent[-1]


def test_forward_blocked_when_detector_never_published(safety, monkeypatch):
    """‼️ THE BUG: a detector that never starts must not read as 'all clear'.

    Before the fix _last_detection_time was None, the staleness check returned
    early, and this exact sequence let full-speed forward motion through on
    every boot where the camera failed to come up."""
    import time as _t
    # Jump past detection_timeout without any detection ever arriving.
    t0 = _t.monotonic()
    monkeypatch.setattr(_t, 'monotonic', lambda: t0 + 60.0)
    safety._check_staleness()

    assert safety.obstacle_blocking is True
    assert _drive_forward(safety).linear.x == 0.0


def test_turning_still_allowed_while_blocked(safety, monkeypatch):
    """The fail-safe blocks forward only — rotating away must stay possible,
    or a blind robot is also a stranded one."""
    import time as _t
    t0 = _t.monotonic()
    monkeypatch.setattr(_t, 'monotonic', lambda: t0 + 60.0)
    safety._check_staleness()

    msg = _ns(linear=_ns(x=0.0, y=0.0, z=0.0),
              angular=_ns(x=0.0, y=0.0, z=0.5))
    safety._cmd_vel_cb(msg)
    out = safety.pubs['cmd_vel_safe'].sent[-1]
    assert out.angular.z == 0.5


def test_fresh_clear_detection_unblocks(safety):
    """A live detector reporting nothing in the way must still let the robot
    drive — the fail-safe must not latch."""
    safety._detections_cb(_ns(data='[]'))
    assert safety.obstacle_blocking is False
    assert _drive_forward(safety).linear.x == pytest.approx(0.2)


def test_close_object_in_centre_blocks_forward(safety):
    """The ordinary path still works after the change."""
    det = ('[{"label":"chair","box_distance":0.3,"img_w":640,'
           '"x1":300,"x2":340}]')
    safety._detections_cb(_ns(data=det))
    assert safety.obstacle_blocking is True
    assert _drive_forward(safety).linear.x == 0.0


# ── recovery_manager_node ───────────────────────────────────────────────────

@pytest.fixture
def recovery():
    action = _mod('nav2_msgs.action',
                  BackUp=_ns(Goal=lambda: _ns(target=None, speed=0.0,
                                              time_allowance=None)),
                  DriveOnHeading=_ns(Goal=lambda: _ns(target=None, speed=0.0,
                                                      time_allowance=None)),
                  NavigateToPose=_ns(Goal=lambda: _ns(pose=None)),
                  Spin=_ns(Goal=lambda: _ns(target_yaw=0.0,
                                            time_allowance=None)))
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'rclpy.action': _mod('rclpy.action',
                             ActionClient=lambda *a, **k: _ns(
                                 wait_for_server=lambda timeout_sec=0.0: False,
                                 send_goal_async=lambda *_: None)),
        'action_msgs': _mod('action_msgs'),
        'action_msgs.srv': _mod('action_msgs.srv',
                                CancelGoal=_ns(Request=lambda: None)),
        'builtin_interfaces': _mod('builtin_interfaces'),
        'builtin_interfaces.msg': _mod('builtin_interfaces.msg',
                                       Duration=lambda sec=0: _ns(sec=sec)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod(
            'geometry_msgs.msg', Twist=_twist(),
            Point=lambda x=0.0, y=0.0, z=0.0: _ns(x=x, y=y, z=z),
            PoseStamped=lambda: _ns(header=_ns(frame_id='', stamp=None),
                                    pose=None)),
        'nav_msgs': _mod('nav_msgs'),
        'nav_msgs.msg': _mod('nav_msgs.msg', Odometry=object, Path=object),
        'nav2_msgs': _mod('nav2_msgs'),
        'nav2_msgs.action': action,
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _std_msgs(),
    }
    mod = _load(f'{PKG}/home_robot/nodes/recovery_manager_node.py', mods)
    node = mod.RecoveryManagerNode()
    return mod, node


def test_exhausted_recovery_returns_to_idle(recovery):
    """‼️ THE BUG: one exhausted recovery used to disable the node for good.

    _check_stuck and _on_trigger both gate on IDLE, and nothing outside
    _recovered() ever restored it, so after the first give-up the robot would
    never attempt to free itself again — silently, until restart."""
    mod, node = recovery
    node._run_recovery(0)

    assert node._status == mod.STATUS_IDLE, (
        'recovery manager stayed latched and will ignore every future stuck')


def test_exhausted_recovery_still_reports_failure(recovery):
    """Reopening the gate must not hide the outcome from subscribers."""
    mod, node = recovery
    node._run_recovery(0)

    published = [m.data for m in node.pubs['recovery/status'].sent]
    assert mod.STATUS_FAILED in published
    spoken = [m.data for m in node.pubs['speech_response'].sent]
    assert any('βοήθεια' in s for s in spoken)


def test_giving_up_does_not_claim_to_be_trying(recovery):
    """The give-up path used to announce "Κόλλησα, προσπαθώ να ξεκολλήσω"
    first and contradict itself one line later."""
    mod, node = recovery
    node._run_recovery(0)

    spoken = [m.data for m in node.pubs['speech_response'].sent]
    assert not any('προσπαθώ' in s for s in spoken)


def test_stuck_detection_works_again_after_a_failure(recovery):
    """The point of the fix, end to end: a second stuck event is still seen."""
    mod, node = recovery
    node._run_recovery(0)
    # _check_stuck bails immediately unless the status gate is open.
    assert node._status == mod.STATUS_IDLE
    node._enabled = True
    node._check_stuck()          # must not raise, must not be short-circuited
    assert node._status == mod.STATUS_IDLE   # no command seen yet -> no stuck


def test_reset_cmd_tracking_clears_the_window(recovery):
    mod, node = recovery
    node._cmd_active = True
    node._cmd_active_since = 123.0
    node._reset_cmd_tracking()
    assert node._cmd_active is False
    assert node._cmd_active_since is None


# ── mission_executor_node ───────────────────────────────────────────────────

@pytest.fixture
def mission():
    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None, ok=lambda: True,
                      try_shutdown=lambda *a, **k: None,
                      time=_ns(Time=lambda: None),
                      duration=_ns(Duration=lambda seconds=0.0: None)),
        'rclpy.node': _mod('rclpy.node', Node=_BaseNode),
        'rclpy.action': _mod('rclpy.action',
                             ActionClient=lambda *a, **k: _ns(
                                 wait_for_server=lambda timeout_sec=0.0: False,
                                 send_goal_async=lambda *_: None)),
        'rclpy.qos': _mod('rclpy.qos',
                          QoSProfile=lambda **k: None,
                          QoSDurabilityPolicy=_ns(TRANSIENT_LOCAL=1),
                          QoSHistoryPolicy=_ns(KEEP_LAST=1)),
        'tf2_ros': _mod('tf2_ros', Buffer=lambda: None,
                        TransformListener=lambda *a, **k: None),
        'action_msgs': _mod('action_msgs'),
        'action_msgs.msg': _mod('action_msgs.msg',
                                GoalStatus=_ns(STATUS_SUCCEEDED=4,
                                               STATUS_CANCELED=5)),
        'ament_index_python': _mod('ament_index_python'),
        'ament_index_python.packages': _mod(
            'ament_index_python.packages',
            get_package_share_directory=lambda _: '/nonexistent'),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod(
            'geometry_msgs.msg', Twist=_twist(),
            PoseStamped=lambda: _ns(
                header=_ns(frame_id='', stamp=None),
                pose=_ns(position=_ns(x=0.0, y=0.0, z=0.0),
                         orientation=_ns(x=0.0, y=0.0, z=0.0, w=1.0)))),
        'nav2_msgs': _mod('nav2_msgs'),
        'nav2_msgs.action': _mod('nav2_msgs.action',
                                 NavigateToPose=_ns(Goal=lambda: _ns(pose=None))),
        'std_msgs': _mod('std_msgs'),
        'std_msgs.msg': _std_msgs(),
        'std_srvs': _mod('std_srvs'),
        'std_srvs.srv': _mod('std_srvs.srv', Empty=_ns(Request=lambda: None)),
    }
    mod = _load(f'{PKG}/home_robot/nodes/mission_executor_node.py', mods)
    node = mod.MissionExecutorNode()
    return mod, node


def test_two_rapid_starts_launch_only_one_mission(mission, monkeypatch):
    """‼️ THE BUG: the IDLE test and the claim were separate steps.

    The state only became non-IDLE later, inside the worker thread, so two
    mission/start messages arriving back to back both passed the gate and two
    threads drove the same robot with competing Nav2 goals."""
    mod, node = mission
    started = []
    monkeypatch.setattr(mod.threading, 'Thread',
                        lambda target, args=(), daemon=False: _ns(
                            start=lambda: started.append(args[0])))

    node._on_mission_start(_ns(data='patrol'))
    node._on_mission_start(_ns(data='patrol'))

    assert started == ['patrol'], f'{len(started)} missions started at once'


def test_slot_released_when_mission_bails_out_early(mission):
    """patrol with no rooms recorded returns without calling _finish. With the
    slot now claimed up front, that path must still hand it back or every later
    mission is refused with 'already running'."""
    mod, node = mission
    node._locations = {}
    node._state = mod.State.NAVIGATING          # as _on_mission_start left it
    node._run_mission('patrol')
    assert node._state == mod.State.IDLE


def test_slot_released_for_an_unknown_mission(mission):
    mod, node = mission
    node._state = mod.State.NAVIGATING
    node._run_mission('πέτα στον Άρη')
    assert node._state == mod.State.IDLE


def test_slot_released_when_a_mission_raises(mission, monkeypatch):
    """A crashing mission must not wedge the executor either."""
    mod, node = mission
    def _boom(_cmd):
        raise RuntimeError('mission blew up')
    monkeypatch.setattr(node, '_dispatch', _boom)
    node._state = mod.State.NAVIGATING
    node._run_mission('patrol')
    assert node._state == mod.State.IDLE


def test_a_new_mission_is_accepted_after_an_early_bail(mission, monkeypatch):
    """The two fixes together: claim, bail, and be ready for the next one."""
    mod, node = mission
    node._locations = {}
    node._state = mod.State.NAVIGATING
    node._run_mission('patrol')

    started = []
    monkeypatch.setattr(mod.threading, 'Thread',
                        lambda target, args=(), daemon=False: _ns(
                            start=lambda: started.append(args[0])))
    node._on_mission_start(_ns(data='patrol'))
    assert started == ['patrol']


def test_navigation_gives_up_when_nav2_never_returns(mission):
    """‼️ THE BUG: this wait had no deadline. A Nav2 server that died after
    accepting the goal parked the mission thread here forever and the state
    never returned to IDLE, so every later mission was refused until restart."""
    mod, node = mission
    node._nav_timeout = 0.3
    cancelled = []
    gh = _ns(accepted=True,
             get_result_async=lambda: _ns(done=lambda: False,   # never completes
                                          result=lambda: None),
             cancel_goal_async=lambda: cancelled.append(True))
    node._nav_ac = _ns(wait_for_server=lambda timeout_sec=0.0: True,
                       send_goal_async=lambda _g: 'fut')
    node._await_future = lambda fut, t: gh

    assert node._navigate_to_xy(1.0, 2.0, 0.0) is False
    assert cancelled, 'the abandoned goal was never cancelled'


def test_spoken_stop_cancels_a_running_mission(mission):
    """A bare "σταμάτα" used to be ignored here unless it also contained
    "αποστολ" — which left the mission driving whenever llm_bridge was down."""
    mod, node = mission
    node._state = mod.State.NAVIGATING
    node._cancel_flag.clear()
    node._on_speech_text(_ns(data='σταμάτα'))
    assert node._cancel_flag.is_set()


def test_idle_chatter_does_not_arm_the_cancel_flag(mission):
    """...but while idle, keep requiring the explicit word, so an unrelated
    "σταμάτα" does not pre-cancel the NEXT mission."""
    mod, node = mission
    node._state = mod.State.IDLE
    node._cancel_flag.clear()
    node._on_speech_text(_ns(data='σταμάτα'))
    assert not node._cancel_flag.is_set()
