"""Tests for the localization watchdog (global_localizer_node).

Why this exists: localization was a ONE-SHOT event at startup. AMCL only
corrects itself while the robot drives — no odometry motion means no
resampling — so a pose that came out 0.2-0.4 m off just sat there being wrong
for the whole session. In RViz that is the recurring complaint "οι κόκκινες
γραμμές είναι λάθος στον χάρτη": the laser lands next to the walls instead of
on them, and only a manual 2D Pose Estimate (or a drive) fixed it.

The watchdog re-measures the live scan against the map every few seconds and
snaps the pose back by itself. That makes it the one piece of this node that
runs unattended forever, so the things worth pinning down are the guards, not
the happy path:

  * it corrects a drifted pose, and the correction really is an improvement
  * it publishes TIGHT covariance — a nudge must not blow AMCL's converged
    particle cloud back open, which would undo what it is protecting
  * it does NOT act on a moving robot (scan and TF are from different instants,
    and a jump under a live Nav2 goal yanks the path sideways)
  * it does NOT act when the pose is already good (no pointless particle resets)
  * a sustained GROSS error escalates to a full global search instead of being
    nudged, but only after several strikes, never on one bad scan
  * when nothing better exists nearby (clutter the map does not have) it backs
    off exponentially instead of burning CPU every tick forever

Robot-free: numpy and scipy are real, ROS is stubbed, no map or lidar hardware.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_localization_watchdog.py -q
"""
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = f'{PKG}/home_robot/nodes/global_localizer_node.py'

STUB_MODULES = ('rclpy', 'rclpy.node', 'rclpy.qos', 'rclpy.time',
                'geometry_msgs', 'geometry_msgs.msg',
                'nav_msgs', 'nav_msgs.msg',
                'sensor_msgs', 'sensor_msgs.msg',
                'std_srvs', 'std_srvs.srv', 'tf2_ros')

MOUNT_YAW, MOUNT_X, MOUNT_Y = math.pi, 0.12, 0.0   # this robot's real mount

RES = 0.05
SIZE = 120           # cells -> 6 m
WALL = 100
FREE = 0


def _install_stubs():
    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def _msg(**defaults):
        return lambda **kw: types.SimpleNamespace(**{**defaults, **kw})

    class _Node:
        def __init__(self, name):
            self._params = {}

        def declare_parameter(self, name, value):
            self._params[name] = value

        def get_parameter(self, name):
            return types.SimpleNamespace(value=self._params[name])

        def create_subscription(self, *a, **k):
            return None

        def create_publisher(self, *a, **k):
            return types.SimpleNamespace(publish=lambda *_: None)

        def create_service(self, *a, **k):
            return None

        def create_timer(self, *a, **k):
            return None

        def get_clock(self):
            return types.SimpleNamespace(
                now=lambda: types.SimpleNamespace(to_msg=lambda: None))

        def get_logger(self):
            return types.SimpleNamespace(info=lambda *a, **k: None,
                                         warn=lambda *a, **k: None,
                                         error=lambda *a, **k: None)

    def _cov_msg(**kw):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=None, frame_id=''),
            pose=types.SimpleNamespace(
                pose=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)),
                covariance=[0.0] * 36))

    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_Node),
        'rclpy.time': _mod('rclpy.time', Time=lambda *a, **k: None),
        'rclpy.qos': _mod('rclpy.qos',
                          QoSProfile=lambda **kw: types.SimpleNamespace(**kw),
                          QoSDurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1),
                          QoSReliabilityPolicy=types.SimpleNamespace(RELIABLE=1)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg',
                                  PoseWithCovarianceStamped=_cov_msg),
        'nav_msgs': _mod('nav_msgs'),
        'nav_msgs.msg': _mod('nav_msgs.msg', OccupancyGrid=_msg()),
        'sensor_msgs': _mod('sensor_msgs'),
        'sensor_msgs.msg': _mod('sensor_msgs.msg', LaserScan=_msg(),
                                Image=_msg(), CameraInfo=_msg()),
        'std_srvs': _mod('std_srvs'),
        'std_srvs.srv': _mod('std_srvs.srv', Empty=object),
        'tf2_ros': _mod('tf2_ros', Buffer=lambda: None,
                        TransformListener=lambda *a, **k: None),
    }
    sys.modules.update(mods)


@pytest.fixture(autouse=True)
def _isolate_stubs():
    """Restore sys.modules — the stubs would otherwise break later tests that
    import the real rclpy (test_smoke's launch parsing)."""
    saved = {n: sys.modules.get(n) for n in STUB_MODULES}
    yield
    for n, m in saved.items():
        if m is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = m


def _load():
    _install_stubs()
    spec = importlib.util.spec_from_file_location('global_localizer_node', NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _room_map():
    """The same deliberately ASYMMETRIC 6x6 m floor plan as the sibling test:
    a bare square room is rotationally ambiguous and would let a wrong pose
    score well, which makes these tests pass for the wrong reason."""
    grid = np.full((SIZE, SIZE), FREE, dtype=np.int8)
    grid[0, :] = grid[-1, :] = WALL
    grid[:, 0] = grid[:, -1] = WALL
    grid[0:70, 40] = WALL
    grid[85, 40:110] = WALL
    grid[95:110, 20:30] = WALL
    grid[25:35, 95:115] = WALL
    return types.SimpleNamespace(
        data=grid.flatten().tolist(),
        info=types.SimpleNamespace(
            width=SIZE, height=SIZE, resolution=RES,
            origin=types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.0, y=0.0))))


def _raycast_scan(map_msg, lx, ly, lyaw, n=360, max_range=8.0):
    grid = np.array(map_msg.data, dtype=np.int8).reshape(SIZE, SIZE)
    ranges = []
    for i in range(n):
        a = lyaw + 2 * math.pi * i / n
        r, hit = 0.0, max_range
        while r < max_range:
            r += RES / 2
            gx = int(round((lx + r * math.cos(a)) / RES))
            gy = int(round((ly + r * math.sin(a)) / RES))
            if not (0 <= gx < SIZE and 0 <= gy < SIZE):
                break
            if grid[gy, gx] == WALL:
                hit = r
                break
        ranges.append(hit)
    return types.SimpleNamespace(
        ranges=ranges, angle_min=0.0, angle_max=2 * math.pi,
        angle_increment=2 * math.pi / n, range_min=0.02, range_max=max_range,
        header=types.SimpleNamespace(frame_id='laser'))


DEFAULTS = {
    'watchdog_max_err':     0.12,
    'watchdog_min_gain':    0.03,
    'watchdog_max_shift':   0.50,
    'watchdog_lost_err':    0.30,
    'watchdog_lost_strikes': 3,
    'watchdog_cooldown':    20.0,
}


@pytest.fixture
def node():
    mod = _load()
    n = mod.GlobalLocalizerNode.__new__(mod.GlobalLocalizerNode)
    n._params = dict(DEFAULTS)
    n._dist = None
    n._lock = _NullLock()
    n._scan = None
    n._map = None
    n._running = False
    n._wd_last_pose = None
    n._wd_strikes = 0
    n._wd_next_ok = 0.0
    n._wd_backoff = 0.0
    logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                   warn=lambda *a, **k: None,
                                   error=lambda *a, **k: None)
    n.get_logger = lambda: logger
    n.get_parameter = lambda name: types.SimpleNamespace(value=n._params[name])
    n.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(to_msg=lambda: None))
    # No TF buffer in a unit test: _base_laser_tf falls back to the documented
    # real mount, and _live_laser_pose is stubbed per-test instead.
    n._tf_buffer = types.SimpleNamespace(
        lookup_transform=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no tf')))
    n.published = []
    n._pub = types.SimpleNamespace(publish=lambda m: n.published.append(m))
    n.global_runs = []
    n._mod = mod
    return n


def _arm(node, truth_base, believed_base, moving=False):
    """Robot really at truth_base; AMCL believes believed_base (both base_link).

    Leaves the node one tick away from acting: the watchdog needs a previous
    sample to judge motion, so this primes it and returns the tick function.
    """
    map_msg = _room_map()
    tlx, tly, tlyaw = node._base_to_laser(*truth_base)
    node._map = map_msg
    node._scan = _raycast_scan(map_msg, tlx, tly, tlyaw)
    node._dist = None

    believed_laser = node._base_to_laser(*believed_base)
    node._live_laser_pose = lambda: believed_laser

    # Prime the motion baseline. A second, displaced sample = still driving.
    node._watchdog()
    if moving:
        moved = (believed_laser[0] + 0.5, believed_laser[1], believed_laser[2])
        node._live_laser_pose = lambda: moved
    return node


def _threading_capture(node):
    """Capture full-global-search launches instead of spawning threads."""
    class _Thread:
        def __init__(self, target=None, daemon=None):
            node.global_runs.append(target)

        def start(self):
            pass
    node._mod.threading = types.SimpleNamespace(Thread=_Thread)


# ── it corrects a drifted pose ────────────────────────────────────────

def test_a_drifted_pose_is_snapped_back_onto_the_walls(node):
    """The whole point: nobody touches RViz and the red lines come good."""
    truth = (1.50, 1.50, 0.70)
    _arm(node, truth, believed_base=(1.72, 1.68, 0.70))
    node._watchdog()

    assert len(node.published) == 1, 'watchdog should have corrected the pose'
    p = node.published[0].pose.pose.position
    before = math.hypot(1.72 - truth[0], 1.68 - truth[1])
    after  = math.hypot(p.x - truth[0], p.y - truth[1])
    assert after < before, f'correction made it worse: {before:.3f} -> {after:.3f}'
    # The watchdog searches a COARSER grid than the startup refine (it runs
    # every few seconds, not once), so one pass lands close rather than exact —
    # and the next pass, 20 s later, tightens it further.
    assert after < 0.15


def test_the_correction_is_published_with_tight_covariance(node):
    """A nudge must not reopen AMCL's converged cloud — that is a reset, not a
    correction, and it would undo the convergence the watchdog is protecting."""
    _arm(node, (1.50, 1.50, 0.70), believed_base=(1.72, 1.68, 0.70))
    node._watchdog()

    cov = node.published[0].pose.covariance
    assert cov[0] <= 0.05 and cov[7] <= 0.05
    assert cov[35] <= 0.02


def test_the_pose_is_in_base_link_not_laser_frame(node):
    """AMCL's /initialpose is base_link. Publishing the laser pose instead puts
    the robot 0.12 m out and 180 deg round, on this yaw=pi mount."""
    truth = (1.50, 1.50, 0.70)
    _arm(node, truth, believed_base=(1.72, 1.68, 0.70))
    node._watchdog()

    q = node.published[0].pose.pose.orientation
    yaw = 2.0 * math.atan2(q.z, q.w)
    assert math.cos(yaw - truth[2]) > 0.99, 'published yaw looks like the laser'


# ── the guards ────────────────────────────────────────────────────────

def test_a_good_pose_is_left_alone(node):
    """No pointless particle resets when the scan already sits on the walls."""
    truth = (1.50, 1.50, 0.70)
    _arm(node, truth, believed_base=truth)
    node._watchdog()
    assert node.published == []


def test_a_moving_robot_is_never_corrected(node):
    """Scan and TF are from different instants while driving, and a jump under
    a live Nav2 goal yanks the path sideways. AMCL is converging anyway."""
    _arm(node, (1.50, 1.50, 0.70), believed_base=(1.72, 1.68, 0.70), moving=True)
    node._watchdog()
    assert node.published == []


def test_motion_clears_accumulated_strikes(node):
    """Strikes must not survive a drive, or the robot re-searches for free."""
    _arm(node, (1.50, 1.50, 0.70), believed_base=(4.30, 4.10, 0.70), moving=True)
    node._wd_strikes = 2
    node._watchdog()
    assert node._wd_strikes == 0


def test_the_cooldown_blocks_back_to_back_corrections(node):
    """Two corrections in a row would fight AMCL instead of letting it settle."""
    _arm(node, (1.50, 1.50, 0.70), believed_base=(1.72, 1.68, 0.70))
    node._watchdog()
    assert len(node.published) == 1
    node._wd_last_pose = None
    node._watchdog()          # prime again
    node._watchdog()          # would act, but is inside the cooldown
    assert len(node.published) == 1


def test_nothing_happens_while_a_global_search_is_running(node):
    _arm(node, (1.50, 1.50, 0.70), believed_base=(1.72, 1.68, 0.70))
    node._running = True
    node._watchdog()
    assert node.published == []


# ── escalation to a full global search ────────────────────────────────

def test_a_grossly_wrong_pose_escalates_after_repeated_strikes(node):
    """Carried to another room: nudging ±0.3 m cannot help, so search the map."""
    _threading_capture(node)
    _arm(node, (1.50, 1.50, 0.70), believed_base=(4.30, 4.10, 0.70))

    for _ in range(DEFAULTS['watchdog_lost_strikes']):
        node._watchdog()

    assert len(node.global_runs) == 1, 'should have re-run global localization'
    assert node.published == [], 'a lost robot must not get a local nudge'


def test_one_bad_scan_does_not_trigger_a_global_search(node):
    """Someone walking past the lidar must not restart localization."""
    _threading_capture(node)
    _arm(node, (1.50, 1.50, 0.70), believed_base=(4.30, 4.10, 0.70))
    node._watchdog()
    assert node.global_runs == []


# ── backing off ───────────────────────────────────────────────────────

def test_it_backs_off_when_no_better_pose_exists_nearby(node):
    """Clutter the map does not have (a new sofa) keeps the error up forever.
    Re-searching every tick would burn CPU for nothing, so the interval grows."""
    # A pose that IS a little off (so the check fires) but where no achievable
    # improvement counts as good enough — the sofa is not in the map, so the
    # residual will not come down no matter how often we look.
    _arm(node, (1.50, 1.50, 0.70), believed_base=(1.72, 1.68, 0.70))
    node._params['watchdog_min_gain'] = 9.9    # nothing can ever be good enough
    node._watchdog()
    first = node._wd_backoff
    assert first >= DEFAULTS['watchdog_cooldown']
    assert node.published == []

    node._wd_next_ok = 0.0
    node._wd_last_pose = None
    node._watchdog()          # prime
    node._watchdog()          # act again
    assert node._wd_backoff > first, 'back-off should grow, not repeat'
