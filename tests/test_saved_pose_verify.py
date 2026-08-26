"""Tests for verifying a saved pose before trusting it (global_localizer_node).

Why this exists: global_localizer deferred to pose_saver's file whenever one
existed, with nothing checking it. The saved pose is only correct if nobody
moved the robot while it was off — and when someone had, the robot booted
confidently wrong and had to be driven to the AprilTag to be corrected. Now the
scan is scored against the map at the saved pose and the full-map search runs
automatically when it doesn't fit.

Two things are worth pinning down:

  * The base_link <-> laser conversion. This lidar is mounted yaw=pi, so a sign
    slip here does not fail loudly — it scores a pose 180 deg out, which is
    exactly the "flip" the node's own comments describe fighting before.
  * The accept/reject decision, including that an unreadable file means
    "search", not "trust".

Robot-free: numpy and scipy are real, ROS is stubbed, no map or lidar hardware.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_saved_pose_verify.py -q
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

# This robot's real lidar mount: yaw=pi, 0.12 m forward of base_link.
MOUNT_YAW, MOUNT_X, MOUNT_Y = math.pi, 0.12, 0.0


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

        def get_logger(self):
            return types.SimpleNamespace(info=lambda *a, **k: None,
                                         warn=lambda *a, **k: None,
                                         error=lambda *a, **k: None)

    mods = {
        'rclpy': _mod('rclpy', init=lambda *a, **k: None,
                      spin=lambda *a, **k: None,
                      try_shutdown=lambda *a, **k: None),
        'rclpy.node': _mod('rclpy.node', Node=_Node),
        # Time() must answer to_msg(): the restored pose is stamped with a zero
        # Time, not now(), or AMCL refuses it. See test_log_tab_noise.py.
        'rclpy.time': _mod(
            'rclpy.time',
            Time=lambda *a, **k: types.SimpleNamespace(
                to_msg=lambda: types.SimpleNamespace(sec=0, nanosec=0))),
        'rclpy.qos': _mod('rclpy.qos',
                          QoSProfile=lambda **kw: types.SimpleNamespace(**kw),
                          QoSDurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1),
                          QoSReliabilityPolicy=types.SimpleNamespace(RELIABLE=1)),
        'geometry_msgs': _mod('geometry_msgs'),
        'geometry_msgs.msg': _mod('geometry_msgs.msg',
                                  PoseWithCovarianceStamped=_msg()),
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


def test_global_match_gate_accepts_clear_line_of_sight():
    mod = _load()
    assert mod._global_match_is_safe(0.10, 0.30)
    assert mod._global_match_is_safe(0.30, 0.30)


def test_global_match_gate_rejects_wrong_room_hypotheses():
    mod = _load()
    assert not mod._global_match_is_safe(0.48, 0.30)
    assert not mod._global_match_is_safe(float('nan'), 0.30)


# ── A tiny synthetic world ────────────────────────────────────────────
#
# 6 m x 6 m room, walls on the border, 0.05 m cells. Scans are generated by
# ray-casting from a known pose, so "the robot really is here" is ground truth.

RES = 0.05
SIZE = 120           # cells -> 6 m
WALL = 100
FREE = 0


def _room_map(mod):
    """A deliberately ASYMMETRIC 6x6 m floor plan.

    Every feature here earns its place: a bare square room is rotationally
    ambiguous, and a divider down the middle makes the two halves
    indistinguishable — a scan from one half then scores *well* against the
    other, so "wrong place" tests pass for the wrong reason. Off-centre walls of
    unequal length plus two differently-placed blocks make each spot in the map
    look like nowhere else, which is what the real flat is like.
    """
    grid = np.full((SIZE, SIZE), FREE, dtype=np.int8)
    grid[0, :] = grid[-1, :] = WALL
    grid[:, 0] = grid[:, -1] = WALL
    grid[0:70, 40] = WALL          # off-centre divider, lower part only
    grid[85, 40:110] = WALL        # a stub wall at a different angle
    grid[95:110, 20:30] = WALL     # block, bottom-left-ish
    grid[25:35, 95:115] = WALL     # block, elsewhere and a different shape
    return types.SimpleNamespace(
        data=grid.flatten().tolist(),
        info=types.SimpleNamespace(
            width=SIZE, height=SIZE, resolution=RES,
            origin=types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.0, y=0.0))))


def _raycast_scan(map_msg, lx, ly, lyaw, n=360, max_range=8.0):
    """Ranges a laser at (lx, ly, lyaw) would see in this map."""
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


@pytest.fixture
def node():
    mod = _load()
    n = mod.GlobalLocalizerNode.__new__(mod.GlobalLocalizerNode)
    n._params = {}
    n._dist = None
    n._lock = _NullLock()
    n._scan = None
    n._map = None
    mod_logger = types.SimpleNamespace(info=lambda *a, **k: None,
                                       warn=lambda *a, **k: None,
                                       error=lambda *a, **k: None)
    n.get_logger = lambda: mod_logger
    n.get_parameter = lambda name: types.SimpleNamespace(value=n._params[name])
    n._params['saved_pose_max_err'] = 0.30
    # No TF in a unit test: exercise the documented fallback mount, which is
    # this robot's real geometry.
    n._tf_buffer = types.SimpleNamespace(
        lookup_transform=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no tf')))
    n._mod = mod
    return n


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ── base_link <-> laser round trip ────────────────────────────────────

def test_base_to_laser_is_the_inverse_of_laser_to_base(node):
    """A sign slip here scores a pose 180 deg out instead of failing loudly."""
    for bx, by, byaw in [(1.0, 2.0, 0.0), (2.5, 1.5, 1.2),
                         (0.5, 0.5, -2.0), (3.0, 3.0, math.pi)]:
        lx, ly, lyaw = node._base_to_laser(bx, by, byaw)
        rx, ry, ryaw = node._laser_to_base(lx, ly, lyaw)
        assert rx == pytest.approx(bx, abs=1e-9)
        assert ry == pytest.approx(by, abs=1e-9)
        assert math.cos(ryaw - byaw) == pytest.approx(1.0, abs=1e-9)


def test_base_to_laser_applies_the_real_mount(node):
    """Facing +x, a laser 0.12 m forward sits 0.12 m ahead and faces backwards."""
    lx, ly, lyaw = node._base_to_laser(1.0, 1.0, 0.0)
    assert (lx, ly) == pytest.approx((1.0 + MOUNT_X, 1.0))
    assert math.cos(lyaw - MOUNT_YAW) == pytest.approx(1.0, abs=1e-9)


def test_mount_offset_rotates_with_the_robot(node):
    """Facing +y, the same 0.12 m offset must land along +y, not +x."""
    lx, ly, _ = node._base_to_laser(1.0, 1.0, math.pi / 2)
    assert (lx, ly) == pytest.approx((1.0, 1.0 + MOUNT_X), abs=1e-9)


# ── Scoring a pose against the scan ───────────────────────────────────

def _score_at(node, base_pose, scan_from_base):
    """median scan->wall error for base_pose, given a scan taken at
    scan_from_base."""
    map_msg = _room_map(node._mod)
    lx, ly, lyaw = node._base_to_laser(*scan_from_base)
    scan = _raycast_scan(map_msg, lx, ly, lyaw)
    lidar = node._mod._lidar_pts(scan)
    node._dist = None
    qx, qy, qyaw = node._base_to_laser(*base_pose)
    return node._wall_err_fn(lidar, map_msg)(qx, qy, qyaw)


def test_the_true_pose_scores_near_zero(node):
    truth = (1.5, 1.5, 0.7)
    assert _score_at(node, truth, truth) < 0.10


def test_a_pose_in_the_wrong_place_scores_badly(node):
    """Carried across the room: the scan cannot fit where it thinks it is."""
    truth = (1.5, 1.5, 0.7)
    assert _score_at(node, (4.5, 4.0, 0.7), scan_from_base=truth) > 0.30


def test_a_rotated_pose_scores_badly(node):
    truth = (1.5, 1.5, 0.7)
    assert _score_at(node, (1.5, 1.5, 0.7 + math.pi / 2), scan_from_base=truth) > 0.30


def test_a_small_drift_still_passes(node):
    """Odometry drift of a few cm must not trigger a full re-search."""
    truth = (1.5, 1.5, 0.7)
    assert _score_at(node, (1.55, 1.53, 0.72), scan_from_base=truth) < 0.30


# ── The accept / reject decision ──────────────────────────────────────

def _prepare(node, saved, truth):
    map_msg = _room_map(node._mod)
    lx, ly, lyaw = node._base_to_laser(*truth)
    node._map = map_msg
    node._scan = _raycast_scan(map_msg, lx, ly, lyaw)
    node._dist = None
    return saved


def _pose_file(tmp_path, x, y, yaw):
    p = tmp_path / 'last_amcl_pose_test.yaml'
    p.write_text(f'x: {x}\ny: {y}\nyaw: {yaw}\n')
    return str(p)


def test_saved_pose_accepted_when_it_fits(node, tmp_path):
    truth = (1.5, 1.5, 0.7)
    _prepare(node, truth, truth)
    fits, err, where = node._saved_pose_fits(_pose_file(tmp_path, *truth))
    assert fits
    assert err < 0.30
    assert '1.50' in where


def test_saved_pose_rejected_when_the_robot_was_moved(node, tmp_path):
    """The whole point: this is what replaces the trip to the AprilTag."""
    _prepare(node, None, truth=(1.5, 1.5, 0.7))
    fits, err, _ = node._saved_pose_fits(_pose_file(tmp_path, 4.5, 4.0, 0.7))
    assert not fits
    assert err > 0.30


def test_unreadable_pose_file_means_search(node, tmp_path):
    """Fail toward looking for yourself, not toward trusting garbage."""
    bad = tmp_path / 'broken.yaml'
    bad.write_text('this: [is not, a pose\n')
    fits, _, where = node._saved_pose_fits(str(bad))
    assert not fits
    assert 'unreadable' in where


def test_missing_keys_mean_search(node, tmp_path):
    p = tmp_path / 'partial.yaml'
    p.write_text('x: 1.0\ny: 2.0\n')          # no yaw
    fits, _, _ = node._saved_pose_fits(str(p))
    assert not fits


def test_returns_none_until_scan_and_map_arrive(node, tmp_path):
    """Undecided must be distinct from 'does not fit' — otherwise the node would
    launch a full search on the first scan-less callback."""
    node._scan = None
    node._map = None
    assert node._saved_pose_fits(_pose_file(tmp_path, 1.5, 1.5, 0.7)) is None


def test_threshold_is_configurable(node, tmp_path):
    truth = (1.5, 1.5, 0.7)
    _prepare(node, None, truth)
    moved = _pose_file(tmp_path, 4.5, 4.0, 0.7)
    node._params['saved_pose_max_err'] = 0.30
    assert not node._saved_pose_fits(moved)[0]
    node._params['saved_pose_max_err'] = 99.0      # accept anything
    assert node._saved_pose_fits(moved)[0]


# ── Wiring ────────────────────────────────────────────────────────────

def test_verification_is_on_by_default_and_can_be_turned_off():
    src = open(NODE).read()
    assert "declare_parameter('verify_saved_pose', True)" in src
    assert "declare_parameter('saved_pose_max_err'" in src
    # The blind-defer path must still exist for anyone who wants it back.
    assert "get_parameter('verify_saved_pose')" in src


def test_undecided_verdict_does_not_fall_through_to_searching():
    """`if verdict is None: return` must come before the fits/err unpacking."""
    src = open(NODE).read()
    block = src[src.index('verdict = self._saved_pose_fits'):]
    block = block[:block.index('self._auto = False')]
    assert 'if verdict is None' in block
    assert block.index('if verdict is None') < block.index('fits, err, where')


# ── Candidate selection (_pick) ───────────────────────────────────────
#
# The bug this replaces: only the top FFT candidate was ever refined, and the
# FFT correlation peak rewards putting scan points near *any* walls. On
# 2026-07-29 that landed the robot 4.4 m away in another room while reporting a
# 0.071 m fit — because that one candidate was the only one measured.

def _cand(err, x, y, yaw=0.0):
    return (err, x, y, yaw)


def test_pick_takes_the_best_fit_when_it_is_clear(node):
    node._params['prior_tie_margin'] = 0.05
    refined = [_cand(0.05, 1.0, 1.0), _cand(0.40, 5.0, 5.0)]
    assert node._pick(refined)[1:3] == (1.0, 1.0)


def test_pick_ignores_a_worse_fit_even_if_nearer_the_prior(node, tmp_path):
    """The prior is a tie-break, not an override — a moved robot must still get
    the pose the scan supports."""
    node._params['prior_tie_margin'] = 0.05
    node._params['map_yaml'] = _pose_file(tmp_path, 5.0, 5.0, 0.0)
    refined = [_cand(0.05, 1.0, 1.0), _cand(0.40, 5.0, 5.0)]
    assert node._pick(refined)[1:3] == (1.0, 1.0)


def test_pick_breaks_a_genuine_tie_toward_the_prior(node, tmp_path, monkeypatch):
    """Two rooms fitting equally well is the real failure mode; staying put is
    likelier than having been carried to the lookalike."""
    node._params['prior_tie_margin'] = 0.05
    monkeypatch.setattr(node, '_saved_pose_laser', lambda: (5.0, 5.0, 0.0))
    refined = [_cand(0.070, 1.0, 1.0), _cand(0.072, 5.0, 5.0)]
    assert node._pick(refined)[1:3] == (5.0, 5.0)


def test_pick_without_a_prior_takes_the_best_fit(node, monkeypatch):
    node._params['prior_tie_margin'] = 0.05
    monkeypatch.setattr(node, '_saved_pose_laser', lambda: None)
    refined = [_cand(0.070, 1.0, 1.0), _cand(0.072, 5.0, 5.0)]
    assert node._pick(refined)[1:3] == (1.0, 1.0)


def test_tie_break_can_be_disabled(node, monkeypatch):
    node._params['prior_tie_margin'] = 0.0
    monkeypatch.setattr(node, '_saved_pose_laser', lambda: (5.0, 5.0, 0.0))
    refined = [_cand(0.070, 1.0, 1.0), _cand(0.070001, 5.0, 5.0)]
    assert node._pick(refined)[1:3] == (1.0, 1.0)


def test_single_candidate_needs_no_prior(node, monkeypatch):
    node._params['prior_tie_margin'] = 0.05
    called = []
    monkeypatch.setattr(node, '_saved_pose_laser',
                        lambda: called.append(1) or (0.0, 0.0, 0.0))
    assert node._pick([_cand(0.05, 1.0, 1.0)])[1:3] == (1.0, 1.0)
    assert not called, 'should not read the pose file when there is no tie'


# ── Angle wrapping ────────────────────────────────────────────────────

def test_wrap_treats_neighbouring_angles_as_close():
    mod = _load()
    assert mod._wrap(math.radians(359) - math.radians(1)) == pytest.approx(
        math.radians(-2), abs=1e-9)
    assert abs(mod._wrap(math.radians(179) - math.radians(-179))) == pytest.approx(
        math.radians(2), abs=1e-9)


def test_wrap_is_identity_in_range():
    mod = _load()
    for a in (0.0, 1.0, -1.0, 3.0):
        assert mod._wrap(a) == pytest.approx(a, abs=1e-9)


# ── The selection metric is the fitting metric ────────────────────────

def test_candidates_are_scored_by_wall_fit_not_fft_score():
    """Regression guard on the original bug: selection must not sort by FFT.

    2026-08-02: the FFT stage is gone entirely. It ranked where the scan
    correlates, and the room the robot was actually in never reached its
    shortlist. _sweep_candidates now scores every standable pose by scan->wall
    fit directly, so the fitting metric and the selection metric are the same
    thing and this bug is structurally impossible. The guard stays: it fails
    loudly if correlation ordering is ever reintroduced.
    """
    src = open(NODE).read()
    block = src[src.index('# ── Exhaustive pose sweep'):
                src.index('best_score, laser_x, laser_y, best_theta')]
    assert '_wall_err_fn' in block, 'selection must measure scan->wall fit'
    assert 'reverse=True' not in block, 'correlation ordering must not decide the pick'
    assert '_fft_match' not in block, 'the FFT stage must not come back'


def test_the_sweep_covers_the_whole_map():
    """The point of the rewrite: every standable cell is a hypothesis.

    The FFT proposed a handful of correlation peaks and the true pose was not
    among them. Checked against a user-confirmed ground truth (robot in the
    middle of the living room): the sweep landed 0.07 m and 3 deg out, in 0.1 s.
    """
    src = open(NODE).read()
    assert 'def _sweep_candidates' in src, 'the exhaustive sweep is gone'
    body = src[src.index('def _sweep_candidates'):src.index('def _visibility_fn')]
    assert '_standable' in body, 'the sweep must enumerate standable cells'
    assert 'np.arange(-math.pi, math.pi' in body, 'every heading must be tried'


def test_line_of_sight_breaks_the_tie():
    """Endpoint fit alone cannot tell repeated room shapes apart: candidates
    routinely tie at 0.050 m. Measured the same day, 4-8% of beams blocked
    where the robot really was against 22-47% everywhere else."""
    src = open(NODE).read()
    assert 'def _visibility_fn' in src, 'the beam model is gone'
    block = src[src.index('# ── Reject what could not have produced this scan'):
                src.index('best_score, laser_x, laser_y, best_theta')]
    assert 'blocked_weight' in block, 'line-of-sight must be weighted into the score'
    assert 'refined = [(c[0], c[3], c[4], c[5]) for c in combined]' in block, \
        '_pick must receive the combined score or the prior silently overrides it'


def test_the_beam_model_compares_ranges_not_contact():
    """"Did the ray touch a wall cell" marks a beam blocked when it merely
    grazes mapped furniture; that version scored 66-92% for every candidate and
    decided nothing. What identifies a wrong room is a beam that should have
    stopped far earlier than it did."""
    src = open(NODE).read()
    body = src[src.index('def _visibility_fn'):src.index('def _refine')]
    assert 'short_by' in body and 'expected' in body, \
        'the beam model must compare expected against measured range'
    assert '< 5.0' in body, \
        'long beams graze everything in a flat and must be excluded'


def test_every_shortlisted_candidate_gets_refined():
    src = open(NODE).read()
    block = src[src.index('refined = []'):src.index('refined.sort')]
    assert 'for err, lx, ly, theta in distinct' in block
    assert 'self._refine' in block
