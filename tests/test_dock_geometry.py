"""Tests for deriving the charging station from the AprilTag above it.

Robot-free: everything here is pure geometry over the calibration file and the
map, which is the point — the bug these guard against (a dock pose 0.96 m from
the real base on the malou map) was invisible on hardware because the robot
just "stopped near the base but not on it", and looked like a docking bug.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dock_geometry.py -q
"""
import math
import os

import pytest

from home_robot import dock_geometry as dg

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where apriltag_relocalizer stores the tag pose for the map in use.
CALIB = os.path.expanduser('~/.ros/room4_tag_map_pose.yaml')

# The calibrated malou tag, as measured 2026-07-22 (verified to 6.1 mm). A
# frozen fixture for the geometry itself — the map moved on, the arithmetic
# it pins down did not. Anything that must track the live map reads CALIB.
MALOU_TAG = dict(x=-0.11230837877346556, y=-0.3591970113160061,
                 z=0.5587916250045776,
                 qx=0.3689896161706194, qy=0.6191975245097218,
                 qz=0.5957590503735292, qw=0.35427707052153706)


def test_tag_normal_points_into_the_room():
    """The tag's outward normal must face where the robot can stand."""
    n = dg.tag_normal_yaw(MALOU_TAG['qx'], MALOU_TAG['qy'],
                          MALOU_TAG['qz'], MALOU_TAG['qw'])
    assert math.degrees(n) == pytest.approx(28.5, abs=1.0)


def test_normal_agrees_with_the_taught_in_front_pose():
    """Ground truth: the pose taught by parking the robot in front of the tag.

    Recorded at x=0.40 y=-0.08 yaw=-147.8 deg. If our normal is right, that
    pose sits out along it and looks back down it. This is the check that
    caught the bad dock entry — the same pose pointed 161 deg away from it.
    """
    n = dg.tag_normal_yaw(MALOU_TAG['qx'], MALOU_TAG['qy'],
                          MALOU_TAG['qz'], MALOU_TAG['qw'])
    hx, hy, hyaw = 0.40, -0.08, math.radians(-147.8)

    # It stands in front of the tag face, not off to the side or behind it.
    assert math.degrees(dg.off_axis(MALOU_TAG['x'], MALOU_TAG['y'],
                                    n, hx, hy)) < 15.

    # And it is aimed at the tag.
    bearing = math.atan2(MALOU_TAG['y'] - hy, MALOU_TAG['x'] - hx)
    assert abs(math.degrees(dg._wrap(bearing - hyaw))) < 10.


def test_recorded_dock_pose_was_wrong():
    """Regression: the pose that used to be in locations.yaml is not the base.

    Kept as a test because the failure was silent for weeks — Nav2 dutifully
    drove to a cramped spot on the far side of the room and every dock attempt
    was blamed on IR homing.
    """
    old = (0.77, 0.007)
    assert math.hypot(old[0] - MALOU_TAG['x'],
                      old[1] - MALOU_TAG['y']) > 0.9


def test_seated_pose_faces_into_the_station():
    n = math.radians(28.5)
    x, y, yaw = dg.seated_pose(0.0, 0.0, n, seat_offset=0.28)
    # Out along the normal by the seat offset...
    assert math.hypot(x, y) == pytest.approx(0.28, abs=1e-6)
    # ...looking back at the base, not away from it.
    assert abs(math.degrees(dg._wrap(yaw - (n + math.pi)))) < 1e-6


def test_approach_point_is_on_the_normal():
    n = math.radians(28.5)
    for d in (0.3, 0.85, 1.5):
        x, y, _ = dg.approach_point(-0.112, -0.359, n, d)
        assert dg.off_axis(-0.112, -0.359, n, x, y) == pytest.approx(0., abs=1e-9)
        assert math.hypot(x + 0.112, y + 0.359) == pytest.approx(d, abs=1e-9)


def test_off_axis_is_symmetric_and_unsigned():
    n = 0.0
    left = dg.off_axis(0., 0., n, math.cos(0.4), math.sin(0.4))
    right = dg.off_axis(0., 0., n, math.cos(-0.4), math.sin(-0.4))
    assert left == pytest.approx(right)
    assert left == pytest.approx(0.4)


# ── find_staging ──────────────────────────────────────────────────────

def test_staging_prefers_the_nearest_reachable_point():
    """A short blind IR run beats a perfectly square one: less room to drift."""
    n = 0.0

    def clearance(x, y):
        # Everything past 0.8 m out is open.
        return 0.5 if math.hypot(x, y) >= 0.8 else 0.0

    got = dg.find_staging(0., 0., n, clearance)
    assert got is not None
    x, y, yaw, dist, clear = got
    assert dist == pytest.approx(0.8, abs=0.06)
    assert clear >= dg.DEFAULT_MIN_CLEARANCE


def test_staging_faces_the_station():
    n = math.radians(28.5)

    def clearance(x, y):
        return 0.5 if math.hypot(x + 0.112, y + 0.359) >= 0.7 else 0.0

    x, y, yaw, dist, _ = dg.find_staging(-0.112, -0.359, n, clearance)
    expect = math.atan2(-0.359 - y, -0.112 - x)
    assert abs(math.degrees(dg._wrap(yaw - expect))) < 1e-6


def test_staging_stays_in_front_of_the_tag():
    """Open floor behind the tag must not be chosen — the camera can't see it."""
    n = 0.0

    def clearance(x, y):
        return 0.5     # everywhere is open, so only the off-axis limit binds

    x, y, yaw, dist, _ = dg.find_staging(0., 0., n, clearance)
    assert dg.off_axis(0., 0., n, x, y) <= dg.DEFAULT_MAX_OFF_AXIS + 1e-9


def test_staging_returns_none_when_walled_in():
    """A base with no reachable approach must say so, not return a bad pose."""
    assert dg.find_staging(0., 0., 0., lambda x, y: 0.0) is None


def test_staging_rejects_anything_nav2_would_refuse():
    """Never hand back a goal below the inflation radius — Nav2 won't plan it."""
    n = 0.0

    def clearance(x, y):
        d = math.hypot(x, y)
        return 0.29 if d < 1.2 else 0.4     # just under the limit up close

    got = dg.find_staging(0., 0., n, clearance)
    assert got is not None
    assert got[4] >= dg.DEFAULT_MIN_CLEARANCE
    assert got[3] >= 1.2


def test_staging_on_the_active_map_when_calibrated():
    """End-to-end over room4 once its tag calibration exists."""
    np = pytest.importorskip('numpy')
    yaml = pytest.importorskip('yaml')
    Image = pytest.importorskip('PIL.Image', reason='pillow not installed')
    ndimage = pytest.importorskip('scipy.ndimage')

    if not os.path.exists(CALIB):
        pytest.skip('room4 tag is not calibrated yet')
    tag = yaml.safe_load(open(CALIB))
    meta = yaml.safe_load(open(f'{PKG}/maps/room4.yaml'))
    res, (ox, oy) = meta['resolution'], meta['origin'][:2]
    img = np.array(Image.open(f'{PKG}/maps/room4.pgm'))
    h, w = img.shape
    dist = ndimage.distance_transform_edt(img >= 65) * res

    def clearance(x, y):
        c, r = int((x - ox) / res), h - 1 - int((y - oy) / res)
        if not (0 <= r < h and 0 <= c < w):
            return -1.0
        return float(dist[r, c])

    n = dg.tag_normal_yaw(tag['qx'], tag['qy'], tag['qz'], tag['qw'])
    got = dg.find_staging(tag['x'], tag['y'], n, clearance)
    assert got is not None, 'no reachable staging pose on room4'
    x, y, yaw, d, c = got
    assert c >= dg.DEFAULT_MIN_CLEARANCE       # Nav2 will accept it
    assert d <= 1.5                            # IR can plausibly bridge it


def test_locations_yaml_matches_the_tag():
    """The shipped dock/dock_staging must still agree with the calibration.

    Against the *live* calibration, not MALOU_TAG: the tag pose is measured
    per map, so pinning it here meant the test compared the malou2 dock with
    the malou tag and failed by 2 m from the 2026-07-28 remap onwards — while
    the dock entry itself was right. A stale calibration on disk is the case
    worth catching, and this is how to catch it.
    """
    yaml = pytest.importorskip('yaml')
    locs = yaml.safe_load(open(f'{PKG}/config/locations.yaml'))
    if not os.path.exists(CALIB):
        assert 'dock' not in locs and 'dock_staging' not in locs
        pytest.skip('room4 tag not calibrated; stale dock goals correctly absent')
    assert 'dock_staging' in locs, 'run scripts/derive_dock_pose.py room4 --write'
    tag = yaml.safe_load(open(CALIB))

    n = dg.tag_normal_yaw(tag['qx'], tag['qy'], tag['qz'], tag['qw'])
    sx, sy, _ = dg.seated_pose(tag['x'], tag['y'], n)
    assert math.hypot(locs['dock']['x'] - sx,
                      locs['dock']['y'] - sy) < 0.02, \
        'dock does not sit under the tag — run scripts/derive_dock_pose.py --write'

    st = locs['dock_staging']
    # Staging must be in front of the tag and further out than the seated pose.
    assert dg.off_axis(tag['x'], tag['y'], n,
                       st['x'], st['y']) <= dg.DEFAULT_MAX_OFF_AXIS
    assert (math.hypot(st['x'] - tag['x'], st['y'] - tag['y'])
            > dg.DEFAULT_SEAT_OFFSET)


def test_dock_staging_is_not_treated_as_a_room():
    """A patrol must not add the charger to its round of rooms."""
    yaml = pytest.importorskip('yaml')
    from home_robot.status_query import rooms_from_locations
    locs = yaml.safe_load(open(f'{PKG}/config/locations.yaml'))
    rooms = rooms_from_locations(locs)
    assert 'dock' not in rooms and 'dock_staging' not in rooms

    src = open(f'{PKG}/home_robot/nodes/mission_executor_node.py').read()
    assert 'rooms_from_locations' in src
    assert "if n != 'dock'" not in src, 'stale room filter misses dock_staging'


# ── Wiring ────────────────────────────────────────────────────────────

def test_voice_dock_goes_through_the_mission_not_straight_to_ir():
    """"πήγαινε να φορτίσεις" must run the whole sequence.

    Publishing `dock` directly starts IR homing wherever the robot stands; from
    another room there is no beam and it sweeps, fails, and reports nothing.
    """
    src = open(f'{PKG}/home_robot/nodes/llm_bridge_node.py').read()
    assert "self.mission_pub.publish(String(data='dock'))" in src
    assert 'self.dock_pub' not in src, \
        'llm_bridge still has a direct dock publisher — it bypasses the approach'


def test_mission_executor_runs_in_the_localize_launch():
    """The dock mission is useless if the executor is off in the launch we use.

    localize.launch.py is what `robot max` starts, and it used to pass
    use_mission:=false — so routing docking through the mission executor would
    have published into the void.
    """
    src = open(f'{PKG}/launch/localize.launch.py').read()
    assert "'use_mission':         'true'," in src, \
        'localize.launch.py must start the mission executor for docking'


def test_dock_mission_waits_for_a_terminal_status():
    """It must not claim success the moment Nav2 returns, as it used to."""
    src = open(f'{PKG}/home_robot/nodes/mission_executor_node.py').read()
    assert '_run_ir_homing' in src
    assert "self._dock_pub.publish(Bool(data=True))" in src
    for terminal in ("'seated'", "'lost'", "'timeout'"):
        assert terminal in src, f'dock mission ignores {terminal}'


def test_driver_reports_every_dock_outcome():
    """Each way the approach can end has to reach the caller."""
    src = open(f'{PKG}/home_robot/nodes/roomba_driver.py').read()
    assert "self.create_publisher(\n            String, 'dock_status'" in src
    for outcome in ("'seated'", "'lost'", "'timeout'"):
        assert f'_publish_dock_status({outcome})' in src, \
            f'driver never publishes {outcome}'


# ── align_twist ───────────────────────────────────────────────────────

def test_align_does_nothing_when_already_square():
    assert dg.align_twist(1.0, 1.0) == 0.0
    assert dg.align_twist(1.0, 1.0 + math.radians(5)) == 0.0


def test_align_turns_the_short_way_across_the_wrap():
    """Near +pi aiming near -pi: 20 deg left, not 340 deg right."""
    assert dg.align_twist(math.radians(170), math.radians(-170)) > 0
    assert dg.align_twist(math.radians(-170), math.radians(170)) < 0
    # And a 2 deg error across the wrap is still "square enough".
    assert dg.align_twist(math.radians(179), math.radians(-179)) == 0.0


def test_align_never_commands_below_the_rotation_floor():
    """Anything under ~0.31 rad/s stalls this base instead of turning it."""
    for err_deg in (13, 45, 90, 179):
        wz = dg.align_twist(0.0, math.radians(err_deg))
        assert abs(wz) >= 0.31, f'{err_deg} deg -> {wz} rad/s would stall'


def test_align_direction_matches_the_error_sign():
    assert dg.align_twist(0.0, math.radians(40)) > 0
    assert dg.align_twist(0.0, math.radians(-40)) < 0
