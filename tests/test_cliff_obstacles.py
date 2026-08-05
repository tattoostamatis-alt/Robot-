"""Tests for the cliff sensors reaching the costmap (roomba_driver.py).

The four cliff sensors have stopped the robot since the beginning and the fact
never left the driver, so the planner kept routing over the same stair. These
pin down the two things that make the marks useful: they must land on the side
the sensor that fired is on, and the costmap must never be allowed to clear
them.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_cliff_obstacles.py -q
"""
import math
import re
import struct
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = (_ROOT / 'home_robot' / 'nodes' / 'roomba_driver.py').read_text()
_NAV2 = yaml.safe_load((_ROOT / 'config' / 'nav2_params.yaml').read_text())


# ── the driver publishes them at all ─────────────────────────────────────

def test_the_four_sensors_are_kept_apart():
    """‼️ The merged `self._cliff` bool cannot say WHERE the drop is.

    Packet 7's four cliff bytes were being OR'd into one flag, which is enough
    to stop but not enough to mark: a mark needs the bearing of whichever
    sensor saw the floor end.
    """
    assert '_cliff_bits' in _DRIVER
    assert re.search(r'self\._cliff_bits = \(bool\(cl\), bool\(cfl\), '
                     r'bool\(cfr\), bool\(cr\)\)', _DRIVER), \
        'the four cliff bytes are not being kept individually'


def test_nothing_is_published_when_no_cliff_is_triggered():
    body = re.search(r'def _publish_cliffs.*?(?=\n    def )', _DRIVER, re.S).group(0)
    assert 'if not any(self._cliff_bits):' in body and 'return' in body, \
        'an empty cloud every 200 ms is noise on a topic that means "drop HERE"'


def test_marks_are_published_in_base_link():
    body = re.search(r'def _publish_cliffs.*?(?=\n    def )', _DRIVER, re.S).group(0)
    assert "frame_id = 'base_link'" in body, \
        'a world frame would need a current pose, which a cliff stop may not have'


# ── the geometry puts them on the right side ─────────────────────────────

def cliff_points(bits, bearings=(65.0, 25.0, -25.0, -65.0), radius=0.15, z=0.10):
    """Reimplements _publish_cliffs' geometry, which is the part worth pinning."""
    pts = []
    for fired, b in zip(bits, bearings):
        if not fired:
            continue
        a = math.radians(b)
        pts.append((radius * math.cos(a), radius * math.sin(a), z))
    return pts


def test_a_left_cliff_marks_on_the_left():
    # REP-103: +y is left. A sign error here puts the stair on the wrong side
    # and the robot swerves INTO it.
    (x, y, _), = cliff_points((True, False, False, False))
    assert y > 0 and x > 0, 'left sensor did not mark front-left'


def test_a_right_cliff_marks_on_the_right():
    (x, y, _), = cliff_points((False, False, False, True))
    assert y < 0 and x > 0, 'right sensor did not mark front-right'


def test_the_two_front_sensors_straddle_the_centre_line():
    (_, yl, _), (_, yr, _) = cliff_points((False, True, True, False))
    assert yl > 0 > yr
    assert abs(yl) == abs(yr), 'the front pair should be symmetric'


def test_all_marks_are_in_front():
    for i in range(4):
        bits = [False] * 4
        bits[i] = True
        (x, _, _), = cliff_points(tuple(bits))
        assert x > 0, f'sensor {i} marked behind the robot'


def test_marks_are_off_the_floor_plane():
    """z=0 sits on the boundary where the costmap's height filter rounds."""
    (_, _, z), = cliff_points((True, False, False, False))
    assert z > 0.0
    assert 'CLIFF_MARK_Z' in _DRIVER


def test_the_packing_matches_the_declared_point_step():
    body = re.search(r'def _publish_cliffs.*?(?=\n    def )', _DRIVER, re.S).group(0)
    assert 'point_step = 12' in body
    assert len(struct.pack('<fff', 1.0, 2.0, 3.0)) == 12


def test_the_real_message_parses_back():
    """The cloud is packed by hand, so run the actual method and read the
    result with the same library the costmap uses. A field offset or a stride
    that is wrong by one produces a message that looks fine and decodes to
    garbage."""
    pytest = __import__('pytest')
    pc2 = pytest.importorskip('sensor_msgs_py.point_cloud2')
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import PointCloud2, PointField

    body = re.search(r'    CLIFF_MARK_Z = .*?\n    def _publish_cliffs.*?'
                     r'(?=\n    def )', _DRIVER, re.S).group(0)
    ns = {'math': math, 'struct': struct,
          'PointCloud2': PointCloud2, 'PointField': PointField}
    exec(compile('class D:\n' + body, 'cliff', 'exec'), ns)   # noqa: S102

    sent = []

    class Fake:
        _cliff_bits = (True, False, False, True)     # left + right fired
        CLIFF_MARK_Z = ns['D'].CLIFF_MARK_Z
        _publish_cliffs = ns['D']._publish_cliffs

        def get_parameter(self, name):
            values = {'cliff_bearings_deg': [65.0, 25.0, -25.0, -65.0],
                      'cliff_radius': 0.15}

            class P:
                value = values[name]
            return P()

        def get_clock(self):
            class C:
                def now(self):
                    class N:
                        def to_msg(self):
                            return Time()
                    return N()
            return C()

        class _Pub:
            def publish(self, m):
                sent.append(m)
        _cliff_pub = _Pub()

    Fake()._publish_cliffs()
    assert len(sent) == 1
    pts = list(pc2.read_points(sent[0], field_names=('x', 'y', 'z'),
                               skip_nans=True))
    assert len(pts) == 2, 'two sensors fired, two marks expected'
    assert pts[0][1] > 0 > pts[1][1], 'left and right landed on the same side'
    assert all(abs(p[2] - Fake.CLIFF_MARK_Z) < 1e-6 for p in pts)


# ── the costmap keeps them ───────────────────────────────────────────────

def test_both_costmaps_listen_for_cliffs():
    local = _NAV2['local_costmap']['local_costmap']['ros__parameters']['voxel_layer']
    glob = _NAV2['global_costmap']['global_costmap']['ros__parameters']['obstacle_layer']
    assert 'cliffs' in local['observation_sources'].split()
    assert 'cliffs' in glob['observation_sources'].split()
    assert local['cliffs']['topic'] == '/cliff_obstacles'
    assert glob['cliffs']['topic'] == '/cliff_obstacles'


def test_cliffs_are_never_cleared_by_a_raytrace():
    """‼️ The whole point. A drop is invisible to every other sensor — the
    LiDAR sweeps straight over a stairwell and reports open floor — so a
    raytrace would erase the mark the moment the robot backed away from it,
    which is exactly when it has to survive."""
    for scope, layer in (('local_costmap', 'voxel_layer'),
                         ('global_costmap', 'obstacle_layer')):
        cfg = _NAV2[scope][scope]['ros__parameters'][layer]['cliffs']
        assert cfg['clearing'] is False, f'{scope} would clear its cliff marks'
        assert cfg['marking'] is True


def test_the_global_costmap_is_the_one_that_remembers():
    """The local map is a rolling window and forgets a stair once the robot
    walks away; without the global source the planner re-routes over the same
    drop on the next trip."""
    glob = _NAV2['global_costmap']['global_costmap']['ros__parameters']
    assert 'cliffs' in glob['obstacle_layer']['observation_sources']
