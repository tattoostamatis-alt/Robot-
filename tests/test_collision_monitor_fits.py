"""The collision monitor and the costmap must agree on the robot's shape.

2026-08-06. The fetch mission could not reach the toualeta or the kouzina in
scripts/fetch_sim_gazebo.py. Nav2 planned a route every time; the robot then
sat still and recovery_manager logged

    STUCK detected — 6.4s with cmd_vel, displacement=0.022m

over and over. Neither the planner nor the doorways were at fault. Two config
files disagreed about how big the robot is:

  * nav2_params costmap:            circle, robot_radius 0.175
    (chassis re-measured Ø0.35 m on 2026-07-31)
  * nav2_params collision_monitor:  square, ±0.22 — half-DIAGONAL 0.311 m

Every comment around the Stop polygon reasons about box edges, and is correct
about them; the corner is the part that was missed. So the planner routed
through gaps the monitor then refused, and cmd_vel was zeroed with a valid
path in hand.

Half-width of the tightest passage to each room, measured on malou2 with
scripts/map_clearance.py:

    toualeta          0.241 m     square 0.311 does NOT fit, circle 0.175 does
    kouzina           0.300 m     square 0.311 does NOT fit, circle 0.175 does
    domatio tou max   0.333 m     both fit
    diadromos         0.433 m     both fit

— which is exactly the set of rooms that failed.

The polygons are octagons now. These tests keep them that way: no corner of a
Stop polygon may claim more space than the tightest passage the robot is
expected to drive through.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_collision_monitor_fits.py -q
"""
import ast
import math
import os

import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(PKG, 'config', 'nav2_params.yaml')

# Measured with scripts/map_clearance.py on maps/malou2.yaml. This is a
# half-width: the largest disc radius that still reaches the room.
TIGHTEST_PASSAGE = 0.241        # toualeta
# The Gazebo world is built from a 2x-downsampled map, which fattens walls;
# the same passage measures 0.200 there. Anything that fits 0.241 in the real
# apartment is what we require — the sim being tighter is a sim artefact.

with open(PARAMS) as fh:
    _P = yaml.safe_load(fh)
_STOP = _P['collision_monitor']['ros__parameters']['Stop']
_COSTMAP_R = _P['local_costmap']['local_costmap']['ros__parameters']['robot_radius']

SUBPOLYGONS = ['moving_forward', 'moving_backward', 'stopped_or_rotating']


def _points(name):
    return ast.literal_eval(_STOP[name]['points'])


def _max_reach(pts):
    return max(math.hypot(x, y) for x, y in pts)


def _half_width(pts):
    return max(abs(y) for _x, y in pts)


def test_costmap_radius_matches_the_measured_chassis():
    """Ø0.35 m, measured 2026-07-31. The polygons below are built around it."""
    assert abs(_COSTMAP_R - 0.175) < 1e-6, (
        'robot_radius changed — re-derive the Stop polygons from the new '
        'chassis size, do not leave them behind like last time')


def test_rotating_in_place_fits_the_tightest_doorway():
    """No direction of travel, so EVERY bearing has to fit.

    This is the one that mattered: a robot that cannot turn inside a doorway
    cannot start moving through it either.
    """
    reach = _max_reach(_points('stopped_or_rotating'))
    assert reach < TIGHTEST_PASSAGE, (
        f'stopped_or_rotating reaches {reach:.3f} m, but the tightest passage '
        f'is {TIGHTEST_PASSAGE} m — the monitor will zero cmd_vel and the '
        f'robot will sit still with a valid plan')


def test_driving_polygons_are_no_wider_than_the_body():
    """Sideways clearance is what a doorway charges you for.

    Forward reach may exceed it — that points down the corridor, into space
    the robot is about to occupy anyway — but the width may not.
    """
    for name in ('moving_forward', 'moving_backward'):
        hw = _half_width(_points(name))
        assert hw <= 0.175 + 1e-9, (
            f'{name} is {hw:.3f} m half-width, wider than the Ø0.35 chassis; '
            f'that is what wedged the robot in the toualeta doorway')
        assert hw < TIGHTEST_PASSAGE


def test_no_polygon_is_a_bare_rectangle():
    """A 4-point box hides its true size in the corners.

    ±0.22 reads as "0.44 m wide, fine for a 0.48 m gap" and is actually
    0.62 m across the diagonal.
    """
    for name in SUBPOLYGONS:
        pts = _points(name)
        assert len(pts) > 4, (
            f'{name} is back to a rectangle — its corners will reach '
            f'{_max_reach(pts):.3f} m while the edges suggest otherwise')


def test_stop_polygons_still_cover_the_physical_robot():
    """Cutting corners must not cut INTO the robot.

    The monitor is a safety device; every sub-polygon still has to enclose the
    real chassis, and keep a margin in the direction it is checking.
    """
    for name in SUBPOLYGONS:
        pts = _points(name)
        assert _half_width(pts) >= 0.175 - 1e-9, (
            f'{name} is narrower than the robot itself')
        assert _max_reach(pts) >= 0.175, (
            f'{name} does not enclose the chassis')
    fwd = _points('moving_forward')
    assert max(x for x, _y in fwd) >= 0.22, (
        'forward stop margin shrank below the 50mm asked for on the front')
