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
# ‼️ 2026-08-08: the three are NOT interchangeable any more, and the split is
# the whole point — see test_rotation_is_not_gated_like_translation.
# Translating sweeps new ground and keeps the full skirt; rotating in place
# sweeps none, so its ring only has to catch actual contact.
MOVING = ['moving_forward', 'moving_backward']
ROTATING = 'stopped_or_rotating'


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


CHASSIS_R = 0.175
# The nominal skirt radius comes from safety_settings, which is what the
# dashboard's Safety tab displays. Reading it here means the tab and the YAML
# cannot drift apart: change one without the other and this file fails.
import home_robot.safety_settings as _ss           # noqa: E402
SKIRT_R = _ss.INFO_ONLY[0]['value']                # 0.215 = chassis + 40 mm
STOP_MARGIN = SKIRT_R - CHASSIS_R
# A 16-point polygon inscribes its circle: the edge midpoints fall ~4.1 mm
# short of the radius. Allow that, and nothing more.
POLY_SLACK = 0.006


def _min_reach(pts):
    """Closest approach of the polygon boundary to the centre.

    For a convex polygon that is the distance to the nearest EDGE, not the
    nearest vertex — a 16-gon's vertices sit on the circle while its edges cut
    slightly inside.
    """
    best = float('inf')
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg = math.hypot(dx, dy)
        if seg < 1e-12:
            continue
        # perpendicular distance from origin to the segment's line
        best = min(best, abs(dx * y1 - dy * x1) / seg)
    return best


def test_every_bearing_keeps_the_full_stop_margin():
    """‼️ The one that caught my own mistake.

    First attempt at this fix cut the corners off but also pulled the sides in
    to ±0.175 — exactly the chassis, i.e. ZERO margin sideways. The monitor
    would not have stopped for anything beside the robot until it was already
    touching. The margin is whatever the Safety tab says it is (40 mm since
    2026-08-06) and it is that in EVERY direction, which is what a round robot
    needs and what the box only ever delivered diagonally.
    """
    floor = SKIRT_R - POLY_SLACK
    for name in MOVING:
        closest = _min_reach(_points(name))
        margin_mm = (closest - CHASSIS_R) * 1000
        assert closest >= floor, (
            f'{name} comes within {closest:.4f} m of centre — only '
            f'{margin_mm:.0f} mm clear of the Ø0.35 chassis, against the '
            f'{STOP_MARGIN * 1000:.0f} mm this polygon exists to keep')
        assert _max_reach(_points(name)) <= SKIRT_R + 1e-6, (
            f'{name} reaches past the {SKIRT_R} m skirt the Safety tab '
            f'reports — the page would be lying about the robot')


def test_rotation_is_not_gated_like_translation():
    """‼️ 2026-08-08 — THE DEADLOCK. Read this before "restoring consistency".

    All three rings used to be the same 0.215. That looks tidy and it froze
    the robot solid: anything 40 mm off the body zeroed every command, and
    since a spin is linear≈0 it fell in the `stopped_or_rotating` bucket, so
    the robot could not even TURN to escape. Seen live on the (-4.43, 1.26)
    goal — four rounds of "STUCK, displacement=0.000m", every recovery killed
    by "Collision Ahead - Exiting Spin" / "Exiting DriveOnHeading". That is
    the back-and-forth the user reported as "goes nowhere, just rocks".

    The geometry: a CIRCLE rotating about its own centre sweeps no new ground
    whatsoever. A spin cannot cause a collision that is not already happening,
    so gating it on clearance is asking the wrong question. Translation is the
    opposite — it does sweep new ground, and keeps the full 40 mm.

    A/B in Gazebo (scratchpad cmtest/e2e.py), robot ~0.19 m off a wall:
        0.215 ring: 0/60 rotate commands survived, 0.0° turned, goal rejected
        0.180 ring: 57/57 survived, 70.4° turned, robot drove 3.88 m away
    """
    rot = _min_reach(_points(ROTATING))
    tightest_moving = min(_min_reach(_points(n)) for n in MOVING)
    assert rot < tightest_moving - 0.02, (
        f'{ROTATING} reaches {rot:.4f} m, essentially the same as the moving '
        f'rings ({tightest_moving:.4f} m). That is the deadlock: the robot '
        f'cannot spin away from an obstacle it is allowed to sit next to.')


def test_the_rotation_ring_still_refuses_actual_contact():
    """The other side of it. Rotating is safe until something is genuinely
    against the chassis — brushing a wall while spinning scrubs the wheels and
    pushes the pose off. So the ring may shrink toward the body, never inside
    it."""
    rot = _min_reach(_points(ROTATING))
    assert rot >= CHASSIS_R, (
        f'{ROTATING} cuts to {rot:.4f} m, inside the {CHASSIS_R} m chassis — '
        f'the robot would happily grind against a wall')
    assert _max_reach(_points(ROTATING)) <= SKIRT_R + 1e-6


def test_the_zone_is_round_not_boxy():
    """A round robot gets a round stop zone.

    Concretely: no bearing may reach more than a hair further than any other.
    A ±0.22 square reaches 0.311 diagonally against 0.22 at the edges — 41%
    more in some directions than others, and that surplus is what refused
    doorways the robot fits through.
    """
    for name in SUBPOLYGONS:
        pts = _points(name)
        lo, hi = _min_reach(pts), _max_reach(pts)
        assert hi - lo < 0.010, (
            f'{name} reaches {hi:.3f} m one way and {lo:.3f} m another '
            f'({(hi - lo) * 1000:.0f} mm apart) — that is a polygon pretending '
            f'to be a circle, and the long bearing will refuse a doorway')


def test_the_zone_fits_the_tightest_doorway():
    """It must still get through the narrowest passage in the house."""
    for name in SUBPOLYGONS:
        reach = _max_reach(_points(name))
        assert reach < TIGHTEST_PASSAGE, (
            f'{name} reaches {reach:.3f} m, but the tightest passage is '
            f'{TIGHTEST_PASSAGE} m — the monitor will zero cmd_vel and the '
            f'robot will sit still holding a valid plan')


def test_stop_polygons_enclose_the_physical_robot():
    """The monitor is a safety device: never cut INTO the body."""
    for name in SUBPOLYGONS:
        assert _min_reach(_points(name)) > CHASSIS_R, (
            f'{name} passes inside the chassis outline')
