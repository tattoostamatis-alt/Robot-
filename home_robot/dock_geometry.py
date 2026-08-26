"""Where the charging station is, derived from the AprilTag above it.

The station's pose used to be taught by hand, by driving the robot onto the
base and recording the pose. That is how `dock` in locations.yaml ended up
0.96 m away from the real base on the `malou` map: recorded facing the wrong
way, it sent Nav2 to a cramped spot on the far side of the room, and every
dock attempt then "stopped near the base but not on it". The taught
in-front-of-the-tag pose gives it away — it points at the tag (3.6 deg off)
and 161 deg away from the recorded dock.

The tag is mounted centred directly above the station, so its calibrated map
pose *is* the station's map pose, to within how well the tag itself is
calibrated (6.1 mm at the last check). Deriving the dock from it instead of
teaching it removes the failure mode entirely, and a remap only has to
recalibrate the tag — which it must do anyway.

Three poses come out of this, and they are different things:

* **station** — where the base physically is: the tag's x/y at floor level.
* **seated** — where the *robot's centre* sits when parked on the base, one
  robot radius plus the base's depth out along the tag's normal. This is a
  reported position, never a Nav2 goal; the map has the base and the wall
  behind it as solid obstacle, so no planner will ever accept it.
* **staging** — the handover point: the closest spot to the station that Nav2
  can actually reach, still square enough to the tag that the camera can see
  it. IR homing takes over from there.

That last one is the crux. The corridor into the base measures 0.05-0.16 m of
clearance while the robot's inscribed radius is 0.175 m and Nav2's inflation is
0.30 m, so Nav2 cannot plan the final approach even in principle. It does not
need to: the robot fits, and `ir_homing` drives it in on the dock's own beams
with the costmap out of the loop.
"""
import math

# Distance from the wall plane (where the tag is) out to the robot's centre
# when it is parked on the base: the Roomba's 0.175 m inscribed radius (chassis
# re-measured Ø0.35 on 2026-07-31) plus the depth the Home Base's ramp holds it
# off the wall.
DEFAULT_SEAT_OFFSET = 0.285

# Nav2 will not plan into anything tighter than its inflation radius, so a
# staging pose below this is one the robot will never be given.
DEFAULT_MIN_CLEARANCE = 0.30

# AprilTag detection degrades badly at a shallow angle to the tag face, and a
# relocalization from a bad detection is worse than none.
DEFAULT_MAX_OFF_AXIS = math.radians(50.)


def tag_normal_yaw(qx, qy, qz, qw):
    """Map-frame bearing of the tag's outward normal, in radians.

    apriltag_ros puts the tag's +Z through its face, pointing back at whatever
    camera can see it — so this is the direction you must approach from.
    """
    # Third column of the rotation matrix: the tag's +Z axis in map coordinates.
    nx = 2. * (qx * qz + qy * qw)
    ny = 2. * (qy * qz - qx * qw)
    return math.atan2(ny, nx)


def seated_pose(tag_x, tag_y, normal_yaw, seat_offset=DEFAULT_SEAT_OFFSET):
    """(x, y, yaw) of the robot's centre when parked on the base.

    Yaw faces *into* the station, which is the reverse of the tag's normal.
    """
    return (tag_x + seat_offset * math.cos(normal_yaw),
            tag_y + seat_offset * math.sin(normal_yaw),
            _wrap(normal_yaw + math.pi))


def approach_point(tag_x, tag_y, normal_yaw, distance):
    """(x, y, yaw) `distance` straight out from the tag, facing back at it."""
    return (tag_x + distance * math.cos(normal_yaw),
            tag_y + distance * math.sin(normal_yaw),
            _wrap(normal_yaw + math.pi))


def off_axis(tag_x, tag_y, normal_yaw, x, y):
    """How far off the tag's face normal the point (x, y) sits, in radians.

    0 is dead in front of the tag; pi/2 is edge-on, where it stops resolving.
    """
    return abs(_wrap(math.atan2(y - tag_y, x - tag_x) - normal_yaw))


def find_staging(tag_x, tag_y, normal_yaw, clearance,
                 min_clearance=DEFAULT_MIN_CLEARANCE,
                 max_off_axis=DEFAULT_MAX_OFF_AXIS,
                 min_dist=0.5, max_dist=1.5,
                 dist_step=0.05, angle_step=math.radians(4.)):
    """Closest pose to the station that Nav2 can reach with the tag in view.

    `clearance(x, y)` returns metres to the nearest obstacle — pass a lookup
    over the map's distance transform. Returns (x, y, yaw, dist, clear) with
    yaw facing the station, or None if the station is walled in beyond reach.

    Nearest wins over squarest: the shorter the blind IR run, the less room
    the approach has to drift.
    """
    best = None
    steps = int(round((max_dist - min_dist) / dist_step)) + 1
    for i in range(steps):
        d = min_dist + i * dist_step
        n_ang = int(round(2. * math.pi / angle_step))
        for j in range(n_ang):
            a = -math.pi + j * angle_step
            if abs(_wrap(a - normal_yaw)) > max_off_axis:
                continue
            x = tag_x + d * math.cos(a)
            y = tag_y + d * math.sin(a)
            c = clearance(x, y)
            if c < min_clearance:
                continue
            cand = (d, abs(_wrap(a - normal_yaw)), x, y, c)
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is not None:
            # Nothing further out can beat a hit at this radius.
            break
    if best is None:
        return None
    d, _, x, y, c = best
    return (x, y, _wrap(math.atan2(tag_y - y, tag_x - x)), d, c)


# This base cannot rotate below ~0.31 rad/s — commanding less just stalls the
# wheels — and overshoots ~14% above it. So an alignment either turns at a
# speed that actually moves the robot, or it does not turn at all.
ROTATION_FLOOR = 0.35
ALIGN_TOLERANCE = math.radians(12.)


def align_twist(current_yaw, target_yaw, tolerance=ALIGN_TOLERANCE,
                speed=ROTATION_FLOOR):
    """Angular velocity to point the robot at the station, 0 when close enough.

    The tolerance is deliberately loose: IR homing sweeps for the beam anyway,
    so hunting for a degree here would cost more (in rotation-floor overshoot)
    than it buys. Getting the robot roughly square to the base just shortens
    the sweep.
    """
    err = _wrap(target_yaw - current_yaw)
    if abs(err) <= tolerance:
        return 0.0
    return speed if err > 0 else -speed


def _wrap(a):
    """Angle to (-pi, pi]."""
    return (a + math.pi) % (2. * math.pi) - math.pi
