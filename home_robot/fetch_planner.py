"""Pure, ROS-free helpers for the "φέρε μου το X" (fetch) mission.

The mission itself lives in mission_executor_node.py; the parts that are plain
geometry / bookkeeping live here so they can be unit-tested without a robot or
a running graph — see tests/test_fetch_planner.py. Everything below is
arithmetic on plain numbers and dicts; the node handles TF, Nav2 and the arm.
"""

import math


def approach_pose(robot_xy, obj_xy, dist=0.4):
    """Where to park to grasp the object at obj_xy.

    Returns (x, y, yaw): a point `dist` metres from the object, on the line
    from the object toward the robot (so we stop *short* of it), facing the
    object. If the robot is already on top of the object the direction is
    undefined, so we fall back to approaching from -x.

    yaw is the map-frame heading the robot should hold — pointing at the
    object, so it lands in the camera/arm workspace ahead of the base.
    """
    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    dx, dy = rx - ox, ry - oy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy, norm = -1.0, 0.0, 1.0
    ux, uy = dx / norm, dy / norm            # unit vector object → robot
    ax, ay = ox + ux * dist, oy + uy * dist  # park this far back along it
    yaw = math.atan2(oy - ay, ox - ax)       # face the object
    return ax, ay, yaw


def memory_target(answer, now, max_age):
    """Decide whether an object_memory/answer instance is usable as a target.

    `answer` is the parsed JSON dict from object_memory_node ({} when the
    object is unknown). Returns {'x','y','room','last_seen'} if it has a
    position and was seen within max_age seconds, else None (→ the caller
    should fall back to a live room-by-room search).
    """
    if not answer:
        return None
    if 'x' not in answer or 'y' not in answer:
        return None
    last_seen = answer.get('last_seen')
    if last_seen is not None and max_age > 0 and (now - last_seen) > max_age:
        return None                          # stale — object may have moved
    return {
        'x': float(answer['x']),
        'y': float(answer['y']),
        'room': answer.get('room'),
        'last_seen': last_seen,
    }


def homing_twist(person, deliver_distance=0.8, center_tol=0.15,
                 ang_gain=1.0, max_ang=0.5, lin_speed=0.12, lin_gain=0.6):
    """Velocity to home in on a person for `follow` delivery.

    `person` is a detected_objects entry (camera frame: x right, z forward, m).
    We rotate to centre them, then drive forward until within deliver_distance.
    Returns (linear_x, angular_z, arrived):
      - angular_z steers to null out the bearing (person on the right → turn
        right = negative/CW under REP-103);
      - linear_x drives forward only once roughly centred and still too far,
        proportional to the *remaining* distance (capped at lin_speed) so the
        robot eases to a gentle stop instead of slamming to zero at the user —
        the standard proportional-follower behaviour;
      - arrived is True when centred AND within deliver_distance → stop & place.
    """
    x = float(person.get('x', 0.0))
    z = float(person.get('z', 0.0))
    if z <= 0.05:
        return 0.0, 0.0, False           # no usable range — hold
    bearing = math.atan2(x, z)           # 0 = ahead, + = to the right
    centred = abs(bearing) <= center_tol
    close   = z <= deliver_distance

    ang = 0.0 if centred else -max(-max_ang, min(max_ang, ang_gain * bearing))
    if centred and not close:
        lin = max(0.0, min(lin_speed, lin_gain * (z - deliver_distance)))
    else:
        lin = 0.0
    return lin, ang, (centred and close)


def nearest_detection(objects, label, max_z):
    """Closest same-label detection in a detected_objects list, or None.

    Objects are in the camera frame (x right, y down, z forward); we rank by z
    (forward distance) and ignore anything farther than max_z or non-positive.
    Used both to verify the object is in grasp range and to pick a live target
    during the search fallback.
    """
    best, best_z = None, float('inf')
    for o in (objects or []):
        if str(o.get('label', '')).lower() != label.lower():
            continue
        z = o.get('z', 0.0)
        if z is None or z <= 0.1 or z > max_z:
            continue
        if z < best_z:
            best, best_z = o, z
    return best
