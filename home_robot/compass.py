"""A compass that works indoors — referenced to the map, not to magnetic north.

The BNO085 has a magnetometer and the firmware deliberately does not use it.
SH2_ROTATION_VECTOR (the magnetometer-fused one) was tried and removed: indoors
the Roomba's DC motors and the metal around the flat corrupted the absolute
heading, the yaw jumped, the EKF rotated odom->base_link, the LiDAR stopped
lining up with the mapped walls and AMCL could not stay converged. That failure
was reported at the time as "the compass doesn't work".

So this compass takes its heading from AMCL instead. The map does not rotate,
and AMCL's yaw is scan-matched against real walls, which makes it far steadier
indoors than any magnetometer near a vacuum motor. The only thing missing is
which way north points *in the map*, and that is a single number the user
supplies once per map: point the robot north, press calibrate.

Pure, so the angle conventions can be tested without a robot. They are easy to
get backwards: ROS yaw grows counter-clockwise from the map's +x axis, while
compass bearings grow CLOCKWISE from north.
"""

import math

__all__ = ['CARDINALS_EL', 'normalize_deg', 'bearing_from_yaw',
           'yaw_for_bearing', 'cardinal_el', 'cardinal_full_el',
           'offset_from_known_bearing']

# Eight points, Greek. Index = round(bearing / 45) % 8.
CARDINALS_EL = ('Β', 'ΒΑ', 'Α', 'ΝΑ', 'Ν', 'ΝΔ', 'Δ', 'ΒΔ')

_FULL_EL = {
    'Β':  'Βορράς',        'ΒΑ': 'Βορειοανατολικά',
    'Α':  'Ανατολή',       'ΝΑ': 'Νοτιοανατολικά',
    'Ν':  'Νότος',         'ΝΔ': 'Νοτιοδυτικά',
    'Δ':  'Δύση',          'ΒΔ': 'Βορειοδυτικά',
}


def normalize_deg(deg):
    """Fold any angle into [0, 360)."""
    return float(deg) % 360.0


def bearing_from_yaw(yaw_rad, north_offset_rad=0.0):
    """Compass bearing (0=N, 90=E, 180=S, 270=W) for a map-frame yaw.

    ``north_offset_rad`` is the map yaw the robot has when it faces north — the
    one calibrated number.

    ‼️ The subtraction is the whole point and is easy to invert: yaw grows
    counter-clockwise, bearings grow clockwise. Facing north gives 0; turning
    LEFT (yaw up) by 90 degrees must read 270 (west), not 90.
    """
    return normalize_deg(math.degrees(north_offset_rad - yaw_rad))


def yaw_for_bearing(bearing_deg, north_offset_rad=0.0):
    """Inverse of bearing_from_yaw: the map yaw that faces a given bearing."""
    yaw = north_offset_rad - math.radians(bearing_deg)
    return math.atan2(math.sin(yaw), math.cos(yaw))      # to (-pi, pi]


def offset_from_known_bearing(yaw_rad, bearing_deg):
    """Calibration: the robot is at ``yaw_rad`` and known to face ``bearing_deg``.

    Returns the north offset to store. Calibrating while facing north
    (bearing 0) simply returns the current yaw, which is the common case.
    """
    off = yaw_rad + math.radians(bearing_deg)
    return math.atan2(math.sin(off), math.cos(off))


def cardinal_el(bearing_deg):
    """Short Greek cardinal for a bearing: 'Β', 'ΝΔ', …"""
    return CARDINALS_EL[int(round(normalize_deg(bearing_deg) / 45.0)) % 8]


def cardinal_full_el(bearing_deg):
    """Spoken Greek name: 'Βορράς', 'Νοτιοδυτικά', …"""
    return _FULL_EL[cardinal_el(bearing_deg)]
