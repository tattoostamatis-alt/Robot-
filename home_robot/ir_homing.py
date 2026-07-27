"""Closed-loop IR homing onto the charging dock.

Replaces the Create 2's built-in Seek Dock (OI opcode 143), which on this
Roomba 879 drove the robot *out* of the approach corridor and never came back
(2026-07-26: omni went 172 -> 161, both directional receivers went to 0).

Pure logic, no ROS and no serial: `step()` takes the three IR receiver bytes
plus whether the wheels actually moved, and returns wheel speeds in mm/s. That
keeps it testable without the robot, which matters because the hardware is
awkward to rig for a dock run.

Two design points worth knowing before changing anything here:

* **Steering is differential, not rotate-then-drive.** This base cannot rotate
  below ~0.31 rad/s and overshoots ~14% above it
  (see the rotation-floor notes), so aiming by pivoting would hunt forever at
  the exact moment precision matters. Driving an arc keeps both wheels turning
  and steers well below that floor. Pivoting is used only when nothing is
  visible and we have to sweep for the beacon.

* **Seating is detected by stall, not by charge.** On this robot the charging
  contacts are disconnected from the mainboard (they feed a power station), so
  charger_state and charging-sources are both permanently 0. Instead: if we are
  commanding forward, both buoys are visible, and the encoders report no
  movement, the robot is physically against the base. Requiring an active
  forward command is what makes this stricter than "stopped and can see IR" —
  a robot that merely halted short of the dock still moves when pushed.
"""

# Home Base emissions. The receivers return 0xA0 | flags; anything else (e.g.
# 162 = virtual wall) is not the dock, so match the known codes explicitly
# rather than masking, which would let 162 through as a buoy-less "dock".
FORCE_FIELD = 0x01
GREEN_BUOY = 0x04
RED_BUOY = 0x08
DOCK_CODES = frozenset({161, 164, 165, 168, 169, 172, 173})

# step() status values
HOMING = 'homing'
SEARCHING = 'searching'
SEATED = 'seated'
LOST = 'lost'


def decode(byte):
    """(force_field, green, red) for a Home Base byte, or None if not the dock."""
    if byte is None or byte not in DOCK_CODES:
        return None
    return (bool(byte & FORCE_FIELD),
            bool(byte & GREEN_BUOY),
            bool(byte & RED_BUOY))


class IrHoming:
    """Drives the final approach from IR receiver bytes.

    Speeds are per-wheel mm/s, the unit `drive_direct` wants, so the caller
    does no conversion and the arc geometry stays visible here.
    """

    def __init__(self,
                 forward_mm_s=100.0,
                 steer_mm_s=25.0,
                 hard_steer_mm_s=50.0,
                 search_mm_s=45.0,      # must clear the ~34 mm/s rotation floor
                 backup_mm_s=-60.0,
                 stall_s=1.2,
                 search_timeout_s=20.0,
                 backup_s=1.0,
                 swap_buoys=False):
        self.forward_mm_s = forward_mm_s
        self.steer_mm_s = steer_mm_s
        self.hard_steer_mm_s = hard_steer_mm_s
        self.search_mm_s = search_mm_s
        self.backup_mm_s = backup_mm_s
        self.stall_s = stall_s
        self.search_timeout_s = search_timeout_s
        self.backup_s = backup_s
        # Which buoy sits on which side is a property of the station, and
        # getting it backwards steers away from the dock. It is a parameter so
        # a bad guess is one launch argument to fix, not a rebuild.
        self.swap_buoys = swap_buoys
        self.reset()

    def reset(self, now=0.0):
        self._lost_since = None
        self._stalled_since = None
        self._backup_until = None
        self._last_status = HOMING

    def _arc(self, bias_mm_s):
        """Forward arc. Positive bias steers left (left wheel slower)."""
        v = self.forward_mm_s
        return (v - bias_mm_s, v + bias_mm_s)

    def step(self, omni, left, right, moved, now):
        """One control tick.

        omni/left/right: raw receiver bytes (packets 17/52/53), None if unread.
        moved:  True if the encoders changed since the last tick.
        now:    monotonic seconds.

        Returns (left_mm_s, right_mm_s, status).
        """
        d_omni = decode(omni)
        d_left = decode(left)
        d_right = decode(right)
        seen = [d for d in (d_omni, d_left, d_right) if d is not None]

        # Backing out of a force-field corner takes priority: we decided to do
        # it, and re-deciding every tick on the same reading would deadlock.
        if self._backup_until is not None:
            if now < self._backup_until:
                return (self.backup_mm_s, self.backup_mm_s, SEARCHING)
            self._backup_until = None

        if not seen:
            if self._lost_since is None:
                self._lost_since = now
            elif (now - self._lost_since) > self.search_timeout_s:
                return (0.0, 0.0, LOST)
            # Sweep in place; this is the one case worth paying the rotation
            # floor for, since an arc would wander away from the dock.
            return (-self.search_mm_s, self.search_mm_s, SEARCHING)
        self._lost_since = None

        green = any(d[1] for d in seen)
        red = any(d[2] for d in seen)
        centred = green and red

        # Seated: commanded forward, centred on both buoys, wheels not turning.
        if centred:
            if not moved:
                if self._stalled_since is None:
                    self._stalled_since = now
                elif (now - self._stalled_since) >= self.stall_s:
                    return (0.0, 0.0, SEATED)
            else:
                self._stalled_since = None
        else:
            self._stalled_since = None

        if centred:
            return (*self._arc(0.0), HOMING)

        # Directional receivers beat buoy colour: which side of the robot the
        # beam lands on is geometry, independent of how the station is laid
        # out, so it needs no swap_buoys guess.
        left_sees = d_left is not None
        right_sees = d_right is not None
        if left_sees != right_sees:
            bias = self.steer_mm_s if left_sees else -self.steer_mm_s
            return (*self._arc(bias), HOMING)

        if green or red:
            # One buoy only: we are off to that beam's side of the corridor.
            # Steer hard, because a single buoy means we are near its edge.
            toward_left = green
            if self.swap_buoys:
                toward_left = not toward_left
            bias = self.hard_steer_mm_s if toward_left else -self.hard_steer_mm_s
            return (*self._arc(bias), HOMING)

        # Force field only: inside the halo that surrounds the station but
        # outside both buoy cones — the state the built-in seek got stuck in.
        # Reverse out and let the sweep re-acquire rather than grinding here.
        self._backup_until = now + self.backup_s
        return (self.backup_mm_s, self.backup_mm_s, SEARCHING)
