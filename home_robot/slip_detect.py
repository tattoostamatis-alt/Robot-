"""Wheel slip / stall detection — the pure, ROS-free core behind
slip_monitor_node.py.

The robot has no way to feel that it is stuck. The bumper only fires on
contact, and a rug, a cable, or a threshold stops it without anything to bump:
the wheels keep turning, odometry keeps integrating, and the robot believes it
is driving across the room while standing still. Nav2 eventually reports
"failed to make progress", which is both late and unspecific.

Two independent tells, because the robot has two kinds of motion:

  ROTATION — the honest one. Wheel odometry says the wheels are turning the
  robot at wz rad/s; the BNO085's gyro says how much it is ACTUALLY turning.
  One rug under one wheel and the two diverge immediately. This needs no
  assumptions about the room.

  LINEAR — harder, because nothing on this robot measures forward motion
  independently. The IMU streams linear_acceleration as literal zeros (the
  SH2_LINEAR_ACCELERATION report is never enabled), and the EKF takes vx
  straight from the wheels, so both would just repeat the wheels' own story.
  What is left is the LiDAR: if the robot is driving at something, the range to
  it must shrink. That works only when there IS something ahead within range
  and the robot is not turning, so this detector reports "unknown" rather than
  guessing whenever those conditions do not hold.

Kept dependency-free (no ROS, no numpy) so the thresholds can be unit-tested
without a robot — see tests/test_slip_detect.py.
"""

from collections import deque

# What "the gyro should have seen something" means. The BNO085 quantises small
# rates to exact 0.0 and streams at only ~6 Hz, so a slow turn is genuinely
# indistinguishable from a stationary robot; below this the comparison is
# meaningless and no verdict is issued at all.
DEFAULT_MIN_WHEEL_WZ = 0.25      # rad/s at the wheels

# Below this fraction of the commanded rotation, the robot is not turning the
# way its wheels say. Deliberately generous: wheel_base calibration error alone
# puts the two within ~20% of each other on a good floor.
DEFAULT_GYRO_RATIO = 0.35

DEFAULT_WINDOW = 1.5             # seconds of history compared
DEFAULT_MIN_GYRO_SAMPLES = 4     # ~0.7 s of a 6 Hz stream

DEFAULT_MIN_VX = 0.08            # m/s below which forward slip is not judged
DEFAULT_RANGE_RATIO = 0.25       # closed less than this fraction of expected
DEFAULT_MAX_RANGE = 3.0          # m — beyond this the LiDAR check is off
DEFAULT_MAX_TURN_WZ = 0.15       # rad/s — turning invalidates the range check


class SlipDetector:
    """Rolling-window comparison of what the wheels claim against what the
    other sensors saw.

    Feed it samples as they arrive (the sources run at different rates: wheels
    ~20 Hz, gyro ~6 Hz, scan ~10 Hz) and ask for a verdict whenever convenient.
    """

    def __init__(self, window=DEFAULT_WINDOW,
                 min_wheel_wz=DEFAULT_MIN_WHEEL_WZ,
                 gyro_ratio=DEFAULT_GYRO_RATIO,
                 min_gyro_samples=DEFAULT_MIN_GYRO_SAMPLES,
                 min_vx=DEFAULT_MIN_VX,
                 range_ratio=DEFAULT_RANGE_RATIO,
                 max_range=DEFAULT_MAX_RANGE,
                 max_turn_wz=DEFAULT_MAX_TURN_WZ):
        self.window = window
        self.min_wheel_wz = min_wheel_wz
        self.gyro_ratio = gyro_ratio
        self.min_gyro_samples = min_gyro_samples
        self.min_vx = min_vx
        self.range_ratio = range_ratio
        self.max_range = max_range
        self.max_turn_wz = max_turn_wz

        self._wheel = deque()     # (t, vx, wz)
        self._gyro = deque()      # (t, gz)
        self._scan = deque()      # (t, front_range)

    # ── input ──────────────────────────────────────────────────────────────
    def feed_wheel(self, t, vx, wz):
        self._wheel.append((t, vx, wz))
        self._trim(self._wheel, t)

    def feed_gyro(self, t, gz):
        self._gyro.append((t, gz))
        self._trim(self._gyro, t)

    def feed_scan(self, t, front_range):
        """`front_range` is the nearest return in a narrow forward sector, or
        None when nothing is in range (an open room)."""
        self._scan.append((t, front_range))
        self._trim(self._scan, t)

    def _trim(self, dq, now):
        cutoff = now - self.window
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def reset(self):
        self._wheel.clear()
        self._gyro.clear()
        self._scan.clear()

    # ── verdict ────────────────────────────────────────────────────────────
    def check_rotation(self):
        """(slipping, detail) — detail carries the numbers for the message.

        `slipping` is None when there is not enough evidence to judge, which is
        a different thing from False and must not be reported as "fine".
        """
        if len(self._gyro) < self.min_gyro_samples or not self._wheel:
            return None, {'reason': 'not enough samples'}

        wheel_wz = _mean(abs(w) for _, _, w in self._wheel)
        if wheel_wz < self.min_wheel_wz:
            return None, {'reason': 'turning too slowly to judge',
                          'wheel_wz': wheel_wz}

        gyro_wz = _mean(abs(g) for _, g in self._gyro)
        ratio = gyro_wz / wheel_wz if wheel_wz else 1.0
        detail = {'wheel_wz': wheel_wz, 'gyro_wz': gyro_wz, 'ratio': ratio}
        return ratio < self.gyro_ratio, detail

    def check_linear(self):
        """Same contract as check_rotation, for forward motion.

        Requires: the robot is driving forward, not turning, and something is
        close enough ahead for the range to be informative. Otherwise None.
        """
        if len(self._wheel) < 2 or len(self._scan) < 2:
            return None, {'reason': 'not enough samples'}

        vx = _mean(v for _, v, _ in self._wheel)
        if vx < self.min_vx:
            return None, {'reason': 'not driving forward', 'vx': vx}
        if _mean(abs(w) for _, _, w in self._wheel) > self.max_turn_wz:
            # A turn changes what the forward sector is pointing at, so the
            # range delta stops being a measure of forward progress.
            return None, {'reason': 'turning'}

        ranges = [(t, r) for t, r in self._scan if r is not None and r <= self.max_range]
        if len(ranges) < 2:
            return None, {'reason': 'nothing ahead to measure against'}

        dt = ranges[-1][0] - ranges[0][0]
        if dt <= 0:
            return None, {'reason': 'no time elapsed'}
        expected = vx * dt                     # metres the gap should have closed
        closed = ranges[0][1] - ranges[-1][1]  # metres it actually closed
        if expected <= 0:
            return None, {'reason': 'no expected motion'}
        ratio = closed / expected
        detail = {'vx': vx, 'expected_m': expected, 'closed_m': closed,
                  'ratio': ratio}
        return ratio < self.range_ratio, detail


def _mean(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
