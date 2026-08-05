#!/usr/bin/env python3
"""Roomba 879 → ROS2 driver via Create 2 Open Interface.

Chassis swapped 692 → 879 on 2026-07-21; the OI protocol is identical, but the
calibration below (wheel_base, mm_per_tick, right_trim) is the 879's and does
NOT carry back to a 692. The header said 692 until 2026-08-01, which is a
trap worth not re-setting: every measured constant in this file was taken on
the 879.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, PointCloud2, PointField
from std_msgs.msg import Bool, Float32, String
import pycreate2
from home_robot import ir_homing
from home_robot.ir_homing import IrHoming
import fcntl
import json
import math
import os
import select
import struct
import termios
import threading
import time


# OI sensor packet 35. Only Passive runs the 5-minute inactivity sleep timer;
# Safe and Full stay awake indefinitely. See _keep_awake().
OI_MODE_OFF, OI_MODE_PASSIVE, OI_MODE_SAFE, OI_MODE_FULL = 0, 1, 2, 3

_BAUD_MAP = {
    9600:   termios.B9600,
    19200:  termios.B19200,
    38400:  termios.B38400,
    57600:  termios.B57600,
    115200: termios.B115200,
}


class _RawTermiosSerial:
    """Drop-in replacement for the pyserial ``Serial`` object that pycreate2's
    SerialCommandInterface talks to, backed by raw ``os``/``termios`` instead.

    Why: on this robot's FTDI FT232R the stock pyserial read path returns
    partial/empty OI responses — the FTDI latency timer (~16ms) can hold a
    short reply just past the driver's fixed post-write sleep, so ``read(4)``
    fires before the bytes land and comes back 0/short. That misframes the
    encoder packets (the "Encoder data not 4 bytes" / "implausible odom delta"
    spam) and starves the EKF of wheel odometry while driving, so the LiDAR
    scan drifts off the map. ``read()`` here *accumulates* until it has exactly
    the requested number of bytes (or the timeout elapses), via ``select`` +
    ``os.read``, so the framing stays aligned regardless of the latency timer.
    Same raw-termios approach as imu_node.py.

    Exposes only what pycreate2's SCI + roomba_driver use: ``is_open``,
    ``write``, ``read``, ``reset_input_buffer``/``reset_output_buffer``,
    ``close``, the ``rts``/``dtr`` modem lines (pycreate2.wake() pulses them
    for the BRC wake — works on FTDI, unlike the CH340 ioctl hang), and the
    ``port``/``baudrate`` attributes its prints read.
    """

    def __init__(self, port, baud, timeout=0.15):
        self.port = port
        self.baudrate = baud
        self.timeout = timeout
        self._rts = False
        self._dtr = False
        self._fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        speed = _BAUD_MAP[baud]
        attrs = termios.tcgetattr(self._fd)
        attrs[0] = 0                                              # iflag: raw input
        attrs[1] = 0                                              # oflag: raw output
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL   # 8N1, no modem ctrl
        attrs[3] = 0                                              # lflag: raw
        attrs[4] = speed                                          # ispeed
        attrs[5] = speed                                          # ospeed
        attrs[6][termios.VMIN] = 0                               # non-blocking reads;
        attrs[6][termios.VTIME] = 0                              # select() gates timing
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)  # blocking mode
        # ‼️ os.open() asserts DTR by default on Linux, i.e. the BRC line is
        # already pulled low the moment the port opens. Leaving it there makes
        # every later `.dtr = True` a no-op -- the robot never sees a falling
        # edge, so _do_brc_pulse pulses nothing (found 2026-07-28). Release it
        # here so the line idles HIGH and self._dtr tells the truth.
        self.dtr = False

    @property
    def is_open(self):
        return self._fd is not None

    def write(self, data):
        return os.write(self._fd, data)

    def read(self, num_bytes):
        """Read exactly num_bytes, accumulating until they arrive or timeout."""
        buf = b''
        deadline = time.monotonic() + self.timeout
        while len(buf) < num_bytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                break
            chunk = os.read(self._fd, num_bytes - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    def reset_input_buffer(self):
        termios.tcflush(self._fd, termios.TCIFLUSH)

    def reset_output_buffer(self):
        termios.tcflush(self._fd, termios.TCOFLUSH)

    def _set_modem_bit(self, bit, level):
        fcntl.ioctl(self._fd,
                    termios.TIOCMBIS if level else termios.TIOCMBIC,
                    struct.pack('I', bit))

    @property
    def rts(self):
        return self._rts

    @rts.setter
    def rts(self, level):
        self._rts = bool(level)
        self._set_modem_bit(termios.TIOCM_RTS, level)

    @property
    def dtr(self):
        return self._dtr

    @dtr.setter
    def dtr(self, level):
        self._dtr = bool(level)
        self._set_modem_bit(termios.TIOCM_DTR, level)

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None


class RoombaDriver(Node):
    def __init__(self):
        super().__init__('roomba_driver')

        self.declare_parameter('port', '/dev/roomba')
        self.declare_parameter('baud', 115200)
        # ── BRC (wake) line ────────────────────────────────────────────────
        # The Roomba sleeps after ~5 min and then ignores serial completely;
        # only a button press or a low pulse on BRC (mini-DIN pin 5) brings it
        # back. On this robot BRC is NOT wired (proven 2026-07-25: 282 wake()
        # calls over 14 min moved nothing on a 99.9% battery, one CLEAN press
        # woke it instantly), so all of this is dormant until someone runs the
        # wire. It is written so that the day the wire exists, nothing else has
        # to change.
        #
        # brc_port: '' means BRC is wired to this same adapter's DTR. Set it to
        # another tty (e.g. a cheap USB-TTL dongle whose DTR goes to pin 5) to
        # drive the line from there instead -- that route leaves the working
        # data cable untouched.
        self.declare_parameter('brc_port', '')
        self.declare_parameter('brc_pulse_s', 0.6)      # OI wants >500ms low
        # How long BRC must sit high before a pulse, so the falling edge is
        # real. See _do_brc_pulse -- without this the pulse is a no-op.
        self.declare_parameter('brc_idle_high_s', 0.2)
        # Settle time between the wake edge and the OI handshake. The robot
        # does not accept 128 the instant BRC goes back high.
        self.declare_parameter('brc_wake_settle_s', 0.5)
        # ‼️ SAFETY, not tuning: per the OI spec, pulsing BRC low THREE times
        # within 5 seconds switches the robot to 19200 baud. pycreate2's wake()
        # does two pulses per call and _keep_awake called it every 2s while the
        # robot was unresponsive -- harmless only because the pin is unwired.
        # With the wire in, that pattern would silently reconfigure the robot
        # and it would look far more broken than merely asleep. One pulse per
        # call, and never more often than this, keeps us well clear.
        self.declare_parameter('brc_min_interval_s', 10.0)
        self.declare_parameter('idle_timeout', 30.0)  # seconds until passive mode
        # How long to let the Create 2's IR dock seek run before giving up and
        # resuming keep-awake. The stock behaviour wanders for a while before
        # finding the beam, so this needs to be generous.
        self.declare_parameter('dock_timeout', 120.0)
        # This robot's battery pack is removed (power bank instead), so
        # charger_state never fires and docking could only ever time out. When
        # true, a stationary robot still seeing the dock's IR beams counts as
        # docked. Set false if a real pack is ever refitted.
        self.declare_parameter('dock_without_battery', True)
        self.declare_parameter('dock_settle_s', 6.0)
        # Closed-loop IR homing instead of OI 143. The stock seek was measured
        # driving this 879 OUT of the approach corridor and never recovering
        # (omni 172 -> 161, both directional receivers to 0), so we steer the
        # final approach ourselves off the same beams. Set false to go back to
        # the built-in behaviour. See home_robot/ir_homing.py.
        self.declare_parameter('use_ir_homing', True)
        # Which buoy is on which side is a property of the station; guessing
        # wrong steers away from it. Flip this if the robot consistently veers
        # off to one side during the approach — no rebuild needed.
        self.declare_parameter('ir_swap_buoys', False)
        self.declare_parameter('ir_forward_mm_s', 100.0)
        self.declare_parameter('ir_search_timeout_s', 20.0)
        # Recalibrated 2026-06-17 via calibrate_odom.py (straight-line +
        # in-place rotation tests, after fixing a bug where the script's
        # blocking input() calls starved the /odom subscription callback
        # of spin time during the motion itself). Previous straight-line
        # calibration (2026-06-13, mm_per_tick=0.359338) was never paired
        # with a rotation test, leaving wheel_base at the uncalibrated
        # spec default — that mismatch was producing the doubled/skewed
        # walls after turns during SLAM mapping.
        # Recalibrated again 2026-06-18 via calibrate_odom.py, full
        # straight-line + in-place rotation tests, this time with a
        # working IMU (MPU9250/6500 at the time, since replaced by a
        # BNO085 — see imu_node.py / firmware/bno085_imu).
        # The rotation test's IMU ground truth measured
        # the robot turning ~2.5 full rotations for the ~1-rotation
        # cmd_vel command (confirmed by direct observation, not a script
        # artifact) — i.e. previous wheel_base values were significantly
        # too large.
        # Recalibrated 2026-07-22 for the Roomba 879 chassis swap (the 692
        # values above were left in place through the swap). mm_per_tick from
        # a tape-measured 2.00 m straight run (odom 3.617 m at the old value
        # -> 0.427692, re-verified at -0.6% over another 2.00 m). wheel_base
        # from a closed-loop in-place spin (rotate_turns.py drives cmd_vel to
        # an exact 1800 deg on /odom) checked against the physical heading
        # mark: it read 5.25 physical turns for 5 odom turns, giving 0.218405,
        # then confirmed dead-on over a 10-turn spin (a 5% error would show as
        # a half-turn miss). IMU ground truth was unavailable (BNO085 silent),
        # so the spin used the floor mark instead.
        self.declare_parameter('wheel_base', 0.218405)
        self.declare_parameter('mm_per_tick', 0.427692)
        # Compensates a physical drive bias (this unit veers right when
        # commanded to drive straight) — independent of wheel_base/
        # mm_per_tick, which are about odometry accuracy, not actual
        # motor output. >1.0 speeds up the right wheel relative to the
        # left. Tune live with `ros2 param set /roomba_driver right_trim
        # <value>` while driving straight, no relaunch needed.
        # Calibrated 2026-06-17 at a higher teleop speed (0.15 m/s) for
        # finer command resolution, then confirmed to still apply at the
        # mapping speed (0.03 m/s) — see right/left round() note below.
        # Re-tuned the same day after the lidar remount; settled on
        # 1.14 after results kept oscillating between attempts (manual
        # push/test noise was bigger than the residual drift itself).
        # Retuned 2026-07-22 for the 879 with the now-working IMU as the
        # drift reference (independent of wheel odom). Auto-drove 1.8m
        # straight via cmd_vel and read IMU-yaw drift: 1.14 veered +31 deg/m
        # LEFT (692 value badly over-corrected), 1.00 -9 deg/m right, 1.031
        # -1.5 deg/m. Least-squares zero-crossing over the three points is
        # 1.033. Drift is ~288 deg/m per trim-unit, so this is touchy —
        # retune the same way if wheels/tyres change.
        self.declare_parameter('right_trim', 1.033)
        # Forward and reverse needed different corrections — not a pure
        # proportional motor gain mismatch, more likely asymmetric
        # caster drag or forward/reverse motor controller response.
        # Applied only when linear.x < 0. Settled on 1.10 the same way
        # as right_trim above (noisy manual tuning oscillated between
        # 0.95 and 1.3 across repeated attempts — stopped chasing exact
        # zero drift and picked a value to move on with).
        self.declare_parameter('right_trim_reverse', 1.10)
        # Per-wheel speed ramp limit, mm/s per second. Joystick/cmd_vel
        # commands set a target speed; the 20Hz _motor_control loop steps
        # the actual drive_direct() output toward that target by at most
        # max_accel * dt per tick instead of jumping straight there — kills
        # both the abrupt lurch on start/reverse and turns the watchdog's
        # stop-on-stale-command into a smooth decel instead of a hard stop
        # (useful since /joy under CPU load sometimes gaps past 0.1s).
        # 2026-07-28: 600 -> 1200 mm/s². At 600 the robot took ~0.8 s to reach
        # full speed and felt sluggish under teleop even at max scale.
        self.declare_parameter('max_accel', 1200.0)  # mm/s^2
        # Soft start. The jolt you feel is entirely in the first fraction of a
        # second: a single ramp rate has to be either gentle off the mark or
        # quick to reach cruising speed, and 2026-07-28 chose quick. So use a
        # gentler rate only while pulling away from a standstill (below
        # soft_start_speed) and the full max_accel above it — smooth departure
        # without going back to the sluggish 600 for the whole range. Applies
        # to turning too, since a spin also starts both wheels from zero.
        # ‼️ Acceleration only. Braking always gets the full rate: making a stop
        # softer means making it longer, which is the wrong trade on a robot
        # that stops for bumpers and e-stops.
        self.declare_parameter('soft_start_accel', 400.0)  # mm/s^2
        self.declare_parameter('soft_start_speed', 120.0)  # mm/s
        # Slew limit on the turn rate itself, rad/s per second. The per-wheel
        # ramp above only softens a turn started from a standstill: begin a turn
        # while already driving and the two wheels are merely asked to diverge,
        # each well inside its own accel limit, so the yaw snaps in. Limiting
        # angular.z directly makes a turn ease in regardless of what the robot
        # was doing beforehand. 2026-07-29, user: "turn more gently".
        # 0 disables it and restores the raw commanded rate.
        self.declare_parameter('max_angular_accel', 2.5)   # rad/s^2
        # OI mode we drive in. 'full' (default since 2026-07-31, at the user's
        # request: "να μην μπαίνει ποτέ για ύπνο"). Safe keeps the Roomba's own
        # cliff/wheel-drop/charger aborts live, but every one of those aborts
        # DROPS THE ROBOT TO PASSIVE — and Passive is where the 5-minute sleep
        # timer lives, so the protections were themselves the thing putting the
        # robot to sleep mid-session. Full never demotes, so it never sleeps.
        # ‼️ Full disables those aborts in the Roomba, so they are reimplemented
        # here in software (cliff_stop / wheel_drop_stop below) off the same
        # 20 Hz sensor read the bumper already uses. Set oi_mode:='safe' to hand
        # the job back to the firmware.
        self.declare_parameter('oi_mode', 'full')
        # Software replacements for what Full mode switches off. These are NOT
        # optional decoration: without them 'full' means the robot will happily
        # drive down a step. Cliff = packets 9-12, wheel drop = packet 7 bits
        # 2/3, both riding the existing encoder query (4 extra bytes, no extra
        # serial transaction).
        self.declare_parameter('cliff_stop', True)
        self.declare_parameter('wheel_drop_stop', True)
        # How long forward stays blocked after a cliff clears, same idea as
        # bump_block_s: stops the robot nosing straight back over the edge.
        self.declare_parameter('cliff_block_s', 1.5)
        # Bumper protection. HW-verified 2026-07-28: packet 7 reports [1, 3] and
        # the light bumper (45) reports too — despite the stale comment in
        # obstacle_safety_node claiming this robot has no bump sensors. Nothing
        # was reading them, so the robot drove into furniture at full speed.
        self.declare_parameter('bump_stop', True)
        # Keep forward blocked this long after the bumper releases, so the
        # rebound doesn't drive straight back into the same obstacle.
        self.declare_parameter('bump_block_s', 1.0)
        # How stale the bumper/cliff readings may be before forward motion is
        # refused. They are only refreshed by get_encoders() at 20 Hz, so a
        # second is twenty missed reads — a stalled link, not a slow one.
        self.declare_parameter('sensor_stale_s', 1.0)
        # Reverse felt too fast at the same scale_linear.x as forward —
        # scales the commanded speed down (not a calibration trim like
        # right_trim_reverse above, just a deliberate speed cap) whenever
        # linear.x < 0. 1.0 = same speed as forward.
        self.declare_parameter('reverse_speed_scale', 0.6)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.dock_timeout = self.get_parameter('dock_timeout').value
        self.dock_without_battery = self.get_parameter('dock_without_battery').value
        self.dock_settle_s = self.get_parameter('dock_settle_s').value
        self._dock_last_pose = (None, None)
        self._dock_still_since = 0.0
        self.use_ir_homing = self.get_parameter('use_ir_homing').value
        self._homing = None                  # IrHoming while an approach runs
        self._homing_last_status = None      # log transitions, not every tick
        self._dock_last_odom = None
        self.mm_per_tick = self.get_parameter('mm_per_tick').value
        self.right_trim = self.get_parameter('right_trim').value
        self.right_trim_reverse = self.get_parameter('right_trim_reverse').value
        self.max_accel = self.get_parameter('max_accel').value
        self.soft_start_accel = self.get_parameter('soft_start_accel').value
        self.soft_start_speed = self.get_parameter('soft_start_speed').value
        self.max_angular_accel = self.get_parameter('max_angular_accel').value
        self.bump_stop = self.get_parameter('bump_stop').value
        self.bump_block_s = self.get_parameter('bump_block_s').value
        self.sensor_stale_s = max(0.1, float(self.get_parameter('sensor_stale_s').value))
        self._write_failures = 0
        self._last_write_ok = 0.0
        self._stale_warned = False
        self._bump = False
        self._bump_warned = False
        self._last_bump_time = 0.0
        self.cliff_stop = self.get_parameter('cliff_stop').value
        self.wheel_drop_stop = self.get_parameter('wheel_drop_stop').value
        self.cliff_block_s = self.get_parameter('cliff_block_s').value
        self._cliff = False
        # Which of the four fired, left to right: (left, front-left,
        # front-right, right). See _publish_cliffs for why the individual bits
        # matter and the merged flag is not enough.
        self._cliff_bits = (False, False, False, False)
        self._cliff_warned = False
        self._last_cliff_time = 0.0
        self._wheel_drop = False
        self._wheel_drop_warned = False
        self.oi_mode = str(self.get_parameter('oi_mode').value).lower()
        if self.oi_mode not in ('safe', 'full'):
            self.get_logger().warn(
                f"oi_mode '{self.oi_mode}' not recognised — falling back to 'safe'")
            self.oi_mode = 'safe'
        self.reverse_speed_scale = self.get_parameter('reverse_speed_scale').value
        self.brc_port = self.get_parameter('brc_port').value
        self.brc_pulse_s = self.get_parameter('brc_pulse_s').value
        self.brc_idle_high_s = self.get_parameter('brc_idle_high_s').value
        self.brc_wake_settle_s = self.get_parameter('brc_wake_settle_s').value
        self.brc_min_interval_s = self.get_parameter('brc_min_interval_s').value
        self._brc_thread = None
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.bot = pycreate2.Create2(port, baud)
        # Swap pycreate2's stock pyserial backend for a raw-termios one: the
        # pyserial read path returned partial/empty OI responses on this FTDI
        # adapter (latency-timer races -> "Encoder data not 4 bytes" / discarded
        # "implausible odom delta"), which starved the EKF of wheel odometry and
        # let the LiDAR scan drift off the map while driving. _RawTermiosSerial
        # accumulates each read to the exact expected length. pycreate2 keeps
        # owning the OI protocol (drive_direct/wake/start/full/get_sensors).
        try:
            self.bot.SCI.ser.close()
        except Exception:
            pass
        self.bot.SCI.ser = _RawTermiosSerial(port, baud)
        # Connection health, used by the keep-awake watchdog (_keep_awake) to
        # auto-recover if the Roomba falls asleep. The one-shot init below is
        # NOT enough on its own: if the robot is already asleep at boot it
        # ignores these commands, and without re-trying the driver would stay
        # stuck reading 0 bytes forever (only a wake + re-init fixes it).
        self._last_ok_time = 0.0
        self._last_oi_mode = None      # packet 35, refreshed by every read
        self._last_brc_pulse = 0.0
        # A base that is asleep answers every query with 0 bytes, forever. At
        # 20 Hz x 3 retries that is 60 warnings and 20 errors a SECOND, all
        # identical — 256 of them filled the dashboard's log tab on 2026-08-02
        # and pushed out everything else, while never once naming the cause.
        # _note_link_failure replaces that with one diagnosis and a periodic
        # reminder; see it for why the advice is a power cycle.
        self._link_down_since = 0.0
        self._link_last_report = 0.0
        self._connect_oi()
        self.get_logger().info(f'Roomba driver started on {port}')

        self._idle = False
        self._stopped = True
        self._estop = False
        # True between a /dock request and the charger engaging (or timeout):
        # the Create 2 drives itself and our housekeeping must not interfere.
        self._docking = False
        self._dock_started = 0.0
        self._last_cmd_time = time.monotonic()
        self._target_left = 0.0
        self._target_right = 0.0
        # Eased turn rate (see _ramp_angular) and when it last advanced.
        self._cur_angular = 0.0
        self._last_angular_time = time.monotonic()
        self._cur_left = 0.0
        self._cur_right = 0.0
        self._last_control_time = time.monotonic()

        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        self.dock_sub = self.create_subscription(Bool, 'dock', self._dock_cb, 10)
        # Soft e-stop (joystick_estop_node): latched so we pick up the current
        # state even if that node published before this driver subscribed. When
        # engaged the wheels are forced to zero regardless of any cmd_vel_safe
        # source (teleop bypasses obstacle_safety straight to cmd_vel_safe, so
        # this driver is the only place a stop can be guaranteed).
        estop_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.estop_sub = self.create_subscription(
            Bool, 'emergency_stop', self._estop_cb, estop_qos)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.battery_pub = self.create_publisher(BatteryState, 'battery/state', 10)
        self.charge_ratio_pub = self.create_publisher(Float32, 'battery/charge_ratio', 10)
        # How the approach is going, for whoever asked for it. IR homing used
        # to decide seated/lost purely in the log, so a caller that had driven
        # the robot to the handover point had no way to tell success from a
        # 120 s timeout — and the mission that requested the dock reported
        # "arrived at the base" either way. Latched, so a subscriber that comes
        # up late still sees how the last attempt ended.
        self.dock_status_pub = self.create_publisher(
            String, 'dock_status',
            QoSProfile(depth=1,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        # Everything the web dashboard's vacuum panel needs, in one JSON topic.
        # These are all fields we already track in memory (packet 7 bumper/cliff,
        # packet 35 OI mode, the e-stop and docking flags), so publishing them
        # costs no extra serial traffic. Latched: a browser that connects between
        # ticks still paints a populated panel instead of dashes.
        self.status_pub = self.create_publisher(
            String, 'roomba/status',
            QoSProfile(depth=1,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        # ── Cliffs as costmap obstacles ──────────────────────────────────
        # The four cliff sensors have always stopped the robot (cliff_stop
        # below) and there it ended: nothing downstream ever heard about it, so
        # the planner kept routing over the same stair and the robot kept
        # driving up to it, stopping, and being sent again. Publishing them as
        # points lets the costmap remember the edge.
        #
        # ‼️ MARKING ONLY on the costmap side (see nav2_params.yaml). A drop is
        # invisible to every other sensor — the LiDAR sweeps straight over a
        # stairwell and reports open floor — so a raytrace would clear the mark
        # the moment the robot backed away, which is precisely when it must not.
        self.declare_parameter('publish_cliff_obstacles', True)
        # Where the four sensors sit, as bearing from straight ahead and
        # distance from base_link. ‼️ ESTIMATED from the 34 cm chassis, NOT
        # measured: the angles put the marks in roughly the right arc, and if a
        # drop ever marks visibly off to one side these are the numbers to fix.
        self.declare_parameter('cliff_bearings_deg', [65.0, 25.0, -25.0, -65.0])
        self.declare_parameter('cliff_radius', 0.15)
        self._cliff_pub = self.create_publisher(
            PointCloud2, 'cliff_obstacles', 5)
        if self.get_parameter('publish_cliff_obstacles').value:
            self.create_timer(0.2, self._publish_cliffs)   # 5 Hz

        self.create_timer(0.05, self._publish_odom)     # 20 Hz
        self.create_timer(0.05, self._motor_control)     # 20 Hz ramped output + watchdog
        self.create_timer(0.5,  self._publish_status)   # 2 Hz dashboard telemetry
        self.create_timer(2.0,  self._publish_battery)  # 0.5 Hz
        self.create_timer(5.0,  self._check_idle)       # idle watchdog
        self.create_timer(2.0,  self._keep_awake)        # auto-reconnect + BRC keep-alive
        self.create_timer(2.0,  self._check_docking)     # exit docking state on charge/timeout
        self.create_timer(0.1,  self._ir_homing_step)    # 10 Hz closed-loop dock approach

        self._x = 0.0
        self._y = 0.0
        self._th = 0.0
        self._prev_left_enc = None
        self._prev_right_enc = None
        self._prev_odom_time = None

    # Guards the dashboard's safety tab can switch, and how long they hold
    # after they release. Kept apart from the calibration trims below because
    # turning one of these off means the robot drives into things: the only
    # legitimate reason is a sensor that reports a permanent bump, and then the
    # log line is the record of who did it and when.
    _SAFETY_FLAGS = ('bump_stop', 'cliff_stop', 'wheel_drop_stop')
    _SAFETY_TIMES = {'bump_block_s': (0.0, 10.0), 'cliff_block_s': (0.0, 10.0)}

    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            limits = self._SAFETY_TIMES.get(p.name)
            if limits is None:
                continue
            try:
                value = float(p.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False, reason=f'{p.name} must be a number')
            if not limits[0] <= value <= limits[1]:
                return SetParametersResult(
                    successful=False,
                    reason=f'{p.name} must be between {limits[0]} and {limits[1]}')
        for p in params:
            if p.name in self._SAFETY_FLAGS:
                setattr(self, p.name, bool(p.value))
                self.get_logger().warn(
                    f'{p.name} turned {"ON" if p.value else "OFF"} at runtime'
                    + ('' if p.value else
                       ' — the base is in FULL mode and will not stop itself'))
            elif p.name in self._SAFETY_TIMES:
                setattr(self, p.name, float(p.value))
                self.get_logger().info(f'{p.name} is now {float(p.value):.1f}s')
        for p in params:
            if p.name == 'right_trim':
                self.right_trim = p.value
            elif p.name == 'right_trim_reverse':
                self.right_trim_reverse = p.value
            elif p.name == 'max_accel':
                self.max_accel = p.value
            elif p.name == 'soft_start_accel':
                self.soft_start_accel = p.value
            elif p.name == 'soft_start_speed':
                self.soft_start_speed = p.value
            elif p.name == 'max_angular_accel':
                self.max_angular_accel = p.value
            elif p.name == 'reverse_speed_scale':
                self.reverse_speed_scale = p.value
        return SetParametersResult(successful=True)

    def _drive(self, right_mm_s, left_mm_s) -> bool:
        """The ONE place wheels are commanded. Returns whether the write landed.

        ‼️ Why this exists at all. The OI holds the last velocity it was given —
        it has no watchdog of its own, and in `full` mode (the default here) the
        Roomba does not act on its own bumper or cliff either. So a serial write
        that throws is not "a dropped frame": the robot carries on at whatever
        it was last told, blind, and the exception takes the 20 Hz timer with
        it, so no zero ever gets sent. A link lost at 200 mm/s meant 200 mm/s
        until somebody caught the robot.

        On failure the only thing worth doing is trying to stop, hard and
        repeatedly, before anything else — hence the immediate zero retries and
        the reconnect. Driving does not resume until a write succeeds.
        """
        try:
            self.bot.drive_direct(int(right_mm_s), int(left_mm_s))
        except Exception as exc:                          # noqa: BLE001
            self._on_write_failure(exc)
            return False
        if self._write_failures:
            self.get_logger().info('Serial link back — wheels under control again')
            self._write_failures = 0
        self._last_write_ok = time.monotonic()
        return True

    def _on_write_failure(self, exc):
        self._write_failures += 1
        # Our own ramp state has to go to zero too, or a link that comes back
        # resumes at the old speed instead of ramping up from a standstill.
        self._target_left = self._target_right = 0.0
        self._cur_left = self._cur_right = 0.0
        self._cur_angular = 0.0

        if self._write_failures == 1:
            self.get_logger().error(
                f'Wheel command failed ({exc!r}) — THE ROOMBA KEEPS ITS LAST '
                'SPEED, so stopping it is now the only thing that matters.')
        # Three fast attempts at a zero: a single dropped byte is common and
        # usually the next write goes through immediately.
        for _ in range(3):
            try:
                self.bot.drive_direct(0, 0)
                self.get_logger().warn('Wheels stopped after a failed write')
                self._last_write_ok = time.monotonic()
                return
            except Exception:                             # noqa: BLE001, S110
                time.sleep(0.02)
        # Still nothing. Re-open the port; _connect_oi() re-enters the drive
        # mode, and the next tick's zero write will land if the port is back.
        if self._write_failures in (4, 20) or self._write_failures % 100 == 0:
            self.get_logger().error(
                f'Wheels have not answered {self._write_failures} times — '
                'reconnecting the serial link. If the robot is still moving, '
                'stop it by hand.')
        try:
            self._connect_oi()
        except Exception:                                 # noqa: BLE001, S110
            pass

    def _sensors_fresh(self) -> bool:
        """Is the bumper/cliff state recent enough to be trusted?

        Both are only refreshed inside get_encoders(). When the link stalls they
        freeze at their last value while `bump_block_s` quietly expires, so the
        forward block lifts on a timer with nobody actually watching the front
        of the robot. Unknown is treated as unsafe.
        """
        if not self._last_ok_time:
            return False              # nothing read yet since startup
        return (time.monotonic() - self._last_ok_time) < self.sensor_stale_s

    def _estop_cb(self, msg: Bool):
        if msg.data == self._estop:
            return
        self._estop = msg.data
        if self._estop:
            # hard stop immediately — no ramp, this is an emergency
            self._target_left = self._target_right = 0.0
            self._cur_left = self._cur_right = 0.0
            self._cur_angular = 0.0
            self._drive(0, 0)   # retries + reconnects on failure; see _drive()
            self._stopped = True
            self.get_logger().warn('EMERGENCY STOP engaged — ignoring cmd_vel until reset')
        else:
            # release: drop any stale target so we don't lurch on the next cmd
            self._target_left = self._target_right = 0.0
            self._cur_angular = 0.0
            self.get_logger().info('Emergency stop released — motion re-enabled')

    def _cmd_vel_cb(self, msg: Twist):
        if self._estop:
            return  # commands are ignored while e-stopped
        self._last_cmd_time = time.monotonic()
        self._stopped = False
        if self._idle:
            self._idle = False
            self._enter_drive_mode()
            self.get_logger().info(f'Roomba: waking up ({self.oi_mode} mode)')

        linear = msg.linear.x * 1000   # m/s → mm/s
        if linear < 0:
            linear *= self.reverse_speed_scale
        angular = self._ramp_angular(msg.angular.z)   # rad/s

        # round(), not int(): at low teleop speeds (mapping uses 30mm/s)
        # the commanded mm/s is small enough that truncating toward zero
        # swallows trim differences below ~3% — e.g. int(30*1.10) and
        # int(30*1.12) both truncate to 33, making the trim untunable at
        # this speed. round() halves that quantization error.
        trim = self.right_trim_reverse if linear < 0 else self.right_trim
        right = (linear + angular * self.wheel_base * 500) * trim
        left  = linear - angular * self.wheel_base * 500

        self._target_right = max(-500.0, min(500.0, right))
        self._target_left  = max(-500.0, min(500.0, left))

    def _ramp_angular(self, requested):
        """Ease the commanded turn rate toward `requested`, rad/s.

        Called per cmd_vel message, so dt is the gap between commands (teleop
        and Nav2 both publish at ~20Hz). A long gap means the previous command
        already timed out and the robot is stopping, so treat it as a fresh
        start rather than integrating across the pause.
        """
        now = time.monotonic()
        dt = now - self._last_angular_time
        self._last_angular_time = now

        if self.max_angular_accel <= 0.0:      # limiter disabled
            self._cur_angular = requested
            return requested
        # 0.25 s is the motor watchdog's timeout: past it the wheels have
        # already been zeroed, so carrying the old rate over would fight the
        # fresh command instead of easing it in. (_stopped is no use here — the
        # caller clears it before we run.)
        if dt > 0.25:
            self._cur_angular = 0.0
            dt = 0.0

        step = self.max_angular_accel * dt
        self._cur_angular += max(-step, min(step, requested - self._cur_angular))
        return self._cur_angular

    # Below this, the "forward" component of a wheel pair is indistinguishable
    # from the right_trim asymmetry: the trim multiplies the right wheel only,
    # so even a pure spin averages slightly positive (~4 mm/s at the 1.2 rad/s
    # teleop ceiling). Treating that as forward would make the guards below
    # block turning on the spot — the one manoeuvre that gets the robot out of
    # what it just hit. 20 mm/s is well clear of the artefact and far below any
    # deliberate forward speed (mapping, the slowest, uses 30 mm/s).
    _FORWARD_EPS_MM_S = 20.0

    @classmethod
    def _is_forward(cls, target_left, target_right):
        """Does this wheel pair move the robot forward overall?

        ‼️ Forward speed is the AVERAGE of the two wheels, not the sign of each
        one. This used to be written `target_left > 0 and target_right > 0`,
        which misses every arc: at this chassis' wheel_base, 0.10 m/s with
        1.0 rad/s of turn comes out left=-9, right=+216 — still 103 mm/s
        forward, i.e. the full commanded speed — but with one wheel negative
        the per-wheel test let it through with the bumper pressed or a cliff
        under the nose. Anything with |angular| > linear/0.109 hits this, and
        MPPI, the spin recovery and doorway manoeuvres all arc constantly, so
        it was the common case rather than an edge one. Found 2026-08-01.
        """
        return (target_left + target_right) / 2.0 > cls._FORWARD_EPS_MM_S

    @staticmethod
    def _strip_forward(target_left, target_right):
        """Cancel the net forward motion of a wheel pair, keeping the turn.

        Subtracting the average from both wheels leaves their difference — the
        rotation — exactly as commanded, so a blocked robot can still turn away
        from the obstacle instead of sitting against it until something else
        gives up. Reverse is never routed here at all (see _is_forward).
        """
        forward = (target_left + target_right) / 2.0
        return target_left - forward, target_right - forward

    def _ramp_step(self, current, target, dt):
        """How much this wheel's speed may change this tick, in mm/s.

        Gentler while accelerating away from a standstill (that is where the
        lurch is felt); full rate everywhere else. Braking and reversing
        direction always get max_accel — a softer stop is a longer stop.
        """
        speeding_up = abs(target) > abs(current)
        same_direction = current * target >= 0
        if speeding_up and same_direction and abs(current) < self.soft_start_speed:
            return self.soft_start_accel * dt
        return self.max_accel * dt

    def _motor_control(self):
        """20Hz ramped output: steps drive_direct() toward the latest cmd_vel
        target by at most max_accel*dt per tick, instead of jumping straight
        there — also doubles as the stale-command watchdog (target -> 0 on
        timeout) so a dropped controller decelerates smoothly instead of
        slamming to a stop."""
        now = time.monotonic()
        dt = now - self._last_control_time
        self._last_control_time = now

        if self._estop:
            # keep the wheels pinned at zero for as long as the e-stop is held
            self._cur_left = self._cur_right = 0.0
            self._drive(0, 0)
            return

        # The dock seek drives the robot itself, so stop fighting it. Note the
        # write below is UNCONDITIONAL: _idle only zeroes the *target*, it
        # never skips the drive_direct call, so without this guard the driver
        # keeps commanding the wheels at 20Hz throughout the dock approach.
        # E-stop above still wins, deliberately.
        #
        # Honest status: suppressing this did NOT make docking work (the robot
        # moved 13cm in 120s before the guard, 21cm after -- both attempts
        # timed out without charging). Kept because commanding the wheels
        # while a built-in behaviour owns them is wrong regardless; the real
        # blocker for docking is still open. See the note in _dock_cb.
        if self._docking:
            return

        target_left, target_right = self._target_left, self._target_right
        if not self._idle and (now - self._last_cmd_time) > 0.25:
            target_left = target_right = 0.0
            self._stopped = True

        # Bumper: block FORWARD motion while the bumper is pressed, and for
        # bump_block_s after it releases so a rebound doesn't immediately drive
        # back into the same obstacle. Reversing and turning stay available —
        # that is how the robot gets out. Safe mode does NOT cover the bumper
        # (only cliff/wheel-drop/charger), so this is the only bump protection
        # the robot has, in teleop and under Nav2 alike.
        # Stale sensors = unknown, and unknown is treated as unsafe. The bumper
        # and cliff are only refreshed inside get_encoders(); if that stalls,
        # they freeze at their last value while bump_block_s quietly expires, so
        # the forward block would lift on a timer with nothing actually watching
        # the front of the robot. Turning and reversing stay available, exactly
        # as with a real bump — this is not a full stop.
        if (self.bump_stop or self.cliff_stop) and not self._sensors_fresh():
            if self._is_forward(target_left, target_right):
                target_left, target_right = self._strip_forward(target_left, target_right)
                if not self._stale_warned:
                    self._stale_warned = True
                    self.get_logger().warn(
                        'No fresh bumper/cliff reading — blocking forward motion '
                        'until the serial link answers again')
        elif self._stale_warned and self._sensors_fresh():
            self._stale_warned = False
            self.get_logger().info('Bumper/cliff readings are fresh again')

        if self.bump_stop and (now - self._last_bump_time) < self.bump_block_s:
            if self._is_forward(target_left, target_right):
                target_left, target_right = self._strip_forward(target_left, target_right)
                if not self._bump_warned:
                    self._bump_warned = True
                    self.get_logger().warn('Bumper pressed — forward motion blocked')
        elif self._bump_warned:
            self._bump_warned = False
            self.get_logger().info('Bumper clear')

        # Cliff: in Full mode the Roomba will drive off a step without this.
        # Blocks FORWARD only — reverse is exactly how the robot backs away
        # from an edge, so taking that away would strand it on the lip.
        if self.cliff_stop and (now - self._last_cliff_time) < self.cliff_block_s:
            if self._is_forward(target_left, target_right):
                target_left, target_right = self._strip_forward(target_left, target_right)
                if not self._cliff_warned:
                    self._cliff_warned = True
                    self.get_logger().warn(
                        'ΓΚΡΕΜΟΣ — μπλοκάρω την εμπρόσθια κίνηση (όπισθεν ελεύθερη)')
        elif self._cliff_warned:
            self._cliff_warned = False
            self.get_logger().info('Cliff clear')

        # Wheel drop: the robot has been picked up, or a wheel is over an edge.
        # Unlike the cliff this stops motion in EVERY direction — there is no
        # safe way to keep driving a robot that is not on the floor, and Safe
        # mode used to enforce exactly that (by dropping to Passive, which is
        # what put it to sleep).
        if self.wheel_drop_stop and self._wheel_drop:
            target_left = target_right = 0.0
            if not self._wheel_drop_warned:
                self._wheel_drop_warned = True
                self.get_logger().warn('Τροχός στον αέρα — σταματώ εντελώς')
        elif self._wheel_drop_warned and not self._wheel_drop:
            self._wheel_drop_warned = False
            self.get_logger().info('Wheels back on the floor')

        step_left = self._ramp_step(self._cur_left, target_left, dt)
        step_right = self._ramp_step(self._cur_right, target_right, dt)
        self._cur_left  += max(-step_left,  min(step_left,  target_left  - self._cur_left))
        self._cur_right += max(-step_right, min(step_right, target_right - self._cur_right))

        self._drive(round(self._cur_right), round(self._cur_left))

    def _check_idle(self):
        if self._docking:      # drive_direct(0,0) would abort the dock approach
            return
        if not self._idle and (time.monotonic() - self._last_cmd_time) > self.idle_timeout:
            self._idle = True
            self._target_left = self._target_right = 0.0
            self._cur_left = self._cur_right = 0.0
            self._drive(0, 0)
            self._stopped = True
            # Wheels stopped, OI mode untouched — this does NOT enter Passive,
            # and it must not: Passive is where the sleep timer lives.
            self.get_logger().info(
                f'Roomba: idle {self.idle_timeout:.0f}s → wheels stopped '
                f'(staying in {self.oi_mode})'
            )

    def _set_brc_low(self, fd: int, low: bool):
        """Drive the BRC line. On a USB-TTL adapter, asserting DTR pulls the
        pin to 0V, which is the active (wake) level BRC wants."""
        fcntl.ioctl(fd,
                    termios.TIOCMBIS if low else termios.TIOCMBIC,
                    struct.pack('I', termios.TIOCM_DTR))

    def _do_brc_pulse(self):
        """One low pulse, long enough for the OI to notice. Runs on its own
        thread: pycreate2's wake() blocks for 3 SECONDS inside what used to be
        a 2s timer callback, which stalled odometry publishing and every other
        timer on this node for as long as the robot stayed asleep."""
        # ‼️ Always release the line and let it settle HIGH first. BRC wakes on
        # a falling edge, and os.open() already asserts DTR (= line low), so
        # driving it low again from low is a no-op: the robot sees BRC held low
        # forever and never a pulse. Verified 2026-07-28 with TIOCMGET
        # read-back -- before this, the only real edge the robot ever got was
        # the port being opened.
        try:
            if self.brc_port:
                fd = os.open(self.brc_port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
                try:
                    self._set_brc_low(fd, False)
                    time.sleep(self.brc_idle_high_s)
                    self._set_brc_low(fd, True)
                    time.sleep(self.brc_pulse_s)
                    self._set_brc_low(fd, False)
                finally:
                    os.close(fd)
            else:
                # Same adapter as the data line: _RawTermiosSerial exposes the
                # modem bits, and toggling them does not disturb bytes in flight.
                self.bot.SCI.ser.dtr = False
                time.sleep(self.brc_idle_high_s)
                self.bot.SCI.ser.dtr = True
                time.sleep(self.brc_pulse_s)
                self.bot.SCI.ser.dtr = False
        except Exception as e:
            self.get_logger().warn(f'BRC pulse failed: {e!r}')

    def _brc_pulse(self):
        """Request a BRC pulse, rate-limited and non-blocking.

        The rate limit is a safety interlock, not a nicety: three pulses inside
        5 seconds put the robot on 19200 baud (OI spec). Skipping a pulse costs
        nothing -- the sleep timer is 5 minutes wide."""
        now = time.monotonic()
        if now - self._last_brc_pulse < self.brc_min_interval_s:
            return None
        if self._brc_thread is not None and self._brc_thread.is_alive():
            return None
        self._last_brc_pulse = now
        self._brc_thread = threading.Thread(target=self._do_brc_pulse, daemon=True)
        self._brc_thread.start()
        return self._brc_thread

    def _enter_drive_mode(self):
        """Put the OI into the mode we drive in — see the `oi_mode` param.

        Safe keeps the Roomba's cliff/wheel-drop/charger aborts active; Full
        turns them off. Both accept drive_direct(); the difference is only who
        is allowed to stop the robot.
        """
        if self.oi_mode == 'full':
            self.bot.full()
        else:
            self.bot.safe()

    def _connect_oi(self):
        """(Re)initialise the Open Interface: wake → Passive → Full. Safe to
        call repeatedly. Called at startup and by _keep_awake to recover
        automatically after the Roomba has gone to sleep (it ignores serial
        while asleep, so a single boot-time init is not enough)."""
        try:
            pulse = self._brc_pulse()   # pulse BRC low — wakes the robot
            if pulse is not None:
                # ‼️ WAIT for the pulse to finish. _brc_pulse runs it on its own
                # thread and returns immediately, so start(128) used to go out
                # ~0.8 s BEFORE the falling edge had even been delivered — the
                # handshake reached a robot that was still asleep, failed, and
                # the next read then looked exactly like "BRC does nothing".
                # Combined with the 10 s rate limit, the 2 s watchdog meant the
                # handshake was NEVER sent at a moment when the robot could
                # hear it. Found 2026-07-28.
                pulse.join(timeout=self.brc_idle_high_s + self.brc_pulse_s + 1.0)
                # The OI needs a moment after the edge before it listens.
                time.sleep(self.brc_wake_settle_s)
            self.bot.start()     # enter OI (Passive)
            time.sleep(0.3)
            self._enter_drive_mode()
            time.sleep(0.2)
        except Exception as e:
            self.get_logger().warn(f'OI (re)connect failed: {e!r}')

    def _keep_awake(self):
        """Never let the Roomba sleep. Three jobs:
          1. If there has been no good sensor/encoder read for >2s the robot
             is asleep or dropped out of OI — re-run wake/start/full so it
             recovers on its own (e.g. right after the user presses CLEAN),
             with no driver restart needed.
          2. If the OI has fallen back to Passive, climb straight back out.
             This is the one that actually keeps the robot awake here: the
             5-minute inactivity sleep timer runs ONLY in Passive, while Safe
             and Full never auto-sleep. Safe is demoted to Passive silently by
             any safety abort — a wheel drop when someone picks the robot up,
             a cliff reading — so "we entered Safe once at startup" is not the
             same as "we are in Safe now".
          3. While healthy, pulse BRC at least once a minute so the OI's
             sleep timer never expires. ‼️ BRC is NOT wired on this chassis
             (proven 2026-07-25: 282 pulses, no response), so this job is a
             no-op here and cannot be what keeps the robot up — job 2 is.

        Getting this wrong is expensive and silent: on 2026-07-31 the robot
        slept at 20:40:24 and a patrol started at 20:49:01 drove a base that
        was no longer listening, reporting nav timeouts for 6 rooms while the
        pose never changed by a millimetre."""
        # Hands off during a dock approach. On the legacy 143 path the robot
        # is in Passive driving itself and _connect_oi()'s full() would cancel
        # the seek; on the IR homing path we are mid-approach in Full and a
        # re-init would stutter the wheels. The homing loop's own 10Hz reads
        # keep _last_ok_time fresh, and Full mode never auto-sleeps.
        if self._docking:
            return
        now = time.monotonic()
        if (now - self._last_ok_time) > 2.0:
            self._connect_oi()
            return
        if self._last_oi_mode == OI_MODE_PASSIVE:
            self.get_logger().warn(
                f'OI dropped to passive — re-entering {self.oi_mode} before the '
                f'sleep timer runs')
            self._enter_drive_mode()
            self._last_oi_mode = None    # re-read it, do not re-assert blindly
            return
        if (now - self._last_brc_pulse) > 60.0:
            self._brc_pulse()

    # How often to repeat the "base is asleep" line while it stays asleep.
    LINK_REPORT_PERIOD = 30.0

    def _note_link_failure(self, what: str, detail: str):
        """One line per outage, then one every 30 s — not 80 a second.

        ‼️ Read the shape of the failure before repeating it. Every read
        returning ZERO bytes is not a flaky cable, it is the documented signature
        of a Roomba that has gone to sleep: the OI stops answering entirely and
        stays that way. On this robot the wake paths are gone — the BRC line is
        dead (all 10 DTR/RTS/polarity/baud combinations were swept on
        2026-07-31) — so nothing the driver can send will bring it back. The
        only thing that works is a power cycle, which is why that is what the
        message says instead of a byte count nobody can act on.

        The old code logged each of the three retries at WARN plus a final
        ERROR, from two pollers, at 20 Hz. That is the noise this replaces.
        """
        now = time.monotonic()
        if not self._link_down_since:
            self._link_down_since = now
            self._link_last_report = 0.0
        if now - self._link_last_report < self.LINK_REPORT_PERIOD:
            return
        self._link_last_report = now
        secs = int(now - self._link_down_since)
        if 'not 9 bytes long, it is: 0' in detail or 'not 80 bytes long, it is: 0' in detail:
            self.get_logger().error(
                f'Η ΒΑΣΗ ΔΕΝ ΑΠΑΝΤΑ ({secs}s): κάθε ερώτημα γυρνά 0 bytes — το '
                'Roomba κοιμάται. Το BRC είναι νεκρό σε αυτό το σκαφάκι, οπότε '
                'ο driver ΔΕΝ μπορεί να το ξυπνήσει: κάνε power cycle (~10s '
                'από το power station). Μέχρι τότε δεν υπάρχει odometry.')
        else:
            self.get_logger().error(
                f'{what} αποτυγχάνει εδώ και {secs}s: {detail}')

    def _note_link_ok(self):
        """Say so once when the base comes back, then go quiet again."""
        if self._link_down_since:
            secs = int(time.monotonic() - self._link_down_since)
            self.get_logger().info(f'Η βάση ξαναπαντά μετά από {secs}s')
            self._link_down_since = 0.0

    def _safe_get_sensors(self):
        """get_sensors() με retry — το Roomba μερικές φορές επιστρέφει 0 bytes."""
        last = None
        for _ in range(3):
            try:
                # Discard any stale/leftover bytes so this read is aligned to
                # the response of the query we're about to send — otherwise a
                # backlog from a delayed previous cycle desyncs every field.
                self.bot.SCI.ser.reset_input_buffer()
                val = self.bot.get_sensors()
                self._last_ok_time = time.monotonic()
                # Packet 35. The only way to know we are still in Safe/Full:
                # a safety abort demotes the OI without saying so anywhere.
                mode = getattr(val, 'open_interface_mode', None)
                if mode is not None:
                    self._last_oi_mode = mode
                self._note_link_ok()
                return val
            except Exception as e:
                last = e
                time.sleep(0.05)
        self._note_link_failure('get_sensors()', str(last))
        return None

    def _dock_cb(self, msg: Bool):
        """Start the final dock approach.

        ‼️ 2026-07-28 — DOCKING CANNOT WORK UNTIL THE BASE IS REPAIRED. Measured
        with the robot parked 1 m away, square on, in Full mode, homing driven
        directly (no Nav2 in the way): Red Buoy (168) seen 100 times, Force
        Field (161) 69 times, **Green Buoy (164) ZERO times**, and never both
        beams together (172). The station emits only one of its two beams.

        A dock needs both: the robot parks where the red and green cones
        overlap, so with one beam missing there is no centre to find. This is
        why the stock OI 143 seek also failed back on 2026-07-22 (13 cm and
        21 cm of travel in 120 s), and why the homing below oscillates — it is
        hunting a signal that is not being transmitted.

        Ruled out by measurement, so do not re-investigate: the map/staging
        pose, the robot's own IR receivers (they read Red cleanly),
        `ir_swap_buoys`, Safe vs Full mode, and serial errors. The fix is
        hardware — inspect the base's emitter with a phone camera (IR shows up
        white/purple on the sensor), clean its window, or replace the base.

        Two routes, chosen by `use_ir_homing`:

        * **IR homing (default)** — we steer, in Full mode, from the dock's own
          IR beams via home_robot/ir_homing.py. This exists because the stock
          seek below does not work on this robot.
        * **OI 143 (legacy)** — hand over to the Create 2's built-in seek.
          Measured 2026-07-22/26: the write succeeds and the OI raises no
          error, but the robot crawls (13cm/120s, 21cm/120s once
          _motor_control stopped fighting it) and never reaches the charger.
          Worse, it steers *out* of the approach corridor: omni went 172
          (Red+Green) to 161 (force field only) with both directional
          receivers at 0, and it never re-acquired. Ruled out along the way:
          an AttributeError crash (pycreate2 has no seek_dock()), _check_idle,
          _keep_awake, _motor_control and the soft e-stop. 143 is a Create 2
          command and this is a consumer Roomba 879, which may simply not
          honour it properly.

        Either way our housekeeping timers would abort the approach within
        seconds, so they are suppressed via _docking until it ends:
          - _check_idle would send drive_direct(0,0) after idle_timeout
          - _keep_awake would re-run _connect_oi(); on the 143 path Full/Safe
            takes back control and cancels the built-in behaviour outright
        """
        if not msg.data:
            return

        if self.use_ir_homing:
            # We steer, so we need Full mode — the opposite of the 143 path,
            # which drops to Passive and hands the wheels to the robot.
            self._connect_oi()
            self._homing = IrHoming(
                forward_mm_s=self.get_parameter('ir_forward_mm_s').value,
                search_timeout_s=self.get_parameter('ir_search_timeout_s').value,
                swap_buoys=self.get_parameter('ir_swap_buoys').value,
            )
            self._homing.reset()
            self._homing_last_status = None
            self._dock_last_odom = None
            self._docking = True
            self._dock_started = time.monotonic()
            self._idle = True                # _motor_control must not fight us
            self._target_left = self._target_right = 0.0
            self._cur_left = self._cur_right = 0.0
            self.get_logger().info('Docking by IR homing — housekeeping suspended')
            self._publish_dock_status('homing')
            return

        try:
            self.bot.SCI.write(143)          # OI opcode 143 = Seek Dock
        except Exception as e:
            self.get_logger().error(f'Seek Dock command failed: {e!r}')
            self._publish_dock_status('lost')
            return
        self._docking = True
        self._dock_started = time.monotonic()
        self._idle = True                    # no motor commands are ours now
        self._target_left = self._target_right = 0.0
        self._cur_left = self._cur_right = 0.0
        self.get_logger().info('Seeking dock (OI 143) — housekeeping suspended')
        self._publish_dock_status('homing')

    def _publish_dock_status(self, status: str):
        """Announce where the dock approach has got to.

        Terminal values are `seated`, `lost` (no beam to follow) and `timeout`;
        `homing` and `searching` are progress. A caller should wait for a
        terminal one rather than assuming the approach worked.
        """
        self.dock_status_pub.publish(String(data=status))

    def _dock_seated(self) -> bool:
        """True when the robot looks parked on the base without charging.

        Two conditions together, because either alone is noisy: the odometry
        pose has not moved for `dock_settle_s` (the Create 2's own seek stops
        driving once it seats), AND an IR receiver still sees the dock. Dock
        beams are 161/164/165/168/169/172/173 in the OI's Infrared Character
        packets; a virtual wall would read 162.
        """
        pose = (round(self._x, 3), round(self._y, 3))
        now = time.monotonic()
        if pose != self._dock_last_pose:
            self._dock_last_pose = pose
            self._dock_still_since = now
            return False
        if (now - self._dock_still_since) < self.dock_settle_s:
            return False
        try:
            vals = []
            for pkt in (17, 52, 53):
                self.bot.SCI.ser.reset_input_buffer()
                self.bot.SCI.ser.write(bytes([142, pkt]))
                time.sleep(0.05)
                b = self.bot.SCI.ser.read(1)
                vals.append(b[0] if b else 0)
        except Exception:                                  # noqa: BLE001
            return False
        return any(v in (161, 164, 165, 168, 169, 172, 173) for v in vals)

    def _check_docking(self):
        """Leave the docking state once the charger engages, or give up.

        Without the timeout a failed dock attempt would leave keep-awake
        suppressed forever and the Roomba would eventually fall asleep with
        no way back short of a restart."""
        if not self._docking:
            return
        if self._homing is not None:
            # IR homing owns this approach and decides seated/lost from the
            # beams itself, on a stricter test than _dock_seated below. Only
            # the timeout at the bottom still applies, as a backstop.
            if (time.monotonic() - self._dock_started) > self.dock_timeout:
                self._drive(0, 0)
                self._docking = False
                self._homing = None
                self.get_logger().warn(
                    f'IR homing gave up after {self.dock_timeout:.0f}s '
                    '— housekeeping resumed')
                self._publish_dock_status('timeout')
                self._connect_oi()
            return
        sensors = self._safe_get_sensors()
        if sensors is not None and getattr(sensors, 'charger_state', 0):
            self._docking = False
            self.get_logger().info('Docked — charging, housekeeping resumed')
            self._publish_dock_status('seated')
            self._connect_oi()
            return
        # ‼️ On THIS robot the battery pack has been removed (2026-07-26) — it
        # runs off a power bank — so charger_state can never become non-zero
        # and the branch above is dead. Docking therefore always ran the full
        # timeout and was logged as a failure even when the robot parked
        # perfectly on the base; _connect_oi() then drove it back out. That is
        # the whole of the long-standing "dock never reaches the charger" bug.
        # With no pack, "docked" has to be judged some other way: the Create 2
        # stops driving once seated, so treat a stationary robot that is still
        # seeing the dock's IR beams as parked.
        if self.dock_without_battery and self._dock_seated():
            self._docking = False
            self.get_logger().info('Docked — seated on base (no battery pack '
                                   'to charge, judged by IR + no motion)')
            self._publish_dock_status('seated')
            self._connect_oi()
            return
        if (time.monotonic() - self._dock_started) > self.dock_timeout:
            self._docking = False
            self.get_logger().warn(
                f'Dock seek gave up after {self.dock_timeout:.0f}s '
                '(not charging) — housekeeping resumed')
            self._publish_dock_status('timeout')
            self._connect_oi()

    # Max plausible per-cycle change at 20Hz (well above the Create 2's
    # ~0.5 m/s top speed) — anything beyond this is a corrupted/misaligned
    # serial read, not real encoder data.
    MAX_DELTA_D = 0.05   # m
    MAX_DELTA_A = math.radians(30)  # rad

    # Packets 19/20 (Distance/Angle) read as 0 on this unit's firmware, so
    # odometry is computed from raw left/right encoder counts (43/44)
    # instead, scaled by the mm_per_tick parameter.

    def _safe_get_ir(self):
        """Read the three IR receivers (17 omni, 52 left, 53 right) at once.

        One Query List, not three individual reads: the homing loop runs at
        10Hz and three round trips cost ~150ms, which would starve the 20Hz
        odometry timer sharing this port. Returns (omni, left, right) or None.
        """
        try:
            self.bot.SCI.ser.reset_input_buffer()
            self.bot.SCI.write(149, (3, 17, 52, 53))
            data = self.bot.SCI.read(3)
            if len(data) != 3:
                return None
            self._last_ok_time = time.monotonic()
            return (data[0], data[1], data[2])
        except Exception:                                  # noqa: BLE001
            return None

    def _ir_homing_step(self):
        """Drive the final dock approach ourselves, 10Hz closed loop.

        Replaces OI 143, which on this 879 drove out of the approach corridor
        and stayed there (measured 2026-07-26). All the judgement lives in
        home_robot/ir_homing.py so it is testable without the robot; this
        method only does I/O.
        """
        if not self._docking or self._homing is None:
            return
        ir = self._safe_get_ir()
        if ir is None:
            return                      # a dropped read is not a decision
        omni, left, right = ir

        # "moved" comes from the odometry the 20Hz timer already maintains,
        # rather than a second encoder read competing for the same port.
        pose = (round(self._x, 3), round(self._y, 3))
        moved = self._dock_last_odom is None or pose != self._dock_last_odom
        self._dock_last_odom = pose

        l_mm, r_mm, status = self._homing.step(omni, left, right, moved,
                                               time.monotonic())

        if status == ir_homing.SEATED:
            self._drive(0, 0)
            self._docking = False
            self._homing = None
            self.get_logger().info(
                'Docked — driven onto the base by IR homing (stalled while '
                'centred on both buoys)')
            self._publish_dock_status('seated')
            self._connect_oi()
            return
        if status == ir_homing.LOST:
            self._drive(0, 0)
            self._docking = False
            self._homing = None
            self.get_logger().warn(
                'IR homing gave up — no dock beam found. Is the base powered?')
            self._publish_dock_status('lost')
            self._connect_oi()
            return

        if status != self._homing_last_status:
            self.get_logger().info(f'IR homing: {status} '
                                   f'(omni={omni} L={left} R={right})')
            self._homing_last_status = status
            self._publish_dock_status(status)
        self._drive(round(r_mm), round(l_mm))

    def _safe_get_encoders(self):
        """Read packets 43/44 (encoder counts) + 7 (bumps/wheel drops), με retry.

        Packet 7 rides along in the SAME Query List as the encoders: one extra
        byte on a round-trip we already make every 50 ms, so the bumper costs
        nothing. A separate poll would add a second serial transaction at 20 Hz
        and starve odometry on this shared port.
        """
        last = None
        for _ in range(3):
            try:
                self.bot.SCI.ser.reset_input_buffer()
                # encoders + bumps/wheel-drops + the four cliff sensors. The
                # cliffs are here rather than in a poll of their own for the
                # same reason packet 7 is: one extra byte each on a round trip
                # we already make every 50 ms, versus a second serial
                # transaction at 20 Hz that would starve odometry.
                self.bot.SCI.write(149, (7, 43, 44, 7, 9, 10, 11, 12))
                # No fixed post-write sleep needed: _RawTermiosSerial.read()
                # blocks (up to its timeout) until all 9 bytes have arrived, so
                # it no longer races the FTDI latency timer.
                data = self.bot.SCI.read(9)
                if len(data) != 9:
                    raise Exception(f'Encoder data not 9 bytes long, it is: {len(data)}')
                left, right, bumps, cl, cfl, cfr, cr = struct.unpack('>hhBBBBB', data)
                # Kept per sensor, not just OR'd into one flag: which of the
                # four fired is what says WHERE the drop is, and that is what
                # the costmap needs to mark. See _publish_cliffs.
                self._cliff_bits = (bool(cl), bool(cfl), bool(cfr), bool(cr))
                # bit0 = bump right, bit1 = bump left, bits 2/3 = wheel drops.
                # In Full mode the Roomba no longer acts on the drops itself, so
                # they are read here instead of being masked off.
                self._bump = bool(bumps & 0x03)
                self._wheel_drop = bool(bumps & 0x0C)
                self._cliff = bool(cl or cfl or cfr or cr)
                if self._bump:
                    self._last_bump_time = time.monotonic()
                if self._cliff:
                    self._last_cliff_time = time.monotonic()
                result = (left, right)
                self._last_ok_time = time.monotonic()
                self._note_link_ok()
                return result
            except Exception as e:
                last = e
                time.sleep(0.05)
        self._note_link_failure('get_encoders()', str(last))
        return None

    # Height the cliff points are published at, in base_link metres. NOT zero:
    # the costmap's obstacle sources filter on height, and a point on the floor
    # plane sits right on the boundary where rounding decides whether it counts.
    # 0.10 m is comfortably inside every layer's band.
    CLIFF_MARK_Z = 0.10

    def _publish_cliffs(self):
        """The four cliff sensors, as points where the floor stops being floor.

        Nothing is published while no sensor is triggered: an empty cloud every
        200 ms would be pure noise on a topic whose only message is "there is a
        drop HERE".
        """
        if not any(self._cliff_bits):
            return
        bearings = self.get_parameter('cliff_bearings_deg').value
        radius = float(self.get_parameter('cliff_radius').value)

        pts = []
        for fired, bearing_deg in zip(self._cliff_bits, bearings):
            if not fired:
                continue
            a = math.radians(float(bearing_deg))
            pts.append((radius * math.cos(a), radius * math.sin(a),
                        self.CLIFF_MARK_Z))
        if not pts:
            return

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        # base_link, not odom: these are positions on the robot itself, and the
        # costmap transforms them. Publishing in a world frame would need the
        # pose to be current, which during a cliff stop it may not be.
        msg.header.frame_id = 'base_link'
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.is_dense = True
        msg.data = b''.join(struct.pack('<fff', *p) for p in pts)
        self._cliff_pub.publish(msg)

    def _publish_odom(self):
        enc = self._safe_get_encoders()
        if enc is None:
            return
        left, right = enc

        now_t = time.monotonic()
        if self._prev_left_enc is None:
            self._prev_left_enc = left
            self._prev_right_enc = right
            self._prev_odom_time = now_t
            return
        dt = now_t - self._prev_odom_time
        self._prev_odom_time = now_t

        d_left = left - self._prev_left_enc
        d_right = right - self._prev_right_enc
        # 16-bit signed counter wraparound
        if d_left > 32768:
            d_left -= 65536
        elif d_left < -32768:
            d_left += 65536
        if d_right > 32768:
            d_right -= 65536
        elif d_right < -32768:
            d_right += 65536

        self._prev_left_enc = left
        self._prev_right_enc = right

        delta_left_m  = d_left  * self.mm_per_tick / 1000.0
        delta_right_m = d_right * self.mm_per_tick / 1000.0

        delta_d = (delta_left_m + delta_right_m) / 2.0
        delta_a = (delta_right_m - delta_left_m) / self.wheel_base

        if abs(delta_d) > self.MAX_DELTA_D or abs(delta_a) > self.MAX_DELTA_A:
            self.get_logger().warn(
                f'Discarding implausible odom delta: d_left={d_left} d_right={d_right} ticks '
                '(corrupted sensor read?)',
                throttle_duration_sec=5.0,
            )
            delta_d = 0.0
            delta_a = 0.0

        self._th += delta_a
        self._x  += delta_d * math.cos(self._th)
        self._y  += delta_d * math.sin(self._th)

        import tf_transformations
        q = tf_transformations.quaternion_from_euler(0, 0, self._th)
        now = self.get_clock().now().to_msg()

        # odom -> base_link TF is published by ekf_node (robot_localization),
        # which fuses this node's wheel velocity with the IMU's absolute yaw
        # — do NOT also broadcast it here, that would fight the EKF's TF
        # (see config/ekf.yaml for why: wheel-only yaw is what produced the
        # skewed/doubled walls after turns during SLAM mapping).
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        # Only twist.linear.x is actually consumed (ekf.yaml's odom0_config
        # fuses just vx) — pose and angular velocity are still published for
        # rqt/debugging but are not fed into the EKF.
        if dt > 0:
            odom.twist.twist.linear.x = delta_d / dt
            odom.twist.twist.angular.z = delta_a / dt
        self.odom_pub.publish(odom)

    def _publish_status(self):
        """In-memory telemetry for the web dashboard — no serial I/O.

        Deliberately excludes battery: this chassis runs off a powerbank and the
        OI's charge fields are garbage, so surfacing them on a dashboard would
        just be a lie in a nicer font.
        """
        now = time.monotonic()
        self.status_pub.publish(String(data=json.dumps({
            'oi_mode':      self._last_oi_mode,   # packet 35: 0 off,1 passive,2 safe,3 full
            'oi_mode_want': self.oi_mode,
            'bump':         self._bump,
            'cliff':        self._cliff,
            'wheel_drop':   self._wheel_drop,
            'estop':        self._estop,
            'docking':      self._docking,
            'idle':         self._idle,
            'stopped':      self._stopped,
            # How stale the last good serial read is. > ~3 s means the base has
            # gone to sleep — the single most useful number here, because a
            # sleeping Roomba fakes navigation bugs (see the nav artifacts note).
            'link_age_s':   round(now - self._last_ok_time, 1)
                            if self._last_ok_time else None,
            'left_mm_s':    round(self._cur_left, 1),
            'right_mm_s':   round(self._cur_right, 1),
        })))

    def _publish_battery(self):
        sensors = self._safe_get_sensors()
        if sensors is None:
            return

        charge    = getattr(sensors, 'battery_charge', None)    # mAh
        capacity  = getattr(sensors, 'battery_capacity', None)  # mAh
        voltage   = getattr(sensors, 'voltage', None)           # mV
        current   = getattr(sensors, 'current', None)           # mA
        charging  = getattr(sensors, 'charger_state', 0)

        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if voltage is not None:
            msg.voltage = voltage / 1000.0
        if current is not None:
            msg.current = current / 1000.0
        if charge is not None:
            msg.charge = charge / 1000.0
        if capacity is not None and capacity > 0:
            msg.capacity = capacity / 1000.0
            msg.design_capacity = msg.capacity
            ratio = charge / capacity if charge is not None else float('nan')
            msg.percentage = ratio
            self.charge_ratio_pub.publish(Float32(data=float(ratio)))
        else:
            msg.percentage = float('nan')

        msg.power_supply_status = (
            BatteryState.POWER_SUPPLY_STATUS_CHARGING
            if charging else BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        )
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_NIMH
        msg.present = True

        self.battery_pub.publish(msg)

    def destroy_node(self):
        self._drive(0, 0)
        self.bot.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = RoombaDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
