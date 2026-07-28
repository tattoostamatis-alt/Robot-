#!/usr/bin/env python3
"""Roomba 692 → ROS2 driver via Create 2 Open Interface."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, String
import pycreate2
from home_robot import ir_homing
from home_robot.ir_homing import IrHoming
import fcntl
import math
import os
import select
import struct
import termios
import threading
import time


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
        # OI mode we drive in. 'safe' (default, 2026-07-28 at the user's
        # request) keeps the Roomba's own protections live: it aborts motion by
        # itself on a cliff, a lifted wheel, or the charger. 'full' disables all
        # of that and lets us drive off a step.
        # ‼️ The trade-off is real: those same protections drop the robot back to
        # PASSIVE, where it ignores drive commands until _connect_oi() runs
        # again. Docking is the known case — touching the charging contacts
        # ends Safe mode.
        self.declare_parameter('oi_mode', 'safe')
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
        self._last_brc_pulse = 0.0
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

        self.create_timer(0.05, self._publish_odom)     # 20 Hz
        self.create_timer(0.05, self._motor_control)     # 20 Hz ramped output + watchdog
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

    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'right_trim':
                self.right_trim = p.value
            elif p.name == 'right_trim_reverse':
                self.right_trim_reverse = p.value
            elif p.name == 'max_accel':
                self.max_accel = p.value
            elif p.name == 'reverse_speed_scale':
                self.reverse_speed_scale = p.value
        return SetParametersResult(successful=True)

    def _estop_cb(self, msg: Bool):
        if msg.data == self._estop:
            return
        self._estop = msg.data
        if self._estop:
            # hard stop immediately — no ramp, this is an emergency
            self._target_left = self._target_right = 0.0
            self._cur_left = self._cur_right = 0.0
            try:
                self.bot.drive_direct(0, 0)
            except Exception as e:  # never let a serial hiccup swallow the e-stop
                self.get_logger().error(f'e-stop drive_direct(0,0) failed: {e!r}')
            self._stopped = True
            self.get_logger().warn('EMERGENCY STOP engaged — ignoring cmd_vel until reset')
        else:
            # release: drop any stale target so we don't lurch on the next cmd
            self._target_left = self._target_right = 0.0
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
        angular = msg.angular.z         # rad/s

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
            self.bot.drive_direct(0, 0)
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

        step = self.max_accel * dt
        self._cur_left  += max(-step, min(step, target_left  - self._cur_left))
        self._cur_right += max(-step, min(step, target_right - self._cur_right))

        self.bot.drive_direct(round(self._cur_right), round(self._cur_left))

    def _check_idle(self):
        if self._docking:      # drive_direct(0,0) would abort the dock approach
            return
        if not self._idle and (time.monotonic() - self._last_cmd_time) > self.idle_timeout:
            self._idle = True
            self._target_left = self._target_right = 0.0
            self._cur_left = self._cur_right = 0.0
            self.bot.drive_direct(0, 0)
            self._stopped = True
            self.get_logger().info(
                f'Roomba: idle {self.idle_timeout:.0f}s → passive mode (low power)'
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
        """Never let the Roomba sleep. Two jobs:
          1. If there has been no good sensor/encoder read for >2s the robot
             is asleep or dropped out of OI — re-run wake/start/full so it
             recovers on its own (e.g. right after the user presses CLEAN),
             with no driver restart needed.
          2. While healthy, pulse BRC at least once a minute so the OI's
             5-minute inactivity sleep timer never expires."""
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
        if (now - self._last_brc_pulse) > 60.0:
            self._brc_pulse()

    def _safe_get_sensors(self):
        """get_sensors() με retry — το Roomba μερικές φορές επιστρέφει 0 bytes."""
        for attempt in range(3):
            try:
                # Discard any stale/leftover bytes so this read is aligned to
                # the response of the query we're about to send — otherwise a
                # backlog from a delayed previous cycle desyncs every field.
                self.bot.SCI.ser.reset_input_buffer()
                val = self.bot.get_sensors()
                self._last_ok_time = time.monotonic()
                return val
            except Exception as e:
                self.get_logger().warn(f'get_sensors() attempt {attempt}: {e!r}')
                time.sleep(0.05)
        self.get_logger().error('get_sensors() failed after 3 attempts')
        return None

    def _dock_cb(self, msg: Bool):
        """Start the final dock approach.

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
            ir = [self.bot.SCI.ser, ]  # keep the port reference alive
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
                self.bot.drive_direct(0, 0)
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
            self.bot.drive_direct(0, 0)
            self._docking = False
            self._homing = None
            self.get_logger().info(
                'Docked — driven onto the base by IR homing (stalled while '
                'centred on both buoys)')
            self._publish_dock_status('seated')
            self._connect_oi()
            return
        if status == ir_homing.LOST:
            self.bot.drive_direct(0, 0)
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
        self.bot.drive_direct(round(r_mm), round(l_mm))

    def _safe_get_encoders(self):
        """Read packets 43/44 (left/right encoder counts) με retry."""
        for attempt in range(3):
            try:
                self.bot.SCI.ser.reset_input_buffer()
                self.bot.SCI.write(149, (2, 43, 44))  # Query List: 2 packets, IDs 43+44
                # No fixed post-write sleep needed: _RawTermiosSerial.read()
                # blocks (up to its timeout) until all 4 bytes have arrived, so
                # it no longer races the FTDI latency timer.
                data = self.bot.SCI.read(4)
                if len(data) != 4:
                    raise Exception(f'Encoder data not 4 bytes long, it is: {len(data)}')
                result = struct.unpack('>hh', data)
                self._last_ok_time = time.monotonic()
                return result
            except Exception as e:
                self.get_logger().warn(f'get_encoders() attempt {attempt}: {e!r}')
                time.sleep(0.05)
        self.get_logger().error('get_encoders() failed after 3 attempts')
        return None

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
        self.bot.drive_direct(0, 0)
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
