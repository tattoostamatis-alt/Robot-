#!/usr/bin/env python3
"""Roomba 692 → ROS2 driver via Create 2 Open Interface."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32
import pycreate2
import fcntl
import math
import os
import select
import struct
import termios
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
        self.declare_parameter('idle_timeout', 30.0)  # seconds until passive mode
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
        self.declare_parameter('wheel_base', 0.23842)
        self.declare_parameter('mm_per_tick', 0.401635)
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
        self.declare_parameter('right_trim', 1.14)
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
        self.declare_parameter('max_accel', 600.0)  # mm/s^2
        # Reverse felt too fast at the same scale_linear.x as forward —
        # scales the commanded speed down (not a calibration trim like
        # right_trim_reverse above, just a deliberate speed cap) whenever
        # linear.x < 0. 1.0 = same speed as forward.
        self.declare_parameter('reverse_speed_scale', 0.6)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.idle_timeout = self.get_parameter('idle_timeout').value
        self.mm_per_tick = self.get_parameter('mm_per_tick').value
        self.right_trim = self.get_parameter('right_trim').value
        self.right_trim_reverse = self.get_parameter('right_trim_reverse').value
        self.max_accel = self.get_parameter('max_accel').value
        self.reverse_speed_scale = self.get_parameter('reverse_speed_scale').value
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
        self._last_cmd_time = time.monotonic()
        self._target_left = 0.0
        self._target_right = 0.0
        self._cur_left = 0.0
        self._cur_right = 0.0
        self._last_control_time = time.monotonic()

        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        self.dock_sub = self.create_subscription(Bool, 'dock', self._dock_cb, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.battery_pub = self.create_publisher(BatteryState, 'battery/state', 10)
        self.charge_ratio_pub = self.create_publisher(Float32, 'battery/charge_ratio', 10)

        self.create_timer(0.05, self._publish_odom)     # 20 Hz
        self.create_timer(0.05, self._motor_control)     # 20 Hz ramped output + watchdog
        self.create_timer(2.0,  self._publish_battery)  # 0.5 Hz
        self.create_timer(5.0,  self._check_idle)       # idle watchdog
        self.create_timer(2.0,  self._keep_awake)        # auto-reconnect + BRC keep-alive

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

    def _cmd_vel_cb(self, msg: Twist):
        self._last_cmd_time = time.monotonic()
        self._stopped = False
        if self._idle:
            self._idle = False
            self.bot.full()
            self.get_logger().info('Roomba: waking up (full mode)')

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

        target_left, target_right = self._target_left, self._target_right
        if not self._idle and (now - self._last_cmd_time) > 0.25:
            target_left = target_right = 0.0
            self._stopped = True

        step = self.max_accel * dt
        self._cur_left  += max(-step, min(step, target_left  - self._cur_left))
        self._cur_right += max(-step, min(step, target_right - self._cur_right))

        self.bot.drive_direct(round(self._cur_right), round(self._cur_left))

    def _check_idle(self):
        if not self._idle and (time.monotonic() - self._last_cmd_time) > self.idle_timeout:
            self._idle = True
            self._target_left = self._target_right = 0.0
            self._cur_left = self._cur_right = 0.0
            self.bot.drive_direct(0, 0)
            self._stopped = True
            self.get_logger().info(
                f'Roomba: idle {self.idle_timeout:.0f}s → passive mode (low power)'
            )

    def _connect_oi(self):
        """(Re)initialise the Open Interface: wake → Passive → Full. Safe to
        call repeatedly. Called at startup and by _keep_awake to recover
        automatically after the Roomba has gone to sleep (it ignores serial
        while asleep, so a single boot-time init is not enough)."""
        try:
            self.bot.wake()      # pulse BRC low — wakes the robot from sleep
            self.bot.start()     # enter OI (Passive)
            time.sleep(0.3)
            self.bot.full()      # Full mode: the OI never auto-sleeps in Full
            time.sleep(0.2)
            self._last_brc_pulse = time.monotonic()
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
        now = time.monotonic()
        if (now - self._last_ok_time) > 2.0:
            self._connect_oi()
            return
        if (now - self._last_brc_pulse) > 60.0:
            try:
                self.bot.wake()
                self._last_brc_pulse = now
            except Exception as e:
                self.get_logger().warn(f'keep-awake BRC pulse failed: {e!r}')

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
        if msg.data:
            self.bot.seek_dock()
            self.get_logger().info('Seeking dock...')

    # Max plausible per-cycle change at 20Hz (well above the Create 2's
    # ~0.5 m/s top speed) — anything beyond this is a corrupted/misaligned
    # serial read, not real encoder data.
    MAX_DELTA_D = 0.05   # m
    MAX_DELTA_A = math.radians(30)  # rad

    # Packets 19/20 (Distance/Angle) read as 0 on this unit's firmware, so
    # odometry is computed from raw left/right encoder counts (43/44)
    # instead, scaled by the mm_per_tick parameter.

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
