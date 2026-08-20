#!/usr/bin/env python3
"""Waveshare RoArm-M3 → ROS2 driver (JSON-over-serial, ESP32).

Confirmed JSON protocol (Waveshare RoArm-M3-S wiki):
  T:100 {"T":100}
      Move all joints to the init pose (blocking on the arm side).
  T:102 {"T":102,"base":.,"shoulder":.,"elbow":.,"wrist":.,"roll":.,"hand":.,
         "spd":S,"acc":A}
      All-joint position control, radians. spd in steps/s (4096 steps/rev,
      0 = firmware default), acc 0-254 in 100 steps/s^2 (0 = instant).
  T:104 {"T":104,"x":..,"y":..,"z":..,"t":..,"r":..,"g":..,"spd":..}
      Cartesian XYZ + wrist/roll/gripper, inverse kinematics (blocking).
  T:105 {"T":105}
      Request feedback: end-effector position + all joint angles + load.
  T:106 {"T":106,"cmd":<rad>,"spd":S,"acc":A}
      Gripper/EoAT angle control, radians. Decreasing "cmd" opens the gripper.
  T:210 {"T":210,"cmd":0|1}
      Torque switch ON/OFF (exact param semantics unconfirmed — use via
      arm/raw_cmd if/when needed, e.g. to let the arm go limp by hand).

Joint angle limits (radians, from the Waveshare wiki):
  base:     -3.14 .. 3.14
  shoulder: -1.57 .. 1.57
  elbow:     0.0  .. 3.14   (init 1.570796)
  wrist:    -1.57 .. 1.57
  roll:     -3.14 .. 3.14
  hand:      1.08 .. 3.14   (gripper, init 3.141593)

Confirmed on the real arm 2026-07-01: T:105 feedback uses the short keys
b/s/e/t/r/g (see FEEDBACK_JOINT_KEYS). Still deliberately unused:
single-joint T:101, whose "joint" index base (0- vs 1-based) is ambiguous
in the docs — T:102 (all joints) and T:106 (gripper) are unambiguous and
used instead; T:210 semantics remain unconfirmed (raw_cmd only).
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32
import serial


JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']

# The RoArm-M3 T:102 command takes the full joint names above, but its T:105
# feedback reports the same joints under abbreviated single-letter keys
# (confirmed against the real hardware 2026-07-01: feedback carries
# x/y/z/tit/b/s/e/t/r/g/... ). Map joint name -> feedback key so
# _parse_feedback() can read the current pose.
FEEDBACK_JOINT_KEYS = {
    'base': 'b', 'shoulder': 's', 'elbow': 'e',
    'wrist': 't', 'roll': 'r', 'hand': 'g',
}

# MECHANICAL limits, from the Waveshare wiki — what the servos can physically
# reach. These are NOT the limits to drive against: the arm is mounted on a
# moving robot next to the LiDAR, so most of that range points it into the
# chassis, the camera or the scan plane. The usable envelope is the `limit_*`
# parameters below, which start here and are meant to be tightened per joint.
MECH_LIMITS = {
    'base':     (-3.14, 3.14),
    'shoulder': (-1.57, 1.57),
    'elbow':    (0.0, 3.14),
    'wrist':    (-1.57, 1.57),
    'roll':     (-3.14, 3.14),
    'hand':     (1.08, 3.14),
}
# Kept for anything still importing the old name (arm_joy reads its own params).
JOINT_LIMITS = MECH_LIMITS


class ArmDriver(Node):
    def __init__(self):
        super().__init__('arm_driver')

        self.declare_parameter('port', '/dev/arm')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('joint_names', JOINT_NAMES)
        # spd/acc as used in the Waveshare T:102/T:106 examples.
        self.declare_parameter('speed', 0)     # steps/s, 0 = firmware default
        self.declare_parameter('accel', 10)    # 0-254, in 100 steps/s^2
        # Per-joint usable envelope, [lo, hi] radians. Read on EVERY clamp, not
        # cached at startup, so limits can be tuned against the real arm live:
        #     ros2 param set /arm_driver limit_shoulder "[-0.6, 1.2]"
        # A limit is never allowed outside MECH_LIMITS — a typo that widens the
        # range past the servo's travel would grind the joint against its stop.
        for _j, (_lo, _hi) in MECH_LIMITS.items():
            self.declare_parameter(f'limit_{_j}', [float(_lo), float(_hi)])

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.speed = self.get_parameter('speed').value
        # NOT cached like speed above: the dashboard's arm speed slider
        # retunes this live (`ros2 param set /arm_driver accel N`) the same
        # way limits() re-reads limit_* on every command, and a value cached
        # once at startup would silently ignore that until a restart.

        self.declare_parameter('feedback_timeout', 3.0)

        # Seconds to wait out the ESP32 boot banner after a reset — see
        # _boot_firmware, which every port open now goes through.
        self.declare_parameter('boot_settle', 2.0)
        self._boot_settle = self.get_parameter('boot_settle').value

        self._port, self._baud = port, baud
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self._boot_firmware(self.ser)
        self._lock = threading.Lock()
        # Backoff for _reopen(); see _read_serial for why this exists at all.
        self._reopen_at = 0.0
        self._reopen_delay = 1.0
        # Watchdog for the ESP32 going silent: it stops answering T:105 while
        # the USB link stays electrically alive (no OSError/SerialException
        # ever fires — ser.in_waiting just reads 0 forever, and the kernel
        # logs nothing). Track last-good feedback so we can force a reopen.
        #
        # 2026-08-19: the cause was found and it was ours — opening the port
        # was resetting the board into its serial bootloader via DTR/RTS. See
        # _boot_firmware, which every open now goes through. The 2026-08-10
        # note here used to say only a physical power-cycle recovered it and
        # blamed "a burst of commands"; both were wrong. This watchdog is now
        # a real recovery path rather than a reopen loop that re-armed the
        # same trap, so it should very rarely have anything to do.
        self._last_feedback_at = time.monotonic()
        self.get_logger().info(f'Arm connected on {port} @ {baud}')

        # Last known joint positions (radians), populated from T:105
        # feedback. Required before sending T:102, since that command sets
        # ALL joints at once — without a real current pose we'd snap
        # un-commanded joints to a guessed value.
        self._current_joints = None
        self._feedback_format_warned = False
        self._limit_warned = set()

        # ── Subscriptions ─────────────────────────────────────────
        self.joint_cmd_sub = self.create_subscription(
            JointState, 'arm/joint_cmd', self._joint_cmd_cb, 10)
        self.gripper_sub = self.create_subscription(
            Float32, 'arm/gripper_cmd', self._gripper_cb, 10)
        # Raw JSON passthrough — for T:100 (init), T:104 (cartesian), T:210
        # (torque), etc. without needing dedicated topics for every command.
        self.raw_cmd_sub = self.create_subscription(
            String, 'arm/raw_cmd', self._raw_cmd_cb, 10)

        # ── Publishers ────────────────────────────────────────────
        self.joint_state_pub = self.create_publisher(JointState, 'arm/joint_states', 10)
        self.raw_feedback_pub = self.create_publisher(String, 'arm/raw_feedback', 10)

        self.create_timer(0.1, self._request_feedback)  # T:105, 10 Hz
        self.create_timer(0.05, self._read_serial)      # drain replies, 20 Hz
        self.create_timer(1.0, self._check_feedback_watchdog)

    # ------------------------------------------------------------------
    # Commands → arm
    # ------------------------------------------------------------------
    def _send_json(self, cmd: dict):
        # _read_serial is the only place that calls _reopen(), on its own
        # 20 Hz timer — a disconnect leaves self.ser None for up to that
        # tick. _request_feedback fires at 10 Hz and hit this window for
        # real 2026-08-14: 'NoneType' object has no attribute 'write' out of
        # a timer callback took rclpy's spin down with it, same failure mode
        # _read_serial was already hardened against (see its docstring).
        if self.ser is None:
            return
        line = (json.dumps(cmd) + '\n').encode('utf-8')
        try:
            with self._lock:
                self.ser.write(line)
        except (OSError, serial.SerialException) as e:
            self.get_logger().warn(f'Arm write failed: {e!r}', throttle_duration_sec=5.0)
            self._mark_disconnected()

    def _joint_cmd_cb(self, msg: JointState):
        if self._current_joints is None:
            self.get_logger().warn(
                'arm/joint_cmd ignored: no feedback received yet, current pose unknown')
            return

        target = dict(self._current_joints)
        for name, pos in zip(msg.name, msg.position):
            if name not in JOINT_LIMITS:
                self.get_logger().warn(f'Unknown joint "{name}" in arm/joint_cmd, ignored')
                continue
            target[name] = self._clamp(name, pos)

        cmd = {'T': 102, **{name: target[name] for name in self.joint_names},
               'spd': self.speed, 'acc': self.get_parameter('accel').value}
        self._send_json(cmd)

    def limits(self, name):
        """The live [lo, hi] for a joint, never wider than the servo's travel.

        Read fresh on every call so `ros2 param set` takes effect on the next
        command — tightening a limit is something you do WITH the arm in front
        of you, and a restart mid-way loses the pose you were measuring.
        """
        lo_m, hi_m = MECH_LIMITS[name]
        try:
            v = list(self.get_parameter(f'limit_{name}').value)
            lo, hi = float(v[0]), float(v[1])
        except Exception:
            return lo_m, hi_m
        if lo > hi:
            lo, hi = hi, lo
        clipped = (max(lo, lo_m), min(hi, hi_m))
        if (lo, hi) != clipped and name not in self._limit_warned:
            self._limit_warned.add(name)
            self.get_logger().warn(
                f'limit_{name}={[lo, hi]} reaches past the servo travel '
                f'{[lo_m, hi_m]} — using the mechanical limit instead')
        return clipped

    def _clamp(self, name, value):
        lo, hi = self.limits(name)
        return max(lo, min(hi, value))

    def _gripper_cb(self, msg: Float32):
        pos = self._clamp('hand', msg.data)
        self._send_json({'T': 106, 'cmd': pos, 'spd': self.speed,
                         'acc': self.get_parameter('accel').value})
        if self._current_joints is not None:
            self._current_joints['hand'] = pos

    def _raw_cmd_cb(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Invalid raw_cmd JSON: {e}')
            return
        self._send_json(cmd)

    # ------------------------------------------------------------------
    # Feedback ← arm
    # ------------------------------------------------------------------
    def _request_feedback(self):
        self._send_json({'T': 105})

    def _mark_disconnected(self):
        """Drop the port and schedule a reopen."""
        try:
            self.ser.close()
        except Exception:                                  # noqa: BLE001
            pass
        self.ser = None
        self._reopen_at = time.monotonic() + self._reopen_delay
        self._reopen_delay = min(self._reopen_delay * 2, 30.0)

    def _boot_firmware(self, ser):
        """Reset the ESP32 into its FIRMWARE, not its serial bootloader.

        ‼️ This is why the arm kept "hanging". The RoArm-M3's CP2102N has its
        DTR/RTS lines wired to the ESP32's auto-reset circuit (RTS->EN,
        DTR->IO0), and pyserial asserts BOTH on open. Every port open is
        therefore a reset with the boot-select line in play, and landing IO0
        low at the moment EN releases boots the serial bootloader — which
        answers nothing, forever, while the USB link stays perfectly healthy
        and the kernel logs not one error. `_check_feedback_watchdog` then
        reopened every ~3 s, each reopen another chance to re-arm the same
        trap, so the driver could hold the board in the bootloader
        indefinitely. That is the "only a physical power cycle recovers it"
        note above: power-cycling worked because it reset the ESP32 while
        nothing was driving those lines. Cycling USB power never did, because
        re-enumeration is followed by another open.

        Verified on the real arm 2026-08-19: a silent board (0 bytes to
        {"T":105} on a raw pyserial probe) came straight back with this
        sequence, printing "All bus servos status checked. / Server Starts."
        and then streaming T:1051 again.
        """
        try:
            ser.dtr = False          # IO0 HIGH -> run firmware, not bootloader
            ser.rts = True           # EN LOW   -> hold in reset
            time.sleep(0.15)
            ser.rts = False          # EN HIGH  -> boot
            # The banner + "All bus servos status checked" takes ~1.5 s; talking
            # over it just loses the first commands.
            time.sleep(self._boot_settle)
            ser.reset_input_buffer()
        except (OSError, serial.SerialException) as e:
            self.get_logger().warn(f'Arm reset lines unavailable: {e!r}')

    def _reopen(self):
        if time.monotonic() < self._reopen_at:
            return
        try:
            self.ser = serial.Serial(self._port, self._baud, timeout=0.1)
            self._boot_firmware(self.ser)
            self._reopen_delay = 1.0
            self._current_joints = None      # pose is unknown again after a drop
            self._last_feedback_at = time.monotonic()  # grace period for the watchdog
            self.get_logger().info(f'Arm reconnected on {self._port}')
        except (OSError, serial.SerialException) as e:
            self._reopen_at = time.monotonic() + self._reopen_delay
            self._reopen_delay = min(self._reopen_delay * 2, 30.0)
            self.get_logger().warn(
                f'Arm reopen failed ({e!r}); retrying in {self._reopen_delay:.0f}s',
                throttle_duration_sec=10.0)

    def _read_serial(self):
        """Drain replies, and survive the arm going away.

        ‼️ This used to catch only JSONDecodeError. `self.ser.in_waiting` raises
        SerialException the moment the adapter disappears — and these
        peripherals hang off an external USB hub that has glitched before — so
        an unplug/hub reset threw out of a TIMER callback, which takes rclpy's
        spin down with it: the whole arm driver died rather than waiting for the
        arm to come back. imu_node has reconnected like this all along.
        """
        if self.ser is None:
            self._reopen()
            return
        while True:
            try:
                with self._lock:
                    if self.ser.in_waiting == 0:
                        return
                    line = self.ser.readline()
            except (OSError, serial.SerialException) as e:
                self.get_logger().warn(f'Arm serial error: {e!r} — reopening')
                self._mark_disconnected()
                return

            if not line:
                return

            self.raw_feedback_pub.publish(String(data=line.decode(errors='replace').strip()))

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._parse_feedback(data)

    def _check_feedback_watchdog(self):
        """Force a reopen if the port is open but the ESP32 has gone quiet.

        This is the software equivalent of the physical board reset: a real
        USB power-cycle (confirmed 2026-08-10 via journalctl) is what
        actually recovered the arm, because it forced a fresh
        serial.Serial() open. Reopening the port here doesn't power-cycle
        the ESP32 itself, but it re-syncs the driver's read/write state and
        has been enough in similar imu_node hangs — worth trying before
        asking for a physical reset.
        """
        if self.ser is None:
            return
        timeout = self.get_parameter('feedback_timeout').value
        if time.monotonic() - self._last_feedback_at > timeout:
            self.get_logger().warn(
                f'Arm: no feedback for >{timeout:.0f}s, port open but ESP32 silent — '
                f'forcing reopen', throttle_duration_sec=timeout)
            self._mark_disconnected()

    def _parse_feedback(self, data: dict):
        # Feedback keys are abbreviated (see FEEDBACK_JOINT_KEYS); fall back to
        # the full name so a custom joint_names param still works.
        def fb_key(name):
            return FEEDBACK_JOINT_KEYS.get(name, name)

        if not all(fb_key(name) in data for name in self.joint_names):
            if not self._feedback_format_warned:
                self.get_logger().warn(
                    f'CMD_SERVO_RAD_FEEDBACK (T:105): unrecognized keys {list(data.keys())} — '
                    f'update _parse_feedback()/FEEDBACK_JOINT_KEYS to match the real field names'
                )
                self._feedback_format_warned = True
            return

        self._current_joints = {name: float(data[fb_key(name)]) for name in self.joint_names}
        self._last_feedback_at = time.monotonic()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [self._current_joints[n] for n in self.joint_names]
        self.joint_state_pub.publish(msg)

    def destroy_node(self):
        try:
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ArmDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
