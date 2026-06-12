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

NOT yet confirmed against real hardware — TODO when the arm arrives:
  - Exact field names of the T:105 feedback response. _parse_feedback()
    first tries the same names as T:102 (base/shoulder/.../hand); if the
    real response uses different keys it logs them once (see
    `_feedback_format_warned`) so the parser can be updated.
  - T:210 torque on/off parameter name/values.
  - Single-joint command T:101's "joint" index base (0- vs 1-based) is
    ambiguous in the docs, so it's deliberately unused — T:102 (all
    joints) and T:106 (gripper) are unambiguous and used instead.
"""

import json
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32
import serial


JOINT_NAMES = ['base', 'shoulder', 'elbow', 'wrist', 'roll', 'hand']

JOINT_LIMITS = {
    'base':     (-3.14, 3.14),
    'shoulder': (-1.57, 1.57),
    'elbow':    (0.0, 3.14),
    'wrist':    (-1.57, 1.57),
    'roll':     (-3.14, 3.14),
    'hand':     (1.08, 3.14),
}


def _clamp(name, value):
    lo, hi = JOINT_LIMITS[name]
    return max(lo, min(hi, value))


class ArmDriver(Node):
    def __init__(self):
        super().__init__('arm_driver')

        self.declare_parameter('port', '/dev/arm')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('joint_names', JOINT_NAMES)
        # spd/acc as used in the Waveshare T:102/T:106 examples.
        self.declare_parameter('speed', 0)     # steps/s, 0 = firmware default
        self.declare_parameter('accel', 10)    # 0-254, in 100 steps/s^2

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.speed = self.get_parameter('speed').value
        self.accel = self.get_parameter('accel').value

        self.ser = serial.Serial(port, baud, timeout=0.1)
        self._lock = threading.Lock()
        self.get_logger().info(f'Arm connected on {port} @ {baud}')

        # Last known joint positions (radians), populated from T:105
        # feedback. Required before sending T:102, since that command sets
        # ALL joints at once — without a real current pose we'd snap
        # un-commanded joints to a guessed value.
        self._current_joints = None
        self._feedback_format_warned = False

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

    # ------------------------------------------------------------------
    # Commands → arm
    # ------------------------------------------------------------------
    def _send_json(self, cmd: dict):
        line = (json.dumps(cmd) + '\n').encode('utf-8')
        with self._lock:
            self.ser.write(line)

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
            target[name] = _clamp(name, pos)

        cmd = {'T': 102, **{name: target[name] for name in self.joint_names},
               'spd': self.speed, 'acc': self.accel}
        self._send_json(cmd)

    def _gripper_cb(self, msg: Float32):
        pos = _clamp('hand', msg.data)
        self._send_json({'T': 106, 'cmd': pos, 'spd': self.speed, 'acc': self.accel})
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

    def _read_serial(self):
        while True:
            with self._lock:
                if self.ser.in_waiting == 0:
                    return
                line = self.ser.readline()

            if not line:
                return

            self.raw_feedback_pub.publish(String(data=line.decode(errors='replace').strip()))

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._parse_feedback(data)

    def _parse_feedback(self, data: dict):
        if not all(name in data for name in self.joint_names):
            if not self._feedback_format_warned:
                self.get_logger().warn(
                    f'CMD_SERVO_RAD_FEEDBACK (T:105): unrecognized keys {list(data.keys())} — '
                    f'update _parse_feedback() to match the real field names'
                )
                self._feedback_format_warned = True
            return

        self._current_joints = {name: float(data[name]) for name in self.joint_names}

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
