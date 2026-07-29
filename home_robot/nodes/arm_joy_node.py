#!/usr/bin/env python3
"""PS5 DualSense right stick → RoArm-M3 manual jog.

  right stick ←→   base      (swing the arm left/right)
  right stick ↑↓   shoulder  (raise/lower the arm)
  R1 held          gripper opens
  R2 held          gripper closes

The arm has no velocity interface — arm_driver.py speaks T:102, which is
*position* control. So this node integrates the stick deflection into a target
angle at a fixed rate and republishes that target, i.e. it turns a velocity
command into a stream of positions. The integrator is deliberately open-loop
(seeded once from arm/joint_states, then advanced on its own) rather than
"current feedback + delta": the servos lag the command, so feeding the measured
angle back in would make the arm creep at a fraction of the requested rate and
stall completely against any load.

Targets are clamped to JOINT_LIMITS here and not only in the driver. Clamping
only downstream would let the integrator wind up past the limit while you hold
the stick, and the arm would then ignore the first part of the way back.

‼️ Sign conventions are UNVERIFIED on hardware. joy publishes left = +1 and
up = +1 (REP-103), but which way the RoArm's base and shoulder count positive
is not documented in the Waveshare wiki. If a stick direction moves the arm the
wrong way, negate that joint's scale_* parameter — no code change needed:

    ros2 param set /arm_joy scale_shoulder -0.5

‼️ R1 DOUBLES AS THE DRIVE DEAD-MAN. config/teleop_twist_joy_ps5.yaml sets
enable_button: 5 (R1) with require_enable_button: true, so R1 is held down for
the whole of every drive — which would make the gripper crawl open every time
the robot is driven anywhere. The gripper was asked for on R1/R2, so rather
than move the binding, R1 only counts as "open the gripper" while the LEFT
stick is centred (drive_lockout_axes). Driving and gripping are never wanted in
the same instant, and this way each button keeps the meaning the user expects.
Set drive_lockout to false to disable that interlock.

DualSense layout (hid-playstation, 8 axes / 13 buttons). Re-confirmed against
the hardware 2026-07-29 by logging raw /dev/input/js0 events:
  axes:    0 LX  1 LY  2 L2  3 RX  4 RY  5 R2  6 dpadX  7 dpadY
  buttons: 0 Cross 1 Circle 2 Triangle 3 Square 4 L1 5 R1 6 L2 7 R2
           8 Share 9 Options 10 PS 11 L3 12 R3
Raw js reports right-stick right as +0.70 and up as -0.72; joy_node negates
both, so on /joy left and up are positive (REP-103), as assumed above.

  ros2 run home_robot arm_joy_node.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Bool, Float32


# Mirrors arm_driver.JOINT_LIMITS. Duplicated rather than imported because the
# nodes are installed as standalone scripts, not as an importable module.
JOINT_LIMITS = {
    'base':     (-3.14, 3.14),
    'shoulder': (-1.57, 1.57),
    'hand':     (1.08, 3.14),   # 1.08 = open, 3.14 = closed
}

JOG_JOINTS = ('base', 'shoulder')


def _clamp(name, value):
    lo, hi = JOINT_LIMITS[name]
    return max(lo, min(hi, value))


class ArmJoy(Node):

    def __init__(self):
        super().__init__('arm_joy')

        self.declare_parameter('joy_topic', '/joy')
        # Right stick. Negate the scales (below) to flip a direction.
        self.declare_parameter('axis_base', 3)          # right stick horizontal
        self.declare_parameter('axis_shoulder', 4)      # right stick vertical
        self.declare_parameter('scale_base', 0.60)      # rad/s at full deflection
        self.declare_parameter('scale_shoulder', 0.50)  # rad/s at full deflection
        self.declare_parameter('gripper_open_button', 5)    # R1
        self.declare_parameter('gripper_close_button', 7)   # R2
        self.declare_parameter('scale_gripper', 0.80)   # rad/s while held
        # See the R1 note in the module docstring: ignore the gripper buttons
        # while the drive stick is pushed, so holding R1 to drive doesn't also
        # open the gripper.
        self.declare_parameter('drive_lockout', True)
        self.declare_parameter('drive_lockout_axes', [0, 1])   # left stick
        # Sticks rest near zero but not exactly; below this we treat it as let go.
        self.declare_parameter('deadzone', 0.12)
        self.declare_parameter('rate', 20.0)            # Hz, command stream
        # If /joy goes quiet (controller off, Bluetooth drop) the last message
        # may have had the stick pushed. Without this the integrator would keep
        # running on that stale deflection and the arm would drive itself into
        # its limit. Treat silence as "sticks centred".
        self.declare_parameter('joy_timeout', 0.5)      # seconds

        self.axis_base = self.get_parameter('axis_base').value
        self.axis_shoulder = self.get_parameter('axis_shoulder').value
        self.scale_base = self.get_parameter('scale_base').value
        self.scale_shoulder = self.get_parameter('scale_shoulder').value
        self.btn_open = self.get_parameter('gripper_open_button').value
        self.btn_close = self.get_parameter('gripper_close_button').value
        self.scale_gripper = self.get_parameter('scale_gripper').value
        self.drive_lockout = self.get_parameter('drive_lockout').value
        self.drive_axes = list(self.get_parameter('drive_lockout_axes').value)
        self.deadzone = self.get_parameter('deadzone').value
        self.joy_timeout = self.get_parameter('joy_timeout').value
        rate = self.get_parameter('rate').value

        # Integrator state. None until arm/joint_states tells us where the arm
        # actually is — jogging from a guessed pose would snap it on the first
        # command, since T:102 sets every joint at once.
        self._target = None       # {'base': rad, 'shoulder': rad}
        self._grip = None         # rad
        self._warned_no_feedback = False

        self._axes = (0.0, 0.0)   # base, shoulder — deadzoned
        self._grip_dir = 0        # -1 open, +1 close
        self._last_joy = None     # rclpy Time of the last /joy message
        self._moving = False      # published motion last tick?
        self._estopped = False

        self.joint_pub = self.create_publisher(JointState, 'arm/joint_cmd', 10)
        self.grip_pub = self.create_publisher(Float32, 'arm/gripper_cmd', 10)

        self.create_subscription(
            Joy, self.get_parameter('joy_topic').value, self._joy_cb, 10)
        self.create_subscription(
            JointState, 'arm/joint_states', self._state_cb, 10)
        # joystick_estop_node latches this (transient-local); match its QoS or
        # we would silently never receive the retained state.
        self.create_subscription(
            Bool, 'emergency_stop', self._estop_cb,
            QoSProfile(depth=1,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        self._dt = 1.0 / rate
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f'Arm joy jog ready: right stick = base/shoulder '
            f'({self.scale_base:.2f}/{self.scale_shoulder:.2f} rad/s), '
            f'button {self.btn_open} opens / {self.btn_close} closes the gripper. '
            f'Waiting for arm/joint_states before moving.')

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        # Rescale so motion starts at zero right at the edge of the deadzone
        # instead of jumping to `deadzone` worth of speed.
        span = 1.0 - self.deadzone
        return (abs(value) - self.deadzone) / span * (1.0 if value > 0 else -1.0)

    @staticmethod
    def _axis(msg, idx):
        return msg.axes[idx] if 0 <= idx < len(msg.axes) else 0.0

    @staticmethod
    def _pressed(msg, idx):
        return 0 <= idx < len(msg.buttons) and msg.buttons[idx] == 1

    def _joy_cb(self, msg: Joy):
        self._axes = (self._deadzone(self._axis(msg, self.axis_base)),
                      self._deadzone(self._axis(msg, self.axis_shoulder)))
        if self.drive_lockout and self._driving(msg):
            self._grip_dir = 0
        else:
            opening = self._pressed(msg, self.btn_open)
            closing = self._pressed(msg, self.btn_close)
            # Both held is ambiguous — do nothing rather than pick a winner.
            self._grip_dir = 0 if opening == closing else (-1 if opening else 1)
        self._last_joy = self.get_clock().now()

    def _driving(self, msg: Joy):
        """True while the drive stick is pushed — see the R1 note up top."""
        return any(self._deadzone(self._axis(msg, ax)) != 0.0
                   for ax in self.drive_axes)

    def _state_cb(self, msg: JointState):
        # Seed the integrator once, then leave it alone: re-seeding from
        # feedback every message would fight the open-loop integration.
        if self._target is None:
            pos = dict(zip(msg.name, msg.position))
            if all(j in pos for j in JOG_JOINTS):
                self._target = {j: _clamp(j, pos[j]) for j in JOG_JOINTS}
                self.get_logger().info(
                    'Arm pose acquired (base=%.2f shoulder=%.2f) — stick is live.'
                    % (self._target['base'], self._target['shoulder']))
        if self._grip is None and 'hand' in msg.name:
            self._grip = _clamp('hand', msg.position[list(msg.name).index('hand')])

    def _estop_cb(self, msg: Bool):
        if msg.data and not self._estopped:
            self.get_logger().warn('E-stop latched — arm jog disabled.')
        elif self._estopped and not msg.data:
            self.get_logger().info('E-stop cleared — arm jog re-enabled.')
        self._estopped = msg.data

    # ------------------------------------------------------------------
    # Integrator → arm
    # ------------------------------------------------------------------
    def _joy_is_stale(self):
        if self._last_joy is None:
            return True
        age = (self.get_clock().now() - self._last_joy).nanoseconds / 1e9
        return age > self.joy_timeout

    def _tick(self):
        if self._estopped or self._joy_is_stale():
            self._moving = False
            return

        if self._target is None:
            if (self._axes != (0.0, 0.0) or self._grip_dir) and not self._warned_no_feedback:
                self._warned_no_feedback = True
                self.get_logger().warn(
                    'Stick moved but no arm/joint_states yet — is arm_driver.py '
                    'running? Refusing to jog from an unknown pose.')
            return

        self._jog_joints()
        self._jog_gripper()

    def _jog_joints(self):
        base_ax, shoulder_ax = self._axes
        moving = base_ax != 0.0 or shoulder_ax != 0.0

        if moving:
            self._target['base'] = _clamp(
                'base', self._target['base'] + base_ax * self.scale_base * self._dt)
            self._target['shoulder'] = _clamp(
                'shoulder',
                self._target['shoulder'] + shoulder_ax * self.scale_shoulder * self._dt)
        elif not self._moving:
            # Idle and already settled — stay off the serial link so the
            # driver's 10 Hz feedback polling isn't competing with us.
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOG_JOINTS)
        msg.position = [self._target[j] for j in JOG_JOINTS]
        self.joint_pub.publish(msg)
        # One extra command after release, so the arm ends exactly on the last
        # target instead of wherever the final in-flight command left it.
        self._moving = moving

    def _jog_gripper(self):
        if not self._grip_dir or self._grip is None:
            return
        new = _clamp('hand', self._grip + self._grip_dir * self.scale_gripper * self._dt)
        if new != self._grip:
            self._grip = new
            self.grip_pub.publish(Float32(data=new))


def main():
    rclpy.init()
    node = ArmJoy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
