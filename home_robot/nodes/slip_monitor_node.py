#!/usr/bin/env python3
"""Says "I am stuck" before anything touches the robot.

The bumper only fires on contact (roomba_driver.py, packet 7), so the one
failure it cannot catch is the common one: a rug, a cable, a door threshold or
a chair leg that holds the robot without being hit. The wheels keep turning,
the encoders keep counting, odometry keeps integrating, and the robot drives
across the room in its own head while standing still in the hall. Nav2 notices
eventually — "failed to make progress" — with no idea why, and by then the
recovery behaviours have already spun in place for a while.

The comparison that catches it is already on the robot and was not being made:
wheel odometry says how fast the wheels are turning the robot, the BNO085's
gyro says how fast it is ACTUALLY turning. See home_robot/slip_detect.py for
the thresholds and why the forward case is the harder one.

Publishes, deliberately WITHOUT stopping the robot:

  /wheel_slip     (String, latched JSON) — for the dashboard, recovery_manager
                  and anything else that wants to react
  speech_response (String) — "κόλλησα" out loud, rate-limited

Not an e-stop. A false positive that halts the robot mid-errand is worse than
a false positive that says something wrong, and the thresholds have never been
tested against a real rug. Whoever wants to act on this can subscribe.
"""
import json
import math
import os
import subprocess
import time

import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

from home_robot.slip_detect import SlipDetector
from home_robot.slip_map import SlipMap, correction_delta, is_significant


class SlipMonitorNode(Node):

    def __init__(self):
        super().__init__('slip_monitor')

        self.declare_parameter('min_wheel_wz', 0.25)
        self.declare_parameter('gyro_ratio', 0.35)
        self.declare_parameter('window', 1.5)
        self.declare_parameter('min_vx', 0.08)
        self.declare_parameter('range_ratio', 0.25)
        # How long a slip must persist before it is announced. One window of
        # bad luck (a gyro sample dropped, a scan that caught a passing leg) is
        # not a stuck robot.
        self.declare_parameter('confirm_seconds', 2.0)
        # Silence between spoken complaints. Being told once is help; being
        # told every two seconds while you free the robot is not.
        self.declare_parameter('speak_cooldown', 25.0)
        self.declare_parameter('announce', True)
        # The forward sector the LiDAR check measures across, in degrees either
        # side of straight ahead. Narrow on purpose: a wide sector picks up
        # walls sliding past during a turn.
        self.declare_parameter('front_sector_deg', 20.0)

        p = self.get_parameter
        self.det = SlipDetector(
            window=p('window').value,
            min_wheel_wz=p('min_wheel_wz').value,
            gyro_ratio=p('gyro_ratio').value,
            min_vx=p('min_vx').value,
            range_ratio=p('range_ratio').value,
        )
        self.confirm_seconds = p('confirm_seconds').value
        self.speak_cooldown = p('speak_cooldown').value
        self.announce = p('announce').value
        self.front_sector = math.radians(p('front_sector_deg').value)

        self._since = {'rotation': None, 'linear': None}
        self._last_spoke = 0.0
        self._state = None          # last published state, to publish on change

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(String, 'wheel_slip', latched)
        self._speech = self.create_publisher(String, 'speech_response', 10)

        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        # BEST_EFFORT to match imu_node's publisher — a RELIABLE subscription
        # would simply never connect to it.
        self.create_subscription(
            Imu, '/imu/data', self._cb_imu,
            QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 5)

        # ── Slip map ───────────────────────────────────────────────────────
        # The live numbers say HOW MUCH the odometry is drifting; this says
        # WHERE. A robot that loses 20 cm every time it crosses one rug has a
        # fixable problem; one that loses 2 cm everywhere has a calibration
        # problem — and those look identical on a live readout.
        self.declare_parameter('slip_map', True)
        self.declare_parameter('slip_map_cell', 0.5)
        self.declare_parameter('slip_map_min_shift', 0.03)
        self.declare_parameter('slip_map_half_life_days', 14.0)
        self._map_pub = self.create_publisher(String, 'slip_map', latched)
        self._slipmap = None
        self._corr_prev = None
        self._map_dirty = False
        if p('slip_map').value:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            self._slipmap = SlipMap(
                path=os.path.expanduser('~/.home_robot/slip_map.json'),
                map_name=self._active_map(),
                cell=p('slip_map_cell').value,
                half_life_days=p('slip_map_half_life_days').value)
            self._slipmap.load()
            # ‼️ 5 Hz, not 1 Hz. The mark lands where the robot was WHEN THE
            # CORRECTION WAS SEEN, not where it was when AMCL made it, so the
            # sampling period is a position error: at 1 Hz and 0.2 m/s that is
            # 20 cm, which is most of a 0.5 m cell, and a test drive put the
            # mark a whole cell past the rug that caused it. At 5 Hz it is 4 cm.
            self.create_timer(0.2, self._sample_correction)
            # Writing on every event would hit the disk during navigation, when
            # the CPU is already the scarce thing; a minute of unsaved history
            # is an acceptable loss.
            self.declare_parameter('slip_map_save_seconds', 60.0)
            self.create_timer(float(p('slip_map_save_seconds').value),
                              self._save_map)
            self._publish_map()

        self.create_timer(0.5, self._evaluate)
        self._publish(False, None, {})
        self.get_logger().info(
            f'Slip monitor up: gyro must show at least '
            f'{self.det.gyro_ratio:.0%} of the wheels above '
            f'{self.det.min_wheel_wz} rad/s, sustained {self.confirm_seconds}s')

    # ── inputs ─────────────────────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        self.det.feed_wheel(time.monotonic(),
                            msg.twist.twist.linear.x,
                            msg.twist.twist.angular.z)

    def _cb_imu(self, msg: Imu):
        self.det.feed_gyro(time.monotonic(), msg.angular_velocity.z)

    def _cb_scan(self, msg: LaserScan):
        """Nearest valid return in a narrow forward sector, in the LiDAR frame.

        ‼️ The C1 is mounted backwards (laser->base_link is a 180° yaw), so
        "forward" in scan angles is ±180°, not 0. Rather than hardcode that,
        the sector is taken around the angle whose base_link bearing is 0 —
        which for this mount is π. Kept as a simple offset because the only
        thing that matters here is that the sector points where the robot
        drives; a remount that changes it will show up as the forward check
        never firing, not as a wrong reading.
        """
        best = None
        n = len(msg.ranges)
        for i in range(n):
            r = msg.ranges[i]
            if not math.isfinite(r) or r < max(0.02, msg.range_min) or r > msg.range_max:
                continue
            a = msg.angle_min + i * msg.angle_increment
            # base_link bearing = laser angle + π (the mount), wrapped.
            b = math.atan2(math.sin(a + math.pi), math.cos(a + math.pi))
            if abs(b) <= self.front_sector and (best is None or r < best):
                best = r
        self.det.feed_scan(time.monotonic(), best)

    # ── verdict ────────────────────────────────────────────────────────────
    def _evaluate(self):
        now = time.monotonic()
        kinds = {'rotation': self.det.check_rotation(),
                 'linear': self.det.check_linear()}

        firing, detail = None, {}
        for kind, (slipping, info) in kinds.items():
            if slipping:
                if self._since[kind] is None:
                    self._since[kind] = now
                # Only counts once it has persisted — see confirm_seconds.
                if now - self._since[kind] >= self.confirm_seconds and firing is None:
                    firing, detail = kind, info
            else:
                # None (cannot judge) clears the timer just like False does:
                # a robot that stopped turning is not still slipping, and
                # holding the timer would fire on the next unrelated turn.
                self._since[kind] = None

        self._publish(firing is not None, firing, detail)

        if firing and self.announce and now - self._last_spoke >= self.speak_cooldown:
            self._last_spoke = now
            self._speech.publish(String(data=(
                'Οι τροχοί μου γυρίζουν αλλά δεν στρίβω — κόλλησα κάπου.'
                if firing == 'rotation' else
                'Προχωράω αλλά δεν κουνιέμαι — κόλλησα κάπου.')))
            self.get_logger().warn(f'slip ({firing}): {detail}')

    # ── slip map ───────────────────────────────────────────────────────────
    @staticmethod
    def _active_map():
        """Which map is loaded, so a remap does not inherit the old house.

        ‼️ A slip map recorded against another map file is worse than none:
        the coordinates only mean anything in the frame they were taken in, so
        loading them would paint the previous house's rug onto this one's
        living room.
        """
        try:
            r = subprocess.run(
                ['ros2', 'param', 'get', '/map_server', 'yaml_filename'],
                capture_output=True, text=True, timeout=10)
            for token in (r.stdout or '').split():
                if token.endswith('.yaml'):
                    return os.path.basename(token)[:-5]
        except Exception:                                 # noqa: BLE001
            pass
        return None

    def _sample_correction(self):
        """Watch map->odom, and when it jumps, note where the robot was.

        map->odom IS the correction: everything AMCL had to undo to put the
        dead reckoning back onto the walls. Only the CHANGE counts — the
        absolute transform also carries wherever the odom origin happened to be
        at boot, which on a healthy robot reads as metres.
        """
        try:
            corr = self._tf_buffer.lookup_transform('map', 'odom',
                                                    rclpy.time.Time())
            pose = self._tf_buffer.lookup_transform('map', 'base_link',
                                                    rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException):
            return                      # not localizing; nothing to record

        t, r = corr.transform.translation, corr.transform.rotation
        cur = (t.x, t.y, math.atan2(2.0 * (r.w * r.z + r.x * r.y),
                                    1.0 - 2.0 * (r.y * r.y + r.z * r.z)))
        shift, turn = correction_delta(self._corr_prev, cur)
        self._corr_prev = cur
        if not is_significant(shift, turn,
                              min_shift=self.get_parameter(
                                  'slip_map_min_shift').value):
            return
        p = pose.transform.translation
        self._slipmap.add(p.x, p.y, shift, turn)
        self._map_dirty = True
        self.get_logger().info(
            f'slip map: {shift*100:.1f} cm / {math.degrees(turn):.1f}° '
            f'correction at ({p.x:.2f}, {p.y:.2f})')
        self._publish_map()

    def _save_map(self):
        if self._slipmap and self._map_dirty:
            self._slipmap.prune()
            self._slipmap.save()
            self._map_dirty = False

    def _publish_map(self):
        if not self._slipmap:
            return
        cells = self._slipmap.decayed()
        self._map_pub.publish(String(data=json.dumps({
            'map': self._slipmap.map_name,
            'cell': self._slipmap.cell,
            # Only what is worth drawing. The tail is thousands of cells the
            # robot merely drove through, and sending them would make the map
            # tab look like the whole house is a problem.
            'cells': [{'x': round(c['x'], 2), 'y': round(c['y'], 2),
                       'm': round(c['m'], 3), 'deg': round(c['deg'], 1),
                       'n': c['n']}
                      for c in cells[:60] if c['m'] >= 0.05],
            'total': len(cells),
        })))

    def _publish(self, slipping, kind, detail):
        state = (slipping, kind)
        if state == self._state:
            return                  # latched topic; only speak up on a change
        self._state = state
        self._pub.publish(String(data=json.dumps({
            'slipping': slipping,
            'kind': kind,
            'detail': {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in detail.items()},
            't': time.time(),
        })))


def main():
    rclpy.init()
    node = SlipMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
