#!/usr/bin/env python3
"""AprilTag reference-tag relocalizer.

One tag (id 0, frame `saloni_tag`) mounted at a fixed spot gives the robot an
absolute pose fix without a manual 2D Pose Estimate or a wrong-room scan-match.

Two phases:
  1. CALIBRATE (once, while correctly localized): call the `~/calibrate_tag`
     service. We look up map->saloni_tag from TF (valid because AMCL is correct)
     and save it to ~/.ros/saloni_tag_map_pose.yaml.
  2. RUNTIME: whenever the tag is seen, compose the saved map->tag with the live
     tag->base_link (which does NOT depend on AMCL — it goes tag->camera->base
     through the detection + static camera extrinsics) to get map->base_link and
     publish it once on /initialpose. AMCL then tracks with lidar + odom.

Services (std_srvs/Empty):
  ~/calibrate_tag  -> record map->tag now (needs correct localization + tag visible)
  ~/relocalize     -> force a one-shot /initialpose from the current tag sighting
"""
import math
import os

import numpy as np
import rclpy
import tf2_ros
import yaml
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import Empty


def quat_to_mat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(4)
    x, y, z, w = x / n, y / n, z / n, w / n
    m = np.eye(4)
    m[0, 0] = 1 - 2 * (y * y + z * z); m[0, 1] = 2 * (x * y - z * w);     m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w);     m[1, 1] = 1 - 2 * (x * x + z * z); m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w);     m[2, 1] = 2 * (y * z + x * w);     m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


def tf_to_mat(t: TransformStamped):
    m = quat_to_mat(t.transform.rotation.x, t.transform.rotation.y,
                    t.transform.rotation.z, t.transform.rotation.w)
    m[0, 3] = t.transform.translation.x
    m[1, 3] = t.transform.translation.y
    m[2, 3] = t.transform.translation.z
    return m


class AprilTagRelocalizer(Node):
    def __init__(self):
        super().__init__('apriltag_relocalizer')
        self.tag_frame = self.declare_parameter('tag_frame', 'saloni_tag').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.tag_id = int(self.declare_parameter('tag_id', 0).value)
        self.auto_seed = bool(self.declare_parameter('auto_seed', True).value)
        # How often auto_seed may re-lock while the tag stays visible. 0 = the
        # old once-per-session behaviour (needed a manual ~/relocalize call after
        # any divergence or a physical move). >0 = re-seed periodically, so just
        # parking the robot in front of the tag re-syncs AMCL again and again
        # without a service call. Throttled so it can't reset AMCL every frame.
        self.auto_seed_interval = float(
            self.declare_parameter('auto_seed_interval', 8.0).value)
        # A periodic auto-seed only fires when the tag DISAGREES with the pose
        # by at least this much. Below it, re-seeding costs an AMCL filter reset
        # and buys nothing but the tag's own jitter. An explicit ~/relocalize
        # ignores both thresholds — asking for it means you want it applied.
        self.auto_seed_min_shift = float(
            self.declare_parameter('auto_seed_min_shift', 0.25).value)   # m
        self.auto_seed_min_yaw = float(
            self.declare_parameter('auto_seed_min_yaw', 8.0).value)      # deg
        default_calib = os.path.expanduser('~/.ros/saloni_tag_map_pose.yaml')
        self.calib_file = self.declare_parameter('calib_file', default_calib).value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_tag = None          # 4x4 map->tag, from calibration
        self._load_calib()
        self.seeded = False
        self.force_once = False
        self._last_seed = 0.0   # monotonic-ish sec of the last auto/forced seed

        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(AprilTagDetectionArray, '/detections',
                                 self._on_detections, 10)
        self.create_service(Empty, '~/calibrate_tag', self._srv_calibrate)
        self.create_service(Empty, '~/relocalize', self._srv_relocalize)

        self.get_logger().info(
            f"apriltag_relocalizer up. tag_frame={self.tag_frame} id={self.tag_id} "
            f"auto_seed={'every %.0fs' % self.auto_seed_interval if (self.auto_seed and self.auto_seed_interval > 0) else ('once' if self.auto_seed else 'off')} "
            f"calibrated={'yes' if self.map_tag is not None else 'NO - call ~/calibrate_tag'}")

    def _load_calib(self):
        try:
            with open(self.calib_file) as f:
                d = yaml.safe_load(f)
            m = quat_to_mat(d['qx'], d['qy'], d['qz'], d['qw'])
            m[0, 3] = d['x']; m[1, 3] = d['y']; m[2, 3] = d['z']
            self.map_tag = m
            self.get_logger().info(f"Loaded tag map-pose from {self.calib_file}")
        except FileNotFoundError:
            self.map_tag = None
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Could not load calib ({e}); needs ~/calibrate_tag")
            self.map_tag = None

    def _tag_visible(self, msg):
        return any(int(d.id) == self.tag_id for d in msg.detections)

    def _on_detections(self, msg):
        if not self._tag_visible(msg):
            return
        now = self.get_clock().now().nanoseconds / 1e9
        fire = False
        if self.force_once:
            fire = True
        elif self.auto_seed:
            if self.auto_seed_interval <= 0.0:
                fire = not self.seeded                       # once per session
            else:
                fire = (now - self._last_seed) >= self.auto_seed_interval
        # An explicit ~/relocalize (force_once) always publishes; a periodic
        # auto-seed only does when it would actually change something.
        forced = self.force_once
        if fire and self._publish_initialpose(forced=forced):
            self.seeded = True
            self.force_once = False
            self._last_seed = now
        elif fire:
            # Gated or failed — still restart the interval, or a robot parked in
            # front of the tag re-evaluates on every single detection frame.
            self._last_seed = now

    def _lookup(self, target, source):
        return self.tf_buffer.lookup_transform(target, source, Time(),
                                               timeout=Duration(seconds=0.3))

    def _pose_delta(self, m_map_base):
        """How far the tag's answer is from where the robot thinks it is.

        Returns (shift_m, yaw_deg), or None when the current pose is unknown
        (no map->base_link yet — then there is nothing to compare and the seed
        should go ahead).
        """
        try:
            cur = self._lookup(self.map_frame, self.base_frame)
        except Exception:  # noqa: BLE001
            return None
        dx = float(m_map_base[0, 3]) - cur.transform.translation.x
        dy = float(m_map_base[1, 3]) - cur.transform.translation.y
        q = cur.transform.rotation
        cur_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
        new_yaw = math.atan2(m_map_base[1, 0], m_map_base[0, 0])
        dyaw = math.atan2(math.sin(new_yaw - cur_yaw), math.cos(new_yaw - cur_yaw))
        return math.hypot(dx, dy), abs(math.degrees(dyaw))

    def _publish_initialpose(self, forced: bool = True):
        if self.map_tag is None:
            self.get_logger().warn("Tag seen but not calibrated - call ~/calibrate_tag "
                                   "while correctly localized first.")
            return False
        try:
            # base_link expressed in the tag frame = tag->base (no AMCL dependency:
            # resolves via tag->camera_optical->camera_link->base_link).
            t_tag_base = self._lookup(self.tag_frame, self.base_frame)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF {self.tag_frame}->{self.base_frame} failed: {e}")
            return False
        m_map_base = self.map_tag @ tf_to_mat(t_tag_base)

        # Don't re-seed a pose that is already right. Measured 2026-07-31 with
        # the robot PARKED in front of the tag: consecutive tag fixes wandered
        # over 16 deg of yaw, and each one reset AMCL's filter, so the laser
        # scan visibly swung off the walls every 8 s. Worse, it fought the
        # localization watchdog — watchdog snapped the scan onto the walls,
        # the tag yanked it back, round and round every cooldown. A periodic
        # re-lock is only worth its disruption when the pose is actually wrong.
        if not forced:
            delta = self._pose_delta(m_map_base)
            if (delta is not None
                    and delta[0] <= self.auto_seed_min_shift
                    and delta[1] <= self.auto_seed_min_yaw):
                self.get_logger().debug(
                    f"Tag agrees with the current pose ({delta[0]:.2f} m, "
                    f"{delta[1]:.1f} deg) — not re-seeding")
                return False

        qx, qy, qz, qw = mat_to_quat(m_map_base)

        msg = PoseWithCovarianceStamped()
        # Stamp 0, not now() — AMCL transforms the pose AT this stamp and a
        # future one is refused outright ("extrapolation into the future"),
        # which drops the tag fix in silence. See global_localizer_node._publish.
        msg.header.stamp = Time().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = float(m_map_base[0, 3])
        msg.pose.pose.position.y = float(m_map_base[1, 3])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        cov = [0.0] * 36
        cov[0] = 0.05   # x variance
        cov[7] = 0.05   # y variance
        cov[35] = 0.06  # yaw variance
        msg.pose.covariance = cov
        self.pub.publish(msg)

        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        self.get_logger().info(
            f"Relocalized from tag -> /initialpose x={m_map_base[0, 3]:.2f} "
            f"y={m_map_base[1, 3]:.2f} yaw={math.degrees(yaw):.1f} deg")
        return True

    def _srv_calibrate(self, req, resp):
        try:
            t = self._lookup(self.map_frame, self.tag_frame)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error("Calibrate failed (need correct localization + "
                                    f"tag visible): {e}")
            return resp
        d = {
            'x': float(t.transform.translation.x),
            'y': float(t.transform.translation.y),
            'z': float(t.transform.translation.z),
            'qx': float(t.transform.rotation.x),
            'qy': float(t.transform.rotation.y),
            'qz': float(t.transform.rotation.z),
            'qw': float(t.transform.rotation.w),
        }
        os.makedirs(os.path.dirname(self.calib_file), exist_ok=True)
        # Atomic — losing this to a power cut means re-calibrating the tag
        # by hand against the map. See record_location._save_locations.
        tmp = self.calib_file + '.tmp'
        with open(tmp, 'w') as f:
            yaml.safe_dump(d, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.calib_file)
        self.map_tag = tf_to_mat(t)
        self.get_logger().info(f"Calibrated: saved map->{self.tag_frame} to {self.calib_file}")
        return resp

    def _srv_relocalize(self, req, resp):
        self.force_once = True
        self.get_logger().info("Relocalize requested - will fire on next tag sighting.")
        return resp


def main():
    rclpy.init()
    node = AprilTagRelocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
