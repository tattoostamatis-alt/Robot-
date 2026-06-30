#!/usr/bin/env python3
"""Global localization — 2D LiDAR + D435 depth scan matching.

Finds the robot's position on the map with no initial pose hint.

Auto-triggers on startup only when pose_saver has no saved pose file for
the active map (pose_saver restores the last good AMCL pose otherwise).
If the robot was moved, call /localize_globally manually or delete the
per-map pose file under ~/.ros/last_amcl_pose_<map>.yaml.

Manual trigger:
  ros2 service call /localize_globally std_srvs/srv/Empty "{}"

Algorithm:
  1. Build Gaussian likelihood field from the occupancy map
  2. Combine 360° LiDAR scan + D435 depth virtual horizontal scan
  3. FFT cross-correlation at COARSE_ANGLES orientations (full map)
  4. Fine angle refinement around the N_CANDIDATES best coarse peaks
  5. Publish best pose to /initialpose so AMCL converges there
"""

import math
import os
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_srvs.srv import Empty
import tf2_ros

COARSE_ANGLES = 36      # 10° step, full 360°
FINE_HALF_DEG = 12.0    # ± degrees around each coarse peak
FINE_STEPS    = 10      # steps within the fine window (≈2.4° each)
N_CANDIDATES  = 5       # top coarse peaks to refine
LF_SIGMA_M    = 0.25    # Gaussian blur radius for likelihood field [m]


def _quat(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2), math.cos(yaw / 2)


def _pose_file_for(map_yaml: str) -> str:
    """Same path convention as pose_saver_node — keep in sync."""
    if not map_yaml:
        return os.path.join(os.path.expanduser('~/.ros'), 'last_amcl_pose.yaml')
    stem = os.path.splitext(os.path.basename(map_yaml))[0]
    return os.path.join(os.path.expanduser('~/.ros'), f'last_amcl_pose_{stem}.yaml')


def _likelihood_field(occ: np.ndarray, res: float) -> np.ndarray:
    """Gaussian distance field: high near obstacles, falls off with sigma."""
    from scipy.ndimage import distance_transform_edt
    dist_m = distance_transform_edt(occ != 100) * res
    return np.exp(-0.5 * (dist_m / LF_SIGMA_M) ** 2).astype(np.float32)


def _lidar_pts(scan: LaserScan) -> np.ndarray:
    """LaserScan → (N, 2) XY points in robot frame."""
    n = len(scan.ranges)
    angles = np.linspace(scan.angle_min, scan.angle_max, n, dtype=np.float32)
    r = np.asarray(scan.ranges, dtype=np.float32)
    ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
    return np.column_stack([r[ok] * np.cos(angles[ok]),
                            r[ok] * np.sin(angles[ok])])


def _depth_pts(depth_m: np.ndarray, info: CameraInfo) -> np.ndarray:
    """D435 depth image → virtual forward scan XY points in robot frame.

    For each image column takes the closest valid obstacle (middle 60% of
    rows to skip floor / ceiling).  Camera frame (Z forward, X right) →
    robot frame (x forward, y left): x = Zc = depth, y = −Xc.
    """
    fx: float = info.k[0]
    cx: float = info.k[2]
    H, W = depth_m.shape

    strip = depth_m[int(H * 0.2): int(H * 0.8), :]
    valid = (strip > 0.15) & (strip < 7.0)
    strip = np.where(valid, strip, np.nan)
    d = np.nanmin(strip, axis=0)          # (W,) closest per column
    ok = np.isfinite(d)
    if not ok.any():
        return np.empty((0, 2), dtype=np.float32)

    u  = np.where(ok)[0].astype(np.float32)
    dv = d[ok]
    x  = dv                               # depth along optical axis = robot forward
    y  = -(u - cx) * dv / fx             # camera right → robot left (negated)
    return np.column_stack([x, y]).astype(np.float32)


def _fft_match(pts: np.ndarray, fft_lf: np.ndarray,
               shape: tuple[int, int], theta: float,
               res: float) -> tuple[float, int, int]:
    """FFT cross-correlation for one rotation angle.

    The scan points are rotated by theta, placed on a map-sized image centred
    at (W//2, H//2), then cross-correlated with the likelihood field.
    Peak position (dy, dx) gives the best translation:
        robot_x = origin_x + (W//2 + dx) * res
        robot_y = origin_y + (H//2 + dy) * res
    """
    H, W = shape
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rpts = (R @ pts.T).T

    px = np.round(rpts[:, 0] / res + W / 2).astype(np.int32)
    py = np.round(rpts[:, 1] / res + H / 2).astype(np.int32)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    scan_img = np.zeros((H, W), dtype=np.float32)
    scan_img[py[ok], px[ok]] = 1.0

    # C = IFFT(FFT(lf) · conj(FFT(S))) → cross-correlation lf ★ S
    corr = np.fft.irfft2(fft_lf * np.conj(np.fft.rfft2(scan_img)), s=(H, W))

    flat      = int(np.argmax(corr))
    iy, ix    = divmod(flat, W)
    score     = float(corr[iy, ix])
    # Unwrap FFT wraparound so (dx, dy) can be negative
    dx = ix if ix <= W // 2 else ix - W
    dy = iy if iy <= H // 2 else iy - H
    return score, dx, dy


class GlobalLocalizerNode(Node):

    def __init__(self):
        super().__init__('global_localizer')

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._map:   OccupancyGrid | None = None
        self._lf:    np.ndarray | None    = None   # likelihood field cache
        self._scan:  LaserScan | None     = None
        self._depth: np.ndarray | None    = None
        self._depth_frame: str            = 'camera_depth_optical_frame'
        self._cam:   CameraInfo | None    = None
        self._lock    = threading.Lock()
        self._running = False
        self._ready_since: float | None = None   # when map+scan first both arrived

        self.declare_parameter('auto_localize', True)
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('defer_to_saved_pose', True)
        self.declare_parameter('depth_weight',  0.5)   # 0 → LiDAR only
        # On a random/cold start, briefly hold the auto-trigger until the D435
        # depth arrives so it joins the very first match (better disambiguation),
        # then fall back to LiDAR-only after depth_timeout if it never shows.
        self.declare_parameter('wait_for_depth', True)
        self.declare_parameter('depth_timeout',  8.0)  # seconds

        latch = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, '/map',          self._cb_map,   latch)
        self.create_subscription(LaserScan,     '/scan',         self._cb_scan,  10)
        self.create_subscription(Image,
            '/camera/camera/depth/image_rect_raw', self._cb_depth, 10)
        self.create_subscription(CameraInfo,
            '/camera/camera/depth/camera_info',    self._cb_cam,   10)

        self._pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_service(Empty, '/localize_globally', self._svc)

        self._auto = self.get_parameter('auto_localize').value
        if self._auto:
            self.get_logger().info(
                'Auto-localize enabled — will find position once map + scan arrive')
        else:
            self.get_logger().info(
                'global_localizer ready  '
                '(ros2 service call /localize_globally std_srvs/srv/Empty "{}")')

    # ── sensor callbacks ───────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        with self._lock:
            self._map = msg
            self._lf  = None          # invalidate cached field
        self.get_logger().info(
            f'Map received: {msg.info.width}×{msg.info.height} '
            f'{msg.info.resolution:.3f} m/px')
        self._maybe_auto()

    def _cb_scan(self, msg: LaserScan):
        with self._lock:
            self._scan = msg
        self._maybe_auto()

    def _cb_depth(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        with self._lock:
            self._depth = arr.astype(np.float32) * 0.001   # mm → m
            if msg.header.frame_id:
                self._depth_frame = msg.header.frame_id

    def _cb_cam(self, msg: CameraInfo):
        with self._lock:
            self._cam = msg

    def _maybe_auto(self):
        if not self._auto:
            return
        with self._lock:
            have_map   = self._map is not None
            have_scan  = self._scan is not None
            have_depth = self._depth is not None and self._cam is not None
        if not (have_map and have_scan):
            return

        if self.get_parameter('defer_to_saved_pose').value:
            pose_file = _pose_file_for(self.get_parameter('map_yaml').value)
            if os.path.exists(pose_file):
                self._auto = False
                self.get_logger().info(
                    f'Saved pose found ({pose_file}) — pose_saver will restore it; '
                    'call /localize_globally if the robot was moved')
                return

        # map+scan ready — optionally hold briefly so the D435 depth can join.
        want_depth = (self.get_parameter('depth_weight').value > 0
                      and self.get_parameter('wait_for_depth').value)
        if want_depth and not have_depth:
            import time
            now = time.monotonic()
            if self._ready_since is None:
                self._ready_since = now
                self.get_logger().info('Map+scan ready — waiting for D435 depth…')
            if now - self._ready_since < self.get_parameter('depth_timeout').value:
                return
            self.get_logger().warn('Depth not received in time — localizing with LiDAR only')

        self._auto = False
        threading.Thread(target=self._run, daemon=True).start()

    # ── service ────────────────────────────────────────────────────

    def _svc(self, _req, resp):
        if not self._running:
            threading.Thread(target=self._run, daemon=True).start()
        return resp

    # ── main algorithm ─────────────────────────────────────────────

    def _transform_pts(self, pts: np.ndarray, src_frame: str,
                       dst_frame: str) -> np.ndarray:
        """2D transform of (N,2) points from src_frame into dst_frame."""
        if pts.size == 0:
            return pts
        try:
            tf = self._tf_buffer.lookup_transform(dst_frame, src_frame,
                                                  rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(
                f'TF {src_frame}->{dst_frame} failed ({e!r}); dropping depth pts')
            return np.empty((0, 2), dtype=np.float32)
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        out = np.empty_like(pts)
        out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + tx
        out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + ty
        return out

    def _laser_to_base(self, lx, ly, lyaw):
        """Convert a laser-frame pose in map to the base_link pose. The FFT
        scan points are in the LiDAR frame, so the match yields the laser pose,
        but AMCL's /initialpose expects base_link. Uses the static
        base_link->laser transform (looked up via TF; falls back to this
        robot's known yaw=π, 0.12 m mount if TF isn't ready)."""
        frame = self._scan.header.frame_id if self._scan is not None else 'laser'
        try:
            tf = self._tf_buffer.lookup_transform('base_link', frame,
                                                  rclpy.time.Time())
            q = tf.transform.rotation
            b = math.atan2(2 * (q.w * q.z + q.x * q.y),
                           1 - 2 * (q.y * q.y + q.z * q.z))
            tx, ty = tf.transform.translation.x, tf.transform.translation.y
        except Exception as e:
            self.get_logger().warn(
                f'base_link->{frame} TF lookup failed ({e!r}); '
                'assuming yaw=π, x=0.12')
            b, tx, ty = math.pi, 0.12, 0.0
        # map->base = map->laser ∘ (base->laser)^-1
        base_yaw = lyaw - b
        t_lb_x = -tx * math.cos(b) - ty * math.sin(b)
        t_lb_y = tx * math.sin(b) - ty * math.cos(b)
        base_x = lx + math.cos(lyaw) * t_lb_x - math.sin(lyaw) * t_lb_y
        base_y = ly + math.sin(lyaw) * t_lb_x + math.cos(lyaw) * t_lb_y
        return base_x, base_y, base_yaw

    def _run(self):
        if self._running:
            return
        self._running = True
        try:
            import time
            t0 = time.monotonic()
            self.get_logger().info('Global localization started…')

            with self._lock:
                map_msg = self._map
                scan    = self._scan
                depth   = self._depth
                cam     = self._cam

            if map_msg is None or scan is None:
                self.get_logger().error('Map or scan not yet available — aborting')
                return

            # Build likelihood field (cached across runs)
            if self._lf is None:
                self.get_logger().info('Building likelihood field…')
                grid = np.array(map_msg.data, dtype=np.int8).reshape(
                    map_msg.info.height, map_msg.info.width)
                self._lf = _likelihood_field(grid, map_msg.info.resolution)

            lf  = self._lf
            res = map_msg.info.resolution
            ox  = map_msg.info.origin.position.x
            oy  = map_msg.info.origin.position.y
            H, W = lf.shape

            # ── Collect scan points (all in the LiDAR frame for FFT) ─
            lidar_frame = scan.header.frame_id or 'laser'
            lidar = _lidar_pts(scan)
            self.get_logger().info(f'  LiDAR: {len(lidar)} pts')

            w = self.get_parameter('depth_weight').value
            depth_pts = np.empty((0, 2), dtype=np.float32)
            if depth is not None and cam is not None and w > 0:
                depth_pts = _depth_pts(depth, cam)
                with self._lock:
                    depth_frame = self._depth_frame
                depth_pts = self._transform_pts(depth_pts, depth_frame, lidar_frame)
                self.get_logger().info(
                    f'  Depth virtual scan: {len(depth_pts)} pts '
                    f'({depth_frame}->{lidar_frame})')

            if len(depth_pts) > 0:
                # Repeat depth points to give them relative weight w vs LiDAR
                n_rep = max(1, round(w * len(lidar) / len(depth_pts)))
                pts = np.vstack([lidar] + [depth_pts] * n_rep)
            else:
                pts = lidar

            # ── Coarse FFT search ──────────────────────────────────
            fft_lf  = np.fft.rfft2(lf)   # precompute — reused for all angles
            coarse  = []
            for i in range(COARSE_ANGLES):
                theta = 2 * math.pi * i / COARSE_ANGLES
                score, dx, dy = _fft_match(pts, fft_lf, (H, W), theta, res)
                coarse.append((score, dx, dy, theta))
            coarse.sort(key=lambda c: c[0], reverse=True)

            # ── Fine search around top candidates ──────────────────
            half = math.radians(FINE_HALF_DEG)
            fine = []
            for _, cdx, cdy, cth in coarse[:N_CANDIDATES]:
                for j in range(FINE_STEPS):
                    theta = cth - half + 2 * half * j / max(FINE_STEPS - 1, 1)
                    score, dx, dy = _fft_match(pts, fft_lf, (H, W), theta, res)
                    fine.append((score, dx, dy, theta))
            fine.sort(key=lambda c: c[0], reverse=True)

            best_score, best_dx, best_dy, best_theta = fine[0]

            # FFT scan points are in the LiDAR frame, so (robot_x, robot_y,
            # best_theta) is the LASER pose in map — convert it to the base_link
            # pose before publishing /initialpose. This lidar is mounted yaw=π
            # (0.12 m forward), so skipping this conversion put the published
            # pose 180° out, which AMCL then locked onto (the "flip").
            laser_x = ox + (W // 2 + best_dx) * res
            laser_y = oy + (H // 2 + best_dy) * res
            robot_x, robot_y, best_theta = self._laser_to_base(
                laser_x, laser_y, best_theta)

            self.get_logger().info(
                f'Done in {time.monotonic() - t0:.1f}s — '
                f'x={robot_x:.2f} y={robot_y:.2f} '
                f'yaw={math.degrees(best_theta):.1f}°  score={best_score:.1f}')

            self._publish(robot_x, robot_y, best_theta)

        except Exception as exc:
            import traceback
            self.get_logger().error(f'Localization error: {exc}')
            traceback.print_exc()
        finally:
            self._running = False

    def _publish(self, x: float, y: float, yaw: float):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qz, qw = _quat(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Conservative uncertainty — scan matching ≈ ±0.5m / ±10°
        msg.pose.covariance[0]  = 0.5    # x  [m²]
        msg.pose.covariance[7]  = 0.5    # y  [m²]
        msg.pose.covariance[35] = 0.2    # yaw [rad²]
        self._pub.publish(msg)
        self.get_logger().info(
            f'Published /initialpose  x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}°')


def main():
    rclpy.init()
    node = GlobalLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
