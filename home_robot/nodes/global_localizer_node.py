#!/usr/bin/env python3
"""Global localization — 2D LiDAR + D435 depth scan matching.

Finds the robot's position on the map with no initial pose hint.

Runs on startup, and figures out on its own whether it needs to: if
pose_saver has a pose for the active map, the scan is scored against the
map at that pose, and it is kept only if it actually fits. If it doesn't
— someone carried the robot to another room — the full map search runs
automatically. So a moved robot re-finds itself without being driven to
the AprilTag and without deleting the saved pose by hand.

Set verify_saved_pose:=false to go back to trusting a saved pose blindly.

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


def _wrap(angle: float) -> float:
    """Angle to (-pi, pi] — so 359° and 1° read as 2° apart, not 358°."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


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
        self._dist:  np.ndarray | None    = None   # distance-to-wall field [m] cache
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
        # Don't take the saved pose on faith — check the scan against the map at
        # that pose first, and search the whole map if it doesn't fit. This is
        # what lets a robot that was carried somewhere find itself on its own,
        # instead of needing to be driven to the AprilTag to be corrected.
        self.declare_parameter('verify_saved_pose', True)
        # Median scan->wall distance, metres. A correctly localized robot sits
        # well under 0.15 m; being in the wrong room puts it far above this.
        self.declare_parameter('saved_pose_max_err', 0.30)
        # How many distinct hypotheses to refine and compare. The FFT peak alone
        # is not trustworthy (see the candidate selection in _run), so more than
        # one has to be measured properly.
        self.declare_parameter('refine_candidates', 4)
        # Two candidates count as the same place within this distance, so the
        # shortlist explores different rooms rather than one room five times.
        self.declare_parameter('candidate_min_separation', 0.8)   # m
        # Fits within this of the best are treated as indistinguishable, and the
        # last known pose breaks the tie. 0 disables the tie-break.
        self.declare_parameter('prior_tie_margin', 0.05)          # m
        # Metres of scan->wall error that a whole unit of blocked-beam fraction
        # is worth. At 1.0 a candidate with 30% of its beams passing through
        # walls must beat a clear one by 0.30 m of fit. 0 disables the check.
        self.declare_parameter('blocked_weight', 1.0)
        # Only used to warn: above this, even the winner is seeing through
        # walls, which usually means the wrong map or moved furniture.
        self.declare_parameter('max_blocked_fraction', 0.30)
        self.declare_parameter('depth_weight',  0.5)   # 0 → LiDAR only
        # On a random/cold start, briefly hold the auto-trigger until the D435
        # depth arrives so it joins the very first match (better disambiguation),
        # then fall back to LiDAR-only after depth_timeout if it never shows.
        self.declare_parameter('wait_for_depth', True)
        self.declare_parameter('depth_timeout',  8.0)  # seconds

        # ── watchdog: keep the red scan glued to the walls, for good ──────
        # One-shot localization at startup is not enough. AMCL only corrects
        # while the robot DRIVES (no odometry motion => no resampling), so a
        # pose that lands 0.2-0.4 m off just sits there being wrong, which is
        # exactly what "οι κόκκινες γραμμές είναι λάθος" looks like in RViz.
        # This timer re-measures the live scan against the map and snaps the
        # pose back onto the walls by itself, without anyone pressing 2D Pose
        # Estimate. Only ever acts on a STATIONARY robot: mid-drive the scan
        # and the TF are from slightly different instants, and a correction
        # published under a running Nav2 goal would yank the path sideways.
        self.declare_parameter('watchdog', True)
        self.declare_parameter('watchdog_period', 5.0)     # s between checks
        # Above this median scan->wall error the pose is worth nudging. Measured
        # in the flat 2026-07-31: a pose that still had 0.20 m of slack in it
        # scored 0.071 m, and a perfect alignment scores ~0.00 (the distance
        # field is zero AT the wall cells), so 0.12 was lax enough to leave
        # visibly-off red lines sitting there. watchdog_min_gain, not this
        # threshold, is what stops pointless corrections.
        self.declare_parameter('watchdog_max_err', 0.06)   # m
        # Only republish if the local search actually buys this much accuracy,
        # so we don't reset AMCL's particles for a third decimal place.
        self.declare_parameter('watchdog_min_gain', 0.03)  # m
        # A snap is a correction, not a teleport: anything bigger than this is
        # a different hypothesis and belongs to the full global search.
        self.declare_parameter('watchdog_max_shift', 0.50) # m
        # Sustained error this large means genuinely lost (carried to another
        # room), so search the whole map instead of nudging.
        self.declare_parameter('watchdog_lost_err', 0.30)  # m
        self.declare_parameter('watchdog_lost_strikes', 3)
        self.declare_parameter('watchdog_cooldown', 20.0)  # s after any action
        self._wd_last_pose:  tuple | None = None   # last laser pose seen
        self._wd_strikes = 0
        self._wd_next_ok = 0.0      # monotonic time the watchdog may act again
        self._wd_backoff = 0.0      # grows when a snap fails to help
        if self.get_parameter('watchdog').value:
            self.create_timer(self.get_parameter('watchdog_period').value,
                              self._watchdog)

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
            self._lf   = None         # invalidate cached fields
            self._dist = None
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
                if not self.get_parameter('verify_saved_pose').value:
                    self._auto = False
                    self.get_logger().info(
                        f'Saved pose found ({pose_file}) — pose_saver will restore it; '
                        'call /localize_globally if the robot was moved')
                    return
                # Trust it only if the scan actually agrees with it. Deferring
                # blindly is what made a moved robot need a trip to the
                # AprilTag: the saved pose is only right if nobody picked the
                # robot up, and nothing was checking.
                verdict = self._saved_pose_fits(pose_file)
                if verdict is None:
                    return          # scan/TF not ready yet — try again next scan
                fits, err, where = verdict
                if fits:
                    self._auto = False
                    self.get_logger().info(
                        f'Saved pose {where} matches the scan '
                        f'(median scan->wall {err:.2f} m) — keeping it')
                    return
                self.get_logger().warn(
                    f'Saved pose {where} does NOT match the scan '
                    f'(median scan->wall {err:.2f} m > '
                    f'{self.get_parameter("saved_pose_max_err").value:.2f} m) — '
                    'the robot was moved; searching the whole map instead')

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

    def _saved_pose_laser(self):
        """The saved base_link pose expressed as a laser pose, or None."""
        pose_file = _pose_file_for(self.get_parameter('map_yaml').value)
        if not os.path.exists(pose_file):
            return None
        try:
            import yaml
            with open(pose_file) as fh:
                saved = yaml.safe_load(fh) or {}
            return self._base_to_laser(float(saved['x']), float(saved['y']),
                                       float(saved['yaw']))
        except Exception:
            return None

    def _pick(self, refined):
        """Choose among refined candidates: best fit, tie-broken by the prior.

        When two hypotheses fit the walls equally well the geometry genuinely
        cannot tell them apart (repeated room shapes do this), and picking by a
        third decimal place is a coin toss. Prefer the one nearer where the
        robot was last seen — it is far likelier to have stayed put than to have
        been carried to the one other place that looks identical. Only applies
        inside prior_tie_margin, so a robot that really was moved still gets the
        pose the scan supports.
        """
        best = refined[0]
        margin = self.get_parameter('prior_tie_margin').value
        rivals = [c for c in refined if c[0] - best[0] <= margin]
        if len(rivals) < 2:
            return best

        prior = self._saved_pose_laser()
        if prior is None:
            self.get_logger().warn(
                f'{len(rivals)} poses fit within {margin:.2f} m and there is no '
                'previous pose to break the tie — taking the best fit; '
                'drive a metre and AMCL will settle it')
            return best

        px, py, _ = prior
        nearest = min(rivals, key=lambda c: math.hypot(c[1] - px, c[2] - py))
        if nearest is not best:
            self.get_logger().warn(
                f'{len(rivals)} poses fit within {margin:.2f} m; choosing the one '
                f'{math.hypot(nearest[1] - px, nearest[2] - py):.2f} m from the '
                f'last known pose over a {best[0]:.3f} m fit '
                f'{math.hypot(best[1] - px, best[2] - py):.2f} m away')
        return nearest

    def _saved_pose_fits(self, pose_file: str):
        """Score the saved pose against the current scan.

        Returns (fits, median_err_m, "x=.. y=.. yaw=..°"), or None if the data
        needed to judge isn't there yet — an unreadable file counts as a
        mismatch (better to search than to trust a pose we can't read).
        """
        try:
            import yaml
            with open(pose_file) as fh:
                saved = yaml.safe_load(fh) or {}
            bx, by, byaw = float(saved['x']), float(saved['y']), float(saved['yaw'])
        except Exception as e:
            self.get_logger().warn(f'Cannot read {pose_file} ({e!r}) — will localize')
            return False, 9.9, '(unreadable)'

        with self._lock:
            scan, map_msg = self._scan, self._map
        if scan is None or map_msg is None:
            return None
        lidar = _lidar_pts(scan)
        if lidar.shape[0] < 50:
            return None

        lx, ly, lyaw = self._base_to_laser(bx, by, byaw)
        err = self._wall_err_fn(lidar, map_msg)(lx, ly, lyaw)
        where = f'x={bx:.2f} y={by:.2f} yaw={math.degrees(byaw):.0f}°'
        return err <= self.get_parameter('saved_pose_max_err').value, err, where

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

    def _base_laser_tf(self):
        """The static base_link->laser transform as (yaw, x, y).

        Falls back to this robot's known mount (yaw=π, 0.12 m forward) if TF
        isn't up yet — getting this wrong puts the pose 180° out, which AMCL
        then locks onto.
        """
        frame = self._scan.header.frame_id if self._scan is not None else 'laser'
        try:
            tf = self._tf_buffer.lookup_transform('base_link', frame,
                                                  rclpy.time.Time())
            q = tf.transform.rotation
            b = math.atan2(2 * (q.w * q.z + q.x * q.y),
                           1 - 2 * (q.y * q.y + q.z * q.z))
            return b, tf.transform.translation.x, tf.transform.translation.y
        except Exception as e:
            self.get_logger().warn(
                f'base_link->{frame} TF lookup failed ({e!r}); '
                'assuming yaw=π, x=0.12')
            return math.pi, 0.12, 0.0

    def _base_to_laser(self, bx, by, byaw):
        """base_link pose in map -> laser pose in map.

        The inverse of _laser_to_base, needed to score a saved base_link pose
        against a scan that lives in the laser frame.
        """
        b, tx, ty = self._base_laser_tf()
        lx = bx + math.cos(byaw) * tx - math.sin(byaw) * ty
        ly = by + math.sin(byaw) * tx + math.cos(byaw) * ty
        return lx, ly, byaw + b

    def _laser_to_base(self, lx, ly, lyaw):
        """Convert a laser-frame pose in map to the base_link pose. The FFT
        scan points are in the LiDAR frame, so the match yields the laser pose,
        but AMCL's /initialpose expects base_link."""
        b, tx, ty = self._base_laser_tf()
        # map->base = map->laser ∘ (base->laser)^-1
        base_yaw = lyaw - b
        t_lb_x = -tx * math.cos(b) - ty * math.sin(b)
        t_lb_y = tx * math.sin(b) - ty * math.cos(b)
        base_x = lx + math.cos(lyaw) * t_lb_x - math.sin(lyaw) * t_lb_y
        base_y = ly + math.sin(lyaw) * t_lb_x + math.cos(lyaw) * t_lb_y
        return base_x, base_y, base_yaw

    def _wall_err_fn(self, lidar: np.ndarray, map_msg: OccupancyGrid):
        """Build a median-scan-to-wall-distance evaluator for this scan + map.

        Returns f(x, y, yaw) -> metres, for a LASER-frame pose. A low value
        means the scan lands on mapped walls, i.e. the robot really is there;
        a high one means it isn't. Used both to polish a fresh match (_refine
        calls it thousands of times, hence hoisting the grid lookups into the
        closure) and to decide whether a saved pose is still valid.
        """
        if self._dist is None:
            from scipy.ndimage import distance_transform_edt
            grid = np.array(map_msg.data, dtype=np.int8).reshape(
                map_msg.info.height, map_msg.info.width)
            # distance from every free cell to the nearest occupied (wall) cell
            self._dist = (distance_transform_edt(grid < 65)
                          * map_msg.info.resolution).astype(np.float32)
        dist = self._dist
        res = map_msg.info.resolution
        ox  = map_msg.info.origin.position.x
        oy  = map_msg.info.origin.position.y
        H, W = dist.shape
        px = lidar[:, 0]
        py = lidar[:, 1]

        def median_err(x: float, y: float, yaw: float) -> float:
            c, s = math.cos(yaw), math.sin(yaw)
            mx = c * px - s * py + x
            my = s * px + c * py + y
            gx = np.round((mx - ox) / res).astype(np.int32)
            gy = np.round((my - oy) / res).astype(np.int32)
            ib = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            if int(ib.sum()) < 50:
                return 9.9
            return float(np.median(dist[gy[ib], gx[ib]]))

        return median_err

    def _standable(self, map_msg: OccupancyGrid):
        """Free cells at least a robot radius clear of any wall, cached."""
        if getattr(self, '_stand_cache', None) is None:
            from scipy.ndimage import distance_transform_edt
            g = np.array(map_msg.data, dtype=np.int16).reshape(
                map_msg.info.height, map_msg.info.width)
            d = distance_transform_edt(g < 65) * map_msg.info.resolution
            self._stand_cache = (g == 0) & (d > 0.20)
        return self._stand_cache

    def _sweep_candidates(self, lidar, map_msg, pos_step=0.30,
                          yaw_step_deg=10.0, shortlist=250):
        """Exhaustive pose sweep: every standable cell x every heading.

        ‼️ This replaces the FFT correlation stage, which was the real reason
        global localization kept landing in the wrong room. The FFT ranks where
        the scan CORRELATES with the likelihood field, and on 2026-08-02 the
        room the robot was actually in never appeared in its shortlist at all —
        so no amount of better scoring downstream could recover it.

        Sweeping every pose sounds slower and is not: the scan-to-wall score
        vectorises over all positions at once for a given heading, so the whole
        map takes ~0.1 s against 17-21 s for the FFT path. Benchmarked against a
        user-confirmed ground truth the same day it landed 0.07 m and 3° out;
        the FFT picked a different room entirely.

        Returns [(fit, x, y, yaw), ...] in the laser frame, best fit first.
        """
        info = map_msg.info
        res, W, H = info.resolution, info.width, info.height
        ox, oy = info.origin.position.x, info.origin.position.y
        if self._dist is None:
            self._wall_err_fn(lidar, map_msg)      # builds the distance field
        dist = self._dist

        ys, xs = np.nonzero(self._standable(map_msg))
        stride = max(1, int(round(pos_step / res)))
        sel = (ys % stride == 0) & (xs % stride == 0)
        px, py = ox + xs[sel] * res, oy + ys[sel] * res
        if px.size == 0:
            return []

        # A third of the beams is plenty to RANK poses; the winner is refined
        # afterwards on the full scan.
        pts = lidar[::3]
        bx, by = pts[:, 0][None, :], pts[:, 1][None, :]
        best = np.full(px.size, 9.9, dtype=np.float32)
        best_yaw = np.zeros(px.size, dtype=np.float32)
        for yaw in np.arange(-math.pi, math.pi, math.radians(yaw_step_deg)):
            c, sn = math.cos(yaw), math.sin(yaw)
            dx = c * bx - sn * by
            dy = sn * bx + c * by
            gx = ((px[:, None] + dx - ox) / res).astype(np.int32)
            gy = ((py[:, None] + dy - oy) / res).astype(np.int32)
            np.clip(gx, 0, W - 1, out=gx)
            np.clip(gy, 0, H - 1, out=gy)
            med = np.median(dist[gy, gx], axis=1)
            upd = med < best
            best[upd] = med[upd]
            best_yaw[upd] = yaw

        order = np.argsort(best)[:shortlist]
        return [(float(best[i]), float(px[i]), float(py[i]), float(best_yaw[i]))
                for i in order]

    def _visibility_fn(self, lidar, map_msg):
        """Build a beam-model evaluator: fraction of beams the map says should
        have stopped much earlier than they did.

        ‼️ This is the question scan->wall distance cannot ask. A likelihood
        field only checks whether endpoints landed near SOME wall; it never
        checks whether the beam could have got there. Stand the robot one room
        over and its returns still land on walls — the neighbours' walls, seen
        straight through the one in between — and it scores perfectly while
        being physically impossible.

        Two details matter, both learned the hard way on 2026-08-02:
          * SHORT beams only. _lidar_pts keeps returns out to the C1's 12 m, and
            in a flat a 10 m beam nearly always grazes something; with those
            included every candidate scored 70-90% and the metric said nothing.
          * Compare RANGES, not "did the ray touch a wall cell". This map has
            furniture painted into it, so mere contact marks almost every beam.
            What identifies a wrong room is a beam that should have stopped
            METRES earlier.

        Measured with the robot in the middle of the living room: 4-8% here,
        22-47% for every candidate elsewhere on the map.
        """
        g = np.array(map_msg.data, dtype=np.int16).reshape(
            map_msg.info.height, map_msg.info.width)
        wall = g >= 65
        res = map_msg.info.resolution
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        H, W = wall.shape

        rho_all = np.hypot(lidar[:, 0], lidar[:, 1])
        near = lidar[(rho_all > 0.15) & (rho_all < 5.0)][::3]
        if near.shape[0] < 30:
            return lambda x, y, yaw: 0.0        # nothing to judge on
        px, py = near[:, 0], near[:, 1]
        rho = np.hypot(px, py)
        steps = 28
        frac = np.linspace(0.04, 0.97, steps)[None, :]
        tol = 0.45                               # m of slack for quantisation

        def blocked(x, y, yaw):
            c, sn = math.cos(yaw), math.sin(yaw)
            ex = c * px - sn * py + x
            ey = sn * px + c * py + y
            # (beams, 1) * (1, steps) -> (beams, steps)
            sx = x + (ex - x)[:, None] * frac
            sy = y + (ey - y)[:, None] * frac
            gx = np.round((sx - ox) / res).astype(np.int32)
            gy = np.round((sy - oy) / res).astype(np.int32)
            ib = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            if int(ib.sum()) < steps * 10:
                return 1.0
            hit = np.zeros(gx.shape, dtype=bool)
            hit[ib] = wall[gy[ib], gx[ib]]
            any_hit = hit.any(axis=1)
            first = np.argmax(hit, axis=1)
            expected = rho * frac[0][first]
            short_by = np.where(any_hit, rho - expected, 0.0)
            return float((short_by > tol).mean())

        return blocked

    def _refine(self, lidar: np.ndarray, lx: float, ly: float, lyaw: float,
                map_msg: OccupancyGrid, span: float = 0.4, step: float = 0.04,
                ang: int = 8, quiet: bool = False):
        """Local grid search that minimises median scan->wall distance.

        Refines a laser-frame pose (lx, ly, lyaw) over ±span m / ±ang° using a
        distance-to-wall field, so the published pose is metric-accurate
        rather than just FFT-pixel-accurate. The watchdog calls this on a
        coarser grid (and quietly) because it runs every few seconds, not once.

        Returns (x, y, yaw) — or (x, y, yaw, err) when quiet, since the caller
        then needs the score to decide whether the nudge is worth publishing.
        """
        if lidar.shape[0] < 50:
            return (lx, ly, lyaw, 9.9) if quiet else (lx, ly, lyaw)
        median_err = self._wall_err_fn(lidar, map_msg)

        best = (median_err(lx, ly, lyaw), lx, ly, lyaw)
        for dx in np.arange(-span, span + 1e-9, step):
            for dy in np.arange(-span, span + 1e-9, step):
                for dth in range(-ang, ang + 1):
                    yaw = lyaw + math.radians(dth)
                    err = median_err(lx + dx, ly + dy, yaw)
                    if err < best[0]:
                        best = (err, lx + dx, ly + dy, yaw)
        if quiet:
            return best[1], best[2], best[3], best[0]
        self.get_logger().info(
            f'  refined: median scan->wall {best[0]:.3f} m '
            f'(shifted {math.hypot(best[1] - lx, best[2] - ly):.2f} m, '
            f'{math.degrees(best[3] - lyaw):+.0f}°)')
        return best[1], best[2], best[3]

    # ── watchdog ───────────────────────────────────────────────────

    def _live_laser_pose(self):
        """Where AMCL currently thinks the laser is, from TF. None if unknown."""
        try:
            tf = self._tf_buffer.lookup_transform('map', 'laser', rclpy.time.Time())
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _watchdog(self):
        """Periodically re-fit the live scan to the map and correct the pose."""
        if self._running:
            return
        import time
        now = time.monotonic()
        if now < self._wd_next_ok:
            return

        with self._lock:
            scan, map_msg = self._scan, self._map
        if scan is None or map_msg is None:
            return
        pose = self._live_laser_pose()
        if pose is None:
            return
        lx, ly, lyaw = pose

        # Stationary only. Also the natural place to act: the robot has just
        # finished a goal / is parked, which is when the user is looking at RViz.
        prev, self._wd_last_pose = self._wd_last_pose, pose
        if prev is None:
            return
        if (math.hypot(lx - prev[0], ly - prev[1]) > 0.02
                or abs(_wrap(lyaw - prev[2])) > math.radians(1.0)):
            self._wd_strikes = 0        # moving: AMCL is correcting itself
            self._wd_backoff = 0.0
            return

        lidar = _lidar_pts(scan)
        if lidar.shape[0] < 50:
            return
        err = self._wall_err_fn(lidar, map_msg)(lx, ly, lyaw)

        if err <= self.get_parameter('watchdog_max_err').value:
            self._wd_strikes = 0
            self._wd_backoff = 0.0
            return

        # Sustained gross error => not a drift, a wrong hypothesis. Search all.
        # This band RETURNS either way: a ±0.3 m nudge cannot reach the right
        # room, and applying one anyway would both corrupt the prior the global
        # search tie-breaks on and start a cooldown, so the strikes needed to
        # escalate could never accumulate — the robot would stay lost, nudging
        # itself sideways every 20 s.
        if err >= self.get_parameter('watchdog_lost_err').value:
            self._wd_strikes += 1
            if self._wd_strikes >= self.get_parameter('watchdog_lost_strikes').value:
                self.get_logger().warn(
                    f'Watchdog: scan->wall {err:.2f} m for '
                    f'{self._wd_strikes} checks — robot looks lost, '
                    're-running global localization')
                self._wd_strikes = 0
                self._wd_next_ok = now + self.get_parameter('watchdog_cooldown').value
                threading.Thread(target=self._run, daemon=True).start()
            return
        self._wd_strikes = 0

        # Drifted a little: snap it back onto the walls.
        nx, ny, nyaw, nerr = self._refine(lidar, lx, ly, lyaw, map_msg,
                                          span=0.30, step=0.05, ang=6, quiet=True)
        shift = math.hypot(nx - lx, ny - ly)
        gain  = err - nerr
        cooldown = self.get_parameter('watchdog_cooldown').value
        if (gain < self.get_parameter('watchdog_min_gain').value
                or shift > self.get_parameter('watchdog_max_shift').value):
            # Nothing better nearby — the error is furniture/clutter the map
            # doesn't have, not a bad pose. Back off so we stop burning CPU on
            # a scene that will not improve until something actually moves.
            self._wd_backoff = min(max(self._wd_backoff * 2.0, cooldown), 300.0)
            self._wd_next_ok = now + self._wd_backoff
            return

        bx, by, byaw = self._laser_to_base(nx, ny, nyaw)
        self.get_logger().info(
            f'Watchdog: scan->wall {err:.2f} -> {nerr:.2f} m — nudging pose '
            f'{shift:.2f} m, {math.degrees(_wrap(nyaw - lyaw)):+.0f}°')
        self._publish(bx, by, byaw, tight=True)
        self._wd_backoff = 0.0
        self._wd_next_ok = now + cooldown
        self._wd_last_pose = None      # pose jumped; re-baseline next tick

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

            # ── Collect scan points, all in the LiDAR frame ─────────
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

            # NB: the sweep and every scorer below use the LiDAR alone. The
            # forward depth points were only ever there to sharpen the FFT
            # correlation peak; the sweep does not correlate, and a 60° forward
            # cone cannot help decide which room a 360° scan came from.
            # ── Exhaustive pose sweep ──────────────────────────────
            # Replaces the FFT correlation stage. See _sweep_candidates: the
            # FFT ranked where the scan CORRELATES, and the room the robot was
            # actually in never even reached the shortlist (2026-08-02). The
            # sweep scores every standable cell at every heading, is ~0.1 s
            # against 17-21 s, and hit a confirmed ground truth to 0.07 m / 3°.
            wall_err = self._wall_err_fn(lidar, map_msg)
            scored = self._sweep_candidates(lidar, map_msg)
            scored.sort(key=lambda c: c[0])

            # Distinct places only — the fine search returns many near-duplicate
            # angles at the same spot, and refining five of those would explore
            # one hypothesis while ignoring the alternatives.
            spread = self.get_parameter('candidate_min_separation').value
            distinct = []
            for cand in scored:
                if all(math.hypot(cand[1] - k[1], cand[2] - k[2]) > spread
                       or abs(_wrap(cand[3] - k[3])) > math.radians(45)
                       for k in distinct):
                    distinct.append(cand)
                if len(distinct) >= self.get_parameter('refine_candidates').value:
                    break

            self.get_logger().info(
                '  candidates by scan->wall fit: ' + ', '.join(
                    f'({c[1]:.1f},{c[2]:.1f},{math.degrees(c[3]):.0f}°)'
                    f'={c[0]:.3f}m' for c in distinct))

            # ── Fine refinement of each candidate ──────────────────
            # The FFT peak is only pixel-accurate and the correlation maximum
            # can sit ~0.5 m off the geometric best fit. Published as
            # /initialpose that leaves AMCL ~0.5 m out, and since AMCL is a
            # particle filter that only converges WITH MOTION, the scan looks
            # off the walls until the robot drives. Refining polishes each
            # hypothesis (LiDAR only — 360°, more reliable than the forward
            # depth for alignment) so they can be compared fairly.
            refined = []
            for err, lx, ly, theta in distinct:
                rx, ry, rtheta = self._refine(lidar, lx, ly, theta, map_msg)
                refined.append((wall_err(rx, ry, rtheta), rx, ry, rtheta))
            refined.sort(key=lambda c: c[0])

            # ── Reject what could not have produced this scan ───────
            # Endpoint fit alone cannot tell rooms apart in a flat where the
            # shapes repeat: candidates routinely tie at 0.050 m. The beam model
            # breaks the tie by asking whether each return was even reachable —
            # 4-8% blocked where the robot really is, 22-47% everywhere else.
            see = self._visibility_fn(lidar, map_msg)
            w_blk = self.get_parameter('blocked_weight').value
            combined = sorted(
                ((fit + w_blk * see(rx, ry, rt), fit, see(rx, ry, rt), rx, ry, rt)
                 for fit, rx, ry, rt in refined), key=lambda c: c[0])
            self.get_logger().info(
                '  with line-of-sight: ' + ', '.join(
                    f'({c[3]:.1f},{c[4]:.1f})={c[0]:.3f}'
                    f'[fit {c[1]:.3f} blk {c[2]*100:.0f}%]' for c in combined[:5]))
            if combined[0][2] > self.get_parameter('max_blocked_fraction').value:
                self.get_logger().warning(
                    f'  even the best pose has {combined[0][2]*100:.0f}% of beams '
                    'passing through walls — is this the right map, or has '
                    'furniture moved?')
            # ‼️ Hand _pick the COMBINED score. It tie-breaks anything within
            # prior_tie_margin by nearness to the last known pose, and on the
            # raw fit everything ties at 0.050 — so the prior would decide and
            # the robot would "find" itself wherever it last was, silently
            # undoing all of the above.
            refined = [(c[0], c[3], c[4], c[5]) for c in combined]

            best_score, laser_x, laser_y, best_theta = self._pick(refined)

            robot_x, robot_y, best_theta = self._laser_to_base(
                laser_x, laser_y, best_theta)

            self.get_logger().info(
                f'Done in {time.monotonic() - t0:.1f}s — '
                f'x={robot_x:.2f} y={robot_y:.2f} '
                f'yaw={math.degrees(best_theta):.1f}°  '
                f'scan->wall={best_score:.3f} m')

            self._publish(robot_x, robot_y, best_theta)

        except Exception as exc:
            import traceback
            self.get_logger().error(f'Localization error: {exc}')
            traceback.print_exc()
        finally:
            self._running = False

    def _publish(self, x: float, y: float, yaw: float, tight: bool = False):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qz, qw = _quat(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        if tight:
            # A watchdog nudge is a small measured correction of a pose AMCL
            # already agrees with, so it must not blow the particle cloud back
            # open — that would undo the very convergence it is protecting.
            msg.pose.covariance[0]  = 0.05
            msg.pose.covariance[7]  = 0.05
            msg.pose.covariance[35] = 0.02
            self._pub.publish(msg)
            return
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
