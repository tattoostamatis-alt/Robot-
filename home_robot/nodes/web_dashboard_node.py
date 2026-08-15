#!/usr/bin/env python3
"""Web dashboard — the whole robot in one browser tab.

    ros2 run home_robot web_dashboard_node.py
    Open: http://<host>:8080/?t=<token>

Started automatically by `robot max` (use_dashboard:=false to skip it).

Two kinds of panel live here:

  * Native panels rendered from ROS topics — map, camera, arm, vacuum, voice,
    system.  Cheap, phone-friendly, and they work with nothing else running.

  * The real Qt GUIs — RViz, MoveIt, Gazebo — streamed as pixels from headless
    VNC displays owned by scripts/gui_session.sh.  These are the actual
    applications, not lookalikes: RViz is RViz, with every display plugin and
    the MoveIt motion-planning panel.

Both go through this one port, so a phone on Tailscale needs a single URL and a
single token.  The VNC leg is bridged in-process (`/vnc/{app}`) rather than by
shelling out to websockify — one fewer daemon to supervise, and the bridge
inherits the token check instead of exposing an unauthenticated port.
"""

import asyncio
import re
import base64
import json
import math
import glob
import os
import secrets
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Set
from urllib.parse import quote

import cv2
import numpy as np
import psutil
import yaml

import rclpy
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rcl_interfaces.msg import Log as RosoutLog
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image, Imu, JointState, LaserScan, PointCloud2
# ‼️ std_msgs/Empty (a message) and std_srvs/Empty (a service) collide by name,
# and this file needs both — the localize/rtabmap service clients and the
# gesture_go publisher. Aliasing the message keeps the pre-existing `Empty` as
# the service, so none of the service call sites below change meaning.
from std_msgs.msg import Bool, Float32, Int16MultiArray, String
from std_msgs.msg import Empty as EmptyMsg
from action_msgs.srv import CancelGoal
from std_srvs.srv import Empty

from home_robot.compass import offset_from_known_bearing
from home_robot import (arm_settings, collision_skirt, keepout_files,
                        keepout_toggle, map_straighten, map_walls3d,
                        mic_settings, room_files, room_segment, safety_settings)
from home_robot.system_settings import (
    bt_args, parse_bt_devices, parse_devices, parse_volume, parse_wifi_list,
    volume_args, wifi_connect_args)
from home_robot.dashboard_i18n import LANGUAGES, as_js_table
from home_robot.status_query import room_el

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Overridable so a second instance can be brought up beside the live one for
# testing without taking the robot's dashboard off 8080.
PORT = int(os.environ.get('HOME_ROBOT_DASHBOARD_PORT', '8080'))

# Which way north points IN THE MAP — one number, calibrated once per map.
_COMPASS_PATH = os.path.expanduser('~/.ros/home_robot_compass.json')
SHARE = get_package_share_directory('home_robot')
# Resolve via the installed share dir — a path relative to __file__ breaks
# under `ros2 run` (lib/home_robot/../config does not exist).
LOCATIONS_FILE = os.path.join(SHARE, 'config', 'locations.yaml')
GUI_SESSION_SH = os.path.join(SHARE, 'scripts', 'gui_session.sh')
# Maps and map_session.sh are read from the SOURCE tree, the same precedence
# localize.launch.py's _resolve_map uses: a map saved during this session lands
# in src/, and `robot stop` kills anything running out of install/ — including,
# if we ran it from there, the very script doing the restart.
SRC_HOME = os.path.expanduser('~/robot_ws/src/home_robot')
# Prefer the source tree so a rebuilt arm model shows up without a colcon
# install, falling back to the installed share dir.
SRC_CONFIG_DIR = (os.path.join(SRC_HOME, 'config')
                  if os.path.isdir(os.path.join(SRC_HOME, 'config'))
                  else os.path.join(SHARE, 'config'))
SRC_MAPS_DIR = (os.path.join(SRC_HOME, 'maps')
                if os.path.isdir(os.path.join(SRC_HOME, 'maps'))
                else os.path.join(SHARE, 'maps'))
NOVNC_DIR = '/usr/share/novnc'
# Vendored locally (not a CDN) so the "Σάρωμα" 3D-scan view still works with
# no internet — the one deliberate exception to this dashboard's otherwise
# self-contained, no-three.js canvas rendering (arm tab etc.): a
# photorealistic textured mesh needs a real WebGL scene graph, which the
# hand-rolled canvas painter's-algorithm renderer cannot give it.
THREE_VENDOR_DIR = os.path.join(SRC_HOME, 'web_static', 'vendor', 'three')

# RTAB-Map keeps ONE live database (house.db) that grows for as long as the
# 3D map session runs — there is no per-map "save" the way the 2D map/AMCL
# side has (malou2.yaml, malou3.yaml, ...). save_snapshot below is what gives
# it an equivalent: a point-in-time copy under RTAB_SAVED_DIR, named by
# timestamp, that mapping can keep growing past without touching.
RTAB_HOUSE_DB  = os.path.expanduser('~/.home_robot/rtabmap/house.db')
RTAB_SAVED_DIR = os.path.expanduser('~/.home_robot/rtabmap/saved')

# Sensors whose USB port can be power-cycled from the System tab, and the label
# each one gets. The list is fixed HERE, on the server: the browser sends a
# name, and a name that is not in this dict never reaches a command line that
# runs as root. See scripts/usb_power.sh for how the ports are resolved (never
# hardcoded — USB enumeration on this machine is not stable).
USB_DEVICES = {
    'camera': '📷 Κάμερα',
    'lidar':  '📡 Lidar',
    'mic':    '🎙️ Μικρόφωνο',
    'imu':    '🧭 IMU',
    'arm':    '🦾 Βραχίονας',
    'roomba': '🧹 Σκούπα',
}
# The root-owned copy, not the one in the source tree. A NOPASSWD sudoers entry
# pointing at a file this user can edit is a root shell with extra steps.
USB_HELPER = '/usr/local/sbin/robot-usb-power'

# Must match the display map in scripts/gui_session.sh.
VNC_PORTS = {'rviz': 5902, 'gazebo': 5903, 'moveit': 5904, 'rtabmap': 5905,
             'rtabview': 5907}
# TigerVNC sessions authenticate from ~/.vnc/passwd, which we cannot read back
# (it is DES-obfuscated).  The browser side needs the cleartext to answer the
# RFB challenge, so it is configured here rather than typed into every tab.
VNC_PASSWORD = os.environ.get('HOME_ROBOT_VNC_PASSWORD', 'RobotView1')

# The arm's USABLE envelope, not its mechanical range — measured by hand on
# 2026-07-31 with the torque cut.  Keep in sync with arm_driver's limit_*
# parameters in bringup.launch.py and with config/arm_joy_ps5.yaml.
# ‼️ On the shoulder, UP is the NEGATIVE direction: `lo` is how high it may go.
ARM_LIMITS = {
    'base':     [-3.015, 0.016],
    'shoulder': [-0.169, 1.570],
    'elbow':    [0.141, 3.079],
    'wrist':    [-1.143, 1.407],
    'roll':     [-1.165, 1.372],
    'hand':     [1.080, 3.140],
}
ARM_JOINTS = ['base', 'shoulder', 'elbow', 'wrist', 'roll']

# ── Access token ───────────────────────────────────────────────────────────────
# ‼️ This dashboard binds 0.0.0.0 and hands out full teleop, click-to-navigate,
# arm control and a live camera feed. It had no authentication at all, so anyone
# on the LAN — or anything reaching it over Tailscale — could drive the robot.
# The token is generated once and PERSISTED, so the phone bookmark keeps working
# across restarts; delete the file to rotate it. Set
# HOME_ROBOT_DASHBOARD_NO_AUTH=1 to go back to the open behaviour on a trusted,
# isolated network.
TOKEN_FILE = os.path.expanduser('~/.home_robot/dashboard_token')
NO_AUTH = os.environ.get('HOME_ROBOT_DASHBOARD_NO_AUTH') == '1'


# A short token is weak on a server bound to 0.0.0.0 that exposes the camera,
# the microphone and the drive controls. This used to REPLACE anything shorter
# than 16 characters, which locked the owner out of their own bookmark on the
# phone — so it warns and obeys instead. Whoever writes this file decides.
_WEAK_TOKEN_LEN = 16


def _load_or_create_token() -> str:
    try:
        with open(TOKEN_FILE) as f:
            tok = f.read().strip()
        if tok:
            if len(tok) < _WEAK_TOKEN_LEN:
                print(f'[dashboard] token is only {len(tok)} characters — weak, '
                      f'but keeping it as configured', flush=True)
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(24)      # 192 bits
    try:
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, 'w') as f:
            f.write(tok + '\n')
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass          # a non-persisted token still secures this run
    return tok


TOKEN = '' if NO_AUTH else _load_or_create_token()


# Once the token has been presented in the query string, it is echoed back as a
# cookie so a bare `http://<host>:8080/` works from then on. Typing the URL by
# hand or copying it out of a wrapped terminal line drops the `?t=...` and the
# reply is a blank 401, which reads as "the dashboard is down" rather than
# "you are missing a query parameter" — that is what it looked like on
# 2026-08-01. SameSite=strict is what keeps the cookie from turning into a
# cross-site hole: without it any page on the LAN could open a websocket to
# /ws (websockets are not subject to CORS) and drive the robot with the
# browser's own cookie. Strict still sends it for a bookmark or a typed
# address, because those have no initiating site.
COOKIE_NAME = 'hr_dash'
COOKIE_MAX_AGE = 365 * 24 * 3600


def _authorised(supplied: Optional[str], cookies: Optional[dict] = None) -> bool:
    if NO_AUTH:
        return True
    if supplied and secrets.compare_digest(supplied, TOKEN):
        return True
    if cookies:
        jar = cookies.get(COOKIE_NAME)
        if jar and secrets.compare_digest(jar, TOKEN):
            return True
    return False


def _lan_ip() -> str:
    """Best-effort local address for the startup banner (was hardcoded)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))   # no packet is actually sent
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return 'localhost'

# ── Locations ──────────────────────────────────────────────────────────────────

def _load_locations() -> dict:
    try:
        with open(os.path.realpath(LOCATIONS_FILE)) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# ── Map display cleanup ──────────────────────────────────────────────────────
# Cosmetic only: both functions below reshape/recolour a COPY of the grid used
# to build the picture the browser gets. self._last_map (the raw OccupancyGrid)
# and everything Nav2/slam_toolbox does with the real /map topic are untouched.

def _despeckle_grid(grid: np.ndarray) -> np.ndarray:
    """Fill small enclosed unknown holes and drop isolated obstacle pixels.

    A raw slam_toolbox map is speckled: a few percent of free-space cells read
    -1 from scan gaps (grey freckles inside an otherwise clean room), and a
    stray return off dust or a mirror leaves a lone occupied pixel in open
    floor. Neither is information — the room isn't unexplored in patches and
    a single pixel isn't a wall. Both just make the picture look dirty.

    The big unknown region OUTSIDE the house must survive untouched (that one
    really is "never scanned"), which is why only unknown blobs that do NOT
    touch the image border are filled — the exterior always does.
    """
    g = grid.copy()

    unknown = (g == -1).astype(np.uint8)
    if unknown.any():
        n, labels = cv2.connectedComponents(unknown, connectivity=4)
        border = set(labels[0, :]) | set(labels[-1, :]) \
            | set(labels[:, 0]) | set(labels[:, -1])
        border.discard(0)
        max_hole_px = 30      # ~0.075 m^2 at a typical 0.05 m/px map
        for lbl in range(1, n):
            if lbl in border:
                continue
            blob = labels == lbl
            if blob.sum() <= max_hole_px:
                g[blob] = 0    # reclassify as free

    occupied = (g == 100).astype(np.uint8)
    if occupied.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            occupied, connectivity=8)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] <= 2:   # 1-2 px noise, not a wall
                g[labels == lbl] = 0

    return g


def _crop_to_content(img: np.ndarray, grid_flipped: np.ndarray,
                      info, margin_px: int = 12):
    """Crop the grey 'never scanned' padding a SLAM map is usually laid out
    on, so the house fills the pane instead of a fraction of it.

    `img`/`grid_flipped` are already flipped to image orientation (row 0 =
    top). Only a translation, never a rotation — resolution is unchanged and
    the client's w2c()/c2w() only ever add an origin offset, so click-to-nav
    and the robot/laser overlays stay exactly correct on the cropped image.
    Returns (image, width, height, origin) — origin unchanged if no crop.
    """
    h, w = grid_flipped.shape
    known = grid_flipped != -1
    rows = np.flatnonzero(known.any(axis=1))
    cols = np.flatnonzero(known.any(axis=0))
    ox, oy, res = info.origin.position.x, info.origin.position.y, info.resolution
    if rows.size == 0 or cols.size == 0:
        return img, w, h, [ox, oy]

    top    = max(0, int(rows[0])  - margin_px)
    bottom = min(h, int(rows[-1]) + 1 + margin_px)
    left   = max(0, int(cols[0])  - margin_px)
    right  = min(w, int(cols[-1]) + 1 + margin_px)
    if top == 0 and bottom == h and left == 0 and right == w:
        return img, w, h, [ox, oy]

    new_origin = [ox + left * res, oy + (h - bottom) * res]
    return img[top:bottom, left:right], right - left, bottom - top, new_origin

# ── Shared state (thread-safe) ─────────────────────────────────────────────────

class State:
    def __init__(self):
        self._lock    = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: Set[WebSocket] = set()
        self.map_png:   Optional[bytes] = None
        self.map_info:  Optional[dict]  = None
        # Room legend + tint state, replayed with the map on connect.
        self.map_rooms: dict = {}
        self.map_tinted: bool = True
        # Keepout zones for the active map, replayed with the map on connect —
        # see home_robot/keepout_files.py.
        self.map_keepout: dict = {}
        self.camera_jpg: Optional[bytes] = None
        # When that frame arrived, and how many have arrived in total. The
        # stream needs both: the age decides whether the picture is still
        # live, and the counter lets it send each frame ONCE instead of
        # re-sending whatever is in the buffer 25 times a second.
        self.camera_at: float = 0.0
        self.camera_seq: int = 0
        # Replayed to every browser that connects, so a tab opened an hour into
        # a session is populated immediately instead of waiting for the next
        # message on each topic. Keyed by message type.
        self.latest: dict = {}
        self.chat: list = []            # rolling voice/LLM transcript
        # Rolling /rosout tail. Until this existed, reading a warning meant
        # finding the terminal that owns the launch — impossible from the phone,
        # and impossible at all once the launch is detached. Only WARN and above
        # are kept: INFO on this graph runs into the thousands per minute (Nav2
        # costmaps alone) and would push everything useful out of the buffer.
        self.logs: list = []

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def add_client(self, ws: WebSocket):
        with self._lock: self._clients.add(ws)

    def remove_client(self, ws: WebSocket):
        with self._lock: self._clients.discard(ws)

    def broadcast(self, msg: dict, remember: bool = True):
        if remember:
            self.latest[msg.get('type', '')] = msg
        if self._loop is None:
            return
        data = json.dumps(msg)
        asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)

    def add_log(self, level: int, name: str, text: str):
        entry = {'level': level, 'name': name, 'text': text, 't': time.time()}
        self.logs.append(entry)
        del self.logs[:-300]
        self.broadcast({'type': 'log', **entry}, remember=False)

    def add_chat(self, role: str, text: str):
        entry = {'role': role, 'text': text, 't': time.time()}
        self.chat.append(entry)
        del self.chat[:-60]
        self.broadcast({'type': 'chat', **entry}, remember=False)

    def send_bytes(self, targets, payload: bytes):
        """Binary frame to a SUBSET of clients — used for the live mic stream.

        Not a broadcast: audio only goes to the tabs that asked to listen, and
        never gets remembered in `latest` (replaying a second of month-old audio
        to a tab that just connected would be nonsense).
        """
        if self._loop is None or not targets:
            return
        asyncio.run_coroutine_threadsafe(
            self._bcast_bytes(list(targets), payload), self._loop)

    async def _bcast_bytes(self, clients, payload: bytes):
        for ws in clients:
            try:
                await ws.send_bytes(payload)
            except Exception:
                with self._lock:
                    self._clients.discard(ws)

    async def _bcast(self, data: str):
        dead = []
        with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        with self._lock:
            for ws in dead:
                self._clients.discard(ws)


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class DashboardNode(Node):
    def __init__(self, state: State, locations: dict):
        super().__init__('web_dashboard')
        self._state     = state
        self._locations = locations
        self._scan_seq  = 0
        self._arm_seq   = 0

        latch = QoSProfile(depth=1,
                           durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                           reliability=QoSReliabilityPolicy.RELIABLE)

        # ── Navigation / map ────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, '/map', self._cb_map, latch)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._cb_pose, 10)
        # Kept as the fast path (it fires the instant AMCL corrects), but it is
        # not the only one — see _poll_tf_pose for why a standing robot needs
        # the TF poll or the map tab shows nothing but the map.
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.create_timer(0.5, self._poll_tf_pose)
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 5)
        self.create_subscription(Path, '/plan', self._cb_plan, 5)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 5)

        # ── Sensor fusion tabs ──────────────────────────────────────────────
        # Two questions the rest of the dashboard cannot answer:
        #   1. "Sensor fusion": the EKF takes vx from the wheels and yaw from the
        #      BNO085 (see config/ekf.yaml) and AMCL corrects what is left. When
        #      the pose walks off, the only useful question is WHICH input lied,
        #      and that needs the three headings side by side — the IMU tab
        #      shows one of them, the map tab shows the result, nothing shows
        #      the disagreement.
        #   2. "Αισθητήρες": the LiDAR sees one horizontal slice at 0.606 m and
        #      the D435 sees a cone from 0.536 m. A table top, a step, or a cat
        #      lives in exactly the gap between them, so what matters is where
        #      the two DISAGREE, not either one alone.
        # Both are gated per socket like the 3D cloud: nothing is computed while
        # nobody is on the tab.
        self._fuse_ws: Set[object] = set()          # sockets on either tab
        self._fuse_cam_ws: Set[object] = set()      # sockets needing the cloud
        self.create_subscription(Odometry, '/odometry/filtered',
                                 self._cb_odom_filtered, 5)
        # Per source: last yaw (rad, unwrapped), sample count, measured rate and
        # the monotonic stamp of the newest sample. Rate/age are what expose the
        # failure the numbers hide — a frozen source keeps publishing its last
        # good heading, so only the clock shows it died.
        self._fz = {k: {'yaw': None, 'raw': None, 'turns': 0.0, 'n': 0,
                        'hz': 0.0, 'seen': 0.0, 'ref': None}
                    for k in ('wheel', 'imu', 'ekf')}
        self._fz_t0     = time.monotonic()
        self._fz_vx     = {'wheel': 0.0, 'ekf': 0.0}
        self._fz_wz     = {'wheel': 0.0, 'imu': 0.0, 'ekf': 0.0}
        self._fz_cov    = {'ekf': None, 'amcl': None}
        # ‼️ map->odom is NOT the correction — it is the correction PLUS wherever
        # the odom origin happened to be when the base powered up. Read raw it
        # showed 366 cm and 123° on a robot that was localizing perfectly
        # (measured 2026-08-05), because the robot had simply driven and turned
        # since boot. Only the CHANGE since the reset is AMCL pulling the pose
        # back, so the reference is captured and subtracted.
        self._fz_corr_ref = None       # (x, y, yaw) at the last reset
        self._fz_corr_max = 0.0        # biggest correction since the reset
        self._fz_yaw_max  = 0.0
        self._fuse_last   = 0.0
        # Angular profiles for the LiDAR/camera comparison, both in base_link
        # bearings so they are directly comparable. (bins, list of metres|None)
        self._prof_lidar  = None
        self._prof_cam    = None
        self._prof_cam_at = 0.0
        self._prof_last   = 0.0
        # Both mounts are static transforms, so they are looked up once and
        # kept. Cached as None until the first successful lookup, which is the
        # normal state for the first second or two after start.
        self._cam_tf      = None       # base_link <- camera depth optical
        self._laser_tf    = None       # base_link <- laser
        self.create_timer(0.25, self._broadcast_fusion)

        # ── IMU (BNO085) ────────────────────────────────────────────────────
        # BEST_EFFORT: the IMU is a 10 Hz firehose of "current truth" — a
        # retransmitted stale sample is worthless, dropping it is correct.
        self._imu_n     = 0        # samples seen, for the measured rate
        self._imu_t0    = time.monotonic()
        self._imu_hz    = 0.0
        self._imu_last  = 0.0      # last broadcast, for the 5 Hz throttle
        self._imu_seen  = 0.0      # monotonic stamp of the newest sample
        # Consecutive gx=gy=gz==0 samples seen WHILE THE ROBOT IS TURNING. A
        # standing robot reads exact zeros all the time (the BNO085 quantises
        # small rates straight to 0), so counting them unconditionally cried
        # "GYRO DEAD" at a robot parked on the carpet — measured on the live
        # stream before this was qualified. Only a turn that produces no yaw
        # rate is evidence of the real failure.
        self._gyro_zero = 0
        self._turning   = False    # set from /odom, see _cb_odom
        self.create_subscription(
            Imu, '/imu/data', self._cb_imu,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        # Fires even when /imu/data goes silent, which is the case that matters:
        # a dead BNO085 is SILENT, so a callback-driven panel would just freeze
        # on the last good reading and look healthy.
        self.create_timer(1.0, self._imu_health)
        # room_markers publishes this latched (TRANSIENT_LOCAL) precisely so a
        # late subscriber gets the current room; asking for volatile threw that
        # away, so the badge stayed '—' until the robot next changed room.
        self.create_subscription(String, '/current_room', self._cb_room, latch)

        # ── Camera / perception ─────────────────────────────────────────────
        # No image subscription here — see _set_camera_sub, toggled by the
        # 'cam_view' message. Detection counts/boxes come from _cb_objects/
        # _cb_poses on /detected_objects & /pose_detections below, which stay
        # subscribed unconditionally: they are tiny JSON strings at a few Hz,
        # nothing like the cost of the raw color stream.
        self.create_subscription(String, '/detected_objects', self._cb_objects, 5)
        # YOLO11n-pose, 17 COCO keypoints per person (pose_node.py, iGPU/ROCm).
        self.create_subscription(String, '/pose_detections', self._cb_poses, 5)
        # Latest overlay geometry, drawn onto the camera JPEG in _cb_camera.
        # (payload, monotonic_stamp) — the stamp is what keeps a box from being
        # painted over a scene the person already walked out of: detections
        # arrive at a few Hz against the camera's 30, and the detector can stop
        # entirely (it is behind use_perception) without saying so.
        self._det_boxes: list = []
        self._det_time  = 0.0
        self._det_poses: list = []
        self._pose_time = 0.0
        self._pose_src_w = 0        # frame width poses were measured in
        self._speaker_key = None    # last speaker snapshot forwarded
        self._overlay   = True        # toggled from the camera tab

        # ── Arm ─────────────────────────────────────────────────────────────
        self.create_subscription(JointState, '/arm/joint_states', self._cb_arm, 10)

        # ── 3D point cloud ──────────────────────────────────────────────────
        # 148814 points x 20 bytes = 3 MB per message at 28 Hz. Nothing about
        # that is sendable to a phone, so _cb_cloud subsamples and quantises,
        # and the browser asks for the stream only while the 3D tab is open
        # (dispatch 'cloud'). depth=1 + BEST_EFFORT: a late cloud is worthless,
        # dropping it is better than queueing it.
        # Which sockets are on the 3D tab. A plain counter leaked: a browser
        # closed while the tab was open never sent its 'off', so the stream ran
        # forever at 150 kB/s with nobody watching.
        self._cloud_ws: Set[object] = set()
        # Tabs on the Κάμερα pane. _cb_camera used to process EVERY color
        # frame (cvtColor + resize + overlay draw + JPEG encode) unconditionally
        # — whether or not the /camera.mjpeg stream this buffer feeds had a
        # single viewer. Same fix as _cloud_ws.
        #
        # ‼️ An early-return inside the callback alone was NOT enough — traced
        # with a temporary /debug/threads (sys._current_frames per thread) and
        # confirmed via /proc/<pid>/task/<tid>/stat deltas: the rclpy spin
        # thread stayed pinned near 65-70% of a core even with the callback
        # body confirmed skipped every time (0/156 "full" runs logged). That
        # baseline turned out to be mostly UNRELATED to the camera (still
        # present with the subscription removed entirely — the volume of this
        # node's ~40 other subscriptions, plausibly the tf2 listener, is the
        # likely rest of the story and remains uninvestigated, root/py-spy
        # would be the next step). What IS attributable to the camera,
        # measured with a controlled on/off A-B test over a live websocket:
        # ~13 points of a core, 2026-08-14. Real, worth having off by default,
        # just smaller than first suspected — DDS deserialises a
        # full-resolution color Image message at ~30 Hz before the callback
        # body ever runs, so the SUBSCRIPTION itself is created/destroyed
        # here instead, same principle as _set_camera_pointcloud toggling the
        # driver's own pointcloud.enable rather than filtering it out locally.
        self._cam_ws: Set[object] = set()
        self._camera_sub = None
        # Tabs currently listening to the microphone. Per socket, like the
        # cloud stream: two phones listening must not switch each other off.
        self._listen_ws: Set[object] = set()
        self._mic_sub = None
        self._cloud_last = 0.0
        # _set_camera_pointcloud turns the D435's pointcloud filter on for as
        # long as someone is on the 3D tab — but ONLY if this node is the one
        # that turned it on.
        #
        # ‼️ Since 2026-08-05 the launch files start the camera with
        # pointcloud.enable:=true, because Nav2's voxel_layer feeds on that
        # topic (see nav2_params.yaml `depth_camera`). Switching it off on
        # leaving the 3D tab would silently blind the costmap to every obstacle
        # the LiDAR's flat slice misses — the exact failure this whole feature
        # exists to prevent, and one that shows up as nothing at all in the
        # logs. So: read the parameter once at startup, and if it is already on,
        # somebody else owns it and this node must never touch it.
        self._cloud_param_on  = False
        self._cloud_param_own = True    # until the startup probe says otherwise
        self._cam_param_cli = self.create_client(
            SetParameters, '/camera/camera/set_parameters')
        self._cam_param_get = self.create_client(
            GetParameters, '/camera/camera/get_parameters')
        self._cam_probe_timer = self.create_timer(3.0, self._probe_cloud_owner)
        self._map_cache = None          # (fetched_at, name); see active_map()
        self._backend_cache = None      # (fetched_at, backend); see llm_backend()
        self._cam_err_at = 0.0          # throttles the decode-failure warning
        self._cam_judge_at = 0.0        # see _judge_frame
        self._cam_state_last = None
        self._room_mask = (None, None, None)   # see _load_room_mask()
        self._room_mask_key = None             # (map name, mtime) it was built from
        self._room_size_warned = False
        self._room_seg_cache = None            # see _segmentation_for()
        self._rooms_tinted = True              # toggled from the map tab
        self._last_map = None                  # so the toggle can redraw
        self._full_map_info = None             # see _room_at_xy()
        # Room-tab code (dispatch(), on the asyncio event loop) needs to know
        # the active map's name, but active_map() shells out to `ros2 param
        # get` on a cache miss (~1-2s, up to a 10s timeout) — blocking that
        # would stall every connected socket, not just the one editing rooms.
        # A background thread that refreshes the cache faster than its own TTL
        # keeps every caller a cache hit instead. See active_map()/MAP_CACHE_TTL.
        threading.Thread(target=self._map_name_keepwarm, daemon=True).start()
        self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self._cb_cloud,
            QoSProfile(depth=1,
                       reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       durability=QoSDurabilityPolicy.VOLATILE))

        # ── Log tail ────────────────────────────────────────────────────────
        # Every node publishes /rosout RELIABLE + TRANSIENT_LOCAL with a 10 s
        # lifespan (checked with `ros2 topic info -v /rosout`, 66 publishers).
        # Subscribe RELIABLE so nothing is dropped, but VOLATILE: transient
        # local here would replay one stale line per publisher at connect,
        # dating the log tail with warnings from before the tab was open.
        self.create_subscription(
            RosoutLog, '/rosout', self._cb_rosout,
            QoSProfile(depth=100,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE))

        # ── Costmap (what Nav2 actually steers around) ──────────────────────
        # The rolling local costmap: 60x60 cells of 0.05 m — a 3x3 m window
        # around the robot — at ~3 Hz. Small enough to send as a PNG per update
        # without thinking about it, and it is the one view that explains why
        # the robot swerved or refused a doorway when the map looks clear.
        #
        # ‼️ 2026-08-14: both subscriptions used to be unconditional, with
        # `_cb_costmap` gating its own work on `_costmap_on` and `_cb_semantic`
        # not gating at all. Same bug class as _cb_camera (see its comment):
        # an in-callback early-return does not stop rclpy's own executor from
        # paying to deserialise and dispatch every message first — confirmed
        # with py-spy record (--pid, --duration 15) showing 89.93% of all
        # sampled time inside rclpy's own wait_for_ready_callbacks, not in any
        # one callback body. So _set_costmap_sub creates/destroys both
        # subscriptions together, same as _set_camera_sub.
        self._costmap_on: Set[object] = set()   # sockets watching the tab
        self._costmap_sub = None
        self._semantic_sub = None
        self._semantic_pts: list = []
        self._semantic_time = 0.0

        # ── RTAB-Map (3D map of the house) ──────────────────────────────────
        # Only ever live while the «3D Χάρτης» tab has its session up; the rest
        # of the time these topics simply have no publisher, which is exactly
        # what _rtab_health reports.
        self._rtab_seen  = 0.0    # monotonic stamp of the last /rtabmap/info
        self._rtab_nodes = 0      # keyframes in the graph
        self._rtab_loops = 0      # loop closures accepted so far
        self._rtab_save_state = 'idle'   # idle|saving|done|error, see _rtab_save_snapshot
        try:
            from rtabmap_msgs.msg import Info as RtabInfo, MapGraph as RtabGraph
            self.create_subscription(RtabInfo, '/rtabmap/info', self._cb_rtab_info, 5)
            self.create_subscription(RtabGraph, '/rtabmap/mapGraph',
                                     self._cb_rtab_graph, 5)
            self._rtab_ok = True
        except ImportError:
            # rtabmap_ros is an apt package, not a dependency of this workspace.
            # Without it the tab still starts the VNC session (rtabmap_viz is a
            # separate process); only this status strip goes dark.
            self._rtab_ok = False
            self.get_logger().info('rtabmap_msgs not installed — 3D map status off')
        self.create_timer(2.0, self._rtab_health)

        # ── Vacuum base ─────────────────────────────────────────────────────
        self.create_subscription(String, '/roomba/status', self._cb_roomba, latch)
        self.create_subscription(String, '/dock_status', self._cb_dock, latch)
        self.create_subscription(Bool, '/emergency_stop', self._cb_estop, latch)

        # ── Voice / LLM ─────────────────────────────────────────────────────
        self.create_subscription(String, '/speech_text', self._cb_heard, 10)
        self.create_subscription(String, '/speech_response', self._cb_said, 10)
        self.create_subscription(String, '/wake_word', self._cb_wake, 10)
        self.create_subscription(Bool, '/tts/speaking', self._cb_speaking, 10)
        self.create_subscription(String, '/situation_context', self._cb_situation, 10)

        # ── Gestures / observations / timeline ──────────────────────────────
        # ‼️ The SHARED mic topic, not the ALSA device. wake_word_node owns the
        # stream; stt_node and sound_event_node read this same topic — this
        # dashboard is just one more subscriber, so unsubscribing costs it
        # nothing (the audio keeps flowing to the others regardless). No
        # unconditional subscription here — see _set_mic_sub, same pattern as
        # _set_camera_sub/_set_costmap_sub. Measured ~13 msgs/s continuously
        # published whether or not anyone is on the dashboard's mic-listen
        # feature, and _cb_mic's own `if not self._listen_ws: return` never
        # stopped rclpy paying to deserialise and dispatch each one first.
        self.create_subscription(String, '/gesture_status', self._cb_gesture, 10)
        self.create_subscription(String, '/observations', self._cb_observations, 10)
        self.create_subscription(String, '/object_memory', self._cb_object_memory, 5)
        self.create_subscription(String, '/vocab/state', self._cb_vocab, latch)
        self.create_subscription(String, '/hand_gesture', self._cb_hand, latch)
        self.create_subscription(String, '/people', self._cb_people, latch)
        self.create_subscription(String, '/acoustic_map', self._cb_acoustic, latch)
        self.create_subscription(String, '/arm/touch', self._cb_touch, latch)
        self.create_subscription(String, '/echo', self._cb_echo, latch)
        self.create_subscription(String, '/self_diagnosis', self._cb_diag, latch)
        # Latched: the accumulated history has to be on screen the moment the
        # page opens, not after the next correction — which on a parked robot
        # never comes.
        self.create_subscription(
            String, '/slip_map',
            lambda m: self._state.broadcast({'type': 'slip_map',
                                             **json.loads(m.data)}), latch)
        self.create_subscription(String, '/llm_quota', self._cb_quota, latch)
        self.create_subscription(String, '/sound_events', self._cb_sound, latch)
        # Both latched by their publishers, so a tab opened long after the fact
        # still paints a full timeline instead of an empty list.
        self.create_subscription(String, '/episodic/timeline', self._cb_timeline, latch)
        self.create_subscription(String, '/episodic/answer',
                                 self._cb_episodic_answer, 10)

        # ── Publishers ──────────────────────────────────────────────────────
        # ‼️ cmd_vel_safe, NOT cmd_vel — the D-pad published to /cmd_vel and the
        # robot never moved, because nothing on this graph relays it: Nav2 drives
        # cmd_vel_nav -> velocity_smoother -> cmd_vel_smoothed -> collision_monitor
        # -> cmd_vel_safe, and roomba_driver subscribes to cmd_vel_safe alone.
        # /cmd_vel had four publishers and no subscriber at all. This is the same
        # wiring the PS5 teleop already uses (localize.launch.py remaps its
        # cmd_vel -> cmd_vel_safe), so the web D-pad now takes the identical path.
        #
        # It does bypass collision_monitor, exactly as the joystick does. What
        # still protects it: roomba_driver's own bumper/cliff/wheel-drop stops,
        # its 0.25 s stale-command watchdog (so a browser that dies mid-press
        # coasts to a stop rather than running away), and the latched e-stop.
        self._vel_pub  = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self._speech_pub = self.create_publisher(String, '/speech_text', 10)
        self._say_pub  = self.create_publisher(String, '/speech_response', 10)
        self._dock_pub = self.create_publisher(Bool, '/dock', 10)
        self._follow_pub = self.create_publisher(Bool, '/follow_command', 10)
        self._nerf_pub = self.create_publisher(Bool, '/nerf/capture', 10)
        self._nerf_train_proc: Optional[subprocess.Popen] = None
        self._nerf_train_lock = threading.Lock()
        self._nerf_train_stop_requested = False
        # ── "✕ Ακύρωση στόχου" ────────────────────────────────────────────────
        # ‼️ This button used to publish ONE empty Twist and nothing else, so it
        # never cancelled anything: bt_navigator still owned the goal and put a
        # fresh command on the wire milliseconds later. Reported 2026-08-04 —
        # "πατάω ακύρωση στόχου και δεν ακυρώνεται". Worse, recovery_manager
        # re-issues the goal it learned from /plan after every escape, so even a
        # real cancel came back to life within 30 s.
        # Cancelling has to reach every layer that can put the robot in motion.
        self._nav_cancel_cli = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self._mission_pub = self.create_publisher(String, '/mission/start', 10)
        self._patrol_pub  = self.create_publisher(Bool, '/patrol_command', 10)
        self._explore_pub = self.create_publisher(Bool, '/explore_command', 10)
        # The signal recovery_manager listens to so it drops the goal instead of
        # re-issuing it. A human cancel is the one case its re-issue must lose to.
        self._cancel_pub = self.create_publisher(String, '/mission/cancel', 10)
        # The gesture "go" button and the timeline search box. Same topics the
        # `goto_pointed` and `recall` voice tools use, so button and voice take
        # one code path.
        self._gesture_go_pub = self.create_publisher(EmptyMsg, "/gesture_go", 10)
        self._recall_pub = self.create_publisher(String, '/episodic/query', 10)
        self._vocab_pub = self.create_publisher(String, '/vocab/set', 10)
        self._enrol_pub = self.create_publisher(String, '/faces/enrol', 10)
        self._forget_pub = self.create_publisher(String, '/faces/forget', 10)
        self._bind_pub = self.create_publisher(
            String, '/gesture_bindings/set', 10)
        self._people_add_pub = self.create_publisher(String, '/people/add', 10)
        self._people_rm_pub = self.create_publisher(String, '/people/remove', 10)
        self._people_enrol_pub = self.create_publisher(String, '/people/enrol', 10)
        self._echo_pub = self.create_publisher(String, '/echo/probe', 10)

        # Map-referenced compass: one calibrated number per map.
        self._last_yaw = None
        self._compass_offset = self._compass_load()
        # Remembered in state.latest, so a tab opening later still gets it.
        self._broadcast_compass()
        # Same topic the llm_bridge `check` tool publishes, so the button and
        # "πήγαινε να δεις αν…" take one code path.
        self._mission_pub = self.create_publisher(String, '/mission/start', 10)
        self.create_subscription(String, '/mission/status',
                                 self._cb_mission, 10)
        # Latched to match fall_monitor_node: a browser opened after the alert
        # must see it, not a stale all-clear.
        self.create_subscription(
            Bool, '/fall_alert', self._cb_fall,
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=QoSReliabilityPolicy.RELIABLE))
        self.create_subscription(String, '/fall_event', self._cb_fall_event, 10)
        self.create_subscription(String, '/speaker_state', self._cb_speaker, 10)
        self.create_subscription(String, '/nerf/status', self._cb_nerf, 5)
        self._arm_cmd_pub = self.create_publisher(JointState, '/arm/joint_cmd', 10)
        self._gripper_pub = self.create_publisher(Float32, '/arm/gripper_cmd', 10)
        self._arm_raw_pub = self.create_publisher(String, '/arm/raw_cmd', 10)
        # The e-stop is latched on the driver's side, so ours must be too or a
        # driver that restarts comes up not knowing the stop is engaged.
        self._estop_pub = self.create_publisher(
            Bool, '/emergency_stop',
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        # "Turn toward whoever spoke". It ships disarmed — a wake word is not a
        # command, and armed-by-default had the robot turning itself during
        # conversations nobody addressed to it. Both ends latched: doa_node
        # remembers the setting across restarts, and this dashboard is normally
        # started after it, so a volatile subscription would show an empty
        # checkbox for a feature that is actually on.
        _doa_qos = QoSProfile(depth=1,
                              durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=QoSReliabilityPolicy.RELIABLE)
        self._doa_rotate_pub = self.create_publisher(
            Bool, '/doa/rotate_enable', _doa_qos)
        self.create_subscription(
            Bool, '/doa/rotate_state', self._cb_doa_rotate, _doa_qos)
        # Hardware VAD, straight off the XVF3800 — see doa_node's own comment on
        # its voice_activity publisher for why this is latched rather than
        # plain volatile QoS (a subscription with the wrong durability just
        # never matches DDS's incompatible-QoS check, and looks identically
        # dead to "nobody is publishing").
        self.create_subscription(
            Bool, '/voice_activity', self._cb_vad, _doa_qos)

        self._loc_client = self.create_client(Empty, '/localize_globally')
        # Created on first use: the services only exist while a mapping session
        # is running, and a client made against a missing service is harmless
        # but pointless to hold for the whole life of the dashboard.
        self._rtab_clients: dict = {}

        # ── safety clearances ────────────────────────────────────────────────
        # The saved settings, the parameter clients that carry them, and the
        # set of nodes they have already been pushed to since this dashboard
        # started. Applying is not a one-shot at boot: the dashboard usually
        # comes up before Nav2 has finished configuring its costmaps, and
        # roomba_driver restarts on its own after a serial drop. So the timer
        # keeps trying until each node has actually taken its values once.
        self._safety = safety_settings.load()
        self._safety_set: dict = {}
        self._safety_get: dict = {}
        self._safety_applied: set = set()
        self._safety_live: dict = {}     # what the nodes report back
        # collision_skirt's live margin — see _current_skirt_margin_mm().
        self._skirt_margin_mm = collision_skirt.MARGIN_DEFAULT_MM
        self._skirt_margin_at = None
        self.create_timer(3.0, self._safety_tick)

        # ── arm envelope + speed ──────────────────────────────────────────────
        # Same reasoning as the safety timer above: arm_driver/arm_joy can come
        # up after the dashboard, or restart on their own (USB glitch), and
        # each time they do they boot with only bringup.launch.py's/
        # arm_joy_ps5.yaml's baked-in values until this pushes what the user
        # actually asked for.
        self._arm_settings = arm_settings.load()
        self._arm_set: dict = {}
        self._arm_get: dict = {}
        self._arm_applied: set = set()
        self._arm_live: dict = {}        # (node, param) -> value, as reported
        self.create_timer(3.0, self._arm_tick)

        # ── microphone settings ────────────────────────────────────────────────
        # Same tick/apply/payload shape as safety_settings above — one Spec
        # table, values scalar, one or two ROS nodes per knob.
        self._mic = mic_settings.load()
        self._mic_set: dict = {}
        self._mic_get: dict = {}
        self._mic_applied: set = set()
        self._mic_live: dict = {}
        self.create_timer(3.0, self._mic_tick)

        self.create_timer(2.0, self._publish_system)

    # ── safety clearances ────────────────────────────────────────────────────

    def _safety_client(self, cache, srv_type, node_name, suffix):
        cli = cache.get(node_name)
        if cli is None:
            cli = self.create_client(srv_type, f'{node_name}/{suffix}')
            cache[node_name] = cli
        return cli

    def _safety_tick(self):
        """Push settings to any node that has not taken them, then read back.

        Read-back is the point of the whole loop: a launch argument
        (`obstacle_safety_distance`) or a hand-edited nav2_params.yaml can
        disagree with the saved file, and a panel that showed the file would be
        showing a number the robot is not using. What gets displayed is always
        what the live node answered.
        """
        for node, params in safety_settings.targets(self._safety).items():
            cli = self._safety_client(self._safety_set, SetParameters,
                                      node, 'set_parameters')
            if not cli.service_is_ready():
                # Node not up (or restarted) — drop it back to un-applied so it
                # gets its values again when it returns.
                self._safety_applied.discard(node)
                continue
            if node not in self._safety_applied:
                self._safety_applied.add(node)
                cli.call_async(self._safety_request(params))
        self._safety_read()
        self._state.broadcast({'type': 'safety', 'v': self._safety_payload(),
                               'skirt_margin_mm': self._current_skirt_margin_mm()})

    @staticmethod
    def _safety_request(params: dict) -> SetParameters.Request:
        req = SetParameters.Request()
        for name, value in params.items():
            if isinstance(value, bool):
                pv = ParameterValue(type=ParameterType.PARAMETER_BOOL,
                                    bool_value=value)
            else:
                pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                    double_value=float(value))
            req.parameters.append(Parameter(name=name, value=pv))
        return req

    def _safety_read(self):
        by_node: dict = {}
        for spec in safety_settings.SPECS:
            for node, param in spec.targets:
                by_node.setdefault(node, []).append(param)
        for node, names in by_node.items():
            cli = self._safety_client(self._safety_get, GetParameters,
                                      node, 'get_parameters')
            if not cli.service_is_ready():
                for name in names:
                    self._safety_live.pop((node, name), None)
                continue
            req = GetParameters.Request()
            req.names = names
            fut = cli.call_async(req)
            fut.add_done_callback(
                lambda f, n=node, ns=list(names): self._safety_got(n, ns, f))

    def _safety_got(self, node, names, fut):
        try:
            values = fut.result().values
        except Exception:                                     # noqa: BLE001
            return
        for name, val in zip(names, values):
            if val.type == ParameterType.PARAMETER_DOUBLE:
                self._safety_live[(node, name)] = val.double_value
            elif val.type == ParameterType.PARAMETER_BOOL:
                self._safety_live[(node, name)] = val.bool_value
            elif val.type == ParameterType.PARAMETER_INTEGER:
                self._safety_live[(node, name)] = float(val.integer_value)

    def _safety_payload(self) -> dict:
        """{key: {'set':…, 'live':…, 'nodes': n_up}} for the browser.

        `live` is None where no node answered, which is how the panel greys a
        row out instead of showing a stale number as if it were in force —
        obstacle_safety_node is off in half the launch configurations, and a
        slider that looks active while nothing reads it is a lie about what is
        guarding the robot.
        """
        out = {}
        for spec in safety_settings.SPECS:
            live = [self._safety_live.get(t) for t in spec.targets]
            answered = [v for v in live if v is not None]
            out[spec.key] = {
                'set': self._safety.get(spec.key, spec.default),
                # Where a knob writes two nodes (the costmaps) they can be out
                # of step — one configured, one not. Report the minimum, i.e.
                # the least clearance actually in force.
                'live': (min(answered) if len(answered) == len(live) else None),
                'nodes': len(answered),
                'total': len(spec.targets),
            }
        return out

    def _safety_apply(self, key, value):
        """One knob, from the browser. Saved first, then pushed."""
        clamped = safety_settings.clamp(key, value)
        if clamped is None:
            return
        self._safety[key] = clamped
        try:
            safety_settings.save(self._safety)
        except OSError as exc:
            self.get_logger().warn(f'could not save safety settings: {exc}')
        spec = safety_settings.BY_KEY[key]
        for node, param in spec.targets:
            cli = self._safety_client(self._safety_set, SetParameters,
                                      node, 'set_parameters')
            if cli.service_is_ready():
                cli.call_async(self._safety_request({param: clamped}))
        self.get_logger().info(f'safety: {key} = {clamped}')
        self._state.broadcast({'type': 'safety', 'v': self._safety_payload(),
                               'skirt_margin_mm': self._current_skirt_margin_mm()})

    def _safety_reset(self):
        for key, value in safety_settings.defaults().items():
            self._safety_apply(key, value)

    # ── microphone settings ──────────────────────────────────────────────────
    # Identical shape to the safety block above (one Spec table, scalar
    # values, live SetParameters) — see mic_settings.py for why THAT module's
    # clamp()/targets() aren't reused directly even though the pattern is.

    def _mic_tick(self):
        for node, params in mic_settings.targets(self._mic).items():
            cli = self._safety_client(self._mic_set, SetParameters,
                                      node, 'set_parameters')
            if not cli.service_is_ready():
                self._mic_applied.discard(node)
                continue
            if node not in self._mic_applied:
                self._mic_applied.add(node)
                cli.call_async(self._mic_request(params))
        self._mic_read()
        self._state.broadcast({'type': 'mic', 'v': self._mic_payload()})

    @staticmethod
    def _mic_request(params: dict) -> SetParameters.Request:
        """Unlike _safety_request: also carries integer and string values —
        doa_node declares led_brightness and every led_color_* as plain
        Python ints, and wake_word_node's model_name is a str, so those have
        to go over the wire as PARAMETER_INTEGER/PARAMETER_STRING, not
        PARAMETER_DOUBLE, or the SetParameters call fails on a type mismatch."""
        req = SetParameters.Request()
        for name, value in params.items():
            if isinstance(value, bool):
                pv = ParameterValue(type=ParameterType.PARAMETER_BOOL,
                                    bool_value=value)
            elif isinstance(value, str):
                pv = ParameterValue(type=ParameterType.PARAMETER_STRING,
                                    string_value=value)
            elif isinstance(value, int):
                pv = ParameterValue(type=ParameterType.PARAMETER_INTEGER,
                                    integer_value=value)
            else:
                pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                    double_value=float(value))
            req.parameters.append(Parameter(name=name, value=pv))
        return req

    def _mic_read(self):
        by_node: dict = {}
        for spec in mic_settings.SPECS:
            for node, param in spec.targets:
                by_node.setdefault(node, []).append(param)
        wm_node, wm_param = mic_settings.WAKE_MODEL_TARGET
        by_node.setdefault(wm_node, []).append(wm_param)
        for node, names in by_node.items():
            cli = self._safety_client(self._mic_get, GetParameters,
                                      node, 'get_parameters')
            if not cli.service_is_ready():
                for name in names:
                    self._mic_live.pop((node, name), None)
                continue
            req = GetParameters.Request()
            req.names = names
            fut = cli.call_async(req)
            fut.add_done_callback(
                lambda f, n=node, ns=list(names): self._mic_got(n, ns, f))

    def _mic_got(self, node, names, fut):
        try:
            values = fut.result().values
        except Exception:                                     # noqa: BLE001
            return
        for name, val in zip(names, values):
            if val.type == ParameterType.PARAMETER_DOUBLE:
                self._mic_live[(node, name)] = val.double_value
            elif val.type == ParameterType.PARAMETER_BOOL:
                self._mic_live[(node, name)] = val.bool_value
            elif val.type == ParameterType.PARAMETER_INTEGER:
                self._mic_live[(node, name)] = val.integer_value
            elif val.type == ParameterType.PARAMETER_STRING:
                self._mic_live[(node, name)] = val.string_value

    def _mic_payload(self) -> dict:
        out = {}
        for spec in mic_settings.SPECS:
            live = [self._mic_live.get(t) for t in spec.targets]
            answered = [v for v in live if v is not None]
            out[spec.key] = {
                'set': self._mic.get(spec.key, spec.default),
                'live': (min(answered) if len(answered) == len(live) else None),
                'nodes': len(answered),
                'total': len(spec.targets),
            }
        wm_target = mic_settings.WAKE_MODEL_TARGET
        wm_live = self._mic_live.get(wm_target)
        out['wake_model'] = {
            'set': self._mic.get('wake_model', mic_settings.WAKE_MODEL_DEFAULT),
            'live': wm_live,
            'nodes': 1 if wm_live is not None else 0,
            'total': 1,
        }
        return out

    def _mic_apply(self, key, value):
        clamped = mic_settings.clamp(key, value)
        if clamped is None:
            return
        self._mic[key] = clamped
        try:
            mic_settings.save(self._mic)
        except OSError as exc:
            self.get_logger().warn(f'could not save mic settings: {exc}')
        for node, params in mic_settings.key_targets(key, clamped).items():
            cli = self._safety_client(self._mic_set, SetParameters,
                                      node, 'set_parameters')
            if cli.service_is_ready():
                cli.call_async(self._mic_request(params))
        self.get_logger().info(f'mic: {key} = {clamped}')
        self._state.broadcast({'type': 'mic', 'v': self._mic_payload()})

    def _mic_reset(self):
        for key, value in mic_settings.defaults().items():
            self._mic_apply(key, value)

    def _current_skirt_margin_mm(self):
        """collision_monitor's live moving-ring margin, mtime-cached like
        _load_room_mask() — so a restart triggered by /safety/skirt/{mm}
        shows up here as soon as the file changes, without re-parsing a
        multi-hundred-line YAML file on every 3s safety tick."""
        path = collision_skirt.default_params_path()
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            return self._skirt_margin_mm
        if self._skirt_margin_at == stamp:
            return self._skirt_margin_mm
        try:
            with open(path) as f:
                mm = collision_skirt.current_margin_mm(f.read())
        except OSError:
            mm = None
        if mm is not None:
            self._skirt_margin_mm = mm
        self._skirt_margin_at = stamp
        return self._skirt_margin_mm

    # ── arm envelope + speed ──────────────────────────────────────────────────

    def _arm_tick(self):
        for node, params in arm_settings.all_targets(self._arm_settings).items():
            cli = self._safety_client(self._arm_set, SetParameters,
                                      node, 'set_parameters')
            if not cli.service_is_ready():
                self._arm_applied.discard(node)
                continue
            if node not in self._arm_applied:
                self._arm_applied.add(node)
                cli.call_async(self._arm_request(params))
        self._arm_read()
        self._state.broadcast({'type': 'arm_settings', 'v': self._arm_payload()})

    @staticmethod
    def _arm_request(params: dict) -> SetParameters.Request:
        """Unlike _safety_request: also carries integer (accel) and
        double-array (limit_<joint>, a [lo, hi] pair) parameter types."""
        req = SetParameters.Request()
        for name, value in params.items():
            if isinstance(value, (list, tuple)):
                pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                    double_array_value=[float(v) for v in value])
            elif isinstance(value, int) and not isinstance(value, bool):
                pv = ParameterValue(type=ParameterType.PARAMETER_INTEGER,
                                    integer_value=value)
            else:
                pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                    double_value=float(value))
            req.parameters.append(Parameter(name=name, value=pv))
        return req

    def _arm_read(self):
        """Read back arm_driver's limit_* + accel. arm_driver, not arm_joy, is
        authoritative for what is actually in force: it is the node that
        refuses an out-of-range command; arm_joy only stops its own jog
        integrator winding up to one (see arm_driver.limits()' docstring)."""
        names = [f'limit_{j}' for j in arm_settings.MECH_LIMITS] + ['accel']
        cli = self._safety_client(self._arm_get, GetParameters,
                                  '/arm_driver', 'get_parameters')
        if not cli.service_is_ready():
            for name in names:
                self._arm_live.pop(('/arm_driver', name), None)
            return
        req = GetParameters.Request()
        req.names = names
        fut = cli.call_async(req)
        fut.add_done_callback(lambda f, ns=names: self._arm_got(ns, f))

    def _arm_got(self, names, fut):
        try:
            values = fut.result().values
        except Exception:                                     # noqa: BLE001
            return
        for name, val in zip(names, values):
            if val.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
                self._arm_live[('/arm_driver', name)] = list(val.double_array_value)
            elif val.type == ParameterType.PARAMETER_INTEGER:
                self._arm_live[('/arm_driver', name)] = val.integer_value
            elif val.type == ParameterType.PARAMETER_DOUBLE:
                self._arm_live[('/arm_driver', name)] = val.double_value

    def _arm_payload(self) -> dict:
        live_accel = self._arm_live.get(('/arm_driver', 'accel'))
        limits = {}
        for j in arm_settings.MECH_LIMITS:
            key = f'limit_{j}'
            limits[j] = {
                'set': self._arm_settings.get(key, list(arm_settings.DEFAULT_LIMITS[j])),
                'live': self._arm_live.get(('/arm_driver', key)),
                'mech': list(arm_settings.MECH_LIMITS[j]),
            }
        return {
            'limits': limits,
            'speed': self._arm_settings.get('speed', arm_settings.SPEED_DEFAULT),
            'speed_live': arm_settings.speed_from_accel(live_accel),
            'nodes': 1 if live_accel is not None else 0,
        }

    def _arm_apply_speed(self, pct):
        clamped = arm_settings.clamp_speed(pct)
        if clamped is None:
            return
        self._arm_settings['speed'] = clamped
        try:
            arm_settings.save(self._arm_settings)
        except OSError as exc:
            self.get_logger().warn(f'could not save arm settings: {exc}')
        for node, params in arm_settings.speed_targets(clamped).items():
            cli = self._safety_client(self._arm_set, SetParameters,
                                      node, 'set_parameters')
            if cli.service_is_ready():
                cli.call_async(self._arm_request(params))
        self.get_logger().info(f'arm speed = {clamped}')
        self._state.broadcast({'type': 'arm_settings', 'v': self._arm_payload()})

    def _arm_apply_limit(self, joint, lo, hi):
        clamped = arm_settings.clamp_limit(joint, lo, hi)
        if clamped is None:
            return
        self._arm_settings[f'limit_{joint}'] = list(clamped)
        try:
            arm_settings.save(self._arm_settings)
        except OSError as exc:
            self.get_logger().warn(f'could not save arm settings: {exc}')
        for node, params in arm_settings.limit_targets(joint, *clamped).items():
            cli = self._safety_client(self._arm_set, SetParameters,
                                      node, 'set_parameters')
            if cli.service_is_ready():
                cli.call_async(self._arm_request(params))
        self.get_logger().info(f'arm limit_{joint} = {list(clamped)}')
        self._state.broadcast({'type': 'arm_settings', 'v': self._arm_payload()})

    def _arm_reset(self):
        for key, value in arm_settings.defaults().items():
            if key == 'speed':
                self._arm_apply_speed(value)
            else:
                self._arm_apply_limit(key[len('limit_'):], value[0], value[1])

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        self._last_map = msg          # kept so the room-tint toggle can redraw
        grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        # Cosmetic pass on a COPY — Nav2/slam_toolbox still get msg.data raw
        # via their own /map subscription, this only affects what gets drawn.
        grid = _despeckle_grid(grid)
        img = np.full((msg.info.height, msg.info.width, 3), 180, dtype=np.uint8)
        img[grid == 0]   = [230, 230, 230]
        img[grid == 100] = [50,  50,  50]
        img[grid == -1]  = [160, 160, 160]
        img = cv2.flip(img, 0)
        grid_flipped = cv2.flip(grid, 0)
        # Tint the free space with each room's colour, so "πήγαινε στην κουζίνα"
        # can be checked against something visible instead of a uniform grey
        # field. Only free cells are tinted: walls stay black or the map stops
        # reading as a floor plan.
        img = self._tint_rooms(img, grid_flipped)
        # room_mask.png is pixel-aligned to this FULL, uncropped grid — kept
        # for _room_at_xy() (map-tab click-to-pick-room), which must do the
        # same origin/resolution math _tint_rooms/_room_from_mask do, not the
        # cropped-and-shifted origin below.
        self._full_map_info = (msg.info.origin.position.x,
                               msg.info.origin.position.y, msg.info.resolution)
        # room_mask.png must match the FULL, uncropped map (see _tint_rooms),
        # so the crop happens last, after tinting.
        img, width, height, origin = _crop_to_content(img, grid_flipped, msg.info)
        _, png = cv2.imencode('.png', img)
        self._state.map_png = png.tobytes()
        info = {
            'width':      width,
            'height':     height,
            'resolution': msg.info.resolution,
            'origin':     origin,
        }
        self._state.map_info = info
        # Drives the legend. Stashed on State rather than only broadcast, so
        # the replay a newly-connected browser gets carries it too — sent only
        # here, every fresh tab showed an empty legend and an unchecked
        # "Χρώματα" box over a map that was in fact tinted.
        _, _, names = self._load_room_mask()
        self._state.map_rooms = {n: list(c) for n, c in (names or {}).items()}
        self._state.map_tinted = self._rooms_tinted
        self._state.map_keepout = self._load_keepout_zones()
        self._state.broadcast({
            'type':  'map',
            **info,
            'image': base64.b64encode(self._state.map_png).decode(),
            'rooms': self._state.map_rooms,
            'tinted': self._state.map_tinted,
            'keepout': self._state.map_keepout,
        }, remember=False)   # replayed from map_png on connect, not from latest

    # How strongly the room colour is mixed into the floor. Low on purpose: the
    # map has to stay readable as a map — obstacles, the plan and the laser are
    # all drawn over it — so this is a wash, not a fill.
    ROOM_TINT = 0.38

    def _load_room_mask(self):
        """maps/<active-map>_room_mask.png as BGR + its per-room alpha, cached.

        Returns (bgr, alpha, names) or (None, None, None). Scoped to whichever
        map map_server currently has loaded (self.active_map(), itself cached —
        see room_files.py's docstring for why the file is per-map at all): the
        same file situational_awareness and room_markers read for THIS map, so
        what the dashboard paints and what the robot calls the room can never
        disagree.

        Reloaded when the file changes on disk, so regenerating it after a
        remap shows up without restarting the dashboard.
        """
        name = self.active_map()
        if not name:
            self._room_mask = (None, None, None)
            self._room_mask_key = None
            return self._room_mask
        mask_path, _ = room_files.paths_for(name)
        try:
            stamp = os.path.getmtime(mask_path)
        except OSError:
            self._room_mask = (None, None, None)
            self._room_mask_key = None
            return self._room_mask
        key = (name, stamp)
        if self._room_mask_key == key:
            return self._room_mask
        try:
            bgr, alpha, names = room_files.load(name)
            self._room_mask = (bgr, alpha, names) if bgr is not None else (None, None, None)
        except Exception as e:
            self.get_logger().warn(f'room_mask unusable: {e}')
            self._room_mask = (None, None, None)
        self._room_mask_key = key
        return self._room_mask

    def _tint_rooms(self, img, grid):
        """Blend the room mask into the free cells of an already-drawn map."""
        if not self._rooms_tinted:
            return img
        bgr, alpha, _ = self._load_room_mask()
        if bgr is None:
            return img
        if bgr.shape[:2] != img.shape[:2]:
            # A mask from a different map would paint rooms in the wrong place,
            # which is worse than no colour at all. test_smoke checks this too.
            if not self._room_size_warned:
                self._room_size_warned = True
                self.get_logger().warn(
                    f'room_mask.png is {bgr.shape[1]}x{bgr.shape[0]} but the map '
                    f'is {img.shape[1]}x{img.shape[0]} — not tinting. '
                    'Regenerate with scripts/make_room_mask.py')
            return img
        paint = alpha & (grid == 0)          # free cells only
        if not paint.any():
            return img
        out = img.astype(np.float32)
        out[paint] = (out[paint] * (1 - self.ROOM_TINT)
                      + bgr[paint] * self.ROOM_TINT)
        return out.astype(np.uint8)

    def _room_at_xy(self, x: float, y: float):
        """Which room name is painted at this map-frame (x, y), or None.

        Same pixel math and nearest-colour match as room_markers_node's
        `_room_from_mask` — kept independent rather than imported so this file
        does not need that node's ROS-only module just for two lines of math.
        Used by the map tab's "click picks a room" toggle, so clicking a
        colour on the map can jump straight to that room's row in the editor
        instead of hunting for it by eye in a name-sorted list.
        """
        bgr, alpha, names = self._load_room_mask()
        if bgr is None or not names or self._full_map_info is None:
            return None
        ox, oy, res = self._full_map_info
        h, w = bgr.shape[:2]
        col = int((x - ox) / res)
        row = h - 1 - int((y - oy) / res)      # mask is stored image-side up
        if not (0 <= col < w and 0 <= row < h) or not alpha[row, col]:
            return None
        b, g, r = bgr[row, col]
        best, best_d = None, float('inf')
        for name, rgb in names.items():
            d = (int(r) - rgb[0]) ** 2 + (int(g) - rgb[1]) ** 2 + (int(b) - rgb[2]) ** 2
            if d < best_d:
                best_d, best = d, name
        return best

    def _save_rooms(self, edits):
        """Rename/recolour rooms from the map tab's editor, for the active map.

        <map>_room_colors.yaml maps name -> [r,g,b] and <map>_room_mask.png is
        painted with those EXACT rgb values per room (scripts/auto_rooms.py or
        the click-to-place tool, see _place_room), so a colour change has to
        repaint the mask, not just the yaml, or the picture and the legend
        would disagree — and situational_awareness/room_markers, which read
        the same two files for this map, would still speak the old colour's
        name for that patch of floor.
        """
        try:
            map_name = self.active_map()
            if not map_name:
                raise ValueError('κανένας αποθηκευμένος χάρτης ενεργός')
            mask_path, colours_path = room_files.paths_for(map_name)
            if not isinstance(edits, list) or not edits:
                raise ValueError('no rooms sent')
            try:
                with open(colours_path) as f:
                    current = yaml.safe_load(f) or {}
            except OSError:
                current = {}

            from PIL import Image
            mask_arr = None
            try:
                mask_arr = np.array(Image.open(mask_path).convert('RGBA'))
            except OSError:
                pass

            new_colours = {}
            seen = set()
            repaints = []   # (match-mask, new_rgb) pairs, applied after the loop
            source = mask_arr.copy() if mask_arr is not None else None
            for e in edits:
                old = str(e.get('old', '')).strip()
                name = str(e.get('name', '')).strip()
                color = e.get('color')
                if (not name or old not in current
                        or not isinstance(color, list) or len(color) != 3):
                    continue
                if name in seen:               # two rows collapsed to one name
                    name = old                  # keep it a no-op rather than merge rooms
                seen.add(name)
                rgb = [max(0, min(255, int(v))) for v in color]
                old_rgb = list(current[old])
                if source is not None and old_rgb != rgb:
                    # Matched against `source`, a snapshot taken before any
                    # repaint — otherwise swapping two rooms' colours would
                    # have the second edit match pixels the first just wrote.
                    match = ((source[:, :, 0] == old_rgb[0])
                             & (source[:, :, 1] == old_rgb[1])
                             & (source[:, :, 2] == old_rgb[2])
                             & (source[:, :, 3] > 50))
                    repaints.append((match, rgb))
                new_colours[name] = rgb

            if not new_colours:
                raise ValueError('nothing usable in the edits')

            for match, rgb in repaints:
                mask_arr[match, 0] = rgb[0]
                mask_arr[match, 1] = rgb[1]
                mask_arr[match, 2] = rgb[2]

            if mask_arr is not None:
                Image.fromarray(mask_arr).save(mask_path)
            with open(colours_path, 'w') as f:
                yaml.safe_dump(new_colours, f, allow_unicode=True)
        except Exception as exc:
            self.get_logger().warn(f'save_rooms: {exc!r}')
            self._state.broadcast({'type': 'room_saved', 'ok': False,
                                   'error': str(exc)})
            return

        self._room_mask_key = None     # force _load_room_mask() to reread
        if self._last_map is not None:
            self._cb_map(self._last_map)   # repaints the tinted picture too
        else:
            _, _, names = self._load_room_mask()
            self._state.map_rooms = {n: list(c) for n, c in (names or {}).items()}
            self._state.broadcast({'type': 'map_rooms', 'rooms': self._state.map_rooms})
        self.get_logger().info(f'rooms saved on {map_name}: {list(new_colours)}')
        self._state.broadcast({'type': 'room_saved', 'ok': True})

    def _segmentation_for(self, map_name: str):
        """Shape-only room labels for map_name (see room_segment.py), cached
        by (name, pgm mtime) so repeated clicks while placing several rooms on
        the same map don't re-run the distance transform each time.

        Returns (labels, resolution, (origin_x, origin_y)), or raises — callers
        are the place_room worker thread, which already turns any exception
        into a `room_saved` failure broadcast.
        """
        yaml_path = os.path.join(SRC_MAPS_DIR, f'{map_name}.yaml')
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        pgm = meta.get('image', f'{map_name}.pgm')
        pgm_path = pgm if os.path.isabs(pgm) else os.path.join(SRC_MAPS_DIR, os.path.basename(pgm))
        stamp = os.path.getmtime(pgm_path)
        key = (map_name, stamp)
        if self._room_seg_cache and self._room_seg_cache[0] == key:
            return self._room_seg_cache[1]

        from PIL import Image
        img = np.array(Image.open(pgm_path))
        if img.ndim == 3:
            img = img[:, :, 0]
        res = float(meta['resolution'])
        origin = (float(meta['origin'][0]), float(meta['origin'][1]))
        labels, _count = room_segment.segment(img, res)
        result = (labels, res, origin)
        self._room_seg_cache = (key, result)
        return result

    def _place_room(self, x, y, name, color):
        """Click-inside-a-room from the map tab: segment the ACTIVE map by
        shape (room_segment.segment — same distance-transform watershed
        auto_rooms.py uses offline), take whichever blob (x, y) landed in, and
        paint just that blob into <map>_room_mask.png / _room_colors.yaml under
        `name`/`color`. Runs in a worker thread (dispatch() is on the asyncio
        event loop — self.active_map() alone can block ~1-2s shelling out to
        `ros2 param get`, and the PIL/scipy work on top of that would stall
        every other websocket for the duration).
        """
        try:
            name = str(name).strip()
            if not name:
                raise ValueError('όνομα δωματίου κενό')
            if not (isinstance(color, list) and len(color) == 3):
                raise ValueError('άκυρο χρώμα')
            rgb = [max(0, min(255, int(v))) for v in color]

            map_name = self.active_map()
            if not map_name:
                raise ValueError('κανένας αποθηκευμένος χάρτης ενεργός')

            labels, res, (ox, oy) = self._segmentation_for(map_name)
            h, w = labels.shape
            col = int((x - ox) / res)
            row = h - 1 - int((y - oy) / res)     # image-side up, same as _room_at_xy
            if not (0 <= col < w and 0 <= row < h):
                raise ValueError('εκτός χάρτη')
            label_id = int(labels[row, col])
            if label_id == 0:
                raise ValueError('Δεν βρέθηκε δωμάτιο εκεί — πολύ κοντά σε τοίχο ή πόρτα.')

            from PIL import Image
            mask_path, colours_path = room_files.paths_for(map_name)
            mask_arr = None
            if os.path.exists(mask_path):
                mask_arr = np.array(Image.open(mask_path).convert('RGBA'))
                if mask_arr.shape[:2] != (h, w):
                    mask_arr = None    # stale mask from a different map size
            if mask_arr is None:
                mask_arr = np.zeros((h, w, 4), np.uint8)
            try:
                with open(colours_path) as f:
                    colours = yaml.safe_load(f) or {}
            except OSError:
                colours = {}

            paint = labels == label_id
            mask_arr[paint, 0] = rgb[0]
            mask_arr[paint, 1] = rgb[1]
            mask_arr[paint, 2] = rgb[2]
            mask_arr[paint, 3] = 255
            colours[name] = rgb

            Image.fromarray(mask_arr).save(mask_path)
            with open(colours_path, 'w') as f:
                yaml.safe_dump(colours, f, allow_unicode=True)
        except Exception as exc:
            self.get_logger().warn(f'place_room: {exc!r}')
            self._state.broadcast({'type': 'room_saved', 'ok': False,
                                   'error': str(exc)})
            return

        self._room_mask_key = None
        if self._last_map is not None:
            self._cb_map(self._last_map)
        else:
            _, _, names = self._load_room_mask()
            self._state.map_rooms = {n: list(c) for n, c in (names or {}).items()}
            self._state.broadcast({'type': 'map_rooms', 'rooms': self._state.map_rooms})
        self.get_logger().info(f'room placed on {map_name}: {name} ({label_id})')
        self._state.broadcast({'type': 'room_saved', 'ok': True})

    def _place_room_rect(self, x1, y1, x2, y2, name, color):
        """Draw-a-rectangle version of _place_room, for spaces
        room_segment.segment's distance-transform watershed can't cleanly
        separate — open-plan areas with no doorway pinch point, where a
        single click either grabs the whole open area or nothing. Same
        two-opposite-corners convention as _add_keepout_zone, but paints
        <map>_room_mask.png pixels directly instead of adding a vector zone:
        rooms have no polygon format anywhere in this file (see
        room_files.py) — a flat pixel mask is what rendering/picking/the
        room list already read, so a raw rectangle rasterizes straight into
        it, no intersection with the flood-fill segmentation needed.

        Unlike _place_room, this does NOT stay inside one recognised blob —
        it is the deliberately blunt tool for when the room shape isn't one.
        It still skips occupied (wall) pixels within the rectangle so a
        painted room doesn't tint the walls bounding it.
        """
        try:
            name = str(name).strip()
            if not name:
                raise ValueError('όνομα δωματίου κενό')
            if not (isinstance(color, list) and len(color) == 3):
                raise ValueError('άκυρο χρώμα')
            rgb = [max(0, min(255, int(v))) for v in color]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            if abs(x2 - x1) < 0.10 or abs(y2 - y1) < 0.10:
                raise ValueError('πολύ μικρό τετράγωνο — μάλλον misclick')

            map_name = self.active_map()
            if not map_name:
                raise ValueError('κανένας αποθηκευμένος χάρτης ενεργός')

            yaml_path = os.path.join(SRC_MAPS_DIR, f'{map_name}.yaml')
            with open(yaml_path) as f:
                meta = yaml.safe_load(f)
            res = float(meta['resolution'])
            ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
            w_px, h_px = self._map_pgm_shape(map_name)

            def to_px(x, y):
                col = int((x - ox) / res)
                row = h_px - 1 - int((y - oy) / res)   # image-side up, same as _place_room
                return col, row

            c1, r1 = to_px(x1, y1)
            c2, r2 = to_px(x2, y2)
            c_lo, c_hi = sorted((c1, c2))
            r_lo, r_hi = sorted((r1, r2))
            c_lo, c_hi = max(0, c_lo), min(w_px, c_hi + 1)
            r_lo, r_hi = max(0, r_lo), min(h_px, r_hi + 1)
            if c_hi <= c_lo or r_hi <= r_lo:
                raise ValueError('εκτός χάρτη')

            from PIL import Image
            pgm_path = os.path.join(SRC_MAPS_DIR, f'{map_name}.pgm')
            gray = np.array(Image.open(pgm_path))
            if gray.ndim == 3:
                gray = gray[:, :, 0]
            not_wall = gray[r_lo:r_hi, c_lo:c_hi] > 50   # exclude occupied/near-black cells

            mask_path, colours_path = room_files.paths_for(map_name)
            mask_arr = None
            if os.path.exists(mask_path):
                mask_arr = np.array(Image.open(mask_path).convert('RGBA'))
                if mask_arr.shape[:2] != (h_px, w_px):
                    mask_arr = None    # stale mask from a different map size
            if mask_arr is None:
                mask_arr = np.zeros((h_px, w_px, 4), np.uint8)
            try:
                with open(colours_path) as f:
                    colours = yaml.safe_load(f) or {}
            except OSError:
                colours = {}

            region = mask_arr[r_lo:r_hi, c_lo:c_hi]
            region[not_wall, 0] = rgb[0]
            region[not_wall, 1] = rgb[1]
            region[not_wall, 2] = rgb[2]
            region[not_wall, 3] = 255
            colours[name] = rgb

            Image.fromarray(mask_arr).save(mask_path)
            with open(colours_path, 'w') as f:
                yaml.safe_dump(colours, f, allow_unicode=True)
        except Exception as exc:
            self.get_logger().warn(f'place_room_rect: {exc!r}')
            self._state.broadcast({'type': 'room_saved', 'ok': False,
                                   'error': str(exc)})
            return

        self._room_mask_key = None
        if self._last_map is not None:
            self._cb_map(self._last_map)
        else:
            _, _, names = self._load_room_mask()
            self._state.map_rooms = {n: list(c) for n, c in (names or {}).items()}
            self._state.broadcast({'type': 'map_rooms', 'rooms': self._state.map_rooms})
        self.get_logger().info(f'room rect placed on {map_name}: {name}')
        self._state.broadcast({'type': 'room_saved', 'ok': True})

    # ── keepout zones ────────────────────────────────────────────────────
    # Areas Nav2 must not enter — see home_robot/keepout_files.py for why the
    # zones are per-map (maps/<map>_keepout_zones.yaml) but the rendered mask
    # Nav2 actually reads is one fixed pair of files (config/keepout_mask.*):
    # filter_mask_server's yaml_filename is set once at LAUNCH time
    # (launch/bringup.launch.py), so there is no per-map path to point it at
    # without also making the launch args per-map, which nothing else needs.

    def _load_keepout_zones(self) -> dict:
        name = self.active_map()
        return keepout_files.load_zones(name) if name else {}

    def _map_pgm_shape(self, map_name: str):
        """(width, height) of a saved map's .pgm, straight from the file —
        independent of whatever's currently live on /map, so a zone can still
        be rasterized right after a map switch before the first /map arrives.
        """
        from PIL import Image
        with Image.open(os.path.join(SRC_MAPS_DIR, f'{map_name}.pgm')) as im:
            return im.size

    def _add_keepout_zone(self, x1, y1, x2, y2, name):
        """Two opposite corners (map frame) -> one rectangular zone, same
        convention as scripts/collect_keepout_clicks.py's RViz workflow.
        Threaded: active_map() + PIL/yaml I/O off the asyncio event loop —
        same reasoning as _save_rooms/_place_room.
        """
        try:
            name = str(name).strip()
            if not name:
                raise ValueError('όνομα ζώνης κενό')
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w < 0.10 or h < 0.10:
                raise ValueError('πολύ μικρή ζώνη — μάλλον misclick')

            map_name = self.active_map()
            if not map_name:
                raise ValueError('κανένας αποθηκευμένος χάρτης ενεργός')

            zones = keepout_files.load_zones(map_name)
            zones[name] = {'shape': 'rect',
                           'x': round((x1 + x2) / 2, 3),
                           'y': round((y1 + y2) / 2, 3),
                           'width': round(w, 2),
                           'height': round(h, 2)}
            keepout_files.save_zones(map_name, zones)
            keepout_files.render_active_mask(
                map_name, os.path.join(SRC_MAPS_DIR, f'{map_name}.yaml'),
                self._map_pgm_shape(map_name))
        except Exception as exc:
            self.get_logger().warn(f'add_keepout_zone: {exc!r}')
            self._state.broadcast({'type': 'keepout_saved', 'ok': False,
                                   'error': str(exc)})
            return
        self._state.map_keepout = zones
        self.get_logger().info(f'keepout zone added on {map_name}: {name}')
        self._state.broadcast({'type': 'keepout_zones', 'zones': zones})
        self._state.broadcast({'type': 'keepout_saved', 'ok': True})

    def _delete_keepout_zone(self, name):
        try:
            name = str(name).strip()
            map_name = self.active_map()
            if not map_name:
                raise ValueError('κανένας αποθηκευμένος χάρτης ενεργός')
            zones = keepout_files.load_zones(map_name)
            if name not in zones:
                raise ValueError(f'δεν βρέθηκε ζώνη "{name}"')
            del zones[name]
            keepout_files.save_zones(map_name, zones)
            keepout_files.render_active_mask(
                map_name, os.path.join(SRC_MAPS_DIR, f'{map_name}.yaml'),
                self._map_pgm_shape(map_name))
        except Exception as exc:
            self.get_logger().warn(f'delete_keepout_zone: {exc!r}')
            self._state.broadcast({'type': 'keepout_saved', 'ok': False,
                                   'error': str(exc)})
            return
        self._state.map_keepout = zones
        self.get_logger().info(f'keepout zone removed on {map_name}: {name}')
        self._state.broadcast({'type': 'keepout_zones', 'zones': zones})
        self._state.broadcast({'type': 'keepout_saved', 'ok': True})

    def _keepout_activate(self, on: bool):
        """Flip both costmaps' keepout_filter to enabled/disabled in
        nav2_params.yaml (home_robot/keepout_toggle.py — Nav2 reads costmap
        plugin config once at on_configure, no live toggle exists) and
        restart with/without use_keepout:=true, which starts/stops the mask
        servers the layer depends on. Same restart shape as _switch_backend/
        the /safety/skirt endpoint — scripts/apply_keepout_toggle.sh does the
        actual `robot stop && robot max ...`.
        """
        path = keepout_toggle.default_params_path()
        try:
            with open(path) as f:
                original = f.read()
            patched = keepout_toggle.patch_enabled(original, on)
        except (OSError, ValueError) as exc:
            self._state.broadcast({'type': 'keepout_activated', 'ok': False,
                                   'error': str(exc)})
            return
        if on:
            map_name = self.active_map()
            if not map_name:
                self._state.broadcast({
                    'type': 'keepout_activated', 'ok': False,
                    'error': 'κανένας αποθηκευμένος χάρτης ενεργός'})
                return
            try:
                keepout_files.render_active_mask(
                    map_name, os.path.join(SRC_MAPS_DIR, f'{map_name}.yaml'),
                    self._map_pgm_shape(map_name))
            except Exception as exc:
                self._state.broadcast({'type': 'keepout_activated', 'ok': False,
                                       'error': str(exc)})
                return
        try:
            with open(path, 'w') as f:
                f.write(patched)
        except OSError as exc:
            self._state.broadcast({'type': 'keepout_activated', 'ok': False,
                                   'error': str(exc)})
            return

        args = ['bash', KEEPOUT_APPLY_SH]
        if on:
            args.append('use_keepout:=true')
        if self.perception_on():
            args.append('use_perception:=true')
        backend = self.llm_backend()
        if backend and backend != 'lemonade':
            args.append(f'llm_backend:={backend}')
        try:
            subprocess.Popen(args, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._state.broadcast({'type': 'keepout_activated', 'ok': False,
                                   'error': str(exc)})
            return
        self._state.broadcast({'type': 'keepout_activated', 'ok': True, 'on': on})

    def _set_costmap_sub(self, on: bool):
        """Subscribe to the costmap + semantic-obstacle streams only while
        _costmap_on is non-empty — see the comment on it in __init__."""
        if on and self._costmap_sub is None:
            self._costmap_sub = self.create_subscription(
                OccupancyGrid, '/local_costmap/costmap', self._cb_costmap, 1)
            self._semantic_sub = self.create_subscription(
                PointCloud2, '/semantic_obstacles', self._cb_semantic,
                QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        elif not on and self._costmap_sub is not None:
            self.destroy_subscription(self._costmap_sub)
            self.destroy_subscription(self._semantic_sub)
            self._costmap_sub = None
            self._semantic_sub = None

    def _cb_semantic(self, msg: PointCloud2):
        """Obstacles the DETECTOR put into the costmap, not the LiDAR.

        semantic_costmap_node inflates people and animals into a cylinder wider
        than their actual footprint so Nav2 keeps a margin, and that is invisible
        in every other view: on the map the robot just appears to give someone a
        strangely wide berth. Only the outline is needed here, so the cloud is
        thinned hard — it is drawn as a scatter over a 3 m window.
        """
        if not self._costmap_on:
            return    # defensive only, see _set_costmap_sub
        try:
            pts = []
            step, n = msg.point_step, msg.width * msg.height
            data = msg.data
            stride = max(1, n // 400)
            for i in range(0, n, stride):
                off = i * step
                x, y = struct.unpack_from('<ff', data, off)
                if math.isfinite(x) and math.isfinite(y):
                    pts.append([round(x, 2), round(y, 2)])
            self._semantic_pts = pts
            self._semantic_time = time.monotonic()
        except (struct.error, IndexError):
            pass

    # Nav2 packs its costmap into an OccupancyGrid: 0 free, 1..98 the inflation
    # gradient, 99 inscribed (the robot's body would touch), 100 lethal, -1
    # unknown. Colouring the gradient is the point — "why did it not fit through
    # there" is answered by how far the blue bleeds out from the walls.
    def _cb_costmap(self, msg: OccupancyGrid):
        if not self._costmap_on:
            return    # defensive only, see _set_costmap_sub
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (24, 24, 27)                       # free space, matching the UI
        img[grid == -1] = (60, 60, 66)              # unknown
        mid = (grid > 0) & (grid < 99)
        if mid.any():
            # Cost 1..98 -> deep blue to cyan. Brighter = closer to the wall.
            g = grid[mid].astype(np.float32) / 98.0
            img[mid] = np.stack([
                (120 + 100 * g).astype(np.uint8),   # B
                (60 + 150 * g).astype(np.uint8),    # G
                np.full(g.shape, 30, np.uint8),     # R
            ], axis=-1)
        img[grid == 99]  = (0, 165, 255)            # inscribed
        img[grid == 100] = (40, 40, 220)            # lethal
        img = cv2.flip(img, 0)                      # ROS y-up -> image y-down
        _, png = cv2.imencode('.png', img)

        # Where the robot sits inside this window. The costmap rolls with the
        # robot but is NOT exactly centred, so taking the middle would put the
        # footprint a few centimetres off — enough to misread a near miss.
        rx = ry = None
        try:
            tf = self._tf_buffer.lookup_transform(
                msg.header.frame_id or 'odom', 'base_link', rclpy.time.Time())
            rx = (tf.transform.translation.x - msg.info.origin.position.x) / msg.info.resolution
            ry = (tf.transform.translation.y - msg.info.origin.position.y) / msg.info.resolution
            ry = h - ry                             # same flip as the image
        except Exception:
            pass

        self._state.broadcast({
            'type': 'costmap',
            'width': w, 'height': h,
            'resolution': round(msg.info.resolution, 4),
            'robot': [round(rx, 1), round(ry, 1)] if rx is not None else None,
            'semantic': (self._semantic_pts
                         if time.monotonic() - self._semantic_time < 2.0 else []),
            'image': base64.b64encode(png.tobytes()).decode(),
        }, remember=False)

    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        p   = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        c   = msg.pose.covariance
        self._fz_cov['amcl'] = (c[0], c[7], c[35])
        self._publish_pose(p.position.x, p.position.y, yaw)

    def _publish_pose(self, x: float, y: float, yaw: float):
        # Kept for compass calibration: the button says "I am facing north",
        # and the server needs the yaw that statement refers to.
        self._last_yaw = yaw
        self._state.broadcast({
            'type': 'pose',
            'x':   round(x, 3),
            'y':   round(y, 3),
            'yaw': round(yaw, 4),
        })

    def _poll_tf_pose(self):
        """Derive the pose from map->base_link when AMCL's topic stays quiet.

        ‼️ /amcl_pose is NOT a position feed — AMCL publishes it only when it
        updates its estimate, which it does only while the robot drives. Open
        the dashboard on a robot standing still (the normal case) and that topic
        never fires, so the map tab drew the map and nothing else: no robot
        marker, no laser (the scan is drawn in the robot's frame, so no pose
        means no dots), and X/Y/Γωνία stuck on '—'. It read like a dead
        localization stack while `tf2_echo map base_link` was answering fine.

        room_markers_node hit exactly this and grew the same 2 Hz TF poll
        (fc6f1cb); the dashboard was left on the topic alone. TF is the same
        estimate AMCL would have published, just always current.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return   # not localized yet — drawing no robot is correct
        t, r = tf.transform.translation, tf.transform.rotation
        self._publish_pose(t.x, t.y, 2.0 * math.atan2(r.z, r.w))

    # Shared angular grid for the LiDAR/camera comparison: 120 bins of 3° over
    # the full circle. Coarse on purpose — the two sensors are 7 cm apart and
    # the depth cone is noisy at the edges, so finer bins compare noise.
    PROF_BINS = 120

    def _profile(self, xy) -> list:
        """Nearest return per 3° bin, from Nx2 base_link points. None = nothing.

        Both sensors are reduced to the same shape so the browser can subtract
        them: whatever is closest in a given direction is what the robot would
        hit going that way, whichever sensor saw it.
        """
        out = [None] * self.PROF_BINS
        if not len(xy):
            return out
        rng = np.hypot(xy[:, 0], xy[:, 1])
        good = (rng > 0.05) & (rng < 8.0) & np.isfinite(rng)
        if not good.any():
            return out
        rng = rng[good]
        bear = np.arctan2(xy[good, 1], xy[good, 0])
        idx = ((bear + math.pi) / (2 * math.pi) * self.PROF_BINS).astype(int)
        np.clip(idx, 0, self.PROF_BINS - 1, out=idx)
        # Sort by range descending so the last write into each bin — which is
        # the one that sticks — is the nearest point in it.
        order = np.argsort(-rng)
        near = np.full(self.PROF_BINS, np.nan)
        near[idx[order]] = rng[order]
        return [None if math.isnan(v) else round(float(v), 3) for v in near]

    def _tf_matrix(self, target: str, source: str):
        """4x4 as (R, t), or None while the transform is missing."""
        try:
            tf = self._tf_buffer.lookup_transform(target, source,
                                                  rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException):
            return None
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        t = tf.transform.translation
        return R, np.array([t.x, t.y, t.z], dtype=np.float64)

    def _cb_scan(self, msg: LaserScan):
        if self._fuse_ws:
            self._scan_profile(msg)
        self._scan_seq += 1
        if self._scan_seq % 3:          # send every 3rd scan (~3 Hz)
            return
        ranges = [round(r, 3) if math.isfinite(r) else 0.0
                  for r in msg.ranges[::3]]   # decimate 3× for bandwidth
        self._state.broadcast({
            'type':      'scan',
            'ranges':    ranges,
            'angle_min': round(msg.angle_min, 6),
            'angle_inc': round(msg.angle_increment * 3, 6),
        })

    def _scan_profile(self, msg: LaserScan):
        """/scan -> base_link bearings, for the sensor-comparison tab.

        ‼️ Not reusable from the map tab's copy: that one hands the browser raw
        `laser`-frame angles and lets draw() add the mount yaw. The C1 is
        mounted backwards (laser->base_link is a 180° yaw, measured), so
        comparing those angles against the camera's would put every disagreement
        on the wrong side of the robot. Transformed properly here instead of
        hardcoding the π, so a remount fixes itself.
        """
        if self._laser_tf is None:
            self._laser_tf = self._tf_matrix('base_link', msg.header.frame_id
                                             or 'laser')
            if self._laser_tf is None:
                return
        R, t = self._laser_tf
        r = np.asarray(msg.ranges, dtype=np.float64)
        if not r.size:
            return
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment
        ok = np.isfinite(r) & (r >= max(0.01, msg.range_min)) & (r <= msg.range_max)
        if not ok.any():
            return
        r, a = r[ok], a[ok]
        pts = np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1)
        base = pts @ R.T + t
        self._prof_lidar = self._profile(base[:, :2])

    def _cb_plan(self, msg: Path):
        # Nav2 republishes the global plan at controller rate; 40 points is
        # plenty to draw a readable line and keeps the socket quiet.
        pts = msg.poses[::max(1, len(msg.poses) // 40)] if msg.poses else []
        self._state.broadcast({
            'type': 'plan',
            'points': [[round(p.pose.position.x, 2), round(p.pose.position.y, 2)]
                       for p in pts],
        })

    @staticmethod
    def _yaw_of(q) -> float:
        """Yaw from a quaternion, the flat-robot case (two_d_mode)."""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _fz_feed(self, key: str, yaw: float):
        """Record one heading sample, unwrapped.

        Wrapped yaw is useless for comparing sources: two headings 1° apart
        read as 179.5 and -179.5 whenever the robot happens to face that way,
        and the difference comes out as 359°. Counting turns keeps the three
        curves continuous so a drift of a few degrees stays a few degrees.
        """
        s = self._fz[key]
        if s['raw'] is not None:
            d = yaw - s['raw']
            if d > math.pi:
                s['turns'] -= 2 * math.pi
            elif d < -math.pi:
                s['turns'] += 2 * math.pi
        s['raw'] = yaw
        s['yaw'] = yaw + s['turns']
        s['n'] += 1
        s['seen'] = time.monotonic()
        if s['ref'] is None:
            s['ref'] = s['yaw']

    def _cb_odom_filtered(self, msg: Odometry):
        """The EKF's own answer — what actually drives the odom->base_link TF."""
        self._fz_feed('ekf', self._yaw_of(msg.pose.pose.orientation))
        self._fz_vx['ekf'] = msg.twist.twist.linear.x
        self._fz_wz['ekf'] = msg.twist.twist.angular.z
        c = msg.pose.covariance
        # Diagonal only: x, y, yaw variances (indices 0, 7, 35 of the 6x6).
        self._fz_cov['ekf'] = (c[0], c[7], c[35])

    def _cb_odom(self, msg: Odometry):
        wz = msg.twist.twist.angular.z
        self._fz_feed('wheel', self._yaw_of(msg.pose.pose.orientation))
        self._fz_vx['wheel'] = msg.twist.twist.linear.x
        self._fz_wz['wheel'] = wz
        # Well clear of the 879's ~0.31 rad/s rotation floor, so this is only
        # true when the wheels are genuinely turning the robot.
        self._turning = abs(wz) > 0.15
        self._state.broadcast({
            'type': 'odom',
            'vx': round(msg.twist.twist.linear.x, 3),
            'wz': round(wz, 3),
        })

    def _cb_imu(self, msg: Imu):
        """BNO085 -> the IMU tab: attitude, turn rate, and honest health.

        ‼️ Two things this panel must NOT pretend about, both by design in
        bno085_imu.ino:
          - The heading is a GAME rotation vector: gyro+accel fusion with the
            magnetometer deliberately left out (indoors the Roomba's DC motors
            wrecked it). So yaw=0 is an arbitrary direction chosen at each boot
            — a *relative* compass, never magnetic north.
          - ax/ay/az are streamed as literal 0.0: SH2_LINEAR_ACCELERATION is
            not enabled, because imu0_config leaves accel out of the EKF. The
            panel says "not streamed" rather than drawing three zeroes as if
            the robot were perfectly still.
        """
        now = time.monotonic()
        self._imu_n += 1
        self._imu_seen = now

        q = msg.orientation
        # ZYX Euler from the quaternion. atan2/asin rather than a matrix so a
        # denormalised quaternion off a garbled serial line cannot raise.
        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)
        sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
        pitch = math.asin(sinp)
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        g = msg.angular_velocity
        a = msg.linear_acceleration
        # Fed at the full stream rate, not the throttled 5 Hz below: the fusion
        # tab's rate counter has to measure the IMU, not this panel's throttle.
        self._fz_feed('imu', yaw)
        self._fz_wz['imu'] = g.z
        # The firmware's own failure mode: enabling several SH2 reports
        # back-to-back silently drops some over the flaky I2C bus, and the
        # symptom is a gyro pinned at exactly 0 while the quaternion keeps
        # updating — the EKF then fights every turn with a constant 0 yaw rate.
        # Only counted while the wheels are actually turning: at rest exact
        # zeros are normal quantisation, not a fault.
        if g.x == 0.0 and g.y == 0.0 and g.z == 0.0:
            if self._turning:
                self._gyro_zero += 1
        else:
            self._gyro_zero = 0

        if now - self._imu_last < 0.2:   # 5 Hz is plenty for a phone
            return
        self._imu_last = now

        self._state.broadcast({
            'type':  'imu',
            'roll':  round(math.degrees(roll), 1),
            'pitch': round(math.degrees(pitch), 1),
            'yaw':   round(math.degrees(yaw), 1),
            'quat':  [round(q.w, 4), round(q.x, 4), round(q.y, 4), round(q.z, 4)],
            'gx':    round(g.x, 4), 'gy': round(g.y, 4), 'gz': round(g.z, 4),
            'ax':    round(a.x, 3), 'ay': round(a.y, 3), 'az': round(a.z, 3),
            'hz':    round(self._imu_hz, 1),
            # ~1 s of turning (the stream runs ~19 Hz) with no yaw rate at all.
            'gyro_dead': self._gyro_zero > 20,
            'turning': self._turning,
            'alive': True,
        })

    def _imu_health(self):
        """Measured rate + the silent-IMU alarm."""
        now = time.monotonic()
        dt = now - self._imu_t0
        if dt > 0:
            self._imu_hz = self._imu_n / dt
        self._imu_n, self._imu_t0 = 0, now
        # Nothing for 2 s. imu_node reopens the serial port on error, so this
        # covers both a wedged BNO085 and a dead imu_node.
        if now - self._imu_seen > 2.0:
            self._state.broadcast({'type': 'imu', 'alive': False, 'hz': 0.0})

    # ── Sensor fusion ──────────────────────────────────────────────────────
    def _fusion_reset(self):
        """Re-zero the three headings against each other, and drop the peaks.

        The absolute values are meaningless (the BNO085's zero is wherever it
        booted, the wheels' zero is wherever the base powered up), so the panel
        only ever shows how far each source has drifted from the others SINCE
        the reset. Pressing it while the robot stands still is the calibration.
        """
        for s in self._fz.values():
            s['ref'] = s['yaw']
        self._fz_corr_ref = None
        self._fz_corr_max = 0.0
        self._fz_yaw_max  = 0.0

    def _broadcast_fusion(self):
        """The EKF's inputs, its output, and what AMCL had to correct.

        Runs at 4 Hz off a timer rather than off any one callback, because the
        interesting failure is a source that STOPPED: a callback-driven panel
        goes quiet exactly when it has something to report.
        """
        if not self._fuse_ws:
            return
        now = time.monotonic()

        dt = now - self._fz_t0
        if dt >= 1.0:                       # measured rates, once a second
            for s in self._fz.values():
                s['hz'] = s['n'] / dt
                s['n']  = 0
            self._fz_t0 = now

        src = {}
        for k, s in self._fz.items():
            rel = (None if s['yaw'] is None or s['ref'] is None
                   else math.degrees(s['yaw'] - s['ref']))
            src[k] = {
                'yaw': None if rel is None else round(rel, 2),
                'hz':  round(s['hz'], 1),
                # No sample ever seen reads as "dead", not as "0.0 s old".
                'age': None if not s['seen'] else round(now - s['seen'], 1),
            }

        def diff(a, b):
            ya, yb = src[a]['yaw'], src[b]['yaw']
            return None if ya is None or yb is None else round(ya - yb, 2)

        # How far AMCL has had to shift the odom frame SINCE the reset: exactly
        # the error the fusion accumulated while you were watching, which is
        # invisible on the map tab (there the robot always looks correctly
        # placed — that is what the correction is for).
        corr = {'ok': False}
        try:
            tf = self._tf_buffer.lookup_transform('map', 'odom',
                                                  rclpy.time.Time())
            t, r = tf.transform.translation, tf.transform.rotation
            yaw = self._yaw_of(r)
            if self._fz_corr_ref is None:
                self._fz_corr_ref = (t.x, t.y, yaw)
            rx, ry, ryaw = self._fz_corr_ref
            d   = math.hypot(t.x - rx, t.y - ry) * 100.0          # cm
            # Shortest way round, or a correction across ±180° reads as 359°.
            cy  = abs(math.degrees(math.atan2(math.sin(yaw - ryaw),
                                              math.cos(yaw - ryaw))))
            self._fz_corr_max = max(self._fz_corr_max, d)
            self._fz_yaw_max  = max(self._fz_yaw_max, cy)
            corr = {'ok': True, 'd': round(d, 1), 'yaw': round(cy, 2),
                    'dmax': round(self._fz_corr_max, 1),
                    'yawmax': round(self._fz_yaw_max, 2)}
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass                # not localized — no correction exists to show

        def sigma(c):
            # Variance -> standard deviation, in units a human reads: the raw
            # numbers are m^2 and rad^2, where a perfectly healthy 0.0004 and
            # an alarming 0.25 look equally like noise.
            if c is None:
                return None
            return [round(math.sqrt(max(0.0, c[0])) * 100, 1),
                    round(math.sqrt(max(0.0, c[1])) * 100, 1),
                    round(math.degrees(math.sqrt(max(0.0, c[2]))), 1)]

        self._state.broadcast({
            'type': 'fusion',
            'src':  src,
            'dyaw': {'wheel': diff('wheel', 'ekf'), 'imu': diff('imu', 'ekf')},
            'vx':   {k: round(v, 3) for k, v in self._fz_vx.items()},
            'wz':   {k: round(v, 3) for k, v in self._fz_wz.items()},
            'cov':  {'ekf': sigma(self._fz_cov['ekf']),
                     'amcl': sigma(self._fz_cov['amcl'])},
            'corr': corr,
            # Same qualification as the IMU tab's gyro alarm: a difference that
            # only appears while the wheels turn is a real fusion fault, one at
            # rest is quantisation.
            'turning': self._turning,
        }, remember=False)

        # The two profiles ride the same timer at half the rate: 120 bins twice
        # over is ~1.5 kB, and the shapes do not change fast enough to be worth
        # 4 Hz on a phone.
        if self._fuse_cam_ws and now - self._prof_last >= 0.4:
            self._prof_last = now
            cam_age = (None if not self._prof_cam_at
                       else round(now - self._prof_cam_at, 1))
            self._state.broadcast({
                'type':  'fuseprof',
                'bins':  self.PROF_BINS,
                'lidar': self._prof_lidar,
                'cam':   self._prof_cam,
                # Distinguishes "the camera agrees with the LiDAR everywhere"
                # from "the camera stopped publishing", which look identical
                # once a stale profile is drawn.
                'cam_age': cam_age,
                'zmin': self.CAM_Z_MIN, 'zmax': self.CAM_Z_MAX,
            }, remember=False)

    # What the picture is, not just whether there is one. Measured on this
    # camera 2026-08-05 from a robot parked 40 cm from a white wall: brightness
    # 116.7, a perfectly normal exposure. The frame was healthy by every
    # measure the dashboard had and looked to a human exactly like a dead
    # camera — "δεν δείχνει" was reported twice for a stream running at 23 fps.
    # Texture is what tells them apart.
    #
    # ‼️ Laplacian variance is RESOLUTION-DEPENDENT, and calibrating it at one
    # size while measuring at another is silently wrong. The same five live
    # frames of that wall:
    #
    #     640x480 -> 6.0      320x240 -> 21.3      160x120 -> 73.3
    #
    # The first attempt took the 640-wide number and applied it to a 160-wide
    # copy, so a blank wall scored 73 against a threshold of 40 and the banner
    # never appeared. Both numbers below are measured at JUDGE_SIZE.
    # For scale, a richly textured reference scores ~4160 at every size, so 60
    # sits an order of magnitude clear of anything with real detail in it.
    JUDGE_SIZE    = (320, 240)
    FLAT_DETAIL   = 60.0     # wall measured 21.3 here
    DARK_MEAN     = 18.0
    BLOWN_MEAN    = 238.0
    _CAM_JUDGE_PERIOD = 1.0  # seconds; the verdict cannot change faster

    def _judge_frame(self, bgr):
        """Say WHY the picture looks empty, once a second.

        Cheap on purpose: one grayscale copy at JUDGE_SIZE, once a second,
        which costs nothing next to the JPEG encode that just ran.
        """
        now = time.monotonic()
        if now - self._cam_judge_at < self._CAM_JUDGE_PERIOD:
            return
        self._cam_judge_at = now
        try:
            small = cv2.cvtColor(cv2.resize(bgr, self.JUDGE_SIZE),
                                 cv2.COLOR_BGR2GRAY)
            mean = float(small.mean())
            detail = float(cv2.Laplacian(small, cv2.CV_64F).var())
        except cv2.error:
            return

        # Only the verdict crosses the wire, never its wording: the sentence
        # lives in the page, where the translation table can reach it. A Greek
        # string built here would render in Greek in all three languages.
        if mean < self.DARK_MEAN:
            state = 'dark'
        elif mean > self.BLOWN_MEAN:
            state = 'blown'
        elif detail < self.FLAT_DETAIL:
            state = 'flat'
        else:
            state = 'ok'
        payload = {'type': 'camstate', 'state': state,
                   'mean': round(mean, 1), 'detail': round(detail, 1)}
        if payload != self._cam_state_last:
            self._cam_state_last = payload
            self._state.broadcast(payload)

    def _set_camera_sub(self, on: bool):
        """Subscribe to the raw color stream only while _cam_ws is non-empty.

        See the comment on _cam_ws in __init__: an early-return inside the
        callback still pays for DDS deserialising every full-resolution color
        frame (~30 Hz) before the callback body runs at all — measured ~13
        points of a core. Creating/destroying the subscription itself is the
        only way to actually stop paying it.
        """
        if on and self._camera_sub is None:
            self._camera_sub = self.create_subscription(
                Image, '/camera/camera/color/image_raw', self._cb_camera, 5)
        elif not on and self._camera_sub is not None:
            self.destroy_subscription(self._camera_sub)
            self._camera_sub = None

    def _cb_camera(self, msg: Image):
        # Defensive only — _set_camera_sub means this should never fire while
        # _cam_ws is empty, but destroy_subscription() does not guarantee an
        # in-flight callback is cancelled.
        if not self._cam_ws:
            return
        try:
            enc = msg.encoding.lower()
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if enc == 'rgb8' else cv2.COLOR_RGBA2BGR if 'rgba' in enc else cv2.COLOR_BGRA2BGR if 'bgra' in enc else -1) \
                  if enc != 'bgr8' else arr
            if bgr is None or not isinstance(bgr, np.ndarray):
                return
            src_w = bgr.shape[1]
            if bgr.shape[1] > 640:
                scale = 640 / bgr.shape[1]
                bgr = cv2.resize(bgr, (640, int(bgr.shape[0] * scale)))
            if self._overlay:
                # Drawn server-side rather than as a canvas over the <img>. The
                # stream is a single MJPEG whose frames the browser never times,
                # so a client-side overlay would drift against it — boxes
                # trailing the person by however long the last frame took. Here
                # they are burned into the same pixels they describe, and the
                # phone gets it for free.
                self._draw_overlay(bgr, src_w)
            _, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self._state.camera_jpg = jpg.tobytes()
            self._state.camera_at = time.monotonic()
            self._state.camera_seq += 1
            # Pushed over the same /ws binary channel as the mic audio — see
            # the 2026-08-14 note on camSetActive client-side: Safari's <img>
            # does not decode multipart/x-mixed-replace (only a top-level
            # navigation to /camera.mjpeg does), so the HTTP MJPEG stream
            # rendered black there despite being perfectly healthy. The
            # websocket is already proven to reach this exact browser (mic
            # audio, pointcloud), so frames ride it instead. JPEG bytes always
            # start with FFD8 (SOI), which is how the client tells a camera
            # frame apart from a PCM audio chunk on the one shared binary
            # channel without a wire-format change to the (working) audio path.
            if self._cam_ws:
                self._state.send_bytes(self._cam_ws, self._state.camera_jpg)
            self._judge_frame(bgr)
        except Exception as e:
            # Was a bare `pass`. A frame this callback cannot decode stops the
            # picture updating for ever, and the silence made that look like a
            # dead camera rather than a dashboard bug. Throttled because a bad
            # encoding repeats at the full frame rate.
            now = time.monotonic()
            if now - self._cam_err_at > 10.0:
                self._cam_err_at = now
                self.get_logger().warn(f'camera frame dropped: {e}')

    # COCO-17 skeleton, the same edge list pose_node.py uses. Duplicated rather
    # than imported: pose_node pulls in torch and ultralytics at import time,
    # and the dashboard must start whether or not perception is even installed.
    _KP_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    # Detections older than this are not drawn. Generous enough for a detector
    # running at a few Hz, short enough that a stopped detector leaves the
    # picture clean instead of freezing a box on screen for ever.
    _OVERLAY_MAX_AGE = 1.5

    def _draw_overlay(self, bgr, src_w: int):
        """Burn YOLO boxes and COCO skeletons into the camera frame."""
        now = time.monotonic()
        h, w = bgr.shape[:2]

        # ONE factor for both axes. The resize above is aspect-preserving
        # (640, height*scale), so x and y share a ratio — deriving them
        # separately from img_w/img_h invites an off-by-a-few-percent stretch
        # that looks like a mis-calibrated camera rather than a bug here.
        def sc(v, iw):
            return int(v * w / float(iw or w))

        if self._det_boxes and now - self._det_time < self._OVERLAY_MAX_AGE:
            for d in self._det_boxes:
                iw = d.get('img_w') or src_w
                x1, y1 = sc(d.get('x1', 0), iw), sc(d.get('y1', 0), iw)
                x2, y2 = sc(d.get('x2', 0), iw), sc(d.get('y2', 0), iw)
                person = d.get('label') == 'person'
                # ‼️ BGR, not RGB — cv2's order. (255,170,60) reads as orange
                # until you remember that, and renders as pale BLUE, which is
                # what the first live capture showed while the caption under the
                # video promised orange.
                colour = (80, 220, 80) if person else (60, 170, 255)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), colour, 2)
                dist = d.get('box_distance')
                label = f"{d.get('label', '?')} {d.get('conf', 0):.2f}"
                if dist is not None:
                    label += f'  {dist:.1f}m'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                ty = max(0, y1 - th - 4)
                # Filled plate behind the text: white-on-white is unreadable, and
                # a living room is mostly pale walls.
                cv2.rectangle(bgr, (x1, ty), (x1 + tw + 6, ty + th + 4), colour, -1)
                cv2.putText(bgr, label, (x1 + 3, ty + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        if self._det_poses and now - self._pose_time < self._OVERLAY_MAX_AGE:
            for p in self._det_poses:
                kps = p.get('keypoints') or []
                if len(kps) < 17:
                    continue
                # pose_detections carries no img_w/img_h, so the width of the
                # frame this callback is holding is the only reference there is.
                pts = [(sc(k.get('x', 0), src_w), sc(k.get('y', 0), src_w),
                        k.get('v', 0.0)) for k in kps]
                for a, b in self._KP_EDGES:
                    xa, ya, va = pts[a]
                    xb, yb, vb = pts[b]
                    if va < 0.5 or vb < 0.5:
                        continue
                    cv2.line(bgr, (xa, ya), (xb, yb), (255, 210, 0), 2, cv2.LINE_AA)
                for x, y, v in pts:
                    if v >= 0.5:
                        cv2.circle(bgr, (x, y), 3, (0, 90, 255), -1, cv2.LINE_AA)

    def _cb_objects(self, msg: String):
        self._state.broadcast({'type': 'objects', 'text': msg.data})
        try:
            dets = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        if not isinstance(dets, list):
            return
        self._det_boxes = [d for d in dets if isinstance(d, dict) and 'x1' in d]
        self._det_time  = time.monotonic()
        # Compact summary for the camera tab's counter, so "is it seeing
        # anything" is answerable without reading the raw JSON dump.
        self._state.broadcast({
            'type': 'vision',
            'objects': len(self._det_boxes),
            'people': sum(1 for d in self._det_boxes if d.get('label') == 'person'),
        })

    def _cb_poses(self, msg: String):
        try:
            poses = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        # pose_node now sends {'persons': [...], 'width', 'height'}; it used to
        # send a bare list. Both are accepted so the overlay survives a stale
        # installed copy of either node.
        if isinstance(poses, dict):
            self._pose_src_w = poses.get('width') or 0
            poses = poses.get('persons') or []
        if isinstance(poses, list):
            self._det_poses = [p for p in poses if isinstance(p, dict)]
            self._pose_time = time.monotonic()

    def _cb_room(self, msg: String):
        self._state.broadcast({'type': 'room', 'name': msg.data})

    def _cancel_navigation(self):
        """Stop going wherever it was going — every layer, not just the wheels.

        There are five separate things that can be driving the robot, and the
        old button (one empty Twist) addressed none of them:

          * bt_navigator owns the NavigateToPose goal. An empty CancelGoal
            request — zero id, zero stamp — cancels every goal on the server;
            the dashboard never sent the goal itself, so it has no handle.
          * mission_executor cancels on an empty `mission/start`.
          * task_planner cancels on `patrol_command=False`.
          * explore_lite keeps its own frontier goals coming.
          * person_follower drives directly.

        And recovery_manager re-issues the goal it read off /plan after every
        escape, which is why a cancel that DID land came back a few seconds
        later. /mission/cancel is what tells it this one was a human's decision.

        The zero Twist goes last: it is the immediate stop, and sending it
        before the owners are told just gets overwritten on their next tick.
        """
        if self._nav_cancel_cli.service_is_ready():
            self._nav_cancel_cli.call_async(CancelGoal.Request())
        else:
            self.get_logger().warn(
                'navigate_to_pose cancel service not up — cancelling the rest anyway')
        self._cancel_pub.publish(String(data='{"source":"dashboard"}'))
        self._mission_pub.publish(String(data=''))
        self._patrol_pub.publish(Bool(data=False))
        self._explore_pub.publish(Bool(data=False))
        self._follow_pub.publish(Bool(data=False))
        self._vel_pub.publish(Twist())
        self.get_logger().info('Navigation cancelled from the dashboard')

    # Roughly what a phone on wifi can take: 4000 points x 9 bytes is ~36 kB,
    # ~48 kB once base64'd, three times a second.
    CLOUD_MAX_POINTS = 4000
    CLOUD_PERIOD = 0.33

    def _probe_cloud_owner(self):
        """Find out once whether the launch already enabled the pointcloud.

        Retries on its own timer rather than assuming: the camera takes several
        seconds to advertise its parameter services, and a single early attempt
        would fail, leave the dashboard believing it owns the parameter, and
        switch the costmap's depth source off the first time someone closed the
        3D tab.
        """
        if not self._cam_param_get.service_is_ready():
            return                      # camera not up yet — try again in 3 s

        def done(fut):
            try:
                vals = fut.result().values
            except Exception:
                return                  # leave the timer running, try again
            if vals and vals[0].bool_value:
                self._cloud_param_own = False
                self._cloud_param_on  = True
                self.get_logger().info(
                    'D435 pointcloud is already on (Nav2 costmap owns it) — '
                    'the 3D tab will use it without switching it off')
            self._cam_probe_timer.cancel()

        req = GetParameters.Request()
        req.names = ['pointcloud.enable']
        self._cam_param_get.call_async(req).add_done_callback(done)

    def _set_camera_pointcloud(self, on: bool):
        """Turn the D435's pointcloud filter on while the 3D tab is watching.

        ‼️ Without this the 3D tab could never show anything. localize.launch.py
        starts the lean depth-only stream with pointcloud.enable:=false (it costs
        CPU and nothing else subscribes), so /camera/camera/depth/color/points
        was advertised but never published — the tab sat on "αναμονή για νέφος
        σημείων…" forever, on every default `robot max`. Confirmed live
        2026-08-02, and confirmed that flipping the parameter at runtime brings
        the topic up at ~18 Hz.

        Enabling it only while someone is looking keeps the default cost at zero,
        which is the same reason the stream itself is gated on _cloud_ws.
        """
        if not self._cloud_param_own:
            return          # Nav2 owns it; see the note where the flag is set
        if on == self._cloud_param_on:
            return
        if not self._cam_param_cli.service_is_ready():
            return          # camera not up (or not the realsense) — nothing to do
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name='pointcloud.enable',
            value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=on),
        )]
        self._cam_param_cli.call_async(req)
        self._cloud_param_on = on
        self.get_logger().info(f'D435 pointcloud {"on" if on else "off"} (3D tab)')

    def _cb_cloud(self, msg: PointCloud2):
        # Two consumers now: the 3D tab wants the coloured points, the sensor
        # tab wants only a top-down profile. Both come off this one decode.
        if not (self._cloud_ws or self._fuse_cam_ws):
            return
        now = time.time()
        if now - self._cloud_last < self.CLOUD_PERIOD:
            return
        self._cloud_last = now

        n = msg.width * msg.height
        if n == 0 or msg.point_step < 20:
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        raw = raw[:n * msg.point_step].reshape(n, msg.point_step)
        # Ceiling, not floor: n // MAX rounds the stride DOWN, which lets the
        # result overshoot the cap by up to 25% (148815 points came out as
        # 4023, not 4000). The cap is a bandwidth budget, so it has to hold.
        stride = max(1, -(-n // self.CLOUD_MAX_POINTS))
        raw = raw[::stride]

        xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3)
        # rgb is a float32 whose BITS are 0x00RRGGBB — reading it as a number
        # gives garbage, so take the bytes. Little-endian, so B G R _.
        bgr = raw[:, 16:19]

        good = np.isfinite(xyz).all(axis=1)
        xyz, bgr = xyz[good], bgr[good]
        if not len(xyz):
            return

        if self._fuse_cam_ws:
            self._cam_profile(xyz, msg.header.frame_id)

        if not self._cloud_ws:
            return
        # Millimetres in int16 covers +-32 m; the D435 stops at 10.
        mm = np.clip(xyz * 1000.0, -32000, 32000).astype('<i2')
        rgb = bgr[:, ::-1].copy()          # BGR -> RGB for the browser
        payload = np.concatenate([mm.view(np.uint8).reshape(len(mm), 6), rgb],
                                 axis=1).tobytes()
        self._state.broadcast({
            'type': 'cloud', 'n': len(mm), 'frame': msg.header.frame_id,
            'total': n, 'data': base64.b64encode(payload).decode(),
        }, remember=False)

    # What counts as an obstacle for the comparison, in metres above the floor.
    # The floor itself has to go: the D435 looks down from 0.536 m, so most of
    # what it returns is carpet, and carpet in a min-range profile reads as a
    # wall half a metre in front of the robot. The ceiling goes for the same
    # reason on the way up. What is left is the band the robot can hit.
    CAM_Z_MIN = 0.06
    CAM_Z_MAX = 1.40

    def _cam_profile(self, xyz, frame: str):
        """D435 points -> the same 3° bearing profile the LiDAR produces."""
        if self._cam_tf is None:
            self._cam_tf = self._tf_matrix(
                'base_link', frame or 'camera_depth_optical_frame')
            if self._cam_tf is None:
                return
        R, t = self._cam_tf
        base = xyz.astype(np.float64) @ R.T + t
        band = base[(base[:, 2] > self.CAM_Z_MIN) & (base[:, 2] < self.CAM_Z_MAX)]
        self._prof_cam    = self._profile(band[:, :2])
        self._prof_cam_at = time.monotonic()

    def _cb_rosout(self, msg: RosoutLog):
        # WARN(30) and up only — see State.logs. Nav2's costmaps alone publish
        # INFO faster than a browser can render it.
        if msg.level < RosoutLog.WARN:
            return
        self._state.add_log(int(msg.level), msg.name, msg.msg)

    def _cb_arm(self, msg: JointState):
        # arm_driver publishes at 10 Hz; halve it for the browser.
        self._arm_seq += 1
        if self._arm_seq % 2:
            return
        self._state.broadcast({
            'type': 'arm',
            'names': list(msg.name),
            'pos':   [round(p, 4) for p in msg.position],
        })

    def _cb_roomba(self, msg: String):
        try:
            self._state.broadcast({'type': 'roomba', **json.loads(msg.data)})
        except (ValueError, TypeError):
            pass

    def _cb_rtab_info(self, msg):
        """One per processed frame — the liveness signal for the 3D map tab."""
        self._rtab_seen = time.monotonic()
        # loop_closure_id is non-zero only on the frame that closed a loop, so
        # it has to be counted as it goes by, not sampled.
        if msg.loop_closure_id:
            self._rtab_loops += 1

    def _cb_rtab_graph(self, msg):
        self._rtab_nodes = len(msg.poses)

    def _rtab_health(self):
        """Whether a mapping session is actually running, and how it is doing.

        Worth its own strip because the tab can look perfectly alive — VNC up,
        rtabmap_viz painted — while the mapper is receiving nothing at all: the
        camera's aligned depth is off, or the RGB-D pair never syncs. The
        keyframe count going up is the only honest 'it is working'.
        """
        if not self._rtab_ok:
            return
        live = (time.monotonic() - self._rtab_seen) < 3.0 if self._rtab_seen else False
        self._state.broadcast({
            'type':  'rtabmap',
            'live':  live,
            'nodes': self._rtab_nodes if live else 0,
            'loops': self._rtab_loops if live else 0,
        })

    def _cb_nerf(self, msg: String):
        try:
            self._state.broadcast({'type': 'nerf', **json.loads(msg.data)})
        except (ValueError, TypeError):
            pass

    # ── NeRF training, launched from the web tab instead of a terminal ─────
    def _nerf_stop_perception(self):
        """Kill just the GPU-holding perception nodes — see NERF_GPU_PROCS.
        Nothing else in the stack (nav2, drivers, this dashboard) is touched,
        so the robot keeps driving/e-stopping normally; it just goes blind
        until the next `robot max`, same trade a live 'robot stop' makes but
        without losing localisation/dashboard/voice too.
        """
        for name in NERF_GPU_PROCS:
            subprocess.run(['pkill', '-f', name])
        self._state.broadcast({'type': 'nerf_train', **self._state.latest.get(
            'nerf_train', {'running': False, 'done': False, 'error': None}),
            'perception_stopped': True}, remember=False)

    # ── Pose / AprilTag — on-demand from the Κάμερα tab ─────────────────────
    # Both ride their own use_X launch flag by default (see bringup.launch.py /
    # localize.launch.py) so `robot max` still starts them — this only adds a
    # LIVE toggle on top, for the case where they were on at boot but are not
    # needed right now (pose+gesture ~25% CPU, apriltag ~29%, measured
    # 2026-08-14). Same fire-and-forget `subprocess.Popen`/`pkill -f` idiom as
    # `_nerf_stop_perception` above and `_switch_backend` below — status is
    # never tracked here, the client reads it straight off the `sys` message's
    # `nodes` list (already broadcast every tick), so a stray click racing a
    # slow start/stop still self-corrects on the next tick instead of lying.
    def _toggle_pose(self, on: bool):
        if on:
            if self._node_running('pose_node'):
                return
            subprocess.Popen(['ros2', 'run', 'home_robot', 'pose_node.py'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
            subprocess.Popen(['ros2', 'run', 'home_robot', 'fall_monitor_node.py',
                               '--ros-args', '-p', 'hold_s:=6.0', '-p', 'speak:=true'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
            subprocess.Popen(['ros2', 'run', 'home_robot', 'gesture_node.py',
                               '--ros-args',
                               '-p', 'auto_goal:=false', '-p', 'motion_gestures:=false',
                               '-p', 'confirm_hits:=5', '-p', 'max_distance:=8.0'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
        else:
            for name in ('pose_node.py', 'fall_monitor_node.py', 'gesture_node.py'):
                subprocess.run(['pkill', '-f', name])

    def _toggle_apriltag(self, on: bool):
        if on:
            if self._node_running('apriltag_node'):
                return
            cfg = os.path.join(SHARE, 'config', 'apriltag.yaml')
            # ‼️ apriltag_node's OWN default node name is 'apriltag', not
            # 'apriltag_node' — the launch file only gets the latter because
            # its Node(name='apriltag_node') implies a `-r __node:=` remap.
            # Without it here, _node_running('apriltag_node') never sees this
            # process (measured 2026-08-14: process alive, node name wrong,
            # so the toggle looked broken from the dashboard's status pill).
            subprocess.Popen(['ros2', 'run', 'apriltag_ros', 'apriltag_node',
                               '--ros-args', '-r', '__node:=apriltag_node',
                               '--params-file', cfg,
                               '-r', 'image_rect:=/camera/camera/color/image_raw',
                               '-r', 'camera_info:=/camera/camera/color/camera_info'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
            subprocess.Popen(['ros2', 'run', 'home_robot', 'apriltag_relocalizer.py',
                               '--ros-args',
                               '-p', 'tag_frame:=saloni_tag', '-p', 'base_frame:=base_link',
                               '-p', 'map_frame:=map'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
        else:
            for name in ('apriltag_node', 'apriltag_relocalizer.py'):
                subprocess.run(['pkill', '-f', name])

    def _nerf_train_start(self):
        with self._nerf_train_lock:
            if self._nerf_train_proc is not None and self._nerf_train_proc.poll() is None:
                return   # already running — the client already has progress via `latest`
            try:
                proc = subprocess.Popen(
                    [sys.executable, '-u', NERF_TRAIN_SCRIPT,
                     '--data', NERF_DATA_DIR, '--steps', '8000'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            except OSError as e:
                self._state.broadcast({'type': 'nerf_train', 'running': False,
                                       'done': False, 'error': repr(e)})
                return
            self._nerf_train_proc = proc
            self._nerf_train_stop_requested = False
        self._state.broadcast({'type': 'nerf_train', 'running': True, 'step': 0,
                               'steps': 8000, 'loss': None, 'psnr': None,
                               'elapsed_min': None, 'eta_min': None,
                               'done': False, 'error': None})
        tail = []
        for line in proc.stdout:
            line = line.rstrip('\n')
            if not line:
                continue
            tail.append(line)
            del tail[:-20]
            m = NERF_STEP_RE.search(line)
            if m:
                self._state.broadcast({
                    'type': 'nerf_train', 'running': True,
                    'step': int(m.group(1)), 'steps': int(m.group(2)),
                    'loss': float(m.group(3)), 'psnr': float(m.group(4)),
                    'elapsed_min': float(m.group(5)), 'eta_min': int(m.group(6)),
                    'done': False, 'error': None})
        proc.wait()
        with self._nerf_train_lock:
            self._nerf_train_proc = None
            stop_requested = self._nerf_train_stop_requested
        if proc.returncode == 0:
            self._state.broadcast({'type': 'nerf_train', 'running': False,
                                   'done': True, 'error': None})
        elif stop_requested:
            # SIGTERM makes this a nonzero exit too — without this check a
            # deliberate "Σταμάτα" click would read as a crash. A checkpoint
            # was already saved at the last 200-step mark, so nothing is lost.
            self._state.broadcast({'type': 'nerf_train', 'running': False,
                                   'done': False, 'error': None})
        else:
            # gpu_is_busy's refusal and every other failure land in plain
            # stdout (stderr is merged in above) — the last few lines are the
            # whole story: which PIDs are on the GPU, or a stack trace tail.
            # Exit code 2 is specifically that refusal (see train_nerf.py),
            # distinct from a real crash — worth its own flag so the client
            # can offer "stop perception" only when that would actually help.
            self._state.broadcast({'type': 'nerf_train', 'running': False,
                                   'done': False, 'gpu_busy': proc.returncode == 2,
                                   'error': '\n'.join(tail[-6:]) or f'exit {proc.returncode}'})

    def _nerf_train_stop(self):
        with self._nerf_train_lock:
            proc = self._nerf_train_proc
            self._nerf_train_stop_requested = True
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _cb_speaker(self, msg: String):
        try:
            snap = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        # 1 Hz and mostly unchanged; only forward transitions so the socket is
        # not carrying an identical message every second all day.
        key = (snap.get('name'), snap.get('identified'),
               None if snap.get('angle') is None else round(snap['angle'] / 15))
        if key == self._speaker_key:
            return
        self._speaker_key = key
        self._state.broadcast({'type': 'speaker', **snap})

    def _cb_fall(self, msg: Bool):
        # remember=True: replayed to a browser that connects later, which is
        # the whole point of a safety alert nobody may be watching live.
        self._state.broadcast({'type': 'fall', 'on': bool(msg.data)})

    def _cb_doa_rotate(self, msg: Bool):
        # remembered, so the switch paints correctly on a tab opened later
        self._state.broadcast({'type': 'doa_rotate', 'on': bool(msg.data)})

    def _cb_vad(self, msg: Bool):
        # remembered too — a tab opened mid-utterance should show "speaking"
        # immediately, not wait for the NEXT transition (this topic only
        # publishes on change, see doa_node's own comment on voice_activity).
        self._state.broadcast({'type': 'vad', 'on': bool(msg.data)})

    def _cb_fall_event(self, msg: String):
        try:
            detail = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'fall_event', **detail}, remember=False)

    def _cb_mission(self, msg: String):
        # Plain state name: idle / navigating / inspecting / done / failed /
        # cancelled. The answer itself comes back as speech, which the browser
        # already shows in the chat pane.
        self._state.broadcast({'type': 'mission', 'state': msg.data})

    def _cb_dock(self, msg: String):
        self._state.broadcast({'type': 'dock', 'status': msg.data})

    def _cb_estop(self, msg: Bool):
        self._state.broadcast({'type': 'estop', 'on': bool(msg.data)})

    def _cb_heard(self, msg: String):
        self._state.add_chat('user', msg.data)

    def _cb_said(self, msg: String):
        self._state.add_chat('robot', msg.data)

    def _cb_wake(self, msg: String):
        self._state.add_chat('wake', msg.data or 'wake')

    def _cb_speaking(self, msg: Bool):
        self._state.broadcast({'type': 'speaking', 'on': bool(msg.data)})

    def _cb_situation(self, msg: String):
        self._state.broadcast({'type': 'situation', 'text': msg.data})

    # ── Gestures / observations / timeline ───────────────────────────────────

    # ── Map-referenced compass ───────────────────────────────────────────────
    # The offset lives on disk, not in a launch parameter: it belongs to the
    # MAP, and a remap invalidates it the same way it invalidates the taught
    # locations. Stored with the map name so switching maps does not silently
    # carry a stale north over.

    def _compass_load(self):
        try:
            with open(_COMPASS_PATH, encoding='utf-8') as fh:
                return json.load(fh).get('offset')
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _compass_store(self, offset):
        self._compass_offset = offset
        try:
            os.makedirs(os.path.dirname(_COMPASS_PATH), exist_ok=True)
            tmp = _COMPASS_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump({'offset': offset}, fh)
            os.replace(tmp, _COMPASS_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save compass offset: {exc}')
        self._broadcast_compass()

    def _compass_calibrate(self, bearing_deg):
        if self._last_yaw is None:
            self.get_logger().warn(
                'compass calibration needs a pose — localize first')
            return
        self._compass_store(
            offset_from_known_bearing(self._last_yaw, bearing_deg))
        self.get_logger().info(
            f'Compass calibrated: facing {bearing_deg:.0f}° at yaw '
            f'{math.degrees(self._last_yaw):.1f}°')

    def _broadcast_compass(self):
        self._state.broadcast({'type': 'compass', 'offset': self._compass_offset})

    def _set_mic_sub(self, on: bool):
        """Subscribe to /mic/audio only while _listen_ws is non-empty — see
        the comment where the (removed) unconditional subscription was."""
        if on and self._mic_sub is None:
            self._mic_sub = self.create_subscription(
                Int16MultiArray, '/mic/audio', self._cb_mic, 30)
        elif not on and self._mic_sub is not None:
            self.destroy_subscription(self._mic_sub)
            self._mic_sub = None

    def _cb_mic(self, msg: Int16MultiArray):
        """Forward raw 16 kHz mono PCM to whoever is listening.

        Sent as int16 little-endian bytes rather than JSON: a 100 ms chunk is
        ~1600 samples, and base64 in a JSON envelope would roughly double the
        bytes and cost a parse per chunk at 10 Hz per client.
        """
        if not self._listen_ws:
            return    # defensive only, see _set_mic_sub
        self._state.send_bytes(self._listen_ws,
                               np.asarray(msg.data, dtype=np.int16).tobytes())

    def _cb_gesture(self, msg: String):
        """Live pointing state. Broadcast as-is; the pane renders the ring."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'gesture', **data})

    def _cb_observations(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'observations', **data})

    def _cb_timeline(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'timeline', **data})

    def _cb_object_memory(self, msg: String):
        """/object_memory carries only CONFIRMED instances already, sorted
        newest-first (object_memory_node.py) — nothing to filter or sort here.
        The room key ('saloni') is translated to a bare Greek noun the same way
        the acoustic map does, so the pane never has to show a raw map key."""
        try:
            items = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        for it in items:
            room = it.get('room')
            it['room_el'] = room_el(room) if room else None
        self._state.broadcast({'type': 'object_memory', 'items': items})

    def _cb_quota(self, msg: String):
        """How many Gemini requests are left today. Latched by llm_bridge, so a
        tab opened at any point gets the number rather than waiting for the next
        thing the owner says."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'quota', **data})

    def _cb_vocab(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'vocab', **data})

    def _cb_sound(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'sound', **data})

    def _cb_hand(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'hand', **data})

    def _cb_people(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'people', **data})

    def _cb_acoustic(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'acoustic', **data})

    def _cb_touch(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'touch', **data})

    # ── system settings (wifi / bluetooth / audio / power) ───────────────────
    # Everything here shells out, so it all runs on a worker thread: an nmcli
    # scan takes seconds and would stall the ROS executor.

    @staticmethod
    def _run(args, timeout=20):
        try:
            r = subprocess.run(args, capture_output=True, text=True,
                               timeout=timeout)
            return r.stdout or '', r.returncode
        except Exception as exc:                          # noqa: BLE001
            return f'{exc}', 1

    def _sys_snapshot(self, note=None, error=None):
        wifi_out, _ = self._run(
            ['nmcli', '-t', '-f', 'IN-USE,SSID,SIGNAL,SECURITY',
             'device', 'wifi', 'list'], timeout=25)
        dev_out, _ = self._run(
            ['nmcli', '-t', '-f', 'DEVICE,TYPE,STATE,CONNECTION', 'device'])
        bt_out, _ = self._run(bt_args('devices'))
        conn_out, _ = self._run(['bluetoothctl', 'devices', 'Connected'])
        vol_out, _ = self._run(['pactl', 'get-sink-volume', '@DEFAULT_SINK@'])
        bt_show, _ = self._run(['bluetoothctl', 'show'])
        ips, _ = self._run(['hostname', '-I'])
        ts, _ = self._run(['tailscale', 'ip', '-4'])

        connected = [d['mac'] for d in parse_bt_devices(conn_out)]
        self._state.broadcast({
            'type': 'sysnet',
            'wifi': parse_wifi_list(wifi_out)[:14],
            'devices': parse_devices(dev_out),
            'bt_on': 'Powered: yes' in bt_show,
            'bt': parse_bt_devices(bt_out, connected)[:14],
            'volume': parse_volume(vol_out),
            'ips': (ips or '').split(),
            'tailscale': (ts or '').strip().splitlines()[:1],
            'note': note,
            'error': error,
        }, remember=False)

    def _power_profile(self, profile: str):
        """Switch the machine-wide power profile — see scripts/power_smooth.sh.

        Forwards to /usr/local/sbin/power-smooth, which does four things (CPU
        band, boost, amd-pstate EPP, iGPU DPM cap) and saves the choice so
        power-smooth.service reapplies it at boot. Needs the one-time
        scripts/install_power_smooth.sh; until then the script exits non-zero
        with instructions rather than silently doing a fraction of the job.
        """
        script = os.path.join(SRC_HOME, 'scripts', 'power_smooth.sh')
        if not os.path.exists(script):
            script = os.path.join(SHARE, 'scripts', 'power_smooth.sh')
        try:
            r = subprocess.run([script, profile], capture_output=True,
                               text=True, timeout=30)
            ok = r.returncode == 0
            out = (r.stdout or '').strip()
            err = '' if ok else (r.stderr or out or 'failed').strip()[:160]
        except Exception as exc:                          # noqa: BLE001
            ok, out, err = False, '', str(exc)[:160]
        if not ok:
            self.get_logger().error(f'power profile {profile}: {err}')
        self._state.broadcast({'type': 'power_profile', 'profile': profile,
                               'ok': ok, 'detail': out.splitlines()[-1] if out else '',
                               'error': err}, remember=False)

    def _usb_power_cycle(self, name: str):
        """Cut power to one USB port for two seconds, from the other side of
        the house.

        The helper is the root-owned copy in /usr/local/sbin, not the one in
        the source tree: a NOPASSWD sudoers entry pointing at a file this user
        can edit would be a root shell with extra steps. See
        scripts/install_usb_power.sh.
        """
        # ‼️ Not self._run: it returns stdout only, and everything that goes
        # wrong here goes to stderr — `sudo -n` refusing for want of a password
        # says nothing on stdout at all, so the panel showed a bare "failed"
        # and sent you hunting for a hardware fault. Measured 2026-08-05.
        if not os.path.exists(USB_HELPER):
            # By far the most likely failure: the one-time root install was
            # never run. Say exactly that, and what to type.
            err = 'δεν έχει εγκατασταθεί — τρέξε: sudo scripts/install_usb_power.sh'
            ok = False
            self.get_logger().error(f'usb power cycle {name}: {USB_HELPER} missing')
        else:
            try:
                r = subprocess.run(['sudo', '-n', USB_HELPER, 'cycle', name],
                                   capture_output=True, text=True, timeout=30)
                ok = r.returncode == 0
                err = '' if ok else (r.stderr or r.stdout or 'failed').strip()[:160]
            except Exception as exc:                      # noqa: BLE001
                ok, err = False, str(exc)[:160]
            if not ok:
                self.get_logger().error(f'usb power cycle {name} failed: {err}')
                if 'password' in err.lower():
                    err = ('λείπει το NOPASSWD — τρέξε: '
                           'sudo scripts/install_usb_power.sh')
        self._state.broadcast({
            'type': 'usb_power', 'device': name, 'ok': ok, 'error': err,
        }, remember=False)

    def _sys_task(self, args, note, refresh=True, timeout=45):
        out, rc = self._run(args, timeout=timeout)
        # ‼️ Never log `args` — a wifi password is in there.
        self.get_logger().info(f'system settings: {note} (rc={rc})')
        if refresh:
            self._sys_snapshot(note=note if rc == 0 else None,
                               error=None if rc == 0 else (out or '')[:200])

    def _cb_echo(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'echo', **data})

    def _cb_diag(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._state.broadcast({'type': 'diagnostics', **data})

    def _cb_episodic_answer(self, msg: String):
        # Not remembered: an answer belongs to the question that was just asked,
        # so replaying it to a tab that connects later would be confusing.
        self._state.broadcast({'type': 'recall_answer', 'text': msg.data},
                              remember=False)

    # ── System panel ─────────────────────────────────────────────────────────

    # Which hwmon chip is which piece of hardware, and what to call it in the
    # UI. Measured on this Krackan mini PC (`sensors`): k10temp/Tctl is the CPU
    # package, amdgpu/edge the integrated GPU, nvme/Composite the SSD, spd5118
    # the two RAM modules, acpitz the board. Anything not listed still shows up
    # under its raw chip name rather than being dropped — a sensor that appears
    # after a kernel upgrade should be visible, not silently missing.
    _TEMP_LABELS = {
        'k10temp':  ('CPU',      '🔥'),
        'amdgpu':   ('iGPU',     '🎮'),
        'nvme':     ('SSD',      '💾'),
        'spd5118':  ('RAM',      '🧠'),
        'acpitz':   ('Μητρική',  '🖥️'),
    }
    # Above this a temperature is drawn as a warning. The NPU box throttles well
    # before anything is damaged, so these are "look at me", not "shut down".
    _TEMP_WARN = {'CPU': 85.0, 'iGPU': 85.0, 'SSD': 70.0, 'RAM': 60.0,
                  'Μητρική': 80.0}

    def _temperatures(self):
        """Every temperature the box exposes, newest reading, as a flat list.

        One entry per sensor, not per chip: the NVMe reports Composite plus two
        internal sensors and both RAM modules report separately, and collapsing
        them to "the first one" (what this used to do) threw away the hottest
        reading — which is the only one worth looking at.
        """
        out = []
        try:
            chips = psutil.sensors_temperatures() or {}
        except Exception:
            return out
        for chip, entries in sorted(chips.items()):
            name, icon = self._TEMP_LABELS.get(chip, (chip, '🌡️'))
            multi = len(entries) > 1
            for i, e in enumerate(entries):
                # "SSD" when there is one reading, "SSD Composite" / "RAM 2"
                # when the chip reports several.
                label = name
                if multi:
                    label = f'{name} {e.label}' if e.label else f'{name} {i + 1}'
                out.append({
                    'name': label, 'icon': icon,
                    'c': round(e.current, 1),
                    'warn': self._TEMP_WARN.get(name),
                    'high': round(e.high, 1) if e.high else None,
                })
        return out

    @staticmethod
    def _disks():
        """Real filesystems only.

        psutil lists every squashfs snap mount — 30+ of them here, all 100%
        full by definition — which would bury the one number that matters (how
        much room is left on the SSD) under pages of noise.
        """
        out = []
        seen = set()
        for p in psutil.disk_partitions():
            if p.fstype in ('squashfs', 'tmpfs', 'devtmpfs', 'overlay'):
                continue
            if p.device in seen:
                continue          # same device bind-mounted twice
            try:
                u = psutil.disk_usage(p.mountpoint)
            except Exception:
                continue
            seen.add(p.device)
            out.append({
                'mount': p.mountpoint,
                'pct':   round(u.percent, 1),
                'free':  round(u.free / 2**30, 1),
                'total': round(u.total / 2**30, 1),
            })
        return out

    def _publish_system(self):
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        # get_node_names() reads the local graph cache — unlike `ros2 node
        # list` it costs nothing and cannot hang when the daemon is stale.
        try:
            nodes = sorted({n for n, _ in self.get_node_names_and_namespaces()})
        except Exception:
            nodes = []
        temps = self._temperatures()
        load = os.getloadavg()
        cores = psutil.cpu_count() or 1
        self._state.broadcast({
            'type':  'sys',
            'cpu':   psutil.cpu_percent(),
            'cpu_cores': cores,
            # Per-core, so a pegged single thread is visible as one full bar
            # instead of a calm-looking average.
            'cpu_each': [round(p) for p in psutil.cpu_percent(percpu=True)],
            'mem':   round(vm.percent, 1),
            'mem_gb': round(vm.used / 2**30, 1),
            'mem_free_gb': round(vm.available / 2**30, 1),
            'mem_total_gb': round(vm.total / 2**30, 1),
            'swap_gb': round(sw.used / 2**30, 1),
            'swap_total_gb': round(sw.total / 2**30, 1),
            # The first temperature is what the old single-value UI showed;
            # keep it so anything still reading `temp` does not break.
            'temp':  temps[0]['c'] if temps else None,
            'temps': temps,
            'disks': self._disks(),
            'load':  round(load[0], 2),
            'load15': [round(x, 2) for x in load],
            # Load is only meaningful against the core count — 8.0 is idle on a
            # 16-thread box and on fire on a 4-thread one.
            'load_pct': round(100 * load[0] / cores),
            'nodes': nodes,
        })

    # ── Handle frontend messages ─────────────────────────────────────────────

    # ── Which map is loaded ─────────────────────────────────────────────────
    # map_server holds the answer in its `yaml_filename` parameter, which is the
    # only source that stays right when the stack is launched by hand with
    # map:=something. `ros2 param get` costs a second or two, hence the cache.
    MAP_CACHE_TTL = 15.0

    def _node_running(self, name: str) -> bool:
        try:
            return any(n == name for n, _ in self.get_node_names_and_namespaces())
        except Exception:
            return False

    def is_mapping(self) -> bool:
        """True while slam_toolbox is building a map (there is no map_server)."""
        return self._node_running('slam_toolbox')

    def perception_on(self) -> bool:
        return self._node_running('object_detector')

    def llm_backend(self):
        """Which LLM backend llm_bridge is running, or None if it is not up.

        ‼️ Carried across a map switch for the same reason use_perception is:
        the switch restarts the whole stack, and `robot max` without the flag
        falls back to 'lemonade' — which starts FastFlowLM and takes 4.7 GB
        back. That happened for real on 2026-08-01: a click on "Ενεργοποίηση"
        silently undid the move to Gemini, with no error anywhere.
        """
        now = time.time()
        if self._backend_cache and now - self._backend_cache[0] < self.MAP_CACHE_TTL:
            return self._backend_cache[1]
        value = None
        try:
            r = subprocess.run(['ros2', 'param', 'get', '/llm_bridge_node',
                                'backend'],
                               capture_output=True, text=True, timeout=10)
            hit = re.search(r'String value is:\s*(\S+)', r.stdout or '')
            if hit:
                value = hit.group(1)
        except Exception:
            pass
        self._backend_cache = (now, value)
        return value

    # ── LLM backend switching ────────────────────────────────────────────────
    # Two halves that must happen in the right order. `robot max` leaves
    # FastFlowLM stopped whenever the backend is gemini (it holds 4.7 GB for a
    # server nothing talks to), so switching TO the local model means starting
    # it and WAITING for it to listen before llm_bridge is told to use it —
    # otherwise the first thing said to the robot hits a closed port.
    FLM_PORT = 52625
    FLM_MODEL = 'qwen3.5:4b'
    FLM_KEEPALIVE = os.path.expanduser('~/bin/flm_keepalive.sh')
    FLM_STOPFILE = '/tmp/flm_keepalive.stop'
    FLM_BOOT_TIMEOUT = 90.0

    def _flm_listening(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            return s.connect_ex(('127.0.0.1', self.FLM_PORT)) == 0

    def _backend_msg(self, state: str, text: str, backend=None):
        self._state.broadcast({'type': 'llm_backend', 'state': state,
                               'text': text, 'backend': backend}, remember=False)

    def _switch_backend(self, backend: str):
        """Start/stop FastFlowLM as needed, then repoint llm_bridge at it."""
        if backend not in ('gemini', 'lemonade'):
            self._backend_msg('err', f'Άγνωστο backend: {backend}')
            return
        self._backend_cache = None          # force a re-read afterwards
        try:
            if backend == 'lemonade':
                if not self._flm_listening():
                    self._backend_msg('busy', 'Ξεκινά το FastFlowLM (NPU)… '
                                              'μπορεί να πάρει ~60 δευτερόλεπτα')
                    try:
                        os.remove(self.FLM_STOPFILE)
                    except OSError:
                        pass
                    if not os.path.exists(self.FLM_KEEPALIVE):
                        self._backend_msg('err', f'Λείπει το {self.FLM_KEEPALIVE}')
                        return
                    subprocess.Popen(
                        ['setsid', 'nohup', self.FLM_KEEPALIVE, self.FLM_MODEL],
                        stdout=open('/tmp/flm_serve.log', 'ab'),
                        stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = time.time() + self.FLM_BOOT_TIMEOUT
                    while time.time() < deadline:
                        if self._flm_listening():
                            break
                        time.sleep(1.0)
                    else:
                        self._backend_msg(
                            'err', 'Το FastFlowLM δεν σήκωσε τη θύρα 52625 — '
                                   'δες το /tmp/flm_serve.log')
                        return
                self._backend_msg('busy', 'Το FastFlowLM ακούει· αλλαγή backend…')
            else:
                self._backend_msg('busy', 'Αλλαγή σε Gemini…')

            r = subprocess.run(
                ['ros2', 'param', 'set', '/llm_bridge_node', 'backend', backend],
                capture_output=True, text=True, timeout=30)
            out = (r.stdout or '') + (r.stderr or '')
            if 'Set parameter successful' not in out:
                self._backend_msg('err', f'Απέτυχε: {out.strip()[:200]}')
                return

            # Only now is it safe to reclaim the 4.7 GB: llm_bridge is no
            # longer pointed at the local server.
            if backend == 'gemini':
                open(self.FLM_STOPFILE, 'a').close()
                subprocess.run(['pkill', '-f', 'flm_keepalive'],
                               capture_output=True)
                subprocess.run(['pkill', '-x', 'flm'], capture_output=True)

            self._backend_cache = None
            label = 'Gemini (cloud)' if backend == 'gemini' else 'Qwen3.5 (NPU)'
            self._backend_msg('ok', f'Ενεργό: {label}', backend)
        except Exception as e:
            self._backend_msg('err', f'Σφάλμα: {e}')

    def _rtab_save_snapshot(self):
        """Copy house.db to a timestamped file under RTAB_SAVED_DIR so this
        point in the map survives whatever mapping does next (including
        'trigger_new_map', which reuses the same live database). Runs on its
        own thread (called from dispatch()) — sqlite3's backup() streams
        page-by-page through SQLite's own Online Backup API, which is safe
        against the mapper writing to house.db concurrently; a plain file
        copy is not guaranteed to be.
        """
        self._rtab_save_state = 'saving'
        if not os.path.exists(RTAB_HOUSE_DB):
            self.get_logger().warn('rtabmap save_snapshot: no house.db yet')
            self._rtab_save_state = 'error'
            return
        os.makedirs(RTAB_SAVED_DIR, exist_ok=True)
        name = 'house_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.db'
        dst = os.path.join(RTAB_SAVED_DIR, name)
        try:
            src_conn = sqlite3.connect(f'file:{RTAB_HOUSE_DB}?mode=ro', uri=True)
            dst_conn = sqlite3.connect(dst)
            with dst_conn:
                src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
            self.get_logger().info(f'rtabmap snapshot saved: {name}')
            self._rtab_save_state = 'done'
        except Exception as e:
            self.get_logger().error(f'rtabmap save_snapshot failed: {e}')
            self._rtab_save_state = 'error'

    def active_map(self):
        if self.is_mapping():
            return None
        now = time.time()
        if self._map_cache and now - self._map_cache[0] < self.MAP_CACHE_TTL:
            return self._map_cache[1]
        name = None
        try:
            r = subprocess.run(['ros2', 'param', 'get', '/map_server',
                                'yaml_filename'],
                               capture_output=True, text=True, timeout=10)
            hit = re.search(r'([A-Za-z0-9_-]+)\.yaml', r.stdout or '')
            if hit:
                name = hit.group(1)
        except Exception:
            pass
        self._map_cache = (now, name)
        return name

    def _map_name_keepwarm(self):
        """Runs forever on its own thread, refreshing active_map()'s cache
        before its 15s TTL lapses — see the comment on the thread that starts
        this in __init__."""
        while True:
            try:
                self.active_map()
            except Exception:
                pass
            time.sleep(self.MAP_CACHE_TTL - 5.0)

    def release_client(self, client):
        """A browser went away — forget anything it had switched on."""
        self._cloud_ws.discard(client)
        self._cam_ws.discard(client)
        self._set_camera_sub(bool(self._cam_ws))
        # Same for a tab that was listening: a closed phone must not leave the
        # microphone streaming to nobody.
        self._listen_ws.discard(client)
        self._set_mic_sub(bool(self._listen_ws))
        # A tab closed on the 3D pane never sends its 'off', so the camera would
        # keep building pointclouds for nobody.
        self._fuse_ws.discard(client)
        self._fuse_cam_ws.discard(client)
        self._set_camera_pointcloud(bool(self._cloud_ws or self._fuse_cam_ws))
        self._costmap_on.discard(client)
        self._set_costmap_sub(bool(self._costmap_on))

    def dispatch(self, msg: dict, client=None):
        t = msg.get('type')
        if t == 'cmd_vel':
            tw = Twist()
            tw.linear.x  = float(msg.get('vx', 0))
            tw.angular.z = float(msg.get('wz', 0))
            self._vel_pub.publish(tw)
        elif t == 'stop':
            self._vel_pub.publish(Twist())
        elif t == 'cancel_nav':
            self._cancel_navigation()
        elif t == 'estop':
            self._estop_pub.publish(Bool(data=bool(msg.get('on'))))
            if msg.get('on'):
                self._vel_pub.publish(Twist())
        elif t == 'nav_goal':
            g = PoseStamped()
            g.header.stamp    = self.get_clock().now().to_msg()
            g.header.frame_id = 'map'
            g.pose.position.x = float(msg['x'])
            g.pose.position.y = float(msg['y'])
            g.pose.orientation.w = 1.0
            self._goal_pub.publish(g)
        elif t == 'goto_room':
            loc = self._locations.get(msg.get('room', ''))
            if loc:
                g = PoseStamped()
                g.header.stamp    = self.get_clock().now().to_msg()
                g.header.frame_id = 'map'
                g.pose.position.x = float(loc['x'])
                g.pose.position.y = float(loc['y'])
                yaw = float(loc.get('yaw', 0))
                g.pose.orientation.z = math.sin(yaw / 2)
                g.pose.orientation.w = math.cos(yaw / 2)
                self._goal_pub.publish(g)
        elif t == 'cloud':
            # Tracked per socket, so two phones on the 3D tab do not have the
            # first one to leave switch off the other's stream.
            if client is not None:
                if msg.get('on'):
                    self._cloud_ws.add(client)
                else:
                    self._cloud_ws.discard(client)
                self._set_camera_pointcloud(
                    bool(self._cloud_ws or self._fuse_cam_ws))
        elif t == 'cam_view':
            # Same idea as 'cloud': the raw color subscription (_set_camera_sub)
            # only exists while at least one socket is on the tab.
            if client is not None:
                if msg.get('on'):
                    self._cam_ws.add(client)
                else:
                    self._cam_ws.discard(client)
                self._set_camera_sub(bool(self._cam_ws))
        elif t == 'fusion':
            # Two switches, not one: the EKF panel is a few hundred bytes at
            # 4 Hz, while the sensor comparison turns the D435's pointcloud
            # filter on and costs real CPU on the camera. A tab that only wants
            # the numbers must not pay for the cloud.
            if client is not None:
                # First viewer re-zeroes everything: every number on the tab is
                # "since when you started looking", and inheriting an hour-old
                # reference from a tab someone closed would open the panel on a
                # metre of correction that had already been explained.
                if msg.get('on') and not self._fuse_ws:
                    self._fusion_reset()
                for key, s in (('on', self._fuse_ws), ('cam', self._fuse_cam_ws)):
                    if msg.get(key):
                        s.add(client)
                    else:
                        s.discard(client)
                self._set_camera_pointcloud(
                    bool(self._cloud_ws or self._fuse_cam_ws))
        elif t == 'fusion_reset':
            self._fusion_reset()
        elif t == 'localize':
            if self._loc_client.service_is_ready():
                self._loc_client.call_async(Empty.Request())
        elif t == 'costmap':
            # Tracked per socket for the same reason the 3D cloud is: a browser
            # closed on the tab never sends its 'off', and a counter would leave
            # the encode running for ever with nobody watching.
            if client is not None:
                if msg.get('on'):
                    self._costmap_on.add(client)
                else:
                    self._costmap_on.discard(client)
                self._set_costmap_sub(bool(self._costmap_on))
        elif t == 'check_room':
            room = str(msg.get('room', '')).strip()
            question = str(msg.get('question', '')).strip() or 'τι βλέπεις;'
            # ':' is the mission string's own separator — a question carrying
            # one would be truncated. Same guard as the llm_bridge tool.
            question = question.replace(':', ' ')
            if room in self._locations:
                self._mission_pub.publish(
                    String(data=f'check:{room}:{question}'))
        elif t == 'sort':
            room = str(msg.get('room', '')).strip()
            self._mission_pub.publish(String(
                data=f'sort:{room}' if room and room in self._locations else 'sort'))
        elif t == 'cancel_mission':
            self._mission_pub.publish(String(data='cancel'))
        elif t == 'room_tint':
            self._rooms_tinted = bool(msg.get('on', True))
            # The map is only published when it changes — on a static map that
            # is once, at startup — so without a redraw the toggle would do
            # nothing until the next remap.
            if self._last_map is not None:
                self._cb_map(self._last_map)
        elif t == 'save_rooms':
            # Threaded: as of the per-map room files, this now calls
            # self.active_map() too, which shells out to `ros2 param get` on a
            # cache miss (~1-2s) — dispatch() runs on the asyncio event loop,
            # so anything blocking here stalls every other connected socket.
            threading.Thread(target=self._save_rooms,
                             args=(msg.get('rooms', []),), daemon=True).start()
        elif t == 'place_room':
            threading.Thread(target=self._place_room,
                             args=(msg.get('x'), msg.get('y'),
                                   msg.get('name', ''), msg.get('color')),
                             daemon=True).start()
        elif t == 'place_room_rect':
            threading.Thread(target=self._place_room_rect,
                             args=(msg.get('x1'), msg.get('y1'),
                                   msg.get('x2'), msg.get('y2'),
                                   msg.get('name', ''), msg.get('color')),
                             daemon=True).start()
        elif t == 'add_keepout_zone':
            threading.Thread(target=self._add_keepout_zone,
                             args=(msg.get('x1'), msg.get('y1'),
                                   msg.get('x2'), msg.get('y2'),
                                   msg.get('name', '')),
                             daemon=True).start()
        elif t == 'delete_keepout_zone':
            threading.Thread(target=self._delete_keepout_zone,
                             args=(msg.get('name', ''),), daemon=True).start()
        elif t == 'keepout_activate':
            threading.Thread(target=self._keepout_activate,
                             args=(bool(msg.get('on')),), daemon=True).start()
        elif t == 'pick_room':
            try:
                name = self._room_at_xy(float(msg.get('x', 0)), float(msg.get('y', 0)))
            except (TypeError, ValueError):
                name = None
            self._state.broadcast({'type': 'room_picked', 'name': name}, remember=False)
        elif t == 'set_backend':
            threading.Thread(target=self._switch_backend,
                             args=(str(msg.get('backend', '')),),
                             daemon=True).start()
        elif t == 'nerf_capture':
            self._nerf_pub.publish(Bool(data=bool(msg.get('on'))))
        elif t == 'nerf_train_start':
            threading.Thread(target=self._nerf_train_start, daemon=True).start()
        elif t == 'nerf_train_stop':
            self._nerf_train_stop()
        elif t == 'nerf_stop_perception':
            threading.Thread(target=self._nerf_stop_perception, daemon=True).start()
        elif t == 'toggle_pose':
            threading.Thread(target=self._toggle_pose, args=(bool(msg.get('on')),),
                              daemon=True).start()
        elif t == 'toggle_apriltag':
            threading.Thread(target=self._toggle_apriltag, args=(bool(msg.get('on')),),
                              daemon=True).start()
        elif t == 'gesture_go':
            # gesture_node holds the point; it re-checks that one exists, so an
            # eager click before any gesture is a logged warning, not a goal.
            self._gesture_go_pub.publish(EmptyMsg())
        elif t == 'recall':
            self._recall_pub.publish(String(data=str(msg.get('when', ''))))
        elif t == 'listen':
            if client is not None:
                if msg.get('on'):
                    self._listen_ws.add(client)
                else:
                    self._listen_ws.discard(client)
                self._set_mic_sub(bool(self._listen_ws))
        elif t == 'sys_rotate_token':
            # Writing the file is enough: the node reads it at startup, so the
            # new token takes effect on the next restart and the current tab
            # keeps working until then.
            new = secrets.token_urlsafe(24)
            try:
                os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
                with open(TOKEN_FILE, 'w') as fh:
                    fh.write(new + '\n')
                os.chmod(TOKEN_FILE, 0o600)
                self.get_logger().warn(
                    'dashboard token rotated — restart the node, then open '
                    'the new link once on every device')
                self._state.broadcast({'type': 'token', 'token': new},
                                      remember=False)
            except OSError as exc:
                self._state.broadcast({'type': 'token', 'error': str(exc)},
                                      remember=False)
        elif t == 'sys_refresh':
            threading.Thread(target=self._sys_snapshot, daemon=True).start()
        elif t == 'sys_wifi_connect':
            args = wifi_connect_args(str(msg.get('ssid', '')),
                                     msg.get('password') or None)
            if args:
                threading.Thread(
                    target=self._sys_task,
                    args=(args, f"wifi -> {msg.get('ssid')}"),
                    daemon=True).start()
        elif t == 'sys_bt':
            args = bt_args(str(msg.get('action', '')), msg.get('mac'))
            if args:
                threading.Thread(
                    target=self._sys_task,
                    args=(args, f"bluetooth {msg.get('action')}"),
                    daemon=True).start()
        elif t == 'sys_volume':
            args = volume_args(msg.get('percent'))
            if args:
                threading.Thread(target=self._sys_task,
                                 args=(args, 'volume'), daemon=True).start()
        elif t == 'sys_power':
            # Reboot and shutdown are one tap from losing a running robot, so
            # the UI confirms and the action is spelled out here explicitly
            # rather than passed through from the browser.
            action = str(msg.get('action', ''))
            cmd = {'reboot': ['sudo', 'systemctl', 'reboot'],
                   'poweroff': ['sudo', 'systemctl', 'poweroff']}.get(action)
            if cmd:
                self.get_logger().warn(f'{action} requested from the dashboard')
                threading.Thread(target=self._sys_task,
                                 args=(cmd, action, False), daemon=True).start()
        elif t == 'usb_power':
            # The device name is checked against a fixed list here rather than
            # forwarded, so nothing the browser sends can become an argument to
            # a command running as root.
            name = str(msg.get('device', ''))
            if name in USB_DEVICES:
                self.get_logger().warn(
                    f'USB power cycle requested from the dashboard: {name}')
                threading.Thread(target=self._usb_power_cycle,
                                 args=(name,), daemon=True).start()
        elif t == 'power_profile':
            # Same rule as the USB names: the profile is checked against a
            # fixed set here, so nothing from the browser becomes an argument.
            prof = str(msg.get('profile', ''))
            if prof in ('off', 'eco', 'flat'):
                self.get_logger().info(f'power profile -> {prof} (dashboard)')
                threading.Thread(target=self._power_profile,
                                 args=(prof,), daemon=True).start()
        elif t == 'echo_probe':
            self._echo_pub.publish(String(data=str(msg.get('room', ''))))
        elif t == 'people_add':
            name = str(msg.get('name', '')).strip()
            if name:
                self._people_add_pub.publish(String(data=name))
        elif t == 'people_remove':
            name = str(msg.get('name', '')).strip()
            if name:
                self._people_rm_pub.publish(String(data=name))
        elif t == 'people_enrol':
            name = str(msg.get('name', '')).strip()
            what = str(msg.get('what', '')).strip()
            if name and what in ('face', 'voice'):
                self._people_enrol_pub.publish(String(data=json.dumps(
                    {'name': name, 'what': what})))
        elif t == 'gesture_bind':
            # Relayed as-is; the gesture nodes validate and are the authority on
            # whether motion may be enabled at all.
            payload = {}
            if msg.get('gesture'):
                payload['bindings'] = {str(msg['gesture']): str(msg.get('action', 'none'))}
            if 'motion_enabled' in msg:
                payload['motion_enabled'] = bool(msg['motion_enabled'])
            if payload:
                self._bind_pub.publish(String(data=json.dumps(payload)))
        elif t == 'safety_set':
            self._safety_apply(str(msg.get('key', '')), msg.get('value'))
        elif t == 'safety_reset':
            self._safety_reset()
        elif t == 'arm_limit_set':
            joint = str(msg.get('joint', ''))
            if joint in arm_settings.MECH_LIMITS:
                self._arm_apply_limit(joint, msg.get('lo'), msg.get('hi'))
        elif t == 'arm_speed_set':
            self._arm_apply_speed(msg.get('value'))
        elif t == 'arm_reset':
            self._arm_reset()
        elif t == 'mic_set':
            self._mic_apply(str(msg.get('key', '')), msg.get('value'))
        elif t == 'mic_reset':
            self._mic_reset()
        elif t == 'doa_rotate':
            # doa_node echoes the new value back on /doa/rotate_state, which is
            # what actually moves the checkbox — so a node that is not running
            # leaves the switch where it was instead of pretending it took.
            self._doa_rotate_pub.publish(Bool(data=bool(msg.get('on'))))
        elif t == 'face_enrol':
            name = str(msg.get('name', '')).strip()
            if name:
                self._enrol_pub.publish(String(data=name))
        elif t == 'face_forget':
            name = str(msg.get('name', '')).strip()
            if name:
                self._forget_pub.publish(String(data=name))
        elif t == 'compass_calibrate':
            self._compass_calibrate(float(msg.get('bearing', 0.0)))
        elif t == 'compass_clear':
            self._compass_store(None)
        elif t == 'vocab':
            # An empty string clears the vocabulary and idles the detector,
            # which is how the Stop button releases the GPU.
            self._vocab_pub.publish(String(data=str(msg.get('what', ''))))
        elif t == 'overlay':
            self._overlay = bool(msg.get('on', True))
        elif t == 'follow':
            # The same topic the llm_bridge 'follow' tool publishes, so the
            # button and "ακολούθησέ με" take one code path. person_follower
            # stops itself after follow_timeout even if nothing sends False.
            self._follow_pub.publish(Bool(data=bool(msg.get('on'))))
            if not msg.get('on'):
                self._vel_pub.publish(Twist())
        elif t == 'rtabmap_cmd':
            # pause/resume stop and restart map building without dropping the
            # graph — the usual reason is to carry the robot past somewhere it
            # cannot drive without polluting the map with garbage frames.
            # trigger_new_map starts a fresh session in the same database.
            cmd = str(msg.get('cmd', ''))
            if cmd in ('pause', 'resume', 'trigger_new_map'):
                cli = self._rtab_clients.get(cmd)
                if cli is None:
                    # ‼️ /rtabmap/rtabmap/… — the namespace AND the node name.
                    # /rtabmap/<cmd> also appeared in `ros2 service list` and is
                    # the obvious guess, but it was a stale graph entry left by
                    # an un-namespaced run: calling it succeeded, returned
                    # cleanly, and did nothing at all. Measured — the frame
                    # counter kept climbing at +24 per 10 s through a "pause".
                    # Against this path it drops to +1.
                    cli = self.create_client(Empty, f'/rtabmap/rtabmap/{cmd}')
                    self._rtab_clients[cmd] = cli
                if cli.service_is_ready():
                    cli.call_async(Empty.Request())
                    if cmd == 'trigger_new_map':
                        self._rtab_loops = 0
                else:
                    self.get_logger().warn(
                        f'/rtabmap/rtabmap/{cmd} not available — '
                        'is the 3D map session up?')
            elif cmd == 'save_snapshot' and self._rtab_save_state != 'saving':
                threading.Thread(target=self._rtab_save_snapshot,
                                  daemon=True).start()
        elif t == 'dock':
            self._dock_pub.publish(Bool(data=bool(msg.get('on', True))))
        elif t == 'arm_joint':
            js = JointState()
            js.name     = [str(msg['joint'])]
            js.position = [float(msg['pos'])]
            self._arm_cmd_pub.publish(js)
        elif t == 'gripper':
            self._gripper_pub.publish(Float32(data=float(msg['pos'])))
        elif t == 'arm_raw':
            # T:210 cmd:0 cuts torque so the arm can be walked by hand; T:100
            # re-inits. Free-form so the panel does not need a topic per command.
            self._arm_raw_pub.publish(String(data=str(msg.get('cmd', ''))))
        elif t == 'ask':
            # Straight onto the STT's own topic, so a typed question takes
            # exactly the path a spoken one does — same gates, same tools.
            self._speech_pub.publish(String(data=str(msg.get('text', ''))))
        elif t == 'say':
            self._say_pub.publish(String(data=str(msg.get('text', ''))))


# ── FastAPI app ────────────────────────────────────────────────────────────────

state     = State()
locations = _load_locations()
ros_node: Optional[DashboardNode] = None
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app: FastAPI):
    state.set_loop(asyncio.get_running_loop())
    yield

app = FastAPI(lifespan=_lifespan)

# noVNC's own HTML/JS, served from the distro package. Nothing secret lives
# here — the RFB stream it opens is what carries the token.
if os.path.isdir(NOVNC_DIR):
    app.mount('/novnc', StaticFiles(directory=NOVNC_DIR), name='novnc')
if os.path.isdir(THREE_VENDOR_DIR):
    app.mount('/vendor/three', StaticFiles(directory=THREE_VENDOR_DIR), name='three')


@app.get('/', response_class=HTMLResponse)
async def index(request: Request, t: str = ''):
    if not _authorised(t, request.cookies):
        return HTMLResponse(
            'Unauthorized — άνοιξε το link με το token: '
            'http://&lt;host&gt;:8080/?t=&lt;token&gt; '
            '(είναι στο ~/.home_robot/dashboard_token)',
            status_code=401)
    # The page keeps carrying the token in its own sub-requests, so nothing
    # depends on the cookie surviving; it only saves the *next* visit.
    #
    # no-store: this page IS the application — the CSS, the JS and the room list
    # are all inlined in it, and it carries the token. Served with no cache
    # directive at all, Safari falls back to heuristic caching and happily
    # re-serves a build from before the last `robot max`, so a fixed layout
    # looks unfixed and the only way out is clearing website data. Nothing here
    # is worth caching: it is regenerated per request and the heavy things
    # (map, camera, clouds) come over their own routes.
    resp = HTMLResponse(_make_html(list(locations.keys()), t or TOKEN),
                        headers={'Cache-Control': 'no-store'})
    if not NO_AUTH:
        resp.set_cookie(COOKIE_NAME, TOKEN, max_age=COOKIE_MAX_AGE,
                        httponly=True, samesite='strict')
    return resp


def _placeholder_jpg(text: str, sub: str = '') -> bytes:
    """A black 640x360 frame carrying a message, as JPEG.

    Sent instead of nothing when the camera has gone quiet. The point is that
    the multipart stream must never stall: see the comment in camera().
    """
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(img, text, (28, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (210, 210, 210), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (28, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (140, 140, 140), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b''


# Latin-1 only: cv2.putText cannot render Greek glyphs — it draws '?' boxes,
# which is precisely the symbol this bug is about.
_CAM_STALE_MSG = ('No camera frames', 'RealSense stopped publishing - check'
                                      ' /camera/camera/color/image_raw')
_CAM_WAIT_MSG = ('Waiting for the camera...', '')

# How long without a frame before the picture is declared stale. Comfortably
# longer than a slow detector cycle, far shorter than a browser's patience.
_CAM_STALE_S = 3.0
# Nothing is sent while the frame is unchanged, so this is the heartbeat that
# keeps the connection alive. Must stay well under the ~2-5 min a browser will
# wait on a stalled response before giving up and drawing a broken image.
_CAM_KEEPALIVE_S = 2.0
# Once the "no frames" placeholder has been sent there is no point redrawing it
# at the keep-alive rate; this is how often it is repeated to hold the
# connection open. Also well under the browser's patience.
_CAM_STALE_REPEAT_S = 10.0


@app.get('/arm_model.json')
async def arm_model(request: Request, t: str = ''):
    """The RoArm-M3's simplified geometry, built by scripts/build_arm_model.py.

    Served rather than inlined: the page is already ~120 kB and this is another
    170: inlining it would slow every page load for a tab most visits never
    open. Cached hard because it only changes when the arm is remodelled.
    """
    if not _authorised(t, request.cookies):
        return Response('Unauthorized', status_code=401)
    path = os.path.join(SRC_CONFIG_DIR, 'arm_model.json')
    if not os.path.exists(path):
        return JSONResponse(
            {'error': 'arm_model.json missing — run scripts/build_arm_model.py'},
            status_code=404)
    with open(path, 'rb') as f:
        return Response(f.read(), media_type='application/json',
                        headers={'Cache-Control': 'public, max-age=86400'})


@app.get('/robot_scan.glb')
async def robot_scan_glb(request: Request, t: str = ''):
    """Photorealistic textured scan of the robot itself (phone LiDAR, e.g.
    Scaniverse), shown at the live pose in the map tab's "Σάρωμα" view in
    place of the plain orange cone marker. One global asset (unlike
    /maps/scan.glb, which is per-map) — the robot doesn't change per map.
    """
    if not _authorised(t, request.cookies):
        return Response('Unauthorized', status_code=401)
    path = os.path.join(SRC_CONFIG_DIR, 'robot_scan.glb')
    if not os.path.exists(path):
        return JSONResponse({'error': 'no robot_scan.glb in config/'}, status_code=404)
    # no-store, not the 24h caching /arm_model.json uses: that one "only
    # changes when the arm is remodelled" (rare); this file is still being
    # iterated on live and a stale cached copy is exactly what made a
    # server-side crop fix look like it hadn't done anything in the browser.
    with open(path, 'rb') as f:
        return Response(f.read(), media_type='model/gltf-binary',
                        headers={'Cache-Control': 'no-store'})


@app.get('/camera.mjpeg')
async def camera(request: Request, t: str = ''):
    if not _authorised(t, request.cookies):
        return Response('Unauthorized', status_code=401)

    async def stream():
        """Never stall, never repeat.

        The old loop sent a frame only `if jpg:` and slept otherwise, which
        gave two bugs at once:

          * when the camera stopped publishing, the response went completely
            silent. The browser waits a few minutes on a stalled stream and
            then aborts it — the '?' after ~3 minutes with the 3D tab (a
            different topic) still working perfectly.
          * while it WAS publishing, it re-sent the same buffer 25x a second
            regardless of whether a new frame had arrived, so a 5 fps camera
            still pushed ~1.2 MB/s per viewer over wifi.

        Now: send a frame when the sequence number moves, otherwise send the
        current picture every couple of seconds as a keep-alive, and swap in a
        placeholder that says what is wrong once frames stop arriving.
        """
        last_seq = -1
        last_send = 0.0
        stale_sent = False
        while True:
            # A browser that closed the tab (or navigated away) leaves this
            # generator running for ever otherwise — one leaked encode loop per
            # tab switch, all of them still touching state.
            if await request.is_disconnected():
                return

            now = time.monotonic()
            jpg = state.camera_jpg
            seq = state.camera_seq
            fresh = jpg is not None and (now - state.camera_at) < _CAM_STALE_S

            if fresh and seq != last_seq:
                last_seq, last_send, stale_sent = seq, now, False
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpg + b'\r\n')
            elif now - last_send >= _CAM_KEEPALIVE_S:
                if fresh:
                    payload = jpg                      # unchanged but alive
                elif not stale_sent or now - last_send >= _CAM_STALE_REPEAT_S:
                    payload = _placeholder_jpg(
                        *(_CAM_STALE_MSG if jpg is not None else _CAM_WAIT_MSG))
                    stale_sent = True
                else:
                    payload = None
                if payload:
                    last_send = now
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + payload + b'\r\n')

            await asyncio.sleep(0.04)   # ~25 fps cap

    return StreamingResponse(
        stream(), media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


# ── GUI sessions (RViz / MoveIt / Gazebo) ─────────────────────────────────────

def _gui_session(action: str, app_name: str, extra_arg: str = '') -> str:
    if not os.path.exists(GUI_SESSION_SH):
        return f'missing {GUI_SESSION_SH}'
    try:
        argv = ['bash', GUI_SESSION_SH, action, app_name]
        if extra_arg:
            argv.append(extra_arg)
        # Gazebo takes ~75 s to come up on this machine (software rendering),
        # so the timeout is generous; the UI polls status rather than blocking.
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        return (r.stdout or r.stderr).strip() or 'ok'
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception as e:
        return f'error: {e!r}'


@app.get('/gui/{app_name}/{action}')
async def gui(request: Request, app_name: str, action: str, t: str = '',
              file: str = ''):
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if app_name not in VNC_PORTS or action not in ('start', 'stop', 'status'):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    # rtabview is the one app that takes an argument: which saved snapshot
    # (see _rtab_save_snapshot) to open. Whitelisted here too, on top of
    # gui_session.sh's own check, since this string reaches a subprocess argv.
    extra_arg = ''
    if app_name == 'rtabview' and action == 'start':
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.db', file):
            return JSONResponse({'error': 'bad snapshot filename'},
                                 status_code=400)
        extra_arg = file
    out = await asyncio.to_thread(_gui_session, action, app_name, extra_arg)
    return {'app': app_name, 'action': action, 'result': out,
            'running': out.startswith('running')}


# ── VNC viewer page ───────────────────────────────────────────────────────────
# We serve our own page instead of the packaged /novnc/vnc.html, which failed
# with a bare "Script error." on some browsers.  Two independent causes, both
# fixed by not using that page:
#
#   * vnc.html loads its UI as <script type="module" crossorigin="anonymous">.
#     A module fetched in CORS mode has its errors MUTED unless the response
#     carries Access-Control-Allow-Origin, and StaticFiles sends none — so
#     every failure inside it reaches window.onerror as the useless string
#     "Script error." with no file, line, or stack.
#   * app/ui.js reads its settings straight out of localStorage with no
#     try/catch (app/webutil.js:159).  When the browser denies site storage —
#     Safari "Block All Cookies" / Lockdown Mode, private windows, strict
#     tracking protection — that throws during UI.start() and noVNC never
#     paints.  Reproduced here with cookies blocked.
#
# This page uses core/rfb.js directly (the protocol half, which touches no
# storage), imports it dynamically so a load failure is catchable, and reports
# whatever goes wrong to the parent frame in words.
VNC_VIEW_HTML = r"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__APP__ — noVNC</title>
<style>
html,body{margin:0;height:100%;background:#0c0c0e;overflow:hidden}
#screen{position:absolute;inset:0}
#msg{position:absolute;left:0;right:0;top:0;z-index:5;padding:8px 12px;
     font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     white-space:pre-wrap;word-break:break-word;display:none}
#msg.err{display:block;background:#3b0d0d;color:#fecaca;border-bottom:1px solid #7f1d1d}
#msg.info{display:block;background:#1c1c20;color:#a1a1aa;border-bottom:1px solid #2c2c32}
</style>
<!-- Classic script, deliberately first and non-module: it is installed before
     anything can throw, and it is what the parent frame listens to. -->
<script>
(function(){
  var APP = __APP_JS__;
  var sent = null;
  window.__vncReport = function(kind, text){
    text = String(text == null ? '' : (text.stack || text.message || text));
    // Only the first failure is interesting; later ones are fallout.
    if (kind === 'error' && sent) return;
    if (kind === 'error') sent = text;
    try {
      parent.postMessage({source:'vncview', app:APP, kind:kind, text:text},
                         location.origin);
    } catch(e){}
    var el = document.getElementById('msg');
    if (!el) return;
    if (kind === 'ok'){ el.className = ''; el.textContent = ''; return; }
    el.className = (kind === 'error') ? 'err' : 'info';
    el.textContent = (kind === 'error' ? '⚠ ' : '') + text;
  };
  window.addEventListener('error', function(e){
    // e.message is "Script error." only for muted cross-origin scripts; this
    // page loads none, so the real text always survives.
    __vncReport('error', (e.message || 'error') +
                (e.filename ? '\n' + e.filename + ':' + e.lineno : ''));
  });
  window.addEventListener('unhandledrejection', function(e){
    __vncReport('error', e.reason);
  });
})();
</script>
</head>
<body>
<div id="screen"></div>
<div id="msg" class="info">Σύνδεση…</div>
<script nomodule>
  __vncReport('error', 'Ο browser δεν υποστηρίζει ES modules — χρειάζεται νεότερη έκδοση.');
</script>
<script type="module">
// Dynamic import so a missing/broken /usr/share/novnc is a caught exception
// with a real message, not a silent dead frame.
let RFB;
try {
  RFB = (await import('/novnc/core/rfb.js')).default;
} catch (e) {
  __vncReport('error', 'Δεν φορτώνει το noVNC (/novnc/core/rfb.js): ' + e);
  throw e;
}

const APP  = __APP_JS__;
const PASS = __PASS_JS__;
const QS   = __QS_JS__;
const url  = (location.protocol === 'https:' ? 'wss' : 'ws') +
             '://' + location.host + '/vnc/' + APP + QS;

let rfb, connected = false;
try {
  rfb = new RFB(document.getElementById('screen'), url,
                {credentials: {password: PASS}});
} catch (e) {
  __vncReport('error', 'RFB: ' + e);
  throw e;
}
// The Xvnc geometry is fixed by gui_session.sh, so scale to the pane rather
// than asking the server to resize (RViz would relayout on every phone turn).
rfb.scaleViewport = true;
rfb.resizeSession  = false;
rfb.showDotCursor  = true;

rfb.addEventListener('connect', () => {
  connected = true;
  __vncReport('ok', '');
  // A reconnect that worked must not leave the counter armed, or the next
  // drop hours later would burn the last retry immediately.
  retry = 0;
  try { history.replaceState(null, '', location.pathname + location.search); }
  catch (e) {}
});
rfb.addEventListener('desktopname',
  e => { if (e.detail && e.detail.name) document.title = e.detail.name; });
// A wrong VNC password lands here, not in 'disconnect'.
rfb.addEventListener('securityfailure', e => {
  const d = e.detail || {};
  __vncReport('error', 'Απορρίφθηκε ο κωδικός VNC (' +
              (d.reason || 'status ' + d.status) + ') — δες το HOME_ROBOT_VNC_PASSWORD.');
});
rfb.addEventListener('credentialsrequired', () => {
  __vncReport('error', PASS ? 'Ο server ζήτησε ξανά κωδικό — λάθος κωδικός VNC.'
                            : 'Ο server ζητά κωδικό VNC και δεν έχει ρυθμιστεί.');
});
// The stock page had reconnect=1; keep that, because a phone that sleeps drops
// the socket every time. The count rides in the location hash — per-session
// browser storage is exactly what may be denied here — so a session that is
// genuinely gone stops after a few tries instead of reloading forever.
const MAX_RETRY = 5;
let retry = parseInt((location.hash.match(/r=(\d+)/) || [])[1] || '0', 10);

// RFB keeps its socket private and its 'disconnect' detail says nothing, so
// every pre-handshake failure looks the same from here — and the three causes
// need three different fixes. Reopen the identical URL once, purely to read
// the close code off it.  Runs at most once, only on a failure.
const CLOSE_MEANING = {
  1008: 'το token απορρίφθηκε — άνοιξε το dashboard με ?t=<token>',
  1011: 'ο server δεν βρήκε τίποτα στη θύρα __PORT__ — η συνεδρία :__DISP__ δεν τρέχει',
  1006: 'το websocket κόπηκε χωρίς κλείσιμο — proxy, extension ή δίκτυο στη μέση',
};
let diagnosed = false;
function diagnose(done){
  if (diagnosed) { done(''); return; }
  diagnosed = true;
  let ws, settled = false;
  const finish = why => { if (settled) return; settled = true;
                          try { ws && ws.close(); } catch (e) {}
                          done(why); };
  try { ws = new WebSocket(url, 'binary'); }
  catch (e) { finish('δεν άνοιξε καν websocket: ' + e); return; }
  const giveUp = setTimeout(() => finish('το websocket δεν απάντησε σε 8s'), 8000);
  ws.onopen = () => { clearTimeout(giveUp);
                      finish('το websocket ανοίγει κανονικά — δοκίμασε ↻'); };
  ws.onclose = e => { clearTimeout(giveUp);
                      finish(CLOSE_MEANING[e.code] ||
                             ('ws close ' + e.code + ' ' + (e.reason || ''))); };
}

function reportNoConnection(){
  diagnose(why => __vncReport('error',
    'Δεν συνδέθηκε στο :__DISP__ (θύρα __PORT__).' +
    (why ? '\n' + why : '') +
    '\nΈλεγχος: scripts/gui_session.sh status ' + APP));
}

rfb.addEventListener('disconnect', () => {
  if (!connected){ reportNoConnection(); return; }
  if (retry >= MAX_RETRY){
    __vncReport('error', 'Χάθηκε η σύνδεση και μετά από ' + MAX_RETRY +
                ' προσπάθειες — η γραφική συνεδρία μάλλον σταμάτησε.');
    return;
  }
  __vncReport('info', 'Χάθηκε η σύνδεση — επανασύνδεση…');
  setTimeout(() => {
    location.hash = 'r=' + (retry + 1);
    location.reload();
  }, 2000 * (retry + 1));
});
// Without this a socket that opens and then goes quiet leaves a black
// rectangle: RFB fires nothing at all while it waits. `diagnosed` keeps this
// from repeating whatever the disconnect handler already said.
setTimeout(() => { if (!connected && !diagnosed) reportNoConnection(); }, 15000);
</script>
</body>
</html>
"""


@app.get('/vncview/{app_name}', response_class=HTMLResponse)
async def vnc_view(request: Request, app_name: str, t: str = ''):
    """The RFB viewer itself. Token-checked so the VNC password, which is
    baked into the page, is never handed to an anonymous caller — and so it
    stops travelling in the iframe's URL the way ?password= used to."""
    if not _authorised(t, request.cookies):
        return HTMLResponse('Unauthorized', status_code=401)
    if app_name not in VNC_PORTS:
        return HTMLResponse('bad request', status_code=400)
    if not os.path.isdir(NOVNC_DIR):
        return HTMLResponse('noVNC is not installed (sudo apt install novnc)',
                            status_code=503)
    # Built from TOKEN, not from `t`: the caller may have been let in by the
    # cookie, in which case `t` is empty (or junk) and the frame's own
    # websocket would arrive unauthenticated.
    token_qs = '' if NO_AUTH else '?t=' + quote(TOKEN, safe='')
    port = VNC_PORTS[app_name]
    # json.dumps for anything that lands inside a JS string literal; the extra
    # </ guard is for a password containing "</script>", which would otherwise
    # close the block no matter how well the string itself is escaped.
    def js(v):
        return json.dumps(v).replace('</', '<\\/')
    html = (VNC_VIEW_HTML
            .replace('__APP_JS__', js(app_name))
            .replace('__APP__', app_name)
            .replace('__PASS_JS__', js(VNC_PASSWORD))
            .replace('__QS_JS__', js(token_qs))
            .replace('__DISP__', str(port - 5900))
            .replace('__PORT__', str(port)))
    # The page carries a credential; keep it out of shared caches.
    return HTMLResponse(html, headers={'Cache-Control': 'no-store'})


# ── Maps ──────────────────────────────────────────────────────────────────────
# The map is baked into map_server at launch: there is no runtime swap, so
# switching one means restarting the stack. That is why the UI confirms first
# and why the work is handed to a detached script — this node is one of the
# processes about to be killed.

MAP_SESSION_SH = os.path.normpath(
    os.path.join(SRC_MAPS_DIR, os.pardir, 'scripts', 'map_session.sh'))

NERF_TRAIN_SCRIPT = os.path.normpath(
    os.path.join(SRC_MAPS_DIR, os.pardir, 'scripts', 'train_nerf.py'))
NERF_DATA_DIR = os.path.expanduser('~/.home_robot/nerf/house')
# Matches train_nerf.py's progress line exactly:
#   step    200/8000  loss 0.01234  psnr 20.1 dB  1.2 min  eta 45 min
NERF_STEP_RE = re.compile(
    r'step\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+psnr\s+([\d.-]+)\s+dB\s+'
    r'([\d.]+)\s+min\s+eta\s+(\d+)\s+min')
# The three Nodes that actually touch the iGPU (see train_nerf.py's
# gpu_is_busy header) — killing just these frees /dev/kfd for training
# without tearing down nav2/drivers/this dashboard. None of the three are
# respawn=True in bringup.launch.py, so this does not cascade.
NERF_GPU_PROCS = ('object_detector.py', 'pose_node.py', 'open_vocab_detector.py')


def _list_maps() -> list:
    """Saved maps, newest first. A map is a .yaml with its .pgm next to it."""
    out = []
    try:
        for f in sorted(os.listdir(SRC_MAPS_DIR)):
            if not f.endswith('.yaml'):
                continue
            name = f[:-5]
            pgm = os.path.join(SRC_MAPS_DIR, name + '.pgm')
            if not os.path.exists(pgm):
                continue          # a yaml with no image cannot be loaded
            out.append({
                'name': name,
                'mtime': os.path.getmtime(os.path.join(SRC_MAPS_DIR, f)),
                'kb': round(os.path.getsize(pgm) / 1024),
                # slam_toolbox can only RESUME (extend) a map that has these.
                'resumable': os.path.exists(os.path.join(SRC_MAPS_DIR,
                                                         name + '.posegraph')),
            })
    except OSError:
        pass
    return sorted(out, key=lambda m: -m['mtime'])


@app.get('/maps')
async def maps(request: Request, t: str = ''):
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if ros_node is None:
        return {'maps': _list_maps(), 'active': None, 'mapping': False}
    # active_map() shells out to `ros2 param get`, which takes a second or two —
    # off the event loop, or every other browser request stalls behind it.
    active = await asyncio.to_thread(ros_node.active_map)
    return {'maps': _list_maps(), 'active': active,
            'mapping': ros_node.is_mapping()}


def _map_pgm_path(name: str) -> Optional[str]:
    for m in _list_maps():
        if m['name'] == name:
            path = os.path.join(SRC_MAPS_DIR, name + '.pgm')
            return path if os.path.exists(path) else None
    return None


def _delete_map(name: str) -> int:
    """Remove every file this map's name owns: the core yaml/pgm/slam-toolbox
    pair, its per-map rooms (room_files.py) and keepout zones
    (keepout_files.py), its low-res Gazebo companion (map_downsample.py,
    "<name>_lo.*"), and any straighten-apply .bak snapshots. Returns how many
    files were actually removed, so the route can tell "nothing here" apart
    from "deleted".
    """
    patterns = [
        name + '.yaml', name + '.pgm', name + '.data', name + '.posegraph',
        name + '_room_mask.png', name + '_room_colors.yaml',
        name + '_keepout_zones.yaml',
        name + '_lo.yaml', name + '_lo.pgm',
        name + '.pgm.bak-*',
    ]
    removed = 0
    for pat in patterns:
        for path in glob.glob(os.path.join(SRC_MAPS_DIR, pat)):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


@app.get('/maps/straighten/{name}')
async def maps_straighten_preview(request: Request, name: str, t: str = ''):
    """Side-by-side PNGs of the saved map as-is and cosmetically straightened —
    lets the Χάρτες tab show a choice right after a save, without writing
    anything. See home_robot/map_straighten.py for what "straightened" means
    and why it must stay a display-only option until the user picks it.

    ‼️ Must be registered BEFORE /maps/{action}/{name} below: both routes
    match /maps/<seg>/<seg>, and FastAPI dispatches to whichever was added to
    the app first. Defined after it, this 400'd on every call ("bad request"
    from maps_action's action-not-in-(switch,save,new) check) and never ran.
    """
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    pgm_path = _map_pgm_path(name)
    if pgm_path is None:
        return JSONResponse({'error': f'no such map: {name}'}, status_code=404)

    def render():
        gray = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        cleaned = map_straighten.straighten(gray)
        upscale = max(1, min(6, 900 // max(gray.shape)))
        return (map_straighten.encode_png(gray, upscale),
                map_straighten.encode_png(cleaned, upscale))

    result = await asyncio.to_thread(render)
    if result is None:
        return JSONResponse({'error': f'could not read {name}.pgm'}, status_code=500)
    original_png, straightened_png = result
    return {'name': name,
            'original': base64.b64encode(original_png).decode(),
            'straightened': base64.b64encode(straightened_png).decode()}


@app.get('/maps/straighten_apply/{name}')
async def maps_straighten_apply(request: Request, name: str, t: str = ''):
    """Replace <name>.pgm with the straightened version, after the user has
    seen both previews and chosen. The old file is kept as a timestamped
    .bak — nothing is destroyed, and restoring it is a plain file copy. Only
    the image changes: resolution/origin/frame in the .yaml are untouched, so
    a currently-active map still needs the existing "Ενεργοποίηση" (switch)
    restart to pick up the new pixels — same rule as any other map edit here.
    """
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    pgm_path = _map_pgm_path(name)
    if pgm_path is None:
        return JSONResponse({'error': f'no such map: {name}'}, status_code=404)

    def apply():
        gray = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        cleaned = map_straighten.straighten(gray)
        backup = pgm_path + '.bak-' + time.strftime('%Y%m%d-%H%M%S')
        os.replace(pgm_path, backup)
        ok = cv2.imwrite(pgm_path, cleaned)
        if not ok:
            os.replace(backup, pgm_path)  # restore rather than leave no map file
            return None
        return backup

    backup = await asyncio.to_thread(apply)
    if backup is None:
        return JSONResponse({'error': f'could not straighten {name}.pgm'}, status_code=500)
    return {'name': name, 'ok': True, 'backup': os.path.basename(backup)}


@app.get('/maps/walls3d')
async def maps_walls3d(request: Request, t: str = '', map: str = ''):
    """Wall footprints for the map tab's 3D view: the active map's clean
    rectilinear polygons (see home_robot/map_walls3d.py), each edge as a
    world-metre quad the browser extrudes into a box. Defaults to whatever
    map Nav2 is currently using; ?map=<name> previews any saved map.
    """
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    name = map or (await asyncio.to_thread(ros_node.active_map) if ros_node else None)
    if not name:
        return JSONResponse({'error': 'no active map'}, status_code=404)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    yaml_path = os.path.join(SRC_MAPS_DIR, name + '.yaml')
    pgm_path = _map_pgm_path(name)
    if pgm_path is None or not os.path.exists(yaml_path):
        return JSONResponse({'error': f'no such map: {name}'}, status_code=404)

    def build():
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        gray = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        boxes = map_walls3d.wall_footprints(
            gray, float(meta['resolution']), meta['origin'][:2])
        return {'name': name, 'walls': [{'corners': c, 'height': h} for c, h in boxes]}

    result = await asyncio.to_thread(build)
    if result is None:
        return JSONResponse({'error': f'could not read {name}.pgm'}, status_code=500)
    return result


@app.get('/maps/scan.glb')
async def maps_scan_glb(request: Request, t: str = '', map: str = ''):
    """Optional photorealistic textured 3D scan (glTF binary) for the map
    tab's 'Σάρωμα' view — a phone LiDAR scan (Scaniverse etc.) dropped in as
    <name>.glb next to that map's .pgm/.yaml, e.g. by scripts/ply_to_map.py's
    companion export. Not every map has one; 404s cleanly when it doesn't,
    same as /arm_model.json's missing-file case. Same name resolution/
    validation as /maps/walls3d (defaults to whatever map Nav2 is using).
    """
    if not _authorised(t, request.cookies):
        return Response('Unauthorized', status_code=401)
    name = map or (await asyncio.to_thread(ros_node.active_map) if ros_node else None)
    if not name:
        return JSONResponse({'error': 'no active map'}, status_code=404)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    path = os.path.join(SRC_MAPS_DIR, name + '.glb')
    if not os.path.exists(path):
        return JSONResponse({'error': f'no 3D scan for {name}'}, status_code=404)
    # no-store: same reasoning as /robot_scan.glb — these get re-cropped/
    # regenerated while iterating, and a 24h cache turns a real fix into
    # "still looks broken" in whatever browser already fetched it once.
    with open(path, 'rb') as f:
        return Response(f.read(), media_type='model/gltf-binary',
                        headers={'Cache-Control': 'no-store'})


@app.get('/rtabmap/saved')
async def rtabmap_saved(request: Request, t: str = ''):
    """Snapshots taken by the 'save_snapshot' rtabmap_cmd — see
    _rtab_save_snapshot. Separate from the live house.db that keeps growing.
    """
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)

    def list_snapshots():
        if not os.path.isdir(RTAB_SAVED_DIR):
            return []
        out = []
        for fn in os.listdir(RTAB_SAVED_DIR):
            if not fn.endswith('.db'):
                continue
            st = os.stat(os.path.join(RTAB_SAVED_DIR, fn))
            out.append({'name': fn, 'mb': round(st.st_size / 1e6, 1),
                        'mtime': st.st_mtime})
        out.sort(key=lambda m: -m['mtime'])
        return out

    snaps = await asyncio.to_thread(list_snapshots)
    state = ros_node._rtab_save_state if ros_node else 'idle'
    return {'saving': state == 'saving', 'error': state == 'error',
            'snapshots': snaps}


@app.get('/rtabmap/delete/{name}')
async def rtabmap_delete(request: Request, name: str, t: str = ''):
    """Remove one saved 3D-map snapshot (see /rtabmap/saved). Same filename
    whitelist as the rtabview GUI route, since this also reaches a file path.
    """
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.db', name):
        return JSONResponse({'error': 'bad snapshot filename'}, status_code=400)
    path = os.path.join(RTAB_SAVED_DIR, name)

    def remove():
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    ok = await asyncio.to_thread(remove)
    if not ok:
        return JSONResponse({'error': f'no such snapshot: {name}'}, status_code=404)
    return {'name': name, 'ok': True}


@app.get('/maps/{action}/{name}')
async def maps_action(request: Request, action: str, name: str, t: str = ''):
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    # Names go into a shell command and a file path, so they are whitelisted
    # rather than escaped.
    if action not in ('switch', 'save', 'new', 'delete') or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    if action in ('switch', 'delete') and name not in {m['name'] for m in _list_maps()}:
        return JSONResponse({'error': f'no such map: {name}'}, status_code=404)

    if action == 'delete':
        # Deleting the active map would leave map_server/AMCL pointed at
        # files that vanish out from under them — force a switch first,
        # same rule the "Ενεργοποίηση" button already enforces the other way.
        active = await asyncio.to_thread(ros_node.active_map) if ros_node else None
        if name == active:
            return JSONResponse({'error': 'cannot delete the active map'},
                                 status_code=400)
        removed = await asyncio.to_thread(_delete_map, name)
        return {'action': action, 'name': name, 'ok': removed > 0,
                'result': f'{removed} files removed'}

    args = ['bash', MAP_SESSION_SH, action]
    if action != 'new':
        args.append(name)
    # Carry the current runtime settings across the restart. Without these the
    # relaunch silently reverts to `robot max` defaults: the 3D tab and object
    # detector switch off, and the LLM falls back to the NPU — which is 4.7 GB
    # of RAM reappearing with nothing to show for it.
    if action in ('switch', 'new') and ros_node:
        if ros_node.perception_on():
            args.append('use_perception:=true')
        backend = await asyncio.to_thread(ros_node.llm_backend)
        if backend and backend != 'lemonade':
            args.append(f'llm_backend:={backend}')
    try:
        # switch/new outlive this process, so they are detached; save is quick
        # and its result is worth waiting for.
        if action == 'save':
            r = await asyncio.to_thread(
                lambda: subprocess.run(args, capture_output=True, text=True,
                                       timeout=120))
            ok = r.returncode == 0
            return {'action': action, 'name': name, 'ok': ok,
                    'result': (r.stdout or r.stderr).strip()[:400]}
        subprocess.Popen(args, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {'action': action, 'name': name, 'ok': True,
                'result': 'restarting'}
    except Exception as e:
        return JSONResponse({'error': repr(e)}, status_code=500)


APPLY_SKIRT_SH = os.path.normpath(
    os.path.join(SRC_MAPS_DIR, os.pardir, 'scripts', 'apply_skirt_margin.sh'))
KEEPOUT_APPLY_SH = os.path.normpath(
    os.path.join(SRC_MAPS_DIR, os.pardir, 'scripts', 'apply_keepout_toggle.sh'))


@app.get('/safety/skirt/{mm}')
async def safety_skirt(request: Request, mm: int, t: str = ''):
    """Rewrite collision_monitor's hard-stop margin and restart — see
    home_robot/collision_skirt.py for why this can't be a live SetParameters
    write like the rest of the Safety tab, and the module docstring there for
    why moving_forward/moving_backward are the only rings this ever touches.
    """
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if mm not in collision_skirt.ALLOWED_MARGINS_MM:
        return JSONResponse({'error': f'mm must be one of '
                             f'{collision_skirt.ALLOWED_MARGINS_MM}'},
                            status_code=400)
    path = collision_skirt.default_params_path()
    try:
        with open(path) as f:
            original = f.read()
        patched = collision_skirt.patch_moving_points(original, mm)
    except (OSError, ValueError) as exc:
        # Never restart on a patch we're not sure about — the robot keeps
        # running its old, still-valid safety config.
        return JSONResponse({'error': str(exc)}, status_code=400)
    try:
        with open(path, 'w') as f:
            f.write(patched)
    except OSError as exc:
        return JSONResponse({'error': str(exc)}, status_code=500)

    # Same runtime-settings carry-forward maps_action does above — without
    # this the relaunch would silently drop perception/LLM backend choices.
    args = ['bash', APPLY_SKIRT_SH]
    if ros_node:
        if ros_node.perception_on():
            args.append('use_perception:=true')
        backend = await asyncio.to_thread(ros_node.llm_backend)
        if backend and backend != 'lemonade':
            args.append(f'llm_backend:={backend}')
    try:
        subprocess.Popen(args, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        return JSONResponse({'error': repr(e)}, status_code=500)
    return {'mm': mm, 'ok': True, 'result': 'restarting'}


@app.websocket('/vnc/{app_name}')
async def vnc_bridge(ws: WebSocket, app_name: str, t: str = ''):
    """noVNC <-> Xvnc. This is websockify, minus the extra daemon.

    noVNC speaks raw RFB over the socket, so all we do is copy bytes both ways
    until either end hangs up.

    ‼️ The subprotocol is echoed, never asserted. This used to accept with a
    hardcoded subprotocol='binary' on the belief that "noVNC opens the socket
    with the 'binary' subprotocol" — it does not. rfb.js defaults
    `_wsProtocols` to [] (core/rfb.js:94 in the installed 1.3.0) and our viewer
    page passes no wsProtocols, so the request carries no Sec-WebSocket-Protocol
    header at all. RFC 6455 §4.1 forbids the server from naming one the client
    did not offer, and Chromium enforces it: the handshake was rejected with

        Response must not include 'Sec-WebSocket-Protocol' header
        if not present in request: binary

    and the RViz/MoveIt/Gazebo panes died at close code 1006 — after the
    "Script error." fix, so the tab reported the failure honestly and was still
    unusable. Verified in a real browser 2026-08-02.
    """
    if not _authorised(t, ws.cookies) or app_name not in VNC_PORTS:
        await ws.close(code=1008)      # policy violation
        return
    offered = ws.scope.get('subprotocols') or []
    await ws.accept(subprotocol='binary' if 'binary' in offered else None)
    try:
        reader, writer = await asyncio.open_connection('127.0.0.1',
                                                       VNC_PORTS[app_name])
    except OSError:
        # The session is not up — tell the tab so it can offer a Start button
        # instead of noVNC's opaque "Failed to connect".
        await ws.close(code=1011)
        return

    async def vnc_to_browser():
        while True:
            data = await reader.read(65536)
            if not data:
                break
            await ws.send_bytes(data)

    async def browser_to_vnc():
        while True:
            data = await ws.receive_bytes()
            writer.write(data)
            await writer.drain()

    tasks = [asyncio.create_task(vnc_to_browser()),
             asyncio.create_task(browser_to_vnc())]
    try:
        # Whichever side closes first ends the session; cancel the other pump
        # so a half-open socket cannot leak a task per reconnect.
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception:
        pass
    finally:
        for task in tasks:
            task.cancel()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ── Telemetry socket ──────────────────────────────────────────────────────────

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket, t: str = ''):
    if not _authorised(t, ws.cookies):
        await ws.close(code=1008)      # policy violation
        return
    await ws.accept()
    state.add_client(ws)

    # Paint the whole UI immediately: the map, then the last value seen on every
    # other topic, then the transcript so far.
    try:
        if state.map_png and state.map_info:
            await ws.send_text(json.dumps({
                'type':  'map',
                **state.map_info,
                'image': base64.b64encode(state.map_png).decode(),
                'rooms': state.map_rooms,
                'tinted': state.map_tinted,
            }))
        for msg in list(state.latest.values()):
            await ws.send_text(json.dumps(msg))
        for entry in list(state.chat)[-25:]:
            await ws.send_text(json.dumps({'type': 'chat', **entry}))
        for entry in list(state.logs)[-150:]:
            await ws.send_text(json.dumps({'type': 'log', **entry}))
        # Which LLM is answering. Off the event loop: it shells out to `ros2
        # param get`, which takes a second or two on a cold cache and would
        # otherwise stall every other socket while this one connects.
        backend = await asyncio.to_thread(ros_node.llm_backend)
        await ws.send_text(json.dumps({
            'type': 'llm_backend', 'state': 'ok' if backend else 'err',
            'backend': backend,
            'text': {'gemini': 'Ενεργό: Gemini (cloud)',
                     'lemonade': 'Ενεργό: Qwen3.5 (NPU)'}.get(
                         backend, 'Το llm_bridge δεν τρέχει')}))
    except Exception:
        state.remove_client(ws)
        return

    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            if ros_node:
                ros_node.dispatch(msg, ws)
    except WebSocketDisconnect:
        pass
    except Exception as exc:                              # noqa: BLE001
        # ‼️ This used to be a bare `pass`. Any exception inside dispatch() —
        # a typo, a missing attribute, a bad cast — closed the socket and left
        # NOTHING anywhere: no log line, no error to the browser, just a button
        # that did nothing. Cost real time chasing the compass calibration.
        # Still swallowed (a dead socket must not take the node down), but it
        # is now visible.
        if ros_node:
            ros_node.get_logger().error(f'dashboard client error: {exc!r}')
    finally:
        state.remove_client(ws)
        if ros_node:
            ros_node.release_client(ws)


# ── HTML frontend ──────────────────────────────────────────────────────────────

def _make_html(rooms: list, token: str = '') -> str:
    """Build the page.

    Written as a plain template with __PLACEHOLDER__ substitution rather than an
    f-string: the page is mostly CSS and JS, and every brace in it would
    otherwise have to be doubled.
    """
    token_qs = f'?t={token}' if token else ''
    return (HTML_TEMPLATE
            .replace('__ROOMS__', json.dumps(rooms))
            .replace('__TOKEN_QS__', json.dumps(token_qs))
            .replace('__ARM_LIMITS__', json.dumps(ARM_LIMITS))
            .replace('__ARM_JOINTS__', json.dumps(ARM_JOINTS))
            # The servo's mechanical ceiling, not the tuned envelope above —
            # the limit sliders must be draggable all the way out to this,
            # not just back within whatever the envelope already is.
            .replace('__ARM_MECH_LIMITS__', json.dumps(
                {j: list(v) for j, v in arm_settings.MECH_LIMITS.items()}))
            .replace('__HAS_NOVNC__', json.dumps(os.path.isdir(NOVNC_DIR)))
            # Sent from the server so the allowed names exist in exactly one
            # place — the browser cannot invent a seventh.
            .replace('__USB_DEVICES__', json.dumps(USB_DEVICES,
                                                   ensure_ascii=False))
            .replace('__SAFETY_SPECS__', json.dumps(
                {s.key: {'kind': s.kind, 'def': s.default, 'lo': s.lo,
                         'hi': s.hi, 'step': s.step,
                         'warn_above': s.warn_above, 'warn_below': s.warn_below}
                 for s in safety_settings.SPECS}))
            .replace('__SAFETY_INFO__', json.dumps(safety_settings.INFO_ONLY))
            .replace('__MIC_SPECS__', json.dumps(
                {s.key: {'kind': s.kind, 'def': s.default, 'lo': s.lo,
                         'hi': s.hi, 'step': s.step,
                         'warn_above': s.warn_above, 'warn_below': s.warn_below}
                 for s in mic_settings.SPECS}))
            .replace('__MIC_INFO__', json.dumps(mic_settings.INFO_ONLY))
            .replace('__WAKE_MODEL_CHOICES__', json.dumps(mic_settings.WAKE_MODEL_CHOICES))
            .replace('__SKIRT_MARGINS__', json.dumps(collision_skirt.ALLOWED_MARGINS_MM))
            .replace('__SKIRT_DEFAULT_MM__', json.dumps(collision_skirt.MARGIN_DEFAULT_MM))
            .replace('__I18N__', json.dumps(as_js_table(), ensure_ascii=False))
            .replace('__LANGS__', json.dumps(LANGUAGES, ensure_ascii=False)))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<!-- viewport-fit=cover is what makes env(safe-area-inset-*) resolve to anything
     other than 0 on a notched iPhone; the tab bar uses it. -->
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Home Robot</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{height:100%;overflow:hidden;background:#141416;color:#e4e4e7;
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',Inter,sans-serif;font-size:14px}
button{font:inherit;color:inherit}
/* ── Header ── */
#hdr{display:flex;align-items:center;gap:10px;padding:0 14px;background:#1c1c20;
  border-bottom:1px solid #2c2c32;height:48px;flex-shrink:0}
#dot{width:9px;height:9px;border-radius:50%;background:#444;transition:.3s;flex-shrink:0}
#dot.on{background:#00e08a;box-shadow:0 0 8px #00e08a}
#title{font-size:15px;font-weight:600;letter-spacing:.2px;white-space:nowrap}
.badge{background:#17324f;color:#7cccff;padding:3px 11px;border-radius:11px;
  font-size:11.5px;white-space:nowrap;border:1px solid #24486e;font-weight:600}
#hdr-spacer{margin-left:auto}
#estop{background:#7f1d1d;border:1px solid #b91c1c;color:#fecaca;padding:7px 16px;
  border-radius:9px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap}
#estop.engaged{background:#dc2626;color:#fff;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.55}}
/* ── Shell ── */
/* ‼️ dvh, not vh. On iOS Safari 100vh is the height the page WOULD have with
   the address bar hidden — it is taller than what you can actually see. With
   the tab bar at the bottom on mobile (column-reverse) that put the tabs
   underneath Safari's bottom bar, and html{overflow:hidden} meant you could not
   even scroll down to find them: "μπαίνει αλλά δεν βλέπω tabs" (2026-08-02).
   100dvh tracks the visible area as that bar shows and hides (iOS 15.4+); the
   vh line above stays as the fallback for anything older. */
#shell{display:flex;height:calc(100vh - 48px);height:calc(100dvh - 48px)}
#tabs{width:150px;background:#1a1a1e;border-right:1px solid #2c2c32;
  padding:8px 0;flex-shrink:0;overflow-y:auto}
.tab{display:flex;align-items:center;gap:9px;padding:10px 14px;cursor:pointer;
  font-size:13px;color:#9b9ba3;border-left:3px solid transparent;user-select:none}
.tab:hover{background:#212127;color:#d4d4d8}
.tab.active{background:#212127;color:#fff;border-left-color:#3b82f6}
.tab .ic{font-size:16px;width:20px;text-align:center}
#panes{flex:1;position:relative;overflow:hidden}
.pane{position:absolute;inset:0;display:none;padding:10px;overflow:auto}
.pane.active{display:flex;flex-direction:column;gap:10px}
/* A soft inner highlight and a real shadow give the cards depth instead of
   reading as flat rectangles on a flat background. */
.card{background:linear-gradient(#1f1f24,#1a1a1e);border:1px solid #2f2f37;
  border-radius:14px;padding:13px 15px;
  box-shadow:0 1px 0 rgba(255,255,255,.03) inset,0 2px 10px rgba(0,0,0,.35)}
.card h3{font-size:11.5px;font-weight:700;color:#9a9aa4;text-transform:uppercase;
  letter-spacing:.9px;margin-bottom:11px;cursor:pointer;position:relative;
  padding-right:18px;user-select:none}
/* The chevron is the only affordance that a card folds. It rotates rather than
   swapping glyphs so the state is obvious mid-animation. */
.card>h3::after{content:'\2304';position:absolute;right:0;top:-2px;
  color:#5a5a66;font-size:15px;transition:transform .15s}
.card.collapsed>h3::after{transform:rotate(-90deg)}
.card.collapsed>h3{margin-bottom:0}
.card.collapsed>*:not(h3){display:none!important}
/* Cards holding a viewer can also be dragged taller. Native resize is pointer
   only — folding is what works on a phone, which is why both exist. */
.card.sizable{resize:vertical;overflow:auto;min-height:130px}
/* ‼️ A card that grows to fill the pane must still never collapse to nothing.
   The pane is a scrolling flex column, and `flex:1` combined with
   `overflow:auto` resolves min-height to 0 — so as soon as other cards were
   added above it, the growing card vanished entirely. That is what happened to
   the map list once the network and Bluetooth cards went in: it was there, at
   zero pixels, at the bottom. The floor makes it always visible and lets the
   PANE scroll instead. */
.card.grow{flex:1 1 auto;overflow:auto;min-height:220px}
/* ‼️ A card must never be squeezed smaller than what is inside it. The pane is
   a flex column, so by default every card shrinks to make the pane fit — and a
   card holding a 300px canvas happily rendered at 130px, showing the top
   quarter of the arm and nothing else. getBoundingClientRect on the canvas
   still reported the full height, which is why this hid so well. Same silent
   crop on the IMU rose and the pointing ring. The pane already scrolls; let it.
   :not(.grow) because that class deliberately gives its space back. */
.pane>.card:not(.grow){flex-shrink:0}
/* Drag bar under a viewer. Native CSS resize is pointer-only and most of the
   use here is a phone, so this is a real handle with touch events. */
.grip{height:16px;margin:-2px 0 7px;border-radius:8px;cursor:ns-resize;
  background:#232329;border:1px solid #2f2f37;display:flex;
  align-items:center;justify-content:center;flex:0 0 auto;touch-action:none}
.grip::after{content:'';width:34px;height:3px;border-radius:2px;background:#4a4a56}
.grip:active{background:#2c2c34}
.grip:active::after{background:#67c4ff}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.btn{background:linear-gradient(#2e2e36,#26262d);border:1px solid #3d3d48;
  color:#d9d9de;padding:8px 15px;border-radius:10px;cursor:pointer;font-size:13px;
  user-select:none;white-space:nowrap;transition:background .12s,border-color .12s}
.btn:hover{background:linear-gradient(#35353e,#2b2b33);border-color:#4a4a57}
.btn:active{transform:translateY(1px)}
.btn:disabled{opacity:.45;cursor:default}
.btn:hover{background:#33333c}
.btn:active{background:#3d3d47}
.btn.pri{background:#1d4ed8;border-color:#2563eb;color:#fff}
.btn.warn{background:#7c2d12;border-color:#9a3412;color:#fed7aa}
/* Flash the row a click-to-pick landed on, so the eye finds it in a list
   sorted by name rather than by map position. Fade-out only (transition on
   the base rule), instant on entry (transition:none on .picked itself). */
#room-edit>[data-room]{transition:background-color 1.3s;border-radius:8px}
#room-edit>[data-room].picked{background-color:#166534;transition:none}
.grid2{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px}
.k{color:#71717a}.v{font-family:ui-monospace,Menlo,monospace;text-align:right;color:#d4d4d8}
.pill{display:inline-block;padding:2px 9px;border-radius:9px;font-size:11px;
  background:#27272e;color:#a1a1aa}
.pill.ok{background:#052e1a;color:#4ade80}
.pill.bad{background:#450a0a;color:#f87171}
/* ── Safety tab: one row per clearance ──
   ‼️ sf- prefixed, NOT .srow/.slab. `.srow` is already the map tab's speed
   sliders (a 96px/1fr/74px grid, right above); reusing the name put every
   label in a 96px column with the help text beside it as a second column, and
   the whole card read as broken on a phone. Measured in WebKit at 390x664 —
   test_dashboard_safety_tab.py::test_the_safety_rows_do_not_reuse_map_classes.
   The value sits on the label's line (right-aligned, monospace) so a column of
   numbers is scannable; the slider gets the full width underneath, because on
   a phone a slider sharing a line with text is too short to aim. */
.sfrow{padding:11px 0;border-top:1px solid #27272e}
.sfrow:first-of-type{border-top:none}
.sfrow.off{opacity:.45}
.sflab{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
  font-size:12.5px;color:#e4e4e7}
.sfval{font-family:ui-monospace,Menlo,monospace;color:#7cccff;font-size:12px;
  white-space:nowrap;flex:0 0 auto}
.sfrow input[type=range]{width:100%;margin:9px 0 0;-webkit-appearance:none;
  appearance:none;height:6px;border-radius:4px;background:#3f3f46;outline:none}
.sfrow input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;
  appearance:none;width:20px;height:20px;border-radius:50%;background:#3b82f6;
  cursor:pointer;border:2px solid #18181b}
.sfrow input[type=range]::-moz-range-thumb{width:20px;height:20px;
  border-radius:50%;background:#3b82f6;cursor:pointer;border:2px solid #18181b}
.sfrow input[type=color]{width:52px;height:30px;margin:8px 0 0;padding:0;
  border:1px solid #3f3f46;border-radius:6px;background:none;cursor:pointer}
.sfhelp{font-size:11.5px;color:#71717a;line-height:1.6;margin-top:6px}
.sfwarn{font-size:11.5px;color:#fbbf24;line-height:1.6;margin-top:6px;display:none}
.sfrow.warned .sfwarn{display:block}
.sftog{display:flex;align-items:center;gap:9px;cursor:pointer;user-select:none}
/* ── System tab: bars, temperatures, disks ── */
.mrow{display:grid;grid-template-columns:64px 1fr auto;gap:10px;align-items:center;
  margin-bottom:9px;font-size:12.5px}
.mrow .lbl{color:#71717a}
.mrow .val{font-family:ui-monospace,Menlo,monospace;color:#d4d4d8;text-align:right;
  white-space:nowrap;font-size:12px}
.bar{height:8px;background:#27272e;border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:5px;background:#3b82f6;
  transition:width .35s ease,background .35s ease}
.bar>i.warn{background:#f59e0b}
.bar>i.bad{background:#ef4444}
/* One cell per core. Fixed 7px columns so 16 threads fit a phone without
   wrapping into something that looks like a second CPU. */
.cores{display:flex;gap:2px;margin:2px 0 11px}
.cores>i{flex:1;min-width:3px;height:14px;background:#27272e;border-radius:2px;
  position:relative;overflow:hidden}
.cores>i>b{position:absolute;bottom:0;left:0;right:0;background:#3b82f6;
  transition:height .35s ease}
.tempgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px}
.tcell{background:#232329;border:1px solid #2c2c32;border-radius:9px;padding:8px 10px}
.tcell .tn{color:#71717a;font-size:11px;display:flex;gap:5px;align-items:center}
.tcell .tv{font-family:ui-monospace,Menlo,monospace;font-size:17px;color:#e4e4e7;
  margin-top:3px}
.tcell.warn{border-color:#7c2d12}.tcell.warn .tv{color:#fbbf24}
.tcell.bad{border-color:#7f1d1d}.tcell.bad .tv{color:#f87171}
/* ── fall alert banner ── */
#fall-bar{display:flex;gap:11px;align-items:center;background:#450a0a;
  border:1px solid #b91c1c;color:#fecaca;border-radius:11px;padding:11px 14px;
  margin-bottom:10px;animation:fallpulse 1.6s ease-in-out infinite}
@keyframes fallpulse{0%,100%{border-color:#b91c1c}50%{border-color:#f87171}}
/* Respect the OS setting — a pulsing red bar is exactly the kind of motion
   people turn animations off for. */
@media (prefers-reduced-motion: reduce){#fall-bar{animation:none}}
/* ── speed sliders (map tab) ── */
.speedbox{margin-top:11px;padding-top:11px;border-top:1px solid #2c2c32}
.srow{display:grid;grid-template-columns:96px 1fr 74px;gap:10px;align-items:center;
  margin-bottom:8px;font-size:12.5px}
.srow .lbl{color:#a1a1aa}
.srow .val{font-family:ui-monospace,Menlo,monospace;color:#d4d4d8;text-align:right;
  font-size:12px}
.srow input[type=range]{-webkit-appearance:none;appearance:none;height:6px;
  border-radius:4px;background:#3f3f46;outline:none}
.srow input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
  width:20px;height:20px;border-radius:50%;background:#3b82f6;cursor:pointer;
  border:2px solid #18181b}
.srow input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;
  background:#3b82f6;cursor:pointer;border:2px solid #18181b}
.pill.warn{background:#422006;color:#fbbf24}
/* ── Map ── */
#map-wrap{position:relative;background:#0c0c0e;border:1px solid #2c2c32;
  border-radius:12px;overflow:hidden;cursor:crosshair;flex:1;min-height:200px}
#map-canvas{width:100%;height:100%;display:block}
/* ‼️ Square, and capped. The costmap is a 60x60 grid of a 3x3 m window, so
   flex:1 stretched it across a 1200x450 pane: every cell became a ~20x7 px
   slab, the window read as a rectangle when it is a square, and the whole
   thing was far bigger than it needed to be to be legible. Sizing by HEIGHT
   with aspect-ratio (width:auto) keeps it square through the drag grip too,
   which pins an explicit height and would otherwise squash it again. */
#cost-wrap{position:relative;background:#0c0c0e;border:1px solid #2c2c32;
  border-radius:12px;overflow:hidden;flex:0 0 auto;align-self:center;
  aspect-ratio:1;height:min(58vh,440px);width:auto;max-width:100%;min-height:200px}
#cost-canvas{width:100%;height:100%;display:block;
  /* 60x60 cells blown up to a phone screen: keep the cells as crisp squares
     rather than letting the browser smear them into a watercolour. */
  image-rendering:pixelated}
.ovl{position:absolute;top:8px;left:10px;font-size:10px;color:#6b6b73;
  background:rgba(0,0,0,.55);padding:3px 8px;border-radius:5px;pointer-events:none}
/* ── Camera ── */
#cam{width:100%;height:100%;object-fit:contain;display:block;background:#0c0c0e}
#cam-wrap{position:relative;flex:1;min-height:200px;background:#0c0c0e;
  border:1px solid #2c2c32;border-radius:12px;overflow:hidden}
/* ── VNC ── */
.vnc-host{flex:1;position:relative;background:#0c0c0e;border:1px solid #2c2c32;
  border-radius:12px;overflow:hidden;min-height:240px}
.vnc-host iframe{width:100%;height:100%;border:0;display:block}
/* Fullscreen. Measured 2026-08-08 in WebKit: RViz runs 1600x900, so on a
   390pt portrait phone the 16:9 image letterboxes to 374x210 and fills only
   43% of the tab — the rest is black. Height was never the problem (486px
   sat unused); the WIDTH is what caps the scale, which is why growing the
   pane changes nothing and only reclaiming the whole viewport does.
   ‼️ position:fixed, NOT Element.requestFullscreen. iPhone Safari does not
   implement that API for anything but <video> — a native-only version is a
   dead button on exactly the device this was asked for. The native call is
   attempted underneath purely as a bonus (it also hides the browser chrome)
   and its absence is not an error. */
.vnc-host.fs{position:fixed;inset:0;z-index:9000;border:0;border-radius:0;
  min-height:0;margin:0}
.vnc-fs-exit{position:fixed;z-index:9001;
  top:calc(env(safe-area-inset-top,0px) + 10px);
  right:calc(env(safe-area-inset-right,0px) + 10px);
  background:rgba(0,0,0,.62);color:#e4e4e7;border:1px solid #3a3a44;
  border-radius:9px;padding:9px 13px;font-size:13px;cursor:pointer;
  /* A 44pt target is the iOS minimum that a thumb hits reliably. */
  min-width:44px;min-height:44px;line-height:1}
/* Mirrors noVNC's own error text out of the iframe so it can be read/copied */
.vnc-err{position:absolute;top:0;left:0;right:0;z-index:5;padding:8px 12px;
  background:#7f1d1d;color:#fecaca;font-size:12px;line-height:1.5;
  font-family:ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;
  max-height:45%;overflow:auto}
.vnc-msg{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:14px;text-align:center;padding:20px;
  color:#a1a1aa;font-size:13px;line-height:1.6}
/* ── Drive pad ── */
#dpad{display:grid;grid-template-columns:repeat(3,50px);grid-template-rows:repeat(3,50px);gap:5px}
.dbtn{background:#2a2a31;border:1px solid #3a3a44;border-radius:9px;display:flex;
  align-items:center;justify-content:center;cursor:pointer;font-size:19px;
  user-select:none;-webkit-user-select:none;touch-action:none}
.dbtn:active{background:#3d3d47}
/* Keyboard driving has no :active, so a held key gets its own lit state —
   without it there is no feedback that the arrow key was even received. */
.dbtn.lit{background:#1d4ed8;border-color:#3b82f6;color:#fff}
.dbtn.ghost{background:transparent;border:none;pointer-events:none}
#bstop{background:#7f1d1d;border-color:#b91c1c;font-size:12px;font-weight:700;color:#fecaca}
/* ── Arm ── */
.joint{display:grid;grid-template-columns:74px 1fr 62px;gap:11px;align-items:center;
  margin-bottom:9px}
.joint label{font-size:13px;color:#a1a1aa}
.joint input[type=range]{width:100%;accent-color:#3b82f6}
.joint .val{font-family:ui-monospace,Menlo,monospace;font-size:12px;text-align:right;color:#d4d4d8}
.joint.off{opacity:.45}
/* ── Chat ── */
#chat{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;min-height:150px}
.msg{max-width:78%;padding:8px 12px;border-radius:13px;font-size:13.5px;line-height:1.45;
  white-space:pre-wrap;word-break:break-word}
.msg.user{align-self:flex-end;background:#1d4ed8;color:#fff;border-bottom-right-radius:4px}
.msg.robot{align-self:flex-start;background:#27272e;border-bottom-left-radius:4px}
.msg.wake{align-self:center;background:transparent;color:#6b6b73;font-size:11px;padding:2px}
#chat-in{display:flex;gap:8px}
#chat-in input{flex:1;background:#27272e;border:1px solid #3a3a44;color:#e4e4e7;
  padding:10px 13px;border-radius:9px;font-size:14px;outline:none}
#chat-in input:focus{border-color:#3b82f6}
/* ── Mobile ── */
@media(max-width:760px){
  #shell{flex-direction:column-reverse}
  /* ‼️ WRAP, never scroll. This was one row with overflow-x:auto, and on iOS
     Safari that is invisible: the scrollbar is not rendered until a scroll is
     already in progress, so there is no hint the row continues. On an iPhone
     only 7 of the 12 tabs fit — Gazebo, Σύστημα, Log and Ρυθμίσεις were simply
     unreachable, and the dashboard read as "it only shows the first page"
     (reported 2026-08-02, reproduced in WebKit at 428pt).
     Wrapped rows cost ~40px each and need no gesture to discover.

     flex-GROW is 0 on purpose. With grow:1 a partly-filled last row stretches
     its cells to fill the width, so at 16 tabs the bottom four were half again
     as wide as the twelve above them and the bar read as broken. Fixed basis
     means every cell is the same size and a short last row simply ends early,
     which looks deliberate.

     The basis only has to keep a cell wider than ~45pt on a 320pt iPhone SE —
     below that the icon and the label collide.

     Was 16.66% (six per row). At 21 tabs that is four rows = 214px, which on a
     568pt iPhone SE is 38% of the screen given to navigation. 14.28% is seven
     per row, so the same 21 tabs need three rows (~160px) and a cell is still
     45.7pt on an SE — measured in WebKit, labels ellipsis rather than collide.
     Past this the basis cannot shrink further and the dashboard needs real
     navigation instead of a flat bar. */
  /* The home indicator sits over the last row on a notched iPhone; without the
     inset the bottom row's labels are half-covered by it. */
  #tabs{width:100%;height:auto;display:flex;flex-wrap:wrap;padding:0;
    border-right:none;border-top:1px solid #2c2c32;
    padding-bottom:env(safe-area-inset-bottom,0px)}
  .tab{flex:0 0 14.28%;flex-direction:column;gap:2px;padding:6px 2px;
    font-size:9.5px;border-left:none;border-top:3px solid transparent;
    justify-content:center;text-align:center;min-width:0}
  /* Long labels (Costmap, Σπίτι 3D) must shrink, not widen the cell and
     push the row back into overflow. */
  .tab>span:last-child{overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;max-width:100%}
  .tab.active{border-left-color:transparent;border-top-color:#3b82f6}
  #title{display:none}
  .pane{padding:8px}
}
</style>
</head>
<body>
<div id="hdr">
  <div id="dot"></div>
  <span id="title">🤖 Home Robot</span>
  <span class="badge" id="room-badge">—</span>
  <span id="hdr-spacer"></span>
  <button id="estop">■ STOP</button>
</div>
<div id="shell">
  <nav id="tabs"></nav>
  <div id="panes">

    <!-- ── Map ─────────────────────────────────────────────────── -->
    <!-- Fall alert. Outside the panes on purpose: it must be visible whichever
         tab is open, because nobody watches the map tab waiting for it. -->
    <div id="fall-bar" style="display:none">
      <span style="font-size:20px">🚨</span>
      <div style="flex:1;min-width:0">
        <div style="font-weight:600">Κάποιος μπορεί να έπεσε</div>
        <div id="fall-detail" style="font-size:11.5px;opacity:.85"></div>
      </div>
      <button class="btn" id="fall-ack">Το είδα</button>
    </div>

    <section class="pane active" id="p-map">
      <div class="row" style="margin-bottom:8px">
        <button class="btn pri" id="map-view-2d">🗺️ 2D</button>
        <button class="btn" id="map-view-scan">📸 Σάρωμα</button>
      </div>
      <div id="map-wrap">
        <canvas id="map-canvas"></canvas>
        <div class="ovl">ΧΑΡΤΗΣ · κλικ για πλοήγηση</div>
      </div>
      <div class="card" id="map-scan3d-card" style="display:none">
        <h3>Φωτορεαλιστικό σάρωμα <span class="badge" id="scan3d-info">—</span></h3>
        <canvas id="scan3d" style="width:100%;height:min(58vh,600px);min-height:260px;
          background:#0a0a0b;border-radius:8px;touch-action:none;cursor:grab;
          display:block"></canvas>
        <div style="color:#71717a;font-size:11.5px;margin-top:6px">
          Σύρε για περιστροφή · ροδέλα για ζουμ · κλικ πάνω στο σπίτι στέλνει το ρομπότ εκεί ·
          πραγματικό υφασμένο mesh (τηλέφωνο), όχι το occupancy grid — το AMCL εντοπίζεται
          στον 2D χάρτη, εδώ βλέπεις μόνο την ίδια θέση πάνω στην πραγματική όψη
        </div>
      </div>
      <div class="card grow">
        <h3>Χάρτες <span class="badge" id="map-active">—</span></h3>
        <div id="map-list" style="margin:8px 0"></div>
        <div class="row" style="margin-top:10px">
          <button class="btn pri" id="b-map-new">🆕 Νέος χάρτης (SLAM)</button>
          <input id="map-save-name" placeholder="όνομα χάρτη"
                 style="flex:1;min-width:120px;background:#18181b;border:1px solid #27272a;
                        color:#e4e4e7;border-radius:8px;padding:8px 10px">
          <button class="btn" id="b-map-save">💾 Αποθήκευση</button>
        </div>
        <div id="map-msg" style="color:#71717a;font-size:11.5px;margin-top:8px"></div>
        <div id="map-straighten" style="display:none;margin-top:12px;padding-top:12px;
                                        border-top:1px solid #27272a">
          <div style="font-size:12.5px;color:#a1a1aa;margin-bottom:8px">
            🧹 Καθαρή εκδοχή — ισιώνει τα σκαλοπάτια των τοίχων. <b>Μόνο εμφάνιση</b> μέχρι
            να διαλέξεις: μπορεί να μετατοπίσει τοίχους λίγα εκατοστά ή να αφαιρέσει ένα
            πραγματικό εμπόδιο που έμοιαζε με θόρυβο σάρωσης — σύγκρινε οπτικά πριν διαλέξεις.
          </div>
          <div class="row" style="gap:12px">
            <div style="flex:1;text-align:center;min-width:0">
              <div style="font-size:11px;color:#71717a;margin-bottom:4px">Πρωτότυπο</div>
              <img id="map-straighten-orig" style="max-width:100%;border-radius:6px;
                                                    border:1px solid #27272a">
            </div>
            <div style="flex:1;text-align:center;min-width:0">
              <div style="font-size:11px;color:#71717a;margin-bottom:4px">Ισιωμένο</div>
              <img id="map-straighten-clean" style="max-width:100%;border-radius:6px;
                                                     border:1px solid #27272a">
            </div>
          </div>
          <div class="row" style="margin-top:10px">
            <button class="btn" id="b-map-straighten-keep">Κράτησε το πρωτότυπο</button>
            <button class="btn pri" id="b-map-straighten-use">Χρήση ισιωμένης εκδοχής</button>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>Δωμάτια
          <label style="float:right;font-size:11.5px;color:#a1a1aa;font-weight:400;
            cursor:pointer;user-select:none">
            <input type="checkbox" id="b-tint" checked style="vertical-align:-2px">
            Χρώματα
          </label>
        </h3>
        <div class="row" id="rooms"></div>
        <div class="row" id="room-legend" style="margin-top:8px;gap:11px"></div>
        <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
          cursor:pointer;user-select:none;margin-top:8px">
          <input type="checkbox" id="b-pick-room" style="vertical-align:-2px">
          🖱️ Κλικ στον χάρτη επιλέγει δωμάτιο (αντί να στέλνει το ρομπότ εκεί)
        </label>
        <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
          cursor:pointer;user-select:none;margin-top:6px">
          <input type="checkbox" id="b-place-room" style="vertical-align:-2px">
          ➕ Κλικ στον χάρτη ΠΡΟΣΘΕΤΕΙ δωμάτιο εδώ, με το όνομα/χρώμα από κάτω
        </label>
        <div class="row" id="place-room-row" style="margin-top:6px;gap:8px;display:none">
          <input type="color" id="pr-color" value="#cc44ff"
            style="width:34px;height:30px;padding:0;border:none;background:none;flex:0 0 auto">
          <input type="text" id="pr-name" placeholder="π.χ. κρεβατοκάμαρα"
            style="flex:1;min-width:100px;background:#232329;border:1px solid #2c2c32;
            border-radius:8px;color:#e4e4e7;padding:6px 9px;font-size:12.5px">
        </div>
        <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
          cursor:pointer;user-select:none;margin-top:6px">
          <input type="checkbox" id="b-place-room-rect" style="vertical-align:-2px">
          ⬜ 2 κλικ στον χάρτη ζωγραφίζουν δωμάτιο σαν τετράγωνο (απέναντι γωνίες)
        </label>
        <div class="row" id="place-room-rect-row" style="margin-top:6px;gap:8px;display:none">
          <input type="color" id="prr-color" value="#cc44ff"
            style="width:34px;height:30px;padding:0;border:none;background:none;flex:0 0 auto">
          <input type="text" id="prr-name" placeholder="π.χ. κρεβατοκάμαρα"
            style="flex:1;min-width:100px;background:#232329;border:1px solid #2c2c32;
            border-radius:8px;color:#e4e4e7;padding:6px 9px;font-size:12.5px">
        </div>
        <div id="room-edit" style="margin-top:10px"></div>
        <div class="row" style="margin-top:8px" id="room-edit-row">
          <button class="btn" id="b-room-save">💾 Αποθήκευση ονομάτων/χρωμάτων</button>
          <span id="room-edit-msg" style="font-size:11.5px;color:#71717a"></span>
        </div>
        <div class="speedbox" style="margin-top:10px">
          <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
            cursor:pointer;user-select:none">
            <input type="checkbox" id="b-kz-add" style="vertical-align:-2px">
            🚫 2 κλικ στον χάρτη ορίζουν απαγορευμένη ζώνη (απέναντι γωνίες)
          </label>
          <div class="row" id="kz-add-row" style="margin-top:6px;gap:8px;display:none">
            <input type="text" id="kz-name" placeholder="π.χ. χαλάκι σκύλου"
              style="flex:1;min-width:100px;background:#232329;border:1px solid #2c2c32;
              border-radius:8px;color:#e4e4e7;padding:6px 9px;font-size:12.5px">
          </div>
          <div class="row" id="kz-list" style="margin-top:8px;flex-direction:column;
            align-items:stretch;gap:4px"></div>
          <div class="row" style="margin-top:8px">
            <button class="btn warn" id="kz-on">🚫 Ενεργοποίηση ζωνών</button>
            <button class="btn" id="kz-off">✅ Απενεργοποίηση</button>
            <span id="kz-msg" style="font-size:11.5px;color:#71717a"></span>
          </div>
          <p style="font-size:11px;color:#71717a;margin-top:6px;line-height:1.5">
            Χρειάζεται επανεκκίνηση (~90s) για να πιάσει η αλλαγή — το Nav2 διαβάζει
            τις ζώνες μόνο στην εκκίνηση.
          </p>
        </div>
        <div class="speedbox" style="margin-top:10px">
          <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
            cursor:pointer;user-select:none">
            <input type="checkbox" id="b-slipmap">
            <span>🩹 Πού γλιστράει <span class="badge" id="sm-badge">—</span></span>
          </label>
          <div id="sm-msg" style="font-size:11.5px;color:#71717a;margin-top:8px;
            line-height:1.6">Δείχνει τα σημεία του σπιτιού όπου το AMCL χρειάστηκε να διορθώσει περισσότερο τη θέση — δηλαδή εκεί που η οδομετρία χάνει έδαφος. Μαζεύεται με τον καιρό και επιβιώνει σε restart. Ένα φωτεινό σημείο σε ένα χαλί είναι πρόβλημα που φτιάχνεται· παντού λίγο, είναι βαθμονόμηση.</div>
        </div>
        <div class="speedbox">
          <div class="row">
            <span style="font-size:12.5px;color:#a1a1aa">🔎 Πήγαινε να δεις</span>
            <select id="ck-room" class="btn" style="padding:6px 9px"></select>
            <input id="ck-q" placeholder="π.χ. είναι κλειστό το παράθυρο;"
              style="flex:1;min-width:140px;background:#232329;border:1px solid #2c2c32;
              border-radius:8px;color:#e4e4e7;padding:7px 10px;font-size:12.5px"
              autocomplete="off">
            <button class="btn pri" id="ck-go">Έλεγξε</button>
            <button class="btn" id="ck-stop">✕</button>
            <button class="btn" id="sort-go" title="Μαζεύει αντικείμενα με τον βραχίονα">🧺 Μάζεψε</button>
          </div>
          <div id="ck-msg" style="font-size:11.5px;color:#71717a;margin-top:6px"></div>
        </div>
      </div>
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <div id="dpad">
            <div class="dbtn ghost"></div><div class="dbtn" id="bf">▲</div><div class="dbtn ghost"></div>
            <div class="dbtn" id="bl">◄</div><div class="dbtn" id="bstop">STOP</div><div class="dbtn" id="br">►</div>
            <div class="dbtn ghost"></div><div class="dbtn" id="bb">▼</div><div class="dbtn ghost"></div>
          </div>
          <div class="grid2" style="flex:1;min-width:150px">
            <span class="k">X</span><span class="v" id="ix">—</span>
            <span class="k">Y</span><span class="v" id="iy">—</span>
            <span class="k">Γωνία</span><span class="v" id="iyaw">—</span>
            <span class="k">Ταχύτητα</span><span class="v" id="ivel">—</span>
          </div>
          <div class="row" style="flex-direction:column;align-items:stretch">
            <button class="btn pri" id="b-loc">🔍 Εντοπισμός</button>
            <button class="btn" id="b-xnav">✕ Ακύρωση στόχου</button>
          </div>
        </div>
        <div class="speedbox">
          <div class="srow">
            <span class="lbl">🚀 Ταχύτητα</span>
            <input type="range" id="sp-lin" min="0.05" max="0.30" step="0.01">
            <span class="val" id="sp-linv">—</span>
          </div>
          <div class="srow">
            <span class="lbl">🔄 Στροφή</span>
            <input type="range" id="sp-ang" min="0.35" max="1.20" step="0.05">
            <span class="val" id="sp-angv">—</span>
          </div>
          <div class="row" style="margin-top:2px">
            <button class="btn" id="sp-slow">🐢 Αργά</button>
            <button class="btn" id="sp-def">Προεπιλογή</button>
            <button class="btn" id="sp-fast">🐇 Γρήγορα</button>
            <span id="sp-note" style="font-size:11px;color:#71717a"></span>
          </div>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ⌨️ Και από πληκτρολόγιο: βελάκια ή WASD (κράτα πατημένο), space = στοπ. Αφήνοντας το πλήκτρο σταματά· αν χαθεί το tab ή το δίκτυο, η βάση σταματά μόνη της σε 0.25s.
        </p>
      </div>
    </section>

    <!-- ── Camera ──────────────────────────────────────────────── -->
    <section class="pane" id="p-cam">
      <div id="cam-wrap">
        <img id="cam" alt="">
        <div class="ovl">📷 RealSense D435 · color</div>
        <!-- ‼️ A healthy camera pointed at a blank wall is indistinguishable
             from a dead one, and was reported as "δεν δείχνει" twice while
             running at 23 fps. This says which it is. -->
        <div class="ovl" id="cam-why" style="top:auto;bottom:8px;left:10px;
             right:10px;display:none;background:rgba(120,53,15,.85);
             color:#fbbf24;font-size:11.5px;line-height:1.5"></div>
      </div>
      <div class="card">
        <h3>Ανίχνευση <span class="badge" id="vis-count">—</span></h3>
        <div class="row">
          <button class="btn pri" id="b-overlay">👁 Πλαίσια/σκελετός</button>
          <button class="btn" id="b-follow">👣 Ακολούθησέ με</button>
          <button class="btn warn" id="b-follow-stop">■ Σταμάτα να ακολουθείς</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Τα πράσινα πλαίσια είναι άνθρωποι, τα πορτοκαλί αντικείμενα· ο κίτρινος σκελετός είναι 17 σημεία COCO. ‼️ Χρειάζονται τα perception nodes — ξεκίνα με «robot max use_perception:=true», αλλιώς η εικόνα μένει καθαρή και ο μετρητής στο 0. Το «Ακολούθησέ με» σταματά μόνο του μετά από 30 δευτερόλεπτα.
        </p>
      </div>
      <div class="card">
        <h3>Πρόσθετη αντίληψη</h3>
        <div class="row">
          <button class="btn" id="b-pose-toggle">Pose/χειρονομίες</button>
          <span class="pill" id="pose-pill">—</span>
        </div>
        <div class="row" style="margin-top:8px">
          <button class="btn" id="b-apriltag-toggle">AprilTag</button>
          <span class="pill" id="apriltag-pill">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Σβηστά γλιτώνουν CPU όταν δεν τα χρειάζεσαι (pose+χειρονομίες ~25%, AprilTag ~29% σε αυτό το μηχάνημα). Το AprilTag χρειάζεται για επαναδιόρθωση θέσης στο σαλόνι μετά από μεγάλο drift.
        </p>
      </div>
      <div class="card">
        <h3>Τι βλέπει</h3>
        <div id="objects" style="font-size:13px;color:#a1a1aa;line-height:1.6">—</div>
      </div>
      <div class="card">
        <h3>Κατάσταση περιβάλλοντος</h3>
        <div id="situation" style="font-size:12.5px;color:#a1a1aa;line-height:1.6">—</div>
      </div>
    </section>

    <!-- ── RViz / MoveIt / Gazebo / RTAB-Map ───────────────────── -->
    <section class="pane" id="p-rviz"></section>
    <section class="pane" id="p-moveit"></section>
    <section class="pane" id="p-gazebo"></section>
    <section class="pane" id="p-rtabmap"></section>

    <!-- ── Arm ─────────────────────────────────────────────────── -->
    <section class="pane" id="p-arm">
      <div class="card">
        <h3>3D <span class="badge" id="arm3d-info">—</span>
          <button class="btn" id="b-arm3d-reset" style="float:right">Επαναφορά όψης</button>
        </h3>
        <!-- ‼️ Sized in vh, not a fixed 300px: on a laptop the arm pane holds six
             cards, so a short viewer left the gripper cropped off the bottom even
             though the canvas itself was fine. The grip below still overrides it. -->
        <canvas id="arm3d" style="width:100%;height:min(52vh,560px);min-height:260px;
          background:#0a0a0b;border-radius:8px;touch-action:none;cursor:grab;
          display:block"></canvas>
        <div style="color:#71717a;font-size:11.5px;margin-top:6px">
          Σύρε για περιστροφή · ροδέλα για ζουμ · κινείται με τις αρθρώσεις από κάτω
        </div>
      </div>
      <div class="card">
        <h3>Αρθρώσεις</h3>
        <div id="joints"></div>
      </div>
      <div class="card">
        <h3>Όρια &amp; ταχύτητα <span class="badge" id="armlim-badge">—</span></h3>
        <div id="armlim"></div>
        <div class="joint" style="margin-top:6px">
          <label>ταχύτητα</label>
          <input type="range" id="armspeed" min="0.5" max="1.5" step="0.05" value="1.0">
          <span class="val" id="armspeed-v">100%</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Η ταχύτητα ρυθμίζει ΚΑΙ το χειριστήριο (jog) ΚΑΙ πόσο απότομα ξεκινούν
          οι κινήσεις από εδώ — όχι το ανώτατο όριο του σερβοκινητήρα, που δεν
          έχει μετρηθεί ασφαλές να ανέβει. Τα όρια των αρθρώσεων δεν μπορούν
          ποτέ να ξεπεράσουν το μηχανικό όριο του σερβοκινητήρα.
        </p>
        <div class="row" style="margin-top:10px">
          <button class="btn warn" id="b-armlim-reset">↺ Επαναφορά προεπιλογών</button>
        </div>
      </div>
      <div class="card">
        <h3>Δαγκάνα</h3>
        <div class="joint">
          <label>άνοιγμα</label>
          <input type="range" id="grip" min="1.08" max="3.14" step="0.01" value="3.14">
          <span class="val" id="grip-v">—</span>
        </div>
        <div class="row">
          <button class="btn" id="b-grip-open">✋ Άνοιγμα</button>
          <button class="btn" id="b-grip-close">🤏 Κλείσιμο</button>
        </div>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Αφή <span class="badge" id="tc-badge">—</span></h3>
        <div class="grid2">
          <span class="k">Πίεση</span><span class="v" id="tc-excess">—</span>
          <span class="k">Σκληρότητα</span><span class="v" id="tc-hard">—</span>
          <span class="k">Βάρος</span><span class="v" id="tc-weight">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Ο βραχίονας στέλνει ΦΟΡΤΙΟ ανά άρθρωση δίπλα σε κάθε γωνία, και μέχρι
          τώρα δεν το διάβαζε κανείς. Η επαφή είναι ΣΚΑΛΟΠΑΤΙ πάνω από το
          φορτίο που κουβαλά κινούμενος στον αέρα· η σκληρότητα είναι φορτίο
          ανά χιλιοστό διαδρομής· το βάρος είναι η διαφορά στον ώμο με κλειστή
          και ανοιχτή δαγκάνα.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ‼️ Οι τιμές είναι ΑΚΑΤΕΡΓΑΣΤΕΣ μονάδες σερβοκινητήρα, όχι γραμμάρια.
          Συγκρίνονται μεταξύ τους και με τίποτα άλλο — γι' αυτό λέει «βαρύ»
          και όχι «180 γραμμάρια». Η μέτρηση είναι ΠΑΘΗΤΙΚΗ: δεν δίνει ποτέ
          εντολή στον βραχίονα.
        </p>
      </div>
      <div class="card">
        <h3>Εντολές</h3>
        <div class="row">
          <button class="btn" id="b-arm-home">🏠 Θέση ανάπαυσης</button>
          <button class="btn warn" id="b-arm-limp">💤 Χαλάρωση (T:210)</button>
          <button class="btn" id="b-arm-init">⚡ Επαναφορά ροπής</button>
          <button class="btn pri" id="b-arm-moveit">🎯 Άνοιγμα MoveIt</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ Στον ώμο το «πάνω» είναι η ΑΡΝΗΤΙΚΗ φορά. Τα όρια είναι τα μετρημένα
          στο χέρι (31/07), όχι του κατασκευαστή. Η χαλάρωση κόβει τη ροπή —
          κράτα τον βραχίονα πριν την πατήσεις.
        </p>
      </div>
    </section>

    <!-- ── Vacuum ──────────────────────────────────────────────── -->
    <section class="pane" id="p-base">
      <div class="card">
        <h3>Βάση Roomba 879</h3>
        <div class="grid2">
          <span class="k">Κατάσταση</span><span class="v" id="r-awake">—</span>
          <span class="k">Σύνδεση</span><span class="v" id="r-link">—</span>
          <span class="k">OI mode</span><span class="v" id="r-mode">—</span>
          <span class="k">Προφυλακτήρας</span><span class="v" id="r-bump">—</span>
          <span class="k">Γκρεμός</span><span class="v" id="r-cliff">—</span>
          <span class="k">Τροχοί</span><span class="v" id="r-wheel">—</span>
          <span class="k">Κινητήρες</span><span class="v" id="r-motors">—</span>
          <span class="k">Docking</span><span class="v" id="r-dock">—</span>
          <span class="k">Έκβαση</span><span class="v" id="r-dockst">—</span>
        </div>
      </div>
      <div class="card">
        <h3>Ενέργειες</h3>
        <div class="row">
          <button class="btn pri" id="b-dock">🔌 Στη βάση</button>
          <button class="btn" id="b-undock">✕ Άκυρο docking</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ Δεν εμφανίζεται μπαταρία: το σασί τρέφεται από powerbank και τα πεδία
          φόρτισης του OI δίνουν σκουπίδια. Αν το «Σύνδεση» ξεπεράσει τα ~3s, η
          βάση κοιμάται — τότε κάθε πρόβλημα πλοήγησης είναι ψεύτικο.
        </p>
      </div>
    </section>

    <!-- ── Costmap ─────────────────────────────────────────────── -->
    <section class="pane" id="p-cost">
      <!-- The drag grip underneath does the same job, but it is pointer-only
           and needs discovering. Two buttons are obvious and work on a phone. -->
      <div class="row" style="justify-content:center;gap:8px;flex:0 0 auto">
        <button class="btn" id="b-cost-smaller" title="μικρότερο">−</button>
        <span style="font-size:11.5px;color:#71717a;min-width:74px;text-align:center"
              id="cost-size">—</span>
        <button class="btn" id="b-cost-bigger" title="μεγαλύτερο">+</button>
      </div>
      <div id="cost-wrap">
        <canvas id="cost-canvas"></canvas>
        <div class="ovl">COSTMAP · 3×3m γύρω από το ρομπότ</div>
      </div>
      <div class="card">
        <h3>Υπόμνημα</h3>
        <div class="row" id="cost-legend" style="flex-wrap:wrap;gap:14px"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Αυτό είναι ό,τι ΒΛΕΠΕΙ ο planner, όχι ο χάρτης. Το μπλε γύρω από τους τοίχους είναι το inflation — αν δύο μπλε ζώνες ενωθούν σε μια πόρτα, το ρομπότ δεν περνά, ακόμη κι αν το άνοιγμα φαίνεται καθαρό στον χάρτη. Τα ροζ σημεία είναι εμπόδια που έβαλε η ΑΝΙΧΝΕΥΣΗ (άνθρωποι παίρνουν μεγαλύτερο περιθώριο), όχι το lidar.
        </p>
      </div>
    </section>

    <!-- ── NeRF capture ────────────────────────────────────────── -->
    <section class="pane" id="p-nerf">
      <div class="card">
        <h3>Καταγραφή <span class="badge" id="nf-state">—</span></h3>
        <div class="grid2">
          <span class="k">Καρέ</span><span class="v" id="nf-frames">—</span>
          <span class="k">Φάκελος</span><span class="v" id="nf-dir">—</span>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="btn pri" id="b-nerf-go">⏺ Ξεκίνα καταγραφή</button>
          <button class="btn warn" id="b-nerf-stop">■ Σταμάτα</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Καταγράφει εικόνες μαζί με τις ΜΕΤΡΗΜΕΝΕΣ πόζες της κάμερας από το TF — γι' αυτό δεν χρειάζεται COLMAP, που είναι το αργό και εύθραυστο μισό κάθε NeRF. Οδήγησε αργά γύρω από τον χώρο κοιτώντας τον από πολλές γωνίες. Κρατά καρέ μόνο όταν η κάμερα έχει όντως μετακινηθεί (12cm ή 8°), αλλιώς 30 πανομοιότυπες λήψεις τον δευτερόλεπτο δεν διδάσκουν τίποτα.
        </p>
      </div>
      <div class="card">
        <h3>Χάρτης κάλυψης</h3>
        <canvas id="nf-map" width="280" height="280" style="width:100%;max-width:280px;border-radius:8px;background:#18181b"></canvas>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Κάθε πράσινη κουκκίδα είναι μια θέση απ' όπου κρατήθηκε καρέ (πάνοψη, x/y). Η πορτοκαλί είναι η πιο πρόσφατη. Βοηθά να δεις ΕΝ ΚΙΝΗΣΕΙ αν καλύπτεις όλο το δωμάτιο πριν πατήσεις «Σταμάτα». Καθαρίζει αυτόματα όταν ξεκινά νέα καταγραφή.
        </p>
      </div>
      <div class="card">
        <h3>Εκπαίδευση <span class="badge" id="nt-state">—</span></h3>
        <div class="grid2">
          <span class="k">Βήμα</span><span class="v" id="nt-step">—</span>
          <span class="k">Loss / PSNR</span><span class="v" id="nt-loss">—</span>
          <span class="k">Χρόνος / ETA</span><span class="v" id="nt-eta">—</span>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="btn pri" id="b-nerf-train-go">▶ Ξεκίνα εκπαίδευση</button>
          <button class="btn warn" id="b-nerf-train-stop">■ Σταμάτα</button>
        </div>
        <p id="nt-error" style="display:none;font-size:11.5px;color:#f87171;margin-top:10px;
           line-height:1.6;white-space:pre-wrap"></p>
        <button class="btn" id="b-nerf-stop-perception"
                style="display:none;margin-top:8px">Σταμάτα αντίληψη και ξαναδοκίμασε</button>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ ΔΕΝ μπορεί να μοιραστεί το iGPU με το perception. Με το object_detector και το pose_node ενεργά, η εκπαίδευση ΡΙΧΝΕΙ την ουρά του ROCm (memory aperture violation) — δεν είναι έλλειψη μνήμης, μετρήθηκαν 5.9GB ελεύθερα. Το «Σταμάτα αντίληψη» σβήνει ΜΟΝΟ τα 3 nodes που χρησιμοποιούν το iGPU (object_detector, pose_node, open_vocab_detector) — το ρομπότ συνεχίζει να οδηγεί/εντοπίζεται κανονικά, απλά τυφλώνεται μέχρι το επόμενο «robot max». Μετρημένη ταχύτητα: ~310ms/βήμα, άρα ένα δωμάτιο θέλει 45 λεπτά έως 2 ώρες. Αποθηκεύει checkpoint κάθε 200 βήματα — «Σταμάτα» και ξανά «Ξεκίνα» συνεχίζει από εκεί, δεν ξαναρχίζει από την αρχή.
        </p>
      </div>
    </section>

    <!-- ── Pointing gestures ───────────────────────────────────── -->
    <section class="pane" id="p-point">
      <div class="card" style="margin-bottom:9px">
        <h3>Δείξε με το χέρι <span class="badge" id="pt-badge">—</span></h3>
        <div class="row" style="align-items:center;gap:16px">
          <canvas id="pt-ring" width="120" height="120"
                  style="width:120px;height:120px;flex:0 0 auto"></canvas>
          <div class="grid2" style="flex:1;min-width:170px">
            <span class="k">Στόχος X</span><span class="v" id="pt-x">—</span>
            <span class="k">Στόχος Y</span><span class="v" id="pt-y">—</span>
            <span class="k">Χέρι</span><span class="v" id="pt-side">—</span>
            <span class="k">Ευθύτητα</span><span class="v" id="pt-straight">—</span>
          </div>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="btn pri" id="b-pt-go">👉 Πήγαινε εκεί</button>
          <span id="pt-msg" style="font-size:11.5px;color:#71717a"></span>
        </div>
      </div>
      <div class="card">
        <h3>Πώς δουλεύει</h3>
        <p style="font-size:11.5px;color:#71717a;line-height:1.6">
          Τεντώνεις το χέρι και δείχνεις στο ΠΑΤΩΜΑ. Το ρομπότ παίρνει τον ώμο και
          τον καρπό σου από τον σκελετό (pose_node), τα ανεβάζει σε 3D με το βάθος
          της D435, και προεκτείνει τη γραμμή ώσπου να συναντήσει το δάπεδο.
          Ο κύκλος γεμίζει καθώς μαζεύονται καρέ που συμφωνούν· γίνεται πράσινος
          όταν κλειδώσει.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ ΔΕΝ οδηγεί μόνο του. Το να δείξεις κάτι γίνεται πολύ εύκολα κατά λάθος
          — απλώνοντας το χέρι για μια κούπα γράφεις την ίδια γεωμετρία — οπότε
          χρειάζεται ρητή επιβεβαίωση: αυτό το κουμπί ή «πήγαινε εκεί» με τη φωνή.
          Θέλει <code>use_pose:=true</code> και ευθυγραμμισμένο βάθος.
        </p>
      </div>
      <div class="card">
        <h3>Τι κάνει η κάθε χειρονομία <span class="badge" id="gb-badge">—</span></h3>
        <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
          cursor:pointer;user-select:none;padding:9px 11px;border-radius:10px;
          background:#232329;border:1px solid #33333d">
          <input type="checkbox" id="gb-motion">
          <span>Να επιτρέπονται χειρονομίες που ΚΙΝΟΥΝ το ρομπότ</span>
        </label>
        <div id="gb-list" style="margin-top:12px"></div>
        <div id="gb-msg" style="color:#71717a;font-size:11.5px;margin-top:8px"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ Οι χειρονομίες που ΣΤΑΜΑΤΟΥΝ δουλεύουν πάντα, ακόμη κι όταν ο
          διακόπτης είναι κλειστός — το αντίθετο θα ήταν η χειρότερη δυνατή
          συμπεριφορά. Όσες ΞΕΚΙΝΟΥΝ κίνηση θέλουν διπλάσιο κράτημα, γιατί ένα
          λάθος «έλα εδώ» στέλνει μηχάνημα πάνω σε άνθρωπο.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Οι στάσεις σώματος διαβάζονται από απόσταση· τα δάχτυλα θέλουν
          κοντινή απόσταση και <code>use_hand_gestures:=true</code>.
        </p>
      </div>
    </section>

    <!-- ── People ──────────────────────────────────────────────── -->
    <section class="pane" id="p-people">
      <div class="card" style="margin-bottom:9px">
        <h3>Ποιος είναι εδώ <span class="badge" id="pp-badge">—</span></h3>
        <div class="grid2">
          <span class="k">Βλέπω</span><span class="v" id="pp-seeing">—</span>
          <span class="k">Ακούω</span><span class="v" id="pp-hearing">—</span>
          <span class="k">Ύψος</span><span class="v" id="pp-height">—</span>
          <span class="k">Γιατί</span><span class="v" id="pp-reason">—</span>
        </div>
      </div>

      <div class="card" style="margin-bottom:9px">
        <h3>Πρόσθεσε άτομο</h3>
        <div class="row">
          <input id="pp-name" placeholder="όνομα"
            style="flex:1;min-width:120px;background:#232329;border:1px solid #33333d;
            border-radius:9px;color:#e4e4e7;padding:8px 11px;font-size:12.5px"
            autocomplete="off">
          <button class="btn pri" id="b-pp-add">➕ Πρόσθεσε</button>
        </div>
        <div id="pp-msg" style="font-size:11.5px;color:#71717a;margin-top:8px"></div>
      </div>

      <div class="card grow">
        <h3>Γνωστά άτομα <span class="badge" id="pp-count">—</span></h3>
        <div id="pp-list"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:12px;line-height:1.6">
          Τρία σήματα, το καθένα τυφλό αλλού. Το ΠΡΟΣΩΠΟ δουλεύει σιωπηλά και
          από απόσταση, αλλά όχι στο σκοτάδι ή από πίσω. Η ΦΩΝΗ δουλεύει στο
          σκοτάδι και πίσω από γωνίες, αλλά μόνο όσο κάποιος μιλάει. Το ΥΨΟΣ
          υπάρχει πάντα.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ‼️ Το ύψος ΔΕΝ αναγνωρίζει από μόνο του — δύο ενήλικες διαφέρουν
          συχνά λίγα εκατοστά. Χρησιμεύει για να ΑΠΟΚΛΕΙΕΙ: «όποιος κι αν
          είναι, δεν είναι το παιδί». Μετριέται από το πάτωμα του χάρτη, οπότε
          θέλει εντοπισμό — χωρίς αυτόν δεν δείχνει ύψος αντί για λάθος ύψος.
        </p>
      </div>
    </section>

    <!-- ── Open-vocabulary search ──────────────────────────────── -->
    <section class="pane" id="p-vocab">
      <div class="card" style="margin-bottom:9px">
        <h3>Ψάξε ό,τι θέλεις <span class="badge" id="vc-badge">—</span></h3>
        <div class="row">
          <input id="vc-q" placeholder="π.χ. κλειδιά, γυαλιά, φορτιστής"
            style="flex:1;min-width:150px;background:#232329;border:1px solid #2c2c32;
            border-radius:8px;color:#e4e4e7;padding:7px 10px;font-size:12.5px"
            autocomplete="off">
          <button class="btn pri" id="b-vc-go">🔎 Ψάξε</button>
          <button class="btn" id="b-vc-stop" title="Ελευθερώνει το iGPU">■</button>
        </div>
        <div class="row" style="margin-top:8px;flex-wrap:wrap;gap:6px" id="vc-chips"></div>
        <div id="vc-msg" style="font-size:11.5px;color:#71717a;margin-top:9px"></div>
      </div>
      <div class="card">
        <h3>Τι βλέπει <span class="badge" id="vc-count">—</span></h3>
        <div id="vc-hits" style="font-size:12.5px;line-height:1.7">
          <span style="color:#71717a">Δεν ψάχνει τίποτα.</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:12px;line-height:1.6">
          Το κανονικό YOLO ξέρει 80 σταθερές κατηγορίες — κούπες, καρέκλες,
          βιβλία. «Κλειδιά», «φορτιστής», «πορτοφόλι» ΔΕΝ υπάρχουν σε αυτές.
          Το YOLO-World παίρνει τη λίστα ως κείμενο, οπότε ψάχνει ό,τι του πεις.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ Ξεκινά ΜΟΝΟ όταν ζητήσεις κάτι και σβήνει μόνο του μετά από 90
          δευτερόλεπτα. Ένας δεύτερος ανιχνευτής που τρέχει συνέχεια στο ίδιο
          iGPU είναι ακριβώς το πρόβλημα φόρτου που έχει ξαναχτυπήσει εδώ.
          Θέλει <code>use_open_vocab:=true</code>.
        </p>
      </div>
    </section>

    <!-- ── Sound events ────────────────────────────────────────── -->
    <section class="pane" id="p-sound">
      <!-- Static markup, same reasoning as the safety tab's .sfrow rows: the
           i18n extractor has to see every label as ordinary text, not text
           built in JS from a table. -->
      <div class="card" style="margin-bottom:9px">
        <h3>Ρυθμίσεις μικροφώνου <span class="badge" id="mic-badge">—</span></h3>
        <div style="margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid #27272e">
          <div class="sflab" style="margin-bottom:2px">Λέξη-αφύπνιση
            <span class="badge" id="wm-badge">—</span></div>
          <select id="wake-model" class="btn"
            style="margin-top:6px;padding:6px 9px;width:100%"></select>
          <div class="sfhelp" style="margin-top:8px">Οι επιλογές εκτός από «Έι
            ρομπότ» είναι έτοιμα αγγλικά μοντέλα του openWakeWord — δουλεύουν
            αμέσως, χωρίς εκπαίδευση, αλλά είναι ονόματα ΑΛΛΩΝ βοηθών (Alexa,
            Jarvis, ...), όχι δικά του. Για ένα εντελώς νέο όνομα/φράση
            χρειάζεται να εκπαιδευτεί καινούργιο μοντέλο
            (<code>training/wake_word_hey_robot/</code>) — δεν αλλάζει από
            εδώ.</div>
        </div>
        <div class="sfrow" data-key="wake_threshold">
          <div class="sflab">Ευαισθησία λέξης-αφύπνισης «Έι ρομπότ»</div>
          <div class="sfhelp">Πιο ψηλά = πιο δύσκολο να «ξυπνήσει» κατά λάθος·
            πιο χαμηλά = πιο εύκολο να μην το προσέξει.</div>
        </div>
        <div class="sfrow" data-key="highpass_hz">
          <div class="sflab">Φιλτράρισμα χαμηλού θορύβου (ανεμιστήρας/ρεύμα)</div>
          <div class="sfhelp">0 = απενεργοποιημένο. Μεγαλύτερη τιμή κόβει πιο
            χαμηλές συχνότητες πριν την ανίχνευση της λέξης-αφύπνισης —
            λιγότερα ψεύτικα ξυπνήματα από θόρυβο.</div>
        </div>
        <div class="sfrow" data-key="barge_in_enabled">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >Να διακόπτεται λέγοντας «Έι ρομπότ» ενώ μιλάει</span></label></div>
          <div class="sfhelp">‼️ Μετρήθηκε ότι 46 στις 81 «διακοπές» ήταν το
            ΙΔΙΟ το ηχείο του ρομπότ, όχι άνθρωπος. Το ρομπότ το απενεργοποιεί
            ξανά μόνο του μετά από 3 άδειες διακοπές στη σειρά.</div>
        </div>
        <div class="sfrow" data-key="barge_in_threshold">
          <div class="sflab">Πόσο σίγουρο πρέπει να είναι για να δεχτεί τη διακοπή</div>
        </div>
        <div class="sfrow" data-key="energy_thresh">
          <div class="sflab">Ευαισθησία έναρξης εγγραφής ομιλίας</div>
          <div class="sfhelp">‼️ Ξαναμετριέται αυτόματα από τον θόρυβο του
            δωματίου σε κάθε επανεκκίνηση — μια τιμή που αλλάζεις εδώ χάνεται
            στο επόμενο <code>robot max</code>.</div>
        </div>
        <div class="sfrow" data-key="use_hw_vad">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >Χρήση του VAD του υλικού (XVF3800)</span></label></div>
        </div>
        <div class="sfrow" data-key="mic_muted">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >🔇 Σίγαση μικροφώνου (privacy mode)</span></label></div>
          <div class="sfhelp">Κόβει την ανίχνευση λέξης-αφύπνισης ΚΑΙ την
            εγγραφή στην πηγή — το ρομπότ δεν ακούει τίποτα όσο είναι
            ενεργό. Το δαχτυλίδι LED γίνεται κόκκινο.</div>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="btn warn" id="b-mic-reset">↺ Επαναφορά προεπιλογών</button>
          <button class="btn" id="b-mic-power">🔌 Power-cycle μικροφώνου</button>
        </div>
        <div id="mic-msg" style="font-size:11.5px;color:#71717a;margin-top:9px"></div>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Δαχτυλίδι LED <span class="badge" id="led-badge">—</span></h3>
        <div class="sfrow" data-key="led_enabled">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >Ενεργό δαχτυλίδι LED</span></label></div>
        </div>
        <div class="sfrow" data-key="led_brightness">
          <div class="sflab">Φωτεινότητα</div>
        </div>
        <div class="sfrow" data-key="led_color_idle_pointer">
          <div class="sflab">Χρώμα δείκτη κατεύθυνσης (σε αναμονή)</div>
        </div>
        <div class="sfrow" data-key="led_color_idle_base">
          <div class="sflab">Χρώμα φόντου (σε αναμονή)</div>
        </div>
        <div class="sfrow" data-key="led_color_listening">
          <div class="sflab">Χρώμα όσο ακούει την εντολή</div>
        </div>
        <div class="sfrow" data-key="led_color_processing">
          <div class="sflab">Χρώμα όσο επεξεργάζεται (STT)</div>
        </div>
        <div class="sfrow" data-key="led_color_muted">
          <div class="sflab">Χρώμα όσο είναι σε σίγαση</div>
        </div>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Τι ακούω <span class="badge" id="sd-badge">—</span></h3>
        <div class="row" style="align-items:center;gap:16px">
          <canvas id="sd-compass" width="120" height="120"
                  style="width:120px;height:120px;flex:0 0 auto"></canvas>
          <div class="grid2" style="flex:1;min-width:170px">
            <span class="k">Κατεύθυνση</span><span class="v" id="sd-bearing">—</span>
            <span class="k">Γωνία</span><span class="v" id="sd-angle">—</span>
            <span class="k">Ομιλία</span><span class="v" id="sd-speech">—</span>
            <span class="k">Παράθυρα</span><span class="v" id="sd-windows">—</span>
            <span class="k">Ανίχνευση ομιλίας (υλικό)</span><span class="v" id="sd-vad">—</span>
          </div>
        </div>
        <div id="sd-cands" style="font-size:11.5px;color:#a1a1aa;margin-top:10px;
          line-height:1.6"></div>
        <div class="row" style="margin-top:12px;align-items:center">
          <button class="btn pri" id="b-listen">🔊 Άκου το μικρόφωνο</button>
          <button class="btn" id="b-listen-stop">■ Σταμάτα</button>
          <span id="listen-msg" style="font-size:11.5px;color:#71717a"></span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ΖΩΝΤΑΝΟΣ ήχος από το δωμάτιο όπου βρίσκεται το ρομπότ — ακούς ό,τι
          ακούει. Ίδιο κανάλι με το «Έι ρομπότ», καθαρισμένο από τον XVF3800.
          Ξεκινά μόνο όταν το πατήσεις και κόβεται μόλις κλείσεις το tab.
        </p>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Αυτόνομη κίνηση <span class="badge" id="dr-badge">—</span></h3>
        <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
          cursor:pointer;user-select:none;padding:9px 11px;border-radius:10px;
          background:#232329;border:1px solid #33333d">
          <input type="checkbox" id="dr-rotate">
          <span>Να στρίβει προς όποιον μιλάει (μετά το «Έι ρομπότ»)</span>
        </label>
        <div id="dr-msg" style="color:#71717a;font-size:11.5px;margin-top:8px"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ Κλειστό από προεπιλογή. Ανοιχτό, το ρομπότ γυρίζει τη βάση του μόλις
          ακούσει το wake word — ΠΡΙΝ πει κανείς εντολή και χωρίς να ρωτήσει.
          Επειδή το «Έι ρομπότ» πιάνεται και μέσα από κουβέντες που δεν του
          απευθύνονται, αυτό φαινόταν σαν να «φεύγει» μόνο του.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Ο φωτεινός δακτύλιος δείχνει ΠΑΝΤΑ την κατεύθυνση της φωνής, ακόμη κι
          όταν ο διακόπτης είναι κλειστός. Η ρύθμιση θυμάται μετά από
          <code>robot max</code>.
        </p>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Ηχοεντοπισμός <span class="badge" id="ec-badge">—</span></h3>
        <div class="grid2">
          <span class="k">Αντήχηση</span><span class="v" id="ec-rt60">—</span>
          <span class="k">Πρώτη ανάκλαση</span><span class="v" id="ec-dist">—</span>
          <span class="k">Ετυμηγορία</span><span class="v" id="ec-verdict">—</span>
        </div>
        <div class="row" style="margin-top:11px">
          <button class="btn pri" id="b-ec-probe">📡 Μέτρησε τον χώρο</button>
          <span id="ec-msg" style="font-size:11.5px;color:#71717a"></span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Το ρομπότ βγάζει ένα σύντομο τσίρπισμα και ακούει την απάντηση του
          δωματίου. Το XVF3800 έχει ακύρωση ηχούς ακριβώς για να ακούει ΕΝΩ
          μιλάει — γι' αυτό γίνεται. Ένας γυμνός διάδρομος αντηχεί· ένα σαλόνι
          με καναπέδες όχι.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ‼️ Αυτό που εμπιστεύεσαι είναι η ΑΛΛΑΓΗ, όχι οι απόλυτοι αριθμοί. Το
          ηχείο, το μικρόφωνο και το ίδιο το σώμα του ρομπότ μπαίνουν το ίδιο
          σε δύο μετρήσεις από το ίδιο σημείο και αλληλοαναιρούνται. Δεν τρέχει
          ΠΟΤΕ μόνο του: το τσίρπισμα ακούγεται, και ρομπότ που τσιρίζει στις
          3 τα ξημερώματα το ξεσυνδέεις. Θέλει <code>use_echo:=true</code>.
        </p>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Από πού ακούγονται <span class="badge" id="am-badge">—</span></h3>
        <div id="am-last" style="font-size:12.5px;line-height:1.7;
          color:#e4e4e7;margin-bottom:9px"></div>
        <div id="am-list" style="font-size:12.5px;line-height:1.7"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Το μικρόφωνο δίνει ΚΑΤΕΥΘΥΝΣΗ, όχι θέση. Μία γωνία από ένα σημείο
          είναι ακτίνα, όχι σημείο — γι' αυτό η θέση εμφανίζεται μόνο αφού
          ακουστεί ο ίδιος ήχος από ΔΥΟ διαφορετικά σημεία. Το δωμάτιο όμως
          βγαίνει από την πρώτη κιόλας φορά, ακολουθώντας την ακτίνα πάνω
          στον χάρτη.
        </p>
      </div>
      <div class="card">
        <h3>Ιστορικό ήχων</h3>
        <div id="sd-feed" style="font-size:12.5px;line-height:1.75">
          <span style="color:#71717a">Τίποτα ακόμη.</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:12px;line-height:1.6">
          Το YAMNet αναγνωρίζει 521 ήχους· εδώ κρατάμε τους δώδεκα που αφορούν
          ένα σπίτι: κουδούνι, σπασμένο γυαλί, συναγερμός, μωρό που κλαίει,
          νερό που τρέχει, κάτι που έπεσε. Η ΟΜΙΛΙΑ δεν αναγγέλλεται ποτέ —
          είναι ο πιο συχνός ήχος σε ένα σπίτι και τον χειρίζεται ήδη η φωνή.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ ΔΕΝ ανοίγει το μικρόφωνο. Διαβάζει το <code>/mic/audio</code> που
          δημοσιεύει ο κόμβος wake word — δεύτερο ALSA stream στην ίδια συσκευή
          θα τσακωνόταν με το «Έι ρομπότ». Θέλει
          <code>use_sound_events:=true</code>.
        </p>
      </div>
    </section>

    <!-- ── Proactive observations ──────────────────────────────── -->
    <section class="pane" id="p-obs">
      <div class="card" style="margin-bottom:9px">
        <h3>Τι πρόσεξε <span class="badge" id="ob-badge">—</span></h3>
        <div id="ob-feed" style="font-size:12.5px;line-height:1.7;color:#e4e4e7">
          <span style="color:#71717a">Καμία παρατήρηση ακόμη.</span>
        </div>
      </div>
      <div class="card">
        <h3>Πώς μαθαίνει</h3>
        <p style="font-size:11.5px;color:#71717a;line-height:1.6">
          Το ρομπότ χτίζει μόνο του μια βάση αναφοράς: πού «ζει» κανονικά κάθε
          αντικείμενο. Ένα αντικείμενο μετράει ως μόνιμο μόνο αφού το δει στο ίδιο
          σημείο σε ΞΕΧΩΡΙΣΤΕΣ επισκέψεις, με απόσταση μεταξύ τους — αλλιώς μια
          παρατεταμένη ματιά σε ένα δωμάτιο θα γινόταν «κανονικότητα» και όλα μετά
          θα έμοιαζαν μετακινημένα.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Σε καινούριο χάρτη θα σιωπά για μέρες. Αυτό είναι το σωστό: δεν ξέρει
          ακόμη τι σημαίνει «κανονικά». Μιλάει ΜΟΝΟ όταν υπάρχει άνθρωπος μπροστά
          του, το πολύ 4 φορές την ώρα, ποτέ 23:00–08:00, και ποτέ την ίδια
          παρατήρηση δύο φορές σε 2 ώρες.
        </p>
      </div>
    </section>

    <!-- ── Object memory ───────────────────────────────────────── -->
    <section class="pane" id="p-objmem">
      <div class="card" style="margin-bottom:9px">
        <h3>Πού είναι τα πράγματα <span class="badge" id="om-badge">—</span></h3>
        <div id="om-list" style="font-size:12.5px;line-height:1.75">
          <span style="color:#71717a">Τίποτα γνωστό ακόμη.</span>
        </div>
      </div>
      <div class="card">
        <h3>Πώς δουλεύει</h3>
        <p style="font-size:11.5px;color:#71717a;line-height:1.6">
          Το ρομπότ θυμάται πού είδε τελευταία κάθε αντικείμενο, από την κάμερα —
          η ίδια μνήμη που χρησιμοποιεί το «φέρε μου το Χ». Ένα αντικείμενο
          μπαίνει εδώ μόνο αφού επιβεβαιωθεί σε αρκετές ξεχωριστές παρατηρήσεις,
          όχι από μία ματιά.
        </p>
      </div>
    </section>

    <!-- ── Episodic timeline ───────────────────────────────────── -->
    <section class="pane" id="p-time">
      <div class="card" style="margin-bottom:9px">
        <h3>Ρώτα τη μνήμη <span class="badge" id="tl-count">—</span></h3>
        <div class="row">
          <input id="tl-q" placeholder="π.χ. τι έγινε σήμερα το πρωί;"
            style="flex:1;min-width:150px;background:#232329;border:1px solid #2c2c32;
            border-radius:8px;color:#e4e4e7;padding:7px 10px;font-size:12.5px"
            autocomplete="off">
          <button class="btn pri" id="b-tl-ask">Ρώτα</button>
        </div>
        <div class="row" style="margin-top:8px;flex-wrap:wrap;gap:6px" id="tl-chips"></div>
        <div id="tl-answer" style="font-size:12.5px;color:#a1a1aa;margin-top:10px;
          line-height:1.6"></div>
      </div>
      <div class="card">
        <h3>Χρονολόγιο</h3>
        <div id="tl-feed" style="font-size:12.5px;line-height:1.75">
          <span style="color:#71717a">Άδειο.</span>
        </div>
      </div>
    </section>

    <!-- ── IMU (BNO085) ────────────────────────────────────────── -->
    <section class="pane" id="p-imu">
      <div class="card">
        <h3>Προσανατολισμός <span class="badge" id="i-health">—</span></h3>
        <div class="row" style="align-items:center;gap:18px;flex-wrap:wrap">
          <canvas id="imu-rose" width="200" height="200"
                  style="width:200px;height:200px;flex:0 0 auto"></canvas>
          <div class="grid2" style="flex:1;min-width:190px">
            <span class="k">Yaw (στροφή)</span><span class="v" id="i-yaw">—</span>
            <span class="k">Pitch (μύτη)</span><span class="v" id="i-pitch">—</span>
            <span class="k">Roll (κλίση)</span><span class="v" id="i-roll">—</span>
            <span class="k">Ρυθμός στροφής</span><span class="v" id="i-gz">—</span>
            <span class="k">Συχνότητα</span><span class="v" id="i-hz">—</span>
          </div>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:12px;line-height:1.6">
          ‼️ ΣΧΕΤΙΚΗ πυξίδα, όχι Βορράς. Το firmware στέλνει GAME_ROTATION_VECTOR — σύντηξη γυροσκοπίου και επιταχυνσιομέτρου χωρίς το μαγνητόμετρο, επίτηδες: μέσα στο σπίτι οι κινητήρες DC της Roomba και τα μέταλλα διέλυαν την απόλυτη γωνία, ο EKF γύριζε και το AMCL δεν κρατούσε σύγκλιση. Το 0° είναι τυχαία κατεύθυνση σε κάθε boot. Για πλοήγηση δεν χρειάζεται αληθινός Βορράς — μόνο σταθερή σχετική γωνία, και το AMCL διορθώνει τη μικρή απόκλιση με scan matching.
        </p>
      </div>
      <div class="card">
        <h3>Πυξίδα <span class="badge" id="cp-badge">—</span></h3>
        <div class="row" style="align-items:center;gap:18px;flex-wrap:wrap">
          <canvas id="cp-rose" width="200" height="200"
                  style="width:200px;height:200px;flex:0 0 auto"></canvas>
          <div class="grid2" style="flex:1;min-width:190px">
            <span class="k">Κατεύθυνση</span><span class="v" id="cp-card">—</span>
            <span class="k">Μοίρες</span><span class="v" id="cp-deg">—</span>
            <span class="k">Βορράς στον χάρτη</span><span class="v" id="cp-off">—</span>
          </div>
        </div>
        <div class="row" style="margin-top:12px;flex-wrap:wrap">
          <button class="btn pri" id="b-cp-north">🧭 Κοιτάω Βορρά</button>
          <select id="cp-dir" class="btn" style="padding:6px 9px">
            <option value="0">Βορρά</option>
            <option value="90">Ανατολή</option>
            <option value="180">Νότο</option>
            <option value="270">Δύση</option>
          </select>
          <button class="btn" id="b-cp-clear">✕ Καθάρισε</button>
        </div>
        <div id="cp-msg" style="font-size:11.5px;color:#71717a;margin-top:8px"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Η πυξίδα ΔΕΝ βγαίνει από μαγνητόμετρο — βγαίνει από τον ΧΑΡΤΗ. Ο χάρτης
          δεν γυρίζει ποτέ και το AMCL διορθώνει τη γωνία πάνω σε αληθινούς
          τοίχους, οπότε μέσα στο σπίτι είναι πολύ σταθερότερο από κάθε
          μαγνητόμετρο δίπλα σε μοτέρ σκούπας.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Λείπει μόνο ΕΝΑ νούμερο: προς τα πού είναι ο Βορράς μέσα στον χάρτη.
          Γύρνα το ρομπότ να κοιτάει Βορρά και πάτα το κουμπί — μία φορά για κάθε
          χάρτη. Αν ξέρεις ότι κοιτάει άλλη κατεύθυνση, διάλεξέ την από τη λίστα.
          ‼️ Θέλει εντοπισμό: χωρίς AMCL δεν υπάρχει γωνία να μετρηθεί.
        </p>
      </div>
      <div class="card">
        <h3>Ακατέργαστες τιμές BNO085</h3>
        <div class="grid2">
          <span class="k">Γυροσκόπιο X/Y/Z</span><span class="v" id="i-gyro">—</span>
          <span class="k">Επιτάχυνση X/Y/Z</span><span class="v" id="i-acc">—</span>
          <span class="k">Quaternion w/x/y/z</span><span class="v" id="i-quat">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Το BNO085 μπορεί να δώσει και μαγνητόμετρο, γραμμική επιτάχυνση, βαρύτητα, βήματα και ταξινόμηση κίνησης — κανένα δεν είναι ενεργό. Το firmware ενεργοποιεί μόνο δύο reports (γωνία και γυροσκόπιο), γιατί όταν ζητούνται πολλά μαζί το I2C ρίχνει σιωπηλά μερικά — έτσι είχε «πεθάνει» το γυροσκόπιο. Η επιτάχυνση στέλνεται ως σταθερό 0.
        </p>
      </div>
      <div class="card">
        <h3>Θέση στο ρομπότ</h3>
        <p style="font-size:11.5px;color:#71717a;line-height:1.6">
          BNO085 σε ESP32 (CH340) στο /dev/imu, τοποθετημένο ανάποδα· nRESET στο GPIO18 ώστε να επανέρχεται μόνο του όταν κολλήσει το πρωτόκολλο. Τροφοδοτεί τον EKF μαζί με το odometry.
        </p>
      </div>
    </section>

    <!-- ── Sensor fusion ───────────────────────────────────────── -->
    <!-- One tab, not two: the EKF panel and the LiDAR/camera comparison are
         the same question asked of different sensors ("do my sensors agree?"),
         and the tab bar was already at 23 entries — a 25th would have pushed
         the phone layout to a fourth row of chips. The camera comparison is
         behind its own switch because it turns the D435 pointcloud filter on. -->
    <section class="pane" id="p-fuse">
      <div class="card">
        <h3>Ποιος λέει τι <span class="badge" id="fz-badge">—</span></h3>
        <canvas id="fz-chart" width="720" height="200"
                style="width:100%;height:200px;display:block;background:#0c0c0e;
                       border:1px solid #2c2c32;border-radius:10px"></canvas>
        <div class="row" style="gap:14px;flex-wrap:wrap;margin-top:9px;
                                font-size:11.5px;color:#a1a1aa">
          <span><b style="color:#fbbf24">╌</b> Τροχοί</span>
          <span><b style="color:#38bdf8">┈</b> IMU</span>
          <span><b style="color:#4ade80">━</b> EKF (αποτέλεσμα)</span>
          <span style="color:#71717a">60 δευτερόλεπτα</span>
        </div>
        <div class="grid2" style="margin-top:12px">
          <span class="k">Τροχοί − EKF</span><span class="v" id="fz-dw">—</span>
          <span class="k">IMU − EKF</span><span class="v" id="fz-di">—</span>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="btn pri" id="b-fz-reset">⟲ Μηδένισε τη σύγκριση</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Οι τρεις γωνίες ΔΕΝ έχουν κοινό μηδέν — το BNO085 ξεκινά από όπου κοιτούσε στο boot, οι τροχοί από όπου άναψε η βάση. Το κουμπί τις ευθυγραμμίζει εδώ και τώρα, οπότε ό,τι ανοίγει μετά είναι πραγματική απόκλιση. Οδήγησε ένα γύρο και γύρνα στο ίδιο σημείο: η γραμμή που δεν επιστρέφει στο μηδέν είναι ο αισθητήρας που λέει ψέματα. Οι τροχοί ανοίγουν πάντα σε χαλί και σε στροφές — γι' αυτό ο EKF παίρνει γωνία μόνο από το IMU (config/ekf.yaml).
        </p>
      </div>

      <div class="card">
        <h3>Πόσο διορθώνει το AMCL <span class="badge" id="fz-corr-badge">—</span></h3>
        <div class="grid2">
          <span class="k">Μετατόπιση από το μηδένισμα</span><span class="v" id="fz-corr">—</span>
          <span class="k">Γωνία από το μηδένισμα</span><span class="v" id="fz-corryaw">—</span>
          <span class="k">Μέγιστο</span><span class="v" id="fz-corrmax">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Πόσο χρειάστηκε να ΞΑΝΑΒΑΛΕΙ το AMCL το ρομπότ πάνω στους τοίχους από τη στιγμή που άνοιξες την καρτέλα — δηλαδή το λάθος που μάζεψε η σύντηξη μόνη της, όσο κοιτούσες. Στον χάρτη δεν φαίνεται ποτέ: εκεί το ρομπότ δείχνει πάντα σωστά τοποθετημένο, επειδή ακριβώς αυτή η διόρθωση το κρατά εκεί. Λίγα εκατοστά ανά διαδρομή είναι φυσιολογικά· δεκάδες σημαίνουν ότι η οδομετρία γλιστράει και το AMCL μπαλώνει.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ‼️ ΔΙΑΦΟΡΑ, όχι απόλυτη τιμή. Το ωμό <code>map→odom</code> περιέχει και το πού έτυχε να είναι η αρχή του <code>odom</code> στο boot: μετρήθηκε 366 cm και 123° σε ρομπότ που εντόπιζε τέλεια — απλώς είχε οδηγήσει από τότε που άναψε. Το κουμπί «Μηδένισε» ξαναπιάνει το σημείο αναφοράς.
        </p>
      </div>

      <div class="card">
        <h3>Ταχύτητες</h3>
        <div class="grid2">
          <span class="k">Μπροστά — τροχοί</span><span class="v" id="fz-vxw">—</span>
          <span class="k">Μπροστά — EKF</span><span class="v" id="fz-vxe">—</span>
          <span class="k">Στροφή — τροχοί</span><span class="v" id="fz-wzw">—</span>
          <span class="k">Στροφή — γυροσκόπιο</span><span class="v" id="fz-wzi">—</span>
          <span class="k">Στροφή — EKF</span><span class="v" id="fz-wze">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Οι τροχοί λένε πόσο ΖΗΤΗΘΗΚΕ να γυρίσει, το γυροσκόπιο πόσο γύρισε ΠΡΑΓΜΑΤΙΚΑ. Όταν οι δύο αριθμοί διαφέρουν σταθερά, η βάση γλιστράει ή έχει κολλήσει σε κάτι. Κάτω από ~0.31 rad/s η 879 δεν στρίβει καθόλου: θα δεις εντολή στροφής στους τροχούς και μηδέν στο γυροσκόπιο, και αυτό είναι το γνωστό κατώφλι, όχι βλάβη.
        </p>
      </div>

      <div class="card">
        <h3>Υγεία πηγών</h3>
        <div class="grid2">
          <span class="k">Τροχοί <code>/odom</code></span><span class="v" id="fz-h-wheel">—</span>
          <span class="k">IMU <code>/imu/data</code></span><span class="v" id="fz-h-imu">—</span>
          <span class="k">EKF <code>/odometry/filtered</code></span><span class="v" id="fz-h-ekf">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ Νεκρή πηγή είναι ΣΙΩΠΗΛΗ, όχι λάθος: ο EKF συνεχίζει να δημοσιεύει την τελευταία καλή γωνία και όλα δείχνουν υγιή. Μόνο η συχνότητα και η ηλικία δείγματος το δείχνουν — γι' αυτό μετριούνται εδώ χωριστά από τις τιμές.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Φυσιολογικές τιμές, μετρημένες σε φρέσκο `robot max`: τροχοί ~20 Hz, IMU ~110 Hz, EKF ~30 Hz. ‼️ Το IMU μετρήθηκε κάποτε στα 6 Hz και πέρασε για φυσιολογικό — ήταν υποβαθμισμένη ροή που κανείς δεν είχε προσέξει. Αν το δεις κάτω από 40, δεν είναι «αργό», είναι χαλασμένο.
        </p>
      </div>

      <div class="card">
        <h3>Αβεβαιότητα</h3>
        <div class="grid2">
          <span class="k">EKF γωνία ±</span><span class="v" id="fz-cov-ekf">—</span>
          <span class="k">AMCL θέση ± / γωνία ±</span><span class="v" id="fz-cov-amcl">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Πόσο σίγουρος δηλώνει ο καθένας ότι είναι, σε εκατοστά και μοίρες (τυπική απόκλιση, όχι το ωμό covariance). Το AMCL φουσκώνει όταν χάνει τον εντοπισμό και ξαναμαζεύει μόλις κλειδώσει σε τοίχους — μια τιμή που μεγαλώνει και δεν ξαναμαζεύει είναι το «οι κόκκινες γραμμές έφυγαν» πριν το δεις στον χάρτη.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ‼️ Η ΘΕΣΗ του EKF δεν εμφανίζεται επίτηδες. Ο EKF δουλεύει στο <code>odom</code>, όπου καμία απόλυτη μέτρηση δεν τον διορθώνει, οπότε η αβεβαιότητα θέσης του μεγαλώνει για πάντα — μετρήθηκε στα ±2607 km σε ρομπότ που εντόπιζε μια χαρά. Δεν είναι βλάβη, είναι ο ορισμός του frame. Η γωνία του παραμένει φραγμένη (το IMU τη διορθώνει), και για τη θέση ο μόνος αριθμός με νόημα είναι του AMCL.
        </p>
      </div>

      <div class="card">
        <h3>LiDAR εναντίον κάμερας <span class="badge" id="fp-badge">—</span></h3>
        <label style="display:flex;align-items:center;gap:9px;font-size:12.5px;
          cursor:pointer;user-select:none;padding:9px 11px;border-radius:10px;
          background:#232329;border:1px solid #33333d">
          <input type="checkbox" id="fp-on">
          <span>Σύγκρινε με το βάθος της D435 (ανάβει το νέφος σημείων)</span>
        </label>
        <canvas id="fp-canvas" width="440" height="440"
                style="display:block;margin:12px auto 0;width:min(100%,420px);
                       aspect-ratio:1;height:auto;background:#0c0c0e;
                       border:1px solid #2c2c32;border-radius:12px"></canvas>
        <div class="grid2" style="margin-top:12px">
          <span class="k">Συμφωνία</span><span class="v" id="fp-agree">—</span>
          <span class="k">Βλέπει μόνο η κάμερα</span><span class="v" id="fp-camonly">—</span>
          <span class="k">Πλησιέστερο κρυφό εμπόδιο</span><span class="v" id="fp-near">—</span>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Κάτοψη γύρω από το ρομπότ, 4 μέτρα ακτίνα, μύτη προς τα πάνω. <b style="color:#e4e4e7">Λευκό</b> = το lidar, <b style="color:#38bdf8">γαλάζιο</b> = η κάμερα, <b style="color:#f87171">κόκκινο</b> = εκεί που η κάμερα βλέπει εμπόδιο ΠΙΟ ΚΟΝΤΑ από το lidar.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Τα κόκκινα είναι ο λόγος που υπάρχει αυτή η κάρτα: το C1 κόβει μία οριζόντια φέτα στα 60.6 cm, οπότε ένα τραπέζι, ένα σκαλί, ένα σκυμμένο κεφάλι ή μια γάτα ζουν ακριβώς στο κενό του. Η κάμερα κοιτάει από τα 53.6 cm και τα πιάνει, αλλά μόνο μπροστά — τα ~87° του κώνου της. Έξω από αυτόν υπάρχει μόνο λευκό, και αυτό είναι σωστό, όχι διαφωνία.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Πολύ κοντά ο κώνος στενεύει μόνος του: το βάθος βγαίνει από δύο φακούς και σε απόσταση μισού μέτρου τα άκρα του καρέ δεν τα βλέπουν και οι δύο, οπότε εκεί δεν υπάρχει μέτρηση. Μετρήθηκε 36° μπροστά σε εμπόδιο στα 40 cm. Δεν είναι βλάβη — κάνε ένα βήμα πίσω και ο κώνος ανοίγει.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Το πάτωμα κόβεται επίτηδες κάτω από <span id="fp-zmin">6</span> cm και το ταβάνι πάνω από <span id="fp-zmax">140</span> cm: χωρίς αυτό η κάμερα «βλέπει» το χαλί μισό μέτρο μπροστά και τα πάντα γίνονται κόκκινα.
        </p>
      </div>
    </section>

    <!-- ── Voice / LLM ─────────────────────────────────────────── -->
    <section class="pane" id="p-llm">
      <div class="card" style="margin-bottom:9px">
        <h3>Ποιος μιλάει <span class="badge" id="sp-badge">—</span></h3>
        <div id="sp-detail" style="font-size:11.5px;color:#71717a;line-height:1.55">
          Θέλει use_diarization (ποιος), DoA (από πού), use_face_detection (ποιον βλέπω). Ό,τι λείπει παραλείπεται.
        </div>
      </div>
      <div class="card" style="margin-bottom:9px">
        <h3>Ποιος απαντά <span class="badge" id="be-badge">—</span></h3>
        <div class="row">
          <button class="btn" id="be-gemini">🌩️ Gemini (cloud)</button>
          <button class="btn" id="be-lemonade">🧠 Qwen3.5 (NPU)</button>
        </div>
        <div id="be-msg" style="font-size:11.5px;color:#71717a;margin-top:8px;
          line-height:1.55"></div>
        <!-- Today's cloud allowance. Hidden while the local model answers —
             Qwen has no quota, and a leftover counter would be a lie. -->
        <div id="q-box" style="display:none;margin-top:10px">
          <div style="display:flex;justify-content:space-between;
            align-items:baseline;font-size:12px;margin-bottom:5px">
            <span style="color:#a1a1aa">Ερωτήσεις σήμερα</span>
            <span id="q-left" style="font-weight:600">—</span>
          </div>
          <div style="height:6px;border-radius:3px;background:#27272a;
            overflow:hidden">
            <div id="q-bar" style="height:100%;width:0;border-radius:3px;
              background:#34d399;transition:width .4s"></div>
          </div>
          <div id="q-note" style="font-size:11px;color:#52525b;margin-top:6px;
            line-height:1.5"></div>
        </div>
        <div style="font-size:11px;color:#52525b;margin-top:6px;line-height:1.5">
          Το Gemini απαντά σε ~0.5s και δεν πιάνει μνήμη. Το Qwen τρέχει τοπικά
          στο NPU — δεν χρειάζεται ίντερνετ, αλλά αργεί ~6s και κρατά 4.7 GB RAM.
          Η αλλαγή σβήνει τις τελευταίες ατάκες της κουβέντας.
        </div>
      </div>
      <div id="chat"></div>
      <div id="chat-in">
        <input id="chat-text" placeholder="Γράψε στο ρομπότ…" autocomplete="off">
        <button class="btn pri" id="b-send">Στείλε</button>
        <button class="btn" id="b-say" title="Να το πει δυνατά χωρίς να το σκεφτεί">🔊</button>
      </div>
    </section>

    <!-- ── System ──────────────────────────────────────────────── -->
    <section class="pane" id="p-sys">
      <div class="card" style="margin-bottom:9px">
        <h3>Αυτοδιάγνωση <span class="badge" id="dg-badge">—</span></h3>
        <div id="dg-list" style="font-size:12.5px;line-height:1.65"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Ελέγχει τους ΓΝΩΣΤΟΥΣ τρόπους που χαλάει αυτό το ρομπότ — αυτούς που
          έχουν ήδη κοστίσει χρόνο. Σχεδόν όλοι είναι ΣΙΩΠΗΛΟΙ: ο κόμβος ζει,
          το topic υπάρχει, και μόνο η συμπεριφορά είναι λάθος. Κενή λίστα
          σημαίνει «τίποτα γνωστό», όχι «όλα καλά».
        </p>
      </div>
      <div class="card">
        <h3>Υπολογιστής</h3>
        <div id="s-bars"></div>
      </div>
      <div class="card">
        <h3>Ομαλό ρεύμα <span class="badge" id="pw-badge">—</span></h3>
        <div class="row" style="flex-wrap:wrap;gap:8px">
          <button class="btn" data-pw="off">🔌 Πρίζα (κανονικό)</button>
          <button class="btn" data-pw="eco">🍃 Ήπιο</button>
          <button class="btn" data-pw="flat">📏 Πολύ σταθερό</button>
        </div>
        <div id="pw-msg" style="font-size:11.5px;color:#71717a;margin-top:10px"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Για όταν το mini PC τρέχει από την powerstation (Anker SOLIX C300 DC). Αυτή δεν έχει inverter — τροφοδοτεί το PC μέσω USB-C Power Delivery. Μια πηγή PD δεν ενοχλείται από το πόσο ρεύμα τραβάς, αλλά από το πόσο απότομα αλλάζει: ένα σκαλοπάτι φορτίου ρίχνει την προστασία της θύρας ή προκαλεί επαναδιαπραγμάτευση PD, και η επαναδιαπραγμάτευση κόβει τη γραμμή αρκετά ώστε το PC να σβήσει ακαριαία.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Τα προφίλ πειράζουν τέσσερα πράγματα, όλα για να μειώσουν το άλμα: στενεύουν τη ζώνη συχνότητας της CPU, κλείνουν το boost, βάζουν τον επεξεργαστή σε λειτουργία εξοικονόμησης, και — μόνο στο «πολύ σταθερό» — κόβουν το ψηλότερο σκαλί της iGPU. Γι' αυτό το «πολύ σταθερό» κοστίζει σε ταχύτητα γραφικών· το «ήπιο» τα αφήνει ήσυχα.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Μετρημένο εδώ σε πραγματικά Watt (αισθητήρας PPT του επεξεργαστή), με ριπές φορτίου σε όλους τους πυρήνες: <b>κανονικό</b> 21.5 W μέσος όρος και διακύμανση 30.0 W, από 7 ως 37· <b>ήπιο</b> 11.9 και 12.1· <b>πολύ σταθερό</b> 10.5 και 11.1. Η διακύμανση πέφτει κατά 63% και η μέση κατανάλωση στο μισό — εδώ δεν πληρώνεις τίποτα για τη σταθερότητα.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ✅ Επιβιώνει σε restart: η επιλογή αποθηκεύεται και ξαναμπαίνει μόνη της σε κάθε boot. Στην πρίζα γύρνα το σε «κανονικό» — δεν χρειάζεται.
        </p>
      </div>
      <div class="card">
        <h3>Ξεκόλλα αισθητήρα <span class="badge" id="usb-badge">—</span></h3>
        <div class="row" style="flex-wrap:wrap;gap:8px" id="usb-buttons"></div>
        <div id="usb-msg" style="font-size:11.5px;color:#71717a;margin-top:10px"></div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Κόβει το ρεύμα στη θύρα USB της συσκευής για δύο δευτερόλεπτα και το ξαναδίνει — ό,τι ακριβώς κάνει το να την βγάλεις και να την ξαναβάλεις, χωρίς να σηκωθεί κανείς. Για αισθητήρα που κόλλησε και δεν ξεκολλάει με restart του κόμβου.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          ‼️ Ο ΟΔΗΓΟΣ ΔΕΝ ΕΠΑΝΕΚΚΙΝΕΙ ΜΟΝΟΣ ΤΟΥ. Η συσκευή θα ξαναεμφανιστεί στο σύστημα, αλλά ο κόμβος που την είχε ανοιχτή κρατά πεθαμένο handle — θέλει και δικό του restart. Η κάμερα είναι η εξαίρεση: ο camera_watchdog την ξαναπιάνει μόνος του.
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:8px;line-height:1.6">
          Η «σκούπα» κόβει το ρεύμα στη σειριακή της βάσης. Αν το ρομπότ κινείται, οι τροχοί συνεχίζουν με την τελευταία εντολή ώσπου να πιάσει το watchdog των 0.25 δευτερολέπτων — μην το πατήσεις εν κινήσει.
        </p>
      </div>
      <div class="card">
        <h3>Θερμοκρασίες</h3>
        <div id="s-temps" class="tempgrid">—</div>
      </div>
      <div class="card">
        <h3>Δίσκοι</h3>
        <div id="s-disks">—</div>
      </div>
      <div class="card grow">
        <h3>Κόμβοι ROS (<span id="s-nodecount">0</span>)</h3>
        <div id="s-nodes" style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px;
          color:#a1a1aa;line-height:1.75;columns:2;column-gap:20px">—</div>
      </div>
    </section>

    <section class="pane" id="p-cloud">
      <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
        <h3>3D κάμερα (D435)
          <span class="badge" id="cloud-info">—</span>
          <button class="btn" id="b-cloud-reset" style="float:right">Επαναφορά όψης</button>
        </h3>
        <canvas id="cloud-canvas" style="flex:1;width:100%;min-height:0;
          background:#0a0a0b;border-radius:8px;touch-action:none;cursor:grab"></canvas>
        <div style="color:#71717a;font-size:11.5px;margin-top:6px">
          Σύρε για περιστροφή · ροδέλα ή τσίμπημα για ζουμ
        </div>
      </div>
    </section>

    <section class="pane" id="p-set">
      <div class="card">
        <h3>Γλώσσα</h3>
        <div class="row" id="lang-buttons"></div>
        <div style="color:#71717a;font-size:11.5px;margin-top:8px">
          Αλλάζει μόνο αυτή τη σελίδα. Το ρομπότ συνεχίζει να μιλά ελληνικά.
        </div>
      </div>

      <div class="card" style="margin-bottom:9px">
        <h3>Δίκτυο <span class="badge" id="sn-badge">—</span></h3>
        <div class="grid2" style="margin-bottom:9px">
          <span class="k">Σύνδεση</span><span class="v" id="sn-conn">—</span>
          <span class="k">Διευθύνσεις</span><span class="v" id="sn-ips">—</span>
          <span class="k">Tailscale</span><span class="v" id="sn-ts">—</span>
        </div>
        <div id="sn-wifi"></div>
        <div class="row" style="margin-top:9px">
          <input id="sn-pass" type="password" placeholder="κωδικός δικτύου"
            style="flex:1;min-width:130px;background:#232329;border:1px solid #33333d;
            border-radius:9px;color:#e4e4e7;padding:8px 11px;font-size:12.5px"
            autocomplete="off">
          <button class="btn" id="b-sn-scan">🔄 Σάρωση</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:9px;line-height:1.6">
          ‼️ Ο κωδικός ταξιδεύει μέσα από αυτή τη σελίδα. Το token του πίνακα
          είναι αδύναμο και ακούει σε ΟΛΟ το τοπικό δίκτυο — σύνδεσε νέο WiFi
          από εδώ μόνο αν εμπιστεύεσαι όποιον είναι στο ίδιο δίκτυο.
        </p>
      </div>

      <div class="card" style="margin-bottom:9px">
        <h3>Bluetooth <span class="badge" id="sb-badge">—</span></h3>
        <div id="sb-list"></div>
        <div class="row" style="margin-top:9px">
          <button class="btn" id="b-sb-on">⏻ Ενεργό</button>
          <button class="btn" id="b-sb-scan">🔍 Αναζήτηση (10s)</button>
        </div>
      </div>

      <div class="card" style="margin-bottom:9px">
        <h3>Ήχος & σύστημα</h3>
        <div class="row" style="align-items:center">
          <span class="k" style="flex:0 0 60px">Ένταση</span>
          <input type="range" id="sv-vol" min="0" max="150" step="5"
            style="flex:1;min-width:120px">
          <span class="v" id="sv-val" style="flex:0 0 46px;text-align:right">—</span>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="btn warn" id="b-sys-reboot">↻ Επανεκκίνηση</button>
          <button class="btn warn" id="b-sys-off">⏻ Τερματισμός</button>
        </div>
        <div id="sn-msg" style="font-size:11.5px;color:#71717a;margin-top:9px"></div>
      </div>

      <div class="card" style="margin-bottom:9px">
        <h3>Κλειδί πρόσβασης <span class="badge" id="tk-badge">—</span></h3>
        <div id="tk-out" style="font-size:12px;line-height:1.6;
          word-break:break-all;color:#a1a1aa"></div>
        <div class="row" style="margin-top:10px">
          <button class="btn warn" id="b-tk-new">🔑 Νέο κλειδί</button>
        </div>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          Ο πίνακας ακούει σε ΟΛΟ το τοπικό δίκτυο και δίνει κάμερα, μικρόφωνο,
          χειριστήριο και χάρτη του σπιτιού. Το κλειδί είναι το μόνο που τον
          προστατεύει. Το νέο ισχύει μετά από επανεκκίνηση — η τρέχουσα καρτέλα
          συνεχίζει να δουλεύει ώσπου τότε. Άνοιξε τον νέο σύνδεσμο μία φορά σε
          κάθε συσκευή.
        </p>
      </div>

    </section>

    <!-- Every row here writes a LIVE ROS parameter the moment the slider is
         released; nothing waits for a restart. The rows are static markup (not
         built from the spec table in JS) so the translation extractor can see
         every label — see tests/test_dashboard_i18n.py. -->
    <section class="pane" id="p-safe">
      <div class="card">
        <h3>Απόσταση από εμπόδια <span class="badge" id="sf-cam-badge">—</span></h3>
        <div class="sfrow" data-key="stop_distance">
          <div class="sflab">Πόσο κοντά πλησιάζει πριν σταματήσει</div>
          <div class="sfhelp">Το ρομπότ κόβει την ευθεία κίνηση όταν δει κάτι
            πιο κοντά από αυτό. Συνεχίζει να στρίβει και να κάνει όπισθεν, ώστε
            να μπορεί να ξεφύγει.</div>
        </div>
        <div class="sfrow" data-key="center_width">
          <div class="sflab">Πόσο φαρδιά κοιτάει μπροστά του</div>
          <div class="sfhelp">Ποιο κομμάτι της εικόνας μετράει ως «μπροστά». Στο
            1.00 τον σταματά ο,τιδήποτε φαίνεται στην άκρη του κάδρου — σε
            διάδρομο αυτό σημαίνει ότι δεν ξεκινά ποτέ.</div>
        </div>
        <div class="sfrow" data-key="detector_timeout">
          <div class="sflab">Σε πόση σιωπή της κάμερας φρενάρει</div>
          <div class="sfhelp">Αν η ανίχνευση σταματήσει (κόλλησε η κάμερα,
            τερμάτισε ο detector), η ευθεία κίνηση μπλοκάρεται μετά από τόση
            ώρα αντί να θεωρηθεί ο δρόμος καθαρός.</div>
        </div>
      </div>

      <div class="card">
        <h3>Απόσταση από τοίχους <span class="badge" id="sf-nav-badge">—</span></h3>
        <div class="sfrow" data-key="inflation_radius">
          <div class="sflab">Πόσο μακριά από τοίχους σχεδιάζει τη διαδρομή</div>
          <div class="sfhelp">Ζώνη γύρω από κάθε εμπόδιο που ο planner αποφεύγει.
            Μεγαλύτερη = περνά πιο κεντραρισμένο.</div>
          <div class="sfwarn">‼️ Οι πόρτες εδώ είναι ~0.78 m. Πάνω από 0.30 m οι
            ζώνες των δύο παραστάδων ενώνονται και το ρομπότ αρνείται να περάσει
            από άνοιγμα που στον χάρτη φαίνεται καθαρό.</div>
        </div>
        <div class="sfrow" data-key="cost_scaling">
          <div class="sflab">Πόσο απότομα χαλαρώνει αυτή η ζώνη</div>
          <div class="sfhelp">Μεγαλύτερο = η αποφυγή σβήνει πιο γρήγορα με την
            απόσταση, άρα δέχεται να περάσει πιο κοντά στον τοίχο.</div>
        </div>
      </div>

      <div class="card">
        <h3>Αισθητήρες επαφής <span class="badge" id="sf-base-badge">—</span></h3>
        <p class="sfhelp" style="margin:0 0 10px">
          ‼️ Η βάση τρέχει σε FULL mode: το ίδιο το Roomba ΔΕΝ σταματά μόνο του
          σε προφυλακτήρα ή σε σκαλί. Αυτοί οι τρεις διακόπτες είναι το μόνο που
          το κάνει. Κλείσ' τους μόνο για χαλασμένο αισθητήρα.
        </p>
        <!-- The checkbox is injected into [data-box]; the label text stays
             here as ordinary markup so i18nCollect() sees a text node it owns.
             Moving it in JS produced a node the translator had not collected,
             and the row stayed Greek in every language. -->
        <div class="sfrow" data-key="bump_stop">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >Στοπ στον προφυλακτήρα</span></label></div>
        </div>
        <div class="sfrow" data-key="cliff_stop">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >Στοπ στον γκρεμό (σκαλί)</span></label></div>
        </div>
        <div class="sfrow" data-key="wheel_drop_stop">
          <div class="sflab"><label class="sftog"><span data-box></span><span
            >Στοπ όταν πέφτει τροχός στο κενό</span></label></div>
        </div>
        <div class="sfrow" data-key="bump_block_s">
          <div class="sflab">Πόσο μένει μπλοκαρισμένο μετά από χτύπημα</div>
          <div class="sfhelp">Κρατά την ευθεία κλειστή τόση ώρα αφού ελευθερωθεί
            ο προφυλακτήρας, ώστε το ρομπότ να μην ξαναμπεί στο ίδιο έπιπλο.</div>
        </div>
        <div class="sfrow" data-key="cliff_block_s">
          <div class="sflab">Πόσο μένει μπλοκαρισμένο μετά από γκρεμό</div>
        </div>
      </div>

      <div class="card">
        <h3>Σκληρά όρια <span class="badge">σταθερά</span></h3>
        <div id="sf-info" class="grid2"></div>
        <p class="sfhelp" style="margin-top:10px">
          ΔΕΝ αλλάζει από εδώ. Ο collision_monitor το διαβάζει μία φορά στην
          εκκίνηση, οπότε ένας διακόπτης εδώ θα έδειχνε νούμερο που το ρομπότ
          δεν χρησιμοποιεί. Αλλάζει στο config/nav2_params.yaml και θέλει
          πλήρη επανεκκίνηση.
        </p>
        <div class="row" style="margin-top:12px">
          <button class="btn warn" id="b-sf-reset">↺ Επαναφορά προεπιλογών</button>
        </div>
        <div id="sf-msg" class="sfhelp" style="margin-top:9px"></div>
      </div>

      <div class="card">
        <h3>🛑 Απόσταση πλήρους στάσης <span class="badge" id="sk-badge">—</span></h3>
        <p class="sfhelp">
          Το τελευταίο όριο: μόλις το lidar δει κάτι μέσα σε αυτή την
          απόσταση γύρω από το σώμα, ο collision_monitor μηδενίζει αμέσως
          την ταχύτητα — ανεξάρτητα από το τι σχεδιάζει ο planner. Δεν είναι
          η «Απόσταση από τοίχους» πιο πάνω: εκείνο επηρεάζει πού περνά η
          διαδρομή, αυτό είναι το φρένο ανάγκης. Μικρότερη απόσταση αφήνει
          το ρομπότ να πλησιάσει περισσότερο πριν σταματήσει απότομα.
        </p>
        <div class="row" style="margin-top:10px;align-items:center">
          <select id="sk-mm" class="btn" style="padding:6px 9px"></select>
          <button class="btn warn" id="b-sk-apply">Εφαρμογή (επανεκκίνηση ~90s)</button>
        </div>
        <div id="sk-msg" class="sfhelp" style="margin-top:9px"></div>
      </div>
    </section>

    <section class="pane" id="p-log">
      <div class="card" style="flex:1;display:flex;flex-direction:column;min-height:0">
        <h3>Προειδοποιήσεις &amp; σφάλματα (/rosout)
          <span class="badge" id="log-count">0</span>
          <button class="btn" id="b-log-clear" style="float:right">Καθάρισε</button>
        </h3>
        <div id="log-list" style="flex:1;overflow:auto;font-family:ui-monospace,Menlo,monospace;
          font-size:11.5px;line-height:1.6"></div>
      </div>
    </section>

  </div>
</div>
<script>
// ── constants ──────────────────────────────────────────────────────────────
const ROOMS      = __ROOMS__;
const TOKEN_QS   = __TOKEN_QS__;    // '' when auth is disabled
const ARM_LIMITS = __ARM_LIMITS__;
const ARM_JOINTS = __ARM_JOINTS__;
const HAS_NOVNC  = __HAS_NOVNC__;
const USB_DEVICES = __USB_DEVICES__;
// ‼️ ANG was 0.10, which is BELOW the 879's ~0.31 rad/s rotation floor — the
// wheels physically do not turn under it, so ◄/► sent a twist that the base
// swallowed in silence. 0.60 rad/s (~34 deg/s) is a deliberate half of the
// PS5's 1.20: a phone has no analogue stick, so the web D-pad is for nudging
// into position, not for crossing the room. See teleop_twist_joy_ps5.yaml.
//
// Now adjustable from the map tab, so these are the DEFAULTS, not the values.
// ANG_MIN is the important one: the slider is floored above the rotation floor
// so it cannot be dragged into the dead band where the base silently ignores
// every turn — that reads as "the robot broke", and has cost real debugging
// time before. LIN_MAX stays modest for the same reason the default is: this
// is a nudge control on a phone with no dead-man switch.
const LIN_DEF = 0.10, ANG_DEF = 0.60;
const LIN_MIN = 0.05, LIN_MAX = 0.30;
const ANG_MIN = 0.35, ANG_MAX = 1.20;
let LIN = LIN_DEF, ANG = ANG_DEF;
try {
  const s = JSON.parse(localStorage.getItem('hr_speed') || '{}');
  if (s.lin) LIN = Math.min(LIN_MAX, Math.max(LIN_MIN, +s.lin));
  if (s.ang) ANG = Math.min(ANG_MAX, Math.max(ANG_MIN, +s.ang));
} catch(e) {}
// ‼️ Must match bringup.launch.py's tf_base_laser: x=0, yaw=pi. This said 0.0
// and drew every scan point mirrored through the robot, so the dots never
// landed on the walls and the dashboard looked like a localization failure
// when localization was fine.
const LASER_X = 0.00, LASER_YAW_OFFSET = Math.PI;

// ── state ──────────────────────────────────────────────────────────────────
let ws=null, mapInfo=null, mapImg=null, pose=null, scan=null, goal=null, plan=null;
let roomsData={};
let kzData={};   // keepout zones for the active map, see keepout_files.py
let nfPoints=[], nfLastFrames=0;   // NeRF capture coverage — kept frame x,y as they arrive
let robotTrail=[];   // where the robot has driven this session — (x,y) in map metres
let driveTimer=null, vx=0, wz=0, estop=false, armPos={};
const $ = id => document.getElementById(id);

// Escape before interpolating anything into innerHTML. The observation feed and
// the timeline render TRANSCRIBED SPEECH and object labels — text the robot
// heard, not text we wrote — so a stray '<' would otherwise be parsed as markup.
// Everything older in this file builds untrusted strings with textContent, which
// is safe by construction; these two panes need markup per row, so they escape.
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// ── tabs ───────────────────────────────────────────────────────────────────
const TABS = [
  ['map',    '🗺️', 'Χάρτης'],
  ['cam',    '📷', 'Κάμερα'],
  ['cloud',  '🧿', '3D'],
  ['rviz',   '🧊', 'RViz'],
  ['moveit', '🎯', 'MoveIt'],
  ['arm',    '🦾', 'Χέρι'],
  ['base',   '🧹', 'Σκούπα'],
  ['imu',    '🧭', 'IMU'],
  // Left in English on purpose, in both languages: "Σύντηξη" alone reads as
  // nuclear fusion, and this is the name the user asked for.
  ['fuse',   '🔀', 'Sensor fusion'],
  ['rtabmap','🏠', 'Σπίτι 3D'],
  ['cost',   '🧱', 'Costmap'],
  ['nerf',   '✨', 'NeRF'],
  ['point',  '👉', 'Χειρονομίες'],
  ['vocab',  '🔎', 'Ψάξε'],
  ['sound',  '👂', 'Ήχοι'],
  ['obs',    '💡', 'Πρόσεξα'],
  ['objmem', '📦', 'Αντικείμενα'],
  ['time',   '🕐', 'Χρονικό'],
  ['people', '🧑', 'Άτομα'],
  ['llm',    '💬', 'Φωνή'],
  ['gazebo', '🌍', 'Gazebo'],
  ['sys',    '📊', 'Σύστημα'],
  ['safe',   '🛡️', 'Ασφάλεια'],
  ['log',    '📜', 'Log'],
  ['set',    '⚙️', 'Ρύθμιση'],
];
// Rebuilt on every language change; the labels come from the same t() as the
// rest of the page. Preserves which tab is active across the rebuild.
function renderTabs(){
  const tabNav = $('tabs');
  const active = (document.querySelector('.tab.active') || {}).dataset;
  tabNav.innerHTML = '';
  TABS.forEach(([id, icon, label]) => {
    const b = document.createElement('div');
    b.className = 'tab' + (active && active.pane === id ? ' active' : '');
    b.dataset.pane = id;
    b.innerHTML = `<span class="ic">${icon}</span><span>${t(label)}</span>`;
    b.onclick = () => showTab(id);
    tabNav.appendChild(b);
  });
}
function showTab(id){
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.pane === id));
  document.querySelectorAll('.pane').forEach(p =>
    p.classList.toggle('active', p.id === 'p-' + id));
  // The map tab holds two views (2D canvas / 3D walls) toggled by
  // setMapView — re-run whichever is current, the same reason `resize()`
  // below exists: a canvas sized while its pane was display:none reads 0
  // for clientWidth/Height, so returning to the tab has to re-measure it.
  if (id === 'map'){ setMapView(mapView); mapsRefresh(); }
  if (id === 'rtabmap'){ rtabSavedRefresh(); }
  if (VNC_APPS[id]) ensureVnc(id);
  cloudSetActive(id === 'cloud');
  camSetActive(id === 'cam');
  costSetActive(id === 'cost');
  fuseSetActive(id === 'fuse');
  if (id === 'cost'){ buildCostLegend(); costShowSize(); }
  // 170 kB of geometry, fetched the first time the tab is opened rather than
  // on every page load — most visits never look at the arm.
  if (id === 'arm'){ armLoad(); armDraw(); }
}

// ── map tab: 2D / 3D toggle ─────────────────────────────────────────────────
// One pane, two views of the SAME active map. The old "Τοίχοι 3D"
// extruded-walls view was dropped per user request (2026-08-15) — the
// photorealistic scan view covers 3D, this canvas only does 2D now.
let mapView = '2d';
function setMapView(view){
  mapView = view;
  $('map-view-2d').classList.toggle('pri', view === '2d');
  $('map-view-scan').classList.toggle('pri', view === 'scan');
  $('map-wrap').style.display = view === '2d' ? '' : 'none';
  $('map-scan3d-card').style.display = view === 'scan' ? '' : 'none';
  if (view === '2d') resize();
  else if (view === 'scan' && window.hrScan3d) window.hrScan3d.activate();
}
$('map-view-2d').onclick = () => setMapView('2d');
$('map-view-scan').onclick = () => setMapView('scan');
// NB: the initial showTab() call lives at the bottom of the script — calling it
// here would touch VNC_APPS before its `const` is initialised (temporal dead
// zone), which throws and leaves the whole page unwired.

// ── VNC panes (RViz / MoveIt / Gazebo) ─────────────────────────────────────
// Each pane is built lazily: an iframe is only created once the tab is opened
// and its session is confirmed up, so a page load does not open three RFB
// streams the user may never look at.
const VNC_APPS = {
  rviz:   {title:'RViz',   note:'Η ίδια συνεδρία :2 που ανοίγει το <code>robot max</code> — και αυτή που βλέπεις από το RealVNC στο κινητό.'},
  moveit: {title:'MoveIt', note:'Ξεκινά <code>arm_moveit.launch.py</code>: move_group + RViz με το Motion Planning panel. Τράβα τη δαγκάνα, Plan, Execute.'},
  gazebo: {title:'Gazebo', note:'‼️ ΠΡΟΣΟΜΟΙΩΣΗ. Δημοσιεύει δικά της /clock, /scan, /odom — μην την ανοίγεις ενώ οδηγείς το πραγματικό ρομπότ.'},
  rtabmap:{title:'RTAB-Map', note:'Χτίζει τρισδιάστατο χάρτη του σπιτιού από την D435. Οδήγησε αργά και κοίτα τους τοίχους· ο χάρτης μεγαλώνει μόνο όσο κινείσαι. ΔΕΝ πειράζει την πλοήγηση: δεν δημοσιεύει TF και ζει σε δικό του namespace.'},
};
const vncState = {};
Object.keys(VNC_APPS).forEach(k => vncState[k] = {frame:null, busy:false});

function vncPane(app){ return $('p-' + app); }

// /vncview reports its own failures by postMessage — no DOM scraping and no
// polling, so a viewer that dies before it paints still says why. The old
// version read noVNC's fallback banner out of the iframe, which could only
// ever repeat what that banner said: "Script error."
window.addEventListener('message', ev => {
  if (ev.origin !== location.origin) return;
  const m = ev.data;
  if (!m || m.source !== 'vncview' || !VNC_APPS[m.app]) return;
  if (m.kind === 'ok') clearVncError(m.app);
  else if (m.kind === 'error') showVncError(m.app, m.text);
});

function clearVncError(app){
  const pane = vncPane(app), b = pane && pane.querySelector('.vnc-err');
  if (b) b.remove();
}

function showVncError(app, text){
  const host = vncPane(app).querySelector('.vnc-host');
  if (!host) return;
  let b = host.querySelector('.vnc-err');
  if (!b){
    b = document.createElement('div');
    b.className = 'vnc-err';
    host.insertBefore(b, host.firstChild);
  }
  b.textContent = '⚠ noVNC: ' + text;
}

function renderVnc(app, mode, detail){
  const pane = vncPane(app), meta = VNC_APPS[app], st = vncState[app];
  pane.innerHTML = '';             // the old frame, and its error banner, go
  const host = document.createElement('div');
  host.className = 'vnc-host';
  pane.appendChild(host);

  if (mode === 'live'){
    const f = document.createElement('iframe');
    // Our own viewer, not noVNC's stock UI — see VNC_VIEW_HTML for why. The
    // VNC credential is baked into that page server-side instead of riding in
    // this URL, where it ended up in history and in every referrer.
    f.src = '/vncview/' + app + TOKEN_QS;
    host.appendChild(f);
    st.frame = f;
    // RViz specifically: always open big rather than making the user hit
    // "Πλήρης οθόνη" every time — see the .vnc-host.fs comment for why a
    // real RViz pane is otherwise mostly letterboxing on a phone. Harmless
    // to call before the VNC connection lands; it's pure CSS layout and
    // requestFullscreen's rejection (no user gesture) is already caught.
    if (app === 'rviz') enterVncFs(app);
  } else {
    const box = document.createElement('div');
    box.className = 'vnc-msg';
    box.innerHTML =
      `<div style="font-size:34px">${mode === 'starting' ? '⏳' : '🖥️'}</div>` +
      `<div><b>${meta.title}</b><br>${detail || ''}</div>` +
      `<div style="max-width:440px;font-size:12px;color:#71717a">${t(meta.note)}</div>`;
    if (mode !== 'starting'){
      const b = document.createElement('button');
      b.className = 'btn pri';
      b.textContent = t('Εκκίνηση') + ' ' + meta.title;
      b.onclick = () => startVnc(app);
      box.appendChild(b);
    }
    host.appendChild(box);
  }

  // Controls strip under the viewport.
  const card = document.createElement('div');
  card.className = 'card';
  const row = document.createElement('div');
  row.className = 'row';
  const mk = (label, cls, fn) => {
    const b = document.createElement('button');
    b.className = 'btn ' + (cls||''); b.textContent = label; b.onclick = fn;
    row.appendChild(b);
  };
  mk(t('↻ Επανασύνδεση'), '', () => refreshVnc(app));
  // Only offered once something is actually painting — fullscreening the
  // "press start" placeholder would just be a black screen with an X on it.
  if (mode === 'live') mk(t('⛶ Πλήρης οθόνη'), '', () => enterVncFs(app));
  if (app !== 'rviz') mk(t('■ Τερματισμός'), 'warn', () => stopVnc(app));
  const s = document.createElement('span');
  s.className = 'pill ' + (mode === 'live' ? 'ok' : '');
  s.textContent = mode === 'live' ? t('ενεργό') : mode === 'starting' ? t('ξεκινά…') : t('σταματημένο');
  row.appendChild(s);
  card.appendChild(row);
  pane.appendChild(card);

  if (app === 'rtabmap'){
    pane.appendChild(rtabControls());
    rtabSavedRefresh();   // now that #rt-saved-list is actually in the DOM
  }
}

// ── Fullscreen for a VNC pane ──────────────────────────────────────────────
// See the .vnc-host.fs comment for why this is CSS-first: on iPhone Safari
// Element.requestFullscreen simply does not exist, so the class is what does
// the work everywhere and the native call is a best-effort extra.
function enterVncFs(app){
  const host = vncPane(app).querySelector('.vnc-host');
  if (!host || host.classList.contains('fs')) return;
  host.classList.add('fs');

  const x = document.createElement('button');
  x.className = 'vnc-fs-exit';
  x.textContent = '✕';
  x.setAttribute('aria-label', t('Έξοδος από πλήρη οθόνη'));
  x.onclick = () => exitVncFs(app);
  host.appendChild(x);

  // Esc is the reflex on a laptop; the native handler below covers the case
  // where the browser grants real fullscreen and eats the key itself.
  host._fsKey = ev => { if (ev.key === 'Escape') exitVncFs(app); };
  document.addEventListener('keydown', host._fsKey);

  // Bonus only: also hides the URL bar where supported. A rejected promise
  // (user gesture rules, iOS) leaves the CSS fullscreen perfectly usable.
  const rq = host.requestFullscreen || host.webkitRequestFullscreen;
  if (rq){ try { const p = rq.call(host); if (p && p.catch) p.catch(() => {}); }
           catch (e) {} }

  // noVNC rescales from a resize event; fixed-position changes do not always
  // emit one inside the iframe, so nudge it once the class has taken effect.
  setTimeout(() => { try { window.dispatchEvent(new Event('resize')); } catch(e){} }, 60);
}

function exitVncFs(app){
  const host = vncPane(app).querySelector('.vnc-host');
  if (!host || !host.classList.contains('fs')) return;
  host.classList.remove('fs');
  const x = host.querySelector('.vnc-fs-exit');
  if (x) x.remove();
  if (host._fsKey){ document.removeEventListener('keydown', host._fsKey); host._fsKey = null; }
  if (document.fullscreenElement || document.webkitFullscreenElement){
    const ex = document.exitFullscreen || document.webkitExitFullscreen;
    if (ex){ try { const p = ex.call(document); if (p && p.catch) p.catch(() => {}); }
             catch (e) {} }
  }
  setTimeout(() => { try { window.dispatchEvent(new Event('resize')); } catch(e){} }, 60);
}

// Leaving fullscreen by the browser's own affordance (Esc it handled, the iOS
// swipe, the Android back gesture) must not strand the pane in the fs class —
// it would stay pinned over the dashboard with no way back.
['fullscreenchange', 'webkitfullscreenchange'].forEach(evt =>
  document.addEventListener(evt, () => {
    if (document.fullscreenElement || document.webkitFullscreenElement) return;
    Object.keys(VNC_APPS).forEach(a => {
      const p = vncPane(a), h = p && p.querySelector('.vnc-host.fs');
      if (h) exitVncFs(a);
    });
  }));

// ── RTAB-Map controls + progress ───────────────────────────────────────────
// The VNC view already carries the full rtabmap_viz, so this strip deliberately
// does NOT duplicate it. It answers the one question the GUI makes you hunt for
// on a phone-sized screen: is the mapper actually taking frames, or is the
// window just sitting there looking alive? A keyframe count that climbs while
// you drive is the only trustworthy answer.
// Built with DOM calls rather than an innerHTML template on purpose: the i18n
// test scans for Greek inside quotes, and HTML attributes written inside a
// template literal next to a t() call read as one long quoted Greek span to it
// (it is explicitly not a full JS tokeniser). textContent keeps every Greek
// string a lone t() argument, which is also what the rest of this file does.
function rtabControls(){
  const card = document.createElement('div');
  card.className = 'card';

  const h = document.createElement('h3');
  h.textContent = t('Χαρτογράφηση') + ' ';
  const badge = document.createElement('span');
  badge.className = 'badge'; badge.id = 'rt-state'; badge.textContent = '—';
  h.appendChild(badge);
  card.appendChild(h);

  const grid = document.createElement('div');
  grid.className = 'grid2';
  [[t('Καρέ-κλειδιά'), 'rt-nodes'], [t('Κλεισίματα βρόχου'), 'rt-loops']]
   .forEach(([label, id]) => {
    const k = document.createElement('span');
    k.className = 'k'; k.textContent = label;
    const v = document.createElement('span');
    v.className = 'v'; v.id = id; v.textContent = '—';
    grid.appendChild(k); grid.appendChild(v);
  });
  card.appendChild(grid);

  const row = document.createElement('div');
  row.className = 'row';
  row.style.marginTop = '10px';
  const mk = (label, cls, cmd) => {
    const b = document.createElement('button');
    b.className = 'btn ' + (cls || '');
    b.textContent = label;
    b.onclick = () => send({type:'rtabmap_cmd', cmd});
    row.appendChild(b);
  };
  mk(t('⏸ Παύση'), '', 'pause');
  mk(t('▶ Συνέχεια'), '', 'resume');
  mk(t('🆕 Νέος χάρτης'), 'warn', 'trigger_new_map');
  const saveBtn = document.createElement('button');
  saveBtn.className = 'btn';
  saveBtn.textContent = t('💾 Αποθήκευση');
  saveBtn.onclick = () => {
    send({type: 'rtabmap_cmd', cmd: 'save_snapshot'});
    rtabSavePoll();
  };
  row.appendChild(saveBtn);
  card.appendChild(row);
  const saveMsg = document.createElement('p');
  saveMsg.id = 'rt-save-msg';
  saveMsg.style.cssText = 'font-size:11.5px;color:#71717a;margin:8px 0 0';
  card.appendChild(saveMsg);
  const p = document.createElement('p');
  p.style.cssText = 'font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6';
  p.textContent = t('Ο χάρτης αποθηκεύεται μόνος του στο ~/.home_robot/rtabmap/house.db. Για εξαγωγή σε .ply/.obj χρησιμόποιησε File → Export στο ίδιο το RTAB-Map παραπάνω. ‼️ Το «Νέος χάρτης» ξεκινά καθαρή συνεδρία — ό,τι έχεις χαρτογραφήσει ως τώρα μένει στη βάση αλλά βγαίνει από τον τρέχοντα χάρτη. Το «💾 Αποθήκευση» παίρνει ένα στιγμιότυπο του μέχρι τώρα χάρτη σε ξεχωριστό αρχείο, χωρίς να διακόψει τη χαρτογράφηση.');
  card.appendChild(p);

  const savedH = document.createElement('h3');
  savedH.style.marginTop = '16px';
  savedH.textContent = t('Αποθηκευμένοι χάρτες');
  card.appendChild(savedH);
  const savedHint = document.createElement('p');
  savedHint.style.cssText = 'font-size:11.5px;color:#71717a;margin:2px 0 8px';
  savedHint.textContent = t('«Προβολή» χτίζει αυτόματα το συναρμολογημένο 3D cloud '
                           + 'σε νέα καρτέλα (~15s) — η πρόοδος φαίνεται live.');
  card.appendChild(savedHint);
  const savedList = document.createElement('div');
  savedList.id = 'rt-saved-list';
  savedList.textContent = t('Φόρτωση…');
  card.appendChild(savedList);
  // NOT rtabSavedRefresh() here — this card is still detached at this point
  // (the caller appends it right after rtabControls() returns), so $('rt-
  // saved-list') would find nothing and the guard would silently no-op.
  // Call it once the card is actually in the document instead.
  return card;
}

function onRtab(m){
  const st = $('rt-state'); if (!st) return;   // tab not built yet
  st.className = 'pill' + (m.live ? ' ok' : '');
  st.textContent = m.live ? t('χαρτογραφεί') : t('ανενεργό');
  $('rt-nodes').textContent = m.live ? m.nodes : '—';
  $('rt-loops').textContent = m.live ? m.loops : '—';
}

// Saved 3D-map snapshots (see _rtab_save_snapshot / /rtabmap/saved). Polls
// itself while a save is in flight — the DB is currently ~200+ MB so the
// sqlite backup takes real seconds, not the instant a click implies.
async function rtabSavedRefresh(){
  const box = $('rt-saved-list'); if (!box) return;   // tab not built yet
  let d;
  try { d = await (await fetch('/rtabmap/saved' + (TOKEN_QS || ''))).json(); }
  catch(e){ return; }
  const msg = $('rt-save-msg');
  if (msg) msg.textContent = d.saving ? t('Αποθήκευση σε εξέλιξη…')
                            : (d.error ? t('Η αποθήκευση απέτυχε — δες τα logs.') : '');
  box.innerHTML = '';
  if (!d.snapshots.length){
    box.textContent = t('Δεν υπάρχουν ακόμα αποθηκευμένοι χάρτες.');
  } else {
    for (const s of d.snapshots){
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:10px;padding:6px 0;'
                        + 'border-bottom:1px solid #27272a';
      const when = new Date(s.mtime * 1000).toLocaleString('el-GR');
      const label = document.createElement('span');
      label.style.flex = '1';
      label.innerHTML = `<b>${s.name}</b>`
        + `<span style="color:#71717a;font-size:11.5px"> · ${when} · ${s.mb} MB</span>`;
      row.appendChild(label);
      const viewBtn = document.createElement('button');
      viewBtn.className = 'btn';
      viewBtn.textContent = t('👁 Προβολή');
      viewBtn.onclick = () => rtabViewOpen(s.name, viewBtn);
      row.appendChild(viewBtn);
      const delBtn = document.createElement('button');
      delBtn.className = 'btn';
      delBtn.textContent = '🗑';
      delBtn.title = t('Διαγραφή χάρτη');
      delBtn.onclick = () => rtabSnapshotDelete(s.name);
      row.appendChild(delBtn);
      box.appendChild(row);
    }
  }
  if (d.saving) setTimeout(rtabSavedRefresh, 3000);
}
function rtabSavePoll(){
  const msg = $('rt-save-msg');
  if (msg) msg.textContent = t('Αποθήκευση σε εξέλιξη…');
  setTimeout(rtabSavedRefresh, 1000);
}

async function rtabSnapshotDelete(name){
  if(!confirm(t('Διαγραφή; ') + name)) return;
  const msg = $('rt-save-msg');
  if (msg) msg.textContent = t('Διαγραφή…');
  try {
    const r = await (await fetch('/rtabmap/delete/' + encodeURIComponent(name)
                                 + (TOKEN_QS || ''))).json();
    if (msg) msg.textContent = r.ok ? '' : t('Απέτυχε') + ': ' + (r.error || '');
  } catch(e){ if (msg) msg.textContent = t('Απέτυχε') + ': ' + e; }
  rtabSavedRefresh();
}

// Opens a SAVED snapshot (read-only, its own process/display — see the
// rtabview app in gui_session.sh) in a new tab streamed over the same noVNC
// bridge the live tabs use. Always stop-then-start: rtabview is a single VNC
// slot, so switching which file it shows needs the old viewer torn down
// first, or 'start' just reports "already running" against the wrong file.
async function rtabViewOpen(name, btn){
  const label = btn ? btn.textContent : '';
  if (btn){ btn.disabled = true; btn.textContent = t('Άνοιγμα…'); }
  try {
    await fetch('/gui/rtabview/stop' + (TOKEN_QS || ''));
    const qs = 'file=' + encodeURIComponent(name)
             + (TOKEN_QS ? '&' + TOKEN_QS.slice(1) : '');
    const r = await (await fetch('/gui/rtabview/start?' + qs)).json();
    if (!r.running){
      alert(t('Άνοιγμα απέτυχε: ') + r.result);
      return;
    }
    window.open('/vncview/rtabview' + TOKEN_QS, '_blank');
  } catch(e){
    alert(t('Άνοιγμα απέτυχε.'));
  } finally {
    if (btn){ btn.disabled = false; btn.textContent = label; }
  }
}

async function guiCall(app, action){
  const r = await fetch(`/gui/${app}/${action}` + TOKEN_QS);
  if (!r.ok) throw new Error('gui ' + r.status);
  return r.json();
}

async function ensureVnc(app){
  const st = vncState[app];
  if (st.frame || st.busy) return;      // already live or mid-flight
  if (!HAS_NOVNC){
    renderVnc(app, 'off', t('Λείπει το noVNC — <code>sudo apt install novnc</code>'));
    return;
  }
  st.busy = true;
  try {
    const j = await guiCall(app, 'status');
    renderVnc(app, j.running ? 'live' : 'off',
              j.running ? '' : t('Η συνεδρία δεν τρέχει.'));
  } catch(e){
    renderVnc(app, 'off', t('Δεν απαντά ο server.'));
  } finally { st.busy = false; }
}

async function startVnc(app){
  const st = vncState[app];
  if (st.busy) return;
  st.busy = true;
  renderVnc(app, 'starting',
    app === 'gazebo' ? t('Το Gazebo θέλει ~75 δευτερόλεπτα σε software rendering…')
                     : t('Ξεκινά η γραφική συνεδρία…'));
  try {
    const j = await guiCall(app, 'start');
    renderVnc(app, j.running ? 'live' : 'off', j.running ? '' : j.result);
  } catch(e){
    renderVnc(app, 'off', t('Απέτυχε η εκκίνηση.'));
  } finally { st.busy = false; }
}

async function stopVnc(app){
  vncState[app].frame = null;
  renderVnc(app, 'off', t('Σταμάτησε.'));
  try { await guiCall(app, 'stop'); } catch(e){}
  ensureVnc(app);
}

function refreshVnc(app){ vncState[app].frame = null; ensureVnc(app); }

// ── map canvas ─────────────────────────────────────────────────────────────
const canvas = $('map-canvas'), ctx = canvas.getContext('2d'), wrap = $('map-wrap');

function scale(){ return mapInfo ? Math.min(canvas.width/mapInfo.width, canvas.height/mapInfo.height) : 1; }
function offX(){  return mapInfo ? (canvas.width  - mapInfo.width  * scale()) / 2 : 0; }
function offY(){  return mapInfo ? (canvas.height - mapInfo.height * scale()) / 2 : 0; }

function w2c(wx, wy){
  if(!mapInfo) return {x:0,y:0};
  const s=scale();
  return {
    x: offX() + (wx - mapInfo.origin[0]) / mapInfo.resolution * s,
    y: offY() + (mapInfo.height - (wy - mapInfo.origin[1]) / mapInfo.resolution) * s,
  };
}
function c2w(cx, cy){
  if(!mapInfo) return null;
  const s=scale();
  return {
    x: mapInfo.origin[0] + (cx - offX()) / s * mapInfo.resolution,
    y: mapInfo.origin[1] + (mapInfo.height - (cy - offY()) / s) * mapInfo.resolution,
  };
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#0c0c0e';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  if(!mapImg||!mapInfo) return;
  const s=scale(), ox=offX(), oy=offY();
  ctx.drawImage(mapImg, ox, oy, mapInfo.width*s, mapInfo.height*s);

  // Where the odometry loses ground. Drawn UNDER everything else: it is
  // history, not something happening now, and it must never be mistaken for an
  // obstacle or a plan.
  if(slipMapOn && slipMap && slipMap.cells && slipMap.cells.length){
    const worst = slipMap.cells[0].m || 1;
    const cellPx = (slipMap.cell || 0.5) / mapInfo.resolution * scale();
    slipMap.cells.forEach(c => {
      const p = w2c(c.x, c.y);
      // Alpha by severity rather than a colour ramp: the map underneath has to
      // stay readable, and "how bad" only needs to be comparable, not precise.
      const a = 0.12 + 0.55 * Math.min(1, c.m / worst);
      ctx.fillStyle = `rgba(248,113,113,${a})`;
      ctx.fillRect(p.x - cellPx/2, p.y - cellPx/2, cellPx, cellPx);
    });
  }

  // Keepout zones — striped so they read as "excluded" rather than "tinted",
  // the way rooms are. A half-drawn zone (first corner clicked, waiting for
  // the second) gets a dashed preview so the pending click isn't invisible.
  if(kzData){
    ctx.fillStyle='rgba(248,113,113,.28)'; ctx.strokeStyle='rgba(248,113,113,.85)'; ctx.lineWidth=1.5;
    Object.values(kzData).forEach(z=>{
      if(z.shape==='circle'){
        const c=w2c(z.x,z.y); const rp=z.radius/mapInfo.resolution*scale();
        ctx.beginPath(); ctx.arc(c.x,c.y,rp,0,Math.PI*2); ctx.fill(); ctx.stroke();
      } else {
        const c=w2c(z.x,z.y);
        const wpx=z.width/mapInfo.resolution*scale(), hpx=z.height/mapInfo.resolution*scale();
        ctx.fillRect(c.x-wpx/2, c.y-hpx/2, wpx, hpx);
        ctx.strokeRect(c.x-wpx/2, c.y-hpx/2, wpx, hpx);
      }
    });
  }
  if(kzCorner){
    const c=w2c(kzCorner.x,kzCorner.y);
    ctx.strokeStyle='#f87171'; ctx.setLineDash([4,3]); ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(c.x,c.y,6,0,Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
  }
  if(prrCorner){
    const c=w2c(prrCorner.x,prrCorner.y);
    ctx.strokeStyle='#cc44ff'; ctx.setLineDash([4,3]); ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(c.x,c.y,6,0,Math.PI*2); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Robot trail — where it has actually driven this session (pose(m) pushes
  // into robotTrail on >5cm movement). History, drawn before the plan so the
  // (brighter, thicker) upcoming route reads on top of it.
  if(robotTrail.length > 1){
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 3.5;
    ctx.beginPath();
    robotTrail.forEach(([x, y], i) => {
      const p = w2c(x, y);
      i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y);
    });
    ctx.stroke();
  }

  // Global plan
  if(plan && plan.length>1){
    ctx.strokeStyle='rgba(96,165,250,.9)'; ctx.lineWidth=2.5;
    ctx.beginPath();
    plan.forEach((p,i)=>{ const c=w2c(p[0],p[1]); i?ctx.lineTo(c.x,c.y):ctx.moveTo(c.x,c.y); });
    ctx.stroke();
  }

  // LiDAR scan
  if(scan && pose){
    ctx.fillStyle='rgba(0,200,255,0.75)';
    const laserYaw = pose.yaw + LASER_YAW_OFFSET;
    const lx = pose.x + LASER_X * Math.cos(pose.yaw);
    const ly = pose.y + LASER_X * Math.sin(pose.yaw);
    for(let i=0;i<scan.ranges.length;i++){
      const r=scan.ranges[i];
      if(!r||r<0.1||r>8) continue;
      const a = laserYaw + scan.angle_min + i*scan.angle_inc;
      const p = w2c(lx + r*Math.cos(a), ly + r*Math.sin(a));
      ctx.fillRect(p.x-1.5, p.y-1.5, 3, 3);
    }
  }

  // Nav goal
  if(goal){
    const g=w2c(goal.x,goal.y);
    ctx.strokeStyle='#ffa040'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(g.x,g.y,10,0,Math.PI*2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(g.x,g.y-15); ctx.lineTo(g.x,g.y+15);
    ctx.moveTo(g.x-15,g.y); ctx.lineTo(g.x+15,g.y);
    ctx.stroke();
  }

  // Robot arrow
  if(pose){
    const rp=w2c(pose.x,pose.y);
    ctx.save();
    ctx.translate(rp.x,rp.y);
    ctx.rotate(-pose.yaw);
    ctx.fillStyle='#00e08a'; ctx.strokeStyle='#003'; ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(17,0); ctx.lineTo(-9,10); ctx.lineTo(-5,0); ctx.lineTo(-9,-10);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
}

function resize(){
  if(!wrap.clientWidth) return;
  canvas.width=wrap.clientWidth; canvas.height=wrap.clientHeight; draw();
}
window.addEventListener('resize',resize);
new ResizeObserver(resize).observe(wrap);

function drawNerfMap(){
  const c = $('nf-map'); if(!c) return;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#18181b'; ctx.fillRect(0, 0, c.width, c.height);
  if(!nfPoints.length) return;
  const xs = nfPoints.map(p=>p[0]), ys = nfPoints.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const pad = 20;
  const spanX = Math.max(maxX-minX, 0.5), spanY = Math.max(maxY-minY, 0.5);
  const scale = Math.min((c.width-2*pad)/spanX, (c.height-2*pad)/spanY);
  const midX = (minX+maxX)/2, midY = (minY+maxY)/2;
  const cx = c.width/2, cy = c.height/2;
  const toPx = ([x,y]) => [cx+(x-midX)*scale, cy-(y-midY)*scale];  // y up on screen
  ctx.fillStyle = '#22c55e';
  for(const p of nfPoints){
    const [px,py] = toPx(p);
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI*2); ctx.fill();
  }
  const [lx,ly] = toPx(nfPoints[nfPoints.length-1]);
  ctx.fillStyle = '#f59e0b';
  ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI*2); ctx.fill();
}

// ── websocket ──────────────────────────────────────────────────────────────
const HANDLERS = {
  map(m){
    mapInfo={width:m.width,height:m.height,resolution:m.resolution,origin:m.origin};
    const i=new Image(); i.onload=()=>{mapImg=i;draw();}; i.src='data:image/png;base64,'+m.image;
    if (m.rooms) { roomsData=m.rooms; roomLegend(m.rooms); renderRoomEditor(m.rooms); }
    if (m.tinted !== undefined) $('b-tint').checked = m.tinted;
    if (m.keepout) { kzData=m.keepout; renderKeepoutList(kzData); }
  },
  map_rooms(m){ roomsData=m.rooms||{}; roomLegend(roomsData); renderRoomEditor(roomsData); },
  room_saved(m){
    $('room-edit-msg').textContent = m.ok ? t('Αποθηκεύτηκε.') : (m.error || t('Αποτυχία.'));
    $('room-edit-msg').style.color = m.ok ? '#4ade80' : '#f87171';
    // place_room reuses this message: clear the name on success so the next
    // click starts a fresh room instead of re-painting the same one.
    if(m.ok && $('b-place-room').checked) $('pr-name').value = '';
  },
  keepout_zones(m){ kzData=m.zones||{}; renderKeepoutList(kzData); draw(); },
  keepout_saved(m){
    $('kz-msg').textContent = m.ok ? t('Αποθηκεύτηκε.') : (m.error || t('Αποτυχία.'));
    $('kz-msg').style.color = m.ok ? '#4ade80' : '#f87171';
    if(m.ok && $('b-kz-add').checked) $('kz-name').value = '';
  },
  keepout_activated(m){
    $('kz-msg').textContent = m.ok
      ? t('Επανεκκίνηση… η σελίδα θα ξανασυνδεθεί μόνη της.')
      : (m.error || t('Αποτυχία.'));
    $('kz-msg').style.color = m.ok ? '#4ade80' : '#f87171';
  },
  room_picked(m){
    if(!m.name){
      $('room-edit-msg').textContent = t('Δεν βρέθηκε δωμάτιο εκεί.');
      $('room-edit-msg').style.color = '#71717a';
      return;
    }
    const row = Array.from(document.querySelectorAll('#room-edit [data-room]'))
      .find(el => el.dataset.room === m.name);
    if(!row) return;
    row.scrollIntoView({behavior:'smooth', block:'nearest'});
    row.classList.add('picked');
    setTimeout(()=>row.classList.remove('picked'), 1500);
    row.querySelector('.re-name').focus();
  },
  pose(m){
    pose=m;
    $('ix').textContent   = m.x.toFixed(2)+' m';
    $('iy').textContent   = m.y.toFixed(2)+' m';
    $('iyaw').textContent = (m.yaw*180/Math.PI).toFixed(0)+'°';
    // Only keep a point once the robot has actually moved 5cm — the pose
    // feed polls TF at ~2Hz even standing still (see _poll_tf_pose's header),
    // and stacking thousands of identical dots buys nothing.
    const last = robotTrail[robotTrail.length - 1];
    if (!last || Math.hypot(m.x - last[0], m.y - last[1]) > 0.05){
      robotTrail.push([m.x, m.y]);
      if (robotTrail.length > 4000) robotTrail.shift();
    }
    draw();
    drawCompass2();
    if (window.hrScan3d) window.hrScan3d.setPose(m);
  },
  scan(m){ scan=m; draw(); },
  plan(m){ plan=m.points; draw(); },
  odom(m){
    $('ivel').textContent = m.vx.toFixed(2)+' m/s · '+m.wz.toFixed(2)+' r/s';
  },
  room(m){
    $('room-badge').textContent = m.name||'—';
  },
  objects(m){ $('objects').textContent = m.text || '—'; },
  situation(m){ $('situation').textContent = m.text || '—'; },
  gesture(m){ gestureState = m; drawPointRing(); renderGestureBindings(); },
  vocab(m){ renderVocab(m); },
  compass(m){ compassOffset = m.offset; drawCompass2(); },
  fusion(m){ renderFusion(m); },
  usb_power(m){ usbResult(m); },
  power_profile(m){ powerResult(m); },
  slip_map(m){ renderSlipMap(m); },
  camstate(m){ renderCamState(m); },
  fuseprof(m){ renderFuseProfile(m); },
  hand(m){ handState = m; renderGestureBindings(); },
  people(m){ renderPeople(m); },
  diagnostics(m){ renderDiagnostics(m); },
  sound(m){ soundState = m; renderSound(m); },
  acoustic(m){ renderAcoustic(m); },
  echo(m){ renderEcho(m); },
  sysnet(m){ renderSysNet(m); },
  token(m){ renderToken(m); },
  touch(m){ renderTouch(m); },
  observations(m){ renderObservations(m); },
  object_memory(m){ renderObjectMemory(m); },
  timeline(m){ renderTimeline(m); },
  recall_answer(m){ $('tl-answer').textContent = m.text || ''; },
  arm(m){
    m.names.forEach((n,i)=>{
      armPos[n]=m.pos[i];
      const sl=$('j-'+n), v=$('jv-'+n);
      if(sl && !sl.dataset.dragging) sl.value=m.pos[i];
      if(v) v.textContent=(m.pos[i]*180/Math.PI).toFixed(0)+'°';
    });
    // Feed the 3D view from the REAL joint states, so what is on screen is
    // where the arm is — not where a slider was last dragged.
    armAngles = Object.assign({}, armPos);
    armDraw();
    if('hand' in armPos){
      const g=$('grip'), gv=$('grip-v');
      if(g && !g.dataset.dragging) g.value=armPos.hand;
      if(gv) gv.textContent=(armPos.hand*180/Math.PI).toFixed(0)+'°';
    }
  },
  roomba(m){
    const OI = {0:'off', 1:'passive', 2:'safe', 3:'full'};
    const flag = (on, bad) => `<span class="pill ${on?(bad?'bad':'ok'):''}">${on?t('ΝΑΙ'):t('όχι')}</span>`;
    const age = m.link_age_s;
    // > ~3 s of silence means the base has gone to sleep. Calling that out here
    // is the whole point of the panel: a sleeping Roomba fakes nav bugs.
    $('r-link').innerHTML = age === null || age === undefined
      ? '<span class="pill">—</span>'
      : `<span class="pill ${age<3?'ok':'bad'}">${age.toFixed(1)}s ${age<3?'':'· '+t('ΚΟΙΜΑΤΑΙ')}</span>`;
    // The one-glance answer to "is the base actually awake?". link_age alone
    // needed decoding; an OI mode of passive/off means asleep even when the
    // serial link is answering, so both have to agree before we say awake.
    const awake = age !== null && age !== undefined && age < 3
                  && (m.oi_mode === 2 || m.oi_mode === 3);
    $('r-awake').innerHTML = age === null || age === undefined
      ? '<span class="pill">—</span>'
      : `<span class="pill ${awake?'ok':'bad'}">${awake?'✅ '+t('ΞΥΠΝΙΑ'):'💤 '+t('ΚΟΙΜΑΤΑΙ')}</span>`;
    $('r-mode').innerHTML = `<span class="pill ${m.oi_mode===3||m.oi_mode===2?'ok':'warn'}">`
      + (OI[m.oi_mode] || '—') + `</span> <span style="color:#52525b">${t('ζητά')} ${m.oi_mode_want}</span>`;
    $('r-bump').innerHTML  = flag(m.bump, true);
    $('r-cliff').innerHTML = flag(m.cliff, true);
    $('r-wheel').innerHTML = flag(m.wheel_drop, true);
    $('r-motors').textContent = `${m.left_mm_s} / ${m.right_mm_s} mm/s`;
    $('r-dock').innerHTML  = flag(m.docking, false);
  },
  imu(m){ onImu(m); },
  rtabmap(m){ onRtab(m); },
  costmap(m){ onCostmap(m); },
  nerf(m){
    const st = $('nf-state'); if(!st) return;
    st.className = 'pill' + (m.active ? ' ok' : '');
    st.textContent = m.active ? t('καταγράφει') : t('σταματημένο');
    $('nf-frames').textContent = `${m.frames} / ${m.max_frames}`;
    $('nf-dir').textContent = m.dir || '—';
    // frames resets to 0 at the start of every session (see nerf_capture_node's
    // _cb_capture) — that is also our signal to clear the old coverage dots.
    // nfLastFrames must reset too, or a new session whose Nth frame matches the
    // previous session's last-seen count gets silently skipped below.
    if(m.frames === 0){ nfPoints.length = 0; nfLastFrames = -1; }
    if(m.last_xy && m.frames !== nfLastFrames){
      nfPoints.push(m.last_xy);
      nfLastFrames = m.frames;
    }
    drawNerfMap();
  },
  nerf_train(m){
    const st = $('nt-state'); if(!st) return;
    st.className = 'pill' + (m.running || m.done ? ' ok' : '');
    st.textContent = m.running ? t('εκπαιδεύεται')
                    : m.done    ? t('ολοκληρώθηκε')
                    : m.error   ? t('σφάλμα')
                    :             t('σταματημένο');
    $('nt-step').textContent = (m.step != null && m.steps != null) ? `${m.step} / ${m.steps}` : '—';
    $('nt-loss').textContent = (m.loss != null) ? `${m.loss.toFixed(4)} / ${m.psnr.toFixed(1)} dB` : '—';
    $('nt-eta').textContent  = (m.elapsed_min != null) ? `${m.elapsed_min.toFixed(1)}′ / ETA ${m.eta_min}′` : '—';
    const errEl = $('nt-error');
    if(m.error){
      errEl.textContent = m.error;
      errEl.style.display = 'block';
      $('b-nerf-stop-perception').style.display = m.gpu_busy ? '' : 'none';
    } else {
      errEl.style.display = 'none';
      $('b-nerf-stop-perception').style.display = 'none';
    }
    $('b-nerf-train-go').disabled = !!m.running;
    $('b-nerf-train-stop').disabled = !m.running;
  },
  vision(m){
    const e = $('vis-count'); if(!e) return;
    // "0 αντικείμενα" and a blank overlay mean the same thing but read very
    // differently — the count is what tells you the detector is alive at all.
    e.textContent = m.people
      ? `${m.objects} · 👤${m.people}`
      : `${m.objects}`;
  },
  dock(m){ $('r-dockst').textContent = m.status || '—'; },
  estop(m){
    estop = !!m.on;
    $('estop').classList.toggle('engaged', estop);
    $('estop').textContent = estop ? t('▶ ΞΕΜΠΛΟΚΑΡΙΣΜΑ') : '■ STOP';
  },
  speaking(m){
    $('title').style.opacity = m.on ? '.55' : '1';
  },
  chat(m){ addMsg(m.role, m.text); },
  log(m){ addLog(m); },
  cloud(m){ onCloud(m); },
  sys(m){
    renderSys(m);
    $('s-nodecount').textContent = m.nodes.length;
    $('s-nodes').textContent = m.nodes.join('\n');
    updatePerceptionToggles(m.nodes);
  },
  llm_backend(m){ onBackend(m); },
  quota(m){ onQuota(m); },
  mission(m){ onMission(m); },
  fall(m){ onFall(m); },
  fall_event(m){ onFallEvent(m); },
  speaker(m){ onSpeaker(m); },
  doa_rotate(m){ onDoaRotate(m); },
  safety(m){ onSafety(m.v); if (m.skirt_margin_mm !== undefined) skirtSetDisplay(m.skirt_margin_mm); },
  arm_settings(m){ onArmSettings(m.v); },
  mic(m){ onMic(m.v); },
  vad(m){ onVad(m.on); },
};

// ── safety clearances ──────────────────────────────────────────────────────
// Two numbers per row and they are not the same thing: what this page ASKED
// for (saved to disk, survives `robot max`) and what the live node ANSWERED.
// They differ whenever a launch argument overrides the file, or the owning node
// is not running at all — and a panel that showed only the first would be a
// picture of a guard that may not exist. The control is greyed out and the
// badge says so rather than moving a slider nothing reads.
const SAFETY_SPECS = __SAFETY_SPECS__;
const SAFETY_INFO  = __SAFETY_INFO__;

function safetyBuild(){
  document.querySelectorAll('#p-safe .sfrow').forEach(row => {
    const key = row.dataset.key, spec = SAFETY_SPECS[key];
    if (!spec) return;
    const lab = row.querySelector('.sflab');
    if (spec.kind === 'bool'){
      // The <label> and its text are already in the markup (so the translator
      // owns that text node) — only the input itself is injected here.
      const box = document.createElement('input');
      box.type = 'checkbox'; box.dataset.input = key;
      row.querySelector('[data-box]').appendChild(box);
      const val = document.createElement('span');
      val.className = 'sfval'; val.dataset.val = key;
      lab.appendChild(val);
      box.addEventListener('change', () =>
        send({type:'safety_set', key, value: box.checked}));
    } else {
      const val = document.createElement('span');
      val.className = 'sfval'; val.dataset.val = key;
      lab.appendChild(val);
      const sl = document.createElement('input');
      sl.type = 'range'; sl.dataset.input = key;
      sl.min = spec.lo; sl.max = spec.hi; sl.step = spec.step;
      sl.value = spec.def;
      row.insertBefore(sl, lab.nextSibling);
      // 'input' paints while dragging (no traffic), 'change' commits on
      // release — one parameter write per gesture instead of forty.
      sl.addEventListener('input',  () => safetyPaintRow(key, +sl.value, null));
      sl.addEventListener('change', () =>
        send({type:'safety_set', key, value:+sl.value}));
    }
  });
  const info = $('sf-info');
  SAFETY_INFO.forEach(row => {
    const k = document.createElement('span');
    k.className = 'k'; k.textContent = row.source;
    const v = document.createElement('span');
    v.className = 'v'; v.textContent = row.value.toFixed(2) + ' ' + row.unit;
    info.appendChild(k); info.appendChild(v);
  });
}

// Decimals that match the step, so 0.5 does not render as "0.5" next to "0.22".
function safetyFmt(key, v){
  const spec = SAFETY_SPECS[key];
  return v.toFixed(spec.step < 0.05 ? 2 : (spec.step < 1 ? 2 : 1));
}

function safetyPaintRow(key, asked, live){
  const spec = SAFETY_SPECS[key];
  const row = document.querySelector('#p-safe .sfrow[data-key="' + key + '"]');
  if (!row || !spec) return;
  const val = row.querySelector('[data-val="' + key + '"]');
  if (spec.kind === 'bool'){
    if (val) val.textContent = asked ? t('ΕΝΕΡΓΟ') : t('ΚΛΕΙΣΤΟ');
  } else {
    // Show the live value beside the asked one only when they disagree —
    // otherwise every row would carry the same number twice.
    const same = live === null || live === undefined
                 || Math.abs(live - asked) < 1e-6;
    if (val) val.textContent = safetyFmt(key, asked)
      + (same ? '' : ' → ' + t('ενεργό') + ' ' + safetyFmt(key, live));
    row.classList.toggle('warned',
      (spec.warn_above !== null && asked > spec.warn_above) ||
      (spec.warn_below !== null && asked < spec.warn_below));
  }
}

function onSafety(v){
  let camUp = 0, navUp = 0, baseUp = 0;
  Object.keys(SAFETY_SPECS).forEach(key => {
    const st = v[key];
    if (!st) return;
    const row = document.querySelector('#p-safe .sfrow[data-key="' + key + '"]');
    const inp = document.querySelector('[data-input="' + key + '"]');
    // Never fight the user's finger: a slider being dragged keeps its value.
    if (inp && document.activeElement !== inp){
      if (SAFETY_SPECS[key].kind === 'bool') inp.checked = !!st.set;
      else inp.value = st.set;
    }
    if (inp) inp.disabled = st.nodes === 0;
    if (row) row.classList.toggle('off', st.nodes === 0);
    safetyPaintRow(key, st.set, st.live);
    if (key === 'inflation_radius' || key === 'cost_scaling') navUp = Math.max(navUp, st.nodes);
    else if (key.startsWith('bump') || key.startsWith('cliff') || key.startsWith('wheel')) baseUp = Math.max(baseUp, st.nodes);
    else camUp = Math.max(camUp, st.nodes);
  });
  $('sf-cam-badge').textContent  = camUp  ? t('ενεργό') : t('δεν τρέχει');
  $('sf-nav-badge').textContent  = navUp  ? t('ενεργό') : t('δεν τρέχει');
  $('sf-base-badge').textContent = baseUp ? t('ενεργό') : t('δεν τρέχει');
}

// ── microphone settings ──────────────────────────────────────────────────────
// Same shape and the same reasoning as the safety block above — see
// mic_settings.py for which knobs made the cut and why the rest didn't.
const MIC_SPECS = __MIC_SPECS__;
const MIC_INFO  = __MIC_INFO__;

// '#rrggbb' <-> the plain 0xRRGGBB int doa_node's led_color_* params hold.
function micHex(v){ return '#' + (v & 0xFFFFFF).toString(16).padStart(6, '0'); }

function micBuild(){
  document.querySelectorAll('#p-sound .sfrow').forEach(row => {
    const key = row.dataset.key, spec = MIC_SPECS[key];
    if (!spec) return;
    const lab = row.querySelector('.sflab');
    if (spec.kind === 'bool'){
      const box = document.createElement('input');
      box.type = 'checkbox'; box.dataset.input = key;
      row.querySelector('[data-box]').appendChild(box);
      const val = document.createElement('span');
      val.className = 'sfval'; val.dataset.val = key;
      lab.appendChild(val);
      box.addEventListener('change', () =>
        send({type:'mic_set', key, value: box.checked}));
    } else if (spec.kind === 'color'){
      const val = document.createElement('span');
      val.className = 'sfval'; val.dataset.val = key;
      lab.appendChild(val);
      const sw = document.createElement('input');
      sw.type = 'color'; sw.dataset.input = key;
      sw.value = micHex(spec.def);
      row.insertBefore(sw, lab.nextSibling);
      // Colour pickers don't drag-then-release like a range slider — 'input'
      // already fires only on a committed pick (closing the swatch), so
      // there's no separate "paint while dragging" step here.
      sw.addEventListener('input', () => {
        const v = parseInt(sw.value.slice(1), 16);
        micPaintRow(key, v, null);
        send({type:'mic_set', key, value: v});
      });
    } else {
      const val = document.createElement('span');
      val.className = 'sfval'; val.dataset.val = key;
      lab.appendChild(val);
      const sl = document.createElement('input');
      sl.type = 'range'; sl.dataset.input = key;
      sl.min = spec.lo; sl.max = spec.hi; sl.step = spec.step;
      sl.value = spec.def;
      row.insertBefore(sl, lab.nextSibling);
      sl.addEventListener('input',  () => micPaintRow(key, +sl.value, null));
      sl.addEventListener('change', () =>
        send({type:'mic_set', key, value:+sl.value}));
    }
  });
}

function micFmt(key, v){
  const spec = MIC_SPECS[key];
  return v.toFixed(spec.step < 0.05 ? 3 : (spec.step < 1 ? 2 : 0));
}

function micPaintRow(key, asked, live){
  const spec = MIC_SPECS[key];
  const row = document.querySelector('#p-sound .sfrow[data-key="' + key + '"]');
  if (!row || !spec) return;
  const val = row.querySelector('[data-val="' + key + '"]');
  if (spec.kind === 'bool'){
    if (val) val.textContent = asked ? t('ΕΝΕΡΓΟ') : t('ΚΛΕΙΣΤΟ');
  } else if (spec.kind === 'color'){
    const same = live === null || live === undefined || live === asked;
    if (val) val.textContent = micHex(asked)
      + (same ? '' : ' → ' + t('ενεργό') + ' ' + micHex(live));
  } else {
    const same = live === null || live === undefined
                 || Math.abs(live - asked) < 1e-6;
    if (val) val.textContent = micFmt(key, asked)
      + (same ? '' : ' → ' + t('ενεργό') + ' ' + micFmt(key, live));
    row.classList.toggle('warned',
      (spec.warn_above !== null && asked > spec.warn_above) ||
      (spec.warn_below !== null && asked < spec.warn_below));
  }
}

function onMic(v){
  let micUp = 0, ledUp = 0;
  Object.keys(MIC_SPECS).forEach(key => {
    const st = v[key];
    if (!st) return;
    const row = document.querySelector('#p-sound .sfrow[data-key="' + key + '"]');
    const inp = document.querySelector('[data-input="' + key + '"]');
    if (inp && document.activeElement !== inp){
      if (MIC_SPECS[key].kind === 'bool') inp.checked = !!st.set;
      else if (MIC_SPECS[key].kind === 'color') inp.value = micHex(st.set);
      else inp.value = st.set;
    }
    if (inp) inp.disabled = st.nodes === 0;
    if (row) row.classList.toggle('off', st.nodes === 0);
    micPaintRow(key, st.set, st.live);
    if (key.startsWith('led_')) ledUp = Math.max(ledUp, st.nodes);
    else micUp = Math.max(micUp, st.nodes);
  });
  $('mic-badge').textContent = micUp ? t('ενεργό') : t('δεν τρέχει');
  $('led-badge').textContent = ledUp ? t('ενεργό') : t('δεν τρέχει');
  if (v.wake_model) onWakeModel(v.wake_model);
}

// ── wake word model picker ───────────────────────────────────────────────────
// Not one of MIC_SPECS: it's a closed set of strings (a <select>), not a
// number/bool/colour — see mic_settings.py's WAKE_MODEL_CHOICES for why.
const WAKE_MODEL_CHOICES = __WAKE_MODEL_CHOICES__;
// Labels, not the bare ids — set as plain text (like usbBuild()'s button
// labels) so the same DOM-walk that translates static markup picks these up
// too, since wakeModelBuild() runs before the first applyLang().
const WAKE_MODEL_LABELS = {
  hey_robot:   'Έι ρομπότ (προεπιλογή, εκπαιδευμένο)',
  alexa:       'Alexa (αγγλικό, χωρίς εκπαίδευση)',
  hey_jarvis:  'Hey Jarvis (αγγλικό, χωρίς εκπαίδευση)',
  hey_mycroft: 'Hey Mycroft (αγγλικό, χωρίς εκπαίδευση)',
  hey_marvin:  'Hey Marvin (αγγλικό, χωρίς εκπαίδευση)',
};

function wakeModelBuild(){
  const sel = $('wake-model');
  if (!sel || sel.childElementCount) return;
  WAKE_MODEL_CHOICES.forEach(id => {
    const o = document.createElement('option');
    o.value = id;
    o.textContent = WAKE_MODEL_LABELS[id] || id;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () =>
    send({type:'mic_set', key:'wake_model', value: sel.value}));
}

function onWakeModel(st){
  const sel = $('wake-model');
  if (!sel) return;
  if (document.activeElement !== sel) sel.value = st.set;
  if (!st.nodes){ $('wm-badge').textContent = t('δεν τρέχει'); return; }
  // A rejected switch (bad/missing model) leaves the NODE's own parameter at
  // the old value — same "asked X, live Y" disagreement the safety sliders
  // show, not a silent failure.
  $('wm-badge').textContent = (st.live && st.live !== st.set)
    ? t('ενεργό') + ': ' + (WAKE_MODEL_LABELS[st.live] || st.live)
    : t('ενεργό');
}

function onVad(on){
  const el = $('sd-vad');
  if (!el) return;
  el.textContent = on ? t('ΝΑΙ') + ' 🔵' : t('όχι');
}

// ── collision_monitor hard-stop skirt ────────────────────────────────────────
// NOT one of SAFETY_SPECS: this needs a restart to take effect (see
// home_robot/collision_skirt.py), so it gets its own control instead of
// pretending to be a live SetParameters slider like the rows above.
const SKIRT_MARGINS = __SKIRT_MARGINS__;
const SKIRT_DEFAULT_MM = __SKIRT_DEFAULT_MM__;

function skirtBuild(){
  const sel = $('sk-mm');
  SKIRT_MARGINS.forEach(mm => {
    const o = document.createElement('option');
    o.value = mm; o.textContent = mm + ' mm';
    sel.appendChild(o);
  });
  // Best guess before the first 'safety' broadcast lands (~3s at worst);
  // skirtSetDisplay() overwrites this with the real, live-read value.
  skirtSetDisplay(SKIRT_DEFAULT_MM);
}

function skirtSetDisplay(mm){
  $('sk-badge').textContent = mm + ' mm';
  // Never fight the user's finger: leave their pick alone while they're
  // choosing, same rule safety sliders follow above.
  if (document.activeElement !== $('sk-mm')) $('sk-mm').value = mm;
}

async function skirtApply(){
  const mm = $('sk-mm').value;
  if (!confirm(t('Αυτό είναι το σκληρό όριο πλήρους στάσης, όχι η απόσταση '
      + 'σχεδίασης διαδρομής. Χρειάζεται πλήρη επανεκκίνηση (~90 δευτερόλεπτα) '
      + 'και μικρότερη απόσταση αφήνει το ρομπότ να πλησιάσει περισσότερο πριν '
      + 'σταματήσει απότομα. Να συνεχίσω;'))) return;
  $('sk-msg').textContent = t('Επανεκκίνηση… η σελίδα θα ξανασυνδεθεί μόνη της.');
  try {
    const r = await (await fetch('/safety/skirt/' + mm + (TOKEN_QS || ''))).json();
    if (r.error) $('sk-msg').textContent = t('Απέτυχε') + ': ' + r.error;
  } catch(e) { /* the server is going down under us; that IS the success case */ }
}
$('b-sk-apply').onclick = skirtApply;

// ── turn-toward-the-speaker switch ─────────────────────────────────────────
// doa_node owns this bit; we only ever paint what it reports. Ticking the box
// sends a request, and the box moves when the answer comes back — if doa_node
// is not running the switch stays put instead of lying about a feature that is
// not there to arm.
function onDoaRotate(m){
  $('dr-rotate').checked = !!m.on;
  $('dr-badge').textContent = m.on ? t('ΕΝΕΡΓΗ') : t('κλειστή');
}

// ── touch ──────────────────────────────────────────────────────────────────
function renderTouch(m){
  $('tc-badge').textContent = m.contact ? t('ΕΠΑΦΗ')
    : (m.baseline === null ? t('περιμένει') : t('ελεύθερος'));
  $('tc-excess').textContent = m.contact ? '+' + m.excess : '—';
  $('tc-hard').textContent = (m.hardness === null) ? '—'
    : m.hardness_el + ' (' + m.hardness + ')';
  $('tc-weight').textContent = (m.weight === null || m.weight === undefined)
    ? (m.have_reference ? '—' : t('χωρίς αναφορά'))
    : m.weight_el + ' (' + m.weight + ')';
}

// ── dashboard key ──────────────────────────────────────────────────────────
function renderToken(m){
  if(m.error){ $('tk-out').textContent = m.error; return; }
  const url = location.origin + '/?t=' + encodeURIComponent(m.token);
  $('tk-badge').textContent = t('νέο κλειδί');
  $('tk-out').innerHTML =
    '<div style="color:#f59e0b;margin-bottom:6px">'
    + esc(t('Κράτησέ το τώρα — μετά την επανεκκίνηση χρειάζεται:')) + '</div>'
    + '<div style="background:#232329;border:1px solid #33333d;border-radius:9px;'
    + 'padding:9px 11px;font-family:monospace;font-size:11.5px">'
    + esc(url) + '</div>';
}

// ── system settings ────────────────────────────────────────────────────────
function bars(signal){
  const n = Math.max(1, Math.min(4, Math.ceil(signal / 25)));
  return '▂▄▆█'.slice(0, n);
}

function renderSysNet(m){
  const wifiDev = (m.devices || []).find(d => d.type === 'wifi');
  $('sn-badge').textContent = wifiDev && wifiDev.state.startsWith('connected')
    ? t('συνδεδεμένο') : t('εκτός');
  $('sn-conn').textContent = wifiDev ? (wifiDev.connection || '—') : '—';
  $('sn-ips').textContent = (m.ips || []).join(', ') || '—';
  $('sn-ts').textContent = (m.tailscale || []).join(', ') || '—';

  const nets = m.wifi || [];
  $('sn-wifi').innerHTML = nets.length ? nets.map(n => `
    <div class="row" style="justify-content:space-between;padding:5px 0;
      border-bottom:1px solid #232329">
      <span style="${n.active ? 'color:#4ade80;font-weight:600' : ''}">
        ${n.secure ? '🔒' : '🔓'} ${esc(n.ssid)}</span>
      <span style="display:flex;gap:9px;align-items:center">
        <span style="color:#71717a">${bars(n.signal)} ${n.signal}%</span>
        ${n.active ? '' : `<button class="btn sn-join" data-ssid="${esc(n.ssid)}"
           style="font-size:11px;padding:4px 9px">${esc(t('Σύνδεση'))}</button>`}
      </span></div>`).join('')
    : '<span style="color:#71717a">' + esc(t('Πάτα σάρωση.')) + '</span>';

  for(const b of document.querySelectorAll('.sn-join')){
    b.onclick = () => {
      send({type:'sys_wifi_connect', ssid: b.dataset.ssid,
            password: $('sn-pass').value});
      $('sn-pass').value = '';
      snMsg(t('Συνδέομαι…'));
    };
  }

  $('sb-badge').textContent = m.bt_on ? t('ενεργό') : t('ανενεργό');
  const bt = m.bt || [];
  $('sb-list').innerHTML = bt.length ? bt.map(d => `
    <div class="row" style="justify-content:space-between;padding:5px 0;
      border-bottom:1px solid #232329">
      <span style="${d.connected ? 'color:#4ade80;font-weight:600' : ''}">
        ${esc(d.name)}</span>
      <button class="btn sb-act" data-mac="${esc(d.mac)}"
        data-act="${d.connected ? 'disconnect' : 'connect'}"
        style="font-size:11px;padding:4px 9px">
        ${esc(d.connected ? t('Αποσύνδεση') : t('Σύνδεση'))}</button>
    </div>`).join('')
    : '<span style="color:#71717a">' + esc(t('Καμία συσκευή.')) + '</span>';

  for(const b of document.querySelectorAll('.sb-act')){
    b.onclick = () => { send({type:'sys_bt', action:b.dataset.act,
                              mac:b.dataset.mac}); snMsg('…'); };
  }

  if(m.volume !== null && m.volume !== undefined && !volDragging){
    $('sv-vol').value = m.volume;
    $('sv-val').textContent = m.volume + '%';
  }
  if(m.error) snMsg(m.error);
  else if(m.note) snMsg(m.note);
}

function snMsg(text){
  $('sn-msg').textContent = text;
  setTimeout(()=>{ if($('sn-msg').textContent === text)
    $('sn-msg').textContent = ''; }, 8000);
}

// ── echolocation ───────────────────────────────────────────────────────────
function renderEcho(m){
  $('ec-badge').textContent = m.busy ? t('μετράει…')
    : ((m.rooms || []).length + ' ' + t('δωμάτια'));
  const l = m.last;
  $('ec-rt60').textContent = (l && l.rt60) ? l.rt60.toFixed(2) + ' s' : '—';
  $('ec-dist').textContent = (l && l.distance) ? l.distance.toFixed(1) + ' m' : '—';
  const v = m.verdict;
  $('ec-verdict').innerHTML = v
    ? `<span style="color:${v.changed ? '#f59e0b' : '#4ade80'}">${esc(v.why)}</span>`
    : '—';
}

// ── acoustic map ───────────────────────────────────────────────────────────
function renderAcoustic(m){
  const sounds = m.sounds || [];
  $('am-badge').textContent = (m.localized || 0) + ' ' + t('εντοπισμένα');

  const last = m.last;
  $('am-last').innerHTML = last
    ? `<div style="padding:8px 10px;border-radius:9px;background:#232329;
         border:1px solid #2f2f37">🔊 ${esc(last.greek || last.event)}
         ${last.room_el ? '<span style="color:#4ade80"> '
           + esc(last.room_el) + '</span>'
           : '<span style="color:#71717a"> ' + Math.round(last.angle)
             + '°</span>'}</div>`
    : '';

  if(!sounds.length){
    $('am-list').innerHTML = '<span style="color:#71717a">'
      + esc(t('Τίποτα ακόμη.')) + '</span>';
    return;
  }
  $('am-list').innerHTML = sounds.map(s => {
    const spot = s.spots && s.spots[0];
    // No spot yet is the honest state, not a failure — say why.
    const where = spot
      ? `<span style="color:#4ade80">${spot.x.toFixed(1)}, ${spot.y.toFixed(1)} m</span>`
        + (s.room ? ` <span style="color:#71717a">· ${esc(s.room)}</span>` : '')
      : `<span style="color:#71717a">${esc(t('θέλει δεύτερο σημείο'))}</span>`;
    return `<div style="display:flex;justify-content:space-between;gap:10px;
      padding:6px 0;border-bottom:1px solid #232329">
      <span>${esc(s.greek || s.event)}
        <span style="color:#52525b">×${s.heard}</span></span>
      <span style="text-align:right">${where}</span></div>`;
  }).join('');
}

// ── resizable viewers ──────────────────────────────────────────────────────
// The big panels are flex:1 so they fill the pane. Dragging the grip pins an
// explicit height instead; double-tapping it gives the pane back its space.
const VIEWERS = ['map-wrap', 'cam-wrap', 'cost-wrap', 'arm3d', 'cloud-canvas', 'scan3d'];

function loadSizes(){
  try { return JSON.parse(localStorage.getItem('hr_sizes') || '{}'); }
  catch(e){ return {}; }
}
function saveSizes(o){
  try { localStorage.setItem('hr_sizes', JSON.stringify(o)); } catch(e){}
}

function setupViewers(){
  const sizes = loadSizes();
  for (const id of VIEWERS){
    const el = $(id);
    if (!el || el.dataset.gripWired) continue;
    el.dataset.gripWired = '1';

    if (sizes[id]) applyViewerHeight(el, sizes[id]);

    const grip = document.createElement('div');
    grip.className = 'grip';
    grip.title = 'drag / double-tap';
    el.parentNode.insertBefore(grip, el.nextSibling);

    let startY = 0, startH = 0;
    const yOf = e => e.touches ? e.touches[0].clientY : e.clientY;
    const down = e => {
      startY = yOf(e);
      startH = el.getBoundingClientRect().height;
      e.preventDefault();
      window.addEventListener('mousemove', move);
      window.addEventListener('touchmove', move, {passive:false});
      window.addEventListener('mouseup', up);
      window.addEventListener('touchend', up);
    };
    const move = e => {
      const h = Math.max(120, Math.min(1400, startH + (yOf(e) - startY)));
      applyViewerHeight(el, h);
      e.preventDefault();
      redrawVisible();
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('touchmove', move);
      window.removeEventListener('mouseup', up);
      window.removeEventListener('touchend', up);
      const o = loadSizes();
      o[id] = Math.round(el.getBoundingClientRect().height);
      saveSizes(o);
      redrawVisible();
    };
    grip.addEventListener('mousedown', down);
    grip.addEventListener('touchstart', down, {passive:false});
    // Double-tap the grip to hand the space back to the pane.
    grip.addEventListener('dblclick', () => {
      el.style.height = ''; el.style.flex = '';
      const o = loadSizes(); delete o[id]; saveSizes(o);
      redrawVisible();
    });
  }
}

function applyViewerHeight(el, h){
  // flex:none is what stops the pane's flex:1 from immediately overriding it.
  el.style.flex = '0 0 auto';
  el.style.height = h + 'px';
}

// ── costmap − / + ──────────────────────────────────────────────────────────
// The drag grip already resizes it, but it is pointer-only and nothing about a
// thin bar says "drag me". Asked for by name: "στο costmap βάλε συν πλην να το
// ρυθμίζω μόνος μου". Steps by a fixed ratio and remembers, through the same
// hr_sizes store the grip writes, so the two controls cannot disagree.
const COST_MIN = 180, COST_MAX = 1200, COST_STEP = 1.18;

function costResize(factor){
  const el = $('cost-wrap'); if (!el) return;
  const now = el.getBoundingClientRect().height;
  const h = Math.round(Math.max(COST_MIN, Math.min(COST_MAX, now * factor)));
  applyViewerHeight(el, h);
  const o = loadSizes(); o['cost-wrap'] = h; saveSizes(o);
  costShowSize();
  drawCost();
}

function costShowSize(){
  const el = $('cost-wrap'), out = $('cost-size');
  if (!el || !out) return;
  out.textContent = Math.round(el.getBoundingClientRect().height) + ' px';
}

// ── foldable cards ─────────────────────────────────────────────────────────
// Every card folds from its header, and the state survives a reload. On a
// phone the panes stack several cards deep and the useful one is often third;
// folding beats scrolling. Cards holding a canvas or an image also get a
// native drag handle, which is pointer-only — hence both mechanisms.
function cardKey(card){
  const pane = card.closest('.pane');
  const idx = [...(pane ? pane.querySelectorAll('.card') : [])].indexOf(card);
  return (pane ? pane.id : '?') + ':' + idx;
}

function loadFolded(){
  try { return new Set(JSON.parse(localStorage.getItem('hr_folded') || '[]')); }
  catch(e){ return new Set(); }
}

function saveFolded(set){
  try { localStorage.setItem('hr_folded', JSON.stringify([...set])); } catch(e){}
}

function setupCards(){
  const folded = loadFolded();
  for (const card of document.querySelectorAll('.pane .card')){
    const h = card.querySelector(':scope > h3');
    if (!h || h.dataset.foldWired) continue;
    h.dataset.foldWired = '1';
    if (card.querySelector('canvas, img')) card.classList.add('sizable');
    if (folded.has(cardKey(card))) card.classList.add('collapsed');
    h.addEventListener('click', e => {
      // ‼️ Some headers carry their own controls — the map's "Χρώματα"
      // checkbox lives inside its h3. Clicking those must not fold the card.
      if (e.target.closest('input,label,select,button,a')) return;
      card.classList.toggle('collapsed');
      const set = loadFolded();
      card.classList.contains('collapsed') ? set.add(cardKey(card))
                                           : set.delete(cardKey(card));
      saveFolded(set);
      if (!card.classList.contains('collapsed')) redrawVisible();
    });
  }
}

// A canvas inside a folded card has no size; on unfold it needs repainting.
function redrawVisible(){
  for (const f of [window.draw, window.armDraw, window.drawCompass2,
                   window.drawPointRing, window.drawCost, window.cloudDraw]){
    if (typeof f === 'function') { try { f(); } catch(e){} }
  }
  window.dispatchEvent(new Event('resize'));
}

// ── people ─────────────────────────────────────────────────────────────────
function renderPeople(m){
  const here = m.here;
  $('pp-badge').textContent = here || t('άγνωστος');
  $('pp-seeing').textContent  = (m.seeing && m.seeing.length)
    ? m.seeing.join(', ') : '—';
  $('pp-hearing').textContent = m.hearing || '—';
  $('pp-height').textContent  = m.measured_height
    ? m.measured_height.toFixed(2) + ' m' : '—';
  $('pp-reason').textContent  = m.reason
    ? m.reason + (m.confidence ? ' · ' + Math.round(m.confidence*100) + '%' : '')
    : '—';

  const list = m.people || [];
  $('pp-count').textContent = list.length + ' ' + t('άτομα');
  if(!list.length){
    $('pp-list').innerHTML = '<span style="color:#71717a">'
      + esc(t('Κανένα άτομο ακόμη. Γράψε ένα όνομα παραπάνω.')) + '</span>';
    return;
  }
  const tick = ok => ok ? '<span style="color:#4ade80">✓</span>'
                        : '<span style="color:#52525b">—</span>';
  $('pp-list').innerHTML = list.map(p => `
    <div style="padding:10px 11px;margin-bottom:8px;border-radius:11px;
      background:${p.name===here?'#16281c':'#232329'};
      border:1px solid ${p.name===here?'#2f6b41':'#2f2f37'}">
      <div class="row" style="justify-content:space-between">
        <span style="font-size:13.5px;font-weight:600;
          ${p.name===here?'color:#7ee2a0':''}">${esc(p.name)}</span>
        <span style="font-size:11px;color:#71717a">
          ${p.complete ? esc(t('πλήρες')) : esc(t('ελλιπές'))}</span>
      </div>
      <div class="grid2" style="margin-top:7px;font-size:12px">
        <span class="k">${esc(t('Πρόσωπο'))}</span><span>${tick(p.face)}</span>
        <span class="k">${esc(t('Φωνή'))}</span><span>${tick(p.voice)}</span>
        <span class="k">${esc(t('Ύψος'))}</span><span>${
          p.height ? p.height.toFixed(2)+' m <span style="color:#52525b">('
                     + p.height_samples + ')</span>'
                   : '<span style="color:#52525b">—</span>'}</span>
      </div>
      <div class="row" style="margin-top:9px">
        <button class="btn pp-face" data-name="${esc(p.name)}"
          style="font-size:11.5px;padding:6px 10px">📷 ${esc(t('Μάθε πρόσωπο'))}</button>
        <button class="btn pp-voice" data-name="${esc(p.name)}"
          style="font-size:11.5px;padding:6px 10px">🎤 ${esc(t('Μάθε φωνή'))}</button>
        <button class="btn pp-del" data-name="${esc(p.name)}"
          style="font-size:11.5px;padding:6px 10px">🗑</button>
      </div>
    </div>`).join('');

  const wire = (cls, fn) => {
    for(const b of document.querySelectorAll('.'+cls)) b.onclick = () => fn(b.dataset.name);
  };
  wire('pp-face',  n => { send({type:'people_enrol', name:n, what:'face'});
    ppMsg(t('Κοίτα την κάμερα: ') + n); });
  wire('pp-voice', n => { send({type:'people_enrol', name:n, what:'voice'});
    ppMsg(t('Μίλα τώρα: ') + n); });
  wire('pp-del',   n => { if(confirm(t('Διαγραφή; ') + n))
    send({type:'people_remove', name:n}); });
}

function ppMsg(text){
  $('pp-msg').textContent = text;
  setTimeout(()=>{ $('pp-msg').textContent=''; }, 6000);
}

// ── gesture bindings editor ────────────────────────────────────────────────
let handState = null;
// Actions that only ever reduce what the robot is doing. Mirrors
// gesture_bindings._SAFE_ACTIONS; the server is still the authority.
const SAFE_ACTIONS = ['none','stop','estop','stop_follow','cancel',
                      'say_hello','listen'];

// A sticker per gesture. The emoji does more than the words: you recognise
// ✌️ instantly and read "δύο δάχτυλα" second, which is the right order when
// you are standing in front of a camera trying to remember what to do.
const GESTURE_ICONS = {
  point_floor:'👇', hand_up:'✋', both_hands_up:'🙌', wave:'👋',
  arms_crossed:'🙅', t_pose:'🧍',
  fist:'✊', point:'☝️', victory:'✌️', three:'3️⃣', open_palm:'🖐️',
  thumbs_up:'👍', thumbs_down:'👎', ok:'👌',
};
// Body poses first, then fingers — the order people will read them in.
const GESTURE_ORDER = ['point_floor','hand_up','both_hands_up','wave',
                       'arms_crossed','t_pose',
                       'fist','point','victory','three','open_palm',
                       'thumbs_up','thumbs_down','ok'];

function gestureLabels(){
  // Both nodes ship their own table; merge whichever are running.
  const a = (gestureState && gestureState.vocab && gestureState.vocab.gesture_labels) || {};
  const b = (handState && handState.gesture_labels) || {};
  return Object.assign({}, a, b);
}

// ‼️ 2026-08-04: "πάω να πατήσω μια χειρονομία να επιλέξω τι θα κάνει και
// τρεμοπαίζει και δεν μπορώ να επιλέξω". /gesture_status arrives at 15 Hz —
// it carries the live hold progress — and this function used to rebuild
// gb-list.innerHTML on every one of them. That destroys and recreates the
// <select> elements fifteen times a second, so an open dropdown is closed
// before you can move the mouse to an option, and the whole list strobes.
//
// The rows are now built ONCE and only mutated afterwards. gbKey is what the
// structure depends on (which gestures exist, which actions are offered); the
// 15 Hz part — highlight, progress bar — only touches style and width.
let gbKey = null;

function renderGestureBindings(){
  const body = (gestureState && gestureState.vocab) || null;
  const hand = handState || null;
  const src = body || hand;
  if(!src || !src.bindings) return;

  const labels = src.action_labels || {};
  const names = gestureLabels();
  const motion = !!src.motion_enabled;
  $('gb-motion').checked = motion;
  $('gb-badge').textContent = motion ? t('κίνηση ΕΝΕΡΓΗ') : t('μόνο ασφαλείς');

  const holding = (body && body.holding) || (hand && hand.holding) || null;
  const progress = Math.max((body && body.progress) || 0,
                            (hand && hand.progress) || 0);

  // Keep the declared order, then anything the nodes added that we do not
  // know about — a new gesture must still appear rather than vanish.
  const keys = Object.keys(src.bindings);
  const rows = GESTURE_ORDER.filter(g => keys.indexOf(g) >= 0)
                 .concat(keys.filter(g => GESTURE_ORDER.indexOf(g) < 0).sort());
  const bodySet = ['point_floor','hand_up','both_hands_up','wave',
                   'arms_crossed','t_pose'];

  const key = JSON.stringify([rows, Object.keys(labels)]);
  if(key !== gbKey){
    gbKey = key;
    gbBuild(rows, labels, names, bodySet);
  }
  gbUpdate(rows, src, holding, progress);
}

function gbBuild(rows, labels, names, bodySet){
  const opts = Object.keys(labels).map(a =>
    `<option value="${esc(a)}">${esc(labels[a])}</option>`).join('');
  $('gb-list').innerHTML = rows.map(g => `
    <div class="gb-row" data-gesture="${esc(g)}"
      style="display:flex;align-items:center;gap:11px;padding:9px 10px;
      margin-bottom:7px;border-radius:11px">
      <div style="font-size:26px;line-height:1;flex:0 0 34px;text-align:center"
        title="${esc(g)}">${GESTURE_ICONS[g]||'✋'}</div>
      <div style="flex:1;min-width:0">
        <div class="gb-name" style="font-size:12.5px">${esc(names[g] || g)}</div>
        <div style="font-size:10.5px;color:#71717a;margin-top:1px">
          ${bodySet.indexOf(g)>=0 ? esc(t('στάση σώματος')) : esc(t('δάχτυλα'))}
          <span class="gb-risk"></span>
        </div>
        <div class="gb-bar" style="height:3px;border-radius:2px;background:#2c2c34;
          margin-top:6px;display:none">
          <div style="height:3px;border-radius:2px;background:#4ade80;width:0%"></div>
        </div>
      </div>
      <select data-gesture="${esc(g)}" class="btn gb-sel"
        style="flex:0 0 148px;padding:6px 8px;font-size:12px">${opts}</select>
    </div>`).join('');

  for(const sel of document.querySelectorAll('.gb-sel')){
    sel.onchange = () => {
      send({type:'gesture_bind', gesture: sel.dataset.gesture, action: sel.value});
      $('gb-msg').textContent = t('Αποθηκεύτηκε.');
      setTimeout(()=>{ $('gb-msg').textContent=''; }, 2500);
    };
  }
}

function gbUpdate(rows, src, holding, progress){
  for(const row of $('gb-list').children){
    const g = row.dataset.gesture;
    const cur = src.bindings[g];
    const live = (g === holding);
    const risky = SAFE_ACTIONS.indexOf(cur) < 0;

    row.style.background = live ? '#16281c' : '#232329';
    row.style.border = '1px solid ' + (live ? '#2f6b41' : '#2f2f37');
    const name = row.querySelector('.gb-name');
    name.style.color = live ? '#7ee2a0' : '#d4d4d8';
    name.style.fontWeight = live ? '600' : '';
    row.querySelector('.gb-risk').innerHTML =
      risky ? ' · <span style="color:#f59e0b">' + esc(t('κινεί')) + '</span>' : '';

    const bar = row.querySelector('.gb-bar');
    bar.style.display = live ? '' : 'none';
    if(live) bar.firstElementChild.style.width = Math.round(progress*100) + '%';

    // ‼️ Never while the user is in it. Writing .value on a focused <select>
    // is what closes an open dropdown — the same bug in miniature, and the
    // reason a rebuilt list could not be used at all.
    const sel = row.querySelector('.gb-sel');
    if(document.activeElement !== sel && sel.value !== cur) sel.value = cur;
    sel.style.borderColor = risky ? '#7c5310' : '';
  }
}

// ── self-diagnosis ─────────────────────────────────────────────────────────
const DG_COLOR = {critical:'#ef4444', warning:'#f59e0b', info:'#71717a'};

function renderDiagnostics(m){
  const f = m.findings || [];
  const crit = f.filter(x=>x.severity==='critical').length;
  $('dg-badge').textContent = !f.length ? t('τίποτα γνωστό')
    : (crit ? crit + ' ' + t('σοβαρά') : f.length + ' ' + t('προειδοποιήσεις'));
  const el = $('dg-list');
  if(!f.length){
    el.innerHTML = '<span style="color:#4ade80">'
      + esc(t('Κανένα γνωστό πρόβλημα.')) + '</span>';
    return;
  }
  el.innerHTML = f.map(x=>`
    <div style="padding:8px 0;border-bottom:1px solid #232329">
      <div style="color:${DG_COLOR[x.severity]||'#a1a1aa'};font-weight:600">
        ${esc(x.title)}</div>
      <div style="color:#a1a1aa;margin-top:3px">${esc(x.detail)}</div>
      <div style="color:#60a5fa;margin-top:4px">→ ${esc(x.fix)}</div>
    </div>`).join('');
}

// ── live microphone ────────────────────────────────────────────────────────
// Raw 16 kHz int16 PCM arrives as BINARY websocket frames and is scheduled
// straight into an AudioContext. No codec, no <audio> element: the stream is
// endless and 100 ms chunks, which media elements handle badly.
const MIC_RATE = 16000;
let audioCtx = null, playHead = 0, listening = false, chunkCount = 0;

function startListening(){
  // ‼️ The AudioContext MUST be created inside the click handler. Browsers
  // (Safari especially) refuse to start audio without a user gesture, and a
  // context created at page load comes up 'suspended' and stays silent.
  if(!audioCtx){
    const AC = window.AudioContext || window.webkitAudioContext;
    if(!AC){ $('listen-msg').textContent = t('Χωρίς υποστήριξη ήχου.'); return; }
    // ‼️ NO forced sampleRate. `new AC({sampleRate:16000})` is accepted and
    // then sits 'suspended' on iOS. The default-rate context resumes, and
    // createBuffer(..., MIC_RATE) still declares 16 kHz, so the browser
    // resamples on playback — which is what we wanted anyway.
    audioCtx = new AC();
    // ‼️ iOS needs an actual sound started inside the gesture before the
    // context will run. One silent sample is enough to unlock it; without
    // this the context reports 'running' but stays mute.
    const unlock = audioCtx.createBufferSource();
    unlock.buffer = audioCtx.createBuffer(1, 1, audioCtx.sampleRate);
    unlock.connect(audioCtx.destination);
    unlock.start(0);
  }
  playHead = 0;
  chunkCount = 0;
  listening = true;
  send({type:'listen', on:true});
  // resume() is a promise: report what actually happened rather than assuming.
  audioCtx.resume().then(updateListenMsg, updateListenMsg);
  updateListenMsg();
}

function updateListenMsg(){
  if(!listening){ $('listen-msg').textContent = ''; return; }
  const state = audioCtx ? audioCtx.state : '—';
  // Showing the state and the chunk count separates the three failure modes
  // that all present as "no sound": no data arriving, data arriving but the
  // context suspended, and everything fine but the phone muted.
  $('listen-msg').textContent =
    t('Ακούς ζωντανά.') + ' [' + state + ' · ' + chunkCount + ']';
}

function stopListening(){
  listening = false;
  send({type:'listen', on:false});
  $('listen-msg').textContent = '';
  if(audioCtx) audioCtx.suspend();
}

function playPcmChunk(buf){
  if(!listening || !audioCtx) return;
  const pcm = new Int16Array(buf);
  if(!pcm.length) return;
  chunkCount++;
  if(chunkCount % 10 === 1) updateListenMsg();
  const frame = audioCtx.createBuffer(1, pcm.length, MIC_RATE);
  const out = frame.getChannelData(0);
  for(let i=0;i<pcm.length;i++) out[i] = pcm[i] / 32768;
  const src = audioCtx.createBufferSource();
  src.buffer = frame; src.connect(audioCtx.destination);
  // Keep a running playhead so consecutive chunks butt up against each other.
  // Starting each chunk at currentTime would leave audible gaps whenever the
  // network jittered. If we fall behind, jump forward rather than accumulate
  // an ever-growing delay.
  const now = audioCtx.currentTime;
  if(playHead < now + 0.05) playHead = now + 0.05;
  src.start(playHead);
  playHead += frame.duration;
}

// ── map-referenced compass ─────────────────────────────────────────────────
// Heading comes from AMCL, not from a magnetometer: the firmware disables the
// magnetometer-fused rotation vector on purpose (the Roomba's motors wrecked it
// indoors and AMCL lost convergence). The map never rotates, so a single
// calibrated offset turns AMCL's yaw into a true bearing.
let compassOffset = null;          // map yaw that faces north, radians

const CARDINALS = ['Β','ΒΑ','Α','ΝΑ','Ν','ΝΔ','Δ','ΒΔ'];

function drawCompass2(){
  const c=$('cp-rose'); if(!c) return;
  const g=c.getContext('2d'), S=c.width, R=S/2-16;
  g.clearRect(0,0,S,S);
  g.beginPath(); g.arc(S/2,S/2,R,0,Math.PI*2);
  g.strokeStyle='#2c2c32'; g.lineWidth=2; g.stroke();

  const haveYaw = pose && typeof pose.yaw === 'number';
  const ready = haveYaw && compassOffset !== null;

  $('cp-off').textContent = compassOffset===null ? '—'
    : (compassOffset*180/Math.PI).toFixed(0)+'°';

  if(!ready){
    $('cp-badge').textContent = compassOffset===null
      ? t('αβαθμονόμητη') : t('χωρίς εντοπισμό');
    $('cp-card').textContent='—'; $('cp-deg').textContent='—';
    g.fillStyle='#52525b'; g.font='600 13px system-ui';
    g.textAlign='center'; g.textBaseline='middle';
    g.fillText('—', S/2, S/2);
    return;
  }

  // ‼️ Same sign rule as home_robot/compass.py: ROS yaw grows counter-
  // clockwise, compass bearings grow clockwise, so this subtracts.
  let bearing = ((compassOffset - pose.yaw)*180/Math.PI) % 360;
  if(bearing < 0) bearing += 360;
  const card = CARDINALS[Math.round(bearing/45) % 8];
  $('cp-badge').textContent = t('βαθμονομημένη');
  $('cp-card').textContent = t(card);
  // Round BEFORE wrapping: 359.98 rounds to 360, which must read 0°.
  $('cp-deg').textContent = (Math.round(bearing) % 360)+'°';

  // The rose turns under a fixed robot: north sits at -bearing from up.
  for(let i=0;i<8;i++){
    const ang = (i*45 - bearing)*Math.PI/180 - Math.PI/2;
    const major = (i%2)===0;
    const rr = R - (major?2:8);
    const x=S/2+Math.cos(ang)*rr, y=S/2+Math.sin(ang)*rr;
    g.fillStyle = i===0 ? '#ef4444' : (i===4 ? '#60a5fa' : '#71717a');
    g.font = (major?'600 13px':'400 10px')+' system-ui';
    g.textAlign='center'; g.textBaseline='middle';
    g.fillText(t(CARDINALS[i]), x, y);
  }
  // The robot always points up — it is the world that rotates around it.
  g.beginPath();
  g.moveTo(S/2, S/2-R+26); g.lineTo(S/2-9, S/2+14); g.lineTo(S/2+9, S/2+14);
  g.closePath(); g.fillStyle='#e4e4e7'; g.fill();
}

// ── pointing gestures ──────────────────────────────────────────────────────
let gestureState = null;

function drawPointRing(){
  const m = gestureState || {};
  const c = $('pt-ring'); if(!c) return;
  const g = c.getContext('2d');
  const S = c.width, R = S/2 - 10;
  g.clearRect(0,0,S,S);

  // Track.
  g.beginPath(); g.arc(S/2,S/2,R,0,Math.PI*2);
  g.strokeStyle='#2c2c32'; g.lineWidth=9; g.stroke();

  const locked = !!(m.confirmed);
  const prog = locked ? 1 : (m.progress || 0);
  if(prog > 0){
    g.beginPath();
    g.arc(S/2,S/2,R,-Math.PI/2,-Math.PI/2 + prog*Math.PI*2);
    // Amber while gathering agreeing frames, green once it locks — the same
    // colours the RViz marker uses, so the two views read alike.
    g.strokeStyle = locked ? '#22c55e' : '#f59e0b';
    g.lineWidth=9; g.lineCap='round'; g.stroke();
  }
  g.fillStyle = locked ? '#22c55e' : (m.pointing ? '#f59e0b' : '#52525b');
  g.font='600 15px system-ui'; g.textAlign='center'; g.textBaseline='middle';
  g.fillText(locked ? '✓' : (m.pointing ? Math.round(prog*100)+'%' : '—'), S/2, S/2);

  const pt = m.confirmed || m.candidate || m.last || null;
  $('pt-x').textContent = pt ? pt.x.toFixed(2)+' m' : '—';
  $('pt-y').textContent = pt ? pt.y.toFixed(2)+' m' : '—';
  $('pt-side').textContent = m.side ? (m.side==='right'?t('δεξί'):t('αριστερό')) : '—';
  $('pt-straight').textContent = m.straightness!=null ? Math.round(m.straightness)+'°' : '—';

  const b=$('pt-badge');
  b.textContent = locked ? t('κλείδωσε') : (m.pointing ? t('δείχνεις…') : t('αδρανές'));
  // The button is only meaningful once a point has been confirmed at least
  // once — `last` survives after the arm drops, which is what it acts on.
  $('b-pt-go').disabled = !m.last;
}

// ── open-vocabulary search ─────────────────────────────────────────────────
function renderVocab(m){
  const active = !!m.active;
  $('vc-badge').textContent = m.error ? t('άγνωστο')
    : (!m.ready ? t('φορτώνει…') : (active ? t('ψάχνει') : t('αδρανές')));
  $('vc-msg').textContent = m.error ? m.error
    : (active ? t('Ψάχνω: ') + (m.greek||m.vocabulary||[]).join(', ') : '');

  const hits = m.hits || [];
  $('vc-count').textContent = active ? hits.length + ' ' + t('ευρήματα') : '—';
  const el = $('vc-hits');
  if(!active){
    el.innerHTML = '<span style="color:#71717a">'+esc(t('Δεν ψάχνει τίποτα.'))+'</span>';
    return;
  }
  if(!hits.length){
    el.innerHTML = '<span style="color:#71717a">'+esc(t('Δεν το βλέπω.'))+'</span>';
    return;
  }
  el.innerHTML = hits.map(h=>{
    // Distance is null whenever depth was unavailable — show a dash rather
    // than 0 m, which would read as "right at the camera".
    const d = (h.z==null) ? '—' : h.z.toFixed(2)+' m';
    return `<div style="padding:6px 0;border-bottom:1px solid #232329">
      <span style="color:#4ade80">${esc(h.label)}</span>
      <span style="color:#71717a"> ${(h.conf*100).toFixed(0)}% · ${esc(d)}</span></div>`;
  }).join('');
}

// ── sound events ───────────────────────────────────────────────────────────
let soundState = null;

function renderSound(m){
  $('sd-badge').textContent = m.listening ? t('ακούει') : t('σε παύση');
  $('sd-bearing').textContent = m.bearing || '—';
  $('sd-angle').textContent   = (m.angle==null) ? '—' : Math.round(m.angle)+'°';
  $('sd-speech').textContent  = m.speech ? (m.speech*100).toFixed(0)+'%' : '—';
  $('sd-windows').textContent = m.windows==null ? '—' : m.windows;

  const cands = m.candidates || [];
  $('sd-cands').textContent = cands.length
    ? cands.map(c=>`${c.greek} ${(c.score*100).toFixed(0)}%`).join(' · ')
    : t('ησυχία');

  const feed = (m.feed||[]).slice().reverse();
  const el = $('sd-feed');
  if(!feed.length){
    el.innerHTML = '<span style="color:#71717a">'+esc(t('Τίποτα ακόμη.'))+'</span>';
  } else {
    el.innerHTML = feed.map(e=>{
      const tm = new Date((e.ts||0)*1000).toLocaleTimeString('el-GR',
        {hour:'2-digit',minute:'2-digit',second:'2-digit'});
      return `<div style="padding:6px 0;border-bottom:1px solid #232329">
        <span style="color:#52525b;font-variant-numeric:tabular-nums">${esc(tm)}</span>
        &nbsp;${esc(e.text||'')}</div>`;
    }).join('');
  }
  drawCompass(m.angle);
}

function drawCompass(angle){
  const c=$('sd-compass'); if(!c) return;
  const g=c.getContext('2d'), S=c.width, R=S/2-10;
  g.clearRect(0,0,S,S);
  g.beginPath(); g.arc(S/2,S/2,R,0,Math.PI*2);
  g.strokeStyle='#2c2c32'; g.lineWidth=2; g.stroke();
  // Nose of the robot, so the wedge is read relative to something.
  g.beginPath(); g.moveTo(S/2,S/2-R); g.lineTo(S/2,S/2-R+9);
  g.strokeStyle='#52525b'; g.lineWidth=3; g.stroke();
  if(angle==null){
    g.fillStyle='#52525b'; g.font='600 12px system-ui';
    g.textAlign='center'; g.textBaseline='middle';
    g.fillText('—', S/2, S/2);
    return;
  }
  // The XVF3800 grows counter-clockwise from straight ahead; canvas angles run
  // clockwise from +x, hence the negation and the quarter-turn offset.
  const rad = -angle*Math.PI/180 - Math.PI/2;
  g.beginPath(); g.moveTo(S/2,S/2);
  g.arc(S/2,S/2,R-3, rad-0.35, rad+0.35); g.closePath();
  g.fillStyle='rgba(96,165,250,.55)'; g.fill();
}

// ── proactive observations ─────────────────────────────────────────────────
function renderObservations(m){
  const feed = (m.feed||[]).slice().reverse();
  $('ob-badge').textContent = (m.learned!=null)
    ? m.learned+' '+t('γνωστά αντικείμενα') : '—';
  const el = $('ob-feed');
  if(!feed.length){
    el.innerHTML = '<span style="color:#71717a">'+esc(t('Καμία παρατήρηση ακόμη.'))+'</span>';
    return;
  }
  el.innerHTML = feed.map(o=>{
    const t = new Date((o.ts||0)*1000).toLocaleTimeString('el-GR',
      {hour:'2-digit',minute:'2-digit'});
    return `<div style="padding:7px 0;border-bottom:1px solid #232329">
      <span style="color:#52525b;font-variant-numeric:tabular-nums">${t}</span>
      &nbsp;${esc(o.text||'')}</div>`;
  }).join('');
}

// ── object memory ────────────────────────────────────────────────────────
function renderObjectMemory(m){
  const items = m.items || [];
  $('om-badge').textContent = items.length
    ? items.length + ' ' + t('αντικείμενα') : '—';
  const el = $('om-list');
  if(!items.length){
    el.innerHTML = '<span style="color:#71717a">'
      + esc(t('Τίποτα γνωστό ακόμη.')) + '</span>';
    return;
  }
  el.innerHTML = items.map(o=>{
    const when = o.last_seen ? new Date(o.last_seen*1000)
      .toLocaleTimeString('el-GR', {hour:'2-digit', minute:'2-digit'}) : '';
    const room = o.room_el ? `<span style="color:#4ade80">${esc(o.room_el)}</span>`
      : `<span style="color:#71717a">${esc(t('άγνωστο δωμάτιο'))}</span>`;
    return `<div style="padding:7px 0;border-bottom:1px solid #232329;
      display:flex;justify-content:space-between;gap:10px">
      <span><span style="font-weight:600">${esc(o.label||'')}</span>
        &nbsp;${room}</span>
      <span style="color:#52525b;font-variant-numeric:tabular-nums">${when}</span>
      </div>`;
  }).join('');
}

// ── episodic timeline ──────────────────────────────────────────────────────
const TL_ICON = {heard:'🗣️', said:'🤖', room:'🚪', observed:'💡',
                 saw:'👤', mission:'🎯'};

function renderTimeline(m){
  $('tl-count').textContent = (m.count!=null) ? m.count+' '+t('γεγονότα') : '—';
  const ev = (m.events||[]).slice().reverse();
  const el = $('tl-feed');
  if(!ev.length){
    el.innerHTML='<span style="color:#71717a">'+esc(t('Άδειο.'))+'</span>'; return;
  }
  el.innerHTML = ev.map(e=>{
    const who = e.who ? `<span style="color:#a78bfa">${esc(e.who)}</span> ` : '';
    return `<div style="padding:6px 0;border-bottom:1px solid #232329">
      <span style="color:#52525b;font-variant-numeric:tabular-nums">${esc(e.clock||'')}</span>
      &nbsp;${TL_ICON[e.kind]||'•'}&nbsp;${who}${esc(e.text||'')}</div>`;
  }).join('');
}

function connect(){
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws${TOKEN_QS}`);
  // Re-arm the cloud stream on reconnect: the node tracks viewers per socket,
  // so after a restart (or a map switch) it has forgotten this tab is open.
  ws.onopen  = ()=>{ $('dot').classList.add('on');
                     if(cloudOn) send({type:'cloud', on:true});
                     // Same reason as the cloud: the node tracks viewers per
                     // socket, so after a restart it has forgotten this tab.
                     if(costOn) send({type:'costmap', on:true});
                     if(fuseOn) fuseSend();
                     if(!overlayOn) send({type:'overlay', on:false}); };
  ws.onclose = ()=>{ $('dot').classList.remove('on');
                     // The server forgets listeners per socket, so a dropped
                     // connection silently stops the audio — reflect that.
                     if(listening) stopListening();
                     setTimeout(connect,2000); };
  ws.binaryType = 'arraybuffer';
  ws.onmessage = e=>{
    // Audio and camera frames both arrive as binary; everything else is JSON.
    // A JPEG always starts with the FFD8 SOI marker, which is how the two
    // binary payloads on this one channel are told apart — see the comment
    // in _cb_camera (web_dashboard_node.py) for why the camera rides this
    // channel at all instead of its own HTTP stream.
    if(e.data instanceof ArrayBuffer){
      const b = new Uint8Array(e.data);
      if(b.length >= 2 && b[0] === 0xFF && b[1] === 0xD8) camShowFrame(b);
      else playPcmChunk(e.data);
      return;
    }
    const m=JSON.parse(e.data);
    const h=HANDLERS[m.type];
    if(h) h(m);
  };
}
function send(o){ if(ws&&ws.readyState===1) ws.send(JSON.stringify(o)); }

// ── map click ──────────────────────────────────────────────────────────────
// Click-modes share the canvas with plain navigation, so they are mutually
// exclusive checkboxes: checking one un-checks the others, rather than
// layering meanings onto the same click with no visual cue.
const CLICK_MODE_BOXES = ['b-pick-room', 'b-place-room', 'b-place-room-rect', 'b-kz-add'];
function syncClickModeRows(){
  $('place-room-row').style.display = $('b-place-room').checked ? '' : 'none';
  $('place-room-rect-row').style.display = $('b-place-room-rect').checked ? '' : 'none';
  $('kz-add-row').style.display = $('b-kz-add').checked ? '' : 'none';
  if(!$('b-kz-add').checked) kzCorner = null;             // abandon a half-drawn zone
  if(!$('b-place-room-rect').checked) prrCorner = null;   // abandon a half-drawn room rect
}
CLICK_MODE_BOXES.forEach(id => $(id).addEventListener('change', () => {
  if($(id).checked) CLICK_MODE_BOXES.filter(o => o !== id).forEach(o => $(o).checked = false);
  syncClickModeRows();
}));
let kzCorner = null;    // first click of a 2-click keepout rectangle, or null
let prrCorner = null;   // first click of a 2-click room rectangle, or null
canvas.addEventListener('click',e=>{
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*canvas.width/r.width;
  const cy=(e.clientY-r.top)*canvas.height/r.height;
  const wp=c2w(cx,cy); if(!wp) return;
  if($('b-kz-add').checked){
    if(!kzCorner){ kzCorner = wp; draw(); return; }
    const name = $('kz-name').value.trim() || ('zone_' + (Object.keys(kzData).length + 1));
    send({type:'add_keepout_zone', x1:kzCorner.x, y1:kzCorner.y, x2:wp.x, y2:wp.y, name});
    kzCorner = null;
    $('kz-msg').textContent = t('Προσθήκη…'); $('kz-msg').style.color = '#71717a';
    return;
  }
  if($('b-place-room-rect').checked){
    if(!prrCorner){ prrCorner = wp; draw(); return; }
    const name = $('prr-name').value.trim();
    if(!name){
      $('room-edit-msg').textContent = t('Δώσε πρώτα όνομα δωματίου.');
      $('room-edit-msg').style.color = '#f87171';
      prrCorner = null; draw();
      return;
    }
    const hex = $('prr-color').value;
    send({type:'place_room_rect', x1:prrCorner.x, y1:prrCorner.y, x2:wp.x, y2:wp.y, name,
          color:[parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)]});
    prrCorner = null;
    $('room-edit-msg').textContent = t('Τοποθέτηση…');
    $('room-edit-msg').style.color = '#71717a';
    return;
  }
  if($('b-place-room').checked){
    const name = $('pr-name').value.trim();
    if(!name){
      $('room-edit-msg').textContent = t('Δώσε πρώτα όνομα δωματίου.');
      $('room-edit-msg').style.color = '#f87171';
      return;
    }
    const hex = $('pr-color').value;
    send({type:'place_room', x:wp.x, y:wp.y, name,
          color:[parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)]});
    $('room-edit-msg').textContent = t('Τοποθέτηση…');
    $('room-edit-msg').style.color = '#71717a';
    return;
  }
  if($('b-pick-room').checked){
    send({type:'pick_room',x:wp.x,y:wp.y});
    return;
  }
  goal=wp; send({type:'nav_goal',x:wp.x,y:wp.y});
  draw();
});

// ── room buttons ───────────────────────────────────────────────────────────
const rdiv=$('rooms');
ROOMS.forEach(name=>{
  const b=document.createElement('button');
  b.className='btn'; b.textContent=name;
  b.onclick=()=>{ send({type:'goto_room',room:name}); goal=null; draw(); };
  rdiv.appendChild(b);
});

// ── room colours ───────────────────────────────────────────────────────────
// The swatches come from the same room_colors.yaml the server tints with, so
// the legend cannot drift from the picture. Greek names are data (they come
// from the map), so they are not routed through t().
function roomLegend(rooms){
  const el = $('room-legend');
  const names = Object.keys(rooms);
  if (!names.length){ el.innerHTML = ''; return; }
  el.innerHTML = names.sort().map(n => {
    const c = rooms[n];
    return '<span style="display:inline-flex;align-items:center;gap:5px;' +
           'font-size:11.5px;color:#a1a1aa">' +
           '<i style="width:11px;height:11px;border-radius:3px;display:inline-block;' +
           'background:rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></i>' + n + '</span>';
  }).join('');
}
// ── room name/colour editor ─────────────────────────────────────────────────
// Edits maps/room_colors.yaml (+ repaints room_mask.png server-side) so a
// remap's room1/room2 placeholders can be named and coloured from the phone
// instead of SSH-editing a YAML file. Greek names are data, not routed
// through t() — same reasoning as roomLegend above.
function renderRoomEditor(rooms){
  const el = $('room-edit');
  const names = Object.keys(rooms).sort();
  if (!names.length){ el.innerHTML = ''; $('room-edit-row').style.display='none'; return; }
  $('room-edit-row').style.display='';
  el.innerHTML = names.map(n => {
    const c = rooms[n];
    const hex = '#' + c.map(v => Math.max(0, Math.min(255, v|0))
                              .toString(16).padStart(2, '0')).join('');
    return '<div class="row" style="gap:8px;margin-top:6px" data-room="' + esc(n) + '">' +
      '<input type="color" class="re-color" value="' + hex + '" ' +
        'style="width:34px;height:30px;padding:0;border:none;background:none;flex:0 0 auto">' +
      '<input type="text" class="re-name" value="' + esc(n) + '" ' +
        'style="flex:1;min-width:100px;background:#232329;border:1px solid #2c2c32;' +
        'border-radius:8px;color:#e4e4e7;padding:6px 9px;font-size:12.5px">' +
      '</div>';
  }).join('');
}
$('b-room-save').onclick = () => {
  const rows = $('room-edit').querySelectorAll('[data-room]');
  const rooms = [];
  rows.forEach(row => {
    const old = row.dataset.room;
    const name = row.querySelector('.re-name').value.trim();
    const hex = row.querySelector('.re-color').value;
    if (!name) return;
    rooms.push({old, name, color:[
      parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)]});
  });
  if (!rooms.length) return;
  send({type:'save_rooms', rooms});
  $('room-edit-msg').textContent = t('Αποθήκευση…');
  $('room-edit-msg').style.color = '#71717a';
};
$('b-tint').onchange = e => send({type:'room_tint', on: e.target.checked});
$('b-slipmap').onchange = e => { slipMapOn = e.target.checked; draw(); };

// ── keepout zones ────────────────────────────────────────────────────────
// Names are data (typed by the user, not routed through t()) — same
// reasoning as roomLegend/renderRoomEditor above.
function renderKeepoutList(zones){
  const el = $('kz-list');
  const names = Object.keys(zones).sort();
  if (!names.length){ el.innerHTML = ''; return; }
  el.innerHTML = names.map(n => {
    const z = zones[n];
    const dims = z.shape === 'circle' ? `r=${z.radius}m` : `${z.width}×${z.height}m`;
    return '<div class="row" style="justify-content:space-between;font-size:11.5px">' +
      '<span>🚫 ' + esc(n) + ' <span style="color:#71717a">(' + dims + ')</span></span>' +
      '<button class="btn" data-kz-del="' + esc(n) + '" style="padding:3px 8px">✕</button>' +
      '</div>';
  }).join('');
  el.querySelectorAll('[data-kz-del]').forEach(b => b.onclick = () => {
    send({type:'delete_keepout_zone', name: b.dataset.kzDel});
    $('kz-msg').textContent = t('Διαγραφή…'); $('kz-msg').style.color = '#71717a';
  });
}
$('kz-on').onclick = () => {
  if (!confirm(t('Θα επανεκκινήσει όλη τη στοίβα (~90 δευτερόλεπτα) με τις '
              + 'απαγορευμένες ζώνες ενεργές. Να συνεχίσω;'))) return;
  send({type:'keepout_activate', on: true});
  $('kz-msg').textContent = t('Ενεργοποίηση…'); $('kz-msg').style.color = '#71717a';
};
$('kz-off').onclick = () => {
  if (!confirm(t('Θα επανεκκινήσει όλη τη στοίβα (~90 δευτερόλεπτα) χωρίς τις '
              + 'απαγορευμένες ζώνες. Να συνεχίσω;'))) return;
  send({type:'keepout_activate', on: false});
  $('kz-msg').textContent = t('Απενεργοποίηση…'); $('kz-msg').style.color = '#71717a';
};

// ── "πήγαινε να δεις" (check mission) ──────────────────────────────────────
// The same mission the voice `check` tool starts: drive to the room, ask the
// VLM, come back, say the answer. Wired here too because typing a question is
// easier than saying it, and because the mission existed for months with no
// way at all to trigger it.
ROOMS.forEach(name => {
  const o = document.createElement('option');
  o.value = name; o.textContent = name;
  $('ck-room').appendChild(o);
});
$('ck-go').onclick = () => {
  const room = $('ck-room').value;
  if (!room) return;
  send({type:'check_room', room, question: $('ck-q').value});
  $('ck-msg').textContent = t('Ξεκίνησε…');
};
// Sorting reuses the room picker: empty-ish selection means "everything".
$('sort-go').onclick = () => {
  send({type:'sort', room: $('ck-room').value});
  $('ck-msg').textContent = t('Ξεκίνησε…');
};
$('ck-stop').onclick = () => {
  send({type:'cancel_mission'});
  $('ck-msg').textContent = t('Ακυρώθηκε.');
};
// Sending on Enter, because a phone keyboard has no comfortable way to reach
// the button once the field is focused.
$('ck-q').addEventListener('keydown', e => {
  if (e.key === 'Enter') $('ck-go').click();
});

const MISSION_EL = {
  idle:'σε αναμονή', navigating:'πηγαίνει…', inspecting:'κοιτάζει…',
  returning:'γυρίζει…', done:'ολοκληρώθηκε', failed:'απέτυχε',
  cancelled:'ακυρώθηκε',
};
function onMission(m){
  const s = (m.state || '').toLowerCase();
  $('ck-msg').textContent = t(MISSION_EL[s] || s);
  $('ck-msg').style.color = s === 'failed' ? '#f87171'
                          : s === 'done'   ? '#4ade80' : '#71717a';
}

// ── drive controls ─────────────────────────────────────────────────────────
function startDrive(v,w){
  vx=v; wz=w;
  if(!driveTimer) driveTimer=setInterval(()=>send({type:'cmd_vel',vx,wz}),100);
  send({type:'cmd_vel',vx:v,wz:w});
}
function stopDrive(){
  clearInterval(driveTimer); driveTimer=null; vx=0; wz=0;
  send({type:'cmd_vel',vx:0,wz:0});
}
// ‼️ Signs, not speeds. Passing LIN/ANG here would capture whatever they were
// when the page wired itself up, and the sliders would move a number nothing
// reads. Multiply at press time instead.
function bindDrive(id,sv,sw){
  const el=$(id);
  const go=()=>startDrive(sv*LIN, sw*ANG), stop=()=>stopDrive();
  el.addEventListener('mousedown',go);
  el.addEventListener('touchstart',e=>{e.preventDefault();go();},{passive:false});
  ['mouseup','mouseleave'].forEach(ev=>el.addEventListener(ev,stop));
  ['touchend','touchcancel'].forEach(ev=>el.addEventListener(ev,stop));
}
bindDrive('bf', 1, 0); bindDrive('bb',-1, 0);
bindDrive('bl', 0, 1); bindDrive('br', 0,-1);

// ── speed sliders ──────────────────────────────────────────────────────────
// Local to the browser (localStorage), not a robot parameter: two people on
// two phones should not fight over how fast the D-pad drives, and the value
// only ever reaches the base as the twist a held button sends.
function speedSave(){
  try { localStorage.setItem('hr_speed', JSON.stringify({lin:LIN, ang:ANG})); } catch(e){}
}
function speedSync(){
  $('sp-lin').value = LIN; $('sp-ang').value = ANG;
  $('sp-linv').textContent = LIN.toFixed(2) + ' m/s';
  $('sp-angv').textContent = ANG.toFixed(2) + ' rad/s';
  // ~34°/s at the default. Degrees are what a person can picture; rad/s is
  // what the twist carries, so show both.
  $('sp-note').textContent = '≈ ' + Math.round(ANG * 57.3) + '°/s';
  // A held button keeps sending the OLD speed until it is released, because
  // the interval closes over the values startDrive was called with. Re-arm it
  // so dragging a slider mid-drive takes effect immediately.
  if (driveTimer) startDrive(Math.sign(vx)*LIN, Math.sign(wz)*ANG);
}
function speedSet(lin, ang){
  LIN = Math.min(LIN_MAX, Math.max(LIN_MIN, lin));
  ANG = Math.min(ANG_MAX, Math.max(ANG_MIN, ang));
  speedSync(); speedSave();
}
$('sp-lin').addEventListener('input', e => speedSet(+e.target.value, ANG));
$('sp-ang').addEventListener('input', e => speedSet(LIN, +e.target.value));
$('sp-slow').onclick = () => speedSet(LIN_MIN, ANG_MIN);
$('sp-def').onclick  = () => speedSet(LIN_DEF, ANG_DEF);
$('sp-fast').onclick = () => speedSet(LIN_MAX, ANG_MAX);
speedSync();

// ── Costmap tab ────────────────────────────────────────────────────────────
// The costmap arrives as a small PNG (60x60) plus the robot's cell position and
// any detector-placed obstacles. Everything is drawn at the grid's own scale and
// blown up by CSS, so a 3 m window stays legible on a phone.
let costImg = null, costMeta = null, costOn = false;

function costSetActive(on){
  if (on === costOn) return;
  costOn = on;
  send({type:'costmap', on});     // the encode only runs while someone looks
}

function onCostmap(m){
  costMeta = m;
  const img = new Image();
  img.onload = () => { costImg = img; drawCost(); };
  img.src = 'data:image/png;base64,' + m.image;
}

function drawCost(){
  const c = $('cost-canvas'); if (!c || !costImg || !costMeta) return;
  const W = costMeta.width, H = costMeta.height;
  if (c.width !== W || c.height !== H){ c.width = W; c.height = H; }
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.drawImage(costImg, 0, 0);

  // Detector-placed obstacles. They come in metres in base_link, so they need
  // the robot's cell AND the resolution to land in the right square.
  const r = costMeta.robot;
  if (r && costMeta.semantic && costMeta.semantic.length){
    g.fillStyle = '#f472b6';
    costMeta.semantic.forEach(([x, y]) => {
      // base_link x is forward, y is left; the image is y-flipped already.
      const cx = r[0] + x / costMeta.resolution;
      const cy = r[1] - y / costMeta.resolution;
      g.fillRect(cx - 0.5, cy - 0.5, 1.5, 1.5);
    });
  }

  if (r){
    g.fillStyle = '#22c55e';
    g.beginPath(); g.arc(r[0], r[1], 2, 0, Math.PI*2); g.fill();
    g.strokeStyle = '#22c55e'; g.lineWidth = 0.7;
    g.beginPath(); g.arc(r[0], r[1], 3.4, 0, Math.PI*2); g.stroke();
  }
}

// ‼️ These are CSS (RGB) and must match what _cb_costmap paints in cv2 (BGR).
// Copying the BGR tuples straight across gave the inflation swatch as green
// while the map drew it cyan — and green is the ROBOT's colour, so the legend
// was pointing at the wrong thing twice over.
const COST_LEGEND = [
  ['#dc2828', 'θανάσιμο'],       // cv2 (40,40,220)
  ['#ffa500', 'ακουμπά'],        // cv2 (0,165,255)
  ['#1ed2dc', 'inflation'],      // cv2 (220,210,30) at its brightest
  ['#f472b6', 'από ανίχνευση'],
  ['#42423c', 'άγνωστο'],        // cv2 (60,60,66)
];
function buildCostLegend(){
  const box = $('cost-legend'); if (!box || box.childElementCount) return;
  COST_LEGEND.forEach(([colour, label]) => {
    const s = document.createElement('span');
    s.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;color:#a1a1aa';
    const sw = document.createElement('span');
    sw.style.cssText = `width:12px;height:12px;border-radius:3px;background:${colour}`;
    s.appendChild(sw);
    s.appendChild(document.createTextNode(t(label)));
    box.appendChild(s);
  });
}

// ── IMU tab ────────────────────────────────────────────────────────────────
// The rose is drawn from the fused yaw and is deliberately labelled with a
// relative 0°, not N/S/E/W: there is no magnetometer in the fix (see the note
// in the pane), and cardinal letters would be a confident lie.
function drawRose(yawDeg, alive){
  const c = $('imu-rose'); if(!c) return;
  const g = c.getContext('2d'), R = 100;
  g.clearRect(0,0,200,200);
  g.translate(R,R);

  g.strokeStyle = '#3a3a44'; g.lineWidth = 2;
  g.beginPath(); g.arc(0,0,88,0,Math.PI*2); g.stroke();

  // Ticks every 30°, drawn in the fixed frame so the needle is what moves.
  g.strokeStyle = '#52525b'; g.lineWidth = 1;
  for(let a=0; a<360; a+=30){
    const r = (a%90===0) ? 12 : 6, t = (a-90)*Math.PI/180;
    g.beginPath();
    g.moveTo(Math.cos(t)*88, Math.sin(t)*88);
    g.lineTo(Math.cos(t)*(88-r), Math.sin(t)*(88-r));
    g.stroke();
  }
  g.fillStyle = '#71717a'; g.font = '11px system-ui'; g.textAlign = 'center';
  g.fillText('0°',   0, -70); g.fillText('90°',  70,   4);
  g.fillText('180°', 0,  78); g.fillText('-90°',-70,   4);

  if(alive){
    // Screen y grows downward, so -90° maps yaw=0 to straight up and the
    // needle turns counter-clockwise for a positive (left) yaw, matching REP-103.
    const t = (-yawDeg - 90) * Math.PI/180;
    g.strokeStyle = '#3b82f6'; g.lineWidth = 4; g.lineCap = 'round';
    g.beginPath(); g.moveTo(0,0);
    g.lineTo(Math.cos(t)*72, Math.sin(t)*72); g.stroke();
    g.fillStyle = '#3b82f6';
    g.beginPath(); g.arc(0,0,6,0,Math.PI*2); g.fill();
    g.fillStyle = '#e4e4e7'; g.font = 'bold 20px system-ui';
    g.fillText(yawDeg.toFixed(0)+'°', 0, 38);
  } else {
    g.fillStyle = '#71717a'; g.font = '13px system-ui';
    g.fillText(t('χωρίς σήμα'), 0, 5);
  }
  g.setTransform(1,0,0,1,0,0);
}

function onImu(m){
  const pill = (cls, txt) => `<span class="pill ${cls}">${txt}</span>`;
  if(!m.alive){
    // A dead BNO085 is SILENT — it does not report an error, it just stops. So
    // the panel must say so loudly instead of freezing on the last good frame.
    $('i-health').innerHTML = pill('bad', t('ΝΕΚΡΟ'));
    ['i-yaw','i-pitch','i-roll','i-gz','i-gyro','i-acc','i-quat']
      .forEach(id => { const e=$(id); if(e) e.textContent='—'; });
    $('i-hz').textContent = '0 Hz';
    drawRose(0, false);
    return;
  }
  $('i-health').innerHTML = m.gyro_dead
    ? pill('bad', t('ΓΥΡΟΣΚΟΠΙΟ ΝΕΚΡΟ'))
    : pill('ok', t('εντάξει'));
  $('i-yaw').textContent   = m.yaw.toFixed(1)+'°';
  $('i-pitch').textContent = m.pitch.toFixed(1)+'°';
  $('i-roll').textContent  = m.roll.toFixed(1)+'°';
  $('i-gz').textContent    = m.gz.toFixed(3)+' rad/s ('
                           + (m.gz*180/Math.PI).toFixed(0)+'°/s)';
  $('i-hz').textContent    = m.hz.toFixed(1)+' Hz';
  // At rest the BNO085 quantises small rates to exact zeros, so three zeroes
  // are only worth flagging while the robot is actually turning.
  const zeroGyro = (m.gx===0 && m.gy===0 && m.gz===0);
  $('i-gyro').textContent  = `${m.gx.toFixed(3)} / ${m.gy.toFixed(3)} / ${m.gz.toFixed(3)}`
                           + (m.gyro_dead ? '  ⚠ '+t('σταθερά 0 ενώ στρίβει')
                              : (zeroGyro && !m.turning ? '  · '+t('ακίνητο') : ''));
  // Not "0.0 / 0.0 / 0.0": the firmware never enables the accel report, so a
  // zero here is an absent reading, not a stationary robot.
  $('i-acc').textContent   = (m.ax===0 && m.ay===0 && m.az===0)
    ? t('δεν στέλνεται (ανενεργό report)')
    : `${m.ax.toFixed(2)} / ${m.ay.toFixed(2)} / ${m.az.toFixed(2)} m/s²`;
  $('i-quat').textContent  = m.quat.map(v=>v.toFixed(3)).join(' / ');
  drawRose(m.yaw, true);
}

// ── Sensor fusion tab ──────────────────────────────────────────────────────
// Two panels behind one switch each. The numbers are cheap (a few hundred
// bytes at 4 Hz) and go on as soon as the tab opens; the LiDAR/camera
// comparison needs the D435's pointcloud filter, so it stays off until asked.
let fuseOn = false, fuseCam = false;
// 60 s of history at the server's 4 Hz. Kept as three parallel arrays so a
// source that drops out leaves a gap in its own line instead of shifting the
// other two.
const FZ_KEEP = 240;
let fzHist = {wheel: [], imu: [], ekf: []};

function fuseSend(){ send({type:'fusion', on: fuseOn, cam: fuseOn && fuseCam}); }

function fuseSetActive(on){
  if (on === fuseOn) return;
  fuseOn = on;
  fuseSend();
}

function fzPill(cls, txt){ return `<span class="pill ${cls}">${txt}</span>`; }

// A source is judged by its clock, not its value: a wedged publisher keeps its
// last good heading on screen for ever, and that is the failure this tab
// exists to catch.
//
// ‼️ `floor` is what each topic MEASURES on this robot — and measure it on a
// FRESH stack. On 2026-08-05 /imu/data read 6.1 Hz and a threshold was built
// on that; after a `robot max` restart the same topic read 108-113 Hz, which
// is what the firmware actually sends. The 6 Hz was a degraded stream nobody
// had noticed, and a floor of 3 would have hidden it for ever. A stale rate is
// exactly the failure this row exists to show, so the floor is set well under
// the healthy rate but well OVER a stream that has fallen over.
function fzHealth(s, floor){
  if (!s || s.age === null || s.age === undefined)
    return fzPill('bad', t('καμία ένδειξη'));
  if (s.age > 2)
    return fzPill('bad', t('ΣΙΩΠΗ') + ' ' + s.age.toFixed(0) + 's');
  const slow = s.hz < floor;
  return fzPill(slow ? 'warn' : 'ok', s.hz.toFixed(1) + ' Hz')
       + (slow ? ' <span style="color:#fbbf24;font-size:11px">'
                 + t('αργό') + '</span>' : '');
}

function renderFusion(m){
  const S = m.src || {};
  ['wheel','imu','ekf'].forEach(k => {
    const v = S[k] && S[k].yaw;
    fzHist[k].push(v === null || v === undefined ? null : v);
    if (fzHist[k].length > FZ_KEEP) fzHist[k].shift();
  });
  drawFzChart();

  const deg = v => (v === null || v === undefined) ? '—'
                 : (v >= 0 ? '+' : '') + v.toFixed(2) + '°';
  $('fz-dw').textContent = deg(m.dyaw.wheel);
  $('fz-di').textContent = deg(m.dyaw.imu);

  // The wheels ALWAYS open up against the fused heading — that is the whole
  // reason the EKF ignores their yaw — so a few degrees is health, not a
  // fault. Only a gap wide enough to matter for a doorway is worth a colour.
  const worst = Math.max(Math.abs(m.dyaw.wheel || 0), Math.abs(m.dyaw.imu || 0));
  const dead = ['wheel','imu','ekf'].some(
    k => !S[k] || S[k].age === null || S[k].age === undefined || S[k].age > 2);
  $('fz-badge').innerHTML = dead ? fzPill('bad', t('λείπει πηγή'))
    : worst > 15 ? fzPill('bad', t('μεγάλη απόκλιση'))
    : worst > 5  ? fzPill('warn', t('αποκλίνουν'))
                 : fzPill('ok', t('συμφωνούν'));

  const c = m.corr || {};
  if (c.ok){
    $('fz-corr').textContent    = c.d.toFixed(1) + ' cm';
    $('fz-corryaw').textContent = c.yaw.toFixed(2) + '°';
    $('fz-corrmax').textContent = c.dmax.toFixed(1) + ' cm · '
                                + c.yawmax.toFixed(2) + '°';
    $('fz-corr-badge').innerHTML = c.d > 50 ? fzPill('bad', t('μεγάλη διόρθωση'))
      : c.d > 15 ? fzPill('warn', t('μαζεύει'))
                 : fzPill('ok', t('εντάξει'));
  } else {
    ['fz-corr','fz-corryaw','fz-corrmax'].forEach(id => $(id).textContent = '—');
    // No map->odom is not a fusion fault: it means nothing is localizing.
    $('fz-corr-badge').innerHTML = fzPill('warn', t('χωρίς εντοπισμό'));
  }

  const ms = v => v.toFixed(3) + ' m/s', rs = v => v.toFixed(3) + ' rad/s';
  $('fz-vxw').textContent = ms(m.vx.wheel);
  $('fz-vxe').textContent = ms(m.vx.ekf);
  $('fz-wzw').textContent = rs(m.wz.wheel);
  $('fz-wzi').textContent = rs(m.wz.imu);
  $('fz-wze').textContent = rs(m.wz.ekf);

  $('fz-h-wheel').innerHTML = fzHealth(S.wheel, 10);   // measured  19.9 Hz
  $('fz-h-imu').innerHTML   = fzHealth(S.imu, 40);     // measured 108-113 Hz
  $('fz-h-ekf').innerHTML   = fzHealth(S.ekf, 15);     // measured  30.0 Hz

  // The EKF's x/y variance is unbounded by construction (see the note in the
  // card), so only its yaw is shown — printing "±2607 km" next to a healthy
  // robot teaches the reader to ignore the whole card.
  $('fz-cov-ekf').textContent  = m.cov.ekf ? '±' + m.cov.ekf[2].toFixed(1) + '°' : '—';
  // AMCL publishes a pose only when it UPDATES one, and it updates only while
  // the robot drives — so on a parked robot this stays empty for ever. Saying
  // why beats a dash that reads as broken.
  $('fz-cov-amcl').textContent = m.cov.amcl
    ? `±${m.cov.amcl[0].toFixed(1)} / ±${m.cov.amcl[1].toFixed(1)} cm · ±${m.cov.amcl[2].toFixed(1)}°`
    : t('μόλις κινηθεί το ρομπότ');
}

// Strip chart of the three headings. Auto-scaled, because the interesting
// range is anything from a tenth of a degree of noise to a 90° runaway, and a
// fixed axis would render one of those two cases as a flat line.
function drawFzChart(){
  const c = $('fz-chart'); if (!c) return;
  const g = c.getContext('2d'), W = c.width, H = c.height, PAD = 26;
  g.clearRect(0, 0, W, H);

  let peak = 2;                            // never zoom in past ±2°
  for (const k in fzHist)
    for (const v of fzHist[k]) if (v !== null) peak = Math.max(peak, Math.abs(v));
  peak *= 1.15;

  const y = v => H/2 - (v / peak) * (H/2 - PAD/2);
  const x = i => PAD + i * (W - PAD - 6) / (FZ_KEEP - 1);

  g.strokeStyle = '#2c2c32'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(PAD, y(0)); g.lineTo(W - 6, y(0)); g.stroke();
  g.fillStyle = '#52525b'; g.font = '10px system-ui'; g.textAlign = 'right';
  g.fillText('+' + peak.toFixed(peak < 10 ? 1 : 0) + '°', PAD - 4, y(peak) + 9);
  g.fillText('0°', PAD - 4, y(0) + 3);
  g.fillText('-' + peak.toFixed(peak < 10 ? 1 : 0) + '°', PAD - 4, y(-peak) - 2);

  const COLOURS = {wheel: '#fbbf24', imu: '#38bdf8', ekf: '#4ade80'};
  // Dashes, not just colours: when the three agree they sit on exactly the
  // same pixels and only the last one drawn is visible — which reads as two
  // dead sources on a robot where everything is fine.
  const DASH = {wheel: [6, 3], imu: [2, 3], ekf: []};
  for (const k in fzHist){
    const h = fzHist[k];
    g.strokeStyle = COLOURS[k];
    g.setLineDash(DASH[k]);
    g.lineWidth = k === 'ekf' ? 2 : 1.5;
    g.beginPath();
    let pen = false;
    // Right-aligned: the newest sample is always at the right edge, so the
    // chart fills leftwards instead of creeping in from the left on connect.
    const off = FZ_KEEP - h.length;
    h.forEach((v, i) => {
      if (v === null){ pen = false; return; }   // gap, not a line through zero
      const px = x(i + off), py = y(v);
      if (pen) g.lineTo(px, py); else g.moveTo(px, py);
      pen = true;
    });
    g.stroke();
  }
  g.setLineDash([]);
}

// ── LiDAR vs camera, top down ──────────────────────────────────────────────
// Both profiles are nearest-return-per-3°-bin in base_link, so the comparison
// is a straight subtraction. Bin 0 is bearing -180°, bin 60 is straight ahead.
const FP_RANGE = 4.0;      // metres drawn
const FP_GAP   = 0.15;     // metres of disagreement worth calling a disagreement

let fpState = null;

function renderFuseProfile(m){
  fpState = m;
  const zn = $('fp-zmin'), zx = $('fp-zmax');
  if (zn) zn.textContent = (m.zmin * 100).toFixed(0);
  if (zx) zx.textContent = (m.zmax * 100).toFixed(0);
  drawFuseProfile();
}

function drawFuseProfile(){
  const c = $('fp-canvas'); if (!c) return;
  const g = c.getContext('2d'), W = c.width, R = W / 2;
  g.clearRect(0, 0, W, W);
  g.save(); g.translate(R, R);
  const px = r => r / FP_RANGE * (R - 12);

  g.strokeStyle = '#26262c'; g.lineWidth = 1;
  for (let r = 1; r <= FP_RANGE; r++){
    g.beginPath(); g.arc(0, 0, px(r), 0, Math.PI*2); g.stroke();
  }
  g.fillStyle = '#52525b'; g.font = '9px system-ui'; g.textAlign = 'left';
  for (let r = 1; r <= FP_RANGE; r++) g.fillText(r + 'm', 3, -px(r) - 3);

  // Robot: a nose-up triangle, so "up is forward" needs no legend.
  g.fillStyle = '#4ade80';
  g.beginPath(); g.moveTo(0, -9); g.lineTo(6, 7); g.lineTo(-6, 7); g.closePath();
  g.fill();

  const m = fpState;
  if (!m || !m.bins){ g.restore(); return; }
  // base_link x forward, y left. Screen: x right, y down. Forward must point
  // up and a positive (left) bearing must go left, hence the swap and the two
  // negations — getting this wrong mirrors every disagreement onto the wrong
  // side of the robot, which is worse than not drawing it.
  const sx = (r, a) => -Math.sin(a) * px(r);
  const sy = (r, a) => -Math.cos(a) * px(r);
  const bearing = i => (i + 0.5) / m.bins * 2 * Math.PI - Math.PI;

  const dot = (i, r, colour, size) => {
    const a = bearing(i);
    g.fillStyle = colour;
    g.beginPath(); g.arc(sx(r, a), sy(r, a), size, 0, Math.PI*2); g.fill();
  };

  let both = 0, agree = 0, camOnly = 0, nearest = null;
  for (let i = 0; i < m.bins; i++){
    const L = m.lidar ? m.lidar[i] : null;
    const C = m.cam   ? m.cam[i]   : null;
    if (L !== null && L !== undefined && L <= FP_RANGE) dot(i, L, '#e4e4e7', 2.1);
    if (C === null || C === undefined) continue;
    const hidden = (L === null || L === undefined) ? false : C < L - FP_GAP;
    if (L !== null && L !== undefined){
      both++;
      if (Math.abs(C - L) <= FP_GAP) agree++;
    }
    if (hidden){
      camOnly++;
      if (nearest === null || C < nearest) nearest = C;
      // The line is the point: it shows how much nearer the camera says the
      // obstacle is, which a lone dot cannot.
      const a = bearing(i);
      g.strokeStyle = '#f87171'; g.lineWidth = 2;
      g.beginPath(); g.moveTo(sx(C, a), sy(C, a)); g.lineTo(sx(L, a), sy(L, a));
      g.stroke();
      if (C <= FP_RANGE) dot(i, C, '#f87171', 3.0);
    } else if (C <= FP_RANGE){
      dot(i, C, '#38bdf8', 2.4);
    }
  }
  g.restore();

  const stale = m.cam_age === null || m.cam_age === undefined || m.cam_age > 3;
  $('fp-badge').innerHTML = !fuseCam ? fzPill('warn', t('ανενεργό'))
    : stale ? fzPill('bad', t('χωρίς βάθος'))
    : camOnly > 0 ? fzPill('warn', camOnly + ' ' + t('κρυφά'))
                  : fzPill('ok', t('συμφωνούν'));
  $('fp-agree').textContent = both
    ? Math.round(agree / both * 100) + '% ' + t('σε') + ' ' + both + ' '
      + t('κατευθύνσεις')
    : '—';
  $('fp-camonly').textContent = stale ? '—'
    : camOnly + ' ' + t('κατευθύνσεις');
  $('fp-near').textContent = nearest === null ? (stale ? '—' : t('κανένα'))
    : nearest.toFixed(2) + ' m';
}

$('b-fz-reset').onclick = () => {
  send({type: 'fusion_reset'});
  // Clear the chart too: the history is in the OLD reference frame, and
  // leaving it would draw a step change that looks like a sensor jump.
  fzHist = {wheel: [], imu: [], ekf: []};
  drawFzChart();
};
$('fp-on').onchange = e => {
  fuseCam = e.target.checked;
  if (!fuseCam){ fpState = null; drawFuseProfile(); }
  fuseSend();
};
drawFzChart();
drawFuseProfile();

// ── Why the picture looks empty ────────────────────────────────────────────
// A camera at 23 fps showing a white wall 40 cm away looks exactly like a dead
// one — measured brightness 117, detail 5.0 against >100 for a normal room.
// Reported as "δεν δείχνει" twice. The banner names what is actually wrong, so
// the answer is "move the robot", not "the camera is broken".
const CAM_WHY = {
  dark:  'σκοτάδι — κοιτάει σε σκοτεινό χώρο ή είναι καλυμμένη',
  blown: 'κατάλευκο — κοιτάει κατευθείαν σε φως ή σε τοίχο από πολύ κοντά',
  flat:  'επίπεδη επιφάνεια χωρίς λεπτομέρεια — μάλλον τοίχος μπροστά της',
};

function renderCamState(m){
  const el = $('cam-why');
  if (!el) return;
  const why = m && CAM_WHY[m.state];
  if (!why){ el.style.display = 'none'; return; }
  el.style.display = 'block';
  el.textContent = '⚠ ' + t('Η κάμερα ΔΟΥΛΕΥΕΙ') + ' — ' + t(why);
}

// ── Slip map ───────────────────────────────────────────────────────────────
// Latched, so the accumulated history is there the moment the page opens
// rather than after the next correction — which on a parked robot never comes.
let slipMap = null, slipMapOn = false;

function renderSlipMap(m){
  slipMap = m;
  const n = (m.cells || []).length;
  const badge = $('sm-badge');
  if (badge) badge.innerHTML = n
    ? `<span class="pill warn">${n}</span>`
    : `<span class="pill ok">${t('καθαρά')}</span>`;
  const msg = $('sm-msg');
  if (msg && n){
    const worst = m.cells[0];
    // The single most useful sentence: how bad, and where. Metres rather than
    // "a lot", because the number is what tells a rug from calibration.
    msg.textContent = t('Χειρότερο σημείο') + `: ${worst.m.toFixed(2)} m `
      + t('συνολικής διόρθωσης σε') + ` ${worst.n} ` + t('περάσματα') + '.';
  }
  if (slipMapOn) draw();
}

// ── CPU frequency band, for the power station ──────────────────────────────
const PW_LABEL = {off: 'κανονικό', eco: 'ήπιο', flat: 'πολύ σταθερό'};

function powerBuild(){
  document.querySelectorAll('#p-sys [data-pw]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('#p-sys [data-pw]').forEach(x => x.disabled = true);
      $('pw-badge').innerHTML = `<span class="pill warn">${t('σε εξέλιξη')}</span>`;
      send({type: 'power_profile', profile: b.dataset.pw});
    };
  });
}

function powerResult(m){
  document.querySelectorAll('#p-sys [data-pw]').forEach(x => x.disabled = false);
  $('pw-badge').innerHTML = m.ok
    ? `<span class="pill ${m.profile === 'off' ? 'ok' : 'warn'}">`
      + t(PW_LABEL[m.profile] || m.profile) + '</span>'
    : `<span class="pill bad">${t('απέτυχε')}</span>`;
  // The script echoes the band it ended up with; showing that rather than a
  // tick is what makes it checkable — the profile name alone would look
  // identical whether or not the cores actually took the setting.
  $('pw-msg').textContent = m.ok ? (m.detail || '') : (m.error || t('απέτυχε'));
}

// ── USB power cycle ────────────────────────────────────────────────────────
// Pulling the plug on one sensor, from wherever you happen to be. Built from
// the server's list so the browser cannot name a device the server would not
// accept anyway.
let usbBusy = null;

function usbBuild(){
  const box = $('usb-buttons');
  if (!box || box.childElementCount) return;
  for (const name in USB_DEVICES){
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = USB_DEVICES[name];
    b.onclick = () => usbCycle(name, b.textContent);
    box.appendChild(b);
  }
}

function usbCycle(name, label){
  if (usbBusy) return;
  // The base is the one that can hurt: cutting its serial link mid-drive
  // leaves the wheels running on the last command until the driver's 0.25 s
  // watchdog catches it.
  if (name === 'roomba' &&
      !confirm(t('Θα κοπεί η σειριακή της βάσης. Αν το ρομπότ κινείται, οι '
                 + 'τροχοί συνεχίζουν ώσπου να πιάσει το watchdog. Να συνεχίσω;')))
    return;
  usbBusy = name;
  $('usb-badge').innerHTML = `<span class="pill warn">${t('σε εξέλιξη')}</span>`;
  $('usb-msg').textContent = label + ' — ' + t('κόβω το ρεύμα…');
  document.querySelectorAll('#usb-buttons .btn').forEach(b => b.disabled = true);
  send({type: 'usb_power', device: name});
}

function usbResult(m){
  usbBusy = null;
  document.querySelectorAll('#usb-buttons .btn').forEach(b => b.disabled = false);
  const label = USB_DEVICES[m.device] || m.device;
  $('usb-badge').innerHTML = m.ok
    ? `<span class="pill ok">${t('έγινε')}</span>`
    : `<span class="pill bad">${t('απέτυχε')}</span>`;
  $('usb-msg').textContent = m.ok
    ? label + ' — ' + t('ξαναήρθε. Ο κόμβος που την είχε ανοιχτή θέλει restart.')
    : label + ' — ' + (m.error || t('απέτυχε'));
}

// ── keyboard driving ───────────────────────────────────────────────────────
// There were no key bindings at all — the D-pad was mouse/touch only, so on a
// laptop "the keys don't work" was literally true. Arrows + WASD, held down.
//
// Auto-repeat matters here: holding a key fires keydown over and over, and
// restarting the 100 ms timer on each repeat would leave a stale interval
// running, so e.repeat is ignored. Release (or losing focus, or the tab going
// to the background) stops the robot — the driver's 0.25 s watchdog is the
// backstop if even that is missed.
// Signs here too, for the same reason bindDrive takes them: the speed is read
// when the key goes down, so the sliders steer the keyboard as well.
const KEYS = {
  ArrowUp:[1,0], ArrowDown:[-1,0], ArrowLeft:[0,1], ArrowRight:[0,-1],
  w:[1,0], s:[-1,0], a:[0,1], d:[0,-1],
  W:[1,0], S:[-1,0], A:[0,1], D:[0,-1],
};
let keyHeld = null;
// Typing "sad" into the chat box must not drive the robot across the room.
const typing = e => {
  const n = e.target;
  return n && (n.tagName === 'INPUT' || n.tagName === 'TEXTAREA' || n.isContentEditable);
};
document.addEventListener('keydown', e=>{
  if(typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
  if(e.key === ' '){                       // space = panic stop
    e.preventDefault(); keyHeld=null; stopDrive(); send({type:'stop'}); return;
  }
  const k = KEYS[e.key];
  if(!k) return;
  e.preventDefault();                      // arrows must not scroll the pane
  if(e.repeat && keyHeld === e.key) return;
  keyHeld = e.key;
  startDrive(k[0]*LIN, k[1]*ANG);
  const el = $({ArrowUp:'bf',w:'bf',W:'bf', ArrowDown:'bb',s:'bb',S:'bb',
                ArrowLeft:'bl',a:'bl',A:'bl', ArrowRight:'br',d:'br',D:'br'}[e.key]);
  if(el) el.classList.add('lit');
});
document.addEventListener('keyup', e=>{
  if(!KEYS[e.key]) return;
  if(keyHeld === e.key){ keyHeld=null; stopDrive(); }
  document.querySelectorAll('.dbtn.lit').forEach(b=>b.classList.remove('lit'));
});
// A key can be held while the tab is switched away, in which case keyup never
// arrives and the robot would drive on with nobody watching it.
const releaseKeys = ()=>{
  if(keyHeld){ keyHeld=null; stopDrive(); }
  document.querySelectorAll('.dbtn.lit').forEach(b=>b.classList.remove('lit'));
};
window.addEventListener('blur', releaseKeys);
document.addEventListener('visibilitychange', ()=>{ if(document.hidden) releaseKeys(); });

$('bstop').addEventListener('click',()=>{ stopDrive(); send({type:'stop'}); });
// ── vision overlay + follow ────────────────────────────────────────────────
let overlayOn = true;
$('b-overlay').addEventListener('click',()=>{
  overlayOn = !overlayOn;
  $('b-overlay').classList.toggle('pri', overlayOn);
  send({type:'overlay', on:overlayOn});
});
// Following drives the robot, so the stop is its own always-visible button
// rather than a second click on the same one — with a person in the frame and
// the robot already moving, a toggle you have to reason about is the wrong
// control. Same reason the header e-stop is not a toggle-shaped thing.
$('b-nerf-go').addEventListener('click',()=>send({type:'nerf_capture', on:true}));
$('b-nerf-stop').addEventListener('click',()=>send({type:'nerf_capture', on:false}));
$('b-nerf-train-go').addEventListener('click',()=>send({type:'nerf_train_start'}));
$('b-nerf-train-stop').addEventListener('click',()=>send({type:'nerf_train_stop'}));
$('b-nerf-stop-perception').addEventListener('click',()=>send({type:'nerf_stop_perception'}));
$('b-follow').addEventListener('click',()=>send({type:'follow', on:true}));
$('b-follow-stop').addEventListener('click',()=>send({type:'follow', on:false}));

// ── Pose / AprilTag on-demand toggles ───────────────────────────────────────
// State is never tracked client-side: each click just asks for the opposite of
// whatever the button currently shows (driven by the last 'sys' message's node
// list, see updatePerceptionToggles below), so a slow start/stop racing a
// second click self-corrects on the next tick instead of getting out of sync.
$('b-pose-toggle').addEventListener('click', () => {
  send({type:'toggle_pose', on: !$('b-pose-toggle').classList.contains('pri')});
});
$('b-apriltag-toggle').addEventListener('click', () => {
  send({type:'toggle_apriltag', on: !$('b-apriltag-toggle').classList.contains('pri')});
});
function updatePerceptionToggles(nodes){
  const poseOn = nodes.includes('pose_node'), tagOn = nodes.includes('apriltag_node');
  const pb = $('b-pose-toggle'), pp = $('pose-pill');
  pb.classList.toggle('pri', poseOn);
  pb.textContent = (poseOn ? '■ ' : '▶ ') + t('Pose/χειρονομίες');
  pp.textContent = poseOn ? t('ενεργό') : t('σβηστό');
  pp.className = 'pill ' + (poseOn ? 'ok' : '');
  const ab = $('b-apriltag-toggle'), ap = $('apriltag-pill');
  ab.classList.toggle('pri', tagOn);
  ab.textContent = (tagOn ? '■ ' : '▶ ') + t('AprilTag');
  ap.textContent = tagOn ? t('ενεργό') : t('σβηστό');
  ap.className = 'pill ' + (tagOn ? 'ok' : '');
}

$('b-sf-reset').addEventListener('click', ()=>{
  if(!confirm(t('Επαναφορά όλων των ρυθμίσεων ασφαλείας στις προεπιλογές;'))) return;
  send({type:'safety_reset'});
  $('sf-msg').textContent = t('Επαναφέρθηκαν.');
});

$('b-mic-reset').addEventListener('click', ()=>{
  if(!confirm(t('Επαναφορά όλων των ρυθμίσεων μικροφώνου στις προεπιλογές;'))) return;
  send({type:'mic_reset'});
  $('mic-msg').textContent = t('Επαναφέρθηκαν.');
});
// Same generic USB power-cycle the System tab's «Ξεκόλλα αισθητήρα» card
// uses (USB_DEVICES/_usb_power_cycle) — a shortcut here, not a second control.
$('b-mic-power').addEventListener('click', ()=>{
  send({type:'usb_power', device:'mic'});
  $('mic-msg').textContent = t('Γίνεται power-cycle…');
});

let volDragging = false;
$('b-tk-new').addEventListener('click', ()=>{
  if(confirm(t('Νέο κλειδί; Ο παλιός σύνδεσμος θα πάψει να δουλεύει.')))
    send({type:'sys_rotate_token'});
});
$('b-sn-scan').addEventListener('click', ()=>{
  send({type:'sys_refresh'}); snMsg(t('Σάρωση…'));
});
$('b-sb-on').addEventListener('click', ()=>send({type:'sys_bt', action:'on'}));
$('b-sb-scan').addEventListener('click', ()=>{
  send({type:'sys_bt', action:'scan'}); snMsg(t('Αναζήτηση…'));
});
$('sv-vol').addEventListener('input', e => {
  volDragging = true; $('sv-val').textContent = e.target.value + '%';
});
$('sv-vol').addEventListener('change', e => {
  volDragging = false;
  send({type:'sys_volume', percent: parseInt(e.target.value, 10)});
});
// Losing a running robot is one tap away, so both of these confirm.
$('b-sys-reboot').addEventListener('click', ()=>{
  if(confirm(t('Επανεκκίνηση του υπολογιστή;')))
    send({type:'sys_power', action:'reboot'});
});
$('b-sys-off').addEventListener('click', ()=>{
  if(confirm(t('Τερματισμός; Θα χρειαστεί να το ανάψεις με το χέρι.')))
    send({type:'sys_power', action:'poweroff'});
});
// Populate when the Settings tab is first opened, not at page load: an nmcli
// scan takes seconds and nobody is looking at it yet.
document.addEventListener('click', e => {
  const tab = e.target.closest('#tabs .tab');
  if(tab && tab.dataset.pane === 'set' && !window.__sysLoaded){
    window.__sysLoaded = true;
    send({type:'sys_refresh'});
  }
}, true);

$('b-ec-probe').addEventListener('click', ()=>{
  send({type:'echo_probe'});
  $('ec-msg').textContent = t('Τσιρίζει…');
  setTimeout(()=>{ $('ec-msg').textContent=''; }, 4000);
});

$('b-pp-add').addEventListener('click', ()=>{
  const n = $('pp-name').value.trim();
  if(!n){ ppMsg(t('Γράψε πρώτα ένα όνομα.')); return; }
  send({type:'people_add', name:n});
  $('pp-name').value = '';
  ppMsg(t('Προστέθηκε: ') + n);
});
$('pp-name').addEventListener('keydown', e => {
  if(e.key === 'Enter') $('b-pp-add').click();
});

$('dr-rotate').addEventListener('change', e => {
  send({type:'doa_rotate', on: e.target.checked});
  // Not applied yet — /doa/rotate_state is what confirms it. Put the box back
  // where it was so a doa_node that never answers cannot leave it showing ON.
  e.target.checked = !e.target.checked;
  $('dr-msg').textContent = t('Στάλθηκε…');
  setTimeout(()=>{ $('dr-msg').textContent=''; }, 2500);
});

$('gb-motion').addEventListener('change', e => {
  send({type:'gesture_bind', motion_enabled: e.target.checked});
  $('gb-msg').textContent = t('Αποθηκεύτηκε.');
  setTimeout(()=>{ $('gb-msg').textContent=''; }, 2500);
});

$('b-listen').addEventListener('click', startListening);
$('b-listen-stop').addEventListener('click', stopListening);

// ── compass calibration ────────────────────────────────────────────────────
$('b-cp-north').addEventListener('click',()=>{
  if(!pose || typeof pose.yaw !== 'number'){
    $('cp-msg').textContent = t('Δεν υπάρχει θέση — κάνε πρώτα εντοπισμό.');
    return;
  }
  const bearing = parseFloat($('cp-dir').value) || 0;
  send({type:'compass_calibrate', bearing:bearing});
  $('cp-msg').textContent = t('Βαθμονομήθηκε.');
  setTimeout(()=>{ $('cp-msg').textContent=''; }, 3000);
});
$('b-cp-clear').addEventListener('click',()=>send({type:'compass_clear'}));

// ── gestures ───────────────────────────────────────────────────────────────
// Acting on a gesture is deliberately a separate, explicit press — gesture_node
// never drives on its own. See the note in the pane.
$('b-pt-go').addEventListener('click',()=>{
  send({type:'gesture_go'});
  $('pt-msg').textContent = t('Στάλθηκε.');
  setTimeout(()=>{ $('pt-msg').textContent=''; }, 2500);
});

// ── open-vocabulary search ─────────────────────────────────────────────────
function askVocab(q){
  send({type:'vocab', what:q});
  $('vc-msg').textContent = t('Ψάχνω: ') + q;
}
$('b-vc-go').addEventListener('click',()=>{
  const q=$('vc-q').value.trim(); if(q) askVocab(q);
});
$('vc-q').addEventListener('keydown',e=>{
  if(e.key==='Enter'){ const q=$('vc-q').value.trim(); if(q) askVocab(q); }
});
// Empty string clears the vocabulary, which is what idles the detector.
$('b-vc-stop').addEventListener('click',()=>send({type:'vocab', what:''}));
// ‼️ Same rule as the timeline chips: DISPLAY translated, SEND Greek. The
// vocabulary mapping is Greek-stem based.
const VC_CHIPS = ['κλειδιά','γυαλιά','φορτιστής','τηλεκοντρόλ','πορτοφόλι'];

// ‼️ Built from applyLang(), NEVER at top level.
//
// These loops used to run inline, calling t() as the page parsed. t() reads
// LANG, which is declared with `let` further down the file, so the call landed
// in the temporal dead zone and threw "Cannot access 'LANG' before
// initialization" — which aborts the WHOLE script. The tab bar is built by
// renderTabs() further down still, so nothing after the throw ever ran and the
// dashboard rendered ZERO tabs. Reported from a phone as "δεν μου δείχνει τα
// κουμπιά"; it was equally broken on desktop.
//
// Rebuilding here also means the chips follow a language switch, which the
// inline version could not do — it painted once, at load.
function renderChips(){
  const build = (host, items, onPick) => {
    const el = $(host);
    if(!el) return;
    el.innerHTML = '';
    items.forEach(q => {
      const b = document.createElement('button');
      b.className = 'btn'; b.textContent = t(q);
      b.style.fontSize = '11.5px'; b.style.padding = '5px 9px';
      b.onclick = () => onPick(q);
      el.appendChild(b);
    });
  };
  build('tl-chips', TL_CHIPS, q => { $('tl-q').value = t(q); askRecall(q); });
  build('vc-chips', VC_CHIPS, q => { $('vc-q').value = t(q); askVocab(q); });
}

// ── timeline ───────────────────────────────────────────────────────────────
function askRecall(q){
  $('tl-answer').textContent = '…';
  send({type:'recall', when:q});
}
$('b-tl-ask').addEventListener('click',()=>askRecall($('tl-q').value.trim()));
$('tl-q').addEventListener('keydown',e=>{
  if(e.key==='Enter') askRecall($('tl-q').value.trim());
});
// Shortcuts for the periods people actually ask about, so the common case is
// one tap on a phone rather than typing Greek into a tiny box.
// ‼️ The chip DISPLAYS a translated label but SENDS the Greek phrase:
// episodic.parse_time_window matches Greek time words, so an English chip on an
// English UI would silently fall through to "no period named".
const TL_CHIPS = ['σήμερα','σήμερα το πρωί','χθες','πριν από 2 ώρες'];

$('b-loc').addEventListener('click',()=>send({type:'localize'}));
// ‼️ 'cancel_nav', not 'stop'. A zero Twist stops the wheels for one tick and
// bt_navigator writes the next command straight over it — "πατάω ακύρωση στόχου
// και δεν ακυρώνεται" (2026-08-04). The server cancels the goal itself.
$('b-xnav').addEventListener('click',()=>{ goal=null; send({type:'cancel_nav'}); draw(); });

// The header stop is a LATCHED e-stop, not a one-shot zero twist: it is the
// only thing that overrides teleop, which bypasses obstacle_safety entirely.
$('estop').addEventListener('click',()=>{
  estop=!estop;
  stopDrive();
  send({type:'estop',on:estop});
});

// ── arm ────────────────────────────────────────────────────────────────────
const jdiv=$('joints');
ARM_JOINTS.forEach(name=>{
  const [lo,hi]=ARM_LIMITS[name];
  const row=document.createElement('div');
  row.className='joint';
  row.innerHTML=`<label>${name}</label>`
    + `<input type="range" id="j-${name}" min="${lo}" max="${hi}" step="0.01" value="${(lo+hi)/2}">`
    + `<span class="val" id="jv-${name}">—</span>`;
  jdiv.appendChild(row);
  const sl=row.querySelector('input');
  // While a finger is on the slider, incoming joint_states must not yank it
  // back — the arm lags the command by design, so echoing feedback into the
  // control would fight the user.
  const hold=()=>sl.dataset.dragging='1';
  const release=()=>{ delete sl.dataset.dragging; };
  ['mousedown','touchstart'].forEach(e=>sl.addEventListener(e,hold));
  ['mouseup','touchend','touchcancel'].forEach(e=>sl.addEventListener(e,release));
  sl.addEventListener('input',()=>{
    $('jv-'+name).textContent=(sl.value*180/Math.PI).toFixed(0)+'°';
    // Move the 3D view immediately on the slider, not on the joint_states
    // echo: the arm lags the command and a preview that waits for hardware
    // feels broken. The next real state message corrects it either way.
    armAngles[name] = parseFloat(sl.value);
    armDraw();
    send({type:'arm_joint',joint:name,pos:parseFloat(sl.value)});
  });
});
const grip=$('grip');
['mousedown','touchstart'].forEach(e=>grip.addEventListener(e,()=>grip.dataset.dragging='1'));
['mouseup','touchend','touchcancel'].forEach(e=>grip.addEventListener(e,()=>delete grip.dataset.dragging));
grip.addEventListener('input',()=>{
  $('grip-v').textContent=(grip.value*180/Math.PI).toFixed(0)+'°';
  armAngles.hand = parseFloat(grip.value);
  armDraw();
  send({type:'gripper',pos:parseFloat(grip.value)});
});
$('b-grip-open').onclick  = ()=>send({type:'gripper',pos:ARM_LIMITS.hand[1]});
$('b-grip-close').onclick = ()=>send({type:'gripper',pos:ARM_LIMITS.hand[0]});
// Rest pose: elbow folded, shoulder down, base centred in its measured range.
$('b-arm-home').onclick = ()=>{
  send({type:'arm_joint',joint:'shoulder',pos:1.2});
  send({type:'arm_joint',joint:'elbow',pos:2.8});
  send({type:'arm_joint',joint:'wrist',pos:0.0});
};
$('b-arm-limp').onclick = ()=>{
  if(confirm(t('Κόβεται η ροπή — ο βραχίονας θα πέσει. Τον κρατάς;')))
    send({type:'arm_raw',cmd:'{"T":210,"cmd":0}'});
};
$('b-arm-init').onclick = ()=>send({type:'arm_raw',cmd:'{"T":210,"cmd":1}'});
$('b-arm-moveit').onclick = ()=>showTab('moveit');

// ── arm envelope + speed ─────────────────────────────────────────────────
// Same pattern as the safety tab (live SetParameters, paint while dragging,
// commit on release, never fight the user's finger) for a different shape of
// knob: a [lo,hi] PAIR per joint instead of one number, and one speed %
// fanned out to arm_joy AND arm_driver together. See
// home_robot/arm_settings.py for why those are shaped this way.
const ARM_MECH_LIMITS = __ARM_MECH_LIMITS__;
const ARM_ALL_JOINTS = [...ARM_JOINTS, 'hand'];   // + gripper, not in ARM_JOINTS

function armLimBuild(){
  const div = $('armlim');
  ARM_ALL_JOINTS.forEach(name=>{
    const [mlo,mhi] = ARM_MECH_LIMITS[name];
    const [lo0,hi0] = ARM_LIMITS[name];
    const row=document.createElement('div');
    row.className='joint';
    row.dataset.joint=name;
    row.innerHTML = `<label>${name}</label>`
      + `<div style="display:flex;gap:6px">`
      + `<input type="range" class="alo" min="${mlo}" max="${mhi}" step="0.01" value="${lo0}">`
      + `<input type="range" class="ahi" min="${mlo}" max="${mhi}" step="0.01" value="${hi0}">`
      + `</div><span class="val"></span>`;
    div.appendChild(row);
    const lo=row.querySelector('.alo'), hi=row.querySelector('.ahi');
    const paint=()=>{
      row.querySelector('.val').textContent =
        (lo.value*180/Math.PI).toFixed(0)+'° … '+(hi.value*180/Math.PI).toFixed(0)+'°';
    };
    [lo,hi].forEach(inp=>{
      ['mousedown','touchstart'].forEach(e=>inp.addEventListener(e,()=>inp.dataset.dragging='1'));
      ['mouseup','touchend','touchcancel'].forEach(e=>inp.addEventListener(e,()=>delete inp.dataset.dragging));
      inp.addEventListener('input', paint);
      inp.addEventListener('change', ()=>{
        // A handle dragged past the other is sorted server-side too; sorting
        // here as well keeps the two from visibly crossing on screen.
        let a=parseFloat(lo.value), b=parseFloat(hi.value);
        if (a>b){ [a,b]=[b,a]; }
        send({type:'arm_limit_set', joint:name, lo:a, hi:b});
      });
    });
    paint();
  });
}

function armLimPaint(joint, set_, live, mech){
  const row = document.querySelector('#armlim .joint[data-joint="'+joint+'"]');
  if (!row || !set_) return;
  const lo=row.querySelector('.alo'), hi=row.querySelector('.ahi');
  if (document.activeElement !== lo) lo.value = set_[0];
  if (document.activeElement !== hi) hi.value = set_[1];
  row.querySelector('.val').textContent =
    (set_[0]*180/Math.PI).toFixed(0)+'° … '+(set_[1]*180/Math.PI).toFixed(0)+'°'
    + (live ? '' : ' '+t('(δεν τρέχει)'));
  row.classList.toggle('off', !live);
  // The position slider for this joint must never allow an angle the
  // envelope no longer covers — otherwise dragging it would silently do
  // nothing once arm_driver clamps the command server-side.
  const posSlider = joint==='hand' ? $('grip') : $('j-'+joint);
  if (posSlider && document.activeElement !== posSlider){
    posSlider.min = set_[0]; posSlider.max = set_[1];
    posSlider.value = Math.min(Math.max(parseFloat(posSlider.value), set_[0]), set_[1]);
  }
}

function onArmSettings(v){
  ARM_ALL_JOINTS.forEach(name=>{
    const st = v.limits[name];
    if (st) armLimPaint(name, st.set, st.live, st.mech);
  });
  $('armlim-badge').textContent = v.nodes ? t('ενεργό') : t('δεν τρέχει');
  const sp = $('armspeed');
  if (document.activeElement !== sp) sp.value = v.speed;
  $('armspeed-v').textContent = Math.round((v.speed_live || v.speed)*100)+'%'
    + (v.nodes ? '' : ' '+t('(δεν τρέχει)'));
}

$('armspeed').addEventListener('input', ()=>
  $('armspeed-v').textContent = Math.round($('armspeed').value*100)+'%');
$('armspeed').addEventListener('change', ()=>
  send({type:'arm_speed_set', value:+$('armspeed').value}));
$('b-armlim-reset').onclick = ()=>{
  if (confirm(t('Επαναφέρει τα όρια των αρθρώσεων ΚΑΙ την ταχύτητα στις '
      + 'μετρημένες προεπιλογές. Να συνεχίσω;')))
    send({type:'arm_reset'});
};
armLimBuild();
$('b-log-clear').onclick  = ()=>{ $('log-list').innerHTML=''; logSeen=0; logLast=null;
  $('log-count').textContent='0'; };
$('b-cloud-reset').onclick = cloudReset;
$('b-cost-smaller').onclick = ()=>costResize(1/COST_STEP);
$('b-cost-bigger').onclick  = ()=>costResize(COST_STEP);
$('b-map-new').onclick  = mapNew;
$('b-map-save').onclick = mapSave;
cloudBind();

// ── vacuum ─────────────────────────────────────────────────────────────────
$('b-dock').onclick   = ()=>send({type:'dock',on:true});
$('b-undock').onclick = ()=>send({type:'dock',on:false});

// ── chat ───────────────────────────────────────────────────────────────────
const chat=$('chat');
function addMsg(role,text){
  const d=document.createElement('div');
  d.className='msg '+role;
  d.textContent = role==='wake' ? t('— ξύπνησε (')+text+') —' : text;
  chat.appendChild(d);
  while(chat.children.length>80) chat.removeChild(chat.firstChild);
  chat.scrollTop=chat.scrollHeight;
}
// ── language ───────────────────────────────────────────────────────────────
// Greek is the source: it is what is written in the markup, and t() is a
// lookup BY the Greek string. An untranslated string therefore renders as
// Greek instead of as a missing key, which is the failure mode that matters
// when a translation is added late.
const I18N  = __I18N__;
const LANGS = __LANGS__;
let LANG = 'el';
try { LANG = localStorage.getItem('hr_lang') || 'el'; } catch(e) {}

function t(s){
  if(LANG === 'el' || !s) return s;
  const e = I18N[String(s).replace(/\s+/g, ' ').trim()];
  return (e && e[LANG]) || s;
}

// Static markup is translated by walking the DOM once. The original Greek is
// kept per node, because translating an already-translated node would look up
// English text in a Greek-keyed table and find nothing — the page would be
// stuck in whatever language it was switched to first.
let i18nNodes = null;
function i18nCollect(){
  i18nNodes = [];
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: n => (n.parentNode && /^(SCRIPT|STYLE)$/.test(n.parentNode.nodeName))
      ? NodeFilter.FILTER_REJECT
      : (n.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT)
  });
  let n;
  while((n = walk.nextNode())) i18nNodes.push({node: n, orig: n.nodeValue});
  for(const el of document.querySelectorAll('[placeholder],[title]')){
    for(const attr of ['placeholder', 'title']){
      const v = el.getAttribute(attr);
      if(v) i18nNodes.push({el: el, attr: attr, orig: v});
    }
  }
}

function applyLang(){
  if(!i18nNodes) i18nCollect();
  for(const rec of i18nNodes){
    // Keep the surrounding whitespace: it is what indents the notes.
    const lead  = rec.orig.match(/^\s*/)[0];
    const trail = rec.orig.match(/\s*$/)[0];
    const val   = lead + t(rec.orig) + trail;
    if(rec.node) rec.node.nodeValue = val;
    else rec.el.setAttribute(rec.attr, t(rec.orig));
  }
  document.documentElement.lang = LANG;
  renderTabs();
  renderChips();
  setupCards();
  setupViewers();
  for(const b of document.querySelectorAll('#lang-buttons .btn'))
    b.classList.toggle('pri', b.dataset.lang === LANG);
  if(document.querySelector('#p-map.active')) mapsRefresh();
}

function setLang(code){
  LANG = code;
  try { localStorage.setItem('hr_lang', code); } catch(e) {}
  applyLang();
}

// ── maps ───────────────────────────────────────────────────────────────────
// Switching a map hot-swaps just the localization/SLAM nodes (map_mode_switch
// on the Python side) — voice, camera, arm, dashboard, VNC and Foxglove stay
// up. Only Nav2 localization is briefly down (a few seconds, not the ~90 s
// full-stack restart this used to be), so the confirm() is lighter than it
// once was but still there — navigation goes idle for a moment either way.
async function mapsRefresh(){
  // active_map() shells out to `ros2 param get`, which takes a second or two
  // (see the Python side) — without this the ΧΑΡΤΕΣ card looks empty/broken
  // for that whole stretch, including the 🧹 straighten button on each row.
  if(!$('map-list').children.length) $('map-list').textContent = t('Φόρτωση…');
  let d;
  try { d = await (await fetch('/maps' + (TOKEN_QS || ''))).json(); }
  catch(e){ return; }
  $('map-active').textContent = d.mapping ? t('χαρτογράφηση…')
                                          : (d.active || '—');
  const box = $('map-list');
  box.innerHTML = '';
  for(const m of d.maps){
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:10px;padding:6px 0;'
                      + 'border-bottom:1px solid #27272a';
    const isActive = m.name === d.active;
    const when = new Date(m.mtime * 1000).toLocaleDateString('el-GR');
    row.innerHTML = `<span style="flex:1">${isActive ? '● ' : ''}<b>${m.name}</b>`
      + `<span style="color:#71717a;font-size:11.5px"> · ${when} · ${m.kb} kB`
      + `${m.resumable ? ' · ' + t('επεκτάσιμος') : ''}</span></span>`;
    if(!isActive){
      const b = document.createElement('button');
      b.className = 'btn'; b.textContent = t('Ενεργοποίηση');
      b.onclick = () => mapSwitch(m.name);
      row.appendChild(b);
    }
    const clean = document.createElement('button');
    clean.className = 'btn'; clean.textContent = '🧹';
    clean.title = t('Καθαρή εκδοχή');
    clean.onclick = () => mapStraightenPreview(m.name);
    row.appendChild(clean);
    if(!isActive){
      const del = document.createElement('button');
      del.className = 'btn'; del.textContent = '🗑';
      del.title = t('Διαγραφή χάρτη');
      del.onclick = () => mapDelete(m.name);
      row.appendChild(del);
    }
    box.appendChild(row);
  }
}

function mapMsg(s){ $('map-msg').textContent = s; }

async function mapSwitch(name){
  if(!confirm(t('Αλλαγή στον χάρτη')
              + ' "' + name + '". ' + t('Ο εντοπισμός επανεκκινείται (λίγα '
              + 'δευτερόλεπτα) — η φωνή, η κάμερα και ο βραχίονας ΔΕΝ '
              + 'διακόπτονται. Να συνεχίσω;'))) return;
  mapMsg(t('Αλλαγή χάρτη…'));
  try { await fetch('/maps/switch/' + encodeURIComponent(name) + (TOKEN_QS || '')); }
  catch(e){ /* fire-and-forget: the swap keeps going server-side either way */ }
  setTimeout(mapsRefresh, 4000);
  setTimeout(mapsRefresh, 10000);
}

async function mapDelete(name){
  if(!confirm(t('Διαγραφή; ') + name)) return;
  mapMsg(t('Διαγραφή…'));
  try {
    const r = await (await fetch('/maps/delete/' + encodeURIComponent(name)
                                 + (TOKEN_QS || ''))).json();
    mapMsg(r.ok ? t('Διαγράφηκε') + ': ' + name : t('Απέτυχε') + ': ' + (r.error || ''));
  } catch(e){ mapMsg(t('Απέτυχε') + ': ' + e); }
  mapsRefresh();
}

async function mapNew(){
  if(!confirm(t('Ξεκινά ΝΕΑ χαρτογράφηση (SLAM). Ο τρέχων χάρτης δεν χάνεται. '
             + 'Μόνο ο εντοπισμός επανεκκινείται (λίγα δευτερόλεπτα) — η φωνή, '
             + 'η κάμερα και ο βραχίονας ΔΕΝ διακόπτονται. Να συνεχίσω;'))) return;
  mapMsg(t('Ξεκινά χαρτογράφηση… Οδήγησε το ρομπότ σε όλο τον χώρο και μετά '
           + 'αποθήκευσε.'));
  try { await fetch('/maps/new/new' + (TOKEN_QS || '')); } catch(e){}
  setTimeout(mapsRefresh, 4000);
  setTimeout(mapsRefresh, 10000);
}

async function mapSave(){
  const name = $('map-save-name').value.trim();
  if(!/^[A-Za-z0-9_-]{1,40}$/.test(name)){
    mapMsg(t('Δώσε όνομα με λατινικά γράμματα, αριθμούς, - ή _')); return;
  }
  $('map-straighten').style.display = 'none';
  mapMsg(t('Αποθήκευση…'));
  try {
    const r = await (await fetch('/maps/save/' + encodeURIComponent(name)
                                 + (TOKEN_QS || ''))).json();
    mapMsg(r.ok ? t('Αποθηκεύτηκε') + ': ' + name : t('Απέτυχε') + ': ' + (r.result || ''));
    mapsRefresh();
    if(r.ok) mapStraightenPreview(name);
  } catch(e){ mapMsg(t('Απέτυχε') + ': ' + e); }
}

async function mapStraightenPreview(name){
  let d;
  try { d = await (await fetch('/maps/straighten/' + encodeURIComponent(name)
                               + (TOKEN_QS || ''))).json(); }
  catch(e){ return; }
  if(!d || d.error) return;
  $('map-straighten-orig').src  = 'data:image/png;base64,' + d.original;
  $('map-straighten-clean').src = 'data:image/png;base64,' + d.straightened;
  $('map-straighten').style.display = '';
  $('b-map-straighten-keep').onclick = () => { $('map-straighten').style.display = 'none'; };
  $('b-map-straighten-use').onclick = async () => {
    if(!confirm(t('Θα αντικαταστήσει τον χάρτη') + ' "' + name + '" '
               + t('με την ισιωμένη εκδοχή (κρατά αντίγραφο ασφαλείας). Αν αυτός ο '
                  + 'χάρτης είναι ήδη ενεργός, θέλει "Ενεργοποίηση" για να φανεί η '
                  + 'αλλαγή. Να συνεχίσω;'))) return;
    mapMsg(t('Εφαρμογή ισιωμένης εκδοχής…'));
    try {
      const r = await (await fetch('/maps/straighten_apply/' + encodeURIComponent(name)
                                   + (TOKEN_QS || ''))).json();
      mapMsg(r.ok ? t('Έγινε') + ' — ' + t('αντίγραφο ασφαλείας') + ': ' + r.backup
                  : t('Απέτυχε') + ': ' + (r.error || ''));
    } catch(e){ mapMsg(t('Απέτυχε') + ': ' + e); }
    $('map-straighten').style.display = 'none';
  };
}

// ── 3D point cloud ─────────────────────────────────────────────────────────
// Canvas 2D, not WebGL and not three.js: the page must stay self-contained
// (no CDN) and 4000 points sorted back-to-front is about a millisecond, so the
// extra machinery would buy nothing. Points arrive as int16 millimetres +
// RGB, in the camera optical frame: +x right, +y DOWN, +z forward.
// ── 3D arm ──────────────────────────────────────────────────────────────────
// Painter's-algorithm triangle rendering on a 2D canvas. No three.js and no
// WebGL: the page has to stay self-contained (no CDN) and this is ~3000
// triangles, which a 2D context sorts and fills comfortably. Same reasoning as
// the point-cloud tab.
//
// Geometry comes from /arm_model.json (scripts/build_arm_model.py), which
// carries the real URDF joint chain — origins, axes and limits — so what is
// drawn matches where the arm actually is, not an artist's impression.
let armModel = null, armYaw = -0.9, armPitch = -0.35, armZoom = 1;
// The arm's reach in metres, measured from the model itself at zero pose.
// ‼️ Do NOT hard-code a scale divisor here. The old one was tuned against a
// model file that was 1000x too large (micrometres labelled as millimetres),
// so when the units were fixed the arm rendered as a single pixel. Measuring
// makes the view correct for whatever geometry is loaded.
let armReach = null;
let armAngles = {};          // joint name -> radians, live from /joint_states
let armLoading = false;

// URDF joint name -> the short name arm_driver/ARM_JOINTS uses.
const ARM_JOINT_MAP = {
  base_link_to_link1: 'base',
  link1_to_link2:     'shoulder',
  link2_to_link3:     'elbow',
  link3_to_link4:     'wrist',
  link4_to_link5:     'roll',
  link5_to_gripper_link: 'hand',
};

// ── tiny 4x4 matrix helpers (column-major, like everything else in graphics) ──
function m4id(){ return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; }
function m4mul(a,b){
  const o = new Array(16);
  for(let c=0;c<4;c++) for(let r=0;r<4;r++){
    let s=0; for(let k=0;k<4;k++) s += a[k*4+r]*b[c*4+k];
    o[c*4+r]=s;
  }
  return o;
}
function m4trans(x,y,z){ const m=m4id(); m[12]=x; m[13]=y; m[14]=z; return m; }
function m4rot(axis, ang){
  // Rodrigues. The URDF gives an arbitrary axis per joint, not always Z.
  const [x,y,z] = axis, c=Math.cos(ang), s=Math.sin(ang), t=1-c;
  const n = Math.hypot(x,y,z) || 1, ax=x/n, ay=y/n, az=z/n;
  return [
    t*ax*ax+c,      t*ax*ay+s*az,  t*ax*az-s*ay, 0,
    t*ax*ay-s*az,   t*ay*ay+c,     t*ay*az+s*ax, 0,
    t*ax*az+s*ay,   t*ay*az-s*ax,  t*az*az+c,    0,
    0,0,0,1];
}
function m4rpy(r,p,y){
  return m4mul(m4rot([0,0,1],y), m4mul(m4rot([0,1,0],p), m4rot([1,0,0],r)));
}
function m4apply(m, v){
  return [m[0]*v[0]+m[4]*v[1]+m[8]*v[2]+m[12],
          m[1]*v[0]+m[5]*v[1]+m[9]*v[2]+m[13],
          m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[14]];
}

function armLoad(){
  if (armModel || armLoading) return;
  armLoading = true;
  $('arm3d-info').textContent = t('φόρτωση…');
  fetch('/arm_model.json' + TOKEN_QS)
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(m => {
      armModel = m;
      armReach = null;                  // remeasure for the new geometry
      const n = Object.values(m.links).reduce((a,v)=>a+v.length/9, 0);
      $('arm3d-info').textContent = Math.round(n) + ' ' + t('τρίγωνα');
      armDraw();
    })
    .catch(e => {
      armLoading = false;
      $('arm3d-info').textContent = t('δεν φορτώθηκε');
      $('arm3d').getContext('2d').fillText?.('', 0, 0);
    });
}

// Walk the joint chain, accumulating each link's world transform.
function armMeasureReach(){
  const saved = armAngles;
  armAngles = {};                       // zero pose, so it is pose-independent
  const tf = armLinkTransforms();
  let r = 0;
  for (const [link, flat] of Object.entries(armModel.links)){
    const m = tf[link]; if (!m) continue;
    for (let i=0;i<flat.length;i+=3){
      const v = m4apply(m, [flat[i]/1000, flat[i+1]/1000, flat[i+2]/1000]);
      const d = Math.hypot(v[0], v[1], v[2]);
      if (d > r) r = d;
    }
  }
  armAngles = saved;
  return r > 1e-4 ? r : 0.5;
}

function armLinkTransforms(){
  const out = {world: m4id()};
  for (const j of armModel.joints){
    const parent = out[j.parent];
    if (!parent) continue;
    let m = m4mul(parent, m4mul(m4trans(j.xyz[0], j.xyz[1], j.xyz[2]),
                                m4rpy(j.rpy[0], j.rpy[1], j.rpy[2])));
    if (j.type !== 'fixed'){
      const short = ARM_JOINT_MAP[j.name];
      let a = short !== undefined && armAngles[short] !== undefined
              ? armAngles[short] : 0;
      if (j.limit) a = Math.max(j.limit[0], Math.min(j.limit[1], a));
      m = m4mul(m, m4rot(j.axis, a));
    }
    out[j.child] = m;
  }
  return out;
}

// Per-link palette AND material: `spec`/`shin` make the grey links read as
// brushed aluminium (bright, tight highlight) and the gripper as plastic
// (dimmer, broader highlight) rather than every link sharing one identical
// sheen — real materials do not all shine the same way, and matching that is
// most of what "looks realistic" buys for free on top of correct shading.
// The gripper also stays blue so the business end is obvious; the segments
// alternate slightly so adjacent links do not merge into one grey mass when
// they fold over each other.
const ARM_MAT_DEFAULT = {rgb: [150, 155, 165], spec: 0.45, shin: 28};
const ARM_COLORS = {
  base_link:         {rgb: [96, 100, 112],  spec: 0.60, shin: 42},
  link1:             {rgb: [150, 155, 168], spec: 0.60, shin: 42},
  link2:             {rgb: [132, 138, 152], spec: 0.60, shin: 42},
  link3:             {rgb: [150, 155, 168], spec: 0.60, shin: 42},
  link4:             {rgb: [132, 138, 152], spec: 0.60, shin: 42},
  link5:             {rgb: [150, 155, 168], spec: 0.60, shin: 42},
  gripper_link:      {rgb: [ 70, 140, 232], spec: 0.22, shin: 10},
  gripper_left_link: {rgb: [ 70, 140, 232], spec: 0.22, shin: 10},
};

// Two lights fixed in VIEW space, not world space — a world-fixed light leaves
// a permanently dark side that orbiting cannot inspect, so the arm would not
// be readable from every angle. KEY is the bright headlight the shading always
// had; FILL is new: a dim, cool light from roughly the opposite side, which is
// what actually sells "in a room" instead of "lit from one bare bulb" — a
// single light leaves every recessed face black, and real ambient bounce
// never does that.
function _armNorm(v){ const n = Math.hypot(v[0],v[1],v[2])||1; return [v[0]/n,v[1]/n,v[2]/n]; }
const ARM_LIGHT      = _armNorm([-0.42, 0.78, -0.46]);
const ARM_LIGHT_COL  = [1.00, 0.97, 0.90];   // slightly warm — the key light
const ARM_FILL       = _armNorm([0.55, -0.30, -0.35]);
const ARM_FILL_COL   = [0.45, 0.55, 0.80];   // cool and dim — bounce, not a second sun

function armDraw(){
  const cv = $('arm3d'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w*dpr || cv.height !== h*dpr){
    cv.width = w*dpr; cv.height = h*dpr;
  }
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  if (!armModel){ return; }

  const tf = armLinkTransforms();
  const cy = Math.cos(armYaw), sy = Math.sin(armYaw);
  const cp = Math.cos(armPitch), sp = Math.sin(armPitch);
  if (armReach === null) armReach = armMeasureReach();
  // Fit the arm to ~80% of the smaller canvas dimension. ‼️ The factor was 1.25,
  // which is 125% — the comment said 80% and the code did the opposite, so a
  // fully extended arm was drawn taller than the canvas and the gripper went
  // off the edge no matter how the view was centred. armZoom is how you get
  // closer, deliberately.
  const scale = (Math.min(w, h) * 0.8 / armReach) * armZoom;
  // Origin is resolved AFTER the geometry is projected — see the centring pass
  // below. Fixed offsets were guesswork and left the gripper off the bottom.
  let ox = 0, oy = 0;

  // World -> view. Screen x is xv, screen up is yv, into-the-screen is zv.
  const toView = v => {
    const xr =  v[0]*cy + v[1]*sy;
    const yr = -v[0]*sy + v[1]*cy;
    return [xr, yr*sp + v[2]*cp, yr*cp - v[2]*sp];
  };
  // Gentle perspective. FOCAL is in metres and the arm is ~0.4 m, so this adds
  // depth without the wide-angle distortion a short focal length would give.
  const FOCAL = 2.2;
  const project = p => {
    const k = FOCAL / (FOCAL + p[2]);
    return [ox + p[0]*scale*k, oy - p[1]*scale*k, k];
  };

  // ── build faces with view-space normals ───────────────────────────────────
  const faces = [];
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (const [link, flat] of Object.entries(armModel.links)){
    const m = tf[link]; if (!m) continue;
    const mat = ARM_COLORS[link] || ARM_MAT_DEFAULT;
    for (let i=0;i<flat.length;i+=9){
      const vv = [];
      for (let k=0;k<3;k++){
        const world = m4apply(m, [flat[i+k*3]/1000, flat[i+k*3+1]/1000,
                                  flat[i+k*3+2]/1000]);
        if (world[0] < minX) minX = world[0];
        if (world[0] > maxX) maxX = world[0];
        if (world[1] < minY) minY = world[1];
        if (world[1] > maxY) maxY = world[1];
        vv.push(toView(world));
      }
      // Surface normal in view space. THIS is what was missing: the old
      // renderer shaded by depth alone, which reads as a flat silhouette
      // because every face of a cylinder at the same distance got the same
      // grey. A normal gives each face its own tilt towards the light.
      const ux = vv[1][0]-vv[0][0], uy = vv[1][1]-vv[0][1], uz = vv[1][2]-vv[0][2];
      const wx = vv[2][0]-vv[0][0], wy = vv[2][1]-vv[0][1], wz = vv[2][2]-vv[0][2];
      let nx = uy*wz - uz*wy, ny = uz*wx - ux*wz, nz = ux*wy - uy*wx;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx/=nl; ny/=nl; nz/=nl;
      // Backface cull: the mesh winds consistently (every link has a positive
      // signed volume), so a face pointing away from the camera is inside the
      // solid. Dropping it halves the fill work AND removes the painter's
      // -algorithm artefacts where a far face overwrote a near one.
      if (nz > 0) continue;

      const p = vv.map(project);
      faces.push([(vv[0][2]+vv[1][2]+vv[2][2])/3, p, mat, [nx,ny,nz]]);
    }
  }
  faces.sort((a,b) => b[0]-a[0]);          // far (large z) first

  // ── centre on what was actually drawn ─────────────────────────────────────
  // The projected bounding box, not a world-space guess: the arm folds and
  // extends, and a fixed origin put the gripper off the bottom of the canvas.
  let bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9;
  for (const f of faces) for (const q of f[1]){
    if (q[0] < bx0) bx0 = q[0];
    if (q[0] > bx1) bx1 = q[0];
    if (q[1] < by0) by0 = q[1];
    if (q[1] > by1) by1 = q[1];
  }
  if (bx1 > bx0){
    ox = w/2 - (bx0 + bx1)/2;
    oy = h/2 - (by0 + by1)/2;
  } else { ox = w/2; oy = h/2; }

  // ── ground: grid, then contact shadow, both before the arm ────────────────
  // Grid squares sized to the arm, not to a fixed 5 cm, so the ground reads
  // the same whatever the model's dimensions turn out to be.
  const gridStep = armReach / 2.5, gridN = 2;
  ctx.lineWidth = 1;
  for (let i = -gridN; i <= gridN; i++){
    for (const axis of [0, 1]){
      const a = [], b = [];
      for (let j = -gridN; j <= gridN; j++){
        const p = axis === 0 ? [i*gridStep, j*gridStep, 0]
                             : [j*gridStep, i*gridStep, 0];
        const q = project(toView(p));
        (j === -gridN ? a : b).push(q);
      }
      const p0 = project(toView(axis === 0 ? [i*gridStep, -gridN*gridStep, 0]
                                           : [-gridN*gridStep, i*gridStep, 0]));
      const p1 = project(toView(axis === 0 ? [i*gridStep,  gridN*gridStep, 0]
                                           : [ gridN*gridStep, i*gridStep, 0]));
      ctx.strokeStyle = (i === 0) ? '#33333d' : '#1e1e24';
      ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
      ctx.stroke();
    }
  }


  // ── contact shadow, sized to the real footprint ───────────────────────────
  if (minX < maxX){
    const c  = project(toView([(minX+maxX)/2, (minY+maxY)/2, 0]));
    const ex = project(toView([maxX + 0.03, (minY+maxY)/2, 0]));
    const ey = project(toView([(minX+maxX)/2, maxY + 0.03, 0]));
    const rx = Math.max(6, Math.hypot(ex[0]-c[0], ex[1]-c[1]));
    const ry = Math.max(4, Math.hypot(ey[0]-c[0], ey[1]-c[1]) * 0.55);
    const g = ctx.createRadialGradient(c[0], c[1], 0, c[0], c[1], Math.max(rx,ry));
    g.addColorStop(0, 'rgba(0,0,0,0.45)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.save();
    ctx.translate(c[0], c[1]); ctx.scale(1, ry/Math.max(rx,ry));
    ctx.translate(-c[0], -c[1]);
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(c[0], c[1], Math.max(rx,ry), 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
  }

  // ── shade and fill ────────────────────────────────────────────────────────
  const L = ARM_LIGHT, F = ARM_FILL;
  for (const [, p, mat, n] of faces){
    const col = mat.rgb;
    // Key light: Lambert diffuse, two-sided via abs so a stray inward-wound
    // triangle shows as lit rather than as a black hole (same reasoning the
    // single-light version had). Fill light is one-sided — it is standing in
    // for ambient bounce, which does not illuminate a face from behind it.
    const kd = Math.abs(n[0]*L[0] + n[1]*L[1] + n[2]*L[2]);
    const fd = Math.max(0, n[0]*F[0] + n[1]*F[1] + n[2]*F[2]);
    // Blinn-Phong specular against the view direction (0,0,-1), key light
    // only — a second specular pass for the dim fill light would not be
    // visible and would just cost another pow() per triangle. Per-material
    // spec/shin is what makes the grey links look metallic and the gripper
    // look like plastic instead of every link sharing one identical sheen.
    let hx = L[0], hy = L[1], hz = L[2] - 1;
    const hl = Math.hypot(hx, hy, hz) || 1;
    const spec = Math.pow(Math.abs((n[0]*hx + n[1]*hy + n[2]*hz)/hl), mat.shin) * mat.spec;
    const r = Math.min(255, col[0]*(0.16 + kd*0.78*ARM_LIGHT_COL[0] + fd*0.40*ARM_FILL_COL[0]) + 255*spec)|0;
    const g2= Math.min(255, col[1]*(0.16 + kd*0.78*ARM_LIGHT_COL[1] + fd*0.40*ARM_FILL_COL[1]) + 255*spec)|0;
    const b = Math.min(255, col[2]*(0.16 + kd*0.78*ARM_LIGHT_COL[2] + fd*0.40*ARM_FILL_COL[2]) + 255*spec)|0;
    ctx.fillStyle = ctx.strokeStyle = `rgb(${r},${g2},${b})`;
    // ‼️ ox/oy are added HERE, not inside project(): the centring pass needs
    // the raw projected box first, and it only exists once every face has been
    // projected. Applying the offset in project() would need the answer before
    // the question.
    ctx.beginPath();
    ctx.moveTo(p[0][0]+ox, p[0][1]+oy);
    ctx.lineTo(p[1][0]+ox, p[1][1]+oy);
    ctx.lineTo(p[2][0]+ox, p[2][1]+oy);
    ctx.closePath();
    ctx.fill();
    // Stroking with the fill colour closes the hairline seams that appear
    // between separately-filled adjacent triangles on a 2D canvas.
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }
}

(function armWireInteraction(){
  const cv = $('arm3d'); if (!cv) return;
  let drag = null;
  const pos = e => e.touches ? [e.touches[0].clientX, e.touches[0].clientY]
                             : [e.clientX, e.clientY];
  const down = e => { drag = pos(e); cv.style.cursor='grabbing'; };
  const move = e => {
    if (!drag) return;
    const [x,y] = pos(e);
    armYaw   += (x - drag[0]) * 0.01;
    armPitch += (y - drag[1]) * 0.01;
    armPitch = Math.max(-1.4, Math.min(1.4, armPitch));
    drag = [x,y];
    e.preventDefault();
    armDraw();
  };
  const up = () => { drag = null; cv.style.cursor='grab'; };
  cv.addEventListener('mousedown', down);
  cv.addEventListener('touchstart', down, {passive:true});
  window.addEventListener('mousemove', move);
  cv.addEventListener('touchmove', move, {passive:false});
  window.addEventListener('mouseup', up);
  cv.addEventListener('touchend', up);
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    armZoom = Math.max(0.4, Math.min(3, armZoom * (e.deltaY > 0 ? 0.9 : 1.1)));
    armDraw();
  }, {passive:false});
  $('b-arm3d-reset').onclick = () => {
    armYaw=-0.9; armPitch=-0.35; armZoom=1; armDraw();
  };
})();


// ── who is speaking ─────────────────────────────────────────────────────────
// Deliberately shows what it does NOT know: a name with no matched face reads
// differently from a confident identification, and the badge should not imply
// the robot can tell two people apart when it cannot.
function onSpeaker(m){
  const bits = [];
  if (m.angle !== null && m.angle !== undefined){
    const side = Math.abs(m.angle) <= 20 ? t('μπροστά')
               : m.angle > 0 ? t('δεξιά') : t('αριστερά');
    bits.push(side + ' (' + Math.round(m.angle) + '°)');
  }
  if (m.faces_visible) bits.push(m.faces_visible + ' ' + t('πρόσωπα'));
  $('sp-badge').textContent = m.name || (bits.length ? t('άγνωστος') : '—');
  $('sp-detail').textContent =
    (m.identified ? '✅ ' + t('ταυτοποιημένος') + ' · ' : '') +
    (bits.join(' · ') || t('κανείς δεν μιλάει'));
}

// ── fall alert ──────────────────────────────────────────────────────────────
// Dismissing hides the banner but does NOT clear the robot's state: the person
// is still on the floor until fall_monitor_node says otherwise. If a NEW alert
// arrives after dismissal it shows again, which is why the flag is keyed to the
// event rather than to a global "muted".
let fallSeen = false;
function onFall(m){
  if (!m.on){ fallSeen = false; $('fall-bar').style.display = 'none'; return; }
  if (fallSeen) return;
  $('fall-bar').style.display = 'flex';
}
function onFallEvent(m){
  fallSeen = false;
  $('fall-bar').style.display = 'flex';
  const when = m.at ? new Date(m.at * 1000).toLocaleTimeString() : '';
  const ang = (m.torso_angle === null || m.torso_angle === undefined)
              ? '' : ' · ' + t('κλίση') + ' ' + m.torso_angle + '°';
  $('fall-detail').textContent = when + ang;
}
$('fall-ack').onclick = () => {
  fallSeen = true;
  $('fall-bar').style.display = 'none';
};

// ── LLM backend switch ──────────────────────────────────────────────────────
// Switching to the NPU model can take a minute (FastFlowLM has to load 4.7 GB
// of weights), so the buttons lock until the server answers rather than
// letting a second impatient click start a second one.
let beBusy = false;
function onBackend(m){
  beBusy = m.state === 'busy';
  $('be-msg').textContent = m.text || '';
  $('be-msg').style.color = m.state === 'err' ? '#f87171'
                          : m.state === 'busy' ? '#fbbf24' : '#71717a';
  const b = m.backend;
  $('be-badge').textContent = b === 'gemini' ? 'Gemini'
                            : b === 'lemonade' ? 'Qwen3.5 (NPU)'
                            : m.state === 'busy' ? '…' : '—';
  ['gemini','lemonade'].forEach(k => {
    const el = $('be-' + k);
    el.classList.toggle('pri', b === k);
    el.disabled = beBusy;
    el.style.opacity = beBusy ? '.5' : '1';
  });
  quotaVisibility(b);
}
// ── how many questions are left today ──────────────────────────────────────
// The Gemini API tells nobody how much quota is left, so llm_bridge counts
// every request it sends and publishes the total. What arrives here is already
// subtracted; this only paints it.
//
// ‼️ The limit is an estimate until a real 429 corrects it, and the wording
// says so ("περίπου"). A counter that quietly rounds itself into confidence is
// worse than a dash — the whole point is to know when to stop asking.
let quotaSeen = null;
function onQuota(m){
  quotaSeen = m;
  const box = $('q-box');
  const limit = m.limit || 0, left = Math.max(0, m.left || 0);
  const used = m.used || 0;
  const num = n => n.toLocaleString('el-GR');
  box.style.display = 'block';
  $('q-left').textContent = left > 0 ? num(left) + t(' από ') + num(limit)
                                     : t('τέλος για σήμερα');
  const pct = limit ? Math.min(100, Math.round(used * 100 / limit)) : 0;
  const bar = $('q-bar');
  bar.style.width = pct + '%';
  bar.style.background = left <= 0 ? '#f87171'
                       : left < limit * 0.1 ? '#fbbf24' : '#34d399';
  const hours = Math.floor((m.resets_in || 0) / 3600);
  const mins  = Math.round(((m.resets_in || 0) % 3600) / 60);
  const when = hours >= 1 ? hours + t(' ώρες') : mins + t(' λεπτά');
  const about = m.source === 'measured' ? '' : t('περίπου ');
  $('q-note').textContent = left > 0
    ? t('Έχουν φύγει ') + num(used) + ' — ' + about + num(left)
      + t(' ακόμη. Μηδενίζει σε ') + when
      + t('. Μία εντολή που κινεί το ρομπότ πιάνει δύο.')
    : t('Το ημερήσιο όριο εξαντλήθηκε. Ξεκλειδώνει σε ') + when
      + t(' — ως τότε γύρνα στο Qwen (NPU).');
}
function quotaVisibility(backend){
  if (backend === 'lemonade' || backend === 'ollama') $('q-box').style.display = 'none';
  else if (quotaSeen) onQuota(quotaSeen);
}
function beSet(backend){
  if (beBusy) return;
  onBackend({state:'busy', text:'Αλλαγή…', backend:null});
  send({type:'set_backend', backend});
}
$('be-gemini').onclick   = () => beSet('gemini');
$('be-lemonade').onclick = () => beSet('lemonade');

// ── System tab rendering ────────────────────────────────────────────────────
// The DOM is built once per shape change and afterwards only widths/text are
// touched: this runs every second, and rebuilding ~40 nodes each time made the
// tab flicker and lost the CSS transitions that make a moving bar readable.
let sysShape = '';

function bar(pct, warn, bad){
  const cls = pct >= (bad || 90) ? 'bad' : pct >= (warn || 75) ? 'warn' : '';
  return '<div class="bar"><i class="' + cls + '" style="width:' +
         Math.max(0, Math.min(100, pct)) + '%"></i></div>';
}

function renderSys(m){
  // ── usage bars ──
  const g = (m.mem_free_gb !== undefined);
  const rows = [
    ['CPU', m.cpu, m.cpu.toFixed(0) + '%' +
       (m.cpu_cores ? ' · ' + m.cpu_cores + ' ' + t('νήματα') : '')],
    ['RAM', m.mem, m.mem + '% · ' +
       (g ? m.mem_free_gb + ' ' + t('GB ελεύθερα') : m.mem_gb + ' GB')],
  ];
  // Load is shown against the core count, because the raw number means nothing
  // on its own: 8.0 is idle on 16 threads and on fire on 4.
  if (m.load_pct !== undefined)
    rows.push(['Φόρτος', m.load_pct, m.load + (m.load15 ? '  (' + m.load15.join(' · ') + ')' : '')]);
  if (m.swap_total_gb)
    rows.push(['Swap', 100 * m.swap_gb / m.swap_total_gb,
               m.swap_gb + ' / ' + m.swap_total_gb + ' GB']);

  const shape = 'b' + rows.length + '_' + (m.cpu_each ? m.cpu_each.length : 0) +
                '_t' + (m.temps ? m.temps.length : 0) +
                '_d' + (m.disks ? m.disks.length : 0);
  const rebuild = shape !== sysShape;
  if (rebuild) sysShape = shape;

  const host = $('s-bars');
  if (rebuild){
    host.innerHTML =
      rows.map((r, i) =>
        '<div class="mrow"><span class="lbl">' + t(r[0]) + '</span>' +
        '<span id="sb' + i + '"></span><span class="val" id="sv' + i + '"></span></div>'
      ).join('') +
      (m.cpu_each ? '<div class="cores" id="s-cores">' +
        m.cpu_each.map(() => '<i><b></b></i>').join('') + '</div>' : '');
  }
  rows.forEach((r, i) => {
    $('sb' + i).innerHTML = bar(r[1]);
    $('sv' + i).textContent = r[2];
  });
  if (m.cpu_each){
    const cells = $('s-cores').children;
    for (let i = 0; i < m.cpu_each.length && i < cells.length; i++){
      const b = cells[i].firstChild;
      b.style.height = m.cpu_each[i] + '%';
      b.style.background = m.cpu_each[i] >= 90 ? '#ef4444'
                         : m.cpu_each[i] >= 70 ? '#f59e0b' : '#3b82f6';
    }
  }

  // ── temperatures ──
  if (m.temps){
    const el = $('s-temps');
    if (rebuild)
      el.innerHTML = m.temps.map((s, i) =>
        '<div class="tcell" id="tc' + i + '"><div class="tn">' + s.icon +
        '<span>' + s.name + '</span></div><div class="tv" id="tt' + i + '"></div></div>'
      ).join('') || '—';
    m.temps.forEach((s, i) => {
      const cell = $('tc' + i); if (!cell) return;
      $('tt' + i).textContent = s.c + '°';
      const w = s.warn || 85;
      cell.className = 'tcell' + (s.c >= w ? ' bad' : s.c >= w - 12 ? ' warn' : '');
    });
  }

  // ── disks ──
  if (m.disks){
    const el = $('s-disks');
    if (rebuild)
      el.innerHTML = m.disks.map((d, i) =>
        '<div class="mrow"><span class="lbl">' + d.mount + '</span>' +
        '<span id="db' + i + '"></span><span class="val" id="dv' + i + '"></span></div>'
      ).join('') || '—';
    m.disks.forEach((d, i) => {
      if (!$('db' + i)) return;
      $('db' + i).innerHTML = bar(d.pct, 80, 92);
      $('dv' + i).textContent = d.free + ' ' + t('GB ελεύθερα') +
                                ' / ' + d.total + ' GB';
    });
  }
}

let cloudPts = null, cloudRGB = null;
let cloudYaw = 0, cloudPitch = -0.15, cloudZoom = 1, cloudOn = false;

function cloudReset(){ cloudYaw = 0; cloudPitch = -0.15; cloudZoom = 1; cloudDraw(); }

function onCloud(m){
  const bin = atob(m.data);
  const n = m.n;
  const xyz = new Float32Array(n * 3);
  const rgb = new Uint8Array(n * 3);
  for(let i = 0; i < n; i++){
    const o = i * 9;
    // int16 little-endian, by hand: this is a binary string, not a buffer.
    for(let k = 0; k < 3; k++){
      let v = bin.charCodeAt(o + k*2) | (bin.charCodeAt(o + k*2 + 1) << 8);
      if(v & 0x8000) v -= 0x10000;
      xyz[i*3 + k] = v / 1000;
    }
    rgb[i*3]     = bin.charCodeAt(o + 6);
    rgb[i*3 + 1] = bin.charCodeAt(o + 7);
    rgb[i*3 + 2] = bin.charCodeAt(o + 8);
  }
  cloudPts = xyz; cloudRGB = rgb;
  $('cloud-info').textContent = m.n + ' / ' + m.total + ' ' + t('σημεία');
  cloudDraw();
}

function cloudDraw(){
  const cv = $('cloud-canvas');
  if(!cv) return;
  const w = cv.clientWidth, h = cv.clientHeight;
  if(cv.width !== w || cv.height !== h){ cv.width = w; cv.height = h; }
  const g = cv.getContext('2d');
  g.fillStyle = '#0a0a0b'; g.fillRect(0, 0, w, h);
  if(!cloudPts){
    g.fillStyle = '#52525b'; g.font = '13px system-ui'; g.textAlign = 'center';
    g.fillText(t('αναμονή για νέφος σημείων…'), w/2, h/2);
    return;
  }
  const n = cloudPts.length / 3;
  const cy = Math.cos(cloudYaw), sy = Math.sin(cloudYaw);
  const cp = Math.cos(cloudPitch), sp = Math.sin(cloudPitch);
  const f = Math.min(w, h) * 0.9 * cloudZoom;

  // Painter's algorithm — without the sort, far points overwrite near ones and
  // the cloud looks inside out.
  const order = new Int32Array(n), depth = new Float32Array(n);
  const px = new Float32Array(n), py = new Float32Array(n);
  let count = 0;
  for(let i = 0; i < n; i++){
    const x = cloudPts[i*3], y = cloudPts[i*3 + 1], z = cloudPts[i*3 + 2];
    if(!(z > 0.05)) continue;
    // Orbit around a point a metre and a half in front of the camera.
    const x1 =  x * cy + (z - 1.5) * sy;
    const z1 = -x * sy + (z - 1.5) * cy;
    const y1 =  y * cp - z1 * sp;
    const z2 =  y * sp + z1 * cp + 1.5;
    if(z2 < 0.15) continue;
    order[count] = i; depth[count] = z2;
    px[count] = w/2 + f * x1 / z2;
    py[count] = h/2 + f * y1 / z2;
    count++;
  }
  const idx = Array.from({length: count}, (_, i) => i)
                   .sort((a, b) => depth[b] - depth[a]);
  for(const j of idx){
    const i = order[j];
    const s = Math.max(1, Math.min(4, 2.2 * cloudZoom / depth[j]));
    g.fillStyle = 'rgb(' + cloudRGB[i*3] + ',' + cloudRGB[i*3+1] + ',' + cloudRGB[i*3+2] + ')';
    g.fillRect(px[j], py[j], s, s);
  }
}

function cloudBind(){
  const cv = $('cloud-canvas');
  let drag = null, pinch = null;
  cv.addEventListener('pointerdown', e => {
    drag = {x: e.clientX, y: e.clientY}; cv.setPointerCapture(e.pointerId);
    cv.style.cursor = 'grabbing';
  });
  cv.addEventListener('pointermove', e => {
    if(!drag) return;
    cloudYaw   += (e.clientX - drag.x) * 0.008;
    cloudPitch += (e.clientY - drag.y) * 0.008;
    cloudPitch = Math.max(-1.4, Math.min(1.4, cloudPitch));
    drag = {x: e.clientX, y: e.clientY};
    cloudDraw();
  });
  const stop = () => { drag = null; cv.style.cursor = 'grab'; };
  cv.addEventListener('pointerup', stop);
  cv.addEventListener('pointercancel', stop);
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    cloudZoom = Math.max(0.25, Math.min(6, cloudZoom * (e.deltaY < 0 ? 1.12 : 0.89)));
    cloudDraw();
  }, {passive: false});
  cv.addEventListener('touchmove', e => {
    if(e.touches.length !== 2) return;
    e.preventDefault();
    const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
                         e.touches[0].clientY - e.touches[1].clientY);
    if(pinch) {
      cloudZoom = Math.max(0.25, Math.min(6, cloudZoom * d / pinch));
      cloudDraw();
    }
    pinch = d;
  }, {passive: false});
  cv.addEventListener('touchend', () => { pinch = null; });
  window.addEventListener('resize', () => { if(cloudOn) cloudDraw(); });
}

// The stream is 150 kB/s, so it runs only while the tab is actually open.
function cloudSetActive(on){
  if(on === cloudOn) return;
  cloudOn = on;
  send({type: 'cloud', on: on});
  if(on) cloudDraw();
}

// Same idea: the backend's raw color subscription (_set_camera_sub) only
// exists while nobody is on this tab (~13 points of a core measured live,
// 2026-08-14 — see the comment on _cam_ws for why an in-callback early-return
// alone was not enough). Clearing src (not just hiding the pane) is what
// actually closes the /camera.mjpeg connection — a display:none <img> keeps
// fetching.
let camOn = false;
let camBlobUrl = null;
function camSetActive(on){
  if(on === camOn) return;
  camOn = on;
  send({type: 'cam_view', on: on});
  $('cam').src = on ? ('/camera.mjpeg' + TOKEN_QS) : '';
  if(!on && camBlobUrl){ URL.revokeObjectURL(camBlobUrl); camBlobUrl = null; }
}

// Safari-only in practice (see _cb_camera's comment in web_dashboard_node.py
// for why the camera also rides the websocket): each frame replaces the
// <img> src with a fresh blob URL. The OLD url is revoked only after the
// new one is assigned, not before — revoking the one still on-screen while
// it might still be decoding risks the image going blank between frames.
function camShowFrame(bytes){
  if(!camOn) return;
  const url = URL.createObjectURL(new Blob([bytes], {type: 'image/jpeg'}));
  const old = camBlobUrl;
  camBlobUrl = url;
  $('cam').src = url;
  if(old) URL.revokeObjectURL(old);
}

// ── log tail ───────────────────────────────────────────────────────────────
// Levels are rcl_interfaces/Log: 30 WARN, 40 ERROR, 50 FATAL. Auto-scroll is
// suppressed when the user has scrolled up to read something — otherwise the
// next warning yanks the line they were reading off the screen.
const LOG_LEVELS = {30:['WARN','#facc15'], 40:['ERROR','#f87171'], 50:['FATAL','#f87171']};
let logSeen = 0;
let logLast = null;      // {key, count, el} — the run currently being folded

// A repeating warning must not push everything else off the tab. The D435
// reports "Incomplete video frame detected! Size 686856 out of 814335 bytes"
// every few seconds under load, and the byte count differs every time, so
// folding on the exact text would never match. Digits are replaced with # to
// get the SHAPE of the message; identical shapes from the same node collapse
// into one line with a counter, and any different message ends the run.
const logShape = m => m.level + '|' + m.name + '|' +
  String(m.text).replace(/\d+/g, '#');

function addLog(m){
  const list = $('log-list');
  const stick = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  const [name, colour] = LOG_LEVELS[m.level] || ['LOG', '#a1a1aa'];
  const ts = new Date(m.t * 1000).toLocaleTimeString('el-GR');
  const key = logShape(m);
  $('log-count').textContent = ++logSeen;
  if(logLast && logLast.key === key){
    logLast.count++;
    // The newest timestamp and the newest numbers, so a folded run still shows
    // what is happening NOW rather than freezing on the first occurrence.
    logLast.el.textContent =
      `${ts}  ${name}  [${m.name}]  ${m.text}  ×${logLast.count}`;
    if(stick) list.scrollTop = list.scrollHeight;
    return;
  }
  const d = document.createElement('div');
  d.style.color = colour;
  d.textContent = `${ts}  ${name}  [${m.name}]  ${m.text}`;
  list.appendChild(d);
  // The folded line is always the LAST one, and the trim always takes the
  // first, so a run can never be folded into a node that has been detached.
  logLast = {key: key, count: 1, el: d};
  while(list.children.length > 300) list.removeChild(list.firstChild);
  if(stick) list.scrollTop = list.scrollHeight;
}
function sendChat(kind){
  const inp=$('chat-text'), text=inp.value.trim();
  if(!text) return;
  send({type:kind,text});
  if(kind==='ask') addMsg('user',text);
  inp.value='';
}
$('b-send').onclick = ()=>sendChat('ask');
$('b-say').onclick  = ()=>sendChat('say');
$('chat-text').addEventListener('keydown',e=>{ if(e.key==='Enter') sendChat('ask'); });

// ── start ──────────────────────────────────────────────────────────────────
// ‼️ Everything that RUNS lives here, at the bottom, after every declaration.
// `let`/`const` are hoisted but not initialised, so calling i18nCollect() from
// higher up the file — where it used to be, next to the button wiring — threw
// "Cannot access 'i18nNodes' before initialization" and killed the whole
// script. Nothing below the throw ran: no tabs, no camera, no websocket, so
// the page rendered its layout and then sat there completely dead. That is
// what it looked like from the MacBook on 2026-08-01.
//
// Collect the ORIGINAL Greek markup before anything writes translated text
// into the page, or a second language switch has no Greek left to look up.
i18nCollect();
for(const [code, name] of LANGS){
  const b = document.createElement('button');
  b.className = 'btn'; b.dataset.lang = code; b.textContent = name;
  b.onclick = () => setLang(code);
  $('lang-buttons').appendChild(b);
}
safetyBuild();             // rows must exist before applyLang() translates them
skirtBuild();
micBuild();                // same reason: rows must exist first
wakeModelBuild();
// The map tab is the one active by default at load, unlike 'set' before it —
// showTab() only refreshes on a CLICK, so the tab that starts already open
// needs its own call or the card sits on "Φόρτωση…" until the user leaves and
// comes back.
mapsRefresh();
usbBuild();                // same: the buttons carry translatable labels
powerBuild();
applyLang();               // also builds the tabs, so it precedes showTab()

// camSetActive (called from showTab) sets $('cam').src, so it only starts
// requesting /camera.mjpeg if the page happens to load straight onto 'cam'.
showTab('map');
resize(); connect();
</script>
<script type="importmap">
{"imports": {"three": "/vendor/three/three.module.min.js"}}
</script>
<script type="module">
// Photorealistic "Σάρωμα" view for the map tab's second button. A separate
// module script (import needs type="module") from the classic one above —
// Three.js is the one deliberate exception to this dashboard's otherwise
// self-contained, no-three.js canvas rendering (arm tab etc.): a phone
// LiDAR scan's baked texture needs a real WebGL scene graph, which the
// hand-rolled canvas painter's-algorithm renderer has no material system
// for. Vendored locally under /vendor/three (see THREE_VENDOR_DIR), not a
// CDN, so this still works with no internet.
//
// A module's top-level scope is its own, but it still chains to the page's
// outer script-scope for name lookup — TOKEN_QS/$/t below are the classic
// script's `const`/`function`, read live, not copied. This runs after that
// script (module execution is deferred to after parsing) so they already
// exist by the time this file's top level runs.
import * as THREE from 'three';
import { GLTFLoader } from '/vendor/three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from '/vendor/three/addons/controls/OrbitControls.js';

(function(){
  const canvas = document.getElementById('scan3d');
  if (!canvas) return;
  const info = document.getElementById('scan3d-info');

  let renderer, scene, camera, controls, marker, goalMarker, scanRoot;
  let robotGroup, robotModelReady = false;
  let trailLine, planLine, trailLen = -1, planLen = -1;
  let loaded = false, loading = false;
  const raycaster = new THREE.Raycaster();

  // Map (x,y) polyline -> a THREE.Line, rebuilding its geometry only when
  // the point count actually changed (robotTrail/plan are appended-to or
  // replaced wholesale, not mutated in place, so a length check is enough
  // to skip the rebuild on the ~55 idle frames between real updates).
  function syncLine(line, pts, prevLen, y){
    if (pts.length === prevLen) return prevLen;
    line.visible = pts.length > 1;
    if (pts.length > 1){
      const arr = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++){
        arr[i*3] = pts[i][0]; arr[i*3+1] = y; arr[i*3+2] = -pts[i][1];
      }
      line.geometry.dispose();
      line.geometry = new THREE.BufferGeometry();
      line.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    }
    return pts.length;
  }

  function ensureScene(){
    if (renderer) return;
    renderer = new THREE.WebGLRenderer({canvas, antialias:true});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0b);
    camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100);
    camera.position.set(4, 3, 4);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.update();
    controls.addEventListener('change', saveCam);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
    const sun = new THREE.DirectionalLight(0xfff4e0, 1.0);
    sun.position.set(3, 6, 2);
    scene.add(sun);
    // Robot marker: cone tip along local +X, matching the map's yaw=0-faces-
    // +X convention (see pose.yaw's use at the laser-pose projection above).
    const cone = new THREE.ConeGeometry(0.12, 0.30, 12);
    cone.rotateZ(-Math.PI / 2);
    marker = new THREE.Mesh(cone, new THREE.MeshStandardMaterial({color: 0xf59e0b}));
    marker.visible = false;
    scene.add(marker);
    // Photorealistic robot marker: loaded lazily below (loadRobotModel), only
    // takes over from the cone once it actually loads — the cone is the
    // fallback if /robot_scan.glb 404s.
    robotGroup = new THREE.Group();
    robotGroup.visible = false;
    scene.add(robotGroup);
    loadRobotModel();
    // Goal marker: flat ring on the floor, same #ffa040 as the 2D canvas's
    // click-to-navigate circle (draw()'s goal marker) — same shared `goal`
    // global, just a different renderer drawing it.
    const ring = new THREE.TorusGeometry(0.16, 0.02, 8, 24);
    ring.rotateX(Math.PI / 2);
    goalMarker = new THREE.Mesh(ring, new THREE.MeshStandardMaterial({color: 0xffa040}));
    goalMarker.visible = false;
    scene.add(goalMarker);
    // Trail (where driven, same colour/source as the 2D canvas's) and plan
    // (upcoming Nav2 route, same blue as the 2D canvas's) — both read the
    // classic script's shared arrays each frame, like goalMarker does.
    trailLine = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color: 0x38bdf8}));
    trailLine.visible = false;
    scene.add(trailLine);
    planLine = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color: 0x60a5fa}));
    planLine.visible = false;
    scene.add(planLine);
    wireClickToNavigate();
    window.addEventListener('resize', resize);
    resize();
    animate();
  }

  // Click-to-navigate: tap the mesh to send a Nav2 goal there, mirroring the
  // 2D canvas's plain single-click-sends behaviour (no confirmation dialog —
  // see its click handler). Distinguishing a tap from an OrbitControls
  // drag-to-orbit needs its own check: both start as the same pointerdown on
  // this canvas, and a native 'click' event still fires after a small drag,
  // which would fire an unwanted goal mid-rotate.
  function wireClickToNavigate(){
    let down = null;
    canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', e => {
      const start = down; down = null;
      if (!start || !loaded || !scanRoot) return;
      if (Math.hypot(e.clientX - start[0], e.clientY - start[1]) > 6) return; // was a drag/orbit
      const r = canvas.getBoundingClientRect();
      const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
      const ny = -((e.clientY - r.top) / r.height) * 2 + 1;
      raycaster.setFromCamera({x: nx, y: ny}, camera);
      const hit = raycaster.intersectObject(scanRoot, true)[0];
      if (!hit) return;
      // Inverse of setPose's map->glTF transform: glTF (x,y,z) -> map (x,-z).
      const wx = hit.point.x, wy = -hit.point.z;
      goal = {x: wx, y: wy};
      send({type: 'nav_goal', x: wx, y: wy});
      draw();
    });
  }

  function resize(){
    if (!renderer) return;
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function animate(){
    requestAnimationFrame(animate);
    // Reads the classic script's shared `goal` (same var the 2D canvas's
    // draw() and click handler use) every frame rather than needing its own
    // change event — cheap, and stays in sync regardless of which view set it.
    if (goalMarker){
      goalMarker.visible = loaded && !!goal;
      if (goal) goalMarker.position.set(goal.x, 0.03, -goal.y);
    }
    if (loaded && trailLine) trailLen = syncLine(trailLine, robotTrail, trailLen, 0.03);
    if (loaded && planLine) planLen = syncLine(planLine, plan || [], planLen, 0.04);
    controls.update();
    renderer.render(scene, camera);
  }

  // Scaniverse exports standard glTF Y-up: measured against this same scan's
  // occupancy grid (scripts/ply_to_map.py), glTF (x, y_up, z) equals the map
  // frame's (x, up, -y) — a -90 deg rotation about X. Rather than rotate the
  // (large) loaded mesh, only the small marker gets that transform per pose.
  function setPose(m){
    if (!marker) return;
    if (!m){
      marker.visible = false;
      if (robotGroup) robotGroup.visible = false;
      return;
    }
    // The photorealistic robot model, once loaded, replaces the cone —
    // never both at once, so a still-loading/missing scan doesn't leave the
    // robot invisible in the meantime.
    marker.visible = loaded && !robotModelReady;
    marker.position.set(m.x, 0.06, -m.y);
    marker.rotation.y = m.yaw;
    if (robotGroup){
      robotGroup.visible = loaded && robotModelReady;
      robotGroup.position.set(m.x, 0.01, -m.y);
      robotGroup.rotation.y = m.yaw;
    }
  }

  // Loads config/robot_scan.glb (a phone LiDAR scan of the physical robot,
  // e.g. Scaniverse) once, independent of the house scan's load(). Same
  // native-Y-up == this scene's world convention as the house mesh (see
  // load()'s comment) — no axis rotation needed, just centre it on its own
  // local (x,z) origin and shift it up so its lowest scanned point sits on
  // the floor, since the raw scan's own origin is wherever Scaniverse put it.
  function loadRobotModel(){
    new GLTFLoader().load('/robot_scan.glb' + (TOKEN_QS || ''),
      (gltf) => {
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const centre = box.getCenter(new THREE.Vector3());
        gltf.scene.position.x -= centre.x;
        gltf.scene.position.z -= centre.z;
        gltf.scene.position.y -= box.min.y;
        robotGroup.add(gltf.scene);
        robotModelReady = true;
        setPose(pose || null);
      },
      undefined,
      () => { /* no robot_scan.glb yet — the cone marker stays the fallback */ });
  }

  // Remembers the camera framing across page loads (localStorage, same
  // pattern as loadSizes()/saveSizes() for the resizable viewer panels) —
  // added on request so "the size it's at right now" survives a reload
  // instead of resetting to the whole-house auto-fit every time.
  const CAM_KEY = 'hr_scan3d_camera';
  function loadCam(){
    try { return JSON.parse(localStorage.getItem(CAM_KEY)); } catch(e){ return null; }
  }
  let saveCamTimer = null;
  function saveCam(){
    clearTimeout(saveCamTimer);
    saveCamTimer = setTimeout(() => {
      try {
        localStorage.setItem(CAM_KEY, JSON.stringify({
          pos: camera.position.toArray(),
          target: controls.target.toArray(),
        }));
      } catch(e){}
    }, 300);   // debounced: 'change' fires on every drag/wheel tick
  }

  function load(){
    if (loading || loaded) return;
    loading = true;
    if (info) info.textContent = t('φόρτωση…');
    new GLTFLoader().load('/maps/scan.glb' + (TOKEN_QS || ''),
      (gltf) => {
        scene.add(gltf.scene);
        scanRoot = gltf.scene;
        // Frame the camera from the mesh's own measured extent, not a fixed
        // guess — a whole-house scan can be anywhere from one room to 15 m
        // across, and the arm tab's fixed-FOCAL mistake (memory:
        // project_robot_arm_3d_model) is exactly the failure mode a
        // hardcoded camera position falls into here: too close, inside the
        // walls, or too far to make anything out. Skipped once a saved
        // camera exists — the user's own last framing wins from then on.
        const box = new THREE.Box3().setFromObject(gltf.scene);
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        const radius = Math.max(size.x, size.y, size.z, 1) * 0.9;
        camera.near = radius / 100;
        camera.far = radius * 20;
        camera.updateProjectionMatrix();
        const saved = loadCam();
        if (saved){
          camera.position.fromArray(saved.pos);
          controls.target.fromArray(saved.target);
        } else {
          camera.position.set(centre.x + radius, centre.y + radius * 0.6, centre.z + radius);
          controls.target.copy(centre);
        }
        controls.update();
        loaded = true; loading = false;
        if (info) info.textContent = t('έτοιμο');
        setPose(pose || null);
      },
      undefined,
      () => {
        loading = false;
        if (info) info.textContent = t('δεν υπάρχει σάρωμα για αυτόν τον χάρτη');
      });
  }

  // Named window.hrScan3d, not window.scan3d: an id="scan3d" element already
  // exists (the canvas above), and browsers auto-expose elements with an id
  // as same-named globals the instant they're parsed — long before this
  // module's imports (fetched over the network) let it get this far. A pose
  // WS message landing in that gap would find the bare canvas element at
  // window.scan3d and crash calling .setPose on it. Prefixed name sidesteps
  // the whole class of DOM-legacy-global collision, not just this one.
  window.hrScan3d = {
    activate(){ ensureScene(); resize(); load(); },
    setPose,
  };
})();
</script>
</body>
</html>"""


# ── main ───────────────────────────────────────────────────────────────────────

def _spin(node: Node):
    try:
        rclpy.spin(node)
    except Exception:
        pass

def main():
    global ros_node
    rclpy.init()
    ros_node = DashboardNode(state, locations)

    t = threading.Thread(target=_spin, args=(ros_node,), daemon=True)
    t.start()

    qs = f'?t={TOKEN}' if TOKEN else ''
    print(f'\n🤖  Web dashboard → http://{_lan_ip()}:{PORT}/{qs}')
    if NO_AUTH:
        print('    ‼️  HOME_ROBOT_DASHBOARD_NO_AUTH=1 — anyone who can reach '
              'this port can drive the robot.')
    else:
        print(f'    token stored in {TOKEN_FILE} (delete it to rotate)')
    if not os.path.isdir(NOVNC_DIR):
        print('    ⚠  noVNC missing — RViz/MoveIt/Gazebo tabs will be empty '
              '(sudo apt install novnc)')
    print()
    # Quiet by default — at 25 fps the camera stream alone would drown the log.
    # HOME_ROBOT_DASHBOARD_ACCESS_LOG=1 turns the request log back on, which is
    # the only way to tell "the browser got a 401" from "the page loaded but the
    # websocket died". Both look identical from outside: a few connections that
    # open and close. Cost a round of guessing on 2026-08-01.
    access = os.environ.get('HOME_ROBOT_DASHBOARD_ACCESS_LOG') == '1'
    uvicorn.run(app, host='0.0.0.0', port=PORT,
                log_level='info' if access else 'warning',
                access_log=access)


if __name__ == '__main__':
    main()
