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
import os
import secrets
import socket
import subprocess
import threading
import time
from typing import Optional, Set
from urllib.parse import quote

import cv2
import numpy as np
import psutil
import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rcl_interfaces.msg import Log as RosoutLog
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Empty

from home_robot.dashboard_i18n import LANGUAGES, as_js_table

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

PORT = 8080
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
SRC_MAPS_DIR = (os.path.join(SRC_HOME, 'maps')
                if os.path.isdir(os.path.join(SRC_HOME, 'maps'))
                else os.path.join(SHARE, 'maps'))
NOVNC_DIR = '/usr/share/novnc'

# Must match the display map in scripts/gui_session.sh.
VNC_PORTS = {'rviz': 5902, 'gazebo': 5903, 'moveit': 5904}
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


def _load_or_create_token() -> str:
    try:
        with open(TOKEN_FILE) as f:
            tok = f.read().strip()
        if tok:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(16)
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

# ── Shared state (thread-safe) ─────────────────────────────────────────────────

class State:
    def __init__(self):
        self._lock    = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: Set[WebSocket] = set()
        self.map_png:   Optional[bytes] = None
        self.map_info:  Optional[dict]  = None
        self.camera_jpg: Optional[bytes] = None
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
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 5)
        self.create_subscription(Path, '/plan', self._cb_plan, 5)
        self.create_subscription(Odometry, '/odom', self._cb_odom, 5)
        # room_markers publishes this latched (TRANSIENT_LOCAL) precisely so a
        # late subscriber gets the current room; asking for volatile threw that
        # away, so the badge stayed '—' until the robot next changed room.
        self.create_subscription(String, '/current_room', self._cb_room, latch)

        # ── Camera / perception ─────────────────────────────────────────────
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._cb_camera, 5)
        self.create_subscription(String, '/detected_objects', self._cb_objects, 5)

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
        self._cloud_last = 0.0
        self._map_cache = None          # (fetched_at, name); see active_map()
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

        # ── Publishers ──────────────────────────────────────────────────────
        self._vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self._speech_pub = self.create_publisher(String, '/speech_text', 10)
        self._say_pub  = self.create_publisher(String, '/speech_response', 10)
        self._dock_pub = self.create_publisher(Bool, '/dock', 10)
        self._arm_cmd_pub = self.create_publisher(JointState, '/arm/joint_cmd', 10)
        self._gripper_pub = self.create_publisher(Float32, '/arm/gripper_cmd', 10)
        self._arm_raw_pub = self.create_publisher(String, '/arm/raw_cmd', 10)
        # The e-stop is latched on the driver's side, so ours must be too or a
        # driver that restarts comes up not knowing the stop is engaged.
        self._estop_pub = self.create_publisher(
            Bool, '/emergency_stop',
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        self._loc_client = self.create_client(Empty, '/localize_globally')

        self.create_timer(2.0, self._publish_system)

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        img = np.full((msg.info.height, msg.info.width, 3), 180, dtype=np.uint8)
        img[grid == 0]   = [230, 230, 230]
        img[grid == 100] = [50,  50,  50]
        img[grid == -1]  = [160, 160, 160]
        img = cv2.flip(img, 0)
        _, png = cv2.imencode('.png', img)
        self._state.map_png = png.tobytes()
        info = {
            'width':      msg.info.width,
            'height':     msg.info.height,
            'resolution': msg.info.resolution,
            'origin':     [msg.info.origin.position.x,
                           msg.info.origin.position.y],
        }
        self._state.map_info = info
        self._state.broadcast({
            'type':  'map',
            **info,
            'image': base64.b64encode(self._state.map_png).decode(),
        }, remember=False)   # replayed from map_png on connect, not from latest

    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        p   = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        self._state.broadcast({
            'type': 'pose',
            'x':   round(p.position.x, 3),
            'y':   round(p.position.y, 3),
            'yaw': round(yaw, 4),
        })

    def _cb_scan(self, msg: LaserScan):
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

    def _cb_plan(self, msg: Path):
        # Nav2 republishes the global plan at controller rate; 40 points is
        # plenty to draw a readable line and keeps the socket quiet.
        pts = msg.poses[::max(1, len(msg.poses) // 40)] if msg.poses else []
        self._state.broadcast({
            'type': 'plan',
            'points': [[round(p.pose.position.x, 2), round(p.pose.position.y, 2)]
                       for p in pts],
        })

    def _cb_odom(self, msg: Odometry):
        self._state.broadcast({
            'type': 'odom',
            'vx': round(msg.twist.twist.linear.x, 3),
            'wz': round(msg.twist.twist.angular.z, 3),
        })

    def _cb_camera(self, msg: Image):
        try:
            enc = msg.encoding.lower()
            arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if enc == 'rgb8' else cv2.COLOR_RGBA2BGR if 'rgba' in enc else cv2.COLOR_BGRA2BGR if 'bgra' in enc else -1) \
                  if enc != 'bgr8' else arr
            if bgr is None or not isinstance(bgr, np.ndarray):
                return
            if bgr.shape[1] > 640:
                scale = 640 / bgr.shape[1]
                bgr = cv2.resize(bgr, (640, int(bgr.shape[0] * scale)))
            _, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self._state.camera_jpg = jpg.tobytes()
        except Exception:
            pass

    def _cb_objects(self, msg: String):
        self._state.broadcast({'type': 'objects', 'text': msg.data})

    def _cb_room(self, msg: String):
        self._state.broadcast({'type': 'room', 'name': msg.data})

    # Roughly what a phone on wifi can take: 4000 points x 9 bytes is ~36 kB,
    # ~48 kB once base64'd, three times a second.
    CLOUD_MAX_POINTS = 4000
    CLOUD_PERIOD = 0.33

    def _cb_cloud(self, msg: PointCloud2):
        if not self._cloud_ws:
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

        # Millimetres in int16 covers +-32 m; the D435 stops at 10.
        mm = np.clip(xyz * 1000.0, -32000, 32000).astype('<i2')
        rgb = bgr[:, ::-1].copy()          # BGR -> RGB for the browser
        payload = np.concatenate([mm.view(np.uint8).reshape(len(mm), 6), rgb],
                                 axis=1).tobytes()
        self._state.broadcast({
            'type': 'cloud', 'n': len(mm), 'frame': msg.header.frame_id,
            'total': n, 'data': base64.b64encode(payload).decode(),
        }, remember=False)

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

    # ── System panel ─────────────────────────────────────────────────────────

    def _publish_system(self):
        temp = None
        try:
            for name, entries in (psutil.sensors_temperatures() or {}).items():
                if entries and name in ('k10temp', 'acpitz', 'coretemp'):
                    temp = round(entries[0].current, 1)
                    break
        except Exception:
            pass
        vm = psutil.virtual_memory()
        # get_node_names() reads the local graph cache — unlike `ros2 node
        # list` it costs nothing and cannot hang when the daemon is stale.
        try:
            nodes = sorted({n for n, _ in self.get_node_names_and_namespaces()})
        except Exception:
            nodes = []
        self._state.broadcast({
            'type':  'sys',
            'cpu':   psutil.cpu_percent(),
            'mem':   round(vm.percent, 1),
            'mem_gb': round(vm.used / 2**30, 1),
            'temp':  temp,
            'load':  round(os.getloadavg()[0], 2),
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

    def release_client(self, client):
        """A browser went away — forget anything it had switched on."""
        self._cloud_ws.discard(client)

    def dispatch(self, msg: dict, client=None):
        t = msg.get('type')
        if t == 'cmd_vel':
            tw = Twist()
            tw.linear.x  = float(msg.get('vx', 0))
            tw.angular.z = float(msg.get('wz', 0))
            self._vel_pub.publish(tw)
        elif t == 'stop':
            self._vel_pub.publish(Twist())
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
        elif t == 'localize':
            if self._loc_client.service_is_ready():
                self._loc_client.call_async(Empty.Request())
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
    resp = HTMLResponse(_make_html(list(locations.keys()), t or TOKEN))
    if not NO_AUTH:
        resp.set_cookie(COOKIE_NAME, TOKEN, max_age=COOKIE_MAX_AGE,
                        httponly=True, samesite='strict')
    return resp


@app.get('/camera.mjpeg')
async def camera(request: Request, t: str = ''):
    if not _authorised(t, request.cookies):
        return Response('Unauthorized', status_code=401)
    async def stream():
        while True:
            jpg = state.camera_jpg
            if jpg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpg + b'\r\n')
            await asyncio.sleep(0.04)   # ~25 fps cap
    return StreamingResponse(stream(),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# ── GUI sessions (RViz / MoveIt / Gazebo) ─────────────────────────────────────

def _gui_session(action: str, app_name: str) -> str:
    if not os.path.exists(GUI_SESSION_SH):
        return f'missing {GUI_SESSION_SH}'
    try:
        # Gazebo takes ~75 s to come up on this machine (software rendering),
        # so the timeout is generous; the UI polls status rather than blocking.
        r = subprocess.run(['bash', GUI_SESSION_SH, action, app_name],
                           capture_output=True, text=True, timeout=120)
        return (r.stdout or r.stderr).strip() or 'ok'
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception as e:
        return f'error: {e!r}'


@app.get('/gui/{app_name}/{action}')
async def gui(request: Request, app_name: str, action: str, t: str = ''):
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    if app_name not in VNC_PORTS or action not in ('start', 'stop', 'status'):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    out = await asyncio.to_thread(_gui_session, action, app_name)
    return {'app': app_name, 'action': action, 'result': out,
            'running': out.startswith('running')}


# ── Maps ──────────────────────────────────────────────────────────────────────
# The map is baked into map_server at launch: there is no runtime swap, so
# switching one means restarting the stack. That is why the UI confirms first
# and why the work is handed to a detached script — this node is one of the
# processes about to be killed.

MAP_SESSION_SH = os.path.normpath(
    os.path.join(SRC_MAPS_DIR, os.pardir, 'scripts', 'map_session.sh'))


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


@app.get('/maps/{action}/{name}')
async def maps_action(request: Request, action: str, name: str, t: str = ''):
    if not _authorised(t, request.cookies):
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    # Names go into a shell command and a file path, so they are whitelisted
    # rather than escaped.
    if action not in ('switch', 'save', 'new') or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', name):
        return JSONResponse({'error': 'bad request'}, status_code=400)
    if action == 'switch' and name not in {m['name'] for m in _list_maps()}:
        return JSONResponse({'error': f'no such map: {name}'}, status_code=404)

    args = ['bash', MAP_SESSION_SH, action]
    if action != 'new':
        args.append(name)
    # Carry the current perception setting across the restart, or switching a
    # map would silently turn the 3D tab and the object detector off.
    if action in ('switch', 'new') and ros_node and ros_node.perception_on():
        args.append('use_perception:=true')
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


@app.websocket('/vnc/{app_name}')
async def vnc_bridge(ws: WebSocket, app_name: str, t: str = ''):
    """noVNC <-> Xvnc. This is websockify, minus the extra daemon.

    noVNC opens the socket with the 'binary' subprotocol and then speaks raw
    RFB over it, so all we do is copy bytes both ways until either end hangs up.
    """
    if not _authorised(t, ws.cookies) or app_name not in VNC_PORTS:
        await ws.close(code=1008)      # policy violation
        return
    await ws.accept(subprotocol='binary')
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
            }))
        for msg in list(state.latest.values()):
            await ws.send_text(json.dumps(msg))
        for entry in list(state.chat)[-25:]:
            await ws.send_text(json.dumps({'type': 'chat', **entry}))
        for entry in list(state.logs)[-150:]:
            await ws.send_text(json.dumps({'type': 'log', **entry}))
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
    except Exception:
        pass
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
            .replace('__VNC_QS__', json.dumps(quote(token_qs, safe='')))
            .replace('__VNC_PASS__', json.dumps(VNC_PASSWORD))
            .replace('__ARM_LIMITS__', json.dumps(ARM_LIMITS))
            .replace('__ARM_JOINTS__', json.dumps(ARM_JOINTS))
            .replace('__HAS_NOVNC__', json.dumps(os.path.isdir(NOVNC_DIR)))
            .replace('__I18N__', json.dumps(as_js_table(), ensure_ascii=False))
            .replace('__LANGS__', json.dumps(LANGUAGES, ensure_ascii=False)))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
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
.badge{background:#1e3a5f;color:#67c4ff;padding:3px 11px;border-radius:11px;
  font-size:12px;white-space:nowrap}
#hdr-spacer{margin-left:auto}
#estop{background:#7f1d1d;border:1px solid #b91c1c;color:#fecaca;padding:7px 16px;
  border-radius:9px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap}
#estop.engaged{background:#dc2626;color:#fff;animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.55}}
/* ── Shell ── */
#shell{display:flex;height:calc(100vh - 48px)}
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
.card{background:#1c1c20;border:1px solid #2c2c32;border-radius:12px;padding:12px 14px}
.card h3{font-size:12px;font-weight:600;color:#8b8b93;text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:9px}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.btn{background:#2a2a31;border:1px solid #3a3a44;color:#d4d4d8;padding:8px 15px;
  border-radius:9px;cursor:pointer;font-size:13px;user-select:none;white-space:nowrap}
.btn:hover{background:#33333c}
.btn:active{background:#3d3d47}
.btn.pri{background:#1d4ed8;border-color:#2563eb;color:#fff}
.btn.warn{background:#7c2d12;border-color:#9a3412;color:#fed7aa}
.grid2{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px}
.k{color:#71717a}.v{font-family:ui-monospace,Menlo,monospace;text-align:right;color:#d4d4d8}
.pill{display:inline-block;padding:2px 9px;border-radius:9px;font-size:11px;
  background:#27272e;color:#a1a1aa}
.pill.ok{background:#052e1a;color:#4ade80}
.pill.bad{background:#450a0a;color:#f87171}
.pill.warn{background:#422006;color:#fbbf24}
/* ── Map ── */
#map-wrap{position:relative;background:#0c0c0e;border:1px solid #2c2c32;
  border-radius:12px;overflow:hidden;cursor:crosshair;flex:1;min-height:200px}
#map-canvas{width:100%;height:100%;display:block}
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
.vnc-msg{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:14px;text-align:center;padding:20px;
  color:#a1a1aa;font-size:13px;line-height:1.6}
/* ── Drive pad ── */
#dpad{display:grid;grid-template-columns:repeat(3,50px);grid-template-rows:repeat(3,50px);gap:5px}
.dbtn{background:#2a2a31;border:1px solid #3a3a44;border-radius:9px;display:flex;
  align-items:center;justify-content:center;cursor:pointer;font-size:19px;
  user-select:none;-webkit-user-select:none;touch-action:none}
.dbtn:active{background:#3d3d47}
.dbtn.ghost{background:transparent;border:none;pointer-events:none}
#bstop{background:#7f1d1d;border-color:#b91c1c;font-size:12px;font-weight:700;color:#fecaca}
/* ── Arm ── */
.joint{display:grid;grid-template-columns:74px 1fr 62px;gap:11px;align-items:center;
  margin-bottom:9px}
.joint label{font-size:13px;color:#a1a1aa}
.joint input[type=range]{width:100%;accent-color:#3b82f6}
.joint .val{font-family:ui-monospace,Menlo,monospace;font-size:12px;text-align:right;color:#d4d4d8}
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
  #tabs{width:100%;height:56px;display:flex;overflow-x:auto;padding:0;
    border-right:none;border-top:1px solid #2c2c32}
  .tab{flex-direction:column;gap:2px;padding:7px 13px;font-size:10px;
    border-left:none;border-top:3px solid transparent;justify-content:center}
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
    <section class="pane active" id="p-map">
      <div id="map-wrap">
        <canvas id="map-canvas"></canvas>
        <div class="ovl">ΧΑΡΤΗΣ · κλικ για πλοήγηση</div>
      </div>
      <div class="card">
        <h3>Δωμάτια</h3>
        <div class="row" id="rooms"></div>
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
      </div>
    </section>

    <!-- ── Camera ──────────────────────────────────────────────── -->
    <section class="pane" id="p-cam">
      <div id="cam-wrap">
        <img id="cam" alt="">
        <div class="ovl">📷 RealSense D435 · color</div>
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

    <!-- ── RViz / MoveIt / Gazebo ──────────────────────────────── -->
    <section class="pane" id="p-rviz"></section>
    <section class="pane" id="p-moveit"></section>
    <section class="pane" id="p-gazebo"></section>

    <!-- ── Arm ─────────────────────────────────────────────────── -->
    <section class="pane" id="p-arm">
      <div class="card">
        <h3>Αρθρώσεις</h3>
        <div id="joints"></div>
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

    <!-- ── Voice / LLM ─────────────────────────────────────────── -->
    <section class="pane" id="p-llm">
      <div id="chat"></div>
      <div id="chat-in">
        <input id="chat-text" placeholder="Γράψε στο ρομπότ…" autocomplete="off">
        <button class="btn pri" id="b-send">Στείλε</button>
        <button class="btn" id="b-say" title="Να το πει δυνατά χωρίς να το σκεφτεί">🔊</button>
      </div>
    </section>

    <!-- ── System ──────────────────────────────────────────────── -->
    <section class="pane" id="p-sys">
      <div class="card">
        <h3>Υπολογιστής</h3>
        <div class="grid2">
          <span class="k">CPU</span><span class="v" id="s-cpu">—</span>
          <span class="k">Μνήμη</span><span class="v" id="s-mem">—</span>
          <span class="k">Θερμοκρασία</span><span class="v" id="s-temp">—</span>
          <span class="k">Load</span><span class="v" id="s-load">—</span>
        </div>
      </div>
      <div class="card" style="flex:1;overflow:auto">
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

      <div class="card" style="flex:1;overflow:auto">
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
const VNC_QS     = __VNC_QS__;      // same, url-encoded for noVNC's ?path=
const VNC_PASS   = __VNC_PASS__;
const ARM_LIMITS = __ARM_LIMITS__;
const ARM_JOINTS = __ARM_JOINTS__;
const HAS_NOVNC  = __HAS_NOVNC__;
const LIN = 0.10, ANG = 0.10;
// ‼️ Must match bringup.launch.py's tf_base_laser: x=0, yaw=pi. This said 0.0
// and drew every scan point mirrored through the robot, so the dots never
// landed on the walls and the dashboard looked like a localization failure
// when localization was fine.
const LASER_X = 0.00, LASER_YAW_OFFSET = Math.PI;

// ── state ──────────────────────────────────────────────────────────────────
let ws=null, mapInfo=null, mapImg=null, pose=null, scan=null, goal=null, plan=null;
let driveTimer=null, vx=0, wz=0, estop=false, armPos={};
const $ = id => document.getElementById(id);

// ── tabs ───────────────────────────────────────────────────────────────────
const TABS = [
  ['map',    '🗺️', 'Χάρτης'],
  ['cam',    '📷', 'Κάμερα'],
  ['cloud',  '🧿', '3D'],
  ['rviz',   '🧊', 'RViz'],
  ['moveit', '🎯', 'MoveIt'],
  ['arm',    '🦾', 'Βραχίονας'],
  ['base',   '🧹', 'Σκούπα'],
  ['llm',    '💬', 'Φωνή/LLM'],
  ['gazebo', '🌍', 'Gazebo'],
  ['sys',    '📊', 'Σύστημα'],
  ['log',    '📜', 'Log'],
  ['set',    '⚙️', 'Ρυθμίσεις'],
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
  if (id === 'map') resize();
  if (VNC_APPS[id]) ensureVnc(id);
  cloudSetActive(id === 'cloud');
  if (id === 'set') mapsRefresh();
}
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
};
const vncState = {};
Object.keys(VNC_APPS).forEach(k => vncState[k] = {frame:null, busy:false});

function vncPane(app){ return $('p-' + app); }

function renderVnc(app, mode, detail){
  const pane = vncPane(app), meta = VNC_APPS[app], st = vncState[app];
  pane.innerHTML = '';
  const host = document.createElement('div');
  host.className = 'vnc-host';
  pane.appendChild(host);

  if (mode === 'live'){
    const f = document.createElement('iframe');
    // noVNC reads these from its query string: connect straight away, scale to
    // the iframe, and answer the VncAuth challenge without prompting (the page
    // itself is already behind the dashboard token).
    f.src = '/novnc/vnc.html?autoconnect=1&reconnect=1&resize=scale'
          + '&path=' + encodeURIComponent('vnc/' + app) + VNC_QS
          + '&password=' + encodeURIComponent(VNC_PASS);
    host.appendChild(f);
    st.frame = f;
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
  if (app !== 'rviz') mk(t('■ Τερματισμός'), 'warn', () => stopVnc(app));
  const s = document.createElement('span');
  s.className = 'pill ' + (mode === 'live' ? 'ok' : '');
  s.textContent = mode === 'live' ? t('ενεργό') : mode === 'starting' ? t('ξεκινά…') : t('σταματημένο');
  row.appendChild(s);
  card.appendChild(row);
  pane.appendChild(card);
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

// ── websocket ──────────────────────────────────────────────────────────────
const HANDLERS = {
  map(m){
    mapInfo={width:m.width,height:m.height,resolution:m.resolution,origin:m.origin};
    const i=new Image(); i.onload=()=>{mapImg=i;draw();}; i.src='data:image/png;base64,'+m.image;
  },
  pose(m){
    pose=m;
    $('ix').textContent   = m.x.toFixed(2)+' m';
    $('iy').textContent   = m.y.toFixed(2)+' m';
    $('iyaw').textContent = (m.yaw*180/Math.PI).toFixed(0)+'°';
    draw();
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
  arm(m){
    m.names.forEach((n,i)=>{
      armPos[n]=m.pos[i];
      const sl=$('j-'+n), v=$('jv-'+n);
      if(sl && !sl.dataset.dragging) sl.value=m.pos[i];
      if(v) v.textContent=(m.pos[i]*180/Math.PI).toFixed(0)+'°';
    });
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
    $('r-mode').innerHTML = `<span class="pill ${m.oi_mode===3||m.oi_mode===2?'ok':'warn'}">`
      + (OI[m.oi_mode] || '—') + `</span> <span style="color:#52525b">${t('ζητά')} ${m.oi_mode_want}</span>`;
    $('r-bump').innerHTML  = flag(m.bump, true);
    $('r-cliff').innerHTML = flag(m.cliff, true);
    $('r-wheel').innerHTML = flag(m.wheel_drop, true);
    $('r-motors').textContent = `${m.left_mm_s} / ${m.right_mm_s} mm/s`;
    $('r-dock').innerHTML  = flag(m.docking, false);
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
    $('s-cpu').textContent  = m.cpu.toFixed(0)+'%';
    $('s-mem').textContent  = m.mem+'% ('+m.mem_gb+' GB)';
    $('s-temp').textContent = m.temp!==null && m.temp!==undefined ? m.temp+'°C' : '—';
    $('s-load').textContent = m.load;
    $('s-nodecount').textContent = m.nodes.length;
    $('s-nodes').textContent = m.nodes.join('\n');
  },
};

function connect(){
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws${TOKEN_QS}`);
  // Re-arm the cloud stream on reconnect: the node tracks viewers per socket,
  // so after a restart (or a map switch) it has forgotten this tab is open.
  ws.onopen  = ()=>{ $('dot').classList.add('on');
                     if(cloudOn) send({type:'cloud', on:true}); };
  ws.onclose = ()=>{ $('dot').classList.remove('on'); setTimeout(connect,2000); };
  ws.onmessage = e=>{
    const m=JSON.parse(e.data);
    const h=HANDLERS[m.type];
    if(h) h(m);
  };
}
function send(o){ if(ws&&ws.readyState===1) ws.send(JSON.stringify(o)); }

// ── map click ──────────────────────────────────────────────────────────────
canvas.addEventListener('click',e=>{
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*canvas.width/r.width;
  const cy=(e.clientY-r.top)*canvas.height/r.height;
  const wp=c2w(cx,cy); if(!wp) return;
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
function bindDrive(id,v,w){
  const el=$(id);
  const go=()=>startDrive(v,w), stop=()=>stopDrive();
  el.addEventListener('mousedown',go);
  el.addEventListener('touchstart',e=>{e.preventDefault();go();},{passive:false});
  ['mouseup','mouseleave'].forEach(ev=>el.addEventListener(ev,stop));
  ['touchend','touchcancel'].forEach(ev=>el.addEventListener(ev,stop));
}
bindDrive('bf', LIN, 0); bindDrive('bb',-LIN, 0);
bindDrive('bl', 0, ANG); bindDrive('br', 0,-ANG);

$('bstop').addEventListener('click',()=>{ stopDrive(); send({type:'stop'}); });
$('b-loc').addEventListener('click',()=>send({type:'localize'}));
$('b-xnav').addEventListener('click',()=>{ goal=null; send({type:'stop'}); draw(); });

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
    send({type:'arm_joint',joint:name,pos:parseFloat(sl.value)});
  });
});
const grip=$('grip');
['mousedown','touchstart'].forEach(e=>grip.addEventListener(e,()=>grip.dataset.dragging='1'));
['mouseup','touchend','touchcancel'].forEach(e=>grip.addEventListener(e,()=>delete grip.dataset.dragging));
grip.addEventListener('input',()=>{
  $('grip-v').textContent=(grip.value*180/Math.PI).toFixed(0)+'°';
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
$('b-log-clear').onclick  = ()=>{ $('log-list').innerHTML=''; logSeen=0; $('log-count').textContent='0'; };
$('b-cloud-reset').onclick = cloudReset;
$('b-map-new').onclick  = mapNew;
$('b-map-save').onclick = mapSave;
cloudBind();

// ‼️ Order matters: collect the ORIGINAL Greek markup before anything writes
// translated text into the page, or the second language switch has nothing
// Greek left to look up.
i18nCollect();
for(const [code, name] of LANGS){
  const b = document.createElement('button');
  b.className = 'btn'; b.dataset.lang = code; b.textContent = name;
  b.onclick = () => setLang(code);
  $('lang-buttons').appendChild(b);
}
applyLang();

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
  for(const b of document.querySelectorAll('#lang-buttons .btn'))
    b.classList.toggle('pri', b.dataset.lang === LANG);
  if(document.querySelector('#p-set.active')) mapsRefresh();
}

function setLang(code){
  LANG = code;
  try { localStorage.setItem('hr_lang', code); } catch(e) {}
  applyLang();
}

// ── maps ───────────────────────────────────────────────────────────────────
// Switching a map restarts the whole stack: map_server takes its map as a
// launch parameter and there is no runtime swap. Hence the confirm() — losing
// navigation for ~90 s should never be one stray tap away.
async function mapsRefresh(){
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
    box.appendChild(row);
  }
}

function mapMsg(s){ $('map-msg').textContent = s; }

async function mapSwitch(name){
  if(!confirm(t('Θα σταματήσει η πλοήγηση και θα ξαναξεκινήσουν όλα με τον χάρτη')
              + ' "' + name + '".\n' + t('Διαρκεί περίπου 90 δευτερόλεπτα. Να συνεχίσω;'))) return;
  mapMsg(t('Επανεκκίνηση… η σελίδα θα ξανασυνδεθεί μόνη της.'));
  try { await fetch('/maps/switch/' + encodeURIComponent(name) + (TOKEN_QS || '')); }
  catch(e){ /* the server is going down under us; that IS the success case */ }
}

async function mapNew(){
  if(!confirm(t('Ξεκινά ΝΕΑ χαρτογράφηση (SLAM). Ο τρέχων χάρτης δεν χάνεται, '
             + 'αλλά η πλοήγηση σταματά μέχρι να αποθηκεύσεις τον καινούργιο. Να συνεχίσω;'))) return;
  mapMsg(t('Ξεκινά χαρτογράφηση… οδήγησε το ρομπότ σε όλο τον χώρο και μετά αποθήκευσε.'));
  try { await fetch('/maps/new/new' + (TOKEN_QS || '')); } catch(e){}
}

async function mapSave(){
  const name = $('map-save-name').value.trim();
  if(!/^[A-Za-z0-9_-]{1,40}$/.test(name)){
    mapMsg(t('Δώσε όνομα με λατινικά γράμματα, αριθμούς, - ή _')); return;
  }
  mapMsg(t('Αποθήκευση…'));
  try {
    const r = await (await fetch('/maps/save/' + encodeURIComponent(name)
                                 + (TOKEN_QS || ''))).json();
    mapMsg(r.ok ? t('Αποθηκεύτηκε') + ': ' + name : t('Απέτυχε') + ': ' + (r.result || ''));
    mapsRefresh();
  } catch(e){ mapMsg(t('Απέτυχε') + ': ' + e); }
}

// ── 3D point cloud ─────────────────────────────────────────────────────────
// Canvas 2D, not WebGL and not three.js: the page must stay self-contained
// (no CDN) and 4000 points sorted back-to-front is about a millisecond, so the
// extra machinery would buy nothing. Points arrive as int16 millimetres +
// RGB, in the camera optical frame: +x right, +y DOWN, +z forward.
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

// ── log tail ───────────────────────────────────────────────────────────────
// Levels are rcl_interfaces/Log: 30 WARN, 40 ERROR, 50 FATAL. Auto-scroll is
// suppressed when the user has scrolled up to read something — otherwise the
// next warning yanks the line they were reading off the screen.
const LOG_LEVELS = {30:['WARN','#facc15'], 40:['ERROR','#f87171'], 50:['FATAL','#f87171']};
let logSeen = 0;
function addLog(m){
  const list = $('log-list');
  const stick = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  const [name, colour] = LOG_LEVELS[m.level] || ['LOG', '#a1a1aa'];
  const d = document.createElement('div');
  const ts = new Date(m.t * 1000).toLocaleTimeString('el-GR');
  d.style.color = colour;
  d.textContent = `${ts}  ${name}  [${m.name}]  ${m.text}`;
  list.appendChild(d);
  while(list.children.length > 300) list.removeChild(list.firstChild);
  $('log-count').textContent = ++logSeen;
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
// Set in JS so the stream carries the same token as the page.
$('cam').src = '/camera.mjpeg' + TOKEN_QS;
showTab('map');
resize(); connect();
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
