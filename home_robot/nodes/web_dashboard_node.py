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
import struct
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
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rcl_interfaces.msg import Log as RosoutLog
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image, Imu, JointState, LaserScan, PointCloud2
# ‼️ std_msgs/Empty (a message) and std_srvs/Empty (a service) collide by name,
# and this file needs both — the localize/rtabmap service clients and the
# gesture_go publisher. Aliasing the message keeps the pre-existing `Empty` as
# the service, so none of the service call sites below change meaning.
from std_msgs.msg import Bool, Float32, String
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty

from home_robot.dashboard_i18n import LANGUAGES, as_js_table

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Overridable so a second instance can be brought up beside the live one for
# testing without taking the robot's dashboard off 8080.
PORT = int(os.environ.get('HOME_ROBOT_DASHBOARD_PORT', '8080'))
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

# Must match the display map in scripts/gui_session.sh.
VNC_PORTS = {'rviz': 5902, 'gazebo': 5903, 'moveit': 5904, 'rtabmap': 5905}
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
        # Room legend + tint state, replayed with the map on connect.
        self.map_rooms: dict = {}
        self.map_tinted: bool = True
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
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._cb_camera, 5)
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
        self._cloud_last = 0.0
        # The camera ships with its pointcloud filter off; _set_camera_pointcloud
        # turns it on for as long as someone is on the 3D tab.
        self._cloud_param_on = False
        self._cam_param_cli = self.create_client(
            SetParameters, '/camera/camera/set_parameters')
        self._map_cache = None          # (fetched_at, name); see active_map()
        self._backend_cache = None      # (fetched_at, backend); see llm_backend()
        self._cam_err_at = 0.0          # throttles the decode-failure warning
        self._room_mask = (None, None, None)   # see _load_room_mask()
        self._room_mask_at = None              # mtime the cache was built from
        self._room_size_warned = False
        self._rooms_tinted = True              # toggled from the map tab
        self._last_map = None                  # so the toggle can redraw
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
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._cb_costmap, 1)
        self._costmap_on: Set[object] = set()   # sockets watching the tab
        self.create_subscription(PointCloud2, '/semantic_obstacles',
                                 self._cb_semantic,
                                 QoSProfile(depth=1,
                                            reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self._semantic_pts: list = []
        self._semantic_time = 0.0

        # ── RTAB-Map (3D map of the house) ──────────────────────────────────
        # Only ever live while the «3D Χάρτης» tab has its session up; the rest
        # of the time these topics simply have no publisher, which is exactly
        # what _rtab_health reports.
        self._rtab_seen  = 0.0    # monotonic stamp of the last /rtabmap/info
        self._rtab_nodes = 0      # keyframes in the graph
        self._rtab_loops = 0      # loop closures accepted so far
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
        self.create_subscription(String, '/gesture_status', self._cb_gesture, 10)
        self.create_subscription(String, '/observations', self._cb_observations, 10)
        self.create_subscription(String, '/vocab/state', self._cb_vocab, latch)
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
        # The gesture "go" button and the timeline search box. Same topics the
        # `goto_pointed` and `recall` voice tools use, so button and voice take
        # one code path.
        self._gesture_go_pub = self.create_publisher(EmptyMsg, "/gesture_go", 10)
        self._recall_pub = self.create_publisher(String, '/episodic/query', 10)
        self._vocab_pub = self.create_publisher(String, '/vocab/set', 10)
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

        self._loc_client = self.create_client(Empty, '/localize_globally')
        # Created on first use: the services only exist while a mapping session
        # is running, and a client made against a missing service is harmless
        # but pointless to hold for the whole life of the dashboard.
        self._rtab_clients: dict = {}

        self.create_timer(2.0, self._publish_system)

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _cb_map(self, msg: OccupancyGrid):
        self._last_map = msg          # kept so the room-tint toggle can redraw
        grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        img = np.full((msg.info.height, msg.info.width, 3), 180, dtype=np.uint8)
        img[grid == 0]   = [230, 230, 230]
        img[grid == 100] = [50,  50,  50]
        img[grid == -1]  = [160, 160, 160]
        img = cv2.flip(img, 0)
        # Tint the free space with each room's colour, so "πήγαινε στην κουζίνα"
        # can be checked against something visible instead of a uniform grey
        # field. Only free cells are tinted: walls stay black or the map stops
        # reading as a floor plan.
        img = self._tint_rooms(img, cv2.flip(grid, 0))
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
        # Drives the legend. Stashed on State rather than only broadcast, so
        # the replay a newly-connected browser gets carries it too — sent only
        # here, every fresh tab showed an empty legend and an unchecked
        # "Χρώματα" box over a map that was in fact tinted.
        _, _, names = self._load_room_mask()
        self._state.map_rooms = {n: list(c) for n, c in (names or {}).items()}
        self._state.map_tinted = self._rooms_tinted
        self._state.broadcast({
            'type':  'map',
            **info,
            'image': base64.b64encode(self._state.map_png).decode(),
            'rooms': self._state.map_rooms,
            'tinted': self._state.map_tinted,
        }, remember=False)   # replayed from map_png on connect, not from latest

    # How strongly the room colour is mixed into the floor. Low on purpose: the
    # map has to stay readable as a map — obstacles, the plan and the laser are
    # all drawn over it — so this is a wash, not a fill.
    ROOM_TINT = 0.38

    def _load_room_mask(self):
        """maps/room_mask.png as BGR + its per-room alpha, cached.

        Returns (bgr, alpha, names) or (None, None, None). The mask is the same
        one situational_awareness and room_markers read, so what the dashboard
        paints and what the robot calls the room can never disagree.

        Reloaded when the file changes on disk, so regenerating it after a
        remap shows up without restarting the dashboard.
        """
        path = os.path.join(SRC_MAPS_DIR, 'room_mask.png')
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            self._room_mask = (None, None, None)
            return self._room_mask
        if self._room_mask_at == stamp:
            return self._room_mask
        try:
            from PIL import Image
            arr = np.array(Image.open(path).convert('RGBA'))
            names = {}
            colours_path = os.path.join(SRC_MAPS_DIR, 'room_colors.yaml')
            if os.path.exists(colours_path):
                with open(colours_path) as f:
                    names = yaml.safe_load(f) or {}
            rgb = arr[:, :, :3].astype(np.float32)
            bgr = rgb[:, :, ::-1]
            alpha = (arr[:, :, 3] > 50)
            self._room_mask = (bgr, alpha, names)
        except Exception as e:
            self.get_logger().warn(f'room_mask unusable: {e}')
            self._room_mask = (None, None, None)
        self._room_mask_at = stamp
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

    def _cb_semantic(self, msg: PointCloud2):
        """Obstacles the DETECTOR put into the costmap, not the LiDAR.

        semantic_costmap_node inflates people and animals into a cylinder wider
        than their actual footprint so Nav2 keeps a margin, and that is invisible
        in every other view: on the map the robot just appears to give someone a
        strangely wide berth. Only the outline is needed here, so the cloud is
        thinned hard — it is drawn as a scatter over a 3 m window.
        """
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
            return                      # nobody is on the tab; skip the encode
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
        self._publish_pose(p.position.x, p.position.y, yaw)

    def _publish_pose(self, x: float, y: float, yaw: float):
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
        wz = msg.twist.twist.angular.z
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

    def _cb_camera(self, msg: Image):
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

    # Roughly what a phone on wifi can take: 4000 points x 9 bytes is ~36 kB,
    # ~48 kB once base64'd, three times a second.
    CLOUD_MAX_POINTS = 4000
    CLOUD_PERIOD = 0.33

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
        # A tab closed on the 3D pane never sends its 'off', so the camera would
        # keep building pointclouds for nobody.
        self._set_camera_pointcloud(bool(self._cloud_ws))
        self._costmap_on.discard(client)

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
                self._set_camera_pointcloud(bool(self._cloud_ws))
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
        elif t == 'set_backend':
            threading.Thread(target=self._switch_backend,
                             args=(str(msg.get('backend', '')),),
                             daemon=True).start()
        elif t == 'nerf_capture':
            self._nerf_pub.publish(Bool(data=bool(msg.get('on'))))
        elif t == 'gesture_go':
            # gesture_node holds the point; it re-checks that one exists, so an
            # eager click before any gesture is a logged warning, not a goal.
            self._gesture_go_pub.publish(EmptyMsg())
        elif t == 'recall':
            self._recall_pub.publish(String(data=str(msg.get('when', ''))))
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
            .replace('__ARM_LIMITS__', json.dumps(ARM_LIMITS))
            .replace('__ARM_JOINTS__', json.dumps(ARM_JOINTS))
            .replace('__HAS_NOVNC__', json.dumps(os.path.isdir(NOVNC_DIR)))
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
.badge{background:#1e3a5f;color:#67c4ff;padding:3px 11px;border-radius:11px;
  font-size:12px;white-space:nowrap}
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
#cost-wrap{position:relative;background:#0c0c0e;border:1px solid #2c2c32;
  border-radius:12px;overflow:hidden;flex:1;min-height:200px}
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
     below that the icon and the label collide. 16.66% is 53pt there, and six
     per row is a comfortable count to scan. */
  /* The home indicator sits over the last row on a notched iPhone; without the
     inset the bottom row's labels are half-covered by it. */
  #tabs{width:100%;height:auto;display:flex;flex-wrap:wrap;padding:0;
    border-right:none;border-top:1px solid #2c2c32;
    padding-bottom:env(safe-area-inset-bottom,0px)}
  .tab{flex:0 0 16.66%;flex-direction:column;gap:2px;padding:6px 2px;
    font-size:9.5px;border-left:none;border-top:3px solid transparent;
    justify-content:center;text-align:center;min-width:0}
  /* Long labels (Βραχίονας, Ρυθμίσεις) must shrink, not widen the cell and
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
      <div id="map-wrap">
        <canvas id="map-canvas"></canvas>
        <div class="ovl">ΧΑΡΤΗΣ · κλικ για πλοήγηση</div>
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
        <canvas id="arm3d" style="width:100%;height:300px;background:#0a0a0b;
          border-radius:8px;touch-action:none;cursor:grab;display:block"></canvas>
        <div style="color:#71717a;font-size:11.5px;margin-top:6px">
          Σύρε για περιστροφή · ροδέλα για ζουμ · κινείται με τις αρθρώσεις από κάτω
        </div>
      </div>
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
        <h3>Εκπαίδευση</h3>
        <p style="font-size:12px;color:#a1a1aa;line-height:1.7">
          Η εκπαίδευση τρέχει ΞΕΧΩΡΙΣΤΑ, από τερματικό:<br>
          <code>scripts/train_nerf.py --data ~/.home_robot/nerf/house --steps 8000</code>
        </p>
        <p style="font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6">
          ‼️ ΔΕΝ μπορεί να μοιραστεί το iGPU με το perception. Με το object_detector και το pose_node ενεργά, η εκπαίδευση ΡΙΧΝΕΙ την ουρά του ROCm (memory aperture violation) — δεν είναι έλλειψη μνήμης, μετρήθηκαν 5.9GB ελεύθερα. Σταμάτα πρώτα το perception· το script αρνείται να ξεκινήσει και μόνο του. Μετρημένη ταχύτητα: ~310ms/βήμα, άρα ένα δωμάτιο θέλει 45 λεπτά έως 2 ώρες.
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
          </div>
        </div>
        <div id="sd-cands" style="font-size:11.5px;color:#a1a1aa;margin-top:10px;
          line-height:1.6"></div>
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
      <div class="card">
        <h3>Υπολογιστής</h3>
        <div id="s-bars"></div>
      </div>
      <div class="card">
        <h3>Θερμοκρασίες</h3>
        <div id="s-temps" class="tempgrid">—</div>
      </div>
      <div class="card">
        <h3>Δίσκοι</h3>
        <div id="s-disks">—</div>
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
const ARM_LIMITS = __ARM_LIMITS__;
const ARM_JOINTS = __ARM_JOINTS__;
const HAS_NOVNC  = __HAS_NOVNC__;
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
  ['arm',    '🦾', 'Βραχίονας'],
  ['base',   '🧹', 'Σκούπα'],
  ['imu',    '🧭', 'IMU'],
  ['rtabmap','🏠', '3D Χάρτης'],
  ['cost',   '🧱', 'Costmap'],
  ['nerf',   '✨', 'NeRF'],
  ['point',  '👉', 'Χειρονομίες'],
  ['vocab',  '🔎', 'Αναζήτηση'],
  ['sound',  '👂', 'Ήχοι'],
  ['obs',    '💡', 'Παρατηρήσεις'],
  ['time',   '🕐', 'Χρονολόγιο'],
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
  costSetActive(id === 'cost');
  if (id === 'cost') buildCostLegend();
  if (id === 'set') mapsRefresh();
  // 170 kB of geometry, fetched the first time the tab is opened rather than
  // on every page load — most visits never look at the arm.
  if (id === 'arm'){ armLoad(); armDraw(); }
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

  if (app === 'rtabmap') pane.appendChild(rtabControls());
}

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
  card.appendChild(row);
  const p = document.createElement('p');
  p.style.cssText = 'font-size:11.5px;color:#71717a;margin-top:10px;line-height:1.6';
  p.textContent = t('Ο χάρτης αποθηκεύεται μόνος του στο ~/.home_robot/rtabmap/house.db. Για εξαγωγή σε .ply/.obj χρησιμόποιησε File → Export στο ίδιο το RTAB-Map παραπάνω. ‼️ Το «Νέος χάρτης» ξεκινά καθαρή συνεδρία — ό,τι έχεις χαρτογραφήσει ως τώρα μένει στη βάση αλλά βγαίνει από τον τρέχοντα χάρτη.');
  card.appendChild(p);
  return card;
}

function onRtab(m){
  const st = $('rt-state'); if (!st) return;   // tab not built yet
  st.className = 'pill' + (m.live ? ' ok' : '');
  st.textContent = m.live ? t('χαρτογραφεί') : t('ανενεργό');
  $('rt-nodes').textContent = m.live ? m.nodes : '—';
  $('rt-loops').textContent = m.live ? m.loops : '—';
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
    if (m.rooms) roomLegend(m.rooms);
    if (m.tinted !== undefined) $('b-tint').checked = m.tinted;
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
  gesture(m){ gestureState = m; drawPointRing(); },
  vocab(m){ renderVocab(m); },
  sound(m){ soundState = m; renderSound(m); },
  observations(m){ renderObservations(m); },
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
  },
  llm_backend(m){ onBackend(m); },
  mission(m){ onMission(m); },
  fall(m){ onFall(m); },
  fall_event(m){ onFallEvent(m); },
  speaker(m){ onSpeaker(m); },
};

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
                     if(!overlayOn) send({type:'overlay', on:false}); };
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
$('b-tint').onchange = e => send({type:'room_tint', on: e.target.checked});

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
$('b-follow').addEventListener('click',()=>send({type:'follow', on:true}));
$('b-follow-stop').addEventListener('click',()=>send({type:'follow', on:false}));

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
['κλειδιά','γυαλιά','φορτιστής','τηλεκοντρόλ','πορτοφόλι'].forEach(q=>{
  const b=document.createElement('button');
  b.className='btn'; b.textContent=t(q);
  b.style.fontSize='11.5px'; b.style.padding='5px 9px';
  b.onclick=()=>{ $('vc-q').value=t(q); askVocab(q); };
  $('vc-chips').appendChild(b);
});

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
['σήμερα','σήμερα το πρωί','χθες','πριν από 2 ώρες'].forEach(q=>{
  const b=document.createElement('button');
  b.className='btn'; b.textContent=t(q);
  b.style.fontSize='11.5px'; b.style.padding='5px 9px';
  b.onclick=()=>{ $('tl-q').value=t(q); askRecall(q); };
  $('tl-chips').appendChild(b);
});

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
$('b-log-clear').onclick  = ()=>{ $('log-list').innerHTML=''; logSeen=0; $('log-count').textContent='0'; };
$('b-cloud-reset').onclick = cloudReset;
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
  // Camera: orbit yaw/pitch, orthographic. Arm is ~400 mm tall.
  const cy = Math.cos(armYaw), sy = Math.sin(armYaw);
  const cp = Math.cos(armPitch), sp = Math.sin(armPitch);
  const scale = Math.min(w, h) / 520 * armZoom * 1.35;
  const ox = w/2, oy = h*0.72;

  const faces = [];
  for (const [link, flat] of Object.entries(armModel.links)){
    const m = tf[link]; if (!m) continue;
    for (let i=0;i<flat.length;i+=9){
      const p = [];
      let depth = 0;
      for (let k=0;k<3;k++){
        // mm -> m, into world, then into view space.
        const v = m4apply(m, [flat[i+k*3]/1000, flat[i+k*3+1]/1000, flat[i+k*3+2]/1000]);
        const xr =  v[0]*cy + v[1]*sy;
        const yr = -v[0]*sy + v[1]*cy;
        const zr =  v[2];
        const ys = yr*sp + zr*cp;
        depth += yr*cp - zr*sp;
        p.push([ox + xr*scale, oy - ys*scale]);
      }
      faces.push([depth/3, p, link]);
    }
  }
  // Painter's algorithm: far to near. With a convex-ish arm this is enough,
  // and it costs one sort instead of a depth buffer.
  faces.sort((a,b) => a[0]-b[0]);

  const zs = faces.length ? [faces[0][0], faces[faces.length-1][0]] : [0,1];
  const span = (zs[1]-zs[0]) || 1;
  for (const [d, p, link] of faces){
    // Cheap depth shading — nearer is brighter. Real lighting would need
    // normals through the transform; this reads as solid and costs nothing.
    const tshade = (d - zs[0]) / span;
    const base = link.startsWith('gripper') ? [90,150,230] : [150,155,165];
    const f = 0.45 + 0.55*tshade;
    ctx.fillStyle = `rgb(${base[0]*f|0},${base[1]*f|0},${base[2]*f|0})`;
    ctx.beginPath();
    ctx.moveTo(p[0][0], p[0][1]);
    ctx.lineTo(p[1][0], p[1][1]);
    ctx.lineTo(p[2][0], p[2][1]);
    ctx.closePath();
    ctx.fill();
  }

  // Ground line, so the arm does not float in nothing.
  ctx.strokeStyle = '#27272e'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(w, oy); ctx.stroke();
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
applyLang();               // also builds the tabs, so it precedes showTab()

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
