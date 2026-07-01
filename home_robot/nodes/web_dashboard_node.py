#!/usr/bin/env python3
"""Web dashboard — RViz-like robot UI for browser and iPhone.

    ros2 run home_robot web_dashboard_node.py
    Open: http://192.168.178.62:8080
"""

import asyncio
import base64
import json
import math
import os
import threading
from typing import Optional, Set

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

PORT = 8080
_HERE = os.path.dirname(os.path.abspath(__file__))
LOCATIONS_FILE = os.path.join(_HERE, '..', 'config', 'locations.yaml')

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

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def add_client(self, ws: WebSocket):
        with self._lock: self._clients.add(ws)

    def remove_client(self, ws: WebSocket):
        with self._lock: self._clients.discard(ws)

    def broadcast(self, msg: dict):
        if self._loop is None:
            return
        data = json.dumps(msg)
        asyncio.run_coroutine_threadsafe(self._bcast(data), self._loop)

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

        latch = QoSProfile(depth=1,
                           durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                           reliability=QoSReliabilityPolicy.RELIABLE)

        self.create_subscription(OccupancyGrid, '/map', self._cb_map, latch)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._cb_pose, 10)
        self.create_subscription(LaserScan, '/scan', self._cb_scan, 5)
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._cb_camera, 5)
        self.create_subscription(String, '/current_room', self._cb_room, 10)

        self._vel_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self._loc_client = self.create_client(Empty, '/localize_globally')

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
        })

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

    def _cb_room(self, msg: String):
        self._state.broadcast({'type': 'room', 'name': msg.data})

    # ── Handle frontend messages ─────────────────────────────────────────────

    def dispatch(self, msg: dict):
        t = msg.get('type')
        if t == 'cmd_vel':
            tw = Twist()
            tw.linear.x  = float(msg.get('vx', 0))
            tw.angular.z = float(msg.get('wz', 0))
            self._vel_pub.publish(tw)
        elif t == 'stop':
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
        elif t == 'localize':
            if self._loc_client.service_is_ready():
                self._loc_client.call_async(Empty.Request())


# ── HTML frontend ──────────────────────────────────────────────────────────────

def _make_html(rooms: list) -> str:
    rooms_json = json.dumps(rooms)
    return f"""<!DOCTYPE html>
<html lang="el">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Home Robot</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}}
html,body{{height:100%;overflow:hidden;background:#1a1a1a;color:#e0e0e0;
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif}}
/* ── Header ── */
#hdr{{display:flex;align-items:center;gap:10px;padding:8px 14px;
  background:#252525;border-bottom:1px solid #333;flex-shrink:0;height:46px}}
#dot{{width:9px;height:9px;border-radius:50%;background:#444;transition:.3s}}
#dot.on{{background:#00ff88;box-shadow:0 0 7px #00ff88}}
#title{{font-size:15px;font-weight:600;letter-spacing:.3px}}
#room-badge{{margin-left:auto;background:#1e3a5f;color:#5bc0ff;
  padding:3px 11px;border-radius:12px;font-size:12px;white-space:nowrap}}
/* ── Layout ── */
#app{{display:grid;height:calc(100vh - 46px);
  grid-template-columns:1fr 300px;
  grid-template-rows:1fr auto auto;
  gap:7px;padding:7px}}
/* ── Map ── */
#map-wrap{{position:relative;background:#0d0d0d;border:1px solid #333;
  border-radius:10px;overflow:hidden;cursor:crosshair;grid-row:1;grid-column:1}}
#map-canvas{{width:100%;height:100%;display:block}}
#map-lbl{{position:absolute;top:8px;left:10px;font-size:10px;color:#555;
  background:rgba(0,0,0,.5);padding:2px 7px;border-radius:4px;pointer-events:none}}
#nav-lbl{{position:absolute;bottom:8px;right:10px;font-size:11px;color:#ffa040;
  background:rgba(0,0,0,.65);padding:3px 8px;border-radius:5px;display:none}}
/* ── Right ── */
#right{{grid-row:1;grid-column:2;display:flex;flex-direction:column;gap:7px}}
#cam-wrap{{flex:1;position:relative;background:#0d0d0d;border:1px solid #333;
  border-radius:10px;overflow:hidden}}
#cam{{width:100%;height:100%;object-fit:cover;display:block}}
#cam-lbl{{position:absolute;top:6px;left:9px;font-size:10px;color:#555;
  background:rgba(0,0,0,.5);padding:2px 7px;border-radius:4px}}
#info{{background:#222;border:1px solid #333;border-radius:10px;padding:11px 13px;
  display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:12px}}
.il{{color:#666}} .iv{{font-family:monospace;text-align:right;color:#ccc}}
/* ── Rooms ── */
#rooms{{grid-row:2;grid-column:1/-1;display:flex;gap:6px;flex-wrap:wrap}}
.rbtn{{background:#1c3a5f;border:1px solid #2a5a9f;color:#8ec8ff;
  padding:6px 15px;border-radius:18px;cursor:pointer;font-size:13px;
  user-select:none;transition:.12s}}
.rbtn:active{{background:#0e2540;transform:scale(.96)}}
/* ── Controls ── */
#ctrl{{grid-row:3;grid-column:1/-1;display:flex;align-items:center;
  justify-content:center;gap:22px;padding:2px 0}}
#dpad{{display:grid;grid-template-columns:repeat(3,48px);
  grid-template-rows:repeat(3,48px);gap:5px}}
.dbtn{{background:#2a2a2a;border:1px solid #444;color:#ccc;border-radius:9px;
  display:flex;align-items:center;justify-content:center;cursor:pointer;
  font-size:20px;user-select:none;-webkit-user-select:none;touch-action:none}}
.dbtn:active{{background:#3a3a3a}}
.dbtn.ghost{{background:transparent;border:none;pointer-events:none}}
#btn-stop{{background:#5c1c1c;border-color:#8c2c2c;font-size:13px;font-weight:700}}
#btn-stop:active{{background:#7c2c2c}}
#actions{{display:flex;flex-direction:column;gap:7px}}
.abtn{{background:#2a2a2a;border:1px solid #444;color:#ccc;padding:9px 16px;
  border-radius:9px;cursor:pointer;font-size:13px;text-align:center;
  user-select:none;white-space:nowrap}}
.abtn:active{{background:#3a3a3a}}
#btn-loc{{border-color:#0088cc;color:#44aaff}}
/* ── Mobile ── */
@media(max-width:700px){{
  #app{{grid-template-columns:1fr;grid-template-rows:180px 1fr auto auto}}
  #right{{grid-row:1;grid-column:1;flex-direction:row}}
  #info{{display:none}}
  #map-wrap{{grid-row:2;grid-column:1}}
}}
</style>
</head>
<body>
<div id="hdr">
  <div id="dot"></div>
  <span id="title">🤖 Home Robot</span>
  <span id="room-badge">—</span>
</div>
<div id="app">
  <div id="map-wrap">
    <canvas id="map-canvas"></canvas>
    <div id="map-lbl">MAP · click to navigate</div>
    <div id="nav-lbl" id="nav-lbl">Navigating…</div>
  </div>
  <div id="right">
    <div id="cam-wrap">
      <img id="cam" src="/camera.mjpeg" alt="">
      <div id="cam-lbl">📷 D435</div>
    </div>
    <div id="info">
      <span class="il">X</span><span class="iv" id="ix">—</span>
      <span class="il">Y</span><span class="iv" id="iy">—</span>
      <span class="il">Yaw</span><span class="iv" id="iyaw">—</span>
      <span class="il">Δωμάτιο</span><span class="iv" id="iroom">—</span>
    </div>
  </div>
  <div id="rooms"></div>
  <div id="ctrl">
    <div id="dpad">
      <div class="dbtn ghost"></div>
      <div class="dbtn" id="bf">▲</div>
      <div class="dbtn ghost"></div>
      <div class="dbtn" id="bl">◄</div>
      <div class="dbtn" id="btn-stop">STOP</div>
      <div class="dbtn" id="br">►</div>
      <div class="dbtn ghost"></div>
      <div class="dbtn" id="bb">▼</div>
      <div class="dbtn ghost"></div>
    </div>
    <div id="actions">
      <div class="abtn" id="btn-loc">🔍 Localize</div>
      <div class="abtn" id="btn-xnav">✕ Ακύρωση</div>
    </div>
  </div>
</div>
<script>
// ── constants ──────────────────────────────────────────────────────────────
const ROOMS = {rooms_json};
const LIN = 0.10, ANG = 0.10;
const LASER_X = 0.00, LASER_YAW_OFFSET = 0.0; // laser TF

// ── state ──────────────────────────────────────────────────────────────────
let ws=null, mapInfo=null, mapImg=null, pose=null, scan=null, goal=null;
let driveTimer=null, vx=0, wz=0;

// ── canvas ─────────────────────────────────────────────────────────────────
const canvas = document.getElementById('map-canvas');
const ctx    = canvas.getContext('2d');
const wrap   = document.getElementById('map-wrap');

function scale(){{ return mapInfo ? Math.min(canvas.width/mapInfo.width, canvas.height/mapInfo.height) : 1; }}
function offX(){{  return mapInfo ? (canvas.width  - mapInfo.width  * scale()) / 2 : 0; }}
function offY(){{  return mapInfo ? (canvas.height - mapInfo.height * scale()) / 2 : 0; }}

function w2c(wx, wy){{
  if(!mapInfo) return {{x:0,y:0}};
  const s=scale();
  return {{
    x: offX() + (wx - mapInfo.origin[0]) / mapInfo.resolution * s,
    y: offY() + (mapInfo.height - (wy - mapInfo.origin[1]) / mapInfo.resolution) * s,
  }};
}}
function c2w(cx, cy){{
  if(!mapInfo) return null;
  const s=scale();
  return {{
    x: mapInfo.origin[0] + (cx - offX()) / s * mapInfo.resolution,
    y: mapInfo.origin[1] + (mapInfo.height - (cy - offY()) / s) * mapInfo.resolution,
  }};
}}

function draw(){{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#0d0d0d';
  ctx.fillRect(0,0,canvas.width,canvas.height);
  if(!mapImg||!mapInfo) return;
  const s=scale(), ox=offX(), oy=offY();
  ctx.drawImage(mapImg, ox, oy, mapInfo.width*s, mapInfo.height*s);

  // LiDAR scan
  if(scan && pose){{
    ctx.fillStyle='rgba(0,200,255,0.75)';
    const laserYaw = pose.yaw + LASER_YAW_OFFSET;
    const lx = pose.x + LASER_X * Math.cos(pose.yaw);
    const ly = pose.y + LASER_X * Math.sin(pose.yaw);
    for(let i=0;i<scan.ranges.length;i++){{
      const r=scan.ranges[i];
      if(!r||r<0.1||r>8) continue;
      const a = laserYaw + scan.angle_min + i*scan.angle_inc;
      const p = w2c(lx + r*Math.cos(a), ly + r*Math.sin(a));
      ctx.fillRect(p.x-1.5, p.y-1.5, 3, 3);
    }}
  }}

  // Nav goal
  if(goal){{
    const g=w2c(goal.x,goal.y);
    ctx.strokeStyle='#ffa040'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(g.x,g.y,10,0,Math.PI*2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(g.x,g.y-15); ctx.lineTo(g.x,g.y+15);
    ctx.moveTo(g.x-15,g.y); ctx.lineTo(g.x+15,g.y);
    ctx.stroke();
  }}

  // Robot arrow
  if(pose){{
    const rp=w2c(pose.x,pose.y);
    ctx.save();
    ctx.translate(rp.x,rp.y);
    ctx.rotate(-pose.yaw);
    ctx.fillStyle='#00ff88'; ctx.strokeStyle='#003';  ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(17,0); ctx.lineTo(-9,10); ctx.lineTo(-5,0); ctx.lineTo(-9,-10);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }}
}}

function resize(){{
  canvas.width=wrap.clientWidth; canvas.height=wrap.clientHeight; draw();
}}
window.addEventListener('resize',resize);
new ResizeObserver(resize).observe(wrap);

// ── websocket ──────────────────────────────────────────────────────────────
function connect(){{
  ws = new WebSocket(`ws://${{location.host}}/ws`);
  ws.onopen  = ()=> document.getElementById('dot').classList.add('on');
  ws.onclose = ()=>{{ document.getElementById('dot').classList.remove('on'); setTimeout(connect,2000); }};
  ws.onmessage = e=>{{
    const m=JSON.parse(e.data);
    if(m.type==='map'){{
      mapInfo={{width:m.width,height:m.height,resolution:m.resolution,origin:m.origin}};
      const i=new Image(); i.onload=()=>{{mapImg=i;draw();}}; i.src='data:image/png;base64,'+m.image;
    }} else if(m.type==='pose'){{
      pose=m;
      document.getElementById('ix').textContent   = m.x.toFixed(2)+'m';
      document.getElementById('iy').textContent   = m.y.toFixed(2)+'m';
      document.getElementById('iyaw').textContent = (m.yaw*180/Math.PI).toFixed(0)+'°';
      draw();
    }} else if(m.type==='scan'){{
      scan=m; draw();
    }} else if(m.type==='room'){{
      document.getElementById('room-badge').textContent = m.name||'—';
      document.getElementById('iroom').textContent      = m.name||'—';
    }}
  }};
}}
function send(o){{ if(ws&&ws.readyState===1) ws.send(JSON.stringify(o)); }}

// ── map click ──────────────────────────────────────────────────────────────
canvas.addEventListener('click',e=>{{
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*canvas.width/r.width;
  const cy=(e.clientY-r.top)*canvas.height/r.height;
  const wp=c2w(cx,cy); if(!wp) return;
  goal=wp; send({{type:'nav_goal',x:wp.x,y:wp.y}});
  document.getElementById('nav-lbl').style.display='block';
  draw();
}});

// ── room buttons ───────────────────────────────────────────────────────────
const rdiv=document.getElementById('rooms');
ROOMS.forEach(name=>{{
  const b=document.createElement('button');
  b.className='rbtn'; b.textContent=name;
  b.onclick=()=>{{ send({{type:'goto_room',room:name}}); goal=null; draw(); }};
  rdiv.appendChild(b);
}});

// ── drive controls ─────────────────────────────────────────────────────────
function startDrive(v,w){{
  vx=v; wz=w;
  if(!driveTimer) driveTimer=setInterval(()=>send({{type:'cmd_vel',vx,wz}}),100);
  send({{type:'cmd_vel',vx:v,wz:w}});
}}
function stopDrive(){{
  clearInterval(driveTimer); driveTimer=null; vx=0; wz=0;
  send({{type:'cmd_vel',vx:0,wz:0}});
}}
function bindDrive(id,v,w){{
  const el=document.getElementById(id);
  const go=()=>startDrive(v,w), stop=()=>stopDrive();
  el.addEventListener('mousedown',go);
  el.addEventListener('touchstart',e=>{{e.preventDefault();go();}},{{passive:false}});
  ['mouseup','mouseleave'].forEach(ev=>el.addEventListener(ev,stop));
  ['touchend','touchcancel'].forEach(ev=>el.addEventListener(ev,stop));
}}
bindDrive('bf', LIN, 0); bindDrive('bb',-LIN, 0);
bindDrive('bl', 0, ANG); bindDrive('br', 0,-ANG);

document.getElementById('btn-stop').addEventListener('click',()=>{{
  stopDrive(); send({{type:'stop'}});
}});
document.getElementById('btn-loc').addEventListener('click',()=>send({{type:'localize'}}));
document.getElementById('btn-xnav').addEventListener('click',()=>{{
  goal=null; send({{type:'stop'}}); draw();
  document.getElementById('nav-lbl').style.display='none';
}});

// ── start ──────────────────────────────────────────────────────────────────
resize(); connect();
</script>
</body>
</html>"""


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

@app.get('/', response_class=HTMLResponse)
async def index():
    return HTMLResponse(_make_html(list(locations.keys())))

@app.get('/camera.mjpeg')
async def camera():
    async def stream():
        while True:
            jpg = state.camera_jpg
            if jpg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                       + jpg + b'\r\n')
            await asyncio.sleep(0.04)   # ~25 fps cap
    return StreamingResponse(stream(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.add_client(ws)

    # Send current map to new client immediately
    if state.map_png and state.map_info:
        await ws.send_text(json.dumps({
            'type':  'map',
            **state.map_info,
            'image': base64.b64encode(state.map_png).decode(),
        }))

    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)
            if ros_node:
                ros_node.dispatch(msg)
    except WebSocketDisconnect:
        pass
    finally:
        state.remove_client(ws)


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

    print(f'\n🤖  Web dashboard → http://192.168.178.62:{PORT}\n')
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')


if __name__ == '__main__':
    main()
