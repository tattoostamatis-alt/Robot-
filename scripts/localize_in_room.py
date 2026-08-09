#!/usr/bin/env python3
"""Find the robot's pose by searching ONE room exhaustively.

The global localizer searches the whole map, and in this flat several rooms
fit the same scan almost equally well (0.176 vs 0.181 on the last boot) — so
it can land in the wrong room and its watchdog then re-lands somewhere else
every few minutes. When a human can say WHICH room the robot is in, that
ambiguity disappears: search only there.

    localize_in_room.py toualeta            # dry run, prints the best pose
    localize_in_room.py toualeta --set      # also publish /initialpose
"""
import argparse, math, sys, time
import numpy as np
import rclpy, yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from ament_index_python.packages import get_package_share_directory
from scipy.ndimage import distance_transform_edt

LATCH = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                   reliability=QoSReliabilityPolicy.RELIABLE,
                   history=QoSHistoryPolicy.KEEP_LAST)

ap = argparse.ArgumentParser()
ap.add_argument('room')
ap.add_argument('--radius', type=float, default=1.2, help='search radius, m')
ap.add_argument('--set', action='store_true', help='publish /initialpose')
args = ap.parse_args()


class L(Node):
    def __init__(self):
        super().__init__('localize_in_room')
        self.m = self.scan = None
        self.create_subscription(OccupancyGrid, '/map', self._m, LATCH)
        self.create_subscription(LaserScan, '/scan', self._s, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    def _m(self, m): self.m = m
    def _s(self, m): self.scan = m


rclpy.init(); n = L()
for _ in range(300):
    rclpy.spin_once(n, timeout_sec=0.05)
    if n.m and n.scan:
        break

locs = yaml.safe_load(open(get_package_share_directory('home_robot') + '/config/locations.yaml'))
if args.room not in locs:
    print(f'άγνωστο δωμάτιο: {args.room}  ({list(locs)})'); sys.exit(1)
cx, cy = locs[args.room]['x'], locs[args.room]['y']

mp = n.m
res = mp.info.resolution
ox, oy = mp.info.origin.position.x, mp.info.origin.position.y
grid = np.array(mp.data, dtype=np.int8).reshape(mp.info.height, mp.info.width)
free = grid >= 0
wall = grid >= 65
dist = distance_transform_edt(~wall) * res          # m to nearest wall
# cells the robot could stand in: known free and >= robot radius from a wall
standable = free & (grid < 65) & (dist >= 0.175)

# live scan points in LASER frame
s = n.scan
ang = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
rng = np.array(s.ranges, dtype=np.float64)
ok = np.isfinite(rng) & (rng >= s.range_min) & (rng <= min(s.range_max, 5.0))
ang, rng = ang[ok], rng[ok]
px, py = rng * np.cos(ang), rng * np.sin(ang)
print(f'{len(px)} σημεία lidar (<5m)')


def endpoints(lx, ly, lyaw):
    c, s_ = math.cos(lyaw), math.sin(lyaw)
    return lx + c*px - s_*py, ly + s_*px + c*py


def fit(lx, ly, lyaw):
    """median scan->wall distance for a LASER pose. Vectorised; this runs
    once per candidate in the sweep, so it must stay cheap."""
    wx, wy = endpoints(lx, ly, lyaw)
    i = ((wx-ox)/res).astype(int)
    j = ((wy-oy)/res).astype(int)
    inb = (i >= 0) & (i < mp.info.width) & (j >= 0) & (j < mp.info.height)
    if inb.sum() < 30:
        return 9.9
    return float(np.median(dist[j[inb], i[inb]]))


def blocked_frac(lx, ly, lyaw, nbeams=180, slack=0.30):
    """Fraction of beams the map says should have stopped much earlier.

    ‼️ Compare DISTANCES, not "did the ray touch a wall". The map has furniture
    drawn into it, so plain contact flags 50-90% of beams from a perfectly good
    pose. What actually separates the right spot from a same-shaped one
    elsewhere is a beam that reads 3 m while the map has a wall at 0.5 m along
    that bearing. `slack` is how much further the real reading may go before it
    counts as seeing through a wall."""
    step = max(1, len(rng)//nbeams)
    r = rng[::step]
    beam_ang = lyaw + ang[::step]
    # march every beam together, one map cell at a time
    steps = int(min(r.max(), 5.0)/res) + 1
    ts = (np.arange(1, steps+1) * res)[:, None]          # (S,1) distance along
    xs = lx + ts*np.cos(beam_ang)[None, :]
    ys = ly + ts*np.sin(beam_ang)[None, :]
    i = ((xs-ox)/res).astype(int)
    j = ((ys-oy)/res).astype(int)
    inb = (i >= 0) & (i < mp.info.width) & (j >= 0) & (j < mp.info.height)
    hit = np.zeros_like(inb)
    hit[inb] = wall[j[inb], i[inb]]
    hit &= ts <= r[None, :]        # only up to where the beam actually ended
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), steps-1)
    d_map = (first + 1) * res
    return float(np.mean(d_map < r - slack))


t0 = time.time()
R = args.radius
cands = []
for gx in np.arange(cx-R, cx+R+1e-9, 0.05):
    for gy in np.arange(cy-R, cy+R+1e-9, 0.05):
        i, j = int((gx-ox)/res), int((gy-oy)/res)
        if not (0 <= i < mp.info.width and 0 <= j < mp.info.height):
            continue
        if not standable[j, i]:
            continue
        for yaw in np.arange(-math.pi, math.pi, math.radians(5)):
            cands.append((fit(gx, gy, yaw), gx, gy, yaw))
cands.sort(key=lambda t: t[0])
print(f'σάρωση {len(cands)} υποψηφίων σε {time.time()-t0:.1f}s')

# re-score the best 40 with the line-of-sight test, which is what separates
# a real pose from a same-shaped spot elsewhere in the room
best = None
shortlist = []
for f, gx, gy, yaw in cands[:60]:
    blk = blocked_frac(gx, gy, yaw)
    score = f + blk
    shortlist.append((score, f, blk, gx, gy, yaw))
    if best is None or score < best[0]:
        best = (score, f, blk, gx, gy, yaw)
shortlist.sort(key=lambda t: t[0])
print('top 5 (score = scan→wall + ποσοστό δεσμών που περνούν μέσα από τοίχο):')
for sc, f, blk, gx, gy, yaw in shortlist[:5]:
    print(f'  ({gx:6.2f},{gy:6.2f}) yaw={math.degrees(yaw):+4.0f}°  '
          f'fit={f:.3f}  blocked={blk*100:3.0f}%  score={sc:.3f}')

score, f2, blk, lx, ly, lyaw = best
print(f'ΚΑΛΥΤΕΡΟ: laser ({lx:.2f}, {ly:.2f}) yaw={math.degrees(lyaw):.0f}°  '
      f'scan→wall={f2:.3f}m  blocked={blk*100:.0f}%  score={score:.3f}')

# base_link->laser on this robot: x=0 y=0 yaw=pi  => same position, yaw-pi
bx, by, byaw = lx, ly, lyaw - math.pi
byaw = math.atan2(math.sin(byaw), math.cos(byaw))
print(f'base_link: ({bx:.2f}, {by:.2f}) yaw={math.degrees(byaw):.0f}°')

if args.set:
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = rclpy.time.Time().to_msg()   # stamp 0, never now()
    msg.pose.pose.position.x = float(bx)
    msg.pose.pose.position.y = float(by)
    msg.pose.pose.orientation.z = math.sin(byaw/2)
    msg.pose.pose.orientation.w = math.cos(byaw/2)
    cov = [0.0]*36
    cov[0] = cov[7] = 0.05      # 22 cm 1-sigma
    cov[35] = 0.05              # ~13 deg
    msg.pose.covariance = cov
    for _ in range(5):
        n.pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.1)
        time.sleep(0.1)
    print('δημοσιεύτηκε /initialpose')

n.destroy_node(); rclpy.shutdown()
