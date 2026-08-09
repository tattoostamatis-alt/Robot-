#!/usr/bin/env python3
"""Print the global costmap between the robot and a named goal.

When both endpoints are cost 0 and the planner still says "Failed to create a
plan from potential when a legal potential was found", something in between
forms an unbroken wall of >=99 cells. This shows it.
"""
import math, sys, rclpy, yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory

LATCH = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                   reliability=QoSReliabilityPolicy.RELIABLE,
                   history=QoSHistoryPolicy.KEEP_LAST)


class P(Node):
    def __init__(self):
        super().__init__('path_block')
        self.g = None
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._g, LATCH)
        self.buf = Buffer(); self.tfl = TransformListener(self.buf, self)

    def _g(self, m): self.g = m


room = sys.argv[1]
rclpy.init(); n = P()
rx = ry = None
for _ in range(400):
    rclpy.spin_once(n, timeout_sec=0.05)
    if n.g is not None:
        try:
            tf = n.buf.lookup_transform('map', 'base_link', rclpy.time.Time())
            rx, ry = tf.transform.translation.x, tf.transform.translation.y
            break
        except Exception:
            pass

g = n.g
res, ox, oy = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y
locs = yaml.safe_load(open(get_package_share_directory('home_robot') + '/config/locations.yaml'))
gx, gy = locs[room]['x'], locs[room]['y']
print(f'ρομπότ ({rx:.2f},{ry:.2f}) → {room} ({gx:.2f},{gy:.2f})')


def c(x, y):
    i, j = int((x-ox)/res), int((y-oy)/res)
    if not (0 <= i < g.info.width and 0 <= j < g.info.height):
        return None
    return g.data[j*g.info.width+i]


# bounding box of the two points, padded
x0, x1 = min(rx, gx)-0.8, max(rx, gx)+0.8
y0, y1 = min(ry, gy)-0.5, max(ry, gy)+0.5
print(f'\nGLOBAL costmap  x {x0:.1f}..{x1:.1f}  y {y0:.1f}..{y1:.1f}')
print('  . <50   : 50-98   # 99   X 100   ? εκτός    R ρομπότ   G στόχος')
nj = int((y1-y0)/res)
ni = int((x1-x0)/res)
for jj in range(nj, -1, -1):
    y = y0 + jj*res
    row = ''
    for ii in range(ni+1):
        x = x0 + ii*res
        v = c(x, y)
        if abs(x-rx) < res and abs(y-ry) < res:
            row += 'R'; continue
        if abs(x-gx) < res and abs(y-gy) < res:
            row += 'G'; continue
        if v is None:
            row += '?'
        elif v >= 100:
            row += 'X'
        elif v >= 99:
            row += '#'
        elif v >= 50:
            row += ':'
        else:
            row += '.'
    print('  ' + row)
n.destroy_node(); rclpy.shutdown()
