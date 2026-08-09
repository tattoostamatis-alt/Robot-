#!/usr/bin/env python3
"""Drive out of a tight spot using the lidar only — no map, no costmap.

Needed when localization is unusable: in a space as tight as this toilet the
scan matches nowhere on the map, so every map-based recovery (planner, BackUp,
Spin) refuses. This steers purely on live /scan: turn toward the roomiest
bearing, creep, stop the moment anything gets close or the bumper trips.

    drive_to_open.py            # dry run
    drive_to_open.py --go       # actually move
"""
import argparse, json, math, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String

TURN = 0.50      # rad/s — the chassis will not rotate at all below ~0.31
CREEP = 0.09     # m/s
STOP_AHEAD = 0.30    # m in a +-35 deg cone: stop
OPEN_ENOUGH = 0.60   # m clear all around the front half: we are out
MAX_TRAVEL = 2.5     # m


class D(Node):
    def __init__(self):
        super().__init__('drive_to_open')
        self.scan = None
        self.bump = False
        self.create_subscription(LaserScan, '/scan', self._s, 10)
        self.create_subscription(String, '/roomba/status', self._st, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)

    def _s(self, m): self.scan = m

    def _st(self, m):
        try:
            d = json.loads(m.data)
            self.bump = bool(d.get('bump') or d.get('wheel_drop'))
        except Exception:
            pass

    def pump(self, n=10):
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.05)

    def sectors(self, width=math.radians(30)):
        s = self.scan
        out = []
        for k in range(int(2*math.pi/width)):
            lo = -math.pi + k*width
            best = float('inf')
            for i, r in enumerate(s.ranges):
                if not (s.range_min <= r <= s.range_max):
                    continue
                a = math.atan2(math.sin(s.angle_min + i*s.angle_increment),
                               math.cos(s.angle_min + i*s.angle_increment))
                if lo <= a < lo+width:
                    best = min(best, r)
            out.append((lo+width/2, best))
        return out

    def cone(self, half=math.radians(35)):
        s = self.scan
        best = float('inf')
        for i, r in enumerate(s.ranges):
            if not (s.range_min <= r <= s.range_max):
                continue
            a = math.atan2(math.sin(s.angle_min + i*s.angle_increment),
                           math.cos(s.angle_min + i*s.angle_increment))
            if abs(a) <= half:
                best = min(best, r)
        return best

    def stop(self):
        for _ in range(3):
            self.pub.publish(Twist())
            time.sleep(0.05)

    def move(self, lin, ang, seconds):
        t = Twist(); t.linear.x = lin; t.angular.z = ang
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            if self.bump:
                self.get_logger().warn('ΠΡΟΦΥΛΑΚΤΗΡΑΣ — σταματώ')
                break
            if lin > 0 and self.cone() < STOP_AHEAD:
                break
            self.pub.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.stop()


ap = argparse.ArgumentParser()
ap.add_argument('--go', action='store_true')
args = ap.parse_args()

rclpy.init(); n = D()
while rclpy.ok() and n.scan is None:
    rclpy.spin_once(n, timeout_sec=0.1)
n.pump(20)

sec = n.sectors()
print('ελεύθερος χώρος (0°=μπροστά):')
for a, r in sec:
    print(f'  {math.degrees(a):+7.0f}°  {r:5.2f} m  ' + '#'*min(int(r*10), 40))
front = min(r for a, r in sec if abs(a) <= math.radians(60))
print(f'μπροστινό ημισφαίριο ελεύθερο: {front:.2f} m   bump={n.bump}')

if not args.go:
    print('(dry-run)')
    raise SystemExit(0)

travelled = 0.0
for step in range(14):
    n.pump(6)
    sec = n.sectors()
    front = min(r for a, r in sec if abs(a) <= math.radians(60))
    if front >= OPEN_ENOUGH and n.cone() >= OPEN_ENOUGH:
        print(f'✔ ανοιχτός χώρος ({front:.2f} m μπροστά) μετά από {travelled:.2f} m')
        break
    if travelled >= MAX_TRAVEL:
        print('όριο διαδρομής')
        break
    if n.bump:
        print('προφυλακτήρας — σταματώ')
        break

    ahead = n.cone()
    best_a, best_r = max(sec, key=lambda t: t[1])

    # ‼️ Only turn when going straight is actually worse. Steering at every
    # step toward "whichever sector is roomiest right now" oscillates: the
    # turn changes which sector that is, so the robot swings -165, +135,
    # -135 and creeps nowhere (measured 2026-08-09 getting out of the
    # toilet — 8 turns for 0.45 m). Commit to forward while forward works,
    # and demand a clearly better bearing (1.6x) before giving that up.
    if ahead >= STOP_AHEAD and best_r < ahead * 1.6:
        d = min(0.25, ahead - STOP_AHEAD)
        print(f'  προχωρώ {d:.2f} m (ελεύθερο {ahead:.2f} m)')
        n.move(CREEP, 0.0, d/CREEP)
        travelled += d
        n.pump(10)
        continue

    if abs(best_a) > math.radians(20) and best_r > STOP_AHEAD:
        print(f'  στρίβω {math.degrees(best_a):+.0f}° (εκεί {best_r:.2f} m, '
              f'μπροστά {ahead:.2f} m)')
        n.move(0.0, math.copysign(TURN, best_a), abs(best_a)/TURN)
        n.pump(10)
        continue

    print(f'  μπλοκαρισμένο παντού (μπροστά {ahead:.2f} m) — στρίβω επί τόπου')
    n.move(0.0, TURN, 0.8)
    n.pump(10)

n.stop()
sec = n.sectors()
print(f'τέλος: μπροστά {n.cone():.2f} m, μέγιστο ελεύθερο '
      f'{max(r for _, r in sec):.2f} m, διανύθηκαν {travelled:.2f} m')
n.destroy_node(); rclpy.shutdown()
