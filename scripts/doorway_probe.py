#!/usr/bin/env python3
"""Who eats the command in a doorway?

The chain is:  MPPI -> cmd_vel_nav -> velocity_smoother -> cmd_vel_smoothed
               -> collision_monitor -> cmd_vel_safe -> wheels

So a hesitation has exactly three possible authors and they are trivially
distinguishable at the topic level:
  * cmd_vel_nav itself small        -> MPPI is creeping (CostCritic)
  * nav big, smoothed small         -> velocity_smoother ramp
  * smoothed big, safe ~0           -> collision_monitor is zeroing it
Logs every sample where they disagree, plus the monitor's own state topic.
"""
import math, time, sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import CollisionMonitorState


class P(Node):
    def __init__(self, seconds):
        super().__init__('doorway_probe')
        self.nav = self.sm = self.safe = None
        self.state = None
        self.scan = None
        self.rows = []
        self.t0 = time.monotonic()
        self.seconds = seconds
        for topic, cb in (('/cmd_vel_nav', self._nav),
                          ('/cmd_vel_smoothed', self._sm),
                          ('/cmd_vel_safe', self._safe)):
            self.create_subscription(Twist, topic, cb, 10)
            self.create_subscription(TwistStamped, topic, self._stamped(cb), 10)
        self.create_subscription(CollisionMonitorState, '/collision_monitor_state',
                                 self._st, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_timer(0.1, self._tick)

    @staticmethod
    def _stamped(cb):
        return lambda m: cb(m.twist)

    def _nav(self, m): self.nav = m
    def _sm(self, m): self.sm = m
    def _safe(self, m): self.safe = m
    def _st(self, m): self.state = m
    def _scan_cb(self, m): self.scan = m

    def _closest(self):
        if self.scan is None:
            return float('nan')
        s = self.scan
        best = float('inf')
        for r in s.ranges:
            if s.range_min <= r <= s.range_max:
                best = min(best, r)
        return best

    def _tick(self):
        t = time.monotonic() - self.t0
        if t > self.seconds:
            self._report()
            rclpy.shutdown()
            return
        n = self.nav.linear.x if self.nav else float('nan')
        s = self.sm.linear.x if self.sm else float('nan')
        f = self.safe.linear.x if self.safe else float('nan')
        pol = self.state.polygon_name if self.state else ''
        act = self.state.action_type if self.state else -1
        self.rows.append((t, n, s, f, pol, act, self._closest()))

    def _report(self):
        print(f'{len(self.rows)} δείγματα\n')
        print('  t     nav   smooth  safe   closest  monitor')
        eaten = ramped = creeping = 0
        for t, n, s, f, pol, act, c in self.rows:
            flag = ''
            if s == s and f == f:
                if abs(s) > 0.05 and abs(f) < 0.01:
                    flag = '  <<< COLLISION_MONITOR ΜΗΔΕΝΙΖΕΙ'; eaten += 1
                elif n == n and abs(n) > 0.05 and abs(s) < abs(n)*0.5:
                    flag = '  <<< SMOOTHER κόβει'; ramped += 1
                elif n == n and 0.001 < abs(n) < 0.06:
                    flag = '  <<< MPPI σέρνεται'; creeping += 1
            if flag or (act and act > 0):
                print(f'  {t:5.1f} {n:6.3f} {s:6.3f} {f:6.3f}  {c:5.2f}   '
                      f'{pol}/{act}{flag}')
        tot = max(len(self.rows), 1)
        print(f'\nΣΥΝΟΨΗ σε {tot} δείγματα:')
        print(f'  collision_monitor μηδένισε : {eaten:4d}  ({100*eaten/tot:.0f}%)')
        print(f'  smoother έκοψε >50%        : {ramped:4d}  ({100*ramped/tot:.0f}%)')
        print(f'  MPPI σερνόταν (<0.06 m/s)  : {creeping:4d}  ({100*creeping/tot:.0f}%)')
        moving = [r for r in self.rows if r[3] == r[3] and abs(r[3]) > 0.02]
        if moving:
            print(f'  μέση ταχύτητα όταν κινείται: '
                  f'{sum(abs(r[3]) for r in moving)/len(moving):.3f} m/s')


secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
rclpy.init()
n = P(secs)
try:
    rclpy.spin(n)
except Exception:
    pass
