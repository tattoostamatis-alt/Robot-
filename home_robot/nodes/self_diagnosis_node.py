#!/usr/bin/env python3
"""Self-diagnosis — the robot names its own known failure modes.

This robot has a documented set of recurring gremlins, and nearly all of them
are SILENT: the node stays alive, the topic stays advertised, the preflight
prints a tick, and only the behaviour is wrong. The symptom never points at the
cause, which is what makes them expensive.

This node measures what it can about itself, runs the catalogue in
home_robot/diagnostics.py over it, and reports the named cause plus the fix.

Topics
  in   /rosout            — recent warnings/errors, matched against signatures
       (topic rates are measured directly, without subscribing to the payloads)
  out  /self_diagnosis     (String, JSON)  — findings for the dashboard
       /speech_response   (String)        — spoken, critical findings only

‼️ It measures topic RATES, not contents. Subscribing to /camera image data
just to count frames would cost more than the whole node; a subscription with
no callback work still increments the message counter, which is all this needs.

‼️ Speaking is heavily gated. A robot that announces every warning is a robot
you mute, and then it cannot tell you the one that mattered. Only `critical`
findings are spoken, once each, with a long cooldown.
"""

import json
import os
import time
from collections import deque

import psutil

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rcl_interfaces.msg import Log as RosoutLog
from std_msgs.msg import String

from home_robot.diagnostics import Health, diagnose

# Topics whose liveness tells us something, and the QoS they are published with.
_WATCH = (
    ('/camera/camera/color/image_raw', 'sensor_msgs/msg/Image'),
    ('/camera/camera/aligned_depth_to_color/image_raw', 'sensor_msgs/msg/Image'),
    ('/scan', 'sensor_msgs/msg/LaserScan'),
    ('/odom', 'nav_msgs/msg/Odometry'),
    ('/imu/data', 'sensor_msgs/msg/Imu'),
    ('/mic/audio', 'std_msgs/msg/Int16MultiArray'),
)

# Log substrings that ARE the signature of a known fault.
# ‼️ 'Already transcribing' was here and is not one of them — it is what a
# second wake word inside the listening window prints, which happens all the
# time. See _stt_stuck in diagnostics.py.
_LOG_SIGNATURES = ('Transcription has not returned', 'TTS attempt', 'Errno 110',
                   'serial', 'ΔΕΝ ΑΠΑΝΤΑ', 'interrupted itself')

_SPEAK_COOLDOWN = 1800.0        # s, per finding


class SelfDiagnosisNode(Node):
    def __init__(self):
        super().__init__('self_diagnosis_node')

        self.declare_parameter('period', 20.0)          # s between sweeps
        self.declare_parameter('speak', True)
        self.declare_parameter('log_window', 300.0)     # s of log history

        self.speak = self.get_parameter('speak').value
        self.log_window = self.get_parameter('log_window').value

        self._counts = {}          # topic -> messages since the last sweep
        self._log_hits = deque()   # (ts, signature)
        self._spoken = {}          # finding key -> ts
        self._last = []
        self._t0 = time.time()

        for topic, type_name in _WATCH:
            self._subscribe_counter(topic, type_name)

        self.create_subscription(
            RosoutLog, '/rosout', self._cb_rosout,
            QoSProfile(depth=50,
                       durability=DurabilityPolicy.VOLATILE,
                       reliability=ReliabilityPolicy.BEST_EFFORT))

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        # ‼️ NOT '/diagnostics'. That name belongs to the ROS convention
        # (diagnostic_msgs/DiagnosticArray) and Nav2 already publishes there —
        # a std_msgs/String on it makes the topic multi-type, and `ros2 topic
        # echo /diagnostics` then refuses to display either.
        self._pub = self.create_publisher(String, 'self_diagnosis', latched)
        self._say_pub = self.create_publisher(String, '/speech_response', 10)

        self.create_timer(self.get_parameter('period').value, self._sweep)
        self.get_logger().info('Self-diagnosis ready')

    def _subscribe_counter(self, topic, type_name):
        """Count messages without deserialising the payload where possible.

        The callback does nothing but increment — that is the entire point:
        rate is the signal, contents are irrelevant and expensive.
        """
        try:
            from rosidl_runtime_py.utilities import get_message
            msg_type = get_message(type_name)
        except Exception as exc:                          # noqa: BLE001
            self.get_logger().warn(f'cannot watch {topic}: {exc}')
            return
        self._counts[topic] = 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            msg_type, topic,
            lambda _m, t=topic: self._counts.__setitem__(t, self._counts[t] + 1),
            qos)

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_rosout(self, msg: RosoutLog):
        if msg.level < 30:                                # WARN and above only
            return
        text = msg.msg or ''
        now = time.time()
        for sig in _LOG_SIGNATURES:
            if sig in text:
                self._log_hits.append((now, sig))
        cutoff = now - self.log_window
        while self._log_hits and self._log_hits[0][0] < cutoff:
            self._log_hits.popleft()

    # ── the sweep ─────────────────────────────────────────────────────────────

    def _snapshot(self, elapsed):
        rates = {t: (c / elapsed if elapsed > 0 else 0.0)
                 for t, c in self._counts.items()}
        for t in self._counts:
            self._counts[t] = 0

        nodes = set()
        try:
            nodes = {n for n, _ns in self.get_node_names_and_namespaces()}
        except Exception:                                 # noqa: BLE001
            pass

        counts = {}
        for _ts, sig in self._log_hits:
            counts[sig] = counts.get(sig, 0) + 1

        # ‼️ swap_pct carries the PAGING RATE, not the used percentage. Swap
        # sitting allocated is normal on this 13 GB box and costs nothing;
        # only si/so actually hurts, and conflating them cries wolf forever.
        try:
            sw = psutil.swap_memory()
            paging = float(getattr(sw, 'sin', 0) + getattr(sw, 'sout', 0))
        except Exception:                                 # noqa: BLE001
            paging = 0.0
        prev = getattr(self, '_prev_paging', None)
        self._prev_paging = paging
        paging_rate = 0.0 if prev is None else max(0.0, paging - prev)

        try:
            vm = psutil.virtual_memory()
            load1 = os.getloadavg()[0]
            cores = psutil.cpu_count() or 1
            cpu = psutil.cpu_percent(interval=None)
        except Exception:                                 # noqa: BLE001
            vm, load1, cores, cpu = None, 0.0, 1, 0.0

        return Health(topic_hz=rates, nodes=nodes, log_counts=counts,
                      cpu_pct=cpu, ram_pct=(vm.percent if vm else 0.0),
                      swap_pct=paging_rate, load1=load1, cores=cores,
                      uptime_s=time.time() - self._t0)

    def _sweep(self):
        elapsed = self.get_parameter('period').value
        health = self._snapshot(elapsed)
        findings = diagnose(health)
        self._last = findings

        now = time.time()
        if self.speak:
            for f in findings:
                if f.severity != 'critical':
                    continue
                if now - self._spoken.get(f.key, -1e9) < _SPEAK_COOLDOWN:
                    continue
                self._spoken[f.key] = now
                self._say_pub.publish(String(data=f'{f.title}. {f.fix}'))
                self.get_logger().warn(f'{f.title} — {f.fix}')
                break                     # one spoken problem at a time

        self._pub.publish(String(data=json.dumps({
            'findings': [f.to_dict() for f in findings],
            'checked': len(self._counts),
            'load1': round(health.load1, 2),
            'cores': health.cores,
            'ram_pct': round(health.ram_pct, 1),
            'paging': round(health.swap_pct, 1),
            'ts': now,
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = SelfDiagnosisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
