#! /usr/bin/env python3
# Adapted from open-navigation/opennav_amd_demonstrations honeybee_watchdogs
# (capture_system_metrics.py), Copyright 2024 Open Navigation LLC, Apache-2.0.
# Retargeted for home_robot + extended with thermal/per-core/fsync logging to
# investigate the mini-PC abrupt shutdowns (see project_npu_minipc_power memory).
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy at
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Black-box system-metrics logger.

Runs on the robot's mini PC and appends one line per tick with CPU/thermal/
memory/disk/load data to a timestamped file under ~/experiment_files. Built to
survive an abrupt power cut: every line is flushed + fsync'd, so the tail of the
file is the last thing the machine saw before it died -- exactly what's needed
to catch a thermal/power event behind the abrupt shutdowns.

Additions over the upstream Open Navigation version:
  * ALL thermal zones (k10temp/Tctl CPU die, amdgpu/edge iGPU, nvme, ...) with
    a WARNING logged when any zone crosses its 'high' threshold
  * hottest-zone summary column + per-core CPU% peak (catch a runaway core)
  * os.fsync after every write (power-cut durable)
  * optional odom correlation (param `odom_topic`, off by default)

Run live:   ros2 run home_robot system_metrics_node.py
Tune rate:  ros2 run home_robot system_metrics_node.py --ros-args -p hz:=2.0
Analyze:    ros2 run home_robot system_metrics_node.py --ros-args -p analyze_dir:=~/experiment_files
"""
import os
import sys
import time

import psutil
import rclpy
from rclpy.node import Node


def parse_metrics_file(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            row = {}
            for pair in line.strip().split(', '):
                if ': ' not in pair:
                    continue
                key, value = pair.split(': ', 1)
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                row[key] = value
            if row:
                rows.append(row)
    return rows


def analyze(directory):
    directory = os.path.expanduser(directory)
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.startswith('system_metrics')]
    if not files:
        print(f'No system_metrics_*.txt files in {directory}')
        return
    rows = [r for path in files for r in parse_metrics_file(path)]
    if not rows:
        print('No parseable rows.')
        return

    def col(name):
        return [r[name] for r in rows if name in r and isinstance(r[name], (int, float))]

    print(f'Files: {len(files)}   Samples: {len(rows)}   (~{len(rows) / 60.0:.1f} min @1Hz)')
    for name in ('CPU', 'Memory', 'Swap', 'Disk', 'MaxTemp', 'CPUTemp', 'GPUTemp', 'Load1'):
        vals = col(name)
        if vals:
            print(f'  {name:8s} avg {sum(vals) / len(vals):6.2f}   max {max(vals):6.2f}')
    # surface the hottest sample -- the prime suspect for a thermal shutdown
    hot = [r for r in rows if 'MaxTemp' in r]
    if hot:
        peak = max(hot, key=lambda r: r['MaxTemp'])
        print(f"\nHottest sample: MaxTemp={peak.get('MaxTemp')}°C "
              f"({peak.get('HotZone', '?')}) at Time={peak.get('Time')} "
              f"CPU={peak.get('CPU')}% Load1={peak.get('Load1')}")


class SystemMetrics(Node):

    def __init__(self):
        super().__init__('system_metrics')
        self.declare_parameter('filepath', '~/experiment_files')
        self.declare_parameter('hz', 1.0)
        self.declare_parameter('odom_topic', '')  # '' disables odom correlation

        self.filepath = os.path.expanduser(self.get_parameter('filepath').value)
        os.makedirs(self.filepath, exist_ok=True)
        self.filename = os.path.join(self.filepath, f'system_metrics_{int(time.time())}.txt')

        self.speed = 0.0
        odom_topic = self.get_parameter('odom_topic').value
        if odom_topic:
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        hz = max(0.1, float(self.get_parameter('hz').value))
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f'Logging system metrics -> {self.filename} at {hz} Hz')

    def _odom_cb(self, msg):
        self.speed = msg.twist.twist.linear.x

    def _read_temps(self):
        """Return (max_temp, hot_zone_name, cpu_temp, gpu_temp, warnings[])."""
        temps = psutil.sensors_temperatures()
        max_temp, hot_zone = -1.0, '?'
        cpu_temp = gpu_temp = -1.0
        warnings = []
        for chip, entries in temps.items():
            for e in entries:
                if e.current is None:
                    continue
                zone = f'{chip}/{e.label}' if e.label else chip
                if e.current > max_temp:
                    max_temp, hot_zone = e.current, zone
                if chip == 'k10temp':
                    cpu_temp = e.current
                elif chip == 'amdgpu':
                    gpu_temp = e.current
                # nvme sensor 1/2 report bogus 65261° 'high' values; ignore those
                if e.high and 0 < e.high < 130 and e.current >= e.high:
                    warnings.append(f'{zone}={e.current}°C>=high({e.high})')
        return max_temp, hot_zone, cpu_temp, gpu_temp, warnings

    def _tick(self):
        cpu = psutil.cpu_percent(interval=0.1)
        per_core = psutil.cpu_percent(percpu=True)
        max_core = max(per_core) if per_core else cpu
        mem = psutil.virtual_memory().percent
        swap = psutil.swap_memory().percent
        disk = psutil.disk_usage('/').percent
        load1 = os.getloadavg()[0]
        freq = psutil.cpu_freq()
        cpu_mhz = round(freq.current) if freq else -1
        max_temp, hot_zone, cpu_temp, gpu_temp, warnings = self._read_temps()

        if warnings:
            self.get_logger().warning('THERMAL: ' + ', '.join(warnings))

        line = (
            f'Time: {int(time.time())}, CPU: {cpu}, MaxCore: {max_core}, '
            f'CPUFreqMHz: {cpu_mhz}, Memory: {mem}, Swap: {swap}, Disk: {disk}, '
            f'Load1: {load1:.2f}, MaxTemp: {max_temp}, HotZone: {hot_zone}, '
            f'CPUTemp: {cpu_temp}, GPUTemp: {gpu_temp}, Speed: {self.speed:.3f}\n'
        )
        # flush + fsync so an abrupt power cut can't lose the last samples
        with open(self.filename, 'a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def main():
    # allow `--ros-args -p analyze_dir:=...` to run the offline analyzer instead
    if '--analyze' in sys.argv:
        idx = sys.argv.index('--analyze')
        analyze(sys.argv[idx + 1] if idx + 1 < len(sys.argv) else '~/experiment_files')
        return
    rclpy.init()
    node = SystemMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
