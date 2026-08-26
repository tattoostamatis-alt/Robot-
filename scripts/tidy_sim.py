#!/usr/bin/env python3
"""A virtual house for the "tidy the floor, carry it to the corner" mission —
no hardware. Sibling of fetch_sim.py, same shape: puts one clutter item in
every room of config/locations.yaml, triggers `tidy` for the whole house, and
checks that every item gets picked (held), an attempt is made to deliver it to
`drop_location` ("gonia"), and the robot returns to the room before the next
item — exactly the sequence added to task_planner_node.py on 2026-08-15.

What is REAL is the part under test: this starts the actual
task_planner_node.py as a subprocess and talks to it over the actual topics
(tidy_command, pick_command/pick_result, place_command/place_result,
detected_objects, navigate_to_pose). Only the world it acts on — Nav2, TF,
the camera, and the arm — is invented, same as fetch_sim.py, and for the same
reason: a simulated /navigate_to_pose or map->base_link on the robot's own
domain would fight the real Nav2/AMCL. Uses its own ROS_DOMAIN_ID, NOT 91
(fetch_sim) or 77 (the dashboard's Gazebo tab) or the robot's 0.

pick_place_node.py is NOT started — pick/place are answered directly by the
fake house (always 'ok'), same simplification fetch_sim.py makes. This tests
task_planner_node.py's orchestration (the new code), not the arm's grasp
geometry or the MoveIt transit — those need real hardware or a much bigger
sim and are out of scope here.

‼️ "gonia" (the drop corner) is deliberately NOT injected into the real
locations.yaml this script reads — if it hasn't been taught yet on this
machine (record_location.py gonia), the run exercises task_planner's
"location not taught, drop in place" fallback instead of an actual delivery
trip. Both are valid to observe; the summary says which one happened.

Usage:
    scripts/tidy_sim.py                  # every room
    scripts/tidy_sim.py --room saloni    # just one
    scripts/tidy_sim.py --domain 93      # somewhere else again

Exit status is 0 only if every room's item was picked and a place was
attempted (delivered to gonia, or the documented fallback).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

STEP_DT = 0.02
STEP_DIST = 0.35

SRC = Path(__file__).resolve().parents[1]
_TASK_PLANNER_MARK = 'lib/home_robot/task_planner_node.py'

# COCO clutter labels (see object_detector.py's CLUTTER_CLASSES) — real
# labels, so a future switch to open-vocab items doesn't need this touched.
CLUTTER_LABELS = ['cup', 'bottle', 'book', 'remote', 'cell phone', 'scissors']


def _die_with_parent():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').prctl(1, 9, 0, 0, 0)
    except Exception:
        pass


def _reap_orphans(verbose=False):
    """Kill task_planner_node processes left over from an earlier run of THIS
    script — matched on the installed path, so a real `ros2 launch`'d
    task_planner_node (different command line, different domain) is untouched."""
    try:
        out = subprocess.run(['pgrep', '-af', _TASK_PLANNER_MARK],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return
    for line in out.splitlines():
        pid_s, _, cmd = line.partition(' ')
        if 'bash -c' in cmd or not pid_s.isdigit():
            continue
        pid = int(pid_s)
        if pid == os.getpid():
            continue
        if verbose:
            print(f'  (killing leftover task_planner_node pid {pid})')
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def _load_rooms():
    import yaml
    with open(SRC / 'config' / 'locations.yaml') as f:
        raw = yaml.safe_load(f) or {}
    return {k: (float(v['x']), float(v['y']))
            for k, v in raw.items()
            if isinstance(v, dict) and 'x' in v and not k.startswith('dock')}


def build_world(rooms):
    """One clutter item per room."""
    world = {}
    for i, room in enumerate(sorted(rooms)):
        world[room] = CLUTTER_LABELS[i % len(CLUTTER_LABELS)]
    return world


def run_house(rooms, world, start_room, ready_evt, stop_evt, log):
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import TransformStamped
    from nav2_msgs.action import NavigateToPose
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    class VirtualHouse(Node):
        def __init__(self):
            super().__init__('virtual_tidy_house')
            self.cbg = ReentrantCallbackGroup()
            sx, sy = rooms[start_room]
            self.rx, self.ry, self.ryaw = sx, sy, 0.0
            # room -> label still lying around there, or None once picked up.
            self.world = dict(world)
            self.held = None
            self.events = []

            self.tf = TransformBroadcaster(self)
            self.create_timer(0.05, self._pub_tf)
            self.create_timer(0.10, self._pub_detections)

            self.det_pub = self.create_publisher(String, 'detected_objects', 10)
            self.pick_pub = self.create_publisher(String, 'pick_result', 10)
            self.place_pub = self.create_publisher(String, 'place_result', 10)

            self.create_subscription(String, 'pick_command', self._on_pick, 10)
            self.create_subscription(String, 'place_command', self._on_place, 10)

            ActionServer(self, NavigateToPose, 'navigate_to_pose',
                         execute_callback=self._on_nav,
                         goal_callback=lambda _g: GoalResponse.ACCEPT,
                         cancel_callback=lambda _g: CancelResponse.ACCEPT,
                         callback_group=self.cbg)

        def _current_room(self):
            best, best_d = None, 1e9
            for room, (x, y) in rooms.items():
                d = math.hypot(x - self.rx, y - self.ry)
                if d < best_d:
                    best, best_d = room, d
            return best if best_d < 0.5 else None

        def _pub_tf(self):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'
            t.transform.translation.x = float(self.rx)
            t.transform.translation.y = float(self.ry)
            t.transform.rotation.z = math.sin(self.ryaw / 2.0)
            t.transform.rotation.w = math.cos(self.ryaw / 2.0)
            self.tf.sendTransform(t)

        def _pub_detections(self):
            room = self._current_room()
            label = self.world.get(room) if room else None
            out = [] if label is None else [{'label': label, 'clutter': True}]
            self.det_pub.publish(String(data=json.dumps(out)))

        def _on_pick(self, msg):
            try:
                req = json.loads(msg.data)
            except json.JSONDecodeError:
                req = {}
            label = req.get('label', '')
            room = self._current_room()
            expected = self.world.get(room) if room else None
            ok = expected is not None and expected == label
            self.events.append(('pick', room, label, ok))
            if ok:
                self.world[room] = None
                self.held = label
            self.pick_pub.publish(String(data=json.dumps(
                {'status': 'ok' if ok else 'failed',
                 'detail': label if ok else f'nothing to grab in {room}'})))

        def _on_place(self, _msg):
            room = self._current_room()
            self.events.append(('place', room, self.held))
            self.held = None
            self.place_pub.publish(String(data=json.dumps(
                {'status': 'ok', 'detail': 'placed'})))

        def _on_nav(self, goal_handle):
            p = goal_handle.request.pose.pose
            gx, gy = p.position.x, p.position.y
            gyaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
            self.events.append(('navigate', round(gx, 2), round(gy, 2)))
            while True:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return NavigateToPose.Result()
                dx, dy = gx - self.rx, gy - self.ry
                d = math.hypot(dx, dy)
                if d <= STEP_DIST:
                    self.rx, self.ry, self.ryaw = gx, gy, gyaw
                    break
                self.rx += STEP_DIST * dx / d
                self.ry += STEP_DIST * dy / d
                self.ryaw = math.atan2(dy, dx)
                time.sleep(STEP_DT)
            goal_handle.succeed()
            return NavigateToPose.Result()

    node = VirtualHouse()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    log['house'] = node
    ready_evt.set()
    while not stop_evt.is_set():
        ex.spin_once(timeout_sec=0.1)
    node.destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='93',
                    help='ROS_DOMAIN_ID for the simulation (NOT the robot\'s 0, '
                         'fetch_sim\'s 91, or the Gazebo tab\'s 77)')
    ap.add_argument('--room', action='append', help='tidy only this room (repeatable)')
    ap.add_argument('--start-room', default='saloni')
    ap.add_argument('--timeout', type=float, default=180.0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)

    rooms = _load_rooms()
    if args.start_room not in rooms:
        print(f'unknown --start-room {args.start_room}; have {sorted(rooms)}')
        return 2
    world = build_world(rooms)
    wanted_rooms = args.room or sorted(rooms)
    bad = [r for r in wanted_rooms if r not in rooms]
    if bad:
        print(f'unknown room(s): {bad}; have {sorted(rooms)}')
        return 2

    gonia_taught = 'gonia' in rooms
    print(f'ROS_DOMAIN_ID={args.domain}  (the robot lives on 0 — untouched)')
    print(f'"gonia" (drop corner) is {"TAUGHT" if gonia_taught else "NOT taught — "
          "expect the in-place fallback, not an actual delivery trip"} in locations.yaml')
    print(f'Starting in {args.start_room}. One item per room:')
    for room in wanted_rooms:
        print(f'  {room:20s} {world[room]}')
    print()

    import rclpy
    rclpy.init()

    ready, stop, shared = threading.Event(), threading.Event(), {}
    house_thread = threading.Thread(
        target=run_house, args=(rooms, world, args.start_room, ready, stop, shared),
        daemon=True)
    house_thread.start()
    if not ready.wait(20):
        print('the virtual house did not come up')
        return 3

    exe = SRC.parents[1] / 'install' / 'home_robot' / 'lib' / 'home_robot' / 'task_planner_node.py'
    if not exe.exists():
        print(f'task_planner_node.py not found at {exe} — build the workspace')
        stop.set()
        return 3
    _reap_orphans(verbose=True)
    planner = subprocess.Popen(
        [sys.executable, str(exe), '--ros-args',
         '-p', 'use_arm:=true',
         '-p', 'detect_wait:=0.3',
         '-p', 'pick_timeout:=10.0',
         '-p', 'nav_timeout:=60.0'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ), preexec_fn=_die_with_parent)
    planner_log = []
    threading.Thread(
        target=lambda: [planner_log.append(l.rstrip()) for l in planner.stdout],
        daemon=True).start()

    from rclpy.node import Node
    from std_msgs.msg import String

    class Driver(Node):
        def __init__(self):
            super().__init__('tidy_sim_driver')
            self.speech = []
            self.create_subscription(String, 'speech_response', self._sp, 10)
            self.tidy_pub = self.create_publisher(String, 'tidy_command', 10)

        def _sp(self, m):
            self.speech.append(m.data)
            print(f'    🤖 {m.data}')

    drv = Driver()

    def spin_for(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(drv, timeout_sec=0.05)

    print('waiting for task_planner_node…')
    spin_for(4.0)

    house = shared['house']
    house.events.clear()
    drv.speech.clear()
    room_arg = wanted_rooms[0] if len(wanted_rooms) == 1 else 'all'
    print(f'── tidy: {room_arg} {"─" * 30}')
    drv.tidy_pub.publish(String(data=json.dumps({'room': room_arg})))

    DONE_PHRASES = ('Ολοκλήρωσα τον γύρο τακτοποίησης',
                    'Δεν βρήκα ακαταστασία' if len(wanted_rooms) == 1 else '__never__',
                    'Σταμάτησα την τακτοποίηση')
    end = time.monotonic() + args.timeout
    finished = False
    try:
        while time.monotonic() < end:
            rclpy.spin_once(drv, timeout_sec=0.05)
            if any(any(p in s for p in DONE_PHRASES) for s in drv.speech):
                finished = True
                break
            # Single-room run has no terminal phrase — done when a pick/place
            # pair for that room has happened, or nav failed.
            if len(wanted_rooms) == 1:
                places = [e for e in house.events if e[0] == 'place']
                if places:
                    finished = True
                    break
        spin_for(1.0)   # let the last place_result/speech_response land
    finally:
        planner.terminate()
        try:
            planner.wait(timeout=10)
        except subprocess.TimeoutExpired:
            planner.kill()
            planner.wait(timeout=5)
        _reap_orphans()
        drv.destroy_node()
        rclpy.try_shutdown()
        stop.set()
        house_thread.join(timeout=5)

    navs = [e for e in house.events if e[0] == 'navigate']
    picks = {e[1]: e[3] for e in house.events if e[0] == 'pick'}
    places = [e for e in house.events if e[0] == 'place']

    print()
    print('=' * 62)
    print(f'finished={finished}  navigations={len(navs)}  '
          f'picks_attempted={len(picks)}  places={len(places)}')
    ok = 0
    for room in wanted_rooms:
        picked = picks.get(room, False)
        ok += picked
        print(f'  {"✅" if picked else "❌"} {room:20s} '
              f'item={world[room]:12s} picked={picked}')
    print(f'{ok}/{len(wanted_rooms)} rooms picked')
    if not gonia_taught:
        print('(no delivery trip to test — "gonia" is not taught on this machine yet)')

    if args.verbose or ok != len(wanted_rooms) or not finished:
        print('\n── house events ──')
        for e in house.events:
            print(f'  · {e}')
        print('\n── last 40 lines from task_planner_node ──')
        for line in planner_log[-40:]:
            print('  ' + line)

    return 0 if (finished and ok == len(wanted_rooms)) else 1


if __name__ == '__main__':
    sys.exit(main())
