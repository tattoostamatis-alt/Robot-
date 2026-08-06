#!/usr/bin/env python3
"""A virtual house for the "φέρε μου το X" (fetch) mission — no hardware.

Puts one object in every room of config/locations.yaml and asks the robot to
go and get each of them, one at a time. Nothing here touches the real robot:
the base, the arm, the camera, Nav2 and object memory are all stand-ins, and
the whole thing runs in its own ROS_DOMAIN_ID.

What is REAL is the part under test. This starts the actual
mission_executor_node as a subprocess and talks to it over the actual topics,
so the state machine, the approach geometry, the retry logic and the grasp
range checks are the shipping code. Only the world it acts on is invented.

    ‼️ ROS_DOMAIN_ID defaults to 91, NOT the robot's 0. A simulated
    /navigate_to_pose on the live domain would take goals meant for the real
    Nav2, and a simulated map->base_link TF would fight AMCL — the same class
    of mistake as the Gazebo tab that hijacked the live robot.

Usage:
    scripts/fetch_sim.py                 # every room, one object each
    scripts/fetch_sim.py --object μπάλα  # just this one
    scripts/fetch_sim.py --keep-going    # don't stop at the first failure
    scripts/fetch_sim.py --domain 92     # somewhere else again

Exit status is 0 only if every requested fetch reported `done`.
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

# The simulated drive is quantised into steps; the robot "teleports" a little
# way along the straight line to the goal every tick. Nothing here models
# obstacles or a planner — this is about the mission logic, not navigation.
STEP_DT = 0.02          # s of wall clock per drive step
STEP_DIST = 0.35        # m of simulated travel per step

SRC = Path(__file__).resolve().parents[1]

# The exact process this script starts, and the only one it may ever kill.
_MISSION_EXE_MARK = 'lib/home_robot/mission_executor_node.py'


def _die_with_parent():
    """PR_SET_PDEATHSIG: child gets SIGKILL if this script dies."""
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').prctl(1, 9, 0, 0, 0)   # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def _reap_orphans(verbose=False):
    """Kill mission_executor processes left over from an earlier run.

    Matched on the INSTALLED path, so this cannot touch a mission_executor
    launched as part of the real robot — that one comes up under `ros2 launch`
    with a different command line and, more to the point, lives on the robot's
    own ROS_DOMAIN_ID where this script never publishes.
    """
    try:
        out = subprocess.run(['pgrep', '-af', _MISSION_EXE_MARK],
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
            print(f'  (killing leftover mission_executor pid {pid})')
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def _load_rooms():
    """Room name -> (x, y) from the real locations.yaml, minus the dock poses."""
    import yaml
    with open(SRC / 'config' / 'locations.yaml') as f:
        raw = yaml.safe_load(f) or {}
    return {k: (float(v['x']), float(v['y']))
            for k, v in raw.items()
            if isinstance(v, dict) and 'x' in v and not k.startswith('dock')}


# One object per room. Greek labels on purpose: this is what the user actually
# says, and the label is compared case-insensitively but otherwise verbatim all
# the way from speech to object memory to the grasp check.
OBJECTS = ['μπάλα', 'κούπα', 'βιβλίο', 'τηλεκοντρόλ', 'μπουκάλι', 'κλειδιά']


def build_world():
    """Place one object in each room, offset so it is never exactly on the
    room's own navigation waypoint — a fetch that only works when the object
    sits precisely on the goal point would be a fetch that never works."""
    rooms = _load_rooms()
    world = {}
    for i, (room, (x, y)) in enumerate(sorted(rooms.items())):
        if i >= len(OBJECTS):
            break
        # 0.6 m away, in a direction that varies per room.
        ang = (i * 2.0 * math.pi) / max(1, min(len(rooms), len(OBJECTS)))
        world[OBJECTS[i]] = {
            'room': room,
            'x': x + 0.6 * math.cos(ang),
            'y': y + 0.6 * math.sin(ang),
        }
    return world, rooms


# ── The virtual house ────────────────────────────────────────────────────────

def run_house(world, rooms, start_room, ready_evt, stop_evt, log, memory=None):
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
            super().__init__('virtual_house')
            self.cbg = ReentrantCallbackGroup()
            sx, sy = rooms[start_room]
            self.rx, self.ry, self.ryaw = sx, sy, 0.0
            self.world = {k: dict(v) for k, v in world.items()}
            # What object memory BELIEVES, which is not always the truth. None
            # means "mirror the world" (the easy case); otherwise it is an
            # explicit dict, and a label absent from it is unknown to memory —
            # that is what forces the room-by-room search fallback.
            self.memory = None if memory is None else {k: dict(v) for k, v in memory.items()}
            self.held = None                 # label currently in the gripper
            self.events = []                 # what the robot asked us to do
            # Rooms whose object memory has been refreshed by actually going
            # and looking. The search fallback re-queries after each drive, so
            # this is what makes "look for it" eventually succeed.
            self.seen_rooms = set()

            self.tf = TransformBroadcaster(self)
            self.create_timer(0.05, self._pub_tf)
            self.create_timer(0.10, self._pub_detections)

            self.mem_pub = self.create_publisher(String, 'object_memory/answer', 10)
            self.det_pub = self.create_publisher(String, 'detected_objects', 10)
            self.pick_pub = self.create_publisher(String, 'pick_result', 10)
            self.place_pub = self.create_publisher(String, 'place_result', 10)

            self.create_subscription(String, 'object_memory/query', self._on_query, 10)
            self.create_subscription(String, 'pick_command', self._on_pick, 10)
            self.create_subscription(String, 'place_command', self._on_place, 10)

            ActionServer(self, NavigateToPose, 'navigate_to_pose',
                         execute_callback=self._on_nav,
                         goal_callback=lambda _g: GoalResponse.ACCEPT,
                         cancel_callback=lambda _g: CancelResponse.ACCEPT,
                         callback_group=self.cbg)

        # ── the robot's own position, as TF ──
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

        # ── what the camera "sees" from where we are ──
        def _pub_detections(self):
            out = []
            for label, o in self.world.items():
                if label == self.held:
                    continue                      # in the gripper, not the scene
                dx, dy = o['x'] - self.rx, o['y'] - self.ry
                # map -> base_link, then base_link -> camera convention
                # (x right, z forward), which is what nearest_detection expects.
                fwd = dx * math.cos(self.ryaw) + dy * math.sin(self.ryaw)
                left = -dx * math.sin(self.ryaw) + dy * math.cos(self.ryaw)
                if fwd <= 0.1 or fwd > 4.0:
                    continue                      # behind us or out of range
                if abs(math.atan2(left, fwd)) > 0.60:
                    continue                      # outside a ~69 deg FOV
                out.append({'label': label, 'x': -left, 'y': 0.0, 'z': fwd})
                # Seeing a thing is what teaches object memory where it is —
                # this is the loop the live object_memory_node closes, and it
                # is what lets the search fallback converge.
                if self.memory is not None:
                    self.memory[label] = {'x': o['x'], 'y': o['y'],
                                          'room': o['room']}
            self.det_pub.publish(String(data=json.dumps(out)))

        # ── object memory ──
        def _on_query(self, msg):
            label = msg.data.strip()
            source = self.world if self.memory is None else self.memory
            o = source.get(label)
            self.events.append(('memory_query', label, 'hit' if o else 'miss'))
            if o is None or label == self.held:
                self.mem_pub.publish(String(data='{}'))
                return
            self.mem_pub.publish(String(data=json.dumps({
                'label': label, 'x': o['x'], 'y': o['y'],
                'room': o.get('room'),
                'last_seen': self.get_clock().now().nanoseconds / 1e9,
            })))

        # ── the arm ──
        def _on_pick(self, msg):
            try:
                req = json.loads(msg.data)
            except json.JSONDecodeError:
                req = {}
            label = req.get('label', '')
            o = self.world.get(label)
            # Refuse exactly as a real arm would: nothing to grab unless the
            # thing is actually within reach of where we parked.
            reach = None
            if o is not None:
                reach = math.hypot(o['x'] - self.rx, o['y'] - self.ry)
            ok = o is not None and reach is not None and reach <= 1.0
            self.events.append(('pick', label, round(reach, 2) if reach else None, ok))
            if ok:
                self.held = label
            self.pick_pub.publish(String(data=json.dumps(
                {'status': 'ok' if ok else 'failed',
                 'reason': None if ok else f'out of reach ({reach:.2f} m)'})))

        def _on_place(self, _msg):
            label = self.held
            self.events.append(('place', label, round(self.rx, 2), round(self.ry, 2)))
            if label:
                # Put it down wherever we are now — that is the delivery.
                self.world[label]['x'] = self.rx + 0.3 * math.cos(self.ryaw)
                self.world[label]['y'] = self.ry + 0.3 * math.sin(self.ryaw)
                self.world[label]['room'] = 'delivered'
                self.held = None
            self.place_pub.publish(String(data=json.dumps({'status': 'ok'})))

        # ── Nav2 stand-in ──
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
                # Whatever we are carrying comes with us.
                if self.held:
                    self.world[self.held]['x'] = self.rx
                    self.world[self.held]['y'] = self.ry
                time.sleep(STEP_DT)
            goal_handle.succeed()
            return NavigateToPose.Result()

    # rclpy.init() is the caller's job — the driver node below shares this
    # process and one context serves both.
    node = VirtualHouse()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    log['house'] = node
    ready_evt.set()
    while not stop_evt.is_set():
        ex.spin_once(timeout_sec=0.1)
    node.destroy_node()


# ── The test driver ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='91',
                    help='ROS_DOMAIN_ID for the simulation (NOT the robot\'s)')
    ap.add_argument('--object', action='append',
                    help='fetch only this object (repeatable)')
    ap.add_argument('--start-room', default='saloni')
    ap.add_argument('--keep-going', action='store_true',
                    help='run every object even after one fails')
    ap.add_argument('--timeout', type=float, default=90.0,
                    help='seconds to allow per fetch')
    ap.add_argument('--verbose', action='store_true',
                    help='dump what the house was asked to do, and the node log')
    ap.add_argument('--scenario', default='known',
                    choices=['known', 'unknown', 'stale', 'missing'],
                    help='known: object memory is correct (the easy case). '
                         'unknown: memory has never seen it, forcing the '
                         'room-by-room search. stale: memory points at the '
                         'wrong room because the object moved. missing: the '
                         'object is not in the house at all.')
    args = ap.parse_args()

    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)

    world, rooms = build_world()
    if args.start_room not in rooms:
        print(f'unknown --start-room {args.start_room}; have {sorted(rooms)}')
        return 2
    wanted = args.object or list(world)

    # Build what object memory believes, per scenario.
    memory = None
    if args.scenario == 'known':
        pass                                   # memory mirrors the world
    elif args.scenario == 'unknown':
        memory = {}                            # never seen anything
    elif args.scenario == 'stale':
        # Every object is remembered in the WRONG room — the position is real,
        # but it belongs to a different object. The robot should drive there,
        # fail to see the thing, and recover.
        labels = list(world)
        memory = {}
        for i, label in enumerate(labels):
            other = world[labels[(i + 1) % len(labels)]]
            memory[label] = {'x': other['x'], 'y': other['y'],
                             'room': other['room']}
    elif args.scenario == 'missing':
        # Ask for something that was never placed.
        wanted = ['ομπρέλα']
        memory = {}

    if args.scenario != 'missing':
        bad = [o for o in wanted if o not in world]
        if bad:
            print(f'unknown object(s): {bad}; have {sorted(world)}')
            return 2

    print(f'ROS_DOMAIN_ID={args.domain}  (the robot lives on 0 — untouched)')
    print(f'Scenario: {args.scenario}')
    print(f'Starting in {args.start_room}. The house:')
    for label, o in world.items():
        note = ''
        if memory is not None:
            m = memory.get(label)
            note = ('   memory: nothing' if m is None
                    else f'   memory says: {m.get("room")}')
        print(f'  {label:12s} in {o["room"]:20s} '
              f'at ({o["x"]:6.2f}, {o["y"]:6.2f}){note}')
    if args.scenario == 'missing':
        print(f'  asking for {wanted[0]!r}, which is not in the house at all')
    print()

    import rclpy
    rclpy.init()

    ready, stop, shared = threading.Event(), threading.Event(), {}
    house_thread = threading.Thread(
        target=run_house,
        args=(world, rooms, args.start_room, ready, stop, shared, memory),
        daemon=True)
    house_thread.start()
    if not ready.wait(20):
        print('the virtual house did not come up')
        return 3

    # The real node under test.
    exe = SRC.parents[1] / 'install' / 'home_robot' / 'lib' / 'home_robot' / 'mission_executor_node.py'
    if not exe.exists():
        print(f'mission_executor_node.py not found at {exe} — build the workspace')
        stop.set()
        return 3
    # ‼️ A leftover mission_executor from an earlier run is not a harmless
    # stray: it subscribes to the same mission/start on the same domain and
    # runs the WHOLE mission a second time, against this same virtual house.
    # The first version of this script hit exactly that after an aborted run —
    # every announcement printed twice, four navigations for a two-leg fetch,
    # and a second `place` that found the gripper already empty. It reads like
    # a bug in the node. Kill any before starting, and make sure this run
    # cannot leave one behind either.
    _reap_orphans(verbose=True)
    mission = subprocess.Popen(
        [sys.executable, str(exe), '--ros-args',
         # Trim the settle waits; they exist to let a real detector stabilise
         # and here the detections are exact from the first message.
         '-p', 'inspect_settle:=0.3',
         '-p', 'pick_timeout:=10.0', '-p', 'place_timeout:=10.0'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ),
        # If this script is killed outright (an outer `timeout`, Ctrl-C in a
        # way that skips the finally), the kernel takes the child with it
        # rather than leaving it to poison the next run.
        preexec_fn=_die_with_parent)
    mission_log = []
    threading.Thread(
        target=lambda: [mission_log.append(l.rstrip()) for l in mission.stdout],
        daemon=True).start()

    from rclpy.node import Node
    from std_msgs.msg import String

    class Driver(Node):
        def __init__(self):
            super().__init__('fetch_sim_driver')
            self.status = []
            self.speech = []
            self.create_subscription(String, 'mission/status', self._st, 10)
            self.create_subscription(String, 'speech_response', self._sp, 10)
            self.start_pub = self.create_publisher(String, 'mission/start', 10)

        def _st(self, m):
            self.status.append(m.data)

        def _sp(self, m):
            self.speech.append(m.data)
            print(f'    🤖 {m.data}')

    drv = Driver()

    def spin_for(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(drv, timeout_sec=0.05)

    print('waiting for mission_executor_node…')
    spin_for(6.0)

    # 'missing' is a success when the robot gives up cleanly instead of
    # hunting forever or claiming it delivered something.
    expected_verdict = 'failed' if args.scenario == 'missing' else 'done'

    results = {}
    try:
        for label in wanted:
            where = world[label]['room'] if label in world else 'nowhere'
            print(f'── fetch: {label}  (in {where}) {"─" * 20}')
            drv.status.clear()
            drv.speech.clear()
            house = shared['house']
            house.events.clear()
            drv.start_pub.publish(String(data=f'fetch:{label}'))

            end = time.monotonic() + args.timeout
            verdict = None
            while time.monotonic() < end:
                rclpy.spin_once(drv, timeout_sec=0.05)
                if 'done' in drv.status:
                    verdict = 'done'
                    break
                if 'failed' in drv.status:
                    verdict = 'failed'
                    break
                if 'cancelled' in drv.status:
                    verdict = 'cancelled'
                    break
            verdict = verdict or 'timeout'

            navs = [e for e in house.events if e[0] == 'navigate']
            picks = [e for e in house.events if e[0] == 'pick']
            places = [e for e in house.events if e[0] == 'place']
            carried = bool(places and places[0][1] == label)
            delivered_near_start = False
            if places:
                sx, sy = rooms[args.start_room]
                delivered_near_start = math.hypot(places[0][2] - sx,
                                                  places[0][3] - sy) < 1.0
            results[label] = {
                'verdict': verdict, 'navigations': len(navs),
                'picked': any(p[3] for p in picks),
                'released': carried,
                'delivered_at_start': delivered_near_start,
            }
            print(f'  → {verdict}  ({len(navs)} navigations, '
                  f'picked={results[label]["picked"]}, '
                  f'released={carried}, back_at_start={delivered_near_start})')
            if args.verbose:
                for e in house.events:
                    print(f'      · {e}')
            print()
            if verdict != expected_verdict and not args.keep_going:
                break
            spin_for(2.5)      # let the node return to idle
    finally:
        mission.terminate()
        try:
            mission.wait(timeout=10)
        except subprocess.TimeoutExpired:
            mission.kill()
            mission.wait(timeout=5)
        _reap_orphans()
        drv.destroy_node()
        rclpy.try_shutdown()
        stop.set()
        house_thread.join(timeout=5)

    print('=' * 62)
    ok = 0
    for label in wanted:
        r = results.get(label)
        if r is None:
            print(f'  {label:12s} not run')
            continue
        if expected_verdict == 'failed':
            # Giving up correctly: said so, and did NOT pretend to carry
            # anything back.
            good = r['verdict'] == 'failed' and not r['picked']
        else:
            good = (r['verdict'] == 'done' and r['picked'] and r['released']
                    and r['delivered_at_start'])
        ok += good
        print(f'  {"✅" if good else "❌"} {label:12s} {r["verdict"]:9s} '
              f'picked={str(r["picked"]):5s} released={str(r["released"]):5s} '
              f'back_at_start={str(r["delivered_at_start"]):5s}')
    label_word = ('clean give-ups' if expected_verdict == 'failed'
                  else 'complete round trips')
    print(f'{ok}/{len(wanted)} {label_word}')

    if ok != len(wanted) or args.verbose:
        print('\n── last 40 lines from mission_executor_node ──')
        for line in mission_log[-40:]:
            print('  ' + line)
    return 0 if ok == len(wanted) else 1


if __name__ == '__main__':
    sys.exit(main())
