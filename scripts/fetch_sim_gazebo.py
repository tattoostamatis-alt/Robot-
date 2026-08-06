#!/usr/bin/env python3
"""Fetch missions over the REAL Nav2 stack, in Gazebo. No hardware.

This is the heavier sibling of fetch_sim.py. There the navigation was a
straight line with no planner and no walls; here the robot drives the actual
malou2 apartment through the actual navigation stack — global planner, MPPI,
costmaps, collision monitor, AMCL, and the doorway recovery manager — while a
simulated lidar sees walls extruded from the same map file the real robot
localises against.

What is real:
  * Gazebo physics and the diff-drive base, driven over cmd_vel_safe
  * the whole Nav2 configuration from nav2_params.yaml
  * mission_executor_node, started as a subprocess and talked to over topics

What is simulated, and why:
  * PERCEPTION. The objects are coloured boxes; a camera would render them
    fine but YOLO would call them "box", not "κούπα". Detections are generated
    from ground truth and the robot's actual AMCL pose, gated by a real field
    of view and range — so the robot still has to arrive, and still has to be
    facing the thing, before it can see it.
  * THE ARM. There is no arm in the sim model. A grasp succeeds only if the
    base genuinely ended up within reach of the object's true position, so the
    approach is tested honestly even though the grasp itself is not.

    ‼️ ROS_DOMAIN_ID defaults to 77 (the sim convention here), never the
    robot's 0 — a simulated Nav2 and a simulated map->base_link on the live
    domain would fight the real ones.

Usage:
    scripts/fetch_sim_gazebo.py                    # one fetch, another room
    scripts/fetch_sim_gazebo.py --object κούπα
    scripts/fetch_sim_gazebo.py --all              # every object in turn
    scripts/fetch_sim_gazebo.py --gui              # watch it in Gazebo + RViz
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
WORLD = SRC / 'worlds' / 'malou2.world'
GROUND_TRUTH = Path(str(WORLD) + '.objects.json')
MAP_YAML = SRC / 'maps' / 'malou2.yaml'

# Where the robot starts and delivers back to: the saloni waypoint.
START = (-1.262, 0.586, 0.0)

# Camera model for the simulated detector.
FOV_HALF = 0.60          # rad, ~69 deg total — the D435's colour FOV
SEE_RANGE = 4.0          # m
GRASP_RANGE = 1.0        # m — must match the mission's fetch_grasp_range

_MISSION_MARK = 'lib/home_robot/mission_executor_node.py'

# The driver node's dedicated executor. Set once in main(); everything that
# needs to service the driver's callbacks goes through spin_drv() so it never
# collides with the perception thread's executor.
_DRV_EX = None


def spin_drv(seconds=0.0):
    """Service the driver node for `seconds` (one pass if 0)."""
    if seconds <= 0:
        _DRV_EX.spin_once(timeout_sec=0.05)
        return
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        _DRV_EX.spin_once(timeout_sec=0.05)


def _die_with_parent():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').prctl(1, 9, 0, 0, 0)
    except Exception:
        pass


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


SIM_DISPLAY = ':7'          # 2-5 belong to the dashboard's VNC tabs


def ensure_display(verbose=False):
    """Give Gazebo an X display, or its renderer aborts on startup.

    ‼️ headless:=true is NOT display-less. sim.launch.py runs Ogre 1.x
    (ogre2/EGL segfaults on this machine's Mesa), and Ogre 1.x renders through
    GLX — it needs an X server even with --headless-rendering, because that
    flag still runs the SERVER's sensors system, which is what renders the
    simulated lidar. Started from a plain terminal with no DISPLAY, gz dies
    with SIGABRT in RenderSystem_GL's dllStartPlugin:

        #11 RenderSystem_GL.so, in dllStartPlugin
        ...
        Aborted (core dumped) — exit code 134

    and every Nav2 node then logs "Invalid frame ID odom" for ever, because
    /odom never appears. The crash scrolls past long before those do, so what
    you notice is the Nav2 spam.

    A headless Xvnc is enough. LIBGL_ALWAYS_SOFTWARE is set because Xvnc has
    no GPU (same reasoning as scripts/gui_session.sh).
    """
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    if os.environ.get('DISPLAY') and os.path.exists(
            '/tmp/.X11-unix/X' + os.environ['DISPLAY'].lstrip(':').split('.')[0]):
        if verbose:
            print(f'  using existing DISPLAY={os.environ["DISPLAY"]}')
        return True

    sock = '/tmp/.X11-unix/X' + SIM_DISPLAY.lstrip(':')
    if not os.path.exists(sock):
        if verbose:
            print(f'  starting headless Xvnc on {SIM_DISPLAY} for Gazebo…')
        # ‼️ EMPTY xstartup. tigervncserver runs ~/.vnc/xstartup by default,
        # and on this machine that file launches RViz — so merely asking for
        # an X server spawned a second RViz at ~110% CPU, on top of gz. Load
        # average hit 37, map_server's Configuring transition never finished,
        # and the lifecycle STARTUP timed out. All we want here is the display.
        stub = '/tmp/fetch_sim_xstartup'
        with open(stub, 'w') as fh:
            fh.write('#!/bin/sh\nexec sleep infinity\n')
        os.chmod(stub, 0o755)
        sh(f'tigervncserver {SIM_DISPLAY} -localhost yes -geometry 1280x720 '
           f'-depth 24 -SecurityTypes None -xstartup {stub} >/dev/null 2>&1')
        for _ in range(20):
            if os.path.exists(sock):
                break
            time.sleep(0.5)
    if not os.path.exists(sock):
        print(f'  could not start an X display on {SIM_DISPLAY}')
        return False
    os.environ['DISPLAY'] = SIM_DISPLAY
    if verbose:
        print(f'  DISPLAY={SIM_DISPLAY}')
    return True


def reap(verbose=False):
    """Kill anything left over from an earlier sim run.

    ‼️ gz ignores SIGTERM (see feedback_gazebo_orphan_clock): an orphaned
    server keeps publishing /clock, and the next launch then sits forever
    waiting for a simulation that is already running somewhere else.
    """
    # ‼️ The Nav2/home_robot nodes the launch starts are NOT children that die
    # with it — three recovery_manager_node processes from earlier runs were
    # found alive at once, each ~20% CPU, which is most of why a later run's
    # map_server never finished configuring.
    pats = ['ros2 launch home_robot sim.launch.py', 'gz sim', 'ruby.*gz',
            _MISSION_MARK, 'parameter_bridge', 'fetch_sim_perception',
            'recovery_manager_node', 'odom_tf_broadcaster', 'goal_pose_bridge',
            'nav2_controller/controller_server', 'nav2_planner/planner_server',
            'nav2_bt_navigator', 'nav2_amcl/amcl', 'nav2_map_server/map_server',
            'nav2_lifecycle_manager', 'nav2_smoother', 'nav2_behaviors',
            'nav2_velocity_smoother', 'nav2_collision_monitor',
            'nav2_waypoint_follower', 'nav2_route', 'opennav_docking',
            'robot_state_publisher']
    for p in pats:
        sh(f'pkill -9 -f "{p}"')
    if verbose:
        print('  (cleaned up any previous sim)')
    time.sleep(2)


# ── simulated perception + arm ───────────────────────────────────────────────

def run_perception(objects, ready_evt, stop_evt, shared):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    import tf2_ros
    from std_msgs.msg import String

    class SimPerception(Node):
        def __init__(self):
            super().__init__('fetch_sim_perception')
            self.set_parameters([rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
            self.objects = {k: dict(v) for k, v in objects.items()}
            self.held = None
            self.events = []
            self.pose = None

            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            q = QoSProfile(depth=10)
            self.det_pub = self.create_publisher(String, 'detected_objects', q)
            self.mem_pub = self.create_publisher(String, 'object_memory/answer', q)
            self.pick_pub = self.create_publisher(String, 'pick_result', q)
            self.place_pub = self.create_publisher(String, 'place_result', q)
            self.create_subscription(String, 'object_memory/query', self._on_query, q)
            self.create_subscription(String, 'pick_command', self._on_pick, q)
            self.create_subscription(String, 'place_command', self._on_place, q)
            self.create_timer(0.2, self._tick)

        def _base_pose(self):
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
            except Exception:
                return None
            t, r = tf.transform.translation, tf.transform.rotation
            yaw = math.atan2(2 * (r.w * r.z + r.x * r.y),
                             1 - 2 * (r.y * r.y + r.z * r.z))
            return t.x, t.y, yaw

        def _tick(self):
            p = self._base_pose()
            if p is None:
                return
            self.pose = p
            rx, ry, ryaw = p
            out = []
            for label, o in self.objects.items():
                if label == self.held:
                    continue
                dx, dy = o['x'] - rx, o['y'] - ry
                fwd = dx * math.cos(ryaw) + dy * math.sin(ryaw)
                left = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
                if fwd <= 0.1 or fwd > SEE_RANGE:
                    continue
                if abs(math.atan2(left, fwd)) > FOV_HALF:
                    continue
                out.append({'label': label, 'x': -left, 'y': 0.0, 'z': fwd})
                # Seeing it is what teaches object memory where it is.
                o['seen'] = True
            self.det_pub.publish(String(data=json.dumps(out)))

        def _on_query(self, msg):
            label = msg.data.strip()
            o = self.objects.get(label)
            known = o is not None and o.get('seen') and label != self.held
            self.events.append(('memory_query', label, 'hit' if known else 'miss'))
            if not known:
                self.mem_pub.publish(String(data='{}'))
                return
            self.mem_pub.publish(String(data=json.dumps({
                'label': label, 'x': o['x'], 'y': o['y'], 'room': o.get('room'),
                'last_seen': self.get_clock().now().nanoseconds / 1e9})))

        def _on_pick(self, msg):
            try:
                label = json.loads(msg.data).get('label', '')
            except json.JSONDecodeError:
                label = ''
            o = self.objects.get(label)
            p = self.pose
            reach = (math.hypot(o['x'] - p[0], o['y'] - p[1])
                     if o and p else None)
            ok = reach is not None and reach <= GRASP_RANGE
            self.events.append(('pick', label,
                                round(reach, 2) if reach is not None else None, ok))
            if ok:
                self.held = label
            self.pick_pub.publish(String(data=json.dumps(
                {'status': 'ok' if ok else 'failed',
                 'reason': None if ok else 'out of reach'})))

        def _on_place(self, _msg):
            p = self.pose or (0.0, 0.0, 0.0)
            self.events.append(('place', self.held, round(p[0], 2), round(p[1], 2)))
            if self.held:
                self.objects[self.held].update(
                    {'x': p[0] + 0.3 * math.cos(p[2]), 'y': p[1] + 0.3 * math.sin(p[2]),
                     'room': 'delivered'})
                self.held = None
            self.place_pub.publish(String(data=json.dumps({'status': 'ok'})))

    # ‼️ Its OWN executor. rclpy.spin_once(node) borrows a shared one, and with
    # the driver node spinning on the main thread at the same time both raise
    # "RuntimeError: Executor is already spinning" the moment they overlap.
    from rclpy.executors import SingleThreadedExecutor
    node = SimPerception()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    shared['perception'] = node
    ready_evt.set()
    while not stop_evt.is_set():
        ex.spin_once(timeout_sec=0.1)
    ex.remove_node(node)
    node.destroy_node()


# ── waiting for the stack ────────────────────────────────────────────────────

def wait_for_nav(drv, rclpy, timeout=420.0):
    """Block until the sim is actually usable, in the order things come up.

    ‼️ Gazebo takes far longer to start than sim.launch.py assumes. On this
    machine a ~500-wall apartment needed **152 s** before /clock produced its
    first tick — the launch file starts Nav2 on a 12 s timer and publishes the
    AMCL seed pose at 22 s, both of which land in a world that does not exist
    yet. The seed is dropped (no TF to transform it into), AMCL never
    initialises, map->base_link never appears, and every costmap logs
    "Invalid frame ID odom ... frame does not exist" forever. It looks exactly
    like a broken Nav2 config; it is a race with world load.

    So: wait for the clock, then for odom, and only then seed AMCL ourselves.
    """
    import tf2_ros
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from geometry_msgs.msg import PoseWithCovarianceStamped

    seen = {'clock': 0, 'odom': 0}
    drv.create_subscription(Clock, '/clock',
                            lambda _m: seen.__setitem__('clock', seen['clock'] + 1), 10)
    drv.create_subscription(Odometry, '/odom',
                            lambda _m: seen.__setitem__('odom', seen['odom'] + 1), 10)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, drv)
    ac = ActionClient(drv, NavigateToPose, 'navigate_to_pose')
    pose_pub = drv.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    spin = spin_drv

    def wait_until(name, pred, budget):
        end = time.monotonic() + budget
        while time.monotonic() < end:
            spin_drv()
            if pred():
                print(f'  {name} after {budget - (end - time.monotonic()):.0f}s')
                return True
        print(f'  TIMEOUT waiting for {name}')
        return False

    t0 = time.monotonic()
    if not wait_until('/clock ticking (Gazebo is stepping physics)',
                      lambda: seen['clock'] > 5, timeout):
        return False
    if not wait_until('/odom flowing (the base is spawned and bridged)',
                      lambda: seen['odom'] > 5, 120.0):
        return False

    # ‼️ Bring the Nav2 lifecycle up OURSELVES, now that the world exists.
    #
    # sim.launch.py activates it on a 12 s TimerAction. With Gazebo still
    # loading, controller_server's activation blocked for 13 s waiting for a
    # transform, planner_server then timed out, and the manager gave up on the
    # whole chain:
    #
    #   [lifecycle_manager_navigation] Failed to change state for node:
    #                                  planner_server
    #   [lifecycle_manager_navigation] Failed to bring up all requested nodes.
    #                                  Aborting bringup.
    #
    # Nothing retries after that. AMCL stayed `unconfigured` for ever and
    # rejected every seed pose with "AMCL is not yet in the active state", so
    # map->base_link never existed. STARTUP is idempotent — it re-runs the
    # configure/activate chain on nodes that never got there.
    #
    # ‼️ A FAILED STARTUP is not necessarily a problem. If the launch timer
    # already got the chain up, a second STARTUP tries to configure nodes that
    # are active and the manager reports failure — "Failed to change state for
    # node: controller_server" one line after "Managed nodes are active". So
    # this is advisory; whether the stack works is decided below, by asking
    # for the TF and the action server.
    _lifecycle_startup(drv, rclpy, 'lifecycle_manager_localization')

    # Seed AMCL now that TF and the clock exist.
    # ‼️ stamp 0, never now(): a current stamp asks TF for a transform in the
    # future and the pose is silently dropped (project_robot_initialpose_dropped
    # — the same bug bit all three publishers on the real robot).
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.pose.pose.position.x = START[0]
    msg.pose.pose.position.y = START[1]
    msg.pose.pose.orientation.w = 1.0
    msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
    msg.pose.covariance[35] = 0.07
    for _ in range(5):
        pose_pub.publish(msg)
        spin(0.5)
    print('  seeded AMCL at the saloni start pose')

    if not wait_until('map->base_link (AMCL localised)',
                      lambda: _has_tf(buf, rclpy), 120.0):
        return False
    if not ac.server_is_ready():
        _lifecycle_startup(drv, rclpy, 'lifecycle_manager_navigation')
    if not wait_until('navigate_to_pose accepting goals',
                      lambda: ac.server_is_ready(), 120.0):
        return False
    print(f'  stack ready in {time.monotonic() - t0:.0f}s total')
    return True


def _lifecycle_startup(node, rclpy, manager, timeout=180.0):
    """Ask a Nav2 lifecycle manager to (re-)run its STARTUP chain."""
    from nav2_msgs.srv import ManageLifecycleNodes

    cli = node.create_client(ManageLifecycleNodes, f'/{manager}/manage_nodes')
    end = time.monotonic() + 30
    while not cli.service_is_ready() and time.monotonic() < end:
        spin_drv()
    if not cli.service_is_ready():
        print(f'  {manager}: manage_nodes service never appeared')
        return False

    req = ManageLifecycleNodes.Request()
    req.command = ManageLifecycleNodes.Request.STARTUP
    fut = cli.call_async(req)
    end = time.monotonic() + timeout
    while not fut.done() and time.monotonic() < end:
        spin_drv()
    if not fut.done():
        print(f'  {manager}: STARTUP timed out')
        return False
    ok = bool(fut.result() and fut.result().success)
    print(f'  {manager}: STARTUP {"ok" if ok else "FAILED"}')
    return ok


def _has_tf(buf, rclpy):
    try:
        buf.lookup_transform('map', 'base_link', rclpy.time.Time(),
                             timeout=rclpy.duration.Duration(seconds=0.1))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='77')
    ap.add_argument('--object', action='append')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--gui', action='store_true', help='show Gazebo and RViz')
    ap.add_argument('--timeout', type=float, default=300.0)
    ap.add_argument('--keep-alive', action='store_true',
                    help='leave the sim running afterwards')
    args = ap.parse_args()

    if not GROUND_TRUTH.exists():
        print(f'{GROUND_TRUTH} missing — run:\n'
              f'  scripts/map_to_world.py maps/malou2.yaml worlds/malou2.world\n'
              f'  scripts/sim_place_objects.py maps/malou2.yaml worlds/malou2.world')
        return 2
    objects = json.loads(GROUND_TRUTH.read_text())

    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    print(f'ROS_DOMAIN_ID={args.domain}  (the robot lives on 0 — untouched)')

    wanted = args.object or (list(objects) if args.all else None)
    if wanted is None:
        # Default: one object, and deliberately not the one in the starting
        # room — the point is to go to ANOTHER room and come back.
        wanted = [l for l, o in objects.items() if o['room'] != 'saloni'][:1]
    bad = [o for o in wanted if o not in objects]
    if bad:
        print(f'unknown object(s) {bad}; have {sorted(objects)}')
        return 2

    print('The house:')
    for label, o in objects.items():
        mark = '←' if label in wanted else ' '
        print(f'  {mark} {label:12s} in {o["room"]:20s} '
              f'at ({o["x"]:6.2f}, {o["y"]:6.2f})')
    print(f'Robot starts at saloni {START[:2]} and delivers back there.\n')

    reap(verbose=True)
    if not ensure_display(verbose=True):
        return 3

    launch = [
        'ros2', 'launch', 'home_robot', 'sim.launch.py',
        f'world:={WORLD}', f'map:={MAP_YAML}',
        f'init_x:={START[0]}', f'init_y:={START[1]}', f'init_yaw:={START[2]}',
        f'use_rviz:={"true" if args.gui else "false"}',
        f'headless:={"false" if args.gui else "true"}',
    ]
    print('starting Gazebo + Nav2 — the world load alone takes 2-3 minutes, '
          'be patient…')
    sim_env = dict(os.environ)
    sim = subprocess.Popen(launch, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True,
                           env=sim_env, preexec_fn=_die_with_parent)
    # Tee to a file as well as memory: when the stack fails to come up the
    # reason is almost always in here, and holding it only in a list means it
    # is gone the moment this process exits.
    sim_log = []
    sim_log_path = '/tmp/fetch_sim_gazebo.log'

    def _pump():
        with open(sim_log_path, 'w') as fh:
            for line in sim.stdout:
                sim_log.append(line.rstrip())
                fh.write(line)
                fh.flush()
    threading.Thread(target=_pump, daemon=True).start()
    print(f'  (full launch log: {sim_log_path})')

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()

    class Driver(Node):
        def __init__(self):
            super().__init__('fetch_sim_driver')
            self.set_parameters([rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
            self.status, self.speech = [], []
            self.create_subscription(String, 'mission/status',
                                     lambda m: self.status.append(m.data), 10)
            self.create_subscription(String, 'speech_response', self._sp, 10)
            self.start_pub = self.create_publisher(String, 'mission/start', 10)

        def _sp(self, m):
            self.speech.append(m.data)
            print(f'    🤖 {m.data}')

    global _DRV_EX
    from rclpy.executors import SingleThreadedExecutor
    drv = Driver()
    _DRV_EX = SingleThreadedExecutor()
    _DRV_EX.add_node(drv)
    mission = None
    stop = threading.Event()
    perc_thread = None
    results = {}

    try:
        if not wait_for_nav(drv, rclpy):
            print('\nNav2 never came up. Last 40 lines:')
            for l in sim_log[-40:]:
                print('  ' + l)
            return 3

        ready, shared = threading.Event(), {}
        perc_thread = threading.Thread(
            target=run_perception, args=(objects, ready, stop, shared), daemon=True)
        perc_thread.start()
        ready.wait(20)
        print('  simulated perception up')

        exe = (SRC.parents[1] / 'install' / 'home_robot' / 'lib' / 'home_robot'
               / 'mission_executor_node.py')
        mission = subprocess.Popen(
            [sys.executable, str(exe), '--ros-args',
             '-p', 'use_sim_time:=true',
             '-p', 'inspect_settle:=1.0',
             '-p', 'pick_timeout:=15.0', '-p', 'place_timeout:=15.0'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=dict(os.environ), preexec_fn=_die_with_parent)
        mission_log = []
        threading.Thread(
            target=lambda: [mission_log.append(l.rstrip()) for l in mission.stdout],
            daemon=True).start()
        time.sleep(5)
        print('  mission_executor up\n')

        perception = shared['perception']
        for label in wanted:
            o = objects[label]
            print(f'── «φέρε μου {label}» (it is in {o["room"]}) ' + '─' * 16)
            drv.status.clear()
            perception.events.clear()
            t0 = time.monotonic()
            drv.start_pub.publish(String(data=f'fetch:{label}'))

            verdict = None
            end = time.monotonic() + args.timeout
            while time.monotonic() < end:
                spin_drv()
                for v in ('done', 'failed', 'cancelled'):
                    if v in drv.status:
                        verdict = v
                        break
                if verdict:
                    break
            verdict = verdict or 'timeout'
            elapsed = time.monotonic() - t0

            picks = [e for e in perception.events if e[0] == 'pick']
            places = [e for e in perception.events if e[0] == 'place']
            picked = any(p[3] for p in picks)
            released = bool(places and places[0][1] == label)
            back = bool(places and math.hypot(places[0][2] - START[0],
                                              places[0][3] - START[1]) < 1.5)
            results[label] = {'verdict': verdict, 'picked': picked,
                              'released': released, 'back': back,
                              'seconds': round(elapsed, 1)}
            print(f'  → {verdict} in {elapsed:.0f}s  picked={picked} '
                  f'released={released} back_at_start={back}')
            for e in perception.events:
                print(f'      · {e}')
            print()
            time.sleep(3)
    finally:
        if not args.keep_alive:
            stop.set()
            if perc_thread:
                perc_thread.join(timeout=5)
            for p in (mission, sim):
                if p:
                    p.terminate()
            time.sleep(2)
            drv.destroy_node()
            rclpy.try_shutdown()
            reap()
        else:
            print('\n--keep-alive: sim still running. Clean up with:')
            print('  pkill -9 -f "gz sim"; pkill -9 -f sim.launch.py')

    print('=' * 64)
    ok = 0
    for label in wanted:
        r = results.get(label)
        if not r:
            print(f'  {label} not run')
            continue
        good = r['verdict'] == 'done' and r['picked'] and r['released'] and r['back']
        ok += good
        print(f'  {"✅" if good else "❌"} {label:12s} {r["verdict"]:8s} '
              f'{r["seconds"]:6.1f}s  picked={str(r["picked"]):5s} '
              f'released={str(r["released"]):5s} back={str(r["back"]):5s}')
    print(f'{ok}/{len(wanted)} complete round trips through the real Nav2 stack')
    return 0 if ok == len(wanted) else 1


if __name__ == '__main__':
    sys.exit(main())
