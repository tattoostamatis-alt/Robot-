#!/usr/bin/env python3
"""Drive to every named room in locations.yaml, one after the other.

`goto_room.py` sends one goal and exits. Doing a whole flat by hand meant
re-running it per room and watching for the failures, which is exactly the
loop that found the saloni/kouzina sticking points — so it lives here now,
with the two things that made those runs finish instead of stall:

  * the costmaps are cleared before every leg. The D435 drops corrupted
    depth frames, and a phantom mark left over from the previous leg is
    what makes the next one report "Collision Ahead" from a standing
    start (see project_robot_saloni_exit_chokepoint). recovery_manager
    clears them during a recovery; nobody cleared them BETWEEN goals.
  * a leg that fails is not the end of the tour. It goes to the back of
    the queue and gets one more attempt after the other rooms — by then
    the robot is approaching from somewhere else entirely, which is
    usually enough.

Speed is measured, not assumed: /odom twist is sampled the whole way, so
the report says what the robot actually did rather than what vx_max says
it may do. If the average sits far below the configured ceiling, the
limit is somewhere downstream (CostCritic in tight rooms, the collision
monitor, or the wheels really slipping) and the numbers per leg say which
leg to look at.

    ros2 run home_robot room_tour.py                     # every room, nearest first
    ros2 run home_robot room_tour.py --rooms kouzina,toualeta
    ros2 run home_robot room_tour.py --order yaml --dwell 5
    ros2 run home_robot room_tour.py --loop 3 --speak
"""

import argparse
import math
import sys
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# dock is not plannable (the base itself is the obstacle) and dock_staging is
# a docking approach, not a room — `robot dock` owns both.
SKIP = {'dock', 'dock_staging'}


def load_locations() -> dict:
    path = get_package_share_directory('home_robot') + '/config/locations.yaml'
    with open(path) as f:
        return yaml.safe_load(f) or {}


class Leg:
    """One room visit and what came of it."""

    def __init__(self, room: str, pose: dict):
        self.room = room
        self.pose = pose
        self.status = 'δεν δοκιμάστηκε'
        self.seconds = 0.0
        self.distance = 0.0
        self.v_avg = 0.0
        self.v_max = 0.0
        self.attempts = 0

    @property
    def ok(self) -> bool:
        return self.status == 'έφτασε'


class RoomTour(Node):
    def __init__(self, args):
        super().__init__('room_tour')
        self._args = args
        self._cb = ReentrantCallbackGroup()

        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                 callback_group=self._cb)
        self._clear_local = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap',
            callback_group=self._cb)
        self._clear_global = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap',
            callback_group=self._cb)

        self._speech = self.create_publisher(String, 'speech_response', 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10,
                                 callback_group=self._cb)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Odometry accumulator, reset at the start of each leg.
        self._lock = threading.Lock()
        self._odom_reset()

        self.legs: list[Leg] = []
        self.done = threading.Event()

    # ── odometry sampling ────────────────────────────────────────────

    def _odom_reset(self):
        with self._lock:
            self._dist = 0.0
            self._v_sum = 0.0
            self._v_n = 0
            self._v_max = 0.0
            self._last_xy = None

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        v = abs(msg.twist.twist.linear.x)
        with self._lock:
            if self._last_xy is not None:
                self._dist += math.dist((p.x, p.y), self._last_xy)
            self._last_xy = (p.x, p.y)
            # Only average while actually driving: the stationary samples
            # during planning/recovery would drag the mean toward zero and
            # hide what the robot does when it IS moving.
            if v > 0.02:
                self._v_sum += v
                self._v_n += 1
                self._v_max = max(self._v_max, v)

    def _odom_stats(self):
        with self._lock:
            avg = self._v_sum / self._v_n if self._v_n else 0.0
            return self._dist, avg, self._v_max

    # ── plumbing ─────────────────────────────────────────────────────

    def _await(self, future, timeout_sec):
        """Wait on a future from the tour thread without spinning here — the
        executor in main() services the callbacks. See
        recovery_manager_node._await_future for why spinning would deadlock.
        """
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return future.result() if future.done() else None

    def clear_costmaps(self):
        for name, client in (('local', self._clear_local),
                             ('global', self._clear_global)):
            if not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f'{name} costmap: το service λείπει')
                continue
            if self._await(client.call_async(ClearEntireCostmap.Request()), 3.0) is None:
                self.get_logger().warn(f'{name} costmap: timeout στο clear')

    def current_xy(self, timeout_sec=5.0):
        """Where the robot is in map frame, or None if TF is not up yet."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                tf = self._tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())
                return (tf.transform.translation.x, tf.transform.translation.y)
            except Exception:
                time.sleep(0.2)
        return None

    def speak(self, text: str):
        self.get_logger().info(text)
        if self._args.speak:
            self._speech.publish(String(data=text))

    # ── one leg ──────────────────────────────────────────────────────

    def _goal_msg(self, leg: Leg) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        # stamp 0 = "the latest TF you have". A now() stamp asks for a
        # transform in the future and gets dropped —
        # project_robot_initialpose_dropped.
        goal.pose.header.stamp = rclpy.time.Time().to_msg()
        goal.pose.pose.position.x = float(leg.pose['x'])
        goal.pose.pose.position.y = float(leg.pose['y'])
        yaw = float(leg.pose.get('yaw', 0.0))
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        return goal

    def drive_to(self, leg: Leg) -> bool:
        """Keep after one room until it is reached or its time is up.

        ‼️ A CANCELED or ABORTED goal is NOT a failed room. recovery_manager
        owns unsticking the robot, and its very first move is to cancel the
        active NavigateToPose so it can BackUp — the owner of that goal (us)
        gets CANCELED. It then re-issues the SAME goal itself, which preempts
        whatever the owner started next. Treating each cancel as a failed room
        turned one pinch point into a cascade: measured 2026-08-09, the robot
        was driving correctly the whole time ("Begin navigating from (-2.74,
        0.94)") while the tour marked five rooms failed in 90 s and skipped
        past each one mid-recovery.

        So the budget for a room is TIME, not attempts: re-send and wait again
        for as long as `--timeout` allows. Re-sending the identical goal while
        recovery_manager has one in flight is harmless — same pose, so the
        preemption is a no-op in terms of where the robot is headed.
        """
        leg.attempts += 1
        self.clear_costmaps()
        self._odom_reset()
        self.speak(f'Πάω στο {leg.room}')
        started = time.monotonic()
        deadline = started + self._args.timeout
        sends = 0

        while rclpy.ok() and time.monotonic() < deadline:
            sends += 1
            handle = self._await(self._nav.send_goal_async(self._goal_msg(leg)), 10.0)
            if handle is None or not handle.accepted:
                leg.status = 'το goal δεν έγινε δεκτό'
                leg.seconds = time.monotonic() - started
                return False

            result = self._await(handle.get_result_async(),
                                 max(deadline - time.monotonic(), 1.0))
            leg.seconds = time.monotonic() - started
            leg.distance, leg.v_avg, leg.v_max = self._odom_stats()

            if result is None:                      # out of time, still going
                self.get_logger().warn(
                    f'{leg.room}: πέρασαν {self._args.timeout:.0f}s — ακυρώνω')
                self._await(handle.cancel_goal_async(), 5.0)
                leg.status = f'timeout ({self._args.timeout:.0f}s)'
                return False

            if result.status == GoalStatus.STATUS_SUCCEEDED:
                leg.status = 'έφτασε'
                xy = self.current_xy(2.0)
                gap = (math.dist(xy, (leg.pose['x'], leg.pose['y']))
                       if xy else float('nan'))
                self.speak(f'Έφτασα στο {leg.room}')
                self.get_logger().info(
                    f'{leg.room}: {leg.seconds:.0f}s, {leg.distance:.1f}m, '
                    f'μέση {leg.v_avg:.2f} m/s, μέγιστη {leg.v_max:.2f} m/s, '
                    f'{gap:.2f}m από τον στόχο, {sends} goals')
                return True

            # Cancelled/aborted: almost always recovery_manager doing its job.
            # Give it room to finish the manoeuvre, then ask again.
            self.get_logger().info(
                f'{leg.room}: status={result.status} μετά {leg.seconds:.0f}s '
                f'— πιθανό recovery, ξαναστέλνω (απομένουν '
                f'{deadline - time.monotonic():.0f}s)')
            time.sleep(3.0)

        leg.status = f'δεν τα κατάφερε σε {self._args.timeout:.0f}s'
        leg.seconds = time.monotonic() - started
        self.get_logger().warn(f'{leg.room}: {leg.status} ({sends} goals)')
        return False

    # ── the tour ─────────────────────────────────────────────────────

    def order_rooms(self, rooms: dict) -> list:
        """Nearest-neighbour from where the robot stands. Not optimal (that
        would be a TSP) but it stops the tour from crossing the flat twice
        for two rooms that share a doorway."""
        if self._args.order == 'yaml':
            return list(rooms.items())

        here = self.current_xy()
        if here is None:
            self.get_logger().warn(
                'δεν βρήκα TF map→base_link — κρατάω τη σειρά του yaml')
            return list(rooms.items())

        remaining = dict(rooms)
        ordered = []
        while remaining:
            name = min(remaining,
                       key=lambda n: math.dist(here, (remaining[n]['x'],
                                                      remaining[n]['y'])))
            pose = remaining.pop(name)
            ordered.append((name, pose))
            here = (pose['x'], pose['y'])
        return ordered

    def run(self):
        try:
            self._run()
        finally:
            self.done.set()

    def _run(self):
        self.get_logger().info('Αναμονή για το Nav2...')
        if not self._nav.wait_for_server(timeout_sec=30.0):
            self.get_logger().error(
                'το navigate_to_pose δεν απαντά — τρέχει το bringup; '
                '(scripts/ensure_nav_active.sh)')
            return

        if self.current_xy() is None:
            self.get_logger().error(
                'κανένα TF map→base_link — δεν τρέχει AMCL/localization. '
                'Χωρίς εντοπισμό ο γύρος δεν έχει νόημα.')
            return

        rooms = load_locations()
        wanted = self._args.rooms
        if wanted:
            missing = [r for r in wanted if r not in rooms]
            if missing:
                self.get_logger().error(f'άγνωστα δωμάτια: {missing}')
                return
            rooms = {r: rooms[r] for r in wanted}
        else:
            rooms = {n: p for n, p in rooms.items() if n not in SKIP}

        if not rooms:
            self.get_logger().error('κανένα δωμάτιο στο locations.yaml')
            return

        for lap in range(1, self._args.loop + 1):
            if self._args.loop > 1:
                self.get_logger().info(f'── γύρος {lap}/{self._args.loop} ──')
            self._one_lap(rooms)
            if not rclpy.ok():
                return

    def _one_lap(self, rooms: dict):
        queue = [Leg(name, pose) for name, pose in self.order_rooms(rooms)]
        self.legs = list(queue)
        self.get_logger().info(
            'σειρά: ' + ' → '.join(leg.room for leg in queue))

        retry: list[Leg] = []
        for leg in queue:
            if not rclpy.ok():
                return
            if not self.drive_to(leg) and self._args.retries > 0:
                retry.append(leg)
            elif self._args.dwell > 0:
                time.sleep(self._args.dwell)

        # Second pass for whatever failed. The approach is different now
        # (the robot is somewhere else), which by itself resolves most of
        # the doorway-approach failures.
        for leg in retry:
            if not rclpy.ok():
                return
            self.get_logger().info(f'ξαναδοκιμάζω το {leg.room}')
            if self.drive_to(leg) and self._args.dwell > 0:
                time.sleep(self._args.dwell)

    # ── report ───────────────────────────────────────────────────────

    def report(self) -> int:
        if not self.legs:
            return 1
        ok = [l for l in self.legs if l.ok]
        width = max(len(l.room) for l in self.legs)
        lines = ['', f'Γύρος: {len(ok)}/{len(self.legs)} δωμάτια', '',
                 f'  {"δωμάτιο".ljust(width)}  κατάσταση          '
                 f'χρόνος   απόσταση  μέση    μέγιστη  προσπ.']
        for l in self.legs:
            lines.append(
                f'  {l.room.ljust(width)}  {l.status.ljust(18)} '
                f'{l.seconds:5.0f}s   {l.distance:5.1f}m  '
                f'{l.v_avg:4.2f}    {l.v_max:4.2f}     {l.attempts}')
        driven = [l for l in self.legs if l.v_max > 0.0]
        if driven:
            avg = sum(l.v_avg for l in driven) / len(driven)
            top = max(l.v_max for l in driven)
            lines += ['', f'  Ταχύτητα συνολικά: μέση {avg:.2f} m/s, '
                          f'μέγιστη {top:.2f} m/s']
        lines.append('')
        for line in lines:
            print(line)
        failed = [l.room for l in self.legs if not l.ok]
        if failed:
            print(f'  Δεν έφτασε: {", ".join(failed)}\n')
            return 1
        return 0


def parse_args(argv):
    p = argparse.ArgumentParser(
        description='Πήγαινε σε όλα τα δωμάτια, ένα-ένα.')
    p.add_argument('--rooms', type=lambda s: [r.strip() for r in s.split(',')],
                   help='λίστα δωματίων με κόμμα (default: όλα)')
    p.add_argument('--order', choices=['nearest', 'yaml'], default='nearest',
                   help='σειρά επίσκεψης (default: nearest)')
    p.add_argument('--timeout', type=float, default=180.0,
                   help='δευτερόλεπτα ανά δωμάτιο πριν ακυρωθεί (default: 180)')
    p.add_argument('--retries', type=int, default=1, choices=[0, 1],
                   help='2η προσπάθεια στο τέλος για όσα απέτυχαν (default: 1)')
    p.add_argument('--dwell', type=float, default=0.0,
                   help='παύση σε κάθε δωμάτιο, δευτερόλεπτα')
    p.add_argument('--loop', type=int, default=1, help='πόσους γύρους')
    p.add_argument('--speak', action='store_true',
                   help='ανακοίνωσε κάθε δωμάτιο με τη φωνή')
    # ros2 run passes --ros-args through; argparse must not choke on it.
    known, _ = p.parse_known_args(argv)
    return known


def main():
    args = parse_args(sys.argv[1:])
    rclpy.init()
    node = RoomTour(args)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    worker = threading.Thread(target=node.run, daemon=True)
    worker.start()

    try:
        while rclpy.ok() and not node.done.is_set():
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        code = node.report()
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(code)


if __name__ == '__main__':
    main()
