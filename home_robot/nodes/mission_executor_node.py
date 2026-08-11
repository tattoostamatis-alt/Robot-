#!/usr/bin/env python3
"""
mission_executor_node.py — Multi-step mission state machine for the home robot.

Missions are triggered by publishing to `mission/start` (String).
Format: "patrol" | "find:<label>" | "fetch:<label>" | "dock" | "check_rooms"
        | "check:<room>:<question>"
Cancel any running mission: publish "" (empty) or "cancel" to `mission/start`.

`check:<room>:<question>` ("πήγαινε στην κουζίνα, δες αν είναι αναμμένο το φως,
γύρνα και πες μου") drives to <room>, asks the VLM <question> about what the
camera sees there (via vision_node: vision/query → vision/answer, Gemini),
drives back to where the command was given, and speaks the answer. Needs the
camera + vision_node + tts_node up.

`fetch:<label>` ("φέρε μου το X") composes the perception + nav + arm stacks:
resolve the object's map position from object memory (falling back to a live
room search), drive to an approach pose short of it, verify it's actually
there, grasp-and-hold it (pick_command hold=true), carry it back to where the
command was given, and release it (place_command). Needs use_perception
(object memory + detector) + use_arm + Nav2 all up. HW-untested.

States per mission: NAVIGATE → INSPECT → REPORT → (next waypoint) → DONE | FAILED

Publishes:
  mission/status  (String) — idle / navigating / inspecting / done / failed / cancelled
  speech_response (String) — TTS feedback
  /navigate_to_pose action → Nav2

Also monitors speech_text for "ακύρωσε" / "σταμάτα αποστολή" to cancel.

Locations are loaded from config/locations.yaml.
Objects are taken from the most recent `detected_objects` / `tracked_objects` message.
"""

import json
import math
import os
import threading
import time
from enum import Enum, auto

import yaml
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty

from home_robot import dock_geometry
from home_robot.sort_planner import SortProgress, SortRules, plan_order
from home_robot.fetch_planner import (approach_pose, memory_target,
                                      nearest_detection, homing_twist,
                                      is_stale_repeat)
from home_robot.status_query import ROOM_NAMES_EL, rooms_from_locations
from home_robot.vocabulary import (greek_accusative, greek_for,
                                   needs_open_vocab, to_prompt)


class State(Enum):
    IDLE        = auto()
    NAVIGATING  = auto()
    INSPECTING  = auto()
    REPORTING   = auto()
    DONE        = auto()
    FAILED      = auto()
    CANCELLED   = auto()


def _load_locations() -> dict:
    try:
        path = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'locations.yaml'
        )
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_sort_rules():
    """config/sort_rules.yaml — where each object label belongs.

    Missing or unreadable is not fatal: every other mission still works, and
    the sort mission says so rather than the node refusing to start.
    """
    try:
        path = os.path.join(
            get_package_share_directory('home_robot'), 'config', 'sort_rules.yaml'
        )
        with open(path) as f:
            return SortRules.from_yaml(yaml.safe_load(f))
    except Exception:
        return None


# The room names come from status_query so speech is identical wherever it is
# produced; the dock is not a room but does get spoken about, so add it here.
LOCATION_NAMES_EL = {**ROOM_NAMES_EL, 'dock': 'βάση φόρτισης'}

CANCEL_PHRASES = {'ακύρωσε', 'ακυρωσε', 'σταμάτα', 'σταματα', 'σταμάτησε', 'σταματησε', 'cancel'}


class MissionExecutorNode(Node):
    def __init__(self):
        super().__init__('mission_executor_node')

        # fetch tuning
        self.declare_parameter('fetch_approach_dist', 0.4)   # m — park this short of the object
        self.declare_parameter('fetch_grasp_range', 1.0)     # m — object must be within this to grasp
        self.declare_parameter('memory_max_age', 300.0)      # s — older object-memory hits are re-verified live
        self.declare_parameter('pick_timeout', 40.0)         # s — wait for the arm to report
        self.declare_parameter('place_timeout', 30.0)
        self.declare_parameter('fetch_max_retries', 1)       # approach+pick retries
        self.declare_parameter('inspect_settle', 2.5)        # s — let the detector stabilise
        # Delivery: 'start_pose' returns to where the command was given; 'follow'
        # then homes in on the user's *current* position (they may have moved).
        self.declare_parameter('delivery_mode', 'start_pose')
        self.declare_parameter('deliver_distance', 0.8)      # m — how close to get to the user
        self.declare_parameter('homing_timeout', 25.0)       # s — give up homing, place anyway
        self.declare_parameter('search_speed', 0.4)          # rad/s — rotate to find the user
        # docking
        self.declare_parameter('dock_timeout', 150.0)        # s — longer than the driver's own 120 s,
                                                             # so its terminal status always wins
        self.declare_parameter('dock_settle', 1.5)           # s — let AMCL settle after the tag fix
        self.declare_parameter('aim_timeout', 8.0)           # s — give up squaring up, IR will sweep
        # Backstop for a NavigateToPose that never returns a result (server
        # died mid-goal). Deliberately far longer than any real room-to-room
        # drive: this is about not hanging forever, not about pacing Nav2.
        self.declare_parameter('nav_timeout', 300.0)         # s
        # sorting ("μάζεψε τα παιχνίδια")
        self.declare_parameter('sort_max_items', 20)     # hard stop on one run
        self.declare_parameter('sort_max_attempts', 2)   # per object, then give up
        self.declare_parameter('sort_max_reach', 6.0)    # m — leave the far side for later
        self._fetch_approach_dist = self.get_parameter('fetch_approach_dist').value
        self._fetch_grasp_range   = self.get_parameter('fetch_grasp_range').value
        self._memory_max_age      = self.get_parameter('memory_max_age').value
        self._pick_timeout        = self.get_parameter('pick_timeout').value
        self._place_timeout       = self.get_parameter('place_timeout').value
        self._fetch_max_retries   = self.get_parameter('fetch_max_retries').value
        self._inspect_settle      = self.get_parameter('inspect_settle').value
        self._delivery_mode       = self.get_parameter('delivery_mode').value
        self._deliver_distance    = self.get_parameter('deliver_distance').value
        self._homing_timeout      = self.get_parameter('homing_timeout').value
        self._search_speed        = self.get_parameter('search_speed').value
        self._dock_timeout        = self.get_parameter('dock_timeout').value
        self._dock_settle         = self.get_parameter('dock_settle').value
        self._aim_timeout         = self.get_parameter('aim_timeout').value
        self._nav_timeout         = self.get_parameter('nav_timeout').value
        self._sort_max_items      = self.get_parameter('sort_max_items').value
        self._sort_max_attempts   = self.get_parameter('sort_max_attempts').value
        self._sort_max_reach      = self.get_parameter('sort_max_reach').value

        self._locations  = _load_locations()
        self._state      = State.IDLE
        self._cancel_flag = threading.Event()
        self._lock       = threading.Lock()
        self._latest_objects: list = []
        self._latest_open_vocab: list = []
        self._object_memory: list = []      # confirmed set, for sorting
        self._sort_rules = _load_sort_rules()
        self._tracked_rx = 0.0    # last tracked_objects arrival (see _on_detected)

        self._nav_ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Request/response state for the fetch mission (callbacks run on the main
        # executor; the mission worker thread waits on these Events — same
        # no-spin pattern as _await_future).
        self._mem_event   = threading.Event(); self._mem_value = None
        self._pick_event  = threading.Event(); self._pick_value = None
        self._place_event = threading.Event(); self._place_value = None
        self._vision_event = threading.Event(); self._vision_value = None
        self._dock_event  = threading.Event(); self._dock_value  = None

        # One-shot AMCL fix off the tag above the base, before the final approach.
        self._reloc_cli = self.create_client(Empty,
                                             '/apriltag_relocalizer/relocalize')

        # Subscribers
        self.create_subscription(String, 'mission/start',     self._on_mission_start,   10)
        self.create_subscription(String, 'speech_text',       self._on_speech_text,     10)
        self.create_subscription(String, 'tracked_objects',   self._on_tracked,         10)
        self.create_subscription(String, 'detected_objects',  self._on_detected,        10)
        # Fetch's own live "is it here" check (COCO covers 80 classes; a slipper
        # is not one of them — see vocabulary.py's HOUSEHOLD_EL). Kept separate
        # from object_memory's copy of the same topic: this one only needs the
        # current frame, not a persisted map position.
        self.create_subscription(String, 'open_vocab_detections', self._on_open_vocab, 10)
        self.create_subscription(String, 'object_memory/answer', self._on_mem_answer,   10)
        # Whole confirmed set, for the sort mission (query answers one label).
        self.create_subscription(String, 'object_memory', self._on_object_memory, 10)
        self.create_subscription(String, 'pick_result',       self._on_pick_result,     10)
        self.create_subscription(String, 'place_result',      self._on_place_result,    10)
        self.create_subscription(String, 'vision/answer',     self._on_vision_answer,    10)
        # Latched by the driver, so a status published before we subscribed
        # (a fast 'lost') still reaches us.
        self.create_subscription(
            String, 'dock_status', self._on_dock_status,
            QoSProfile(depth=1,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))

        # Publishers
        self._status_pub  = self.create_publisher(String, 'mission/status', 10)
        self._speech_pub  = self.create_publisher(String, 'speech_response', 10)
        self._mem_query_pub = self.create_publisher(String, 'object_memory/query', 10)
        self._vision_query_pub = self.create_publisher(String, 'vision/query', 10)
        self._pick_pub    = self.create_publisher(String, 'pick_command', 10)
        # Idle by default (GPU cost) — a fetch for a non-COCO object turns this
        # on for the duration of the mission and clears it when done.
        self._vocab_pub   = self.create_publisher(String, 'vocab/set', 10)
        self._place_pub   = self.create_publisher(String, 'place_command', 10)
        # For the follow-delivery final approach (Nav2 gets us to the area; this
        # closes the last metre onto the user).
        # ‼️ RELATIVE 'cmd_vel'; bringup remaps it to cmd_vel_safe when
        # obstacle_safety_node is not running to relay it. The comment here
        # used to claim "obstacle_safety relays it to cmd_vel_safe, same as
        # person_follower" and both halves were wrong: person_follower
        # publishes to cmd_vel_safe directly, and localize.launch.py declares
        # use_obstacle_safety default FALSE, so nothing was relaying at all —
        # /cmd_vel had zero subscribers on the live robot 2026-08-03.
        self._cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Hands the final approach to the driver's closed-loop IR homing.
        self._dock_pub = self.create_publisher(Bool, 'dock', 10)

        self.get_logger().info('Mission executor ready — topics: mission/start, mission/status')

    # ── Subscriptions ────────────────────────────────────────────────

    def _on_objects(self, msg: String):
        try:
            self._latest_objects = json.loads(msg.data) or []
        except json.JSONDecodeError:
            pass

    # ‼️ tracked_objects is the PREFERRED source; detected_objects is a fallback
    # for when the tracker is not running. Both used to be wired to the same
    # callback with no preference at all despite the comment claiming one, so
    # with perception up every detection cycle was processed TWICE — once
    # tracked, once raw — alternating between the two views of the same scene.
    _TRACKED_TTL = 2.0        # s: consider the tracker live for this long

    def _on_tracked(self, msg):
        self._tracked_rx = time.monotonic()
        self._on_objects(msg)

    def _on_detected(self, msg):
        if time.monotonic() - self._tracked_rx < self._TRACKED_TTL:
            return            # the tracker is live; ignore the raw duplicate
        self._on_objects(msg)

    def _on_open_vocab(self, msg: String):
        try:
            self._latest_open_vocab = json.loads(msg.data) or []
        except json.JSONDecodeError:
            pass

    def _on_dock_status(self, msg: String):
        self._dock_value = msg.data
        self._dock_event.set()

    def _on_mem_answer(self, msg: String):
        try:
            self._mem_value = json.loads(msg.data)
        except json.JSONDecodeError:
            self._mem_value = None
        self._mem_event.set()

    def _on_pick_result(self, msg: String):
        try:
            self._pick_value = json.loads(msg.data)
        except json.JSONDecodeError:
            self._pick_value = None
        self._pick_event.set()

    def _on_place_result(self, msg: String):
        try:
            self._place_value = json.loads(msg.data)
        except json.JSONDecodeError:
            self._place_value = None
        self._place_event.set()

    def _on_vision_answer(self, msg: String):
        self._vision_value = msg.data.strip()
        self._vision_event.set()

    def _on_speech_text(self, msg: String):
        """Cancel on a spoken stop.

        A bare "σταμάτα" used to be ignored here — it had to contain "αποστολ"
        as well — on the grounds that llm_bridge's stop gate relays a cancel to
        us anyway. It does, but only while llm_bridge is running: with the LLM
        off (or busy failing) the spoken stop reached nobody and the mission
        drove on. So while a mission is actually running, any stop phrase
        cancels it; when idle, keep requiring "αποστολ" so unrelated chatter
        does not arm the flag for the next mission.
        """
        text = msg.data.lower().strip()
        if not any(p in text for p in CANCEL_PHRASES):
            return
        with self._lock:
            running = self._state != State.IDLE
        if running or 'αποστολ' in text:
            self._cancel_flag.set()

    def _on_mission_start(self, msg: String):
        cmd = msg.data.strip()
        if not cmd or cmd.lower() in {'cancel', 'ακύρωσε'}:
            self._cancel_flag.set()
            return

        # ‼️ Claim the slot INSIDE the lock. The check used to be under the lock
        # but the state only became non-IDLE later, inside the worker thread, so
        # two mission/start messages arriving back to back both passed the gate
        # and two threads drove the same robot with competing Nav2 goals. The
        # claim and the test have to be one atomic step.
        with self._lock:
            if self._state != State.IDLE:
                self.get_logger().warn(f'Mission already running ({self._state}), ignoring "{cmd}"')
                return
            self._state = State.NAVIGATING

        self._cancel_flag.clear()
        threading.Thread(target=self._run_mission, args=(cmd,), daemon=True).start()

    def _run_mission(self, cmd: str):
        """Run `cmd`, and release the mission slot no matter how it ends.

        Several missions bail out early without calling _finish (no rooms
        recorded, unknown dock, unrecognised command). Now that the slot is
        claimed up front, those paths would strand the state at NAVIGATING and
        every later mission would be refused with "already running" — so the
        release is guaranteed here rather than trusted to each mission body.
        """
        try:
            self._dispatch(cmd)
        except Exception:                                   # noqa: BLE001
            self.get_logger().exception(f'Mission "{cmd}" crashed')
        finally:
            with self._lock:
                if self._state != State.IDLE:
                    self._state = State.IDLE
                    self._status_pub.publish(String(data='idle'))

    def _dispatch(self, cmd: str):
        if cmd == 'patrol':
            self._mission_patrol()
        elif cmd.startswith('find:'):
            label = cmd[5:].strip()
            self._mission_find(label)
        elif cmd.startswith('fetch:'):
            label = cmd[6:].strip()
            self._mission_fetch(label)
        elif cmd.startswith('goto_object:'):
            label = cmd[12:].strip()
            self._mission_goto_object(label)
        elif cmd == 'dock':
            self._mission_dock()
        elif cmd.startswith('sort'):
            # "sort" = everything with a rule; "sort:<destination>" = only the
            # things that belong there ("μάζεψε τα παιχνίδια").
            group = cmd[5:].strip() or None if cmd.startswith('sort:') else None
            self._mission_sort(group)
        elif cmd == 'check_rooms':
            self._mission_check_rooms()
        elif cmd.startswith('check:'):
            room, _, question = cmd[6:].strip().partition(':')
            self._mission_check(room.strip(), question.strip() or 'τι βλέπεις;')
        else:
            self.get_logger().warn(f'Unknown mission: {cmd}')
            self._speak(f'Δεν ξέρω αποστολή "{cmd}".')

    # ── Mission: patrol ──────────────────────────────────────────────

    def _mission_patrol(self):
        self.get_logger().info('Mission: patrol')
        locs = rooms_from_locations(self._locations)
        if not locs:
            self._speak('Δεν έχω καταχωρισμένα δωμάτια ακόμα.')
            return

        self._set_state(State.NAVIGATING)
        self._speak('Ξεκινώ περιπολία.')

        findings = {}
        for room in locs:
            if self._cancel_flag.is_set():
                break
            name_el = LOCATION_NAMES_EL.get(room, room)
            self._speak(f'Πηγαίνω {name_el}.')
            if not self._navigate_to(room):
                continue

            self._set_state(State.INSPECTING)
            time.sleep(2.0)  # let detector stabilize

            nearby = [o['label'] for o in self._latest_objects if o.get('z', 99.) < 3.5]
            findings[name_el] = nearby

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Η περιπολία ακυρώθηκε.')
            return

        # Report
        self._set_state(State.REPORTING)
        if any(findings.values()):
            parts = [f'{r}: {", ".join(items)}' for r, items in findings.items() if items]
            self._speak('Αναφορά περιπολίας. ' + '. '.join(parts))
        else:
            self._speak('Ολοκλήρωσα περιπολία. Δεν βρήκα τίποτα ξεχωριστό.')

        self._finish(State.DONE, '')

    # ── Mission: find ────────────────────────────────────────────────

    def _mission_find(self, label: str):
        self.get_logger().info(f'Mission: find "{label}"')
        locs = rooms_from_locations(self._locations)
        if not locs:
            self._speak('Δεν έχω καταχωρισμένα δωμάτια.')
            return

        self._set_state(State.NAVIGATING)
        self._speak(f'Ψάχνω για {label}.')

        for room in locs:
            if self._cancel_flag.is_set():
                break
            name_el = LOCATION_NAMES_EL.get(room, room)
            self._speak(f'Ελέγχω {name_el}.')
            if not self._navigate_to(room):
                continue

            self._set_state(State.INSPECTING)
            time.sleep(2.5)

            found = [o for o in self._latest_objects
                     if o.get('label', '').lower() == label.lower() and o.get('z', 99.) < 4.]
            if found:
                dist = min(o.get('z', 0.) for o in found)
                self._finish(State.DONE, f'Βρήκα {label} στο {name_el}, περίπου {dist:.1f} μέτρα μπροστά!')
                return

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Αναζήτηση ακυρώθηκε.')
        else:
            self._finish(State.FAILED, f'Δεν βρήκα {label} πουθενά.')

    # ── Mission: goto_object ("πήγαινε στην καρέκλα") ────────────────

    def _mission_goto_object(self, label: str):
        """Drive to a thing rather than to a taught place.

        `goto` only accepts the rooms and waypoints someone taught by hand, and
        `find` walks every room in turn looking with the camera — so until now
        the robot could not use the one thing that already knew the answer.
        object_memory_node has been recording where each detected object sits in
        the MAP frame all along; asking it first turns "πήγαινε στην καρέκλα"
        into a single navigation instead of a tour of the flat.

        The room-by-room search stays as the fallback, because remembered
        positions go stale: a cup is somewhere else by tomorrow. memory_target
        applies that age limit, and anything it rejects falls through to looking
        for real.
        """
        self.get_logger().info(f'Mission: goto_object "{label}"')
        self._set_state(State.NAVIGATING)

        # _resolve_location asks object memory first and only then walks the
        # rooms — the same order fetch uses, so both take the fast path when the
        # robot already knows and neither trusts a stale position.
        target = self._resolve_location(label)
        if target is None:
            self._finish(State.CANCELLED if self._cancel_flag.is_set() else State.FAILED,
                         'Ακυρώθηκε.' if self._cancel_flag.is_set()
                         else f'Δεν ξέρω πού είναι {label}.')
            return

        # Falling back to a metre behind the object keeps approach_pose defined
        # when TF is briefly unavailable; the same fallback fetch uses.
        rob = self._lookup_base_pose() or (target['x'] - 1.0, target['y'], 0.0)

        # Park short of it and face it, exactly as fetch does — driving at the
        # remembered point itself would end with the object under the bumper.
        ax, ay, yaw = approach_pose((rob[0], rob[1]), (target['x'], target['y']),
                                    dist=self._fetch_approach_dist)
        room = target.get('room')
        where = f' στο {LOCATION_NAMES_EL.get(room, room)}' if room else ''
        self._speak(f'Πάω στο {label}{where}.')

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Ακυρώθηκε.')
            return
        if self._navigate_to_xy(ax, ay, yaw):
            self._finish(State.DONE, f'Έφτασα στο {label}.')
        else:
            self._finish(State.FAILED, f'Δεν μπόρεσα να φτάσω στο {label}.')

    # ── Mission: fetch ("φέρε μου το X") ─────────────────────────────

    def _mission_fetch(self, label: str):
        """Entry point: normalize the label and arm open-vocab if it's needed.

        object_detector only ever emits the 80 COCO classes, so a slipper —
        which has no COCO class at all (see vocabulary.py's HOUSEHOLD_EL) —
        was never findable by fetch no matter how thoroughly it searched: not
        by the live check below, and not by object_memory (same detector).
        `to_prompt` turns whatever the LLM said («παντόφλα», «slipper», or a
        wrong COCO guess like «shoe») into the one label every downstream
        check, object_memory, and pick_place_node's grasp all agree on.
        open_vocab_detector is idle by default (a second model on the same
        GPU) — armed for the duration of this mission only, cleared after.
        """
        prompt = to_prompt(label) or label
        use_open_vocab = needs_open_vocab(prompt)
        if use_open_vocab:
            self._vocab_pub.publish(String(data=json.dumps([prompt])))
        try:
            self._mission_fetch_inner(prompt)
        finally:
            if use_open_vocab:
                self._vocab_pub.publish(String(data=json.dumps([])))

    def _mission_fetch_inner(self, label: str):
        self.get_logger().info(f'Mission: fetch "{label}"')
        # `label` is the internal matching key (English prompt) from here on —
        # everything spoken uses these two declined forms instead.
        label_el  = greek_for(label)
        label_acc = greek_accusative(label)
        # Deliver back to where the command was given = the robot's pose now.
        start_pose = self._lookup_base_pose()

        self._set_state(State.NAVIGATING)
        self._speak(f'Πάω να φέρω {label_acc}.')

        # 1. RESOLVE — where is it?
        target = self._resolve_location(label)
        if target is None:
            self._finish(State.CANCELLED if self._cancel_flag.is_set() else State.FAILED,
                         'Ακυρώθηκε.' if self._cancel_flag.is_set() else f'Δεν ξέρω πού είναι {label_el}.')
            return

        # 2-4. approach → verify → grasp-and-hold, with retries
        #
        # ‼️ A wrong memory used to be unrecoverable. The loop re-queried
        # object memory after a failed approach, and object memory answers
        # from its own store — so it returned the SAME coordinates, the robot
        # drove to the SAME empty spot, and the retry budget went entirely on
        # one wrong place. It then gave up without ever running the
        # room-by-room search that finds the object, which is the search it
        # runs happily when memory has simply never seen the thing. Verified
        # in scripts/fetch_sim.py --scenario stale: 5 of 6 objects came back,
        # and the one that failed was the one whose remembered position was
        # nowhere near the truth.
        #
        # Now a cleared position is remembered in `tried`, and a memory answer
        # that points back at one of them is not accepted as progress — we go
        # and look instead. A position found by actually looking gets a fresh
        # retry budget, because it is evidence rather than a guess.
        picked = False
        tried = []            # approach targets that turned out to be empty
        searched = False      # room-by-room fallback already used once
        reached = False       # did we ever actually get to the object?
        attempt = 0
        while attempt <= self._fetch_max_retries and not self._cancel_flag.is_set():
            attempt += 1
            rob = self._lookup_base_pose() or (target['x'] - 1.0, target['y'], 0.0)
            ax, ay, yaw = approach_pose((rob[0], rob[1]), (target['x'], target['y']),
                                        self._fetch_approach_dist)
            if not self._navigate_to_xy(ax, ay, yaw):
                reached = False
                continue
            reached = True

            self._set_state(State.INSPECTING)
            time.sleep(self._inspect_settle)
            det = nearest_detection(self._latest_objects + self._latest_open_vocab,
                                    label, self._fetch_grasp_range)
            if det is not None:
                if self._do_pick(label, hold=True):
                    picked = True
                    break
                continue

            # Not here. Note the dud, then ask memory for something better.
            tried.append((target['x'], target['y']))
            self._speak(f'Δεν βλέπω {label_acc} εδώ, ξανακοιτάω.')
            t2 = self._query_object_memory(label)
            if t2 and not is_stale_repeat(t2, tried):
                target = t2
                continue

            # Memory only has the place we just cleared. Go and look.
            if searched:
                break
            searched = True
            self._speak(f'Ψάχνω στα δωμάτια για {label_acc}.')
            t3 = self._search_rooms_for(label, tried)
            if t3 is None:
                break
            target = t3
            attempt = 0

        if not picked:
            # ‼️ Say WHICH half failed. This used to be "Δεν κατάφερα να πιάσω
            # X" for every outcome, including the common one where Nav2 never
            # got the robot there at all — it wedges in narrow doorways
            # (reproduced in scripts/fetch_sim_gazebo.py: 4 of 6 objects, with
            # "STUCK detected ... displacement=0.022m" in the log). Blaming the
            # grasp sends whoever hears it to look at the arm, which is fine
            # and was never involved.
            if self._cancel_flag.is_set():
                msg = 'Ακυρώθηκε.'
            elif not reached:
                msg = f'Δεν μπόρεσα να φτάσω στο {label_el}.'
            else:
                msg = f'Δεν κατάφερα να πιάσω {label_acc}.'
            self._finish(State.CANCELLED if self._cancel_flag.is_set() else State.FAILED,
                         msg)
            return

        # 5. carry back to the user (skip the drive if cancelled, but still
        #    release below so the arm isn't left holding the object)
        if not self._cancel_flag.is_set():
            self._set_state(State.NAVIGATING)
            self._speak(f'Σου φέρνω {label_acc}.')
            if start_pose is not None:
                self._navigate_to_xy(*start_pose)      # Nav2 to the area
            if self._delivery_mode == 'follow':
                self._home_in_on_person()              # close onto the user now

        # 6. DELIVER — release the object
        self._do_place()

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, f'Ακυρώθηκε — άφησα {label_acc}.')
        else:
            self._finish(State.DONE, f'Ορίστε {label_acc}.')

    def _resolve_location(self, label: str):
        """Return {'x','y','room',...} for `label`, or None. Object memory first,
        then a live room-by-room search (object memory updates as we look)."""
        t = self._query_object_memory(label)
        if t:
            return t
        self._speak('Δεν το έχω στη μνήμη, ψάχνω στα δωμάτια.')
        return self._search_rooms_for(label)

    def _search_rooms_for(self, label: str, tried=None):
        """Drive room to room until object memory has a position we trust.

        `tried` is a list of (x, y) already visited and found empty; a memory
        answer pointing back at one of them is not news, so the search keeps
        going rather than declaring success on the same wrong spot.
        """
        for room in rooms_from_locations(self._locations):
            if self._cancel_flag.is_set():
                return None
            if not self._navigate_to(room):
                continue
            self._set_state(State.INSPECTING)
            time.sleep(self._inspect_settle)
            t = self._query_object_memory(label)
            if t and not is_stale_repeat(t, tried):
                return t
        return None

    def _query_object_memory(self, label: str, timeout: float = 1.5):
        self._mem_event.clear()
        self._mem_value = None
        self._mem_query_pub.publish(String(data=label))
        if not self._mem_event.wait(timeout):
            return None
        return memory_target(self._mem_value, now=self._now(), max_age=self._memory_max_age)

    def _do_pick(self, label: str, hold: bool) -> bool:
        self._pick_event.clear()
        self._pick_value = None
        self._pick_pub.publish(String(data=json.dumps({'label': label, 'hold': hold})))
        if not self._pick_event.wait(self._pick_timeout):
            self._speak('Ο βραχίονας άργησε να απαντήσει.')
            return False
        return (self._pick_value or {}).get('status') == 'ok'

    def _do_place(self) -> bool:
        self._place_event.clear()
        self._place_value = None
        self._place_pub.publish(String(data='{}'))
        if not self._place_event.wait(self._place_timeout):
            return False
        return (self._place_value or {}).get('status') == 'ok'

    def _home_in_on_person(self):
        """Follow-delivery final approach: rotate to find the user, then drive
        onto them until within deliver_distance. Bounded by homing_timeout;
        always stops the base on exit. Falls through (places where it is) if the
        user is never found — better than driving forever."""
        self._set_state(State.INSPECTING)
        self._speak('Σε ψάχνω για να σου το δώσω.')
        deadline = time.monotonic() + self._homing_timeout
        arrived = False
        while time.monotonic() < deadline and not self._cancel_flag.is_set():
            person = nearest_detection(self._latest_objects, 'person', max_z=5.0)
            tw = Twist()
            if person is None:
                tw.angular.z = self._search_speed          # scan for the user
            else:
                lin, ang, arrived = homing_twist(
                    person, deliver_distance=self._deliver_distance)
                tw.linear.x, tw.angular.z = lin, ang
                if arrived:
                    self._cmd_vel_pub.publish(Twist())
                    break
            self._cmd_vel_pub.publish(tw)
            time.sleep(0.2)
        self._cmd_vel_pub.publish(Twist())                 # always stop
        if not arrived and not self._cancel_flag.is_set():
            self._speak('Δεν σε βρήκα, το αφήνω εδώ.')

    def _lookup_base_pose(self):
        """Current map-frame (x, y, yaw) of the robot, or None if TF isn't ready."""
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2. * (q.w * q.z + q.x * q.y),
                         1. - 2. * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ── Mission: dock ────────────────────────────────────────────────

    def _mission_dock(self):
        """Drive to the base and park on it, in three handovers.

        This used to be a single `_navigate_to('dock')` that announced "Έφτασα
        στη βάση φόρτισης" the moment Nav2 returned — it never published
        `dock`, so the IR homing that actually parks the robot was never
        started, and the mission reported success from wherever Nav2 happened
        to stop. Two things have to be true for a dock to work, and neither was:

        1. **Nav2 cannot do the final approach.** The corridor into the base
           measures 0.05-0.22 m of clearance against a 0.30 m inflation radius,
           so no plan into it will ever be accepted. Nav2's job ends at
           `dock_staging`, roughly 0.85 m out, which it can reach.
        2. **Something has to steer the last stretch.** That is `ir_homing`,
           driving on the dock's own beams with the costmap out of the loop —
           legitimate here because the robot physically fits (0.17 m inscribed)
           even where Nav2 refuses to plan.

        Between the two, a relocalization off the tag above the base: AMCL can
        have drifted on the way over, and the whole approach is aimed from the
        map pose. The tag is the one thing in the room that fixes it to
        millimetres.
        """
        staging = 'dock_staging' if 'dock_staging' in self._locations else 'dock'
        if staging not in self._locations:
            self._speak('Δεν έχω καταχωρίσει τη βάση φόρτισης.')
            return

        self._set_state(State.NAVIGATING)
        self._speak('Πηγαίνω στη βάση φόρτισης.')

        if not self._navigate_to(staging):
            if self._cancel_flag.is_set():
                self._finish(State.CANCELLED, 'Ακύρωση πλεύσης προς βάση.')
            else:
                self._finish(State.FAILED,
                             'Δεν μπόρεσα να πλησιάσω τη βάση φόρτισης.')
            return
        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Ακύρωση πλεύσης προς βάση.')
            return

        # Re-fix on the tag before aiming. Best-effort: a dock attempt from a
        # slightly stale pose is worth trying, a refusal to dock is not.
        self._relocalize_on_tag()
        self._aim_at_dock()
        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Ακύρωση πλεύσης προς βάση.')
            return

        self._set_state(State.INSPECTING)
        outcome = self._run_ir_homing()

        if outcome == 'seated':
            self._finish(State.DONE, 'Κάθισα στη βάση φόρτισης.')
        elif outcome == 'cancelled':
            self._finish(State.CANCELLED, 'Ακύρωση πλεύσης προς βάση.')
        elif outcome == 'lost':
            self._finish(State.FAILED,
                         'Δεν βρίσκω τη δέσμη της βάσης. Είναι στην πρίζα;')
        else:
            self._finish(State.FAILED,
                         'Έφτασα μπροστά στη βάση αλλά δεν κατάφερα να καθίσω.')

    def _relocalize_on_tag(self):
        """Ask the relocalizer for a one-shot fix off the tag above the base.

        Silent no-op when the service is absent — the relocalizer is optional
        in some launches, and its absence should cost accuracy, not the dock.
        """
        if not self._reloc_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(
                'apriltag relocalizer not available — docking on the AMCL pose')
            return
        future = self._reloc_cli.call_async(Empty.Request())
        if self._await_future(future, 5.0) is None:
            self.get_logger().warn('relocalize call timed out — carrying on')
            return
        # AMCL needs a moment to settle on the seeded pose before we read a
        # heading off it.
        time.sleep(self._dock_settle)
        self.get_logger().info('Relocalized on the dock tag')

    def _aim_at_dock(self):
        """Turn to face the base, so IR homing starts inside the beam cone.

        Skipped entirely when the pose is unknown: IR homing sweeps for the
        beam on its own, so a missing TF costs time, not the attempt.
        """
        target = self._locations.get('dock')
        if target is None:
            return
        deadline = time.monotonic() + self._aim_timeout
        while time.monotonic() < deadline and not self._cancel_flag.is_set():
            pose = self._lookup_base_pose()
            if pose is None:
                return
            x, y, yaw = pose
            desired = math.atan2(float(target['y']) - y, float(target['x']) - x)
            wz = dock_geometry.align_twist(yaw, desired)
            if wz == 0.0:
                break
            twist = Twist()
            twist.angular.z = wz
            self._cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        self._cmd_vel_pub.publish(Twist())

    def _run_ir_homing(self):
        """Hand over to IR homing and wait for how it actually ended.

        Returns 'seated' | 'lost' | 'timeout' | 'cancelled'. The driver's own
        `dock_timeout` bounds the approach; the wait here is longer so a
        terminal status always beats the local deadline, and only a driver
        that died silently falls through to 'timeout'.
        """
        self._dock_event.clear()
        self._dock_value = None
        self._speak('Μπαίνω στη βάση.')
        self._dock_pub.publish(Bool(data=True))

        deadline = time.monotonic() + self._dock_timeout
        while time.monotonic() < deadline:
            if self._cancel_flag.is_set():
                self._cmd_vel_pub.publish(Twist())
                return 'cancelled'
            if self._dock_event.wait(0.2):
                self._dock_event.clear()
                status = self._dock_value
                if status in ('seated', 'lost', 'timeout'):
                    return status
                # 'homing' / 'searching' are progress, keep waiting
        return 'timeout'

    # ── Mission: check_rooms ─────────────────────────────────────────

    def _mission_check_rooms(self):
        self.get_logger().info('Mission: check_rooms')
        locs = rooms_from_locations(self._locations)
        if not locs:
            self._speak('Δεν έχω δωμάτια.')
            return

        self._set_state(State.NAVIGATING)
        self._speak('Ελέγχω όλα τα δωμάτια.')

        for room in locs:
            if self._cancel_flag.is_set():
                break
            name_el = LOCATION_NAMES_EL.get(room, room)
            self._speak(f'Ελέγχω {name_el}.')
            if not self._navigate_to(room):
                continue
            self._set_state(State.INSPECTING)
            time.sleep(2.0)
            nearby = [o['label'] for o in self._latest_objects if o.get('z', 99.) < 3.5]
            if nearby:
                self._speak(f'{name_el}: βλέπω {", ".join(nearby[:3])}.')
            else:
                self._speak(f'{name_el}: τίποτα.')

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Έλεγχος ακυρώθηκε.')
        else:
            self._finish(State.DONE, 'Ολοκλήρωσα έλεγχο δωματίων.')

    # ── Mission: check ("πήγαινε δες X και πες μου") ─────────────────

    def _mission_sort(self, group=None):
        """Tidy up: pick each known object and carry it where it belongs.

        Fetch in a loop, with two differences that matter over a run lasting
        minutes: the plan is REBUILT after every pick (the robot has moved and
        object_memory may have been updated), and a failure moves on to the
        next object instead of ending the mission. An object that keeps failing
        is abandoned after MAX_ATTEMPTS — otherwise the robot loops on the one
        thing it cannot grasp while the rest of the floor stays untidied.
        """
        if self._sort_rules is None or not self._sort_rules.known_labels:
            self._finish(State.FAILED,
                         'Δεν ξέρω πού πάει τι. Λείπει το sort_rules.yaml.')
            return
        if group is not None and not self._sort_rules.labels_for(group):
            self._finish(State.FAILED, f'Δεν ξέρω τι πάει στο {group}.')
            return

        self.get_logger().info(f'Mission: sort group={group!r}')
        self._speak('Ξεκινάω να μαζεύω.' if group is None
                    else f'Μαζεύω ό,τι πάει {LOCATION_NAMES_EL.get(group, group)}.')
        progress = SortProgress(max_attempts=self._sort_max_attempts)

        for _ in range(self._sort_max_items):
            if self._cancel_flag.is_set():
                break

            # Re-plan every round: the robot has moved, so "nearest" changed,
            # and object_memory may have seen or lost things meanwhile.
            objects = self._sort_candidates()
            here = self._lookup_base_pose()
            robot_xy = (here[0], here[1]) if here else (0.0, 0.0)
            ordered = plan_order(objects, robot_xy, self._sort_rules,
                                 group=group, max_reach=self._sort_max_reach)
            target = progress.next_target(ordered)
            if target is None:
                break

            ok = self._sort_one(target)
            progress.record(target, ok)

        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Σταμάτησα το μάζεμα.')
            return
        self._finish(State.DONE, progress.summary_el())

    def _sort_one(self, target) -> bool:
        """Approach, grasp, carry to the destination, release. False on any
        failure — the caller decides whether to retry or move on."""
        label = target['label']
        dest = target['destination']
        self._set_state(State.NAVIGATING)

        here = self._lookup_base_pose()
        robot_xy = (here[0], here[1]) if here else (0.0, 0.0)
        ax, ay, ayaw = approach_pose(robot_xy, (target['x'], target['y']),
                                     dist=self._fetch_approach_dist)
        if not self._navigate_to_xy(ax, ay, ayaw):
            return False
        if self._cancel_flag.is_set():
            return False

        # hold=True: keep it in the gripper for the drive, same as fetch.
        self._set_state(State.INSPECTING)
        if not self._do_pick(label, hold=True):
            return False

        self._set_state(State.NAVIGATING)
        if not self._navigate_to(dest):
            # Already holding the object. Put it down where we are rather than
            # driving around with it indefinitely — a known place beats a
            # gripper that never opens.
            self._do_place()
            return False
        return self._do_place()

    def _on_object_memory(self, msg: String):
        """Whole confirmed set, republished by object_memory_node at ~1 Hz.

        Subscribed rather than queried: sorting needs EVERYTHING it knows
        about, and object_memory/query answers one label at a time.
        """
        try:
            items = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(items, list):
            self._object_memory = items

    def _sort_candidates(self):
        """Everything object_memory currently knows about, as plain dicts."""
        return list(self._object_memory)

    def _mission_check(self, room: str, question: str):
        """Go to <room>, ask the VLM <question> about what the camera sees
        there, drive back to where the command was given, and speak the answer.
        Reuses vision_node (vision/query → vision/answer). Needs the camera +
        vision_node + tts_node running."""
        self.get_logger().info(f'Mission: check room="{room}" q="{question}"')
        if room not in self._locations:
            self._finish(State.FAILED,
                         f'Δεν ξέρω πού είναι {LOCATION_NAMES_EL.get(room, room)}.')
            return

        name_el = LOCATION_NAMES_EL.get(room, room)
        # Where the command was given (= robot's pose now) so we can report back.
        start_pose = self._lookup_base_pose()

        # 1. go to the room
        self._set_state(State.NAVIGATING)
        self._speak(f'Πάω {name_el} να δω.')
        if not self._navigate_to(room):
            self._finish(State.CANCELLED if self._cancel_flag.is_set() else State.FAILED,
                         'Ακυρώθηκε.' if self._cancel_flag.is_set()
                         else f'Δεν μπόρεσα να φτάσω {name_el}.')
            return

        # 2. inspect — ask the VLM about the live frame
        self._set_state(State.INSPECTING)
        time.sleep(self._inspect_settle)
        answer = self._ask_vision(question) if not self._cancel_flag.is_set() else None
        if answer is None and not self._cancel_flag.is_set():
            answer = 'δεν μπόρεσα να δω καθαρά'

        # 3. come back to the user
        if not self._cancel_flag.is_set() and start_pose is not None:
            self._set_state(State.NAVIGATING)
            self._speak('Γυρίζω πίσω.')
            self._navigate_to_xy(*start_pose)

        # 4. report by speaking
        if self._cancel_flag.is_set():
            self._finish(State.CANCELLED, 'Η αποστολή ακυρώθηκε.')
        else:
            self._set_state(State.REPORTING)
            self._finish(State.DONE, f'Πήγα {name_el}. {answer}')

    def _ask_vision(self, question: str, timeout: float = 20.0):
        """Ask vision_node a free-form question about the current camera frame.
        Publishes on vision/query, waits (no-spin, main executor services the
        callback) for vision/answer. Returns the answer, or None on timeout."""
        self._vision_event.clear()
        self._vision_value = None
        self._vision_query_pub.publish(String(data=question))
        if not self._vision_event.wait(timeout):
            self.get_logger().warn('vision/answer timed out')
            return None
        return self._vision_value

    # ── Navigation ───────────────────────────────────────────────────

    def _await_future(self, future, timeout_sec):
        """Wait for an async future from this worker thread WITHOUT spinning —
        the node's main executor (rclpy.spin in main) services the callbacks
        that complete it. Spinning here would attach the node to a second
        executor and the future would never complete. Returns None on timeout."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return future.result() if future.done() else None

    def _navigate_to(self, location_name: str) -> bool:
        pose_data = self._locations.get(location_name)
        if not pose_data:
            self.get_logger().warn(f'Location "{location_name}" not in locations.yaml')
            return False
        return self._navigate_to_xy(float(pose_data.get('x', 0.)),
                                    float(pose_data.get('y', 0.)),
                                    float(pose_data.get('yaw', 0.)))

    def _navigate_to_xy(self, x: float, y: float, yaw: float) -> bool:
        if not self._nav_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('navigate_to_pose action server not available')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.)

        future = self._nav_ac.send_goal_async(goal)
        gh = self._await_future(future, 10.)
        if gh is None or not gh.accepted:
            return False

        # Poll until done or cancelled. Do NOT spin here — the node's main
        # executor (rclpy.spin in main) services the action callbacks that
        # complete result_future; spinning from this worker thread would
        # attach the node to a second executor and stall the future.
        result_future = gh.get_result_async()
        # ‼️ Bounded. This wait used to have no deadline at all: if the Nav2
        # server died after accepting the goal, the future never completed, the
        # mission thread parked here forever and the state never returned to
        # IDLE — so every later mission was refused until a restart. The cap is
        # a backstop against a dead server, not a navigation budget, hence the
        # generous default.
        deadline = time.monotonic() + self._nav_timeout
        while not result_future.done():
            if self._cancel_flag.is_set():
                gh.cancel_goal_async()
                return False
            if time.monotonic() > deadline:
                self.get_logger().warn(
                    f'navigate_to_pose produced no result in {self._nav_timeout:.0f}s '
                    '— giving up on this goal (server dead?)')
                gh.cancel_goal_async()
                return False
            time.sleep(0.1)

        result = result_future.result()
        return result is not None and result.status == GoalStatus.STATUS_SUCCEEDED

    # ── Helpers ──────────────────────────────────────────────────────

    def _set_state(self, state: State):
        self._state = state
        self._status_pub.publish(String(data=state.name.lower()))

    def _finish(self, final_state: State, message: str):
        self._set_state(final_state)
        if message:
            self._speak(message)
        self.get_logger().info(f'Mission finished: {final_state.name}')
        time.sleep(2.0)
        self._state = State.IDLE
        self._status_pub.publish(String(data='idle'))

    def _speak(self, text: str):
        self._speech_pub.publish(String(data=text))
        self.get_logger().info(f'[speech] {text}')


def main():
    rclpy.init()
    node = MissionExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
