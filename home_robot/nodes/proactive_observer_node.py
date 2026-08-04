#!/usr/bin/env python3
"""Proactive observations — the robot speaks first.

Everything else in the voice stack is reactive: wake word, question, answer.
This node watches `object_memory` against a learned baseline of where things
normally live, and volunteers the differences that matter:

    «Το ποτήρι είναι στο σαλόνι, συνήθως είναι στην κουζίνα.»

Topics
  in   /object_memory     (String, JSON)  — confirmed instances, map frame
       /pose_detections   (String, JSON)  — is anybody here to talk to
  out  /speech_response   (String)        — spoken, via tts_node
       /observations      (String, JSON)  — the feed the dashboard renders

The baseline is learned, not configured, and persists across runs in
`~/.ros/home_robot_baseline.json`. On a fresh map the node says nothing at all
until it has watched a room over several separate visits — which is correct:
it has no idea yet what "normal" looks like.

‼️ The gating is the feature, not the detection. See home_robot/proactive.py.
A robot that reports every change gets muted within a day, so the defaults here
are deliberately conservative: at most 4 remarks an hour, 3 minutes apart, never
the same remark twice in 2 hours, nothing between 23:00 and 08:00, and nothing
at all unless a person is actually in view.
"""

import json
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from home_robot.proactive import (Baseline, Observation, ObservationGate,
                                  find_observations)
from home_robot.routines import RoutineLearner
from home_robot.status_query import ROOM_LOCATIVE_EL

_STATE_PATH = os.path.expanduser('~/.ros/home_robot_baseline.json')
_ROUTINE_PATH = os.path.expanduser('~/.ros/home_robot_routines.json')

# How long after the last skeleton we still consider somebody present. People
# step out of frame constantly; a hard "visible right now" test would silence
# the node almost permanently.
_PERSON_MEMORY_S = 20.0


def _locative(room):
    """'kouzina' -> 'στην κουζίνα', with a safe fallback for unmapped rooms."""
    return ROOM_LOCATIVE_EL.get(room, f'στο {room}' if room else 'κάπου αλλού')


def render_greek(obs):
    """The sentence for an observation. Kept here rather than in the pure module
    so the geometry stays language-free."""
    if obs.kind == 'moved_room':
        return (f'Το {obs.label} είναι {_locative(obs.room)}, '
                f'συνήθως είναι {_locative(obs.from_room)}.')
    if obs.kind == 'moved':
        return f'Το {obs.label} έχει μετακινηθεί {_locative(obs.room)}.'
    return f'Κάτι άλλαξε με το {obs.label}.'


def render_routine_el(key, expected_hour, hours_late):
    """The sentence for a routine that has not happened. Absence is the news."""
    when = f'{int(expected_hour):02d}:{int(round((expected_hour % 1) * 60)):02d}'
    kind, _, name = key.partition(':')
    if kind == 'person':
        return (f'Ο {name} συνήθως είναι εδώ γύρω στις {when}. '
                f'Δεν τον έχω δει σήμερα.')
    if kind == 'sound':
        return f'Συνήθως ακούω {name} γύρω στις {when}. Σήμερα όχι.'
    if kind == 'room':
        return (f'Συνήθως περνάω {ROOM_LOCATIVE_EL.get(name, name)} γύρω στις '
                f'{when}. Σήμερα δεν έχω πάει.')
    return f'Το «{key}» συνήθως γίνεται γύρω στις {when}. Σήμερα δεν έγινε.'


class ProactiveObserverNode(Node):
    def __init__(self):
        super().__init__('proactive_observer_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('min_interval', 180.0)      # s between remarks
        self.declare_parameter('max_per_hour', 4)
        self.declare_parameter('repeat_cooldown', 7200.0)  # s
        self.declare_parameter('quiet_start_hour', 23)
        self.declare_parameter('quiet_end_hour', 8)
        self.declare_parameter('require_person', True)
        self.declare_parameter('moved_threshold', 1.5)     # m
        self.declare_parameter('state_path', _STATE_PATH)

        self.enabled        = self.get_parameter('enabled').value
        self.require_person = self.get_parameter('require_person').value
        self.moved_threshold = self.get_parameter('moved_threshold').value
        self.state_path     = self.get_parameter('state_path').value

        self.gate = ObservationGate(
            min_interval=self.get_parameter('min_interval').value,
            max_per_hour=self.get_parameter('max_per_hour').value,
            repeat_cooldown=self.get_parameter('repeat_cooldown').value,
            quiet_hours=(self.get_parameter('quiet_start_hour').value,
                         self.get_parameter('quiet_end_hour').value))

        self.baseline = self._load_baseline()
        self.routines = self._load_routines()
        self._last_person_at = 0.0
        self._feed = []            # recent observations, for the dashboard

        self.create_subscription(String, '/object_memory', self._cb_memory, 5)
        self.create_subscription(String, '/pose_detections', self._cb_poses, 5)
        # Signals the household's rhythm is made of. Deliberately NOT the
        # robot's own position: /current_room says where the ROBOT went, which
        # is its patrol schedule, not the family's habits.
        self.create_subscription(String, '/current_speaker', self._cb_speaker, 10)
        self.create_subscription(String, '/sound_events', self._cb_sound, 10)
        self.create_subscription(String, '/face_identities', self._cb_faces, 10)

        self._say_pub  = self.create_publisher(String, '/speech_response', 10)
        self._feed_pub = self.create_publisher(String, 'observations', 10)

        # Persist periodically: a baseline that only survives a clean shutdown
        # would be lost on every power cut, and this machine has those.
        self.create_timer(120.0, self._save_baseline)
        self.create_timer(120.0, self._save_routines)
        # Checked every few minutes, not every second: "this did not happen
        # today" cannot become true between one tick and the next.
        self.create_timer(300.0, self._check_routines)

        self.get_logger().info(
            f'Proactive observer ready — {len(self.baseline.known_labels())} '
            f'objects with a learned home, '
            f'{len(self.routines.established())} learned routines, '
            f'enabled={self.enabled}')

    # ── state ─────────────────────────────────────────────────────────────────

    def _load_baseline(self):
        try:
            with open(self.state_path, encoding='utf-8') as fh:
                return Baseline.from_dict(json.load(fh))
        except FileNotFoundError:
            return Baseline()
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt state file must not stop the node from coming up; the
            # baseline simply relearns.
            self.get_logger().warn(f'baseline unreadable ({exc}) — relearning')
            return Baseline()

    def _save_baseline(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.baseline.to_dict(), fh, ensure_ascii=False)
            os.replace(tmp, self.state_path)          # atomic
        except OSError as exc:
            self.get_logger().warn(f'could not save baseline: {exc}',
                                   throttle_duration_sec=300.0)

    def _load_routines(self):
        try:
            with open(_ROUTINE_PATH, encoding='utf-8') as fh:
                return RoutineLearner.from_dict(json.load(fh))
        except FileNotFoundError:
            return RoutineLearner()
        except (json.JSONDecodeError, OSError) as exc:
            self.get_logger().warn(f'routines unreadable ({exc}) — relearning')
            return RoutineLearner()

    def _save_routines(self):
        self.routines.prune()
        try:
            os.makedirs(os.path.dirname(_ROUTINE_PATH), exist_ok=True)
            tmp = _ROUTINE_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.routines.to_dict(), fh, ensure_ascii=False)
            os.replace(tmp, _ROUTINE_PATH)
        except OSError as exc:
            self.get_logger().warn(f'could not save routines: {exc}',
                                   throttle_duration_sec=300.0)

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_speaker(self, msg: String):
        name = (msg.data or '').strip()
        if name and name.lower() != 'unknown':
            self.routines.observe(f'person:{name}')

    def _cb_faces(self, msg: String):
        """Seeing somebody is a better rhythm signal than hearing them: people
        come home long before they say anything."""
        try:
            data = json.loads(msg.data) or {}
        except json.JSONDecodeError:
            return
        for face in data.get('faces', []) or []:
            name = face.get('name')
            if name and name != 'unknown':
                self.routines.observe(f'person:{name}')

    def _cb_sound(self, msg: String):
        try:
            data = json.loads(msg.data) or {}
        except json.JSONDecodeError:
            return
        for key in data.get('fired', []) or []:
            self.routines.observe(f'sound:{key}')

    def _check_routines(self):
        """Speak about a routine that should have happened and did not."""
        if not self.enabled:
            return
        now = time.time()
        hour = time.localtime(now).tm_hour
        person = self._person_present()
        for key, expected, late in self.routines.overdue(now=now):
            # Reuses Observation so the gate's dedup/cooldown/quiet-hours
            # logic is shared — a missing routine is just another thing
            # worth saying at most once in a while.
            obs = Observation('routine_missing', key, importance=0.7)
            if not self.gate.allow(obs, now=now, person_present=person,
                                   hour=hour):
                continue
            text = render_routine_el(key, expected, late)
            self._say_pub.publish(String(data=text))
            self.gate.record(obs, now=now)
            self._push_feed(obs, text, now)
            self.get_logger().info(f'Routine missing: {text}')
            break            # one remark per cycle


    def _cb_poses(self, msg: String):
        # ‼️ pose_node publishes {"persons": [...], "width": W, "height": H}.
        # Testing the payload's truthiness marked somebody present on EVERY
        # frame — the dict is non-empty even with zero people in it — which
        # defeated the "say it to someone" gate entirely.
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        persons = payload.get('persons', []) if isinstance(payload, dict) else payload
        if persons:
            self._last_person_at = time.time()

    def _person_present(self):
        if not self.require_person:
            return True
        return (time.time() - self._last_person_at) < _PERSON_MEMORY_S

    def _cb_memory(self, msg: String):
        try:
            instances = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(instances, list):
            return

        # Compare against the baseline BEFORE folding today's sightings in —
        # otherwise the observation is diluted by the very evidence it is about.
        observations = find_observations(self.baseline, instances,
                                         moved_threshold=self.moved_threshold)

        for inst in instances:
            label = inst.get('label')
            if label and inst.get('x') is not None:
                self.baseline.observe(label, inst['x'], inst['y'],
                                      inst.get('room'))

        if not self.enabled or not observations:
            return

        now = time.time()
        hour = time.localtime(now).tm_hour
        person = self._person_present()

        for obs in observations:
            if not self.gate.allow(obs, now=now, person_present=person,
                                   hour=hour):
                continue
            text = render_greek(obs)
            self._say_pub.publish(String(data=text))
            self.gate.record(obs, now=now)
            self._push_feed(obs, text, now)
            self.get_logger().info(f'Observed: {text}')
            break          # one remark per cycle, never a monologue

    def _push_feed(self, obs, text, now):
        self._feed.append({
            'text': text, 'kind': obs.kind, 'label': obs.label,
            'room': obs.room, 'from_room': obs.from_room,
            'x': obs.x, 'y': obs.y, 'ts': now,
        })
        del self._feed[:-30]
        self._feed_pub.publish(String(data=json.dumps(
            {'latest': self._feed[-1], 'feed': self._feed[-10:],
             'learned': len(self.baseline.known_labels())},
            ensure_ascii=False)))


def main():
    rclpy.init()
    node = ProactiveObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_baseline()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
