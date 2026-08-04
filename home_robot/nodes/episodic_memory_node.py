#!/usr/bin/env python3
"""Episodic memory — "τι έγινε σήμερα;", "πότε ήρθε ο Μαξ;"

The robot already remembers facts (rag_memory) and places (object_memory).
Neither is indexed by time, so nothing could answer a question about *when*.
This node keeps a timeline of what happened and recalls slices of it.

Topics
  in   /speech_text       (String)  — what a person said
       /speech_response   (String)  — what the robot answered
       /current_room      (String)  — where the robot is
       /current_speaker   (String)  — who is talking (diarization)
       /observations      (String, JSON) — proactive remarks
       /episodic/query    (String)  — a question in Greek
  out  /episodic/answer   (String)  — the spoken answer
       /episodic/timeline (String, JSON) — the feed the dashboard renders

Persists to ~/.ros/home_robot_episodic.json so the timeline survives the power
cuts this machine is prone to.

‼️ The speaker is attached from the LAST `/current_speaker` seen, which is how
diarization publishes it — slightly before the transcript of the same utterance.
Anything older than `speaker_ttl` is treated as unknown rather than blamed on
whoever spoke last, so a question at noon is not attributed to the person who
happened to talk at breakfast.
"""

import json
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

from home_robot.episodic import (
    KIND_HEARD, KIND_OBSERVED, KIND_ROOM, KIND_SAID, EpisodicLog,
    parse_time_window,
)
from home_robot.status_query import ROOM_LOCATIVE_EL, room_el

_STATE_PATH = os.path.expanduser('~/.ros/home_robot_episodic.json')


def _clock(ts):
    return time.strftime('%H:%M', time.localtime(ts))


def _day_label(ts, now):
    """'σήμερα' / 'χθες' / a date, for rendering an answer."""
    today = time.strftime('%Y-%m-%d', time.localtime(now))
    day   = time.strftime('%Y-%m-%d', time.localtime(ts))
    if day == today:
        return 'σήμερα'
    if time.strftime('%Y-%m-%d', time.localtime(now - 86400.0)) == day:
        return 'χθες'
    return time.strftime('%d/%m', time.localtime(ts))


def summarize(events, now=None, max_lines=6):
    """A spoken Greek answer for a set of recalled events."""
    now = time.time() if now is None else now
    if not events:
        return 'Δεν θυμάμαι κάτι για τότε.'

    lines = []
    for ev in events[-max_lines:]:
        when = _clock(ev.ts)
        if ev.kind == KIND_HEARD:
            # Name in apposition rather than "ο {name} είπε": the article is
            # gendered and diarization only knows a label, so "ο Μαρία" is one
            # enrolment away. "ο κάποιος είπε" was already wrong for the
            # unknown-speaker case. The colon form needs neither.
            if ev.who:
                lines.append(f'{when}, {ev.who}: «{ev.text}»')
            else:
                lines.append(f'{when}, κάποιος είπε «{ev.text}»')
        elif ev.kind == KIND_SAID:
            lines.append(f'{when}, απάντησα «{ev.text}»')
        elif ev.kind == KIND_ROOM:
            lines.append(f'{when}, ήμουν {ROOM_LOCATIVE_EL.get(ev.room, ev.text)}')
        elif ev.kind == KIND_OBSERVED:
            lines.append(f'{when}, παρατήρησα: {ev.text}')
        else:
            lines.append(f'{when}, {ev.text}')

    head = _day_label(events[0].ts, now)
    body = '. '.join(lines)
    extra = ''
    if len(events) > max_lines:
        extra = f' (και άλλα {len(events) - max_lines})'
    return f'{head.capitalize()}: {body}.{extra}'


class EpisodicMemoryNode(Node):
    def __init__(self):
        super().__init__('episodic_memory_node')

        self.declare_parameter('max_events', 5000)
        self.declare_parameter('max_age_days', 14)
        self.declare_parameter('speaker_ttl', 30.0)      # s
        self.declare_parameter('state_path', _STATE_PATH)
        self.declare_parameter('log_robot_replies', True)

        self.speaker_ttl = self.get_parameter('speaker_ttl').value
        self.state_path  = self.get_parameter('state_path').value
        self.log_replies = self.get_parameter('log_robot_replies').value

        self.log = self._load()
        self._speaker = None
        self._speaker_at = 0.0

        self.create_subscription(String, '/speech_text', self._cb_heard, 10)
        self.create_subscription(String, '/speech_response', self._cb_said, 10)
        self.create_subscription(String, '/current_speaker', self._cb_speaker, 10)
        self.create_subscription(String, '/observations', self._cb_observation, 10)

        latched = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/current_room', self._cb_room, latched)
        self.create_subscription(String, '/episodic/query', self._cb_query, 10)

        self._answer_pub   = self.create_publisher(String, 'episodic/answer', 10)
        self._timeline_pub = self.create_publisher(String, 'episodic/timeline', latched)

        self.create_timer(60.0, self._save)
        self.create_timer(5.0, self._publish_timeline)

        self.get_logger().info(f'Episodic memory ready — {len(self.log)} events '
                               f'remembered')

    # ── state ─────────────────────────────────────────────────────────────────

    def _load(self):
        max_events = self.get_parameter('max_events').value
        max_age    = self.get_parameter('max_age_days').value
        try:
            with open(self.state_path, encoding='utf-8') as fh:
                log = EpisodicLog.from_dict(json.load(fh))
            log.max_events, log.max_age_days = max_events, max_age
            return log
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as exc:
            self.get_logger().warn(f'timeline unreadable ({exc}) — starting fresh')
        return EpisodicLog(max_events=max_events, max_age_days=max_age)

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(self.log.to_dict(), fh, ensure_ascii=False)
            os.replace(tmp, self.state_path)          # atomic
        except OSError as exc:
            self.get_logger().warn(f'could not save timeline: {exc}',
                                   throttle_duration_sec=300.0)

    # ── inputs ────────────────────────────────────────────────────────────────

    def _cb_speaker(self, msg: String):
        name = (msg.data or '').strip()
        if name and name.lower() != 'unknown':
            self._speaker = name
            self._speaker_at = time.time()

    def _current_speaker(self):
        if self._speaker and (time.time() - self._speaker_at) < self.speaker_ttl:
            return self._speaker
        return None

    def _cb_heard(self, msg: String):
        text = (msg.data or '').strip()
        if text:
            self.log.add(KIND_HEARD, text, who=self._current_speaker())

    def _cb_said(self, msg: String):
        if not self.log_replies:
            return
        text = (msg.data or '').strip()
        if text:
            self.log.add(KIND_SAID, text)

    def _cb_room(self, msg: String):
        room = (msg.data or '').strip()
        if room:
            # ROOM_LOCATIVE_EL carries the inflected article — "στην κουζίνα",
            # not "στο κουζίνα". Composing 'στο ' + the bare noun produced
            # "ήμουν στο τουαλέτα", which is exactly why the two tables exist.
            where = ROOM_LOCATIVE_EL.get(room, f'στο {room_el(room)}')
            self.log.add(KIND_ROOM, f'ήμουν {where}', room=room)

    def _cb_observation(self, msg: String):
        try:
            latest = (json.loads(msg.data) or {}).get('latest') or {}
        except json.JSONDecodeError:
            return
        text = latest.get('text')
        if text:
            self.log.add(KIND_OBSERVED, text, room=latest.get('room'))

    # ── recall ────────────────────────────────────────────────────────────────

    def _cb_query(self, msg: String):
        phrase = (msg.data or '').strip()
        now = time.time()
        window = parse_time_window(phrase, now=now)
        # No period named at all: default to today rather than all of history,
        # so "τι έγινε;" is a question about the day and not about a fortnight.
        if window is None and not phrase:
            window = parse_time_window('σήμερα', now=now)

        events = self.log.query(phrase or None, window=window, now=now)
        answer = summarize(events, now=now)
        self._answer_pub.publish(String(data=answer))
        self.get_logger().info(f'Recall «{phrase}» → {len(events)} events')

    def _publish_timeline(self):
        events = self.log.recent(60)
        self._timeline_pub.publish(String(data=json.dumps({
            'count': len(self.log),
            'events': [{**e.to_dict(), 'clock': _clock(e.ts)} for e in events],
        }, ensure_ascii=False)))


def main():
    rclpy.init()
    node = EpisodicMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
