"""Household sound events — turning 521 AudioSet classes into things worth saying.

YAMNet recognises 521 sounds. A home robot cares about maybe a dozen, and the
other 509 are noise that would make it unbearable. This module is the filter,
and it is pure so the judgement can be tested with score vectors instead of a
microphone.

Design notes:

  * **Grouped, not per-class.** "Glass", "Shatter", "Breaking" and "Smash,
    crash" are four AudioSet classes for one household event. Scoring the group
    by its strongest member means a sound that splits its confidence across
    near-synonyms still fires.
  * **Max over frames, not mean.** YAMNet scores every 0.48 s. Breaking glass
    lasts a fraction of a second, so averaging a 2 s window buries it under the
    silence around it. Sustained sounds (running water) clear the bar anyway.
  * **Speech is not an event.** It is the most common class in a house by far
    and the voice stack already handles it; reporting it would be constant.
    It is tracked only as a presence hint.
"""

import math
import time

__all__ = ['EVENTS', 'classify', 'SoundEventGate', 'bearing_el', 'describe']

# event key -> (Greek noun phrase, importance 0..1, AudioSet class names)
# Importance drives both ordering and whether the robot speaks unprompted.
EVENTS = {
    'doorbell': ('το κουδούνι της πόρτας', 0.95,
                 ('Doorbell', 'Ding-dong', 'Ding')),
    'knock': ('χτύπημα στην πόρτα', 0.85,
              ('Knock', 'Door')),
    'glass': ('σπασμένο γυαλί', 1.0,
              ('Glass', 'Shatter', 'Breaking', 'Smash, crash')),
    'alarm': ('συναγερμός', 1.0,
              ('Alarm', 'Smoke detector, smoke alarm', 'Fire alarm',
               'Siren', 'Civil defense siren', 'Alarm clock')),
    'emergency': ('σειρήνα από όχημα', 0.5,
                  ('Police car (siren)', 'Ambulance (siren)',
                   'Fire engine, fire truck (siren)')),
    'phone': ('το τηλέφωνο', 0.7,
              ('Telephone', 'Telephone bell ringing', 'Ringtone')),
    'water': ('νερό που τρέχει', 0.6,
              ('Water tap, faucet', 'Sink (filling or washing)', 'Water',
               'Boiling')),
    'baby': ('μωρό που κλαίει', 1.0,
             ('Baby cry, infant cry', 'Crying, sobbing')),
    'scream': ('φωνή', 0.95,
               ('Screaming',)),
    'dog': ('σκύλος', 0.4,
            ('Dog', 'Bark', 'Whimper (dog)')),
    'thud': ('κάτι έπεσε', 0.8,
             ('Thump, thud', 'Slam')),
    'microwave': ('ο φούρνος μικροκυμάτων', 0.3,
                  ('Microwave oven',)),
}

# Tracked for presence, never announced. See the module docstring.
SPEECH_CLASSES = ('Speech', 'Conversation', 'Narration, monologue',
                  'Child speech, kid speaking')


def classify(scores, class_names, threshold=0.35, speech_threshold=0.5):
    """Score vector -> [(event_key, score), ...] strongest first, plus speech.

    ``scores`` is either one (521,) vector or a (frames, 521) block; a block is
    reduced with max per class, for the reason in the module docstring.

    Returns (events, speech_score).
    """
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 2:
        arr = arr.max(axis=0)
    index = {name: i for i, name in enumerate(class_names)}

    def group_score(names):
        vals = [arr[index[n]] for n in names if n in index]
        return max(vals) if vals else 0.0

    events = []
    for key, (_, _, names) in EVENTS.items():
        score = group_score(names)
        if score >= threshold:
            events.append((key, round(float(score), 3)))
    events.sort(key=lambda kv: (-kv[1], kv[0]))

    speech = group_score(SPEECH_CLASSES)
    return events, (round(float(speech), 3) if speech >= speech_threshold else 0.0)


# The XVF3800 reports 0-359 degrees. 0 is straight ahead and the angle grows
# counter-clockwise, so the right-hand side of the robot is the 270 sector.
_BEARINGS = (
    (337.5, 360.0, 'μπροστά'), (0.0, 22.5, 'μπροστά'),
    (22.5, 67.5, 'μπροστά αριστερά'),
    (67.5, 112.5, 'αριστερά'),
    (112.5, 157.5, 'πίσω αριστερά'),
    (157.5, 202.5, 'πίσω'),
    (202.5, 247.5, 'πίσω δεξιά'),
    (247.5, 292.5, 'δεξιά'),
    (292.5, 337.5, 'μπροστά δεξιά'),
)


def bearing_el(angle_deg):
    """Greek direction word for a DoA angle, or None if there is no angle.

    None matters: without the XVF3800's DoA the robot knows WHAT it heard but
    not WHERE, and saying a made-up direction is worse than omitting it.
    """
    if angle_deg is None or (isinstance(angle_deg, float) and math.isnan(angle_deg)):
        return None
    a = float(angle_deg) % 360.0
    for lo, hi, word in _BEARINGS:
        if lo <= a < hi:
            return word
    return 'μπροστά'


def describe(event_key, angle_deg=None):
    """The sentence for a detected event: «Ακούω το κουδούνι της πόρτας, δεξιά.»"""
    noun = EVENTS.get(event_key, (event_key, 0.0, ()))[0]
    where = bearing_el(angle_deg)
    if where:
        return f'Ακούω {noun}, {where}.'
    return f'Ακούω {noun}.'


class SoundEventGate:
    """Debounce and rate-limit sound events.

    A doorbell rings for two seconds and YAMNet sees it in four consecutive
    windows; without this the robot would announce it four times. Each event key
    has its own cooldown so a doorbell does not suppress a smoke alarm.

    ``min_repeats`` requires an event to be seen in more than one window before
    it counts, which is what separates a real sound from a single-window
    misfire. Events at or above ``instant_importance`` bypass that: waiting for
    a second window of breaking glass or a smoke alarm is the wrong trade.

    ‼️ But only when the score is convincing. On the first live run the robot
    announced «Ακούω μωρό που κλαίει» from a single 0.53 window of ordinary room
    noise — 'baby' is importance 1.0, so it skipped the repeat check on a
    marginal score. ``instant_min_score`` is the second half of that bypass: a
    critical event fires on one window only if it is also *confident*, and
    otherwise falls back to needing corroboration like everything else.
    """

    def __init__(self, cooldown=20.0, min_repeats=2, repeat_window=3.0,
                 instant_importance=1.0, instant_min_score=0.75):
        self.cooldown = cooldown
        self.min_repeats = min_repeats
        self.repeat_window = repeat_window
        self.instant_importance = instant_importance
        self.instant_min_score = instant_min_score
        self._last_fired = {}       # key -> ts
        self._pending = {}          # key -> [ts, ...]

    def _importance(self, key):
        return EVENTS.get(key, ('', 0.0, ()))[1]

    def update(self, events, now=None):
        """Feed one window's [(key, score)]. Returns the keys to announce now."""
        now = time.time() if now is None else now
        fired = []
        for key, score in events:
            if now - self._last_fired.get(key, -1e9) < self.cooldown:
                continue

            if (self._importance(key) >= self.instant_importance
                    and score >= self.instant_min_score):
                self._last_fired[key] = now
                self._pending.pop(key, None)
                fired.append(key)
                continue

            seen = [t for t in self._pending.get(key, [])
                    if now - t <= self.repeat_window]
            seen.append(now)
            self._pending[key] = seen
            if len(seen) >= self.min_repeats:
                self._last_fired[key] = now
                self._pending.pop(key, None)
                fired.append(key)

        fired.sort(key=lambda k: -self._importance(k))
        return fired

    def reset(self):
        self._last_fired.clear()
        self._pending.clear()
