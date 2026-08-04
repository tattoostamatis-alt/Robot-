"""Proactive observation — deciding when the robot should speak first.

Everything here is pure, so the judgement calls can be tested without a robot,
a camera or a clock.

The hard part of a robot that volunteers information is not noticing changes —
`object_memory` already tracks where things are. It is noticing *few enough* of
them. A robot that reports every difference is worse than a silent one: it is
noise, and noise gets muted. Three ideas do the filtering:

  * **A baseline has to be earned.** An object counts as "normally here" only
    after being seen in the same spot on several *separate* occasions, well
    apart in time. Without the separation rule a single long look at a room
    would mint a baseline from one sighting, and the very next visit would
    report everything in it as moved.
  * **Only differences that mean something.** A cup that shifted 20 cm along
    the counter is not news. A cup that moved to another room is.
  * **Say it to someone.** Observations are held until a person is actually
    present; announcing to an empty room burns the observation for nothing.
"""

import math
import time

__all__ = ['Observation', 'Baseline', 'ObservationGate', 'find_observations']

# How far an object must move before the move is worth mentioning, in metres.
# Below this it is detector jitter or somebody nudging a mug.
_MOVED_THRESHOLD = 1.5

# Sightings closer together than this belong to the same visit, so they cannot
# add independent evidence to a baseline. Twenty minutes comfortably separates
# "the robot is parked here staring" from "the robot came back later".
_VISIT_GAP_S = 1200.0

# Sightings needed, on separate visits, before a position is "normal".
_BASELINE_VISITS = 3


class Observation:
    """Something worth saying, with enough structure to render or rank it."""

    __slots__ = ('kind', 'label', 'room', 'from_room', 'importance', 'x', 'y')

    def __init__(self, kind, label, room=None, from_room=None,
                 importance=0.5, x=None, y=None):
        self.kind = kind                # 'moved_room' | 'moved' | 'new_in_room'
        self.label = label
        self.room = room                # where it is now
        self.from_room = from_room      # where it used to be
        self.importance = importance    # 0..1, used to rank and to gate
        self.x, self.y = x, y

    def key(self):
        """Identity for de-duplication: the same object making the same kind of
        news is one observation, however many times it is re-derived."""
        return (self.kind, self.label, self.room, self.from_room)

    def __repr__(self):                                  # pragma: no cover
        return (f'<Observation {self.kind} {self.label} '
                f'{self.from_room}->{self.room} imp={self.importance}>')

    def __eq__(self, other):
        return isinstance(other, Observation) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class Baseline:
    """Where things normally are, learned from repeated visits.

    Keyed by object label. Each entry keeps the settled position, the room, and
    the timestamps of the visits that support it.
    """

    def __init__(self, visits_required=_BASELINE_VISITS, visit_gap=_VISIT_GAP_S):
        self.visits_required = visits_required
        self.visit_gap = visit_gap
        self._entries = {}       # label -> {x, y, room, visits:[ts], settled:bool}

    def observe(self, label, x, y, room, now=None):
        """Fold in a sighting. Returns True once this label has a settled home."""
        now = time.time() if now is None else now
        e = self._entries.get(label)
        if e is None:
            self._entries[label] = {
                'x': x, 'y': y, 'room': room, 'visits': [now], 'settled': False}
            return False

        # A sighting during the same visit refines the position but earns no
        # extra credit towards settling.
        if now - e['visits'][-1] >= self.visit_gap:
            e['visits'].append(now)

        # Moving the stored position towards the sighting keeps the baseline
        # current for objects that drift slowly (a chair nudged week to week).
        n = len(e['visits'])
        e['x'] += (x - e['x']) / max(2, n)
        e['y'] += (y - e['y']) / max(2, n)
        if room:
            e['room'] = room

        e['settled'] = len(e['visits']) >= self.visits_required
        return e['settled']

    def home_of(self, label):
        """(x, y, room) where this label normally lives, or None if not settled."""
        e = self._entries.get(label)
        if e is None or not e['settled']:
            return None
        return e['x'], e['y'], e['room']

    def is_settled(self, label):
        return self.home_of(label) is not None

    def known_labels(self):
        return [l for l, e in self._entries.items() if e['settled']]

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self):
        return {'visits_required': self.visits_required,
                'visit_gap': self.visit_gap,
                'entries': self._entries}

    @classmethod
    def from_dict(cls, data):
        b = cls(visits_required=data.get('visits_required', _BASELINE_VISITS),
                visit_gap=data.get('visit_gap', _VISIT_GAP_S))
        b._entries = data.get('entries', {}) or {}
        return b


def find_observations(baseline, instances, moved_threshold=_MOVED_THRESHOLD):
    """Compare what is visible now against what is normal.

    ``instances`` are object_memory dicts: label, x, y, room. Returns a list of
    Observation, most important first. Objects with no settled baseline produce
    nothing — the robot stays quiet about a room it has not learned yet, which
    is the correct behaviour on a fresh map.
    """
    out = []
    for inst in instances:
        label = inst.get('label')
        if not label:
            continue
        home = baseline.home_of(label)
        if home is None:
            continue

        hx, hy, hroom = home
        x, y = inst.get('x'), inst.get('y')
        room = inst.get('room')
        if x is None or y is None:
            continue

        if room and hroom and room != hroom:
            # A different room is the strongest signal that something is out of
            # place, and the easiest sentence to say usefully.
            out.append(Observation('moved_room', label, room=room,
                                   from_room=hroom, importance=0.9, x=x, y=y))
            continue

        if math.dist((x, y), (hx, hy)) >= moved_threshold:
            out.append(Observation('moved', label, room=room, from_room=hroom,
                                   importance=0.5, x=x, y=y))

    out.sort(key=lambda o: o.importance, reverse=True)
    return out


class ObservationGate:
    """Decides whether an observation may be spoken *now*.

    Rate limiting is the whole feature. The gate enforces, in order:
      * nothing while no one is there to hear it,
      * nothing during quiet hours,
      * a minimum gap between two spoken observations,
      * a cap per rolling hour,
      * and never the same observation twice within a cooldown.
    """

    def __init__(self, min_interval=180.0, max_per_hour=4,
                 repeat_cooldown=7200.0, quiet_hours=(23, 8)):
        self.min_interval = min_interval
        self.max_per_hour = max_per_hour
        self.repeat_cooldown = repeat_cooldown
        self.quiet_hours = quiet_hours
        self._last_spoken_at = 0.0
        self._spoken_at = []          # rolling timestamps for the hourly cap
        self._seen = {}               # observation key -> last spoken ts

    def _in_quiet_hours(self, hour):
        start, end = self.quiet_hours
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end          # window wraps midnight

    def allow(self, obs, now=None, person_present=True, hour=None):
        """True if this observation should be spoken right now."""
        now = time.time() if now is None else now
        if not person_present:
            return False
        if hour is not None and self._in_quiet_hours(hour):
            return False
        if now - self._last_spoken_at < self.min_interval:
            return False

        self._spoken_at = [t for t in self._spoken_at if now - t < 3600.0]
        if len(self._spoken_at) >= self.max_per_hour:
            return False

        last = self._seen.get(obs.key())
        if last is not None and now - last < self.repeat_cooldown:
            return False
        return True

    def record(self, obs, now=None):
        """Mark an observation as spoken. Call only after actually saying it."""
        now = time.time() if now is None else now
        self._last_spoken_at = now
        self._spoken_at.append(now)
        self._seen[obs.key()] = now
