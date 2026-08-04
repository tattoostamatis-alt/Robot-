"""Who is who — one identity built from face, voice and body.

Three weak signals, each blind in a different way:

  * **Face** works silently and at a distance, and fails in the dark, from
    behind, and when somebody walks past looking away.
  * **Voice** works in the dark and around corners, and only exists while the
    person is actually talking — which is a small fraction of the time.
  * **Height** is always available whenever the skeleton is, and is nearly
    useless on its own: two adults in a household are often within a few
    centimetres. It earns its place as a VETO, not as an identifier — it is
    very good at saying "whoever that is, it is not the eight-year-old".

This module fuses them and is pure, so the fusion rules can be tested without a
camera, a microphone or a robot.

‼️ Height is measured against the MAP frame, where z = 0 is the floor. That
makes the head keypoint's z its height above the floor directly — no need to
see the feet, which are out of frame or behind furniture most of the time.
"""

import math

__all__ = ['Person', 'PersonRegistry', 'HEAD_TO_CROWN', 'fuse']

# The nose keypoint sits below the top of the head. Measured on adults at
# roughly 11 cm; it is a constant offset, not a per-person one, so it does not
# affect telling people apart — only the absolute number shown in the UI.
HEAD_TO_CROWN = 0.11

# Below this many samples the height statistics mean nothing yet.
_MIN_SAMPLES = 8
# A height must differ from someone's learned height by at least this much
# before it is allowed to veto a match, whatever the standard deviation says.
# Posture, shoes and a slouch move a real person by several centimetres.
_MIN_VETO_GAP = 0.10


class Person:
    """One enrolled person and what has been learned about them."""

    __slots__ = ('name', 'has_face', 'has_voice', '_n', '_mean', '_m2',
                 'last_seen')

    def __init__(self, name):
        self.name = name
        self.has_face = False
        self.has_voice = False
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.last_seen = 0.0

    # ── height statistics (Welford, so it never stores the samples) ───────────

    def observe_height(self, metres):
        """Fold one height measurement in. Implausible values are ignored.

        The bounds are deliberately wide — a toddler and a tall adult both have
        to fit — but they reject the two failure modes that actually happen: a
        depth reading that landed on the wall behind the person, and a skeleton
        whose head keypoint jumped to the floor.
        """
        if not (0.5 <= metres <= 2.3):
            return False
        self._n += 1
        delta = metres - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (metres - self._mean)
        return True

    @property
    def samples(self):
        return self._n

    @property
    def height(self):
        """Learned height in metres, or None while there is not enough data."""
        return self._mean if self._n >= _MIN_SAMPLES else None

    @property
    def height_sd(self):
        if self._n < 2:
            return 0.0
        return math.sqrt(self._m2 / (self._n - 1))

    def height_conflicts(self, metres):
        """True when this measurement cannot plausibly be this person.

        Used only to REJECT, never to select — see the module note. Requires a
        settled height, and a gap larger than both three standard deviations
        and a fixed floor, so a slouch or thick shoes cannot veto anybody.
        """
        learned = self.height
        if learned is None or metres is None:
            return False
        gap = abs(metres - learned)
        return gap > max(_MIN_VETO_GAP, 3.0 * self.height_sd)

    def to_dict(self):
        return {'name': self.name, 'has_face': self.has_face,
                'has_voice': self.has_voice, 'n': self._n,
                'mean': self._mean, 'm2': self._m2,
                'last_seen': self.last_seen}

    @classmethod
    def from_dict(cls, d):
        p = cls(d.get('name', ''))
        p.has_face = bool(d.get('has_face'))
        p.has_voice = bool(d.get('has_voice'))
        p._n = int(d.get('n', 0))
        p._mean = float(d.get('mean', 0.0))
        p._m2 = float(d.get('m2', 0.0))
        p.last_seen = float(d.get('last_seen', 0.0))
        return p


def fuse(face_name, voice_name, height, registry):
    """Decide who this is. Returns (name, confidence, reason).

    name is None when nothing is confident enough. The reason is carried
    through to the UI because "why does it think that" is the first question
    anyone asks when it gets somebody wrong.
    """
    face = face_name or None
    voice = voice_name or None

    # Height never identifies; it only removes a candidate.
    def vetoed(name):
        p = registry.get(name)
        return bool(p and p.height_conflicts(height))

    if face and voice and face == voice:
        if vetoed(face):
            return None, 0.0, 'ύψος ασυμβίβαστο'
        return face, 0.98, 'πρόσωπο + φωνή'

    if face and voice and face != voice:
        # Disagreement. Prefer the face: a voice can come out of a television
        # or a phone on speaker, a face in front of the camera cannot.
        if vetoed(face):
            return (None, 0.0, 'ύψος ασυμβίβαστο') if vetoed(voice) \
                else (voice, 0.55, 'φωνή (το πρόσωπο διαφωνεί)')
        return face, 0.6, 'πρόσωπο (η φωνή διαφωνεί)'

    if face:
        return (None, 0.0, 'ύψος ασυμβίβαστο') if vetoed(face) \
            else (face, 0.85, 'πρόσωπο')

    if voice:
        return (None, 0.0, 'ύψος ασυμβίβαστο') if vetoed(voice) \
            else (voice, 0.8, 'φωνή')

    return None, 0.0, 'άγνωστος'


class PersonRegistry:
    """Everyone the robot has been introduced to."""

    def __init__(self):
        self._people = {}

    def __len__(self):
        return len(self._people)

    def __contains__(self, name):
        return name in self._people

    def names(self):
        return sorted(self._people)

    def get(self, name):
        return self._people.get(name)

    def add(self, name):
        """Create a person if new. Returns the Person either way."""
        name = (name or '').strip()
        if not name:
            return None
        if name not in self._people:
            self._people[name] = Person(name)
        return self._people[name]

    def remove(self, name):
        return self._people.pop(name, None) is not None

    def mark_face(self, name, present=True):
        p = self.add(name)
        if p:
            p.has_face = present

    def mark_voice(self, name, present=True):
        p = self.add(name)
        if p:
            p.has_voice = present

    def observe(self, name, height=None, now=None):
        """Record a sighting: refreshes last_seen and folds in the height."""
        p = self._people.get(name)
        if p is None:
            return False
        if now is not None:
            p.last_seen = now
        if height is not None:
            p.observe_height(height)
        return True

    def identify(self, face_name=None, voice_name=None, height=None):
        return fuse(face_name, voice_name, height, self)

    def summary(self):
        """What the UI shows: who is known and how completely."""
        out = []
        for name in self.names():
            p = self._people[name]
            out.append({
                'name': name,
                'face': p.has_face,
                'voice': p.has_voice,
                'height': round(p.height, 2) if p.height is not None else None,
                'height_samples': p.samples,
                'last_seen': p.last_seen,
                # Three ticks is a complete profile; the UI shows the gaps.
                'complete': p.has_face and p.has_voice and p.height is not None,
            })
        return out

    def to_dict(self):
        return {'people': [p.to_dict() for p in self._people.values()]}

    @classmethod
    def from_dict(cls, data):
        r = cls()
        for d in (data or {}).get('people', []) or []:
            p = Person.from_dict(d)
            if p.name:
                r._people[p.name] = p
        return r
