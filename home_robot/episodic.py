"""Episodic memory — what happened in this house, and when.

`rag_memory` answers "what do I know about X" and `object_memory` answers
"where is X". Neither can answer "πότε γύρισε ο Μαξ;" or "τι έγινε σήμερα;",
because neither is indexed by time. This module is the missing axis.

Pure and clock-injectable so the whole thing tests without a robot.

The interesting part is not storage — it is turning a spoken Greek phrase into
a time window. "Σήμερα το πρωί", "χθες", "πριν από δύο ώρες" and "τώρα" are how
people actually ask, and each has to become a concrete (start, end) pair before
any filtering can happen.
"""

import re
import time
import unicodedata

__all__ = ['Event', 'EpisodicLog', 'parse_time_window', 'strip_accents']

# Event kinds, kept short because they are also the dashboard's filter chips.
KIND_HEARD    = 'heard'       # somebody said something
KIND_SAID     = 'said'        # the robot answered
KIND_ROOM     = 'room'        # the robot changed room
KIND_OBSERVED = 'observed'    # a proactive observation
KIND_SAW      = 'saw'         # a person was recognised
KIND_MISSION  = 'mission'     # a mission started or finished

_DAY = 86400.0
_HOUR = 3600.0


def strip_accents(text):
    """Greek text without diacritics, lowercased.

    Speech-to-text accents are unreliable ("ΤΩΡΑ" transcribes unaccented, "τώρα"
    accented), so every phrase match happens on the stripped form.
    """
    if not text:
        return ''
    decomposed = unicodedata.normalize('NFD', text.lower())
    return ''.join(c for c in decomposed if not unicodedata.combining(c))


def _day_bounds(now, days_ago=0):
    """(start, end) of a local calendar day, `days_ago` before today."""
    lt = time.localtime(now)
    midnight = now - (lt.tm_hour * _HOUR + lt.tm_min * 60 + lt.tm_sec)
    start = midnight - days_ago * _DAY
    return start, start + _DAY


# Relative windows expressed in hours back from now.
_RELATIVE = (
    ('τελευταια ωρα', 1.0),
    ('τελευταιο τεταρτο', 0.25),
    ('μολις', 0.25),
    ('τωρα', 0.5),
)

_NUMBER_WORDS = {
    'μια': 1, 'ενα': 1, 'δυο': 2, 'τρεις': 3, 'τρια': 3, 'τεσσερις': 4,
    'τεσσερα': 4, 'πεντε': 5, 'εξι': 6, 'επτα': 7, 'εφτα': 7, 'οκτω': 8,
    'οχτω': 8, 'εννεα': 9, 'εννια': 9, 'δεκα': 10,
}


def parse_time_window(phrase, now=None):
    """Turn a Greek phrase into (start, end) epoch seconds.

    Returns None when the phrase names no period, so the caller can fall back to
    a default (usually "today") rather than silently searching all of history.
    """
    now = time.time() if now is None else now
    p = strip_accents(phrase)
    if not p:
        return None

    # Order matters: "χθες το βράδυ" must be tested before plain "χθες", and
    # "προχθες" contains "χθες" as a substring.
    if 'προχθες' in p:
        return _day_bounds(now, 2)

    if 'χθες' in p or 'χτες' in p:
        start, end = _day_bounds(now, 1)
        return _narrow_to_part_of_day(p, start, end)

    if 'σημερα' in p or 'μερα' == p:
        start, end = _day_bounds(now, 0)
        return _narrow_to_part_of_day(p, start, end)

    if 'αποψε' in p:
        start, _ = _day_bounds(now, 0)
        return start + 18 * _HOUR, now

    # "πριν από N ώρες/λεπτά/μέρες"
    m = re.search(r'πριν(?:\s+απο)?\s+(\d+|[α-ω]+)\s*(ωρ|λεπτ|μερ|μέρ)', p)
    if m:
        raw = m.group(1)
        count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)
        if count:
            unit = m.group(2)
            span = {'ωρ': _HOUR, 'λεπτ': 60.0}.get(unit, _DAY)
            return now - count * span, now

    for needle, hours in _RELATIVE:
        if needle in p:
            return now - hours * _HOUR, now

    # A bare part-of-day with no other qualifier means today's.
    start, end = _day_bounds(now, 0)
    narrowed = _narrow_to_part_of_day(p, start, end, require_match=True)
    if narrowed is not None:
        w_start, w_end = narrowed
        if w_start > now:
            # The period has not begun today, so the speaker means the last one
            # that actually happened — at 14:30, "τι έγινε το βράδυ" is asking
            # about last night. Clipping to now instead would invert the window
            # and silently answer "nothing".
            return w_start - _DAY, w_end - _DAY
        # Never claim a window that runs into the future.
        return w_start, min(w_end, now)

    return None


def _narrow_to_part_of_day(phrase, start, end, require_match=False):
    """Cut a whole day down to morning / noon / afternoon / evening."""
    parts = (
        (('πρωι',), 5, 12),
        (('μεσημερ',), 12, 16),
        (('απογευμα',), 16, 19),
        (('βραδ', 'νυχτα', 'απογευματ'), 19, 24),
    )
    for needles, h0, h1 in parts:
        if any(n in phrase for n in needles):
            return start + h0 * _HOUR, start + h1 * _HOUR
    return None if require_match else (start, end)


class Event:
    """One thing that happened, at one moment."""

    __slots__ = ('ts', 'kind', 'text', 'room', 'who')

    def __init__(self, ts, kind, text, room=None, who=None):
        self.ts = ts
        self.kind = kind
        self.text = text
        self.room = room
        self.who = who

    def to_dict(self):
        return {'ts': self.ts, 'kind': self.kind, 'text': self.text,
                'room': self.room, 'who': self.who}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('ts', 0.0), d.get('kind', ''), d.get('text', ''),
                   d.get('room'), d.get('who'))

    def __repr__(self):                                  # pragma: no cover
        return f'<Event {self.kind} {self.text!r} @{self.ts}>'


class EpisodicLog:
    """A bounded, time-ordered log of events with recall by window and keyword."""

    def __init__(self, max_events=5000, max_age_days=14):
        self.max_events = max_events
        self.max_age_days = max_age_days
        self._events = []

    def __len__(self):
        return len(self._events)

    def add(self, kind, text, room=None, who=None, ts=None):
        """Append an event. Returns it, or None if it was suppressed.

        Consecutive identical events are collapsed: `/current_room` republishes
        on a timer, and without this the log would fill with "μπήκα στο σαλόνι"
        repeated hundreds of times and drown everything else.
        """
        ts = time.time() if ts is None else ts
        if self._events:
            last = self._events[-1]
            if last.kind == kind and last.text == text and last.who == who:
                return None
        ev = Event(ts, kind, text, room, who)
        self._events.append(ev)
        self._trim(now=ts)
        return ev

    def _trim(self, now):
        cutoff = now - self.max_age_days * _DAY
        if self._events and self._events[0].ts < cutoff:
            self._events = [e for e in self._events if e.ts >= cutoff]
        if len(self._events) > self.max_events:
            del self._events[:-self.max_events]

    def all(self):
        return list(self._events)

    def recent(self, count=20):
        return list(self._events[-count:])

    def between(self, start, end):
        return [e for e in self._events if start <= e.ts < end]

    def query(self, phrase=None, window=None, kinds=None, who=None,
              now=None, limit=50):
        """Recall events by time window, kind, speaker and keyword.

        `window` wins over `phrase`; if neither yields a window, the search
        covers everything retained rather than nothing — "τι έγινε με τα
        κλειδιά" is a keyword question with no time in it at all.
        """
        now = time.time() if now is None else now
        if window is None and phrase:
            window = parse_time_window(phrase, now=now)

        events = self.between(*window) if window else list(self._events)

        if kinds:
            kinds = set(kinds)
            events = [e for e in events if e.kind in kinds]
        if who:
            target = strip_accents(who)
            events = [e for e in events if e.who and strip_accents(e.who) == target]

        if phrase:
            keywords = _content_words(phrase)
            if keywords:
                events = [e for e in events
                          if _matches_keywords(e, keywords)] or events

        return events[-limit:]

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self):
        return {'max_events': self.max_events,
                'max_age_days': self.max_age_days,
                'events': [e.to_dict() for e in self._events]}

    @classmethod
    def from_dict(cls, data):
        log = cls(max_events=data.get('max_events', 5000),
                  max_age_days=data.get('max_age_days', 14))
        log._events = [Event.from_dict(d) for d in data.get('events', []) or []]
        log._events.sort(key=lambda e: e.ts)
        return log


# Words that carry no search signal — asking "τι έγινε σήμερα" must not filter
# the day down to events containing the literal word "έγινε".
_STOPWORDS = frozenset("""
τι ποιος ποια ποιο ποτε που πως γιατι ειπε ειπα εγινε γινε ημουν ησουν ηταν
ο η το οι τα του της των τον την στο στη στην στον στα στις σε με και για
απο ενα μια ενας μου σου του μας σας τους θυμασαι ξερεις εχει ειναι
σημερα χθες χτες προχθες τωρα αποψε πρωι βραδυ μεσημερι απογευμα μερα ωρα
ωρες λεπτα λεπτο πριν μολις τελευταια τελευταιο
""".split())


def _content_words(phrase):
    words = re.findall(r'[\wά-ώΆ-Ώ]+', strip_accents(phrase))
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def _matches_keywords(event, keywords):
    haystack = strip_accents(f'{event.text} {event.who or ""} {event.room or ""}')
    return any(k in haystack for k in keywords)
