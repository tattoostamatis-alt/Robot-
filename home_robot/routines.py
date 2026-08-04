"""Learning the household's rhythm, and noticing when it breaks.

The episodic timeline records what happened and when. That makes it an archive,
not intelligence — it can answer "τι έγινε χθες" but never "κάτι είναι
παράξενο σήμερα". This module is the layer that turns one into the other.

A routine is a thing that usually happens, in a usual time window, on usual
days: the kitchen is used around 08:00 on weekdays, Max comes home near 18:00.
Once that is learned, its ABSENCE becomes information.

Pure and clock-injectable, because every interesting case here is about time
and none of them are convenient to reproduce on a real robot.

The three judgements that make this usable rather than annoying:

  * **Circular time.** 23:50 and 00:10 are twenty minutes apart, not
    twenty-three hours. Averaging clock times linearly puts the mean of those
    two at noon, which would make every late-night routine nonsense. Times are
    therefore averaged as unit vectors on a circle.
  * **A routine must be earned.** Something seen twice is a coincidence. The
    default needs it on most of the days it could have happened, over at least
    a week, before its absence is worth a word.
  * **Absence is only interesting once the window has passed.** Saying "Max is
    not home" at 17:00 when he usually arrives at 18:00 is noise; saying it at
    19:30 is the point.
"""

import math
import time

__all__ = ['Routine', 'RoutineLearner', 'circular_mean_hours',
           'hours_between', 'DAY_NAMES_EL']

_DAY = 86400.0
_HOUR = 3600.0

DAY_NAMES_EL = ('Δευτέρα', 'Τρίτη', 'Τετάρτη', 'Πέμπτη', 'Παρασκευή',
                'Σάββατο', 'Κυριακή')


def hours_between(a, b):
    """Shortest distance between two clock hours, in hours (0..12).

    ‼️ Circular. 23.9 and 0.1 are 0.2 apart, not 23.8.
    """
    d = abs((a - b) % 24.0)
    return min(d, 24.0 - d)


def circular_mean_hours(hours):
    """Mean clock time of a list of hours, respecting the wrap at midnight.

    Returns (mean_hour, spread_hours) where spread is a circular standard
    deviation — small when the events cluster, large when they are scattered
    across the day. A scattered "routine" is not one, and the caller uses the
    spread to reject it.
    """
    if not hours:
        return 0.0, 12.0
    angles = [h * math.pi / 12.0 for h in hours]           # 24h -> 2pi
    sx = sum(math.sin(a) for a in angles) / len(angles)
    cx = sum(math.cos(a) for a in angles) / len(angles)
    mean = math.atan2(sx, cx) * 12.0 / math.pi % 24.0
    r = math.hypot(sx, cx)                                 # 1 = perfectly tight
    if r >= 1.0:
        return mean, 0.0
    # Circular standard deviation, converted from radians to hours.
    spread = math.sqrt(-2.0 * math.log(r)) * 12.0 / math.pi
    return mean, min(spread, 12.0)


class Routine:
    """One learned habit: a key that happens around a time, on certain days."""

    __slots__ = ('key', 'hours', 'days', 'dates', 'last_ts')

    def __init__(self, key):
        self.key = key
        self.hours = []          # clock hour of each occurrence
        self.days = set()        # weekday indexes seen
        self.dates = set()       # 'YYYY-MM-DD' of each occurrence, deduped
        self.last_ts = 0.0

    def observe(self, ts):
        lt = time.localtime(ts)
        date = time.strftime('%Y-%m-%d', lt)
        if date in self.dates:
            # Once per day: a person walking in and out of the kitchen ten times
            # is one "kitchen was used today", not ten pieces of evidence.
            self.last_ts = max(self.last_ts, ts)
            return False
        self.dates.add(date)
        self.days.add(lt.tm_wday)
        self.hours.append(lt.tm_hour + lt.tm_min / 60.0)
        self.last_ts = max(self.last_ts, ts)
        return True

    @property
    def occurrences(self):
        return len(self.dates)

    def typical(self):
        """(mean_hour, spread_hours) of when this usually happens."""
        return circular_mean_hours(self.hours)

    def to_dict(self):
        return {'key': self.key, 'hours': self.hours,
                'days': sorted(self.days), 'dates': sorted(self.dates),
                'last_ts': self.last_ts}

    @classmethod
    def from_dict(cls, d):
        r = cls(d.get('key', ''))
        r.hours = list(d.get('hours', []))
        r.days = set(d.get('days', []))
        r.dates = set(d.get('dates', []))
        r.last_ts = d.get('last_ts', 0.0)
        return r


class RoutineLearner:
    """Learns routines from timestamped event keys and reports what is missing.

    ``min_occurrences`` days must have carried the event, spread over at least
    ``min_span_days``, and the times must cluster inside ``max_spread_hours``,
    before it counts as a routine at all.
    """

    def __init__(self, min_occurrences=4, min_span_days=7,
                 max_spread_hours=2.5, overdue_hours=1.5, max_age_days=60):
        self.min_occurrences = min_occurrences
        self.min_span_days = min_span_days
        self.max_spread_hours = max_spread_hours
        self.overdue_hours = overdue_hours
        self.max_age_days = max_age_days
        self._routines = {}

    # ── learning ──────────────────────────────────────────────────────────────

    def observe(self, key, ts=None):
        ts = time.time() if ts is None else ts
        r = self._routines.get(key)
        if r is None:
            r = self._routines[key] = Routine(key)
        return r.observe(ts)

    def _span_days(self, routine):
        if len(routine.dates) < 2:
            return 0.0
        days = sorted(routine.dates)
        first = time.mktime(time.strptime(days[0], '%Y-%m-%d'))
        last = time.mktime(time.strptime(days[-1], '%Y-%m-%d'))
        return (last - first) / _DAY

    def is_established(self, key):
        r = self._routines.get(key)
        if r is None or r.occurrences < self.min_occurrences:
            return False
        if self._span_days(r) < self.min_span_days:
            # Four times in one afternoon is not a daily habit.
            return False
        _, spread = r.typical()
        return spread <= self.max_spread_hours

    def established(self):
        return [k for k in self._routines if self.is_established(k)]

    def describe(self, key):
        """(mean_hour, spread, occurrences, weekdays) for a learned routine."""
        r = self._routines.get(key)
        if r is None:
            return None
        mean, spread = r.typical()
        return mean, spread, r.occurrences, sorted(r.days)

    # ── noticing ──────────────────────────────────────────────────────────────

    def overdue(self, now=None):
        """Established routines that should have happened today and have not.

        Returns [(key, expected_hour, hours_late)], most overdue first. Only
        routines whose window has clearly passed are reported: a habit at 18:00
        is not "missing" at 17:00.
        """
        now = time.time() if now is None else now
        lt = time.localtime(now)
        today = time.strftime('%Y-%m-%d', lt)
        now_h = lt.tm_hour + lt.tm_min / 60.0

        out = []
        for key in self.established():
            r = self._routines[key]
            if lt.tm_wday not in r.days:
                continue                      # not a day this usually happens
            if today in r.dates:
                continue                      # already happened
            mean, spread = r.typical()
            # Late by more than the window plus the grace period.
            late = now_h - (mean + max(spread, 0.5) + self.overdue_hours)
            # Only within the same day: before the usual time, and after
            # midnight-ish rollover, there is nothing to say.
            if late > 0 and now_h >= mean:
                out.append((key, mean, late))
        out.sort(key=lambda kv: -kv[2])
        return out

    def is_unusual_time(self, key, ts=None):
        """True when an established routine fires far from its usual hour."""
        ts = time.time() if ts is None else ts
        if not self.is_established(key):
            return False
        mean, spread = self._routines[key].typical()
        lt = time.localtime(ts)
        hour = lt.tm_hour + lt.tm_min / 60.0
        return hours_between(hour, mean) > max(spread * 2.0, 2.0)

    # ── housekeeping ──────────────────────────────────────────────────────────

    def prune(self, now=None):
        """Forget occurrences older than max_age_days, and empty routines.

        A household's rhythm changes — school terms, jobs, seasons. A routine
        learned in March must not still be asserted in July.
        """
        now = time.time() if now is None else now
        cutoff = time.strftime('%Y-%m-%d',
                               time.localtime(now - self.max_age_days * _DAY))
        for key, r in list(self._routines.items()):
            keep = {d for d in r.dates if d >= cutoff}
            if len(keep) != len(r.dates):
                dropped = sorted(r.dates - keep)
                r.dates = keep
                # hours/days are positional only in aggregate; rebuild what we
                # can by dropping the oldest entries in the same proportion.
                r.hours = r.hours[len(dropped):]
                if not keep:
                    del self._routines[key]

    # ── persistence ───────────────────────────────────────────────────────────

    def to_dict(self):
        return {'min_occurrences': self.min_occurrences,
                'min_span_days': self.min_span_days,
                'max_spread_hours': self.max_spread_hours,
                'overdue_hours': self.overdue_hours,
                'routines': [r.to_dict() for r in self._routines.values()]}

    @classmethod
    def from_dict(cls, data):
        obj = cls(min_occurrences=data.get('min_occurrences', 4),
                  min_span_days=data.get('min_span_days', 7),
                  max_spread_hours=data.get('max_spread_hours', 2.5),
                  overdue_hours=data.get('overdue_hours', 1.5))
        for d in data.get('routines', []) or []:
            r = Routine.from_dict(d)
            if r.key:
                obj._routines[r.key] = r
        return obj
