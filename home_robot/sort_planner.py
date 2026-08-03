"""Pure planning for sorting tasks — "μάζεψε τα παιχνίδια".

Sorting is fetch run repeatedly with a rule about where each thing belongs, so
this module is only the two parts fetch does not have: deciding WHICH objects
are in scope and WHERE each one goes, and keeping track across a run that may
take several minutes and will partly fail.

ROS-free on purpose (same reason as fetch_planner.py): the ordering, the rules
and the give-up logic are where the bugs live, and none of them need a robot to
test.

The rules live in config/sort_rules.yaml as `destination: [labels...]`, an
inversion of the obvious mapping. Written label-first, adding a room means
touching every label that belongs to it; written destination-first, a room is
one line and a label appears exactly once — and a label appearing twice is then
a visible duplicate rather than a silent last-one-wins.
"""
import math

# Objects further than this from the robot's current position are left for a
# later pass. Not a hard limit on the room — just an ordering that avoids
# crossing the house between two items that sit next to each other.
DEFAULT_MAX_REACH = 6.0

# How many times one object may fail to be picked before it is abandoned. Two
# is deliberate: one retry covers a bad grasp angle or a servo that did not
# settle, and beyond that the object is probably not graspable at all (too big,
# against a wall) and the robot would loop on it for ever while the rest of the
# floor stays untidied.
MAX_ATTEMPTS = 2


class SortRules:
    """label -> destination, built from the destination-first YAML."""

    def __init__(self, mapping=None, default=None):
        self._by_label = {}
        self.default = default
        self.duplicates = []
        for dest, labels in (mapping or {}).items():
            for label in labels or []:
                key = str(label).strip().lower()
                if not key:
                    continue
                if key in self._by_label and self._by_label[key] != dest:
                    # Kept and reported rather than silently overwritten: a
                    # label in two rooms is a config mistake the user must see.
                    self.duplicates.append((key, self._by_label[key], dest))
                    continue
                self._by_label[key] = dest

    @classmethod
    def from_yaml(cls, data):
        data = data or {}
        return cls(mapping=data.get('rules') or {},
                   default=data.get('default'))

    def destination(self, label):
        """Where this label belongs, or the default, or None."""
        return self._by_label.get(str(label).strip().lower(), self.default)

    def labels_for(self, destination):
        return sorted(k for k, v in self._by_label.items() if v == destination)

    @property
    def known_labels(self):
        return sorted(self._by_label)


def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def plan_order(objects, robot_xy, rules, *, group=None,
               max_reach=DEFAULT_MAX_REACH):
    """Which objects to pick, in which order.

    `objects` are object_memory entries: {'label', 'x', 'y', ...}. `group`
    optionally narrows to one destination ("μάζεψε τα παιχνίδια" = only the
    things that belong in the toy box).

    Nearest-first from where the robot stands. Not a travelling-salesman
    solution — greedy is within a metre or two indoors, and re-planning after
    every pick (which the mission does, because the robot has moved and the
    world may have changed) makes a globally optimal route meaningless anyway.

    Objects with no rule and no default are skipped: the robot must not invent
    a destination for something it has never been told about.
    """
    out = []
    for o in objects or []:
        label = (o.get('label') or '').strip().lower()
        if not label:
            continue
        dest = rules.destination(label)
        if dest is None:
            continue
        if group is not None and dest != group:
            continue
        try:
            xy = (float(o['x']), float(o['y']))
        except (KeyError, TypeError, ValueError):
            continue
        d = _dist(robot_xy, xy)
        if d > max_reach:
            continue
        out.append({'label': label, 'x': xy[0], 'y': xy[1],
                    'destination': dest, 'distance': round(d, 2)})
    # Tie-break on label so the order is deterministic for a given scene —
    # otherwise two objects at the same distance swap between runs and a test
    # (or a person watching) sees nondeterminism that is not real.
    out.sort(key=lambda e: (e['distance'], e['label']))
    return out


class SortProgress:
    """Tracks a run: what is done, what failed, what to try next.

    A sort takes minutes and WILL partly fail — an object rolls away, a grasp
    misses, someone walks through. The useful behaviour is to keep going and
    report honestly at the end, not to abort on the first failure or to loop
    for ever on the one thing that cannot be picked.
    """

    def __init__(self, max_attempts=MAX_ATTEMPTS):
        self.max_attempts = int(max_attempts)
        self._attempts = {}
        self.done = []
        self.failed = []

    @staticmethod
    def key(entry):
        # Position included: two cups in different rooms are two jobs. Rounded
        # to 10 cm so object_memory jitter does not turn one cup into three.
        return (entry['label'], round(entry['x'], 1), round(entry['y'], 1))

    def next_target(self, ordered):
        """First entry still worth attempting, or None when the run is over."""
        for entry in ordered:
            k = self.key(entry)
            if k in self.done or k in self.failed:
                continue
            if self._attempts.get(k, 0) >= self.max_attempts:
                continue
            return entry
        return None

    def record(self, entry, ok):
        k = self.key(entry)
        if ok:
            self.done.append(k)
            return
        self._attempts[k] = self._attempts.get(k, 0) + 1
        if self._attempts[k] >= self.max_attempts:
            self.failed.append(k)

    def summary_el(self):
        """One spoken Greek sentence. Never claims more than it did."""
        n_ok, n_bad = len(self.done), len(self.failed)
        if not n_ok and not n_bad:
            return 'Δεν βρήκα τίποτα να μαζέψω.'
        if not n_bad:
            return (f'Μάζεψα {n_ok} '
                    + ('αντικείμενο.' if n_ok == 1 else 'αντικείμενα.'))
        if not n_ok:
            return (f'Δεν κατάφερα να πιάσω {n_bad} '
                    + ('αντικείμενο.' if n_bad == 1 else 'αντικείμενα.'))
        return (f'Μάζεψα {n_ok}, αλλά {n_bad} '
                + ('δεν το κατάφερα.' if n_bad == 1 else 'δεν τα κατάφερα.'))
