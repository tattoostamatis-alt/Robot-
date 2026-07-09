"""Spatial object memory — the pure, ROS-free core behind object_memory_node.py.

Keeps a running map of *where things are* in the world frame: every time the
detector reports an object, we either fold the sighting into the nearest
existing instance of the same label (same cup seen again → update in place) or
start a new instance (a second cup elsewhere → its own entry). Positions are
smoothed with an exponential moving average so a jittery per-frame detection
settles onto a stable coordinate.

Deliberately dependency-free (no ROS, no numpy) so the bookkeeping can be
unit-tested without a robot or a running graph — see tests/test_object_memory.py.
The ROS node handles the camera→map TF, room naming, persistence and RAG push;
everything here is plain arithmetic on a list of dicts.
"""

import math
import time


class ObjectMemory:
    """A collection of remembered object instances in a single (world) frame.

    Parameters
    ----------
    merge_distance:
        A new sighting of the same label within this many metres of an existing
        instance updates that instance; farther away it becomes a new one.
    ema_alpha:
        Weight of each new sighting on the smoothed position (0..1). Higher =
        snappier/noisier, lower = smoother/laggier.
    min_conf:
        Sightings below this detector confidence are ignored.
    confirm_count:
        How many sightings an instance needs before it is *confirmed*. Below
        this it is "tentative": still tracked (so repeated sightings can confirm
        it) but hidden from confirmed-only queries/outputs, so a single spurious
        detection never pollutes the RAG store or a "where is X?" answer. This is
        the SORT/DeepSORT `n_init` track-confirmation idea. Default 1 keeps the
        old behaviour (every sighting confirmed immediately).
    clock:
        Monotonic-ish time source (seconds, float). Injectable for tests.
    """

    def __init__(self, merge_distance: float = 0.6, ema_alpha: float = 0.35,
                 min_conf: float = 0.5, confirm_count: int = 1, clock=time.time):
        self.merge_distance = float(merge_distance)
        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self.min_conf = float(min_conf)
        self.confirm_count = max(1, int(confirm_count))
        self._clock = clock
        self._instances: list[dict] = []
        self._next_id = 1

    def is_confirmed(self, inst: dict) -> bool:
        """True once an instance has been seen `confirm_count` times."""
        return inst.get('count', 0) >= self.confirm_count

    # ── ingest ────────────────────────────────────────────────────────────
    def observe(self, label: str, x: float, y: float, z: float,
                conf: float = 1.0, room: str | None = None,
                now: float | None = None) -> dict | None:
        """Record one world-frame sighting. Returns the touched instance, or
        None if it was dropped for low confidence."""
        if conf < self.min_conf:
            return None
        now = self._clock() if now is None else now

        inst = self._nearest(label, x, y)
        if inst is None:
            inst = {
                'id': self._next_id, 'label': label,
                'x': x, 'y': y, 'z': z,
                'conf': conf, 'room': room,
                'count': 1, 'first_seen': now, 'last_seen': now,
            }
            self._next_id += 1
            self._instances.append(inst)
            return inst

        # Confidence-weighted EMA: a low-confidence sighting pulls the stored
        # position less than a crisp one, so a marginal detection can't yank a
        # well-established instance across the room.
        a = self.ema_alpha * min(1.0, max(0.0, conf))
        inst['x'] = (1 - a) * inst['x'] + a * x
        inst['y'] = (1 - a) * inst['y'] + a * y
        inst['z'] = (1 - a) * inst['z'] + a * z
        inst['conf'] = max(inst['conf'], conf)
        inst['count'] += 1
        inst['last_seen'] = now
        if room is not None:
            inst['room'] = room
        return inst

    def _nearest(self, label: str, x: float, y: float) -> dict | None:
        best, best_d = None, self.merge_distance
        for inst in self._instances:
            if inst['label'] != label:
                continue
            d = math.hypot(inst['x'] - x, inst['y'] - y)
            if d <= best_d:
                best, best_d = inst, d
        return best

    # ── maintenance ───────────────────────────────────────────────────────
    def prune(self, max_age: float, now: float | None = None) -> int:
        """Drop instances unseen for longer than max_age seconds. Returns the
        number removed. max_age <= 0 disables pruning."""
        if max_age <= 0:
            return 0
        now = self._clock() if now is None else now
        before = len(self._instances)
        self._instances = [i for i in self._instances
                           if now - i['last_seen'] <= max_age]
        return before - len(self._instances)

    # ── query ─────────────────────────────────────────────────────────────
    def query(self, label: str, now: float | None = None,
              confirmed_only: bool = False) -> dict | None:
        """Best current guess for where `label` is: most recently seen instance
        of that label (ties broken by confidence). None if unknown.

        confirmed_only skips tentative (single-sighting) instances so a spurious
        detection is never returned as fact."""
        now = self._clock() if now is None else now
        matches = [i for i in self._instances if i['label'] == label
                   and (not confirmed_only or self.is_confirmed(i))]
        if not matches:
            return None
        return max(matches, key=lambda i: (i['last_seen'], i['conf']))

    def all(self, confirmed_only: bool = False) -> list[dict]:
        """All instances, most-recently-seen first. confirmed_only hides
        tentative instances still short of confirm_count sightings."""
        items = self._instances if not confirmed_only else \
            [i for i in self._instances if self.is_confirmed(i)]
        return sorted(items, key=lambda i: i['last_seen'], reverse=True)

    # ── persistence ───────────────────────────────────────────────────────
    def to_list(self) -> list[dict]:
        return [dict(i) for i in self._instances]

    def load_list(self, data: list[dict]) -> None:
        """Replace contents from a previously saved to_list()."""
        self._instances = [dict(i) for i in (data or [])]
        self._next_id = 1 + max((i.get('id', 0) for i in self._instances), default=0)
