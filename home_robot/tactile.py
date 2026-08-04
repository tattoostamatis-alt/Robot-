"""Feeling with the arm — contact, hardness and weight from joint load.

The RoArm-M3's feedback carries a load figure per joint (tB, tS, tE, tT, tR,
tG) alongside the angles. Almost nothing reads it, so the arm grabs blind and
hopes. These are the three things that load can actually tell you, and they are
each a different measurement:

  * **Contact** is a STEP. Load sits at a baseline while moving through air and
    jumps when something resists. Detecting the jump needs a baseline, which is
    why this is stateful rather than a threshold on the raw number — the
    baseline drifts with pose, since holding the arm out horizontally loads the
    shoulder far more than holding it down.
  * **Hardness** is the SLOPE of load against travel. Press in a millimetre: a
    table spikes, a cushion barely moves. This needs both numbers, which is why
    it takes a short probing motion and not a single reading.
  * **Weight** is the load the SHOULDER and elbow carry once the object is
    lifted and still. Reading it while moving measures the acceleration too.

Pure, so all three can be tested from sequences of numbers.

‼️ These are raw servo units, not newtons. Nothing here is calibrated against a
known mass, so `weight_estimate` returns a comparable NUMBER, not grams, and the
labels are relative — "heavier than the last thing" is honest, "180 g" would not
be. Calibrating would need a set of known weights and a session with the arm.
"""

import math

__all__ = ['LOAD_KEYS', 'ContactDetector', 'hardness', 'weight_estimate',
           'describe_touch']

# Feedback keys, confirmed against the live arm 2026-08-04: base, shoulder,
# elbow, tilt/wrist, roll, gripper.
LOAD_KEYS = ('tB', 'tS', 'tE', 'tT', 'tR', 'tG')

# Joints that carry an object's weight. The base only spins and the gripper
# only pinches, so neither says anything about how heavy the thing is.
_LIFT_KEYS = ('tS', 'tE')


def total_load(feedback, keys=LOAD_KEYS):
    """Sum of |load| over the given joints. Missing keys count as zero."""
    if not feedback:
        return 0.0
    return sum(abs(float(feedback.get(k, 0.0) or 0.0)) for k in keys)


class ContactDetector:
    """Spots the load STEP that means the arm has touched something.

    Keeps a running baseline of "load while moving freely" and fires when the
    reading rises above it by more than `threshold` for `confirm` samples in a
    row. The baseline updates only while NOT in contact — otherwise pressing on
    a table for a few seconds would teach it that the table is normal.
    """

    def __init__(self, threshold=45.0, confirm=3, alpha=0.15):
        self.threshold = threshold
        self.confirm = confirm
        self.alpha = alpha
        self._baseline = None
        self._hits = 0
        self._contact = False
        self._peak = 0.0

    def update(self, feedback):
        """Feed one feedback frame. Returns True on the sample contact begins."""
        load = total_load(feedback)
        if self._baseline is None:
            self._baseline = load
            return False

        excess = load - self._baseline
        if excess > self.threshold:
            self._hits += 1
            self._peak = max(self._peak, excess)
            if self._hits >= self.confirm and not self._contact:
                self._contact = True
                return True
        else:
            self._hits = 0
            if self._contact:
                self._contact = False
                self._peak = 0.0
            # ‼️ Only learn the baseline when free. Updating it during contact
            # would let a steady press become the new "normal" and the arm
            # would stop noticing it.
            self._baseline += self.alpha * (load - self._baseline)
        return False

    @property
    def in_contact(self):
        return self._contact

    @property
    def baseline(self):
        return self._baseline

    @property
    def excess(self):
        """How far above the free-moving baseline the load is peaking."""
        return round(self._peak, 1)

    def reset(self):
        self._baseline = None
        self._hits = 0
        self._contact = False
        self._peak = 0.0


def hardness(samples):
    """Load per millimetre of travel, from [(travel_mm, load), ...].

    A table spikes over a millimetre; a cushion barely moves. Returns None when
    the probe did not actually travel — dividing by a travel of zero would
    report everything as infinitely hard, which is exactly the wrong failure.
    """
    pts = [(float(d), float(l)) for d, l in (samples or [])]
    if len(pts) < 2:
        return None
    span = max(d for d, _ in pts) - min(d for d, _ in pts)
    if span < 0.5:
        return None
    # Least squares slope; more robust than first-to-last on noisy servo data.
    n = len(pts)
    mx = sum(d for d, _ in pts) / n
    my = sum(l for _, l in pts) / n
    num = sum((d - mx) * (l - my) for d, l in pts)
    den = sum((d - mx) ** 2 for d, _ in pts)
    if den < 1e-9:
        return None
    return num / den


def describe_hardness(slope):
    """Greek label for a hardness slope. Relative, not absolute — see module."""
    if slope is None:
        return 'άγνωστο'
    if slope < 8:
        return 'μαλακό'
    if slope < 30:
        return 'μέτριο'
    return 'σκληρό'


def weight_estimate(held_feedback, empty_feedback):
    """How much more the lifting joints carry with the object than without.

    Both frames must be from the SAME pose and both must be still: the arm
    holding a cup out at arm's length loads the shoulder far more than holding
    it close, and moving adds acceleration to the reading. Returns a comparable
    number in servo units, never grams.
    """
    if not held_feedback or not empty_feedback:
        return None
    return round(total_load(held_feedback, _LIFT_KEYS)
                 - total_load(empty_feedback, _LIFT_KEYS), 1)


def describe_weight(value):
    if value is None:
        return 'άγνωστο'
    if value < 15:
        return 'πολύ ελαφρύ'
    if value < 60:
        return 'ελαφρύ'
    if value < 150:
        return 'μέτριο'
    return 'βαρύ'


def describe_touch(slope=None, weight=None):
    """One sentence about what the arm just felt."""
    parts = []
    if slope is not None:
        parts.append(describe_hardness(slope))
    if weight is not None:
        parts.append(describe_weight(weight))
    if not parts:
        return 'Δεν κατάλαβα τι άγγιξα.'
    return 'Το άγγιξα: ' + ', '.join(parts) + '.'
