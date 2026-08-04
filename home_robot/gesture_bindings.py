"""Which gesture does what — the configurable part.

Both vocabularies (body poses in gesture_vocab.py, finger gestures in
hand_gestures.py) produce a name. This module maps a name to a robot command,
holds the defaults, and enforces the one rule that matters:

‼️ **Gestures that START motion are not equal to gestures that STOP it.**

A false positive on "stop" costs nothing — the robot was going to stop anyway
and you say it again. A false positive on "come here" sends a machine across
the room at a person who did not ask for it. So starting actions need a longer
hold and can be switched off wholesale, while stopping actions stay responsive.
This is why `is_safe` exists and why `hold_frames_for` returns different
numbers for the two classes.

Pure and persistable; the dashboard edits the mapping and the nodes read it.
"""

__all__ = ['ACTIONS', 'ACTION_EL', 'DEFAULT_BINDINGS', 'is_safe',
           'GestureBindings']

# What a gesture can be bound to. The value is the topic contract the node
# implements; keep it in step with gesture_node's _dispatch.
ACTIONS = (
    'none',            # explicitly bound to nothing
    'stop',            # halt motion
    'estop',           # latched emergency stop
    'goto_pointed',    # drive to the last pointed floor spot
    'follow',          # start following the person
    'stop_follow',
    'dock',            # return to the charger
    'patrol',
    'cancel',          # cancel the current mission/goal
    'say_hello',       # speak a greeting — harmless, good for testing
    'listen',          # start a voice turn without the wake word
)

ACTION_EL = {
    'none':         '— τίποτα —',
    'stop':         'Σταμάτα',
    'estop':        'ΣΤΟΠ ανάγκης',
    'goto_pointed': 'Πήγαινε εκεί που δείχνω',
    'follow':       'Ακολούθησέ με',
    'stop_follow':  'Σταμάτα να ακολουθείς',
    'dock':         'Πήγαινε στη βάση',
    'patrol':       'Περιπολία',
    'cancel':       'Ακύρωση αποστολής',
    'say_hello':    'Πες γεια',
    'listen':       'Άκουσέ με (χωρίς wake word)',
}

# Actions that only ever REDUCE what the robot is doing. A false positive here
# is an inconvenience; everything else can put a moving robot near a person.
_SAFE_ACTIONS = frozenset({'none', 'stop', 'estop', 'stop_follow', 'cancel',
                           'say_hello', 'listen'})

# Frames a gesture must be held. Deliberately asymmetric — see the module note.
HOLD_SAFE = 5
HOLD_MOTION = 12

DEFAULT_BINDINGS = {
    # Body poses — readable across a room.
    'hand_up':       'stop',
    'both_hands_up': 'estop',
    'arms_crossed':  'cancel',
    'point_floor':   'none',      # the pointing pane already handles this one
    'wave':          'say_hello',
    't_pose':        'none',
    # Finger gestures — for close up.
    'fist':          'stop',
    'open_palm':     'stop',
    'point':         'none',
    'victory':       'none',
    'three':         'none',
    'thumbs_up':     'goto_pointed',
    'thumbs_down':   'cancel',
    'ok':            'listen',
}


def is_safe(action):
    """True when a false positive on this action cannot start motion."""
    return action in _SAFE_ACTIONS


def hold_frames_for(action):
    """How long the gesture must be held before this action fires."""
    return HOLD_SAFE if is_safe(action) else HOLD_MOTION


class GestureBindings:
    """The gesture -> action mapping, with validation and persistence."""

    def __init__(self, mapping=None, motion_enabled=True):
        # ‼️ motion_enabled is the master switch for everything that can move
        # the robot. Off by default in the launch file: gesture-driven motion
        # should be something you turn on deliberately.
        self.motion_enabled = motion_enabled
        self._map = dict(DEFAULT_BINDINGS)
        if mapping:
            for gesture, action in mapping.items():
                self.set(gesture, action)

    def set(self, gesture, action):
        """Bind a gesture. Unknown actions are rejected rather than stored —
        a typo would otherwise silently disable the gesture."""
        if action not in ACTIONS:
            return False
        self._map[gesture] = action
        return True

    def action_for(self, gesture):
        """The action to run, or None when there is nothing to do.

        Returns None — not the action — when a motion action is bound but
        motion is disabled, so the caller never has to re-check the switch.
        """
        action = self._map.get(gesture, 'none')
        if action == 'none':
            return None
        if not is_safe(action) and not self.motion_enabled:
            return None
        return action

    def hold_for(self, gesture):
        return hold_frames_for(self._map.get(gesture, 'none'))

    def as_dict(self):
        return dict(self._map)

    def to_dict(self):
        return {'motion_enabled': self.motion_enabled, 'bindings': self._map}

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(mapping=data.get('bindings'),
                   motion_enabled=bool(data.get('motion_enabled', True)))
