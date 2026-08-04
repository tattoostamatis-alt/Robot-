"""Spoken directions for the arm: "κάνε τον βραχίονα λίγο μπροστά".

2026-08-04, from the owner: «μιλάω στο Gemini και του λέω κάνε τον βραχίονα
λίγο μπροστά και δεν κάνει τίποτα, ή κάνω κάτι λάθος;» — not a phrasing
problem. There was no arm tool. `pick` and `fetch` drive the whole
pick-and-place pipeline at a named object; nothing could simply move the arm.
The dashboard sliders and the PS5 stick could, and that was the entire list.

A person says "λίγο μπροστά", not "shoulder to -0.35 rad". So this maps
direction words onto joint deltas, and the caller adds them to the arm's
CURRENT pose and clamps — relative, because "λίγο μπροστά" only means anything
relative to where the arm already is.

‼️ Shoulder "up" is NEGATIVE. Measured on the hardware and written down in
[[project_robot_arm_joy]] after the PS5 stick moved the arm the wrong way; the
same sign trap applies here, and getting it backwards means the arm dives into
the chassis when someone asks for up.

The limits live in arm_driver (MECH_LIMITS, tightened per joint by the limit_*
parameters). This module deliberately does NOT copy them: a second set of
numbers is a second thing to keep in sync, and the driver clamps anyway.
"""

__all__ = ['DIRECTIONS', 'AMOUNTS', 'deltas_for', 'describe']

# How far one step goes, in radians. "λίγο" is the default because it is what
# people say when they are adjusting something they can see.
AMOUNTS = {
    'λίγο':      0.25,
    'κανονικά':  0.50,
    'πολύ':      0.90,
}
DEFAULT_AMOUNT = 'λίγο'

# direction -> {joint: multiplier}. Multipliers, not radians, so `amount`
# scales the whole gesture.
#
# forward/back move shoulder and elbow together: dropping the shoulder alone
# swings the gripper down as much as out, which does not read as "μπροστά" to
# anyone watching. Opening the elbow at the same time keeps the hand roughly
# level as it extends.
DIRECTIONS = {
    'μπροστά':  {'shoulder': +1.0, 'elbow': -0.6},
    'πίσω':     {'shoulder': -1.0, 'elbow': +0.6},
    'πάνω':     {'shoulder': -1.0},          # ‼️ negative is up
    'κάτω':     {'shoulder': +1.0},
    'αριστερά': {'base': +1.0},
    'δεξιά':    {'base': -1.0},
    'μέσα':     {'elbow': +1.0},             # fold up
    'έξω':      {'elbow': -1.0},             # extend
}


def deltas_for(direction, amount=None):
    """{joint: radians} for a spoken direction, or None if unknown."""
    mult = DIRECTIONS.get(direction)
    if mult is None:
        return None
    step = AMOUNTS.get(amount or DEFAULT_AMOUNT, AMOUNTS[DEFAULT_AMOUNT])
    return {joint: step * factor for joint, factor in mult.items()}


def describe(direction, amount=None):
    """What the robot says back, in the words it was asked in."""
    return f'{amount or DEFAULT_AMOUNT} {direction}'
