"""Keyword recognition for the handful of motion/arm commands said the most —
run ahead of the LLM for latency, same reasoning as [[stop_command]].

2026-08-13, by owner request: unlike `goto`/`move`/`arm` called by the model,
a command matched here also skips [[motion_confirm]] entirely. That gate
exists because an overheard "πάμε στο σαλόνι" is not addressed to the robot;
these patterns are narrow and verb-anchored enough (goto needs a movement verb
*and* a known room, turn needs a turn verb *and* a direction, arm needs the
word "βραχίονα") that they are not the shape of sentence confirm_motion was
built to catch. Confirmation stays in place for anything the model itself
decides to call — this module only shortcuts the phrasings below.

Scope is deliberately narrow: bare goto/move/turn/arm, not `tidy`/`follow`/
`fetch`/`find`/`patrol`/`explore`. Those need open-vocabulary object names or
multi-step reasoning the model already handles reliably (see stop_command.py's
docstring — everything except `stop` was stable through the model), so gating
them here would only add parsing risk for no reliability gain. If a sentence
mentions one of those, this module deliberately backs off and lets the model
have it.
"""
import re

from home_robot import arm_motion
from home_robot.stop_command import strip_accents

__all__ = ['parse_simple_motion', 'parse_arm_command']

# Bare, accent-stripped stems that name a room in speech. Matched with \w* so
# declensions land on the same key: 'κουζίνα'/'κουζίνας'/'κουζίνες'. Kept in
# sync by hand with ROOM_NAMES_EL in status_query.py, same as every other
# Greek stem table in this codebase (HOUSEHOLD_EL, arm_motion.DIRECTIONS).
ROOM_STEMS = {
    'σαλον':   'saloni',
    'κουζιν':  'kouzina',
    'διαδρομ': 'diadromos',
    'τουαλετ': 'toualeta',
    'μπαμπα':  'domatio tou mbamba',   # "δωμάτιο του μπαμπά" — "μπαμπά" alone is distinctive
    'μαξ':     'domatio tou max',      # "δωμάτιο του Μαξ" — "Μαξ" alone is distinctive
}

# Negated or hypothetical mentions are conversation, not a command — same
# guard as stop_command.py's NEGATION_RE: "μη πας στην κουζίνα", "αν γυρίσεις
# αριστερά", "όταν πάω στο σαλόνι, ...".
_NEGATION_RE = re.compile(r'\b(μη|μην|γιατι|οταν|αν|θα)\b')

# Anything naming a different tool backs this module off entirely — a bare
# regex has no business guessing between "πήγαινε στην κουζίνα" (goto) and
# "πήγαινε στην κουζίνα και φέρε μου το ποτήρι" (goto + fetch, two tools).
_OTHER_TOOL_RE = re.compile(
    r'\b(φερ\w*|δεις|δω|κοιτα\w*|ψαξ\w*|βρες|μαζεψ\w*|τακτοποι\w*|'
    r'ακολουθ\w*|πιασ\w*|σηκωσ\w*|περιπολ\w*|εξερευν\w*|καθαρ\w*)\b')

# ‼️ Imperative only. 'παμε' ("let's go") and 'παω' ("I'm going", never a
# command) are deliberately excluded even though they read as goto verbs —
# 'παμε' is the literal trigger word from the overheard-conversation incident
# motion_confirm.py documents: «Λοιπόν, πάμε στο σαλόνι, πάμε στο διάδρομο.»,
# said between two people, not to the robot. Every other gate here still asks
# the model to run through motion_confirm's yes/no when the phrasing is this
# ambiguous; only the unambiguous imperative skips it.
# 'αντε' and 'γυρνα' added 2026-08-13 after testing more real phrasings —
# both are second-person imperative, same shape as 'πήγαινε', not the
# reported/narrated shape 'πάμε' is. 'έλα' ("come") was tested too and left
# out on purpose: it is used constantly as a bare conversational filler
# ("έλα ντε", "έλα τώρα") unrelated to movement, which 'άντε'/'γύρνα' are not.
_GOTO_VERB_RE = re.compile(r'\b(πηγαιν\w*|αντε|γυρνα\w*)\b')

_FORWARD_RE  = re.compile(r'\b(μπροστα|μπρος|εμπρος)\w*\b')
_BACKWARD_RE = re.compile(r'\b(πισω|οπισθεν)\w*\b')
_MOVE_VERB_RE = re.compile(r'\b(κανε|προχωρα\w*|κινησου)\b')

_TURN_VERB_RE = re.compile(r'\b(στριψ\w*|γυρνα\w*|στροφ\w*)\b')
_LEFT_RE  = re.compile(r'\bαριστερα\w*\b')
_RIGHT_RE = re.compile(r'\bδεξια\w*\b')

_ARM_WORD_RE = re.compile(r'\bβραχιον\w*\b')
# Deliberately its own (smaller) exclusion list, not _OTHER_TOOL_RE: 'σηκωσ'
# ("lift") is excluded there because "σήκωσε το ποτήρι" is a pick command,
# but "σήκωσε τον βραχίονα" is just this module's own up-direction phrased
# differently, and _ARM_WORD_RE already requires "βραχίονα" to be present —
# the collision _OTHER_TOOL_RE guards against for goto does not apply here.
# 'πιασ' ("grab") stays excluded: "πιάσε" is never itself a direction word,
# so keeping it out only blocks true pick sentences, at no cost to arm ones.
_ARM_OTHER_TOOL_RE = re.compile(
    r'\b(φερ\w*|δεις|δω|κοιτα\w*|ψαξ\w*|βρες|μαζεψ\w*|τακτοποι\w*|'
    r'ακολουθ\w*|πιασ\w*|περιπολ\w*|εξερευν\w*|καθαρ\w*)\b')

# Exact accent-stripped match, not a stem: these are adverbs ("πάνω", "λίγο")
# that do not inflect, so a truncated stem would only add false-positive risk
# (e.g. an "εξ" stem for "έξω" matching "εξερεύνηση") for no benefit.
_ARM_DIRECTION_STEMS = {strip_accents(d): d for d in arm_motion.DIRECTIONS}
_ARM_AMOUNT_STEMS = {strip_accents(a): a for a in arm_motion.AMOUNTS}

# Verbs that imply a direction on their own — "σήκωσε τον βραχίονα" ("lift
# the arm") never says the word "πάνω", but it means exactly that. Checked
# only when no explicit direction word matched.
_ARM_IMPLIED_DIRECTION = {
    'σηκωσ':          'πάνω',
    'κατεβασ':        'κάτω',
    'χαμηλωσ':        'κάτω',
}


def parse_simple_motion(text: str, known_rooms) -> tuple[str, dict] | None:
    """('goto', {'location': ...}) / ('move', {'direction': ..., 'duration': ...}) or None.

    `known_rooms` is the current map's room set (`rooms_from_locations`,
    status_query.py) — a room word for a location the loaded map does not
    have must fall through to the model, which can at least say so.
    """
    n = strip_accents(text)
    if _NEGATION_RE.search(n) or _OTHER_TOOL_RE.search(n):
        return None

    if _GOTO_VERB_RE.search(n):
        for stem, room in ROOM_STEMS.items():
            if room in known_rooms and re.search(rf'\b{stem}\w*\b', n):
                return ('goto', {'location': room})

    # A direction word alone is not enough — "τι είναι αυτό μπροστά μου;" is a
    # question, not a command. Either verb regex covers real phrasings: "κάνε
    # λίγο μπροστά" (_MOVE_VERB_RE) and "πήγαινε λίγο μπροστά" (_GOTO_VERB_RE,
    # already required above to reach here without a room match).
    has_move_verb = _MOVE_VERB_RE.search(n) or _GOTO_VERB_RE.search(n)
    if has_move_verb and _FORWARD_RE.search(n) and not _BACKWARD_RE.search(n):
        return ('move', {'direction': 'forward', 'duration': 1.0})
    if has_move_verb and _BACKWARD_RE.search(n) and not _FORWARD_RE.search(n):
        return ('move', {'direction': 'backward', 'duration': 1.0})

    # 2026-08-13: also accept goto/move verbs here, not just στριψε/γύρνα/
    # στροφή — "Πήγαινε αριστερά" and "Κάνε αριστερά" are at least as common
    # spoken as "Στρίψε αριστερά", and πήγαινε/κάνε/άντε are already trusted
    # imperatives above.
    if has_move_verb or _TURN_VERB_RE.search(n):
        left, right = _LEFT_RE.search(n), _RIGHT_RE.search(n)
        if left and not right:
            return ('move', {'direction': 'left', 'duration': 1.0})
        if right and not left:
            return ('move', {'direction': 'right', 'duration': 1.0})
        # No direction, or both — ambiguous, ask the model.

    return None


def parse_arm_command(text: str) -> tuple[str, dict] | None:
    """('arm', {'direction': ..., 'amount': ...}) or None.

    Requires the word "βραχίονα" so this never collides with
    parse_simple_motion's own "μπροστά"/"πίσω" — the direction words are the
    same, "κάνε τον βραχίονα μπροστά" and "πήγαινε μπροστά" only differ by
    that one word.
    """
    n = strip_accents(text)
    if not _ARM_WORD_RE.search(n) or _NEGATION_RE.search(n) or _ARM_OTHER_TOOL_RE.search(n):
        return None

    direction = amount = None
    for word, full in _ARM_DIRECTION_STEMS.items():
        if re.search(rf'\b{word}\b', n):
            direction = full
            break
    if direction is None:
        for stem, implied in _ARM_IMPLIED_DIRECTION.items():
            if re.search(rf'\b{stem}\w*\b', n):
                direction = implied
                break
    if direction is None:
        return None
    for word, full in _ARM_AMOUNT_STEMS.items():
        if re.search(rf'\b{word}\b', n):
            amount = full
            break
    return ('arm', {'direction': direction, 'amount': amount})
