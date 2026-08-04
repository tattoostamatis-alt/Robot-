"""Ask before driving off, because most of what the robot hears is not for it.

2026-08-04, from the owner: "το ρομπότ πάει συνέχεια βόλτες χωρίς να του πω".
The episodic timeline showed exactly how:

    18:59:32  heard (Speaker_5): «Λοιπόν, πάμε στο σαλόνι, πάμε στο διάδρομο.»
    18:59:43  said:              «Πηγαίνω στο σαλόνι και μετά στον διάδρομο.»

Nobody was talking to the robot. stt_node records for up to 12 s after a wake
word and returns whatever the room said in that window, so a family
conversation arrives as one utterance and the model quite reasonably finds an
instruction inside it. The same afternoon it also answered «Νερό, νερό θες?»,
«Αυτό!» and half a dozen other overheard fragments — harmless when the answer
is words, not harmless when the answer is a moving robot.

Talking back to something that was not addressed to you costs nothing. Driving
across the house does. So motion tools ask first, and a wrong guess simply gets
no answer and expires.

The confirmation is a keyword gate, not another model call, for the reason
[[stop_command]] documents: the model's own tool calls are the unreliable part.
Kept deliberately narrow — a bare "ναι"/"όχι" and the handful of ways people
actually say them. Anything else is treated as the conversation having moved
on, which cancels the pending action rather than guessing.
"""
import re

from home_robot.status_query import room_locative_el
from home_robot.stop_command import strip_accents
from home_robot.vocabulary import greek_accusative, greek_to

__all__ = ['MOTION_TOOLS', 'PENDING_TTL', 'is_affirmative', 'is_negative',
           'confirm_question']

# Tools that put the robot on the floor and move it. `stop` is NOT here: it is
# the one command that must never wait for a second sentence. Neither are the
# arm tools — the arm is in front of you and stays where the robot is.
#
# ‼️ The first version of this list held eight tools and missed five, which is
# the same bug one level up: `follow`, `fetch`, `check`, `goto_object` and
# `sort` all put the robot on the floor too, and they went through unasked. The
# overheard sentence does not have to say «πάμε στο σαλόνι» — «φέρε μου το
# ποτήρι» and «δες αν έκλεισα το παράθυρο» were just as unaddressed, and
# `follow` then chased whoever it saw for the next 30 s. Anything that ends in
# a Nav2 goal or a Twist belongs here; audit this list whenever a tool is added.
MOTION_TOOLS = frozenset({
    'goto', 'goto_pointed', 'dock', 'patrol', 'explore', 'tidy', 'find', 'move',
    'follow', 'fetch', 'check', 'goto_object', 'sort',
})

# How long a pending confirmation stays live. Long enough to answer a spoken
# question, short enough that a "ναι" to something else entirely — an hour
# later, in another conversation — cannot launch it.
PENDING_TTL = 25.0

# 'ναι' has to be a whole word: it is inside «ναι» but also inside plenty of
# other things once accents are stripped. 'εντάξει'/'οκ'/'πήγαινε'/'άντε'
# ('παμε' deliberately absent — «πάμε» is what the overheard sentence said).
_YES_RE = re.compile(
    r'\b(ναι|οκ|ok|ενταξει|βεβαι\w*|φυσικα|αμε|παρακαλω|'
    r'πηγαινε|προχωρα|αντε|ξεκινα|κανε το|καντο)\b')

# 'όχι', 'άσε', 'μη(ν)', 'ακύρωσε', 'σταμάτα' (also caught by the stop gate,
# but a "σταμάτα" answering a question should cancel it, not just brake).
_NO_RE = re.compile(
    r'\b(οχι|ασε|ασ|αστο|μη|μην|ακυρ\w*|σταματ\w*|αργοτερα|τιποτα)\b')


def is_affirmative(text: str) -> bool:
    """True if this reads as a yes to the question just asked."""
    n = strip_accents(text)
    return bool(_YES_RE.search(n) and not _NO_RE.search(n))


def is_negative(text: str) -> bool:
    n = strip_accents(text)
    return bool(_NO_RE.search(n))


# What the robot asks. Phrased so the answer is a plain yes or no — an
# open question ("τι να κάνω;") invites a sentence the gate cannot read.
_QUESTIONS = {
    'goto':         'Να πάω{where};',
    'goto_pointed': 'Να πάω εκεί που έδειξες;',
    'dock':         'Να πάω στη βάση;',
    'patrol':       'Να κάνω περιπολία στο σπίτι;',
    'explore':      'Να ξεκινήσω εξερεύνηση;',
    'tidy':         'Να ξεκινήσω τακτοποίηση{where};',
    'find':         'Να ψάξω {what};',
    'move':         'Να μετακινηθώ;',
    'follow':       'Να σε ακολουθήσω;',
    'fetch':        'Να πάω να φέρω {what};',
    'check':        'Να πάω{where} να δω;',
    'goto_object':  'Να πάω {what};',
    'sort':         'Να μαζέψω τα σκόρπια πράγματα{where};',
}


def confirm_question(tool: str, args: dict | None = None) -> str:
    """The yes/no question for a pending motion tool."""
    args = args or {}
    template = _QUESTIONS.get(tool, 'Να το κάνω;')
    where, what = '', ''
    if tool == 'goto':
        loc = args.get('location')
        # ‼️ Greek AND inflected. The first live run asked «Να πάω στο
        # saloni;» — the robot reading its own map keys out loud — and the
        # naive fix gave «στο διάδρομος». room_locative_el already carries the
        # article for exactly this reason.
        where = f' {room_locative_el(loc)}' if loc else ''
    elif tool == 'tidy':
        room = args.get('room')
        where = f' {room_locative_el(room)}' if room and room != 'all' else ''
    elif tool == 'find':
        # `find` is open vocabulary and the model passes the word the person
        # actually said, in Greek — so it is already speakable, unlike the two
        # below, which arrive as COCO labels.
        what = args.get('object') or 'κάτι'
    elif tool == 'check':
        room = args.get('room')
        where = f' {room_locative_el(room)}' if room else ''
    elif tool == 'sort':
        dest = args.get('destination')
        where = f' {room_locative_el(dest)}' if dest and dest != 'all' else ''
    elif tool == 'fetch':
        what = greek_accusative(args.get('object') or '') or 'κάτι'
    elif tool == 'goto_object':
        # 'chair' -> «Να πάω στην καρέκλα;». Without a target the question is
        # «Να πάω εκεί;», which is still answerable — the object is missing
        # from the tool call, not from the room.
        what = greek_to(args.get('object') or '') if args.get('object') else 'εκεί'
    return template.format(where=where, what=what)
