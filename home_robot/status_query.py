"""Recognise "how are you / how much battery" questions and answer them from
real telemetry instead of letting the LLM improvise.

Same failure as the stop tool (see stop_command.py): qwen3vl-it:4b answers
status questions conversationally rather than calling `system_status`, and
because it has no data it invents a number. Measured on the same question,
hours apart: "Έχω 85% μπαταρία" and "Έχω 87% μπαταρία" — both fabricated, and
both while the Roomba was asleep and publishing no battery topic at all.

A wrong-but-confident battery reading is worse than no reading: it is the one
number you would act on before sending the robot across the house.

Unlike stop, the answer here needs data, so this module only decides *that*
the question was asked and how to phrase the result; the caller fetches the
telemetry and passes it to `format_status`.
"""
import re
import unicodedata


def _norm(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')


# Deliberately does NOT include 'φορτισ': "πήγαινε να φορτίσεις" is the dock
# command, not a status question, and must still reach the LLM's tool call.
_BATTERY_RE = re.compile(r'\bμπαταρι\w*')
_STATUS_RE = re.compile(
    r'\b(καταστασ\w*|θερμοκρασι\w*|uptime|cpu|επεξεργαστ\w*|'
    r'μνημ\w*|ram|δισκο\w*|φορτ[ιω]μεν\w*)')
# Movement verbs mean it is an instruction that happens to mention a resource
# ("πήγαινε να φορτίσεις", "πήγαινε στη βάση"), so leave those alone.
_COMMAND_RE = re.compile(r'\b(πηγαιν\w*|παμε|παω|τακτοποι\w*|καθαρ\w*|ακολουθ\w*)')


def is_status_query(text: str) -> bool:
    """True if the user is asking about the robot's own condition."""
    n = _norm(text)
    if _COMMAND_RE.search(n):
        return False
    return bool(_BATTERY_RE.search(n) or _STATUS_RE.search(n))


def wants_battery(text: str) -> bool:
    """True if the question is specifically about charge."""
    return bool(_BATTERY_RE.search(_norm(text)))


def format_status(status: dict, battery_only: bool) -> str:
    """One short Greek sentence from real telemetry.

    `status` is the dict produced by the system_status tool. Missing keys are
    reported as missing — never guessed.

    ‼️ The Roomba's battery reading is NEVER reported. The pack has been
    removed from this robot (2026-07-26) and it runs off a power bank, so
    battery_percent/charger_state are meaningless leftovers: the OI still
    publishes a figure and it is nonsense — 41% at 14.44 V, when 14.44 V is a
    full pack. Quoting it would be exactly the confidently-wrong number this
    whole module exists to prevent.
    """
    if battery_only:
        return ('Δεν έχω μπαταρία — δουλεύω με powerbank, οπότε δεν μετράω '
                'στάθμη.')

    bits = []
    if status.get('cpu_percent') is not None:
        bits.append(f'CPU {status["cpu_percent"]:.0f}%')
    if status.get('cpu_temp_c') is not None:
        bits.append(f'θερμοκρασία {status["cpu_temp_c"]:.0f} βαθμοί')
    if status.get('ram_percent') is not None:
        bits.append(f'μνήμη {status["ram_percent"]:.0f}%')
    if status.get('disk_percent') is not None:
        bits.append(f'δίσκος {status["disk_percent"]:.0f}%')
    if not bits:
        return 'Δεν μπορώ να διαβάσω την κατάστασή μου αυτή τη στιγμή.'
    return 'Είμαι εντάξει: ' + ', '.join(bits) + '.'
