"""Keyword recognition for the spoken emergency stop.

Why this exists instead of just letting the LLM call its `stop` tool:
measured 2026-07-26 against qwen3vl-it:4b (FastFlowLM/NPU) with the full
15-tool schema at temperature 0.1, deterministic across repeats —

    "Σταμάτα."              → stop tool          ✓
    "Σταμάτησε."            → stop tool          ✓
    "Σταμάτα!"              → text "Σταμάτησα."  ✗
    "Μαξ, σταμάτα!"         → text "Σταμάτησα."  ✗
    "Στοπ!"                 → text "Σταμάτησα."  ✗
    "Σταμάτα σε παρακαλώ."  → text "Σταμάτησα."  ✗
    "Έι ρομπότ, σταμάτα."   → text "Σταμάτησα."  ✗

Roughly two thirds of natural phrasings answered with a *false* confirmation
while the robot kept driving. Every other tool (goto/dock/tidy/follow/
system_status) was stable under the same prefixes, so this is specific to
`stop` — and `stop` is the one that must not be probabilistic.

No other layer covers it: the joystick e-stop is a physical button, and
mission_executor's CANCEL_PHRASES additionally require the word 'αποστολ'.
"""
import re
import unicodedata


def strip_accents(text: str) -> str:
    """Lowercase and drop Greek accents so keyword matching is robust."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text.lower())
        if unicodedata.category(c) != 'Mn'
    )


# 'σταματ' covers σταμάτα/σταμάτησε/σταματήστε/σταματάς; 'ακινητ' covers
# ακίνητος/ακίνητη. 'στοπ' is what the STT writes for a spoken English "stop".
STOP_RE = re.compile(r'\b(σταματ\w*|στοπ|ακινητ\w*|αλτ)\b')

# Negated or hypothetical mentions are conversation, not a command:
# "μη σταματάς", "γιατί σταμάτησες;", "όταν σταματήσεις, γύρνα".
NEGATION_RE = re.compile(r'\b(μη|μην|γιατι|οταν|αν|θα)\b')


def is_stop_command(text: str) -> bool:
    """True if the utterance is a direct order to halt right now."""
    normalized = strip_accents(text)
    return bool(STOP_RE.search(normalized) and not NEGATION_RE.search(normalized))
