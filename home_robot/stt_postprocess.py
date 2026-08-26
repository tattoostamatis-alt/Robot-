"""Clean prompt leakage out of Whisper transcriptions.

faster-whisper is given a domain `initial_prompt` (room names, "πήγαινε
στο…") because it markedly improves Greek command accuracy. The cost is that
on a long buffer with little clear speech the decoder falls back on that
prompt and emits it as if it were heard. Measured on real mic audio
2026-07-26 — every one of these came out of `stt_node` as a "Heard:" line:

    "Εντολές προς το ρομπότ Μαξ."                        (7×, pure echo)
    "Μαξ, πήγαινε στην κουζίνα, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ."
    "Πήγαινε στη βάση, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ."
    "Στον δωμάτιο του Μαξ, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ."

The first is meta-text the user can never have said. The others are a real
command with prompt fragments stapled on — which is worse than noise, because
"πήγαινε στην κουζίνα, στο δωμάτιο του Μαξ" is a *plausible* instruction and
the LLM will happily act on the wrong half of it.

Note: this could NOT be reproduced offline. Silence, shaped noise, and a 12 s
buffer at the same RMS (0.038) the node logged all decoded cleanly, with and
without condition_on_previous_text. So the trigger is something about the live
XVF3800 capture, and this module is a net rather than a root-cause fix — the
repetition signature is what it keys on, not the cause.
"""
import re
import unicodedata

# Meta-text from the head of the initial_prompt. A user talking to the robot
# never utters this, so any transcription containing it is discarded whole.
META_MARKERS = (
    'εντολες προς το ρομποτ',
    # Stock Whisper hallucinations on quiet/noisy Greek recordings.  These
    # are video-credit phrases, never commands addressed to the robot.
    'υποτιτλοι authorwave',
    'ευχαριστουμε που παρακολουθησατε',
)

# Prompt fragments that may legitimately appear ONCE (they are real commands)
# but signal leakage when the decoder loops on them.
PROMPT_FRAGMENTS = (
    'στην κουζινα', 'στο σαλονι', 'στον διαδρομο', 'στην τουαλετα',
    'στο δωματιο του μαξ', 'στο δωματιο του μπαμπα', 'στη βαση',
)


def _norm(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w\s]', '', s).strip()


def is_babble(text: str) -> bool:
    """True for a transcription that is a loop of repeated fragments.

    Seen live 2026-07-30, straight out of stt_node after a wake:

        "Ροδικρον, μάλταν, νου, μάλτα, δαμμου, ροδικοσύγμα, τάο, δίτρο,
         σύγμα, ροδικρον, τίτρα, ροδικρον, νου, δίτρο, σύγμα, καπέ, λαμβά,
         εξελον, ροδικοσύγμα, τάο, ροδικοσύγμα."

    Whisper decoding room noise into mangled Greek letter names — nothing the
    user said, but the LLM receives it as a command and answers. The prompt
    filter below cannot catch it: none of these are prompt fragments. What
    gives it away is structure rather than vocabulary — many short comma-
    separated tokens, most of them repeats. Real speech does not repeat a third
    of its words.
    """
    tokens = [_norm(c) for c in re.split(r'\s*,\s*', text.strip()) if c.strip()]
    if len(tokens) < 8:
        return False
    unique = len(set(tokens))
    if unique / len(tokens) > 0.75:          # varied enough to be real speech
        return False
    return all(len(t.split()) <= 2 for t in tokens)


def clean(text: str) -> str:
    """Return the usable part of a transcription, or '' if it is all leakage.

    Keeps the leading clauses (what the user actually said) and drops the
    trailing prompt echo. A clause repeated two or more times is the tell: a
    person says "πήγαινε στην κουζίνα" once, not three times.
    """
    if not text or not text.strip():
        return ''

    if any(m in _norm(text) for m in META_MARKERS):
        return ''

    if is_babble(text):
        return ''

    clauses = [c for c in re.split(r'\s*,\s*', text.strip()) if c.strip()]
    if len(clauses) < 2:
        return text.strip()

    seen: dict[str, int] = {}
    for c in clauses:
        n = _norm(c)
        seen[n] = seen.get(n, 0) + 1

    kept = []
    for c in clauses:
        n = _norm(c)
        # Cut at the first clause that both repeats and belongs to the prompt.
        if seen[n] >= 2 and any(f in n for f in PROMPT_FRAGMENTS):
            break
        # Plain stutter of the immediately preceding clause.
        if kept and _norm(kept[-1]) == n:
            continue
        kept.append(c)

    if not kept:
        return ''
    out = ', '.join(kept).strip()
    if not out.endswith(('.', '!', ';', '?')) and text.rstrip().endswith(('.', '!', ';', '?')):
        out += text.rstrip()[-1]
    return out
