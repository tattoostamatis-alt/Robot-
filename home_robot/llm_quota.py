"""How many Gemini requests are left today, counted here because nobody tells us.

2026-08-04, from the owner: "να βλέπω πόσες μπορώ ακόμα μέχρι το όριο". There
is no answer to ask for — the Gemini API returns no remaining-quota header and
has no usage endpoint, so the only honest counter is one that counts every
request as it leaves the house. That means:

  * It is shared. Two nodes call Gemini — llm_bridge for conversation and
    vision_node for "τι βλέπεις;" — and both spend from the same free-tier
    bucket, so the count lives in a file under a lock, not in either process.
  * It survives restarts. `robot max` restarts the stack several times a day
    and the quota does not reset with it.
  * It counts REQUESTS, not sentences. One spoken command that triggers a tool
    is two requests: the call, then the follow-up that phrases the answer. That
    is what Google is counting, so it is what the robot reports.

‼️ The daily limit is a guess until the first 429 corrects it. The free-tier
number is not discoverable through the API and changes per model, so the
default here is the documented Flash-Lite free allowance and `note_rate_limit`
overwrites it with whatever the rejection actually says. A counter that lies
low is worse than no counter, so the number carries its source: 'default' until
a real 429 makes it 'measured'.

The day rolls over at midnight in America/Los_Angeles, which is where Google
resets free-tier daily quota — not at local midnight, which would show a full
tank nine hours early.
"""
import fcntl
import json
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

__all__ = ['record', 'snapshot', 'note_rate_limit', 'set_limits',
           'DEFAULT_RPD', 'DEFAULT_RPM', 'STATE_FILE']

STATE_FILE = os.path.expanduser('~/.home_robot/gemini_usage.json')

# Free tier, gemini-*-flash-lite. Treated as an estimate — see note_rate_limit.
DEFAULT_RPD = 1000
DEFAULT_RPM = 15

QUOTA_TZ = ZoneInfo('America/Los_Angeles')

# The quota values inside a 429. The SDK stringifies the error, so this reads
# the body as text rather than pretending to know its object shape. A real
# rejection lists one violation per limit that was hit:
#
#   'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
#   'quotaValue': '1000'
#
# ‼️ One 429 can carry BOTH the per-minute and the per-day violation, so the
# value has to be tied to the id next to it. Reading the first number in the
# body would happily record "your daily limit is 15".
_QUOTA_ID_RE = re.compile(r'quota[_ ]?[Ii]d["\']?\s*[:=]\s*["\']?([\w-]+)')
_QUOTA_VALUE_RE = re.compile(r'quota[_ ]?[Vv]alue["\']?\s*[:=]\s*["\']?(\d+)')
_PER_DAY_RE = re.compile(r'per[_ ]?day', re.I)
_PER_MINUTE_RE = re.compile(r'per[_ ]?minute', re.I)


def _violations(text):
    """[(scope, value)] out of a 429 body: scope is 'day', 'minute' or ''.

    Each value is attributed to the nearest quotaId BEFORE it, which is the
    order Google writes them in. A value with no id in front of it is reported
    with an empty scope and the caller decides what it can mean.
    """
    ids = [(m.start(), m.group(1)) for m in _QUOTA_ID_RE.finditer(text)]
    out = []
    for m in _QUOTA_VALUE_RE.finditer(text):
        prior = [name for pos, name in ids if pos < m.start()]
        name = prior[-1] if prior else ''
        scope = ('day' if _PER_DAY_RE.search(name)
                 else 'minute' if _PER_MINUTE_RE.search(name) else '')
        out.append((scope, int(m.group(1))))
    return out


def _quota_day(now=None):
    """Today's date in Google's reset timezone, as 'YYYY-MM-DD'."""
    stamp = datetime.fromtimestamp(now or time.time(), QUOTA_TZ)
    return stamp.strftime('%Y-%m-%d')


def _seconds_to_reset(now=None):
    now = now or time.time()
    stamp = datetime.fromtimestamp(now, QUOTA_TZ)
    midnight = (stamp + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return max(0.0, midnight.timestamp() - now)


def _blank(now):
    return {'day': _quota_day(now), 'count': 0, 'recent': [],
            'rpd': DEFAULT_RPD, 'rpm': DEFAULT_RPM, 'source': 'default',
            'last_error': ''}


def _roll(state, now):
    """Apply the two resets: the daily one and the rolling minute."""
    today = _quota_day(now)
    if state.get('day') != today:
        state['day'] = today
        state['count'] = 0
    state['recent'] = [t for t in state.get('recent', []) if now - t < 60.0]
    return state


def _with_state(mutate):
    """Read-modify-write under a lock, and never let a broken file stop a call.

    ‼️ Every failure path here returns a usable snapshot instead of raising.
    This runs on the request path of the LLM: a counter that cannot count is a
    cosmetic problem, and it must never be the reason the robot stops talking.
    """
    now = time.time()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    try:
        with open(STATE_FILE, 'a+') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                try:
                    state = json.loads(raw) if raw.strip() else _blank(now)
                except (ValueError, TypeError):
                    state = _blank(now)          # truncated by a power cut
                if not isinstance(state, dict):
                    state = _blank(now)
                _roll(state, now)
                mutate(state, now)
                fh.seek(0)
                fh.truncate()
                json.dump(state, fh)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        state = _blank(now)
        mutate(state, now)
    return _view(state, now)


def _view(state, now):
    """The shape everything downstream reads. Numbers, already subtracted."""
    rpd = int(state.get('rpd') or DEFAULT_RPD)
    rpm = int(state.get('rpm') or DEFAULT_RPM)
    used = int(state.get('count') or 0)
    minute = len(state.get('recent') or [])
    return {
        'used': used,
        'limit': rpd,
        'left': max(0, rpd - used),
        'minute_used': minute,
        'minute_limit': rpm,
        'minute_left': max(0, rpm - minute),
        'source': state.get('source', 'default'),
        'resets_in': int(_seconds_to_reset(now)),
        'day': state.get('day', _quota_day(now)),
        'last_error': state.get('last_error', ''),
    }


def record(n: int = 1) -> dict:
    """Count n requests that were just sent, and return the new numbers."""
    def _mutate(state, now):
        state['count'] = int(state.get('count') or 0) + n
        state.setdefault('recent', []).extend([now] * n)
    return _with_state(_mutate)


def snapshot() -> dict:
    """The current numbers without spending anything."""
    return _with_state(lambda state, now: None)


def set_limits(rpd=None, rpm=None, source=None) -> dict:
    """Point the counter at the limits this key actually has."""
    def _mutate(state, now):
        if rpd:
            state['rpd'] = int(rpd)
        if rpm:
            state['rpm'] = int(rpm)
        if source:
            state['source'] = source
    return _with_state(_mutate)


def is_rate_limit(err) -> bool:
    text = str(err)
    return '429' in text or 'RESOURCE_EXHAUSTED' in text.upper()


def note_rate_limit(err) -> dict | None:
    """Learn the real limit from a rejection. Returns the new numbers, or None.

    A 429 is the one moment the server states the quota out loud. Two things
    happen: the daily count is pinned to the limit (we are out — whatever the
    local count said, it was low), and the limit itself is believed from now
    on. Per-minute rejections are common and mean nothing about the day, so
    only a PerDay violation moves the daily number.
    """
    text = str(err)
    if not is_rate_limit(text):
        return None
    found = _violations(text)
    daily = next((v for scope, v in found if scope == 'day'), None)
    minute = next((v for scope, v in found if scope == 'minute'), None)
    day_hit = bool(daily) or bool(_PER_DAY_RE.search(text))

    def _mutate(state, now):
        state['last_error'] = text[:400]
        if minute:
            state['rpm'] = minute
        if daily:
            state['rpd'] = daily
            state['source'] = 'measured'
        if day_hit:
            # The day is over regardless of what the local count said — it can
            # only ever have been low (a crash between a request and its fsync
            # loses one; another machine on the same key loses all of them).
            state['count'] = max(int(state.get('count') or 0),
                                 int(state.get('rpd') or DEFAULT_RPD))
    return _with_state(_mutate)


def describe(view=None) -> str:
    """One Greek sentence, for speech and for the log."""
    v = view or snapshot()
    left, limit = v['left'], v['limit']
    about = '' if v['source'] == 'measured' else 'περίπου '
    if left <= 0:
        hours, minutes = divmod(int(v['resets_in']) // 60, 60)
        # «σε 0 ώρες» is what floor division gives in the last hour of the day,
        # which is the hour someone is most likely to be waiting it out.
        when = f'{hours} ώρες' if hours else f'{minutes} λεπτά'
        return ('Έφτασα το ημερήσιο όριο του Gemini. Ξεκλειδώνει σε '
                f'{when}.')
    return (f'Μου απομένουν {about}{left} ερωτήσεις από τις {limit} '
            f'σήμερα.')
