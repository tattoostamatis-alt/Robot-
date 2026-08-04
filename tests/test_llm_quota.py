"""Tests for the Gemini request counter (home_robot/llm_quota.py).

2026-08-04, from the owner: "να βλέπω πόσες μπορώ ακόμα μέχρι το όριο". The API
offers no remaining-quota header and no usage endpoint, so the number is
counted here — which means these tests are the only thing standing between the
owner and a confident wrong number.

What they police, in order of how badly it would mislead:

  * A 429 that carries BOTH violations must not record "your daily limit is 15".
  * A per-minute rejection must not declare the day over.
  * Two processes (llm_bridge and vision_node) share one bucket.
  * A file truncated by one of this machine's power cuts must not stop the LLM.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_llm_quota.py -q
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pytest

from home_robot import llm_quota

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def state_file(tmp_path, monkeypatch):
    path = str(tmp_path / 'gemini_usage.json')
    monkeypatch.setattr(llm_quota, 'STATE_FILE', path)
    return path


# ── counting ─────────────────────────────────────────────────────────────────

def test_a_fresh_day_starts_full():
    v = llm_quota.snapshot()
    assert v['used'] == 0
    assert v['left'] == v['limit'] == llm_quota.DEFAULT_RPD


def test_every_request_comes_off_the_total():
    llm_quota.record()
    llm_quota.record()
    v = llm_quota.record()
    assert v['used'] == 3
    assert v['left'] == llm_quota.DEFAULT_RPD - 3


def test_the_count_survives_a_restart(state_file):
    """`robot max` restarts the stack several times a day. The quota does not
    restart with it, so neither may the counter."""
    llm_quota.record(5)
    assert json.load(open(state_file))['count'] == 5
    assert llm_quota.snapshot()['used'] == 5


def test_the_two_nodes_share_one_bucket(state_file):
    """llm_bridge and vision_node are separate processes spending the same
    free-tier allowance. A counter held in memory would report half of it."""
    code = (f'import sys; sys.path.insert(0, {PKG!r});'
            'from home_robot import llm_quota;'
            f'llm_quota.STATE_FILE = {state_file!r};'
            'llm_quota.record(2)')
    for _ in range(3):
        subprocess.run([sys.executable, '-c', code], check=True)
    assert llm_quota.snapshot()['used'] == 6


def test_the_minute_window_rolls_off(monkeypatch):
    llm_quota.record(4)
    assert llm_quota.snapshot()['minute_used'] == 4
    later = time.time() + 61
    monkeypatch.setattr(llm_quota.time, 'time', lambda: later)
    v = llm_quota.snapshot()
    assert v['minute_used'] == 0
    assert v['used'] == 4, 'the minute window must not reset the day'


def test_the_day_rolls_over_at_pacific_midnight(state_file):
    """‼️ Not at local midnight. Google resets free-tier daily quota at
    midnight America/Los_Angeles, and rolling over at Athens midnight would
    show a full tank ten hours early — the one moment the number is trusted."""
    llm_quota.record(7)
    state = json.load(open(state_file))
    yesterday = (datetime.now(llm_quota.QUOTA_TZ) - timedelta(days=1))
    state['day'] = yesterday.strftime('%Y-%m-%d')
    json.dump(state, open(state_file, 'w'))
    assert llm_quota.snapshot()['used'] == 0


def test_the_reset_countdown_is_a_real_time():
    v = llm_quota.snapshot()
    assert 0 < v['resets_in'] <= 24 * 3600


# ── learning the real limit from a rejection ─────────────────────────────────

_DAILY_429 = """google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED.
{'error': {'code': 429, 'message': 'You exceeded your current quota.',
 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type':
 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric':
 'generativelanguage.googleapis.com/generate_content_free_tier_requests',
 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
 'quotaDimensions': {'model': 'gemini-flash-lite'}, 'quotaValue': '250'}]}]}}"""

_MINUTE_429 = _DAILY_429.replace('PerDayPerProject', 'PerMinutePerProject') \
                        .replace("'250'", "'15'")

_BOTH_429 = """429 RESOURCE_EXHAUSTED {'violations': [
 {'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier',
  'quotaValue': '15'},
 {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
  'quotaValue': '250'}]}"""


def test_an_ordinary_error_is_not_a_rate_limit():
    assert llm_quota.note_rate_limit(RuntimeError('connection reset')) is None
    assert not llm_quota.is_rate_limit(RuntimeError('500 INTERNAL'))


def test_the_daily_limit_is_believed_from_the_rejection():
    """The default is a guess; this is the number stated by the server."""
    v = llm_quota.note_rate_limit(_DAILY_429)
    assert v['limit'] == 250
    assert v['left'] == 0, 'a daily 429 means the day is over'
    assert v['source'] == 'measured'


def test_a_per_minute_rejection_does_not_end_the_day():
    """These are routine — three quick questions in a row will do it — and
    treating one as "out of quota" would send the owner to the NPU for the rest
    of the day for nothing."""
    llm_quota.record(10)
    v = llm_quota.note_rate_limit(_MINUTE_429)
    assert v['minute_limit'] == 15
    assert v['used'] == 10
    assert v['limit'] == llm_quota.DEFAULT_RPD
    assert v['left'] > 0


def test_both_violations_in_one_body_land_in_the_right_place():
    """‼️ The failure this guards: reading the first number in the body and
    recording a daily limit of 15."""
    v = llm_quota.note_rate_limit(_BOTH_429)
    assert v['limit'] == 250
    assert v['minute_limit'] == 15


def test_a_429_with_no_numbers_still_ends_the_day():
    v = llm_quota.note_rate_limit('429 RESOURCE_EXHAUSTED: quota exceeded per day')
    assert v['left'] == 0


def test_the_learned_limit_sticks_for_later_requests():
    llm_quota.note_rate_limit(_DAILY_429)
    llm_quota.set_limits(rpd=250)          # same limit, new day
    llm_quota._with_state(lambda s, now: s.update(count=0))
    v = llm_quota.record()
    assert v['limit'] == 250
    assert v['left'] == 249


# ── it must never be the reason the robot goes quiet ─────────────────────────

def test_a_truncated_file_does_not_raise(state_file):
    """‼️ This machine loses power mid-write often enough to have its own memory
    entry. A half-written JSON file must cost the count, not the conversation."""
    with open(state_file, 'w') as fh:
        fh.write('{"day": "2026-08-04", "coun')
    v = llm_quota.record()
    assert v['used'] == 1


def test_a_directory_where_the_file_should_be_does_not_raise(state_file):
    os.makedirs(state_file, exist_ok=True)
    v = llm_quota.record()
    assert v['used'] >= 0, 'a broken path must still return usable numbers'


def test_nonsense_contents_do_not_raise(state_file):
    with open(state_file, 'w') as fh:
        fh.write('[1, 2, 3]')
    assert llm_quota.record()['used'] == 1


# ── what it says out loud ────────────────────────────────────────────────────

def test_it_says_the_number_in_greek():
    llm_quota.record(3)
    said = llm_quota.describe()
    assert 'ερωτήσεις' in said
    assert str(llm_quota.DEFAULT_RPD - 3) in said
    assert 'περίπου' in said, 'an estimated limit must not sound certain'


def test_a_measured_limit_drops_the_hedge():
    llm_quota.set_limits(rpd=250, source='measured')
    assert 'περίπου' not in llm_quota.describe()


def test_running_out_says_when_it_comes_back():
    llm_quota.note_rate_limit(_DAILY_429)
    said = llm_quota.describe()
    assert 'όριο' in said
    assert 'ώρες' in said or 'λεπτά' in said
    assert ' 0 ' not in said, 'floor division said "σε 0 ώρες" in the last hour'


def test_the_last_hour_of_the_day_is_counted_in_minutes(monkeypatch):
    """Ten minutes before the reset is exactly when someone asks."""
    soon = datetime.now(llm_quota.QUOTA_TZ).replace(hour=23, minute=50)
    monkeypatch.setattr(llm_quota.time, 'time', lambda: soon.timestamp())
    llm_quota.note_rate_limit(_DAILY_429)
    assert 'λεπτά' in llm_quota.describe()


# ── wiring: nothing may reach Gemini uncounted ───────────────────────────────

_BRIDGE = open(f'{PKG}/home_robot/nodes/llm_bridge_node.py').read()
_VISION = open(f'{PKG}/home_robot/nodes/vision_node.py').read()
_DASH = open(f'{PKG}/home_robot/nodes/web_dashboard_node.py').read()


def test_every_gemini_request_goes_through_the_counted_door():
    """‼️ A second call site added later would make the number read low, and a
    low number is exactly the failure mode that matters: it says "keep asking"
    right up until the robot goes quiet mid-sentence.

    One spoken command is TWO requests — the tool call and the follow-up that
    phrases the answer — and both must be counted."""
    body = _BRIDGE.split('def _gemini_generate')[1]
    direct = body.count('_gemini_client.models.generate_content')
    assert direct == 1, 'a Gemini call bypasses _gemini_generate'
    assert _BRIDGE.count('self._gemini_generate(') >= 2, \
        'the follow-up call is not counted'


def test_the_bridge_publishes_the_count():
    assert "'/llm_quota'" in _BRIDGE
    assert 'TRANSIENT_LOCAL' in _BRIDGE.split("'/llm_quota'")[1][:200], \
        'not latched — a dashboard opened later would show a dash'


def test_running_out_is_spoken_not_only_logged():
    """Silence looks exactly like the STT being deaf, which has cost this
    project a debugging session before."""
    handler = _BRIDGE.split('def _handle_text_gemini')[1].split('\n    def ')[0]
    assert 'is_rate_limit' in handler
    assert 'llm_quota.describe()' in handler


def test_vision_spends_from_the_same_bucket():
    """"τι βλέπεις;" costs a request like anything else. A counter that only
    watched the conversation would read low all day."""
    assert 'llm_quota.record()' in _VISION
    assert 'llm_quota.note_rate_limit' in _VISION


def test_the_dashboard_shows_it():
    assert "'/llm_quota', self._cb_quota" in _DASH
    assert 'def _cb_quota' in _DASH
    assert 'quota(m){ onQuota(m); }' in _DASH


def test_the_counter_is_hidden_when_the_local_model_answers():
    """Qwen on the NPU has no quota. A leftover counter next to it would be a
    lie, and the switch between backends is one click away."""
    assert 'quotaVisibility' in _DASH
    fn = _DASH.split('function quotaVisibility')[1].split('function ')[0]
    assert "'lemonade'" in fn and "display = 'none'" in fn
