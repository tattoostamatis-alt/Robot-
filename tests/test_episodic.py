"""Unit tests for episodic memory (home_robot/episodic.py)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.episodic import (  # noqa: E402
    EpisodicLog, Event, parse_time_window, strip_accents,
)

HOUR = 3600.0
DAY = 86400.0


def _at(hour, minute=0, day_offset=0):
    """Epoch seconds for a local wall-clock time, relative to a fixed noon."""
    base = time.mktime((2026, 8, 4, 0, 0, 0, 0, 0, -1))
    return base + day_offset * DAY + hour * HOUR + minute * 60


NOW = _at(14, 30)          # 2026-08-04, 14:30 local


# ── strip_accents ─────────────────────────────────────────────────────────────

def test_strips_greek_accents_and_lowercases():
    assert strip_accents('ΤΩΡΑ') == 'τωρα'
    assert strip_accents('τώρα') == 'τωρα'
    assert strip_accents('Χθές Το Βράδυ') == 'χθες το βραδυ'


def test_strip_accents_handles_empty():
    assert strip_accents('') == ''
    assert strip_accents(None) == ''


# ── parse_time_window ─────────────────────────────────────────────────────────

def test_today_covers_the_calendar_day():
    start, end = parse_time_window('τι έγινε σήμερα', now=NOW)
    assert start <= _at(0, 1) < end
    assert start <= NOW < end


def test_yesterday_is_the_previous_day():
    start, end = parse_time_window('τι έγινε χθες', now=NOW)
    assert start == _at(0, 0, day_offset=-1)
    assert end == _at(0, 0)


def test_yesterday_spelled_chtes():
    # STT produces both spellings.
    assert parse_time_window('χτες', now=NOW) == parse_time_window('χθες', now=NOW)


def test_day_before_yesterday_is_not_swallowed_by_yesterday():
    # 'προχθες' contains 'χθες' as a substring — the ordering must handle it.
    start, _ = parse_time_window('προχθές', now=NOW)
    assert start == _at(0, 0, day_offset=-2)


def test_this_morning_narrows_to_the_morning():
    start, end = parse_time_window('σήμερα το πρωί', now=NOW)
    assert start == _at(5)
    assert end == _at(12)


def test_yesterday_evening_narrows_within_yesterday():
    start, end = parse_time_window('χθες το βράδυ', now=NOW)
    assert start == _at(19, 0, day_offset=-1)
    assert end == _at(24, 0, day_offset=-1)


def test_bare_part_of_day_means_today():
    start, end = parse_time_window('το πρωί', now=NOW)
    assert start == _at(5)
    assert end == _at(12)


def test_a_part_of_day_still_ahead_means_the_last_one_that_happened():
    # At 14:30 "το βράδυ" cannot mean tonight — nothing has happened yet. It
    # means last night, not an inverted (and therefore empty) window.
    start, end = parse_time_window('το βράδυ', now=NOW)
    assert start < end <= NOW
    assert start == _at(19, 0, day_offset=-1)


def test_a_part_of_day_in_progress_is_clipped_to_now():
    # At 14:30, "το απόγευμα" (16-19) has not started either, but "το μεσημέρι"
    # (12-16) is underway: keep today's, ending at now rather than in the future.
    start, end = parse_time_window('το μεσημέρι', now=NOW)
    assert start == _at(12)
    assert end == NOW


def test_relative_hours():
    start, end = parse_time_window('πριν από 2 ώρες', now=NOW)
    assert end == NOW
    assert start == NOW - 2 * HOUR


def test_relative_hours_written_as_a_word():
    start, _ = parse_time_window('πριν από δύο ώρες', now=NOW)
    assert start == NOW - 2 * HOUR


def test_relative_minutes():
    start, _ = parse_time_window('πριν από 30 λεπτά', now=NOW)
    assert start == NOW - 30 * 60


def test_relative_days():
    start, _ = parse_time_window('πριν από 3 μέρες', now=NOW)
    assert start == NOW - 3 * DAY


def test_just_now():
    start, end = parse_time_window('μόλις τώρα', now=NOW)
    assert end == NOW
    assert NOW - start <= HOUR


def test_phrase_without_a_time_returns_none():
    # The caller needs to tell "no period named" from "an empty period".
    assert parse_time_window('πού είναι τα κλειδιά μου', now=NOW) is None


def test_empty_phrase_returns_none():
    assert parse_time_window('', now=NOW) is None
    assert parse_time_window(None, now=NOW) is None


# ── EpisodicLog ───────────────────────────────────────────────────────────────

def _log_with_a_day():
    log = EpisodicLog()
    log.add('heard', 'καλημέρα', who='max', ts=_at(8, 15))
    log.add('room', 'μπήκα στην κουζίνα', room='kouzina', ts=_at(8, 30))
    log.add('heard', 'πού είναι τα κλειδιά', who='mpampas', ts=_at(11, 0))
    log.add('observed', 'Το ποτήρι είναι στο σαλόνι', ts=_at(13, 0))
    log.add('heard', 'γεια σου', who='max', ts=_at(14, 0))
    return log


def test_events_are_recorded_in_order():
    log = _log_with_a_day()
    assert len(log) == 5
    assert [e.ts for e in log.all()] == sorted(e.ts for e in log.all())


def test_consecutive_duplicates_are_collapsed():
    # /current_room republishes on a timer; the log must not fill with it.
    log = EpisodicLog()
    log.add('room', 'μπήκα στο σαλόνι', room='saloni', ts=1000.0)
    assert log.add('room', 'μπήκα στο σαλόνι', room='saloni', ts=1001.0) is None
    assert len(log) == 1


def test_a_repeat_after_something_else_is_kept():
    log = EpisodicLog()
    log.add('room', 'σαλόνι', ts=1000.0)
    log.add('room', 'κουζίνα', ts=1010.0)
    assert log.add('room', 'σαλόνι', ts=1020.0) is not None
    assert len(log) == 3


def test_between_filters_by_window():
    log = _log_with_a_day()
    morning = log.between(_at(5), _at(12))
    assert len(morning) == 3


def test_query_with_a_time_phrase():
    log = _log_with_a_day()
    got = log.query('τι έγινε σήμερα το πρωί', now=NOW)
    assert len(got) == 3


def test_query_filters_by_kind():
    log = _log_with_a_day()
    got = log.query('σήμερα', kinds=['heard'], now=NOW)
    assert len(got) == 3
    assert all(e.kind == 'heard' for e in got)


def test_query_filters_by_speaker():
    log = _log_with_a_day()
    got = log.query('σήμερα', who='max', now=NOW)
    assert len(got) == 2
    assert all(e.who == 'max' for e in got)


def test_speaker_match_ignores_accents():
    log = EpisodicLog()
    log.add('heard', 'γεια', who='Μάξ', ts=NOW - 10)
    assert len(log.query('σήμερα', who='μαξ', now=NOW)) == 1


def test_keyword_query_without_a_time_searches_everything():
    log = _log_with_a_day()
    got = log.query('κλειδιά', now=NOW)
    assert len(got) == 1
    assert 'κλειδιά' in got[0].text


def test_stopwords_do_not_filter_the_results_away():
    # "τι έγινε σήμερα" is pure stopwords + a date; it must return the day, not
    # only events containing the word "έγινε".
    log = _log_with_a_day()
    assert len(log.query('τι έγινε σήμερα', now=NOW)) == 5


def test_keyword_with_no_match_falls_back_to_the_window():
    # Better to answer "here is the period you asked about" than nothing.
    log = _log_with_a_day()
    got = log.query('σήμερα ελικόπτερο', now=NOW)
    assert len(got) == 5


def test_query_respects_the_limit():
    log = EpisodicLog()
    for i in range(100):
        log.add('heard', f'γεγονός {i}', ts=NOW - 100 + i)
    assert len(log.query('σήμερα', now=NOW, limit=10)) == 10


def test_old_events_are_pruned_by_age():
    log = EpisodicLog(max_age_days=7)
    log.add('heard', 'παλιό', ts=NOW - 30 * DAY)
    log.add('heard', 'νέο', ts=NOW)
    assert len(log) == 1
    assert log.all()[0].text == 'νέο'


def test_log_is_bounded_by_count():
    log = EpisodicLog(max_events=10)
    for i in range(50):
        log.add('heard', f'γεγονός {i}', ts=NOW + i)
    assert len(log) == 10
    assert log.all()[-1].text == 'γεγονός 49'


def test_recent_returns_the_tail():
    log = _log_with_a_day()
    assert len(log.recent(2)) == 2
    assert log.recent(2)[-1].text == 'γεια σου'


# ── persistence ───────────────────────────────────────────────────────────────

def test_log_survives_a_round_trip():
    log = _log_with_a_day()
    restored = EpisodicLog.from_dict(log.to_dict())
    assert len(restored) == len(log)
    assert [e.text for e in restored.all()] == [e.text for e in log.all()]
    assert restored.all()[0].who == 'max'


def test_from_dict_tolerates_empty_state():
    assert len(EpisodicLog.from_dict({})) == 0


def test_from_dict_sorts_out_of_order_events():
    data = {'events': [
        Event(NOW, 'heard', 'δεύτερο').to_dict(),
        Event(NOW - 100, 'heard', 'πρώτο').to_dict(),
    ]}
    restored = EpisodicLog.from_dict(data)
    assert [e.text for e in restored.all()] == ['πρώτο', 'δεύτερο']
