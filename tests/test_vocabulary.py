"""Unit tests for open-vocabulary name handling (home_robot/vocabulary.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.vocabulary import (  # noqa: E402
    COCO_NAMES, greek_for, needs_open_vocab, normalize_vocabulary,
    strip_accents, to_prompt,
)


# ── strip_accents ─────────────────────────────────────────────────────────────

def test_strips_and_lowercases():
    assert strip_accents('ΚΛΕΙΔΙΆ') == 'κλειδια'
    assert strip_accents('γυαλιά') == 'γυαλια'


def test_handles_empty():
    assert strip_accents('') == ''
    assert strip_accents(None) == ''


# ── to_prompt ─────────────────────────────────────────────────────────────────

def test_maps_greek_household_objects():
    assert to_prompt('κλειδιά') == 'keys'
    assert to_prompt('γυαλιά') == 'eyeglasses'
    assert to_prompt('φορτιστής') == 'phone charger'
    assert to_prompt('τηλεκοντρόλ') == 'remote control'
    assert to_prompt('πορτοφόλι') == 'wallet'


def test_inflections_land_on_the_same_prompt():
    # Greek inflects; a table of exact forms would miss most of these.
    for form in ('κλειδί', 'κλειδιά', 'κλειδιών', 'ΚΛΕΙΔΙΑ', 'κλειδια'):
        assert to_prompt(form) == 'keys', form


def test_articles_and_possessives_are_stripped():
    assert to_prompt('τα κλειδιά μου') == 'keys'
    assert to_prompt('το πορτοφόλι του') == 'wallet'


def test_multiword_key_beats_the_shorter_stem():
    # 'γυαλια ηλιου' must not be swallowed by the 'γυαλι' stem.
    assert to_prompt('γυαλιά ηλίου') == 'sunglasses'
    assert to_prompt('γυαλιά') == 'eyeglasses'


def test_english_passes_through():
    # The LLM emits COCO labels in English; they must keep working.
    assert to_prompt('cup') == 'cup'
    assert to_prompt('cell_phone') == 'cell phone'


def test_unknown_greek_returns_none():
    # Better to decline than to invent a prompt and search for nothing.
    assert to_prompt('ελικόπτερο') is None


def test_empty_returns_none():
    assert to_prompt('') is None
    assert to_prompt(None) is None


# ── needs_open_vocab ──────────────────────────────────────────────────────────

def test_coco_classes_do_not_need_open_vocab():
    # object_detector already finds these, continuously and for free.
    for label in ('cup', 'bottle', 'chair', 'person', 'book'):
        assert not needs_open_vocab(label), label


def test_coco_multiword_labels_are_recognised():
    assert not needs_open_vocab('cell phone')
    assert not needs_open_vocab('cell_phone')


def test_non_coco_things_need_open_vocab():
    for label in ('keys', 'wallet', 'phone charger', 'eyeglasses'):
        assert needs_open_vocab(label), label


def test_empty_label_needs_nothing():
    assert not needs_open_vocab('')
    assert not needs_open_vocab(None)


def test_coco_set_is_the_expected_size():
    assert len(COCO_NAMES) == 80


# ── normalize_vocabulary ──────────────────────────────────────────────────────

def test_normalizes_a_mixed_list():
    assert normalize_vocabulary(['κλειδιά', 'wallet']) == ['keys', 'wallet']


def test_accepts_a_bare_string():
    assert normalize_vocabulary('κλειδιά') == ['keys']


def test_deduplicates():
    # Two spellings of the same thing must not cost two classes of inference.
    assert normalize_vocabulary(['κλειδιά', 'κλειδί', 'keys']) == ['keys']


def test_drops_unrecognised_entries():
    assert normalize_vocabulary(['κλειδιά', 'ελικόπτερο']) == ['keys']


def test_caps_the_vocabulary():
    # Every class costs inference time; a caller looping over a phrase could
    # otherwise hand over hundreds.
    assert len(normalize_vocabulary([f'thing{i}' for i in range(50)], limit=8)) == 8


def test_empty_input_is_empty_output():
    assert normalize_vocabulary([]) == []
    assert normalize_vocabulary(None) == []


# ── greek_for ─────────────────────────────────────────────────────────────────

def test_speaks_greek_back():
    assert greek_for('keys') == 'τα κλειδιά'
    assert greek_for('wallet') == 'το πορτοφόλι'


def test_unknown_prompt_is_returned_unchanged():
    assert greek_for('helicopter') == 'helicopter'


def test_round_trip_for_every_mapped_household_item():
    """Every Greek stem must reach a prompt, and every prompt must be speakable
    or deliberately left as-is — a silent gap here becomes a robot that finds
    something and cannot say what."""
    from home_robot.vocabulary import HOUSEHOLD_EL
    for stem, prompt in HOUSEHOLD_EL.items():
        assert to_prompt(stem) == prompt, stem
        assert greek_for(prompt)          # never empty
