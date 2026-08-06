"""Unit tests for open-vocabulary name handling (home_robot/vocabulary.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from home_robot.vocabulary import (  # noqa: E402
    COCO_NAMES, greek_accusative, greek_for, greek_to, needs_open_vocab,
    normalize_vocabulary, strip_accents, to_prompt,
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


# ── greek_accusative ──────────────────────────────────────────────────────────

def test_neuter_and_plural_are_unchanged_but_for_nothing():
    assert greek_accusative('bottle') == 'το μπουκάλι'
    assert greek_accusative('keys') == 'τα κλειδιά'


def test_feminine_takes_the_accusative_article():
    assert greek_accusative('cup') == 'την κούπα'
    assert greek_accusative('chair') == 'την καρέκλα'


def test_masculine_loses_its_sigma():
    """«Να φέρω τον καναπές;» is the giveaway that nothing inflected it."""
    assert greek_accusative('couch') == 'τον καναπέ'
    assert greek_accusative('dog') == 'τον σκύλο'
    assert greek_accusative('person') == 'τον άνθρωπο'


def test_the_final_n_follows_the_next_word():
    """«την βαλίτσα» is the sound of a machine reading a table. The ν stays
    before a vowel or a stop and goes before everything else."""
    assert greek_accusative('suitcase') == 'τη βαλίτσα'      # β  -> drop
    assert greek_accusative('towel') == 'την πετσέτα'         # π  -> keep
    assert greek_accusative('banana') == 'την μπανάνα'        # μπ -> keep
    assert greek_accusative('umbrella') == 'την ομπρέλα'      # vowel -> keep
    assert greek_accusative('lamp') == 'τη λάμπα'             # λ  -> drop


def test_an_unknown_label_is_left_alone():
    """No article is the honest output for a label we cannot inflect — an
    invented one would be wrong out loud, every time."""
    assert greek_accusative('snowboard') == 'snowboard'
    assert greek_to('snowboard') == 'στο snowboard'


# ── greek_to ──────────────────────────────────────────────────────────────────

def test_going_somewhere_merges_the_preposition():
    assert greek_to('chair') == 'στην καρέκλα'
    assert greek_to('couch') == 'στον καναπέ'
    assert greek_to('bed') == 'στο κρεβάτι'
    assert greek_to('suitcase') == 'στη βαλίτσα'


def test_every_greek_label_survives_both_inflections():
    """Whatever the detector reports, both questions must come out as Greek
    with an article and no leftover English."""
    from home_robot.vocabulary import _EN_TO_EL
    for prompt, nominative in _EN_TO_EL.items():
        acc, to = greek_accusative(prompt), greek_to(prompt)
        assert acc and to, prompt
        assert not acc.startswith(('ο ', 'η ', 'οι ')), f'{prompt}: {acc}'
        assert to.startswith('σ'), f'{prompt}: {to}'
        assert len(acc.split()) == len(nominative.split()), prompt


# ── footwear: the 2026-08-06 «πιάσε την παντόφλα» failure ─────────────────────

def test_a_slipper_is_a_name_the_robot_knows():
    """COCO has no footwear class at all, so a slipper is only ever reachable
    through the open-vocabulary detector. It used to map to nothing, the LLM
    guessed the COCO-ish 'shoe' (also not a COCO class), and pick died with
    «δεν βλέπω κάτι να σηκώσω» with the slipper in plain view."""
    for spoken in ('παντόφλα', 'παντόφλες', 'την παντόφλα', 'ΠΑΝΤΟΦΛΑ'):
        assert to_prompt(spoken) == 'slipper', spoken


def test_footwear_always_routes_to_the_open_vocabulary_detector():
    """The guard that makes the difference: if any of these were mistaken for
    COCO classes, pick would wait forever on a detector that never emits them."""
    for prompt in ('slipper', 'shoe', 'shoes'):
        assert prompt not in COCO_NAMES, prompt
        assert needs_open_vocab(prompt), prompt


def test_a_slipper_can_be_spoken_back():
    assert greek_for('slipper') == 'η παντόφλα'
    assert greek_accusative('slipper') == 'την παντόφλα'
