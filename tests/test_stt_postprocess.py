"""Tests for Whisper prompt-leakage cleanup (see home_robot/stt_postprocess.py).

Every string in the "real capture" tests below was produced by stt_node on
live mic audio on 2026-07-26 and published to speech_text as if the user had
said it.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_stt_postprocess.py -q
"""
import os

import pytest

from home_robot.stt_postprocess import clean

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = f'{PKG}/home_robot/nodes'


# ── pure echo of the prompt: must be dropped entirely ─────────────────

@pytest.mark.parametrize('raw', [
    'Εντολές προς το ρομπότ Μαξ.',
    'Εντολές προς το ρομπότ Μαξ',
    'Εντολές προς το ρομπότ.',
    'Εντολές προς το ρομπότ Μαξ, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ.',
    'Εντολές προς το ρομπότ Μαξ, στον διάδρομο, στην κουζίνα, στο δωμάτιο του Μαξ.',
    'Εντολές προς το ρομπότ Μαξ, στο σαλόνι, στο δωμάτιο του Μαξ.',
])
def test_meta_echo_is_discarded(raw):
    assert clean(raw) == ''


# ── real command + echoed tail: keep the command, drop the tail ───────

@pytest.mark.parametrize('raw,expected', [
    ('Μαξ, πήγαινε στην κουζίνα, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ.',
     'Μαξ, πήγαινε στην κουζίνα.'),
    ('Μαμπά, πήγαινε στην κουζίνα, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ.',
     'Μαμπά, πήγαινε στην κουζίνα.'),
    ('Πήγαινε στη βάση, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ.',
     'Πήγαινε στη βάση.'),
    ('Στον δωμάτιο του Μαξ, στο δωμάτιο του Μαξ, στο δωμάτιο του Μαξ.',
     'Στον δωμάτιο του Μαξ.'),
])
def test_repeated_prompt_fragments_are_trimmed(raw, expected):
    assert clean(raw) == expected


# ── ordinary speech must survive untouched ────────────────────────────

@pytest.mark.parametrize('raw', [
    'Πού είσαι?',
    'Τι ώρα είναι?',
    'Κουζίνα!',
    'Ευχαριστώ.',
    'Σταμάτα!',
    'Μαξ, σταμάτα!',
    'Πήγαινε στην κουζίνα.',
    'Αυτός ξέρει Γερμανικά, η μητέρα του σκληρή τηλεόραση.',
    'Όχι, ήθελε τέτοιο.',
])
def test_normal_speech_is_untouched(raw):
    assert clean(raw) == raw


def test_single_mention_of_a_room_is_kept():
    # One mention is a legitimate command — only repetition signals leakage.
    raw = 'Πήγαινε στην κουζίνα, στο δωμάτιο του Μαξ.'
    assert clean(raw) == raw


def test_empty_and_blank():
    assert clean('') == ''
    assert clean('   ') == ''


def test_immediate_stutter_is_collapsed():
    assert clean('Λουκάκο, Λουκάκο, Λουκάκο μίχη!') == 'Λουκάκο, Λουκάκο μίχη!'


# ── wiring contract with stt_node ─────────────────────────────────────

def test_stt_node_cleans_before_publishing():
    src = open(f'{NODES}/stt_node.py').read()
    assert 'from home_robot.stt_postprocess import clean' in src
    cleaned = src.index('text = clean(raw)')
    published = src.index('self.text_pub.publish')
    assert cleaned < published, 'transcription must be cleaned before it is published'


def test_stt_node_disables_previous_text_conditioning():
    src = open(f'{NODES}/stt_node.py').read()
    assert 'condition_on_previous_text=False' in src


def test_initial_prompt_has_no_meta_framing():
    # The framing sentence was the single most-leaked string; it must not
    # come back, or clean() would start discarding whole transcriptions.
    src = open(f'{NODES}/stt_node.py').read()
    call = src[src.index('self._whisper.transcribe'):]
    call = call[:call.index(')\n')]
    assert 'Εντολές προς το ρομπότ' not in call
