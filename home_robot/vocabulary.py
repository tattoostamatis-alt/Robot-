"""Open-vocabulary object names — Greek in, YOLO-World prompts out.

YOLO-World detects whatever you name, but it names things in English, and the
person talking to this robot does not. This module is the translation layer, and
it is deliberately pure so the mapping can be tested without a GPU.

Two problems it solves:

  * **Greek to prompt.** "τα κλειδιά" has to become "keys" — stripped of the
    article, of the accent the STT may or may not have produced, and of the
    inflection ("κλειδιών", "κλειδιά"). Matching on a stem handles the cases a
    dictionary of exact forms would miss.
  * **Knowing when it is even needed.** COCO's 80 classes already cover cups,
    bottles and chairs, and object_detector runs on them continuously for free.
    Open-vocabulary inference is worth its GPU time only for the things COCO
    genuinely lacks — keys, glasses, chargers, remotes, wallets, medicine. So
    `needs_open_vocab` decides, and the node stays idle the rest of the time.
"""

import re
import unicodedata

__all__ = [
    'COCO_NAMES', 'HOUSEHOLD_EL', 'strip_accents', 'to_prompt',
    'needs_open_vocab', 'normalize_vocabulary', 'greek_for',
]

# The 80 classes object_detector already covers. Anything here must NOT wake the
# open-vocabulary model — it would be a second, slower detector for a thing the
# first one already found.
COCO_NAMES = frozenset("""
person bicycle car motorcycle airplane bus train truck boat traffic_light
fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow
elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee
skis snowboard sports_ball kite baseball_bat baseball_glove skateboard
surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana
apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch
potted_plant bed dining_table toilet tv laptop mouse remote keyboard
cell_phone microwave oven toaster sink refrigerator book clock vase scissors
teddy_bear hair_drier toothbrush
""".split())

# Greek stem -> English prompt, for the household things COCO does not have.
# Stems, not whole words: Greek inflects, and "κλειδιά"/"κλειδιών"/"κλειδί" must
# all land on the same prompt. Written accent-free — every lookup is stripped
# first, because the STT's accents are not reliable.
HOUSEHOLD_EL = {
    'κλειδ':        'keys',
    'γυαλι':        'eyeglasses',
    'φορτιστ':      'phone charger',
    'καλωδι':       'cable',
    'τηλεκοντρολ':  'remote control',
    'χειριστηρι':   'remote control',
    'πορτοφολ':     'wallet',
    'χαρτ':         'papers',
    'φαρμακ':       'medicine box',
    'χαπ':          'pill bottle',
    'σκουπιδ':      'trash',
    'καλαθ':        'basket',
    'πετσετ':       'towel',
    'μαξιλαρ':      'pillow',
    'κουβερτ':      'blanket',
    'παπουτσ':      'shoes',
    'καλτσ':        'socks',
    'ρουχ':         'clothes',
    'τσαντ':        'bag',
    'ομπρελ':       'umbrella',
    'κουτ':         'box',
    'σακουλ':       'plastic bag',
    'πιατ':         'plate',
    'κουταλ':       'spoon',
    'πιρουν':       'fork',
    'μαχαιρ':       'knife',
    'ποτηρ':        'glass',
    'κουπ':         'cup',
    'μπουκαλ':      'bottle',
    'βιβλι':        'book',
    'περιοδικ':     'magazine',
    'στυλο':        'pen',
    'μολυβ':        'pencil',
    'ψαλιδ':        'scissors',
    'γυαλια ηλιου': 'sunglasses',
    'ταμπλετ':      'tablet',
    'ακουστικ':     'headphones',
    'ποντικ':       'computer mouse',
    'πληκτρολογ':   'keyboard',
    'λαμπ':         'lamp',
    'κερ':          'candle',
    'φυτο':         'plant',
    'γλαστρ':       'flower pot',
    'πιπιλ':        'pacifier',
    'παιχνιδ':      'toy',
    'μπαλ':         'ball',
    'κουκλ':        'doll',
    'τουβλακ':      'building blocks',
    'σκυλ':         'dog',
    'γατ':          'cat',
}

# Reverse lookup for speaking a result back, best-effort.
_EN_TO_EL = {
    'keys': 'τα κλειδιά', 'eyeglasses': 'τα γυαλιά',
    'phone charger': 'ο φορτιστής', 'cable': 'το καλώδιο',
    'remote control': 'το τηλεκοντρόλ', 'wallet': 'το πορτοφόλι',
    'papers': 'τα χαρτιά', 'medicine box': 'τα φάρμακα',
    'pill bottle': 'τα χάπια', 'trash': 'τα σκουπίδια',
    'basket': 'το καλάθι', 'towel': 'η πετσέτα', 'pillow': 'το μαξιλάρι',
    'blanket': 'η κουβέρτα', 'shoes': 'τα παπούτσια', 'socks': 'οι κάλτσες',
    'clothes': 'τα ρούχα', 'bag': 'η τσάντα', 'umbrella': 'η ομπρέλα',
    'box': 'το κουτί', 'plastic bag': 'η σακούλα', 'plate': 'το πιάτο',
    'glass': 'το ποτήρι', 'cup': 'η κούπα', 'bottle': 'το μπουκάλι',
    'book': 'το βιβλίο', 'magazine': 'το περιοδικό', 'pen': 'το στυλό',
    'pencil': 'το μολύβι', 'scissors': 'το ψαλίδι',
    'sunglasses': 'τα γυαλιά ηλίου', 'tablet': 'το τάμπλετ',
    'headphones': 'τα ακουστικά', 'computer mouse': 'το ποντίκι',
    'keyboard': 'το πληκτρολόγιο', 'lamp': 'η λάμπα', 'candle': 'το κερί',
    'plant': 'το φυτό', 'flower pot': 'η γλάστρα', 'toy': 'το παιχνίδι',
    'ball': 'η μπάλα', 'doll': 'η κούκλα',
}

# Greek articles and possessives that ride along with a spoken object name and
# must not reach the prompt. "φέρε μου τα κλειδιά μου" -> "κλειδια".
_STOPWORDS = frozenset("""
το τα ο η οι του της των τον την ενα μια ενας μου σου του μας σας τους
αυτο αυτα εκεινο καποιο
""".split())


def strip_accents(text):
    """Lowercased and accent-free — the form every lookup happens in."""
    if not text:
        return ''
    decomposed = unicodedata.normalize('NFD', str(text).lower())
    return ''.join(c for c in decomposed if not unicodedata.combining(c))


def _words(text):
    return [w for w in re.findall(r'[\wά-ώΆ-Ώ]+', strip_accents(text))
            if w not in _STOPWORDS]


def to_prompt(text):
    """The English YOLO-World prompt for a spoken Greek object name.

    Returns None when nothing is recognised, so the caller can decline rather
    than search for a prompt it invented. Falls through unchanged for text that
    is already English, which is how the LLM's COCO labels keep working.
    """
    if not text:
        return None
    stripped = strip_accents(text)

    # Multi-word keys first ('γυαλια ηλιου' must beat the 'γυαλι' stem).
    for stem, prompt in sorted(HOUSEHOLD_EL.items(), key=lambda kv: -len(kv[0])):
        if ' ' in stem and stem in stripped:
            return prompt

    for word in _words(text):
        for stem, prompt in sorted(HOUSEHOLD_EL.items(), key=lambda kv: -len(kv[0])):
            if ' ' not in stem and word.startswith(stem):
                return prompt

    # Already-English input (COCO labels from the LLM, or a literal prompt).
    if re.fullmatch(r'[a-z0-9 _-]+', stripped.strip()):
        return stripped.strip().replace('_', ' ')
    return None


def needs_open_vocab(label):
    """True when this target is NOT something object_detector already finds.

    COCO runs continuously and costs nothing extra; open-vocabulary inference
    costs a second model on the same GPU. Only pay it for what COCO lacks.
    """
    if not label:
        return False
    normalized = strip_accents(label).strip().replace(' ', '_')
    return normalized not in COCO_NAMES


def normalize_vocabulary(items, limit=16):
    """Clean a requested vocabulary into prompts YOLO-World can be set to.

    Deduplicated and capped: every extra class costs inference time, and a
    caller looping over a phrase can otherwise hand over hundreds.
    """
    if isinstance(items, str):
        items = [items]
    out = []
    for item in items or []:
        prompt = to_prompt(item)
        if prompt and prompt not in out:
            out.append(prompt)
    return out[:limit]


def greek_for(prompt):
    """Speakable Greek for an English prompt; the prompt itself if unknown."""
    return _EN_TO_EL.get(prompt, prompt)
