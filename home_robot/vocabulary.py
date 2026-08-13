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
    'greek_accusative', 'greek_to',
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
    # ‼️ COCO has NO footwear class at all. 'shoe' sits in object_detector's
    # CLUTTER_CLASSES but yolo11n-seg can never emit it, so a slipper is only
    # ever findable through the open-vocabulary detector. Listed here (before
    # 'παντελον', which shares no stem) because on 2026-08-06 «πιάσε την
    # παντόφλα» reached the LLM, which guessed the COCO-ish label 'shoe', and
    # pick died with "δεν βλέπω κάτι να σηκώσω".
    'παντοφλ':      'slipper',
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
    'slipper': 'η παντόφλα', 'slippers': 'οι παντόφλες', 'shoe': 'το παπούτσι',
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
    # The COCO labels the robot actually says out loud — `goto_object` and
    # `fetch` take them straight from the detector, and without these the
    # spoken confirmation was «Να πάω στο chair;». Only the ones that exist in
    # a home; the remaining 32 (giraffe, snowboard) would just be noise.
    'chair': 'η καρέκλα', 'couch': 'ο καναπές', 'sofa': 'ο καναπές',
    'bed': 'το κρεβάτι', 'dining_table': 'το τραπέζι',
    'dining table': 'το τραπέζι', 'table': 'το τραπέζι',
    'tv': 'η τηλεόραση', 'laptop': 'το λάπτοπ', 'mouse': 'το ποντίκι',
    'cell_phone': 'το κινητό', 'cell phone': 'το κινητό',
    'refrigerator': 'το ψυγείο', 'microwave': 'ο φούρνος μικροκυμάτων',
    'oven': 'ο φούρνος', 'sink': 'ο νεροχύτης', 'toilet': 'η τουαλέτα',
    'potted_plant': 'το φυτό', 'potted plant': 'το φυτό', 'vase': 'το βάζο',
    'clock': 'το ρολόι', 'backpack': 'το σακίδιο', 'handbag': 'η τσάντα',
    'suitcase': 'η βαλίτσα', 'teddy_bear': 'το αρκουδάκι',
    'teddy bear': 'το αρκουδάκι', 'sports_ball': 'η μπάλα',
    'sports ball': 'η μπάλα',
    'bowl': 'το μπολ', 'fork': 'το πιρούνι', 'knife': 'το μαχαίρι',
    'spoon': 'το κουτάλι', 'wine_glass': 'το ποτήρι',
    'wine glass': 'το ποτήρι',
    'banana': 'η μπανάνα', 'apple': 'το μήλο', 'orange': 'το πορτοκάλι',
    'person': 'ο άνθρωπος', 'cat': 'η γάτα', 'dog': 'ο σκύλος',
    'remote': 'το τηλεκοντρόλ', 'toothbrush': 'η οδοντόβουρτσα',
    'toaster': 'η τοστιέρα', 'hair_drier': 'το πιστολάκι',
    'hair drier': 'το πιστολάκι', 'carrot': 'το καρότο',
    'sandwich': 'το σάντουιτς', 'donut': 'το ντόνατ', 'cake': 'η τούρτα',
    'pizza': 'η πίτσα', 'broccoli': 'το μπρόκολο',
}

# Nominative -> accusative, for the article and for the one masculine ending
# pattern that changes. Everything else in the table above is neuter or a
# feminine noun in -α/-η, which are identical in both cases.
# 'οι' is feminine everywhere in this table ('οι κάλτσες'); a masculine plural
# would want 'τους'.
_ACC_ARTICLE = {'ο': 'τον', 'η': 'την', 'το': 'το', 'οι': 'τις', 'τα': 'τα'}
_MASC_ACC = (('ής', 'ή'), ('ές', 'έ'), ('ας', 'α'), ('ος', 'ο'))

# Final-ν rule, feminine only: «την κούπα» but «τη βαλίτσα». The ν survives
# before a vowel and before a stop — κ π τ ξ ψ and the digraphs γκ μπ ντ τσ τζ
# — and is dropped before everything else. Masculine 'τον' keeps its ν always
# in modern usage (it is what tells it apart from 'το'), so it is not listed.
# Without this the robot says «τη» and «την» in exactly the wrong places, which
# is audible in a way a missing article is not.
_KEEPS_N = ('γκ', 'μπ', 'ντ', 'τσ', 'τζ')
_KEEPS_N_ONE = frozenset('αεηιουω' 'κπτξψ')

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


def greek_accusative(prompt):
    """'cup' -> 'την κούπα'. What the robot FETCHES, not what it is called.

    The table stores nominative ('η κούπα') because that is how you name a
    thing. Every sentence that acts on it needs the accusative, and «Να φέρω η
    κούπα;» is exactly the kind of robot-Greek room_locative_el was written to
    stop. Unknown prompts fall through untouched — a label with no article is
    an English fallback, and inventing one would be worse than saying it plain.
    """
    phrase = greek_for(prompt)
    parts = str(phrase).split()
    if len(parts) < 2 or parts[0] not in _ACC_ARTICLE:
        return phrase
    article, head, tail = parts[0], parts[1], parts[2:]
    if article == 'ο':
        for nom, acc in _MASC_ACC:
            if head.endswith(nom):
                head = head[:-len(nom)] + acc
                break
    return ' '.join([_final_n(_ACC_ARTICLE[article], head), head] + tail)


def _final_n(article, word):
    """'την' before «κούπα», 'τη' before «βαλίτσα». Feminine singular only."""
    if article != 'την':
        return article
    stem = strip_accents(word.lower())
    if stem[:2] in _KEEPS_N or stem[:1] in _KEEPS_N_ONE:
        return article
    return 'τη'


def greek_to(prompt):
    """'chair' -> 'στην καρέκλα'. Where the robot GOES."""
    acc = greek_accusative(prompt)
    first = str(acc).split(' ')[0]
    # 'σ' + the accusative article is the whole rule: στον/στην/στη/στο/στα/στις
    # — the final-ν decision was already made one call down, and it is the same
    # decision here ('στη βαλίτσα', 'στην κούπα').
    if first in set(_ACC_ARTICLE.values()) | {'τη'}:
        return 'σ' + acc
    return f'στο {acc}'
