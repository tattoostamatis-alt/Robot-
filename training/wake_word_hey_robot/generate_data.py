#!/usr/bin/env python3
"""Generate synthetic TTS training data for the "Έι ρομπότ"/"Hey robot" wake word.

Uses edge-tts (Greek + English neural voices) with rate/pitch variants
to synthesize positive ("Έι ρομπότ"/"Hey robot") and negative (other
words/phrases, including phonetically-similar hard negatives) clips,
then converts each to 16kHz mono PCM WAV via ffmpeg.

Output: wav/*.wav + manifest.tsv (label<TAB>path)
"""

import asyncio
import os
import subprocess

import edge_tts


OUT_RAW = 'raw'
OUT_WAV = 'wav'
os.makedirs(OUT_RAW, exist_ok=True)
os.makedirs(OUT_WAV, exist_ok=True)

GREEK_VOICES = ['el-GR-AthinaNeural', 'el-GR-NestorasNeural']
ENGLISH_VOICES = ['en-US-AriaNeural', 'en-US-GuyNeural', 'en-US-JennyNeural',
                  'en-GB-RyanNeural', 'en-GB-SoniaNeural', 'en-US-AndrewNeural']

# Wake phrase is "Έι ρομπότ" / "Hey robot" (replaced "Ρομπότ Μαξ" on 2026-07-25).
# Both spellings of the Greek interjection ("Έι" / "Χέι") are synthesized, since
# the two Greek voices render them slightly differently.
POS_GREEK = ['Έι ρομπότ', 'Έι ρομπότ!', 'Χέι ρομπότ', 'Έι ρομπότ έλα εδώ',
             'Έι ρομπότ άκου με', 'Έι ρομποτ']
# Keep the English positives short and phrase-final: adding trailing-word
# variants ("Hey robot hello", "…are you there") taught the model that "robot"
# anywhere in the window is enough, which made "my robot" score 0.8+.
POS_ENGLISH = ['Hey robot', 'Hey robot!', 'Hey robot come here', 'Hey robot listen to me']

NEG_GREEK = ['Γεια σου', 'Πήγαινε στην κουζίνα', 'Σταμάτα', 'Καθάρισε το δωμάτιο',
             'Τι ώρα είναι', 'Άσε με ήσυχο', 'Πήγαινε στο σαλόνι', 'Επέστρεψε στη βάση',
             'Ένα δύο τρία', 'Ευχαριστώ πολύ', 'Πόσο κάνει αυτό', 'Δεν καταλαβαίνω',
             'Βοήθεια', 'Άναψε το φως', 'Κλείσε την πόρτα',
             # phonetically close to "ρομπότ"
             'Ρόμπα', 'Μπότες', 'Ρόδα', 'Ρόμπερτ']
NEG_ENGLISH = ['Hello', 'Stop', 'Robert', 'Rowboat', 'Rope', 'Roll it', 'Report',
               'Rabbit', 'Go to the kitchen', 'What time is it', 'Thank you',
               'Turn on the light']

# Critical hard negatives: either half of the wake phrase spoken ALONE must NOT
# trigger. Weighted heavily (all voices, 3 variants) so the model learns that
# only "Έι" AND "ρομπότ" together fire — plain "ρομπότ" (very common in normal
# commands to the robot) or a bare "έι"/"hey" stays silent. The retired
# "Ρομπότ Μαξ" phrase is in here too, so the old wake word no longer fires.
NEG_HARD_GREEK = ['Ρομπότ', 'Το ρομπότ', 'ρομπότ μου', 'Ένα ρομπότ', 'Το ρομπότ μου',
                  'Πού είναι το ρομπότ', 'Το ρομπότ καθαρίζει',
                  'Έι', 'Έι εσύ', 'Έι ακούς', 'Ρομπότ Μαξ', 'Μαξ']
NEG_HARD_ENGLISH = ['Robot', 'The robot', 'My robot', 'A robot', 'Where is the robot',
                    'Hey', 'Hey you', 'Hey there', 'Robot Max', 'Hey Max']

VARIANTS_3 = [('-15%', '-25Hz'), ('+0%', '+0Hz'), ('+15%', '+25Hz')]
VARIANTS_2 = [('-15%', '-25Hz'), ('+15%', '+25Hz')]
VARIANTS_1 = [('+0%', '+0Hz')]
# edge-tts only ships two Greek voices, so Greek data is thin compared to the
# six English ones — the misclassified clips were nearly all Greek, and nearly
# all from the same voice at the slow rate. Extra rate/pitch variants are the
# only cheap way to widen Greek coverage.
VARIANTS_5 = [('-20%', '-40Hz'), ('-10%', '-15Hz'), ('+0%', '+0Hz'),
              ('+10%', '+15Hz'), ('+20%', '+40Hz')]

# ‼️ The false-trigger case that matters, added 2026-07-30 after a live test:
# with the model deployed, ordinary speech containing "ρομπότ" woke the robot 11
# times in 4.3 minutes. The window the model sees is ~1.94s ending at the word,
# so what it must reject is a sentence ENDING in "ρομπότ" — and the old negative
# set barely had any ("Το ρομπότ καθαρίζει" ends elsewhere). These are exactly
# that shape: a carrier phrase, then "ρομπότ" last, the way you address it.
# "εντάξει ρομπότ" is in here on purpose: "…ξει ρομπότ" is the nearest miss to
# "έι ρομπότ" that normal Greek produces.
NEG_FINAL_GREEK = [
    'πήγαινε στην κουζίνα ρομπότ', 'σταμάτα ρομπότ', 'έλα εδώ ρομπότ',
    'καλημέρα ρομπότ', 'καληνύχτα ρομπότ', 'ευχαριστώ ρομπότ', 'άκου ρομπότ',
    'πάμε ρομπότ', 'όχι ρομπότ', 'ναι ρομπότ', 'ωραία ρομπότ', 'μπράβο ρομπότ',
    'τι κάνεις ρομπότ', 'πού είσαι ρομπότ', 'γεια σου ρομπότ', 'περίμενε ρομπότ',
    'γύρνα πίσω ρομπότ', 'καθάρισε το σαλόνι ρομπότ', 'πήγαινε στη βάση σου ρομπότ',
    'πρόσεχε ρομπότ', 'βοήθησέ με ρομπότ', 'εντάξει ρομπότ', 'ρε ρομπότ',
    'καλό ρομπότ', 'κακό ρομπότ', 'αυτό το ρομπότ', 'το δικό μου ρομπότ',
    'ρομπότ έλα', 'ρομπότ σταμάτα', 'ρομπότ πήγαινε', 'ρομπότ άκου',
]

jobs = []  # (label, text, voice, rate, pitch, name)


def add_jobs(label, texts, voices, variants):
    for v in voices:
        for t in texts:
            for r, p in variants:
                idx = len(jobs)
                name = f'{label}_{idx:04d}_{v}'
                jobs.append((label, t, v, r, p, name))


add_jobs('pos', POS_GREEK, GREEK_VOICES, VARIANTS_5)
add_jobs('pos', POS_ENGLISH, ENGLISH_VOICES, VARIANTS_3)
add_jobs('neg', NEG_GREEK, GREEK_VOICES, VARIANTS_3)
add_jobs('neg', NEG_ENGLISH, ENGLISH_VOICES, VARIANTS_1)
add_jobs('neg', NEG_HARD_GREEK, GREEK_VOICES, VARIANTS_5)
add_jobs('neg', NEG_HARD_ENGLISH, ENGLISH_VOICES, VARIANTS_3)
# appended LAST so the indices of every existing clip stay put
add_jobs('neg', NEG_FINAL_GREEK, GREEK_VOICES, VARIANTS_5)

print(f'Total jobs: {len(jobs)} '
      f'(pos={sum(1 for j in jobs if j[0]=="pos")}, '
      f'neg={sum(1 for j in jobs if j[0]=="neg")})')

sem = asyncio.Semaphore(8)


async def gen_one(job):
    label, text, voice, rate, pitch, name = job
    mp3_path = f'{OUT_RAW}/{name}.mp3'
    if os.path.exists(f'{OUT_WAV}/{name}.wav'):
        return label, name          # already synthesized — keep it
    async with sem:
        try:
            c = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await c.save(mp3_path)
        except Exception as e:
            print(f'FAILED {name}: {e}')
            return None
    return label, name


async def main():
    results = await asyncio.gather(*(gen_one(j) for j in jobs))

    manifest = []
    for r in results:
        if r is None:
            continue
        label, name = r
        mp3_path = f'{OUT_RAW}/{name}.mp3'
        wav_path = f'{OUT_WAV}/{name}.wav'
        if not os.path.exists(wav_path):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3_path,
                            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', wav_path], check=True)
        manifest.append((label, wav_path))

    # ‼️ Real recordings live in the same manifest and are NOT regenerable —
    # carry them across instead of overwriting the file out from under them.
    real = []
    if os.path.exists('manifest.tsv'):
        real = [tuple(l.strip().split('\t')) for l in open('manifest.tsv')
                if l.strip() and 'real_' in l]
    with open('manifest.tsv', 'w') as f:
        for label, path in manifest + real:
            f.write(f'{label}\t{path}\n')
    if real:
        print(f'kept {len(real)} real recordings in the manifest')

    n_pos = sum(1 for label, _ in manifest if label == 'pos')
    n_neg = sum(1 for label, _ in manifest if label == 'neg')
    print(f'Done: {len(manifest)} clips (pos={n_pos}, neg={n_neg})')


if __name__ == '__main__':
    asyncio.run(main())
