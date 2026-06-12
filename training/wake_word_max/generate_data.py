#!/usr/bin/env python3
"""Generate synthetic TTS training data for the "Max"/"Μαξ" wake word.

Uses edge-tts (Greek + English neural voices) with rate/pitch variants
to synthesize positive ("Μαξ"/"Max") and negative (other words/phrases,
including phonetically-similar hard negatives) clips, then converts
each to 16kHz mono PCM WAV via ffmpeg.

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

POS_GREEK = ['Μαξ', 'Μαξ!', 'Έι Μαξ', 'Γεια σου Μαξ', 'Μαξ ελα εδω', 'Μαξ ακου με']
POS_ENGLISH = ['Max', 'Max!', 'Hey Max', 'Okay Max']

NEG_GREEK = ['Γεια σου', 'Πήγαινε στην κουζίνα', 'Σταμάτα', 'Καθάρισε το δωμάτιο',
             'Τι ώρα είναι', 'Άσε με ήσυχο', 'Πήγαινε στο σαλόνι', 'Επέστρεψε στη βάση',
             'Ένα δύο τρία', 'Ευχαριστώ πολύ', 'Πόσο κάνει αυτό', 'Δεν καταλαβαίνω',
             'Βοήθεια', 'Άναψε το φως', 'Κλείσε την πόρτα']
NEG_ENGLISH = ['Hello', 'Stop', 'Mark', 'Mac', 'Mix', 'Tax', 'Fax', 'Mask', 'Sax',
               'Jack', 'Alex', 'Go to the kitchen', 'What time is it', 'Thank you',
               'Turn on the light']

VARIANTS_3 = [('-15%', '-25Hz'), ('+0%', '+0Hz'), ('+15%', '+25Hz')]
VARIANTS_2 = [('-15%', '-25Hz'), ('+15%', '+25Hz')]
VARIANTS_1 = [('+0%', '+0Hz')]

jobs = []  # (label, text, voice, rate, pitch, name)


def add_jobs(label, texts, voices, variants):
    for v in voices:
        for t in texts:
            for r, p in variants:
                idx = len(jobs)
                name = f'{label}_{idx:04d}_{v}'
                jobs.append((label, t, v, r, p, name))


add_jobs('pos', POS_GREEK, GREEK_VOICES, VARIANTS_3)
add_jobs('pos', POS_ENGLISH, ENGLISH_VOICES, VARIANTS_3)
add_jobs('neg', NEG_GREEK, GREEK_VOICES, VARIANTS_2)
add_jobs('neg', NEG_ENGLISH, ENGLISH_VOICES, VARIANTS_1)

print(f'Total jobs: {len(jobs)} '
      f'(pos={sum(1 for j in jobs if j[0]=="pos")}, '
      f'neg={sum(1 for j in jobs if j[0]=="neg")})')

sem = asyncio.Semaphore(8)


async def gen_one(job):
    label, text, voice, rate, pitch, name = job
    mp3_path = f'{OUT_RAW}/{name}.mp3'
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
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3_path,
                        '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', wav_path], check=True)
        manifest.append((label, wav_path))

    with open('manifest.tsv', 'w') as f:
        for label, path in manifest:
            f.write(f'{label}\t{path}\n')

    n_pos = sum(1 for label, _ in manifest if label == 'pos')
    n_neg = sum(1 for label, _ in manifest if label == 'neg')
    print(f'Done: {len(manifest)} clips (pos={n_pos}, neg={n_neg})')


if __name__ == '__main__':
    asyncio.run(main())
