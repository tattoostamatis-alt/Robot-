#!/usr/bin/env python3
"""How does the wake model hold up when the phrase is said slowly or quickly?

Synthesizes "Έι ρομπότ" across a wide range of speaking rates and streams each
clip through the model the way wake_word_node.py does (1280-sample chunks,
score = max over the stream), rather than the right-aligned training buffer
evaluate.py uses. The runtime view is the one that matters here: a slow phrase
runs past the model's ~1.94s receptive field, so the model only ever sees its
tail — and the tail alone ("ρομπότ") is a deliberate hard negative.

‼️ Clean TTS is useless as a test: every clip scores 1.000, because these are
the same voices the model trained on at full volume. Each clip is therefore
also scored through extract_features.augment (gain, reverb, mic lowpass, room
noise) — the same distortions that stand in for a real mic across a room. Read
the augmented columns, not the clean one.

    python3 rate_sweep.py [--model hey_robot.onnx] [--out rate_sweep.json]
"""
import argparse, asyncio, json, os, pathlib, subprocess, sys, wave

import numpy as np

RATES = ['-40%', '-30%', '-20%', '-10%', '+0%', '+10%', '+20%', '+30%', '+40%']
VOICES = ['el-GR-AthinaNeural', 'el-GR-NestorasNeural']
PHRASES = ['Έι ρομπότ', 'Έι ρομπότ!']
CHUNK = 1280
CACHE = pathlib.Path('rate_sweep_wav')


async def synth(text, voice, rate, path):
    import edge_tts
    for attempt in range(3):          # edge-tts fails ~20% of the time here
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(str(path))
            return True
        except Exception as e:
            if attempt == 2:
                print(f'  edge-tts failed for {path.name}: {e}', file=sys.stderr)
    return False


def to_wav16k(src, dst):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
                    '-ac', '1', '-ar', '16000', str(dst)], check=True)


def load(path):
    with wave.open(str(path), 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def stream_score(model, audio):
    """Max score over the clip, exactly how wake_word_node consumes the mic."""
    model.reset()
    best = 0.0
    padded = np.concatenate([np.zeros(CHUNK * 8, dtype=np.int16), audio,
                             np.zeros(CHUNK * 4, dtype=np.int16)])
    for i in range(0, len(padded) - CHUNK + 1, CHUNK):
        for score in model.predict(padded[i:i + CHUNK]).values():
            best = max(best, float(score))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='hey_robot.onnx')
    ap.add_argument('--out', default='rate_sweep.json')
    ap.add_argument('--threshold', type=float, default=0.50)
    ap.add_argument('--n-aug', type=int, default=8)
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    jobs = [(p, v, r) for r in RATES for v in VOICES for p in PHRASES]
    todo = []
    for text, voice, rate in jobs:
        stem = f'{voice.split("-")[-1]}_{rate.replace("%","").replace("+","p").replace("-","m")}_{"excl" if text.endswith("!") else "plain"}'
        wav = CACHE / f'{stem}.wav'
        if not wav.exists():
            todo.append((text, voice, rate, wav))

    async def synth_all():
        for text, voice, rate, wav in todo:
            mp3 = wav.with_suffix('.mp3')
            if await synth(text, voice, rate, mp3):
                to_wav16k(mp3, wav)
                os.remove(mp3)
    if todo:
        print(f'synthesizing {len(todo)} clips…', flush=True)
        asyncio.run(synth_all())

    from openwakeword.model import Model

    from extract_features import augment, load_wav_mono16k
    model = Model(wakeword_model_paths=[args.model], vad_threshold=0.5)
    noise = load_wav_mono16k('noise/room_noise.wav')
    rng = np.random.default_rng(0)

    rows = []
    print(f'\n{"rate":>6} {"voice":<9} {"ph":<5} {"dur":>5}  {"clean":>5}  '
          f'{"aug mean":>8} {"aug min":>7}  fires')
    for text, voice, rate in jobs:
        stem = f'{voice.split("-")[-1]}_{rate.replace("%","").replace("+","p").replace("-","m")}_{"excl" if text.endswith("!") else "plain"}'
        wav = CACHE / f'{stem}.wav'
        if not wav.exists():
            continue
        audio = load(wav)
        clean = stream_score(model, audio)
        aug = [stream_score(model, augment(audio, noise, rng))
               for _ in range(args.n_aug)]
        fires = sum(s >= args.threshold for s in aug)
        rows.append({'rate': rate, 'voice': voice, 'text': text,
                     'dur': round(len(audio) / 16000, 2),
                     'clean': round(clean, 3),
                     'aug': [round(s, 3) for s in aug],
                     'fires': fires})
        print(f'{rate:>6} {voice.split("-")[-1][:8]:<9} '
              f'{"excl" if text.endswith("!") else "plain":<5} '
              f'{len(audio) / 16000:5.2f}s  {clean:5.3f}  {np.mean(aug):8.3f} '
              f'{min(aug):7.3f}  {fires}/{args.n_aug}')

    print(f'\nby rate (augmented, threshold {args.threshold}):')
    for r in RATES:
        a = [s for x in rows if x['rate'] == r for s in x['aug']]
        if a:
            fires = sum(s >= args.threshold for s in a)
            print(f'  {r:>6}  mean {np.mean(a):.3f}  min {min(a):.3f}  '
                  f'fires {fires}/{len(a)}  ({100 * fires / len(a):.0f}%)')
    allaug = [s for x in rows for s in x['aug']]
    fires = sum(s >= args.threshold for s in allaug)
    print(f'\nTOTAL augmented: {fires}/{len(allaug)} fire '
          f'({100 * fires / len(allaug):.0f}%)')
    pathlib.Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
