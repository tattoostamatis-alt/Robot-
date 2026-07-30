#!/usr/bin/env python3
"""Where should the wake threshold sit? Misses vs false triggers, measured.

rate_sweep.py showed that at threshold 0.50 roughly one in six *augmented*
positives fails to fire, and that speaking rate is not what decides it —
level, distance and room noise are. Lowering the threshold is the cheap fix,
but only if the negatives (especially "ρομπότ" alone, which is in half the
commands to this robot) stay silent.

Both sides are scored the way wake_word_node.py consumes the mic — streamed in
1280-sample chunks, score = max over the stream — through the same augment()
the training set uses, so the numbers describe a mic across a room rather than
clean TTS.

    python3 threshold_sweep.py [--n-aug 3] [--model hey_robot.onnx]
"""
import argparse, json, pathlib

import numpy as np

from extract_features import augment, load_wav_mono16k
from rate_sweep import stream_score

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='hey_robot.onnx')
    ap.add_argument('--n-aug', type=int, default=3)
    ap.add_argument('--out', default='threshold_sweep.json')
    args = ap.parse_args()

    from openwakeword.model import Model
    model = Model(wakeword_model_paths=[args.model], vad_threshold=0.5)
    noise = load_wav_mono16k('noise/room_noise.wav')
    rng = np.random.default_rng(1)

    manifest = [ln.strip().split('\t') for ln in open('manifest.tsv') if ln.strip()]
    pos = [p for label, p in manifest if label == 'pos']
    neg = [p for label, p in manifest if label == 'neg']
    print(f'{len(pos)} positives, {len(neg)} negatives, {args.n_aug} augmented '
          f'copies each', flush=True)

    scores = {'pos': [], 'neg': []}
    for kind, paths in (('pos', pos), ('neg', neg)):
        for i, path in enumerate(paths):
            audio = load_wav_mono16k(path)
            for _ in range(args.n_aug):
                scores[kind].append(stream_score(model, augment(audio, noise, rng)))
            if (i + 1) % 50 == 0:
                print(f'  {kind} {i + 1}/{len(paths)}', flush=True)

    p = np.array(scores['pos'])
    n = np.array(scores['neg'])
    print(f'\n{"thresh":>7} {"fires":>7} {"misses":>7}   {"false":>6} {"rate":>7}')
    rows = []
    for t in THRESHOLDS:
        hit = float((p >= t).mean())
        fp = int((n >= t).sum())
        rows.append({'threshold': t, 'hit_rate': round(hit, 4),
                     'false_triggers': fp, 'n_neg': len(n)})
        print(f'{t:7.2f} {100 * hit:6.1f}% {int((p < t).sum()):7d}   '
              f'{fp:6d} {100 * fp / len(n):6.2f}%')

    print(f'\npositives : mean {p.mean():.3f}  p10 {np.percentile(p, 10):.3f}  '
          f'min {p.min():.3f}')
    print(f'negatives : max  {n.max():.3f}  p99 {np.percentile(n, 99):.3f}  '
          f'mean {n.mean():.3f}')
    pathlib.Path(args.out).write_text(json.dumps(
        {'rows': rows, 'pos': p.round(4).tolist(), 'neg': n.round(4).tolist()},
        indent=2))


if __name__ == '__main__':
    main()
