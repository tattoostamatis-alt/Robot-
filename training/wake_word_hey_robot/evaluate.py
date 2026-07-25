#!/usr/bin/env python3
"""Sanity-check the exported hey_robot.onnx via openwakeword.model.Model on
held-out manifest clips (right-aligned 2.0s buffers, same as training)
plus pure noise — print scores grouped by label."""

import wave

import numpy as np
from openwakeword.model import Model

WINDOW_SAMPLES = 32000


def load_wav_mono16k(path):
    with wave.open(path, 'rb') as w:
        nch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return audio


def trim_silence(audio, thresh=150, pad=800):
    """Same edge-silence trim as extract_features.py, so eval sees the same
    buffers the model was trained on."""
    loud = np.nonzero(np.abs(audio) > thresh)[0]
    if len(loud) == 0:
        return audio
    return audio[max(0, loud[0] - pad):min(len(audio), loud[-1] + pad)]


def make_buffer(audio):
    if len(audio) > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:]
    leading = WINDOW_SAMPLES - len(audio)
    buf = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    buf[leading:leading + len(audio)] = audio
    return buf


def main():
    model = Model(wakeword_model_paths=['hey_robot.onnx'])
    name = list(model.models.keys())[0]
    print('model name:', name)

    manifest = [line.strip().split('\t') for line in open('manifest.tsv')]

    pos_scores, neg_scores = [], []
    n_skipped = 0
    for label, path in manifest:
        audio = trim_silence(load_wav_mono16k(path))
        if len(audio) > WINDOW_SAMPLES:
            n_skipped += 1  # doesn't fit the window — see extract_features.py
            continue
        audio = make_buffer(audio)
        model.reset()
        # feed enough silence first so prediction_buffer warm-up (5 frames) passes,
        # then feed the real buffer in 1280-sample chunks
        for _ in range(6):
            model.predict(np.zeros(1280, dtype=np.int16))
        score = None
        for i in range(0, WINDOW_SAMPLES, 1280):
            preds = model.predict(audio[i:i + 1280])
            score = preds[name]
        (pos_scores if label == 'pos' else neg_scores).append((score, path))

    print(f'{n_skipped} clips skipped (longer than the 2s window)')

    pos_scores.sort()
    neg_scores.sort(reverse=True)

    print(f'\nPositive clips (n={len(pos_scores)}): min/mean/max = '
          f'{min(s for s, _ in pos_scores):.3f} / '
          f'{np.mean([s for s, _ in pos_scores]):.3f} / '
          f'{max(s for s, _ in pos_scores):.3f}')
    print('5 lowest-scoring positives (potential misses):')
    for s, p in pos_scores[:5]:
        print(f'  {s:.3f}  {p}')

    print(f'\nNegative clips (n={len(neg_scores)}): min/mean/max = '
          f'{min(s for s, _ in neg_scores):.3f} / '
          f'{np.mean([s for s, _ in neg_scores]):.3f} / '
          f'{max(s for s, _ in neg_scores):.3f}')
    print('5 highest-scoring negatives (potential false triggers):')
    for s, p in neg_scores[:5]:
        print(f'  {s:.3f}  {p}')

    # pure noise / silence
    model.reset()
    for _ in range(6):
        model.predict(np.zeros(1280, dtype=np.int16))
    rng = np.random.default_rng(123)
    noise = (rng.standard_normal(WINDOW_SAMPLES) * 50).astype(np.int16)
    score = None
    for i in range(0, WINDOW_SAMPLES, 1280):
        preds = model.predict(noise[i:i + 1280])
        score = preds[name]
    print(f'\nPure noise score: {score:.3f}')


if __name__ == '__main__':
    main()
