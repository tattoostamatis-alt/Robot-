#!/usr/bin/env python3
"""Interactive recorder for REAL "Έι ρομπότ"/"Hey robot" wake-word samples on the robot mic.

Captures from the reSpeaker XVF3800 ASR-beam channel (ch 4) — the exact signal
`wake_word_node.py` runs detection on — at 16 kHz mono, auto-trims leading/
trailing silence so the spoken word sits near the end of the clip (matching how
extract_features.py right-aligns), lets you audition each take (keep / redo),
saves to wav/ and appends to manifest.tsv with a `real_` prefix.

After recording, re-run the pipeline:
    python3 extract_features.py && python3 train.py && python3 evaluate.py
    cp max.onnx ../../config/models/max.onnx

Usage:
    python3 record_real.py --label pos --count 30           # say "Έι ρομπότ"
    python3 record_real.py --label neg --count 30 --tag robot  # hard negatives
"""

import argparse
import glob
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
RECORD_SECS = 2.0            # captured window per take
FRAME = 320                  # 20 ms RMS frames for silence trimming
PAD = 1600                   # 0.1 s padding kept around voiced region
DEVICE_HINT = 'XVF3800'      # substring match; overridable with --device


def find_device(hint):
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and hint.lower() in d['name'].lower():
            return i, d['max_input_channels']
    return None, None


def trim_silence(audio, thresh_ratio=0.15):
    """Trim to the voiced region using per-frame RMS; keep PAD around it."""
    n = len(audio) // FRAME
    if n == 0:
        return audio
    frames = audio[:n * FRAME].reshape(n, FRAME).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    if rms.max() < 200:            # essentially silence
        return None
    thresh = max(rms.max() * thresh_ratio, 150)
    voiced = np.where(rms > thresh)[0]
    if len(voiced) == 0:
        return None
    start = max(0, voiced[0] * FRAME - PAD)
    end = min(len(audio), (voiced[-1] + 1) * FRAME + PAD)
    return audio[start:end]


def save_wav(path, audio):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


def next_index(label, tag):
    stem = f'real_{label}_{tag}_' if tag else f'real_{label}_'
    existing = glob.glob(os.path.join('wav', stem + '*.wav'))
    idx = 0
    for p in existing:
        try:
            idx = max(idx, int(os.path.basename(p)[len(stem):-4]) + 1)
        except ValueError:
            pass
    return idx, stem


def beep(freq=660, dur=0.12):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    try:
        sd.play((0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', choices=['pos', 'neg'], required=True)
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--tag', default='', help='optional group tag, e.g. speaker or "mac"')
    ap.add_argument('--channel', type=int, default=4, help='mic channel (4 = XVF3800 ASR beam)')
    ap.add_argument('--device', default=DEVICE_HINT, help='input device name substring or index')
    args = ap.parse_args()

    if args.device.isdigit():
        dev_idx = int(args.device)
        dev_ch = sd.query_devices(dev_idx)['max_input_channels']
    else:
        dev_idx, dev_ch = find_device(args.device)
    if dev_idx is None:
        sys.exit(f'No input device matching "{args.device}". Available:\n' +
                 '\n'.join(f'  {i}: {d["name"]}' for i, d in enumerate(sd.query_devices())
                           if d['max_input_channels'] > 0))
    ch = args.channel if args.channel < dev_ch else 0
    print(f'Device {dev_idx} ({sd.query_devices(dev_idx)["name"]}), '
          f'{dev_ch} ch, using channel {ch}')

    os.makedirs('wav', exist_ok=True)
    idx, stem = next_index(args.label, args.tag)
    word = 'Έι ρομπότ' if args.label == 'pos' else (args.tag or 'the negative phrase')
    print(f'\nRecording {args.count} "{args.label}" samples — say: "{word}"')
    print('After each: [Enter]=keep  r=redo  s=skip  q=quit\n')

    saved = 0
    manifest_lines = []
    while saved < args.count:
        input(f'[{saved + 1}/{args.count}] Press Enter, then speak after the beep... ')
        beep()
        rec = sd.rec(int(RECORD_SECS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                     channels=dev_ch, dtype='int16', device=dev_idx)
        sd.wait()
        audio = rec[:, ch].copy()
        trimmed = trim_silence(audio)
        if trimmed is None:
            print('  ⚠ no speech detected — redo')
            continue
        peak = int(np.abs(trimmed).max())
        print(f'  captured {len(trimmed)/SAMPLE_RATE:.2f}s, peak={peak} '
              f'({peak/32768*100:.0f}% FS) — playing back...')
        try:
            sd.play(trimmed, SAMPLE_RATE); sd.wait()
        except Exception:
            pass
        choice = input('  [Enter]=keep  r=redo  s=skip  q=quit: ').strip().lower()
        if choice == 'q':
            break
        if choice == 'r':
            continue
        if choice == 's':
            saved += 1
            continue
        path = os.path.join('wav', f'{stem}{idx:04d}.wav')
        save_wav(path, trimmed)
        manifest_lines.append(f'{args.label}\t{path}')
        print(f'  ✓ saved {path}')
        idx += 1
        saved += 1

    if manifest_lines:
        # ensure the file ends in exactly one newline before appending, so we
        # never introduce a blank line (extract_features.py can't parse those)
        need_nl = False
        if os.path.exists('manifest.tsv') and os.path.getsize('manifest.tsv') > 0:
            with open('manifest.tsv', 'rb') as f:
                f.seek(-1, os.SEEK_END)
                need_nl = f.read(1) != b'\n'
        with open('manifest.tsv', 'a') as f:
            if need_nl:
                f.write('\n')
            f.write('\n'.join(manifest_lines) + '\n')
        print(f'\nAppended {len(manifest_lines)} lines to manifest.tsv')
    print('Done.')


if __name__ == '__main__':
    main()
