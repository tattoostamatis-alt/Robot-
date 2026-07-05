#!/usr/bin/env python3
"""Continuous hard-negative capture for the "Μαξ"/"Max" wake word.

Records one uninterrupted stretch of REAL general speech from the reSpeaker
XVF3800 ASR-beam channel (ch 4 — the exact signal wake_word_node.py detects on),
then auto-slices it into overlapping 2.0 s windows and keeps the ones that
actually contain voiced speech. Each kept window becomes a negative example
(`real_neg_<tag>_NNNN.wav`) appended to manifest.tsv.

Unlike record_real.py this needs NO per-take interaction: you just talk normally
for the whole duration (say anything EXCEPT "Max/Μαξ"). Audio cues:
  3 short beeps + 1 long beep  -> START talking
  2 beeps                      -> STOP

After it finishes, re-run the pipeline:
    python3 extract_features.py && python3 train.py && python3 evaluate.py
    cp max.onnx ../../config/models/max.onnx

Usage:
    python3 record_negatives_continuous.py --secs 180 --tag conv
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
WIN_SAMPLES = 32000          # 2.0 s window, matches extract_features.py buffer
FRAME = 320                  # 20 ms RMS frames
DEVICE_HINT = 'XVF3800'


def find_device(hint):
    for i, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] > 0 and hint.lower() in d['name'].lower():
            return i, d['max_input_channels']
    return None, None


def beep(freq=660, dur=0.12):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    try:
        sd.play((0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass


def save_wav(path, audio):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


def voiced_stats(win):
    """Return (peak, fraction of voiced 20ms frames) for a window."""
    n = len(win) // FRAME
    if n == 0:
        return 0, 0.0
    frames = win[:n * FRAME].reshape(n, FRAME).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    peak = int(np.abs(win).max())
    if rms.max() < 150:
        return peak, 0.0
    thresh = max(rms.max() * 0.15, 150)
    return peak, float((rms > thresh).mean())


def next_index(stem):
    idx = 0
    for p in glob.glob(os.path.join('wav', stem + '*.wav')):
        try:
            idx = max(idx, int(os.path.basename(p)[len(stem):-4]) + 1)
        except ValueError:
            pass
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=180.0, help='record duration')
    ap.add_argument('--hop', type=float, default=1.0, help='window hop (s)')
    ap.add_argument('--tag', default='conv', help='group tag')
    ap.add_argument('--channel', type=int, default=4, help='mic channel (4 = ASR beam)')
    ap.add_argument('--device', default=DEVICE_HINT)
    ap.add_argument('--min-voiced', type=float, default=0.35,
                    help='keep window only if this fraction of frames is voiced')
    ap.add_argument('--lead', type=float, default=2.0, help='count-in before recording')
    args = ap.parse_args()

    if args.device.isdigit():
        dev_idx = int(args.device)
        dev_ch = sd.query_devices(dev_idx)['max_input_channels']
    else:
        dev_idx, dev_ch = find_device(args.device)
    if dev_idx is None:
        sys.exit(f'No input device matching "{args.device}".')
    ch = args.channel if args.channel < dev_ch else 0
    print(f'Device {dev_idx} ({sd.query_devices(dev_idx)["name"]}), {dev_ch} ch, channel {ch}')

    os.makedirs('wav', exist_ok=True)
    stem = f'real_neg_{args.tag}_'
    idx = next_index(stem)

    print(f'\nRecording {args.secs:.0f}s of general speech. Talk normally — '
          f'ANYTHING EXCEPT "Max/Μαξ".')
    print('Cue: 3 short beeps + 1 long beep = START ;  2 beeps at the end = STOP.\n')
    time.sleep(args.lead)
    for _ in range(3):
        beep(660, 0.10); time.sleep(0.4)
    beep(880, 0.35)                       # GO
    time.sleep(0.15)

    rec = sd.rec(int(args.secs * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                 channels=dev_ch, dtype='int16', device=dev_idx)
    sd.wait()
    beep(440, 0.15); time.sleep(0.12); beep(440, 0.15)   # STOP

    audio = rec[:, ch].astype(np.int16).copy()
    hop = int(args.hop * SAMPLE_RATE)
    saved, skipped_silent = 0, 0
    manifest_lines = []
    peaks = []
    for start in range(0, len(audio) - WIN_SAMPLES + 1, hop):
        win = audio[start:start + WIN_SAMPLES]
        peak, vfrac = voiced_stats(win)
        if vfrac < args.min_voiced or peak < 300:
            skipped_silent += 1
            continue
        path = os.path.join('wav', f'{stem}{idx:04d}.wav')
        save_wav(path, win)
        manifest_lines.append(f'neg\t{path}')
        peaks.append(peak)
        idx += 1
        saved += 1

    if manifest_lines:
        need_nl = False
        if os.path.exists('manifest.tsv') and os.path.getsize('manifest.tsv') > 0:
            with open('manifest.tsv', 'rb') as f:
                f.seek(-1, os.SEEK_END)
                need_nl = f.read(1) != b'\n'
        with open('manifest.tsv', 'a') as f:
            if need_nl:
                f.write('\n')
            f.write('\n'.join(manifest_lines) + '\n')

    pk = f'{np.mean(peaks)/32768*100:.0f}% FS mean peak' if peaks else 'n/a'
    print(f'\nSaved {saved} negative windows ({skipped_silent} silent windows '
          f'skipped), {pk}.')
    print(f'Appended {len(manifest_lines)} lines to manifest.tsv')
    if saved < 20:
        print('⚠ Few windows kept — talk more continuously / closer to the mic, '
              'or lower --min-voiced.')


if __name__ == '__main__':
    main()
