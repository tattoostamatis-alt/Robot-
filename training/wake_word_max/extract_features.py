#!/usr/bin/env python3
"""Build (16,96) openWakeWord embedding windows + labels from manifest.tsv.

Each clip is right-aligned into a fixed 2.0s (32000-sample @ 16kHz) buffer
with 0/0.2s of trailing silence after the speech — mimicking the streaming
case where `predict()` looks at the most recent 2s of audio right after (or
shortly after) the word/phrase ends. Each buffer maps to exactly one (16, 96)
embedding window via openWakeWord's melspectrogram + embedding models.

A model trained only on clean, full-volume TTS scores ~0 on real microphone
audio (verified empirically: a "Max" clip played over speakers and captured
via the laptop mic scored ~0 even after gain-matching). To generalize beyond
the dry TTS recording conditions, each base buffer also gets several
augmented copies: random gain (quiet/distant mic pickup), synthetic reverb
(random decaying-noise impulse response), a speaker/mic-style lowpass, and
mixing with real captured room noise (noise/room_noise.wav). A handful of
pure low-level noise buffers are added as extra negatives (silence should
never trigger).
"""

import wave

import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
from openwakeword.utils import AudioFeatures


WINDOW_SAMPLES = 32000  # 2.0s -> exactly 16 embedding frames
TRAILING_VARIANTS = [0, 3200]  # 0 / 0.2s of trailing silence after speech
N_SILENCE_NEG = 40
N_AUG = 4  # augmented copies per base buffer, in addition to the clean original


def load_wav_mono16k(path):
    with wave.open(path, 'rb') as w:
        assert w.getframerate() == 16000, path
        assert w.getsampwidth() == 2, path
        nch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1).astype(np.int16)
    return audio


def make_buffer(audio, trailing):
    avail = WINDOW_SAMPLES - trailing
    if len(audio) > avail:
        if trailing > 0:
            return None  # doesn't fit with trailing silence — skip this variant
        audio = audio[-avail:]  # trim leading context, right-align the end
    leading = avail - len(audio)
    buf = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    buf[leading:leading + len(audio)] = audio
    return buf


def augment(buf_int16, noise_source, rng):
    """Apply random gain + optional reverb/lowpass + room-noise mixing."""
    x = buf_int16.astype(np.float64)

    # random gain — simulate varying mic distance/volume (quiet to loud)
    cur_rms = np.sqrt((x ** 2).mean()) + 1e-6
    target_rms = 10 ** rng.uniform(np.log10(80), np.log10(4000))
    x = x * (target_rms / cur_rms)

    # optional synthetic reverb (decaying-noise impulse response)
    if rng.random() < 0.5:
        rir_len = int(rng.integers(800, 3200))  # 50-200ms @ 16kHz
        tau = rir_len / rng.uniform(2, 5)
        t = np.arange(rir_len)
        rir = rng.standard_normal(rir_len) * np.exp(-t / tau)
        rir = rir / (np.sqrt((rir ** 2).sum()) + 1e-9)
        x = fftconvolve(x, rir, mode='full')[:len(x)]

    # optional lowpass — simulate speaker/mic frequency rolloff
    if rng.random() < 0.5:
        cutoff = rng.uniform(3000, 7000)
        b, a = butter(2, cutoff / (16000 / 2), btype='low')
        x = lfilter(b, a, x)

    # mix in real captured room noise at a random SNR
    snr_db = rng.uniform(5, 25)
    start = int(rng.integers(0, len(noise_source) - len(x) + 1))
    noise = noise_source[start:start + len(x)].astype(np.float64)
    sig_rms = np.sqrt((x ** 2).mean()) + 1e-6
    noise_rms = np.sqrt((noise ** 2).mean()) + 1e-6
    noise_scale = (sig_rms / (10 ** (snr_db / 20))) / noise_rms
    x = x + noise * noise_scale

    return np.clip(x, -32768, 32767).astype(np.int16)


def main():
    manifest = [line.strip().split('\t') for line in open('manifest.tsv')]
    noise_source = load_wav_mono16k('noise/room_noise.wav')
    rng = np.random.default_rng(0)

    buffers = []
    labels = []
    for label, path in manifest:
        audio = load_wav_mono16k(path)
        for trailing in TRAILING_VARIANTS:
            buf = make_buffer(audio, trailing)
            if buf is None:
                continue
            buffers.append(buf)
            labels.append(1 if label == 'pos' else 0)
            for _ in range(N_AUG):
                buffers.append(augment(buf, noise_source, rng))
                labels.append(1 if label == 'pos' else 0)

    for _ in range(N_SILENCE_NEG):
        buffers.append((rng.standard_normal(WINDOW_SAMPLES) * 50).astype(np.int16))
        labels.append(0)

    X = np.stack(buffers)
    y = np.array(labels, dtype=np.float32)
    print(f'Total examples: {len(y)} (pos={int(y.sum())}, neg={int((1 - y).sum())})')

    af = AudioFeatures()
    embeddings = af.embed_clips(X, batch_size=64)
    print('embeddings shape:', embeddings.shape)

    np.savez('features.npz', X=embeddings.astype(np.float32), y=y)
    print('Saved features.npz')


if __name__ == '__main__':
    main()
