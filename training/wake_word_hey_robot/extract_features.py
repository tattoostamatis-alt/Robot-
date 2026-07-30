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
FRAME = 320   # 20ms RMS frames, as in record_real.py — used by fit_window()
PAD = 1600    # 0.1s kept around the voiced region


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


def trim_silence(audio, thresh=150, pad=800):
    """Strip edge silence (edge-tts pads every clip) so more clips fit the
    2s window on their own merits instead of being dropped for length."""
    loud = np.nonzero(np.abs(audio) > thresh)[0]
    if len(loud) == 0:
        return audio
    lo = max(0, loud[0] - pad)
    hi = min(len(audio), loud[-1] + pad)
    return audio[lo:hi]


def make_buffer(audio, trailing):
    avail = WINDOW_SAMPLES - trailing
    if len(audio) > avail:
        # Don't left-truncate to make it fit: that silently relabels the clip.
        # A slow "Έι ρομπότ!" (2.26s) loses its "Έι" and is still taught as a
        # positive — i.e. "ρομπότ" alone means wake — while a slow "my robot"
        # (2.16s) loses its "m" and lands acoustically on top of the wake
        # phrase as a negative. Those two mislabelings were the whole source
        # of the near-miss false triggers. Skip anything that doesn't fit.
        return None
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


def fit_window(audio):
    """Trim hard enough to fit the 2.0s window — but never truncate.

    The fixed threshold above is tuned for edge-tts, whose padding is digital
    silence. Real XVF3800 recordings carry a room-noise floor (median RMS ~88,
    against thresh=150 — close enough that breaths and reverb tails survive the
    trim), so 25 of the first 45 real takes came out longer than the window
    even though the speech in them is a median 1.50s. Dropping those would have
    thrown away most of the real data, and specifically the slow takes it was
    recorded for.

    So retry with progressively stricter relative thresholds, keyed to each
    clip's own peak — the same idea record_real.py uses to find the voiced
    region. If it still does not fit, the clip is dropped, NOT cut: truncating
    is what turned a slow "Έι ρομπότ" into a positive example of bare "ρομπότ"
    on 2026-07-25.
    """
    if len(audio) <= WINDOW_SAMPLES:
        return audio
    n = len(audio) // FRAME
    if n:
        frames = audio[:n * FRAME].reshape(n, FRAME).astype(np.float64)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        for ratio in (0.15, 0.25, 0.35):
            voiced = np.where(rms > max(rms.max() * ratio, 150))[0]
            if len(voiced) == 0:
                continue
            start = max(0, voiced[0] * FRAME - PAD)
            end = min(len(audio), (voiced[-1] + 1) * FRAME + PAD)
            candidate = audio[start:end]
            if len(candidate) <= WINDOW_SAMPLES:
                return candidate
    return audio          # still too long — caller drops it


def main():
    manifest = [line.strip().split('\t') for line in open('manifest.tsv')]
    noise_source = load_wav_mono16k('noise/room_noise.wav')
    rng = np.random.default_rng(0)

    buffers = []
    labels = []
    clean = []  # 1 = untouched TTS buffer, 0 = augmented/synthetic noise
    n_skipped = 0
    n_rescued = 0
    for label, path in manifest:
        audio = trim_silence(load_wav_mono16k(path))
        if len(audio) > WINDOW_SAMPLES:
            audio = fit_window(audio)
            if len(audio) > WINDOW_SAMPLES:
                n_skipped += 1
            else:
                n_rescued += 1
        for trailing in TRAILING_VARIANTS:
            buf = make_buffer(audio, trailing)
            if buf is None:
                continue
            buffers.append(buf)
            labels.append(1 if label == 'pos' else 0)
            clean.append(1)
            for _ in range(N_AUG):
                buffers.append(augment(buf, noise_source, rng))
                labels.append(1 if label == 'pos' else 0)
                clean.append(0)

    for _ in range(N_SILENCE_NEG):
        buffers.append((rng.standard_normal(WINDOW_SAMPLES) * 50).astype(np.int16))
        labels.append(0)
        clean.append(0)

    X = np.stack(buffers)
    y = np.array(labels, dtype=np.float32)
    clean = np.array(clean, dtype=np.float32)
    print(f'Total examples: {len(y)} (pos={int(y.sum())}, neg={int((1 - y).sum())}); '
          f'{n_rescued} over-length clips re-trimmed to fit, '
          f'{n_skipped} still too long and dropped')

    af = AudioFeatures()
    embeddings = af.embed_clips(X, batch_size=64)
    print('embeddings shape:', embeddings.shape)

    np.savez('features.npz', X=embeddings.astype(np.float32), y=y, clean=clean)
    print('Saved features.npz')


if __name__ == '__main__':
    main()
