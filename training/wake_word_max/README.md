# Custom "Max"/"Μαξ" wake word model

A fully custom openWakeWord model for the wake word "Max" / "Μαξ", trained
entirely on synthetic speech (no real recordings yet). Output: `max.onnx`,
deployed to `../../config/models/max.onnx` and loaded by `wake_word_node.py`
as the default model.

## Pipeline

1. `generate_data.py` — synthesizes positive ("Μαξ"/"Max", several phrasings)
   and negative (other words/phrases, including phonetically-similar hard
   negatives like "Mac"/"Mark"/"Σταμάτα") clips with edge-tts (Greek + English
   neural voices, multiple rate/pitch variants), converts to 16kHz mono WAV.
   -> `raw/*.mp3`, `wav/*.wav`, `manifest.tsv`

2. `extract_features.py` — right-aligns each clip into a 2.0s (32000-sample)
   buffer (0/0.2s trailing silence variants), runs openWakeWord's
   melspectrogram + embedding models to get (16, 96) embedding windows, and
   builds a training set. Each base buffer also gets `N_AUG` augmented
   copies: random gain (quiet/distant mic pickup), synthetic reverb (random
   decaying-noise impulse response), a speaker/mic-style lowpass, and mixing
   with real captured room noise (`noise/room_noise.wav`).
   -> `features.npz`

   **Why augmentation matters**: a model trained only on clean, full-volume
   TTS scores ~0 on real microphone audio. Verified empirically — a "Μαξ"
   clip played over laptop speakers and captured via the laptop mic scored
   ~0 even after gain-matching the levels. The augmented model keeps strong
   separation (clean held-out: pos mean 0.94, neg mean 0.003) and degrades
   gracefully under realistic noise/reverb (pos mean 0.90, neg mean 0.03,
   with some individual clips crossing the 0.5 threshold in either
   direction).

3. `train.py` — trains a small MLP classifier ((16,96) -> 64 -> 16 -> 1,
   sigmoid) on the embeddings and exports to ONNX (opset 17, input
   `[1,16,96]`, output `[1,1]`) — the same shape openWakeWord's pretrained
   wakeword models use, so it drops straight into
   `Model(wakeword_model_paths=[...])`.
   -> `max.onnx`

4. `evaluate.py` — sanity-checks `max.onnx` on held-out manifest clips
   (clean) plus pure noise.

## Known limitations / how to improve

- **No real recordings**: all training data is synthetic TTS. Real speech
  (different pitch, accent, mic, room) will likely score lower than the
  synthetic positives. If false negatives/positives are a problem on the
  actual robot, the highest-value next step is collecting real "Μαξ"/"Max"
  recordings from the actual users on the actual mic (XVF3800) and adding
  them to `manifest.tsv` as additional positives (and any false-triggering
  phrases as new negatives), then re-running steps 2-4.
- **Dev-machine audio loopback is not a valid test**: on this machine, audio
  played through the speakers and captured via the mic comes back ~17dB
  *below* the room noise floor (i.e. inaudible) — confirmed by comparing RMS
  levels of a captured clip vs. a captured silence/noise reference. Don't
  use that loopback setup to judge model quality; test on the actual robot
  hardware (XVF3800) with real spoken audio instead.
- **Threshold tuning**: `threshold` (default 0.5, see `wake_word_node.py`
  parameters) may need adjusting once real-world data is available — under
  the realistic noise/reverb test, a few hard negatives ("Mark", "Aria"-voice
  clips) score 0.4-0.7, so a slightly higher threshold (e.g. 0.6) trades
  fewer false triggers for slightly more missed activations.
- `noise/room_noise.wav` is a single 4s room-noise sample from this dev
  machine's mic. Capturing additional/longer real noise samples (robot idle,
  fan running, TV on, etc.) and mixing them into augmentation would improve
  robustness further.

## Re-running the pipeline

```bash
cd training/wake_word_max
python3 generate_data.py      # ~258 clips via edge-tts + ffmpeg
python3 extract_features.py   # -> features.npz (~2200 examples with augmentation)
python3 train.py               # -> max.onnx
python3 evaluate.py            # sanity check
cp max.onnx ../../config/models/max.onnx
cd ../.. && colcon build --packages-select home_robot --symlink-install
```
