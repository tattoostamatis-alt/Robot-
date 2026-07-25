# Custom "Έι ρομπότ" / "Hey robot" wake word model

A fully custom openWakeWord model for the wake phrase "Έι ρομπότ" / "Hey
robot", trained entirely on synthetic speech. Output: `hey_robot.onnx`,
deployed to `../../config/models/hey_robot.onnx` and loaded by
`wake_word_node.py` as the default model.

Replaced the previous "Ρομπότ Μαξ" phrase on 2026-07-25 at the user's request.
Both halves of the phrase spoken alone are trained in as heavily-weighted
**hard negatives** — plain "ρομπότ" matters most, since it turns up in ordinary
commands to the robot — as is the retired "Ρομπότ Μαξ", so the old wake word no
longer fires. Held-out eval: positives min 0.609 / mean 0.991, negatives max
0.382 (next-highest 0.009, mean 0.001), pure noise 0.000.

## Pipeline

1. `generate_data.py` — synthesizes positive ("Έι ρομπότ"/"Hey robot", several
   phrasings) and negative clips with edge-tts (Greek + English neural voices,
   multiple rate/pitch variants), converts to 16kHz mono WAV.
   -> `raw/*.mp3`, `wav/*.wav`, `manifest.tsv`

   Negatives include phonetic near-misses ("ρόμπα", "μπότες", "Robert",
   "rowboat") and the hard negatives above. edge-tts ships only two Greek
   voices against six English ones, so Greek phrases get five rate/pitch
   variants (`VARIANTS_5`) to compensate — the misclassified clips were
   overwhelmingly Greek before that.

2. `extract_features.py` — trims edge silence, right-aligns each clip into a
   2.0s (32000-sample) buffer (0/0.2s trailing silence variants), runs
   openWakeWord's melspectrogram + embedding models to get (16, 96) embedding
   windows, and builds a training set. Each base buffer also gets `N_AUG`
   augmented copies: random gain (quiet/distant mic pickup), synthetic reverb
   (random decaying-noise impulse response), a speaker/mic-style lowpass, and
   mixing with real captured room noise (`noise/room_noise.wav`).
   -> `features.npz`

   **Why augmentation matters**: a model trained only on clean, full-volume
   TTS scores ~0 on real microphone audio. Verified empirically — a clip
   played over laptop speakers and captured via the laptop mic scored ~0 even
   after gain-matching the levels.

   **Clips that don't fit the window are dropped, never truncated.** This was
   a real bug, fixed 2026-07-25: an over-length clip used to be trimmed from
   the left and kept its label, so a slow "Έι ρομπότ!" (2.26s) became a
   *positive* example of bare "ρομπότ", and a slow "my robot" (2.16s) became a
   *negative* that sounds like "…y robot". That single mislabeling was behind
   every near-miss false trigger seen while retraining ("my robot" scoring
   0.8-0.9). Fixing it moved the worst negative from 0.86 to 0.02. Trimming
   the edge silence edge-tts pads onto every clip then let all clips fit
   again, with no data lost.

3. `train.py` — trains a small MLP classifier ((16,96) -> 64 -> 16 -> 1,
   sigmoid) on the embeddings and exports to ONNX (opset 17, input
   `[1,16,96]`, output `[1,1]`) — the same shape openWakeWord's pretrained
   wakeword models use, so it drops straight into
   `Model(wakeword_model_paths=[...])`.
   -> `hey_robot.onnx`

   Trains `N_SEEDS` models and keeps the best. Two things make that worth
   doing: training was full-batch (300 Adam steps total, barely converged and
   wildly seed-dependent — identical data gave anything between "negatives max
   0.06" and "negatives max 0.85"), now mini-batch; and runs are ranked by
   **separation margin** on the clean, non-augmented validation clips (10th
   percentile of positives minus the worst negative). Ranking by val_loss
   picks models that still fire on near-misses; ranking by error counts picks
   over-conservative models with every score bunched near the threshold.

4. `evaluate.py` — sanity-checks `hey_robot.onnx` on held-out manifest clips
   (clean) plus pure noise.

## Known limitations / how to improve

- **No real recordings**: all training data is synthetic TTS. Real speech
  (different pitch, accent, mic, room) will score lower than the synthetic
  positives — this is why `bringup.launch.py` runs this model at threshold
  0.50 rather than the 0.60 the old "max" model used (that one had 30 real
  XVF3800 recordings in it). If false negatives/positives show up on the
  robot, the highest-value next step is `record_real.py`: collect real
  "Έι ρομπότ" recordings on the actual mic (XVF3800 ch4), add them to
  `manifest.tsv` as positives (and any false-triggering phrases as negatives),
  then re-run steps 2-4.
- **Dev-machine audio loopback is not a valid test**: on this machine, audio
  played through the speakers and captured via the mic comes back ~17dB
  *below* the room noise floor (i.e. inaudible). Don't use that loopback setup
  to judge model quality; test on the actual robot hardware with real speech.
- `noise/room_noise.wav` is a single 4s room-noise sample from this dev
  machine's mic. Capturing longer/more varied real noise (robot idle, fan
  running, TV on) and mixing it into augmentation would improve robustness.

## Re-running the pipeline

```bash
cd training/wake_word_hey_robot
python3 generate_data.py      # ~616 clips via edge-tts + ffmpeg
python3 extract_features.py   # -> features.npz (~6200 examples with augmentation)
python3 train.py               # -> hey_robot.onnx
python3 evaluate.py            # sanity check
cp hey_robot.onnx ../../config/models/hey_robot.onnx
cd ../.. && colcon build --packages-select home_robot --symlink-install
```
