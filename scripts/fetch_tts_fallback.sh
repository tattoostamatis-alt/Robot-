#!/usr/bin/env bash
# Download the offline Greek voice used when edge-tts fails.
#
# edge-tts is a free, unauthenticated endpoint that fails ~20% of the time
# (measured 2026-07-26). Without a local voice the robot goes silent on those
# answers, which reads as "it isn't responding" even though STT, the LLM and
# the reply all worked. See home_robot/tts_fallback.py.
#
#   ./scripts/fetch_tts_fallback.sh
#
# ~61 MB, gitignored (*.onnx), CPU-only, ~0.1 s per utterance.
# Requires the piper runtime:
#   pip install --user --break-system-packages piper-tts

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/models/piper"
VOICE="el_GR-joy-medium"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/el/el_GR/joy/medium"

mkdir -p "$DIR"
# HF_HUB_DISABLE_XET: Xet-backed downloads stall at 0 progress on this network.
export HF_HUB_DISABLE_XET=1

for f in "$VOICE.onnx" "$VOICE.onnx.json"; do
  if [ -s "$DIR/$f" ]; then
    echo "already have $f"
    continue
  fi
  echo "downloading $f ..."
  curl -fL --progress-bar -o "$DIR/$f" "$BASE/$f"
done

echo
echo "Voice in $DIR"
python3 - "$DIR/$VOICE.onnx" <<'PY'
import sys
from home_robot.tts_fallback import PiperFallback
out = PiperFallback(sys.argv[1]).synthesize('Η εφεδρική φωνή δουλεύει.')
if out is None:
    sys.exit('FAILED — voice did not synthesise')
audio, rate = out
print(f'OK — {len(audio)/rate:.1f}s of audio at {rate} Hz')
PY
