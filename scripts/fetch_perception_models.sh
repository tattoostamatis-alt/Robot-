#!/usr/bin/env bash
# Pull the models and Python deps the open-vocabulary detector and the sound
# event node need. Both download on first use anyway; running this ahead of time
# keeps that cost out of a live session.
#
# Nothing here is committed to the repo — these are 25-354 MB of weights, cached
# the same way the faster-whisper models already are.
#
# Usage:  scripts/fetch_perception_models.sh
set -e

echo "→ open-vocabulary detection (use_open_vocab:=true)"

# ‼️ YOLO-World's set_classes() encodes the class names with CLIP ViT-B/32.
# Without the `clip` module ultralytics tries to pip-install it at runtime and
# fails under PEP 668 ("externally-managed-environment"), so set_classes throws
# "No module named 'clip'" and the detector silently never hunts for anything.
# clip-anytorch is the maintained PyPI packaging of OpenAI's CLIP.
if python3 -c "import clip" 2>/dev/null; then
  echo "  ✔ clip already installed"
else
  echo "  → installing clip-anytorch (user site)"
  pip install --user --break-system-packages --no-input clip-anytorch
fi

# yolov8s-worldv2.pt (~25 MB) and the CLIP weights (~354 MB). The CLIP download
# is the slow one and it happens inside the first set_classes() call, which is
# why open_vocab_detector warms it up at startup rather than on first request.
python3 - <<'PY'
import os
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
from ultralytics import YOLOWorld
m = YOLOWorld('yolov8s-worldv2.pt')
m.set_classes(['keys'])          # forces the CLIP download + encode
print('  ✔ YOLO-World + CLIP cached')
PY

echo "→ sound events (use_sound_events:=true)"
# YAMNet as ONNX (~15 MB): 521 AudioSet classes, waveform in. The waveform-input
# export matters — the mel-patch exports would need YAMNet's exact frontend
# reimplemented, and any mismatch quietly degrades every classification.
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import hf_hub_download
for f in ('yamnet.onnx', 'yamnet_class_map.csv'):
    hf_hub_download('zeropointnine/yamnet-onnx', f)
print('  ✔ YAMNet cached')
PY

echo "→ face identity (use_face_detection:=true)"
# SFace (~37 MB): 128-d embeddings from a 112x112 crop aligned with the SAME
# five landmarks YuNet already publishes. That pairing is why alignment works.
HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download('opencv/face_recognition_sface',
                'face_recognition_sface_2021dec.onnx')
print('  ✔ SFace cached')
PY

echo "✔ done — all features can now start offline"
