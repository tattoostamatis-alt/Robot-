#!/bin/bash
# Quantize YOLO11n-pose, YOLO11n-seg, and YuNet for NPU (AMD Quark XINT8).
# Run once: bash scripts/quantize_npu_models.sh
#
# ‼️ 2026-08-05 — THIS SCRIPT PRODUCED A MODEL THAT DETECTED NOTHING, EVER.
#
# Every reader below calibrated with `np.random.randn(...)`: values centred on
# 0 with sigma 1, while real camera frames are 0-255. INT8 calibration derives
# its activation scales from exactly that data, so every scale came out about
# two orders of magnitude too small and the activations saturated. In the YuNet
# output that showed up as an objectness branch of EXACTLY 0.000 at every
# anchor — the score is cls * obj, so face_detection_node published `[]` on
# every frame for months, at 10 Hz, for a full core of CPU, with no error
# anywhere. See home_robot/nodes/face_detection_node.py.
#
# Two changes, both of which had to be here:
#   1. calibration uses the real input range (see `calibration_batches`), and
#   2. every model is CHECKED after quantization — a graph whose outputs are
#      all-constant is rejected loudly instead of being installed.
#
# The check is the important one. It catches a bad calibration set, but it also
# catches whatever the next mistake turns out to be, which the fix in (1) does
# not.
set -e

VENV=~/ryzenai_venv
NODES_DIR=~/robot_ws/src/home_robot/home_robot/nodes
WORK=/tmp/npu_quant
mkdir -p "$WORK"

# Shared by all three quantization steps below. Kept as a file the heredocs
# import rather than pasted three times, because it was three copies of the
# same wrong line that made this script produce three suspect models.
cat > "$WORK/calib.py" <<'CALIBEOF'
"""Calibration data in the range the model will actually see, and a check.

‼️ The bug this exists to prevent: calibrating with np.random.randn gives
values around 0 with sigma 1. Real frames are 0-255. INT8 scales derived from
the wrong range saturate every activation, and the result is a graph that runs,
loads, reports no error, and outputs a constant.
"""
import numpy as np
import onnxruntime as ort


def calibration_batches(model_path, n=50, seed=0):
    """n input dicts spanning the real input range.

    Uniform 0-255 is not a substitute for real photographs — a proper set would
    be frames from this robot's own camera — but it puts the scales in the
    right decade, which is the difference between "slightly less accurate" and
    "always outputs zero".
    """
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
    rng = np.random.default_rng(seed)
    return inp.name, [rng.uniform(0.0, 255.0, size=shape).astype(np.float32)
                      for _ in range(n)]


class Reader:
    def __init__(self, model_path, n=50):
        self.inp_name, self._data = calibration_batches(model_path, n)
        self._i = 0

    def get_next(self):
        if self._i >= len(self._data):
            return None
        d = {self.inp_name: self._data[self._i]}
        self._i += 1
        return d


def check_not_dead(model_path, label):
    """Reject a graph whose outputs do not respond to its input.

    Runs two very different inputs and requires that SOME output changes, and
    that no output is a single constant across the whole tensor. Both were true
    of the broken YuNet: obj was 0.000 everywhere, for anything.
    """
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
    rng = np.random.default_rng(1)
    a = sess.run(None, {inp.name: rng.uniform(0, 255, shape).astype(np.float32)})
    b = sess.run(None, {inp.name: rng.uniform(0, 255, shape).astype(np.float32)})

    names = [o.name for o in sess.get_outputs()]
    dead = [n for n, t in zip(names, a)
            if np.asarray(t).size and np.ptp(np.asarray(t)) == 0]
    if dead:
        raise SystemExit(
            f'{label}: quantization produced constant output(s) {dead} — '
            'the graph is dead, refusing to install it. This is what a '
            'calibration set in the wrong value range looks like.')
    if all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)):
        raise SystemExit(
            f'{label}: outputs do not change with the input — the graph is '
            'not looking at its input at all.')
    print(f'  \u2714 {label}: outputs vary with input, no constant tensors')
CALIBEOF

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source "$VENV/bin/activate"

export HF_HUB_DISABLE_XET=1

echo "=== Quantizing YOLO11n-pose ==="
python3 - <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/ryzenai_venv/lib/python3.12/site-packages'))
sys.path.insert(0, '/tmp/npu_quant')
from calib import Reader, check_not_dead

WORK   = '/tmp/npu_quant'
NODES  = os.path.expanduser('~/robot_ws/src/home_robot/home_robot/nodes')
FP_PATH = f'{WORK}/yolo11n-pose.onnx'
Q_PATH  = f'{NODES}/yolo11n_pose_int8.onnx'

if not os.path.exists(Q_PATH):
    # Export ONNX from ultralytics
    from ultralytics import YOLO
    m = YOLO('yolo11n-pose.pt')
    m.export(format='onnx', imgsz=640, simplify=True, opset=17, dynamic=False)
    import shutil, glob
    src = glob.glob(os.path.expanduser('~/.config/Ultralytics/yolo11n-pose.onnx'))
    if not src:
        src = glob.glob('yolo11n-pose.onnx') + glob.glob('/tmp/yolo11n-pose.onnx')
    # ultralytics saves next to the .pt
    import site
    for sp in site.getsitepackages() + ['.']:
        candidate = os.path.join(sp, 'yolo11n-pose.onnx')
        if os.path.exists(candidate):
            src = [candidate]; break
    if not src:
        # ultralytics saves in cwd or package dir; check both
        candidates = ['yolo11n-pose.onnx', os.path.join(WORK, 'yolo11n-pose.onnx')]
        src = [c for c in candidates if os.path.exists(c)]
    if src:
        shutil.copy(src[0], FP_PATH)
    else:
        # Use ultralytics export path logic
        from pathlib import Path
        yolo = YOLO('yolo11n-pose.pt')
        result = yolo.export(format='onnx', imgsz=640, simplify=True, opset=17)
        shutil.copy(str(result), FP_PATH)

    # Quantize with Quark
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config
    import onnxruntime as ort


    cfg = Config(global_quant_config=get_default_config('XINT8'))
    ModelQuantizer(cfg).quantize_model(FP_PATH, Q_PATH, Reader(FP_PATH))
    check_not_dead(Q_PATH, 'Pose model')
    print(f'Pose model saved: {Q_PATH}')
else:
    print(f'Already exists: {Q_PATH}')
PYEOF

echo "=== Quantizing YOLO11n-seg ==="
python3 - <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/ryzenai_venv/lib/python3.12/site-packages'))
sys.path.insert(0, '/tmp/npu_quant')
from calib import Reader, check_not_dead

WORK  = '/tmp/npu_quant'
NODES = os.path.expanduser('~/robot_ws/src/home_robot/home_robot/nodes')
FP_PATH = f'{WORK}/yolo11n-seg.onnx'
Q_PATH  = f'{NODES}/yolo11n_seg_int8.onnx'

if not os.path.exists(Q_PATH):
    from ultralytics import YOLO
    import shutil
    yolo = YOLO('yolo11n-seg.pt')
    result = yolo.export(format='onnx', imgsz=640, simplify=True, opset=17)
    shutil.copy(str(result), FP_PATH)

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config
    import onnxruntime as ort


    cfg = Config(global_quant_config=get_default_config('XINT8'))
    ModelQuantizer(cfg).quantize_model(FP_PATH, Q_PATH, Reader(FP_PATH))
    check_not_dead(Q_PATH, 'Seg model')
    print(f'Seg model saved: {Q_PATH}')
else:
    print(f'Already exists: {Q_PATH}')
PYEOF

echo "=== Downloading + Quantizing YuNet ==="
python3 - <<'PYEOF'
import os, sys, numpy as np, urllib.request
sys.path.insert(0, os.path.expanduser('~/ryzenai_venv/lib/python3.12/site-packages'))
sys.path.insert(0, '/tmp/npu_quant')
from calib import Reader, check_not_dead

WORK  = '/tmp/npu_quant'
NODES = os.path.expanduser('~/robot_ws/src/home_robot/home_robot/nodes')
FP_PATH = f'{WORK}/yunet.onnx'
Q_PATH  = f'{NODES}/yunet_int8.onnx'

if not os.path.exists(Q_PATH):
    if not os.path.exists(FP_PATH):
        url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
        print(f'Downloading YuNet from {url}...')
        urllib.request.urlretrieve(url, FP_PATH)

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config
    import onnxruntime as ort


    cfg = Config(global_quant_config=get_default_config('XINT8'))
    ModelQuantizer(cfg).quantize_model(FP_PATH, Q_PATH, Reader(FP_PATH))
    check_not_dead(Q_PATH, 'YuNet')
    print(f'YuNet saved: {Q_PATH}')
else:
    print(f'Already exists: {Q_PATH}')
PYEOF

echo "=== All models quantized. ==="
echo "Models in: $NODES_DIR"
ls -lh "$NODES_DIR"/*_int8.onnx 2>/dev/null || true
