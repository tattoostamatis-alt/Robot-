#!/bin/bash
# Quantize YOLO11n-pose, YOLO11n-seg, and YuNet for NPU (AMD Quark XINT8).
# Run once: bash scripts/quantize_npu_models.sh
set -e

VENV=~/ryzenai_venv
NODES_DIR=~/robot_ws/src/home_robot/home_robot/nodes
WORK=/tmp/npu_quant
mkdir -p "$WORK"

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source "$VENV/bin/activate"

export HF_HUB_DISABLE_XET=1

echo "=== Quantizing YOLO11n-pose ==="
python3 - <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/ryzenai_venv/lib/python3.12/site-packages'))

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

    class DummyReader:
        def __init__(self, model_path, n=50):
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.inp_name = sess.get_inputs()[0].name
            self._data = [np.random.randn(1,3,640,640).astype(np.float32) for _ in range(n)]
            self._i = 0
        def get_next(self):
            if self._i >= len(self._data): return None
            d = {self.inp_name: self._data[self._i]}; self._i += 1; return d

    cfg = Config(global_quant_config=get_default_config('XINT8'))
    ModelQuantizer(cfg).quantize_model(FP_PATH, Q_PATH, DummyReader(FP_PATH))
    print(f'Pose model saved: {Q_PATH}')
else:
    print(f'Already exists: {Q_PATH}')
PYEOF

echo "=== Quantizing YOLO11n-seg ==="
python3 - <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/ryzenai_venv/lib/python3.12/site-packages'))

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

    class DummyReader:
        def __init__(self, model_path, n=50):
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.inp_name = sess.get_inputs()[0].name
            self._data = [np.random.randn(1,3,640,640).astype(np.float32) for _ in range(n)]
            self._i = 0
        def get_next(self):
            if self._i >= len(self._data): return None
            d = {self.inp_name: self._data[self._i]}; self._i += 1; return d

    cfg = Config(global_quant_config=get_default_config('XINT8'))
    ModelQuantizer(cfg).quantize_model(FP_PATH, Q_PATH, DummyReader(FP_PATH))
    print(f'Seg model saved: {Q_PATH}')
else:
    print(f'Already exists: {Q_PATH}')
PYEOF

echo "=== Downloading + Quantizing YuNet ==="
python3 - <<'PYEOF'
import os, sys, numpy as np, urllib.request
sys.path.insert(0, os.path.expanduser('~/ryzenai_venv/lib/python3.12/site-packages'))

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

    class DummyReader:
        def __init__(self, model_path, n=50):
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            inp = sess.get_inputs()[0]
            self.inp_name = inp.name
            # YuNet default input: (1, 3, 192, 320) or (1, 192, 320, 3)
            shape = [1 if isinstance(d,str) else d for d in inp.shape]
            self._data = [np.random.randn(*shape).astype(np.float32) for _ in range(n)]
            self._i = 0
        def get_next(self):
            if self._i >= len(self._data): return None
            d = {self.inp_name: self._data[self._i]}; self._i += 1; return d

    cfg = Config(global_quant_config=get_default_config('XINT8'))
    ModelQuantizer(cfg).quantize_model(FP_PATH, Q_PATH, DummyReader(FP_PATH))
    print(f'YuNet saved: {Q_PATH}')
else:
    print(f'Already exists: {Q_PATH}')
PYEOF

echo "=== All models quantized. ==="
echo "Models in: $NODES_DIR"
ls -lh "$NODES_DIR"/*_int8.onnx 2>/dev/null || true
