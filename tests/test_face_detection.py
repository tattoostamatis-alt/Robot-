"""Face detection has to actually detect a face.

‼️ There was no test file here at all, and that is precisely how the node
shipped publishing `[]` on every frame for a full core of CPU. The quantized
graph's objectness output was exactly 0.000 at every anchor, the score is
cls * obj, so nothing could ever clear any threshold — and nothing downstream
complained, because "no faces in view" is a perfectly ordinary answer.

The first test below is the one that was missing: give it a picture with a face
in it and require a face back. Everything else is detail.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_face_detection.py -q
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = f'{PKG}/home_robot/nodes/face_detection_node.py'

# The node prepends the venv (OpenCV 4.11) before importing cv2 — the system
# 4.6 cannot load this model. Do the same here so the test exercises what the
# robot runs.
_VENV = '/home/dimi/ryzenai_venv/lib/python3.12/site-packages'
if os.path.isdir(_VENV) and _VENV not in sys.path:
    sys.path.insert(0, _VENV)
import cv2  # noqa: E402

# A face that ships with ROS, so this needs no fixture of its own and no
# picture of a real person in the repo.
LENA = '/opt/ros/jazzy/share/rqt_image_view/resource/lena.png'


def _load_module():
    """Import the node with ROS stubbed — only the pure helpers are wanted."""
    stubs = {
        'rclpy': types.ModuleType('rclpy'),
        'rclpy.node': types.ModuleType('rclpy.node'),
        'sensor_msgs': types.ModuleType('sensor_msgs'),
        'sensor_msgs.msg': types.ModuleType('sensor_msgs.msg'),
        'std_msgs': types.ModuleType('std_msgs'),
        'std_msgs.msg': types.ModuleType('std_msgs.msg'),
        'cv_bridge': types.ModuleType('cv_bridge'),
    }
    stubs['rclpy'].init = lambda *a, **k: None
    stubs['rclpy'].spin = lambda *a, **k: None
    stubs['rclpy'].try_shutdown = lambda *a, **k: None
    stubs['rclpy.node'].Node = type('Node', (), {'__init__': lambda s, n: None})
    stubs['sensor_msgs.msg'].Image = object
    stubs['std_msgs.msg'].String = object
    stubs['cv_bridge'].CvBridge = object

    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('face_detection_node', NODE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


fd = _load_module()

needs_model = pytest.mark.skipif(
    not os.path.isfile(fd._MODEL_PATH),
    reason='YuNet model not fetched (the node downloads it on first start)')
needs_lena = pytest.mark.skipif(not os.path.isfile(LENA),
                                reason='no ROS test image on this box')


@pytest.fixture(scope='module')
def detector():
    cv2.setNumThreads(1)
    return cv2.FaceDetectorYN.create(
        fd._MODEL_PATH, '', (640, 480), fd._SCORE_THR, fd._NMS_THR, fd._TOP_K)


def _face_crop():
    lena = cv2.imread(LENA)
    h, w = lena.shape[:2]
    return lena[int(h * .25):int(h * .85), int(w * .25):int(w * .80)]


def _scene(face_px):
    """A 640x480 frame with one face `face_px` tall, centred on flat grey."""
    canvas = np.full((480, 640, 3), 120, dtype=np.uint8)
    face = _face_crop()
    scale = face_px / face.shape[0]
    small = cv2.resize(face, (int(face.shape[1] * scale), face_px))
    fh, fw = small.shape[:2]
    y, x = (480 - fh) // 2, (640 - fw) // 2
    canvas[y:y + fh, x:x + fw] = small
    return canvas


# ── the test that was missing ────────────────────────────────────────────────

@needs_model
@needs_lena
def test_a_picture_with_a_face_in_it_produces_a_face(detector):
    """‼️ THE regression. The old int8 graph returned [] here, forever, and
    nothing in the repo would have noticed."""
    _n, raw = detector.detect(_scene(160))
    faces = fd.to_json(raw)
    assert faces, 'no face detected in an image that plainly contains one'
    assert faces[0]['score'] >= fd._SCORE_THR


@needs_model
@needs_lena
def test_it_still_sees_a_face_across_a_room(detector):
    """70 px of a 480 px frame is a person at roughly 2 m — the distance
    someone stands at to talk to the robot. Detection has to survive there, or
    the input size has been cut too far."""
    _n, raw = detector.detect(_scene(70))
    assert fd.to_json(raw), 'a face at ~2 m is no longer detected'


@needs_model
def test_flat_grey_produces_no_faces(detector):
    """The other half: a detector that answers "face" to everything is as
    useless as one that never does."""
    _n, raw = detector.detect(np.full((480, 640, 3), 120, dtype=np.uint8))
    assert fd.to_json(raw) == []


@needs_model
@needs_lena
def test_the_box_lands_on_the_face(detector):
    """A box in the wrong place still counts as a detection everywhere else in
    the stack, and face_identity would then embed a crop of the wall."""
    _n, raw = detector.detect(_scene(160))
    box = fd.to_json(raw)[0]
    cx = (box['x1'] + box['x2']) / 2
    cy = (box['y1'] + box['y2']) / 2
    assert 220 < cx < 420, f'face centre x={cx:.0f} is not near the middle'
    assert 140 < cy < 340, f'face centre y={cy:.0f} is not near the middle'


# ── the JSON contract with face_identity and the dashboard ───────────────────

def _row(x, y, w, h, score=0.9):
    """One FaceDetectorYN row: box, five landmark pairs, score."""
    return [x, y, w, h,
            x + 10, y + 10, x + 30, y + 10,     # right eye, left eye
            x + 20, y + 20,                      # nose
            x + 12, y + 30, x + 28, y + 30,      # mouth corners
            score]


def test_json_carries_corners_not_width_and_height():
    face = fd.to_json([_row(100, 50, 40, 60)])[0]
    assert (face['x1'], face['y1']) == (100, 50)
    assert (face['x2'], face['y2']) == (140, 110)


def test_json_carries_exactly_five_landmarks():
    """face_identity_node._align returns None for anything else, so the face is
    silently dropped from recognition rather than reported as broken."""
    face = fd.to_json([_row(100, 50, 40, 60)])[0]
    assert len(face['landmarks']) == 5
    assert all(set(p) == {'x', 'y'} for p in face['landmarks'])


def test_landmark_order_is_passed_through_untouched():
    """‼️ SFace's alignment template is built in YuNet's order (right eye, left
    eye, nose, right mouth, left mouth). Reordering still yields embeddings —
    they just stop telling people apart, with no error anywhere."""
    face = fd.to_json([_row(0, 0, 40, 60)])[0]
    assert [(p['x'], p['y']) for p in face['landmarks']] == \
        [(10, 10), (30, 10), (20, 20), (12, 30), (28, 30)]


def test_the_score_survives():
    assert fd.to_json([_row(0, 0, 10, 10, 0.834)])[0]['score'] == 0.83


def test_no_faces_is_an_empty_list_not_a_crash():
    assert fd.to_json(None) == []
    assert fd.to_json([]) == []


def test_boxes_are_scaled_back_to_the_full_frame():
    """Detection may run on a resized copy; every consumer indexes the original
    image, so coordinates that stay in the resized frame point at the wrong
    pixels — a bug that looks like bad detection, not bad arithmetic."""
    face = fd.to_json([_row(100, 50, 40, 60)], 2.0, 1.5)[0]
    assert (face['x1'], face['y1'], face['x2'], face['y2']) == (200, 75, 280, 165)
    assert face['landmarks'][0] == {'x': 220, 'y': 90}


# ── the model has to be there, and say so when it is not ─────────────────────

def test_an_existing_model_is_not_downloaded_again(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'x')
    assert fd.ensure_model(str(path), 'http://invalid.invalid/nope') is True


def test_a_failed_download_reports_false_and_leaves_no_stub(tmp_path):
    """A zero-byte model would load as a broken detector instead of a missing
    one, which is the harder failure to read."""
    path = tmp_path / 'model.onnx'
    assert fd.ensure_model(str(path), 'http://invalid.invalid/nope') is False
    assert not path.exists()
    assert not (tmp_path / 'model.onnx.part').exists()


# ── the settings that made it cost a whole core ──────────────────────────────

def test_it_defaults_to_one_thread():
    """Measured: 640x480 is 43 ms wall / 43 ms CPU on one thread and 25 ms wall
    / 51 ms CPU on four. Extra threads buy latency this node has no use for."""
    src = open(NODE).read()
    assert "declare_parameter('threads', 1)" in src


def test_it_does_not_run_faster_than_a_face_moves():
    src = open(NODE).read()
    assert "declare_parameter('process_every_n', 6)" in src, \
        'back to 10 Hz — twice the CPU for detections nothing consumes faster'


def test_the_quantized_model_is_gone():
    """‼️ yunet_int8.onnx detected nothing, ever. If the node points back at it,
    it returns to burning a core to publish empty lists. Checked on the paths,
    not on the source text — the docstring names the file on purpose, so that
    whoever finds it lying in the nodes directory knows why not to use it."""
    assert 'yunet_int8' not in fd._MODEL_PATH
    assert 'yunet_int8' not in fd._MODEL_URL
