"""Tests for the dashboard's 3D point-cloud stream (web_dashboard_node).

The camera publishes 148 815 points x 20 bytes at 28 Hz — 3 MB a message. What
reaches the browser is subsampled to ~4000 points and quantised to int16
millimetres + RGB, which is nine bytes a point. Both halves of that conversion
are hand-rolled binary, so both are tested here: the Python encoder directly,
and the browser's decoder by running the real JS through node.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_dashboard_cloud.py -q
"""
import base64
import json
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

_SRC = (Path(__file__).resolve().parents[1]
        / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()

_ENC = re.search(r'^    CLOUD_MAX_POINTS = .*?^    def _cb_cloud.*?'
                 r'(?=\n\n    def )', _SRC, re.M | re.S)
assert _ENC, 'could not extract _cb_cloud from web_dashboard_node.py'

# PointCloud2 is only an annotation on the callback, but the class body is
# evaluated, so it has to resolve to something.
_ns = {'np': np, 'time': time, 'base64': base64, 'PointCloud2': object}
exec(compile('class Dash:\n' + _ENC.group(0),  # noqa: S102 - test extraction
             'dash_cloud', 'exec'), _ns)
Dash = _ns['Dash']

POINT_STEP = 20          # what realsense actually publishes: x y z _ rgb


class FakeState:
    def __init__(self):
        self.sent = []

    def broadcast(self, msg, remember=True):
        self.sent.append(msg)


class FakeMsg:
    def __init__(self, xyz, bgr):
        n = len(xyz)
        buf = np.zeros((n, POINT_STEP), dtype=np.uint8)
        buf[:, 0:12] = np.asarray(xyz, dtype='<f4').view(np.uint8).reshape(n, 12)
        buf[:, 16:19] = np.asarray(bgr, dtype=np.uint8)
        self.data = buf.tobytes()
        self.width, self.height = n, 1
        self.point_step = POINT_STEP

        class H:
            frame_id = 'camera_depth_optical_frame'
        self.header = H()


def encode(xyz, bgr, viewers=True, fusion_viewers=False):
    d = Dash()
    d._state = FakeState()
    d._cloud_ws = {'a browser'} if viewers else set()
    # The Sensor fusion tab shares this one decode but wants only a top-down
    # profile out of it, never the 3D payload — see _cam_profile.
    d._fuse_cam_ws = {'a fusion tab'} if fusion_viewers else set()
    d._cam_profile = lambda *a, **k: None   # its TF/profile machinery is not here
    d._cloud_last = 0.0
    d._cb_cloud(FakeMsg(xyz, bgr))
    return d._state.sent


def decode(msg):
    raw = base64.b64decode(msg['data'])
    out = []
    for i in range(msg['n']):
        x, y, z = struct.unpack_from('<hhh', raw, i * 9)
        out.append(((x / 1000, y / 1000, z / 1000),
                    (raw[i*9+6], raw[i*9+7], raw[i*9+8])))
    return out


# ── the stream only runs for someone who is watching ─────────────────────────

def test_nothing_is_sent_with_no_viewers():
    """150 kB/s to a phone that is not on the 3D tab is the whole reason the
    browser has to ask for this stream."""
    assert encode([(0.1, 0.2, 1.0)], [(1, 2, 3)], viewers=False) == []


def test_a_viewer_gets_a_cloud():
    assert len(encode([(0.1, 0.2, 1.0)], [(1, 2, 3)])) == 1


def test_a_fusion_viewer_alone_gets_no_3d_payload():
    """The Sensor fusion tab needs the same decode but not the 150 kB/s.

    It asks for the cloud to be produced (it wants the top-down profile out of
    it), so the callback runs — but sending it the full 3D payload as well
    would put the 3D tab's bandwidth on a tab that never draws it.
    """
    assert encode([(0.1, 0.2, 1.0)], [(1, 2, 3)],
                  viewers=False, fusion_viewers=True) == []


def test_a_fusion_viewer_does_not_stop_the_3d_tab():
    both = encode([(0.1, 0.2, 1.0)], [(1, 2, 3)],
                  viewers=True, fusion_viewers=True)
    assert len(both) == 1


def test_the_rate_is_throttled_below_the_camera():
    """The camera runs at 28 Hz; forwarding all of it would be 4 MB/s."""
    d = Dash()
    d._state = FakeState()
    d._cloud_ws = {'x'}
    d._fuse_cam_ws = set()
    d._cam_profile = lambda *a, **k: None
    d._cloud_last = 0.0
    for _ in range(10):                    # ten frames back to back
        d._cb_cloud(FakeMsg([(0.0, 0.0, 1.0)], [(1, 2, 3)]))
    assert len(d._state.sent) == 1, 'throttle is not holding'
    assert Dash.CLOUD_PERIOD >= 0.2


# ── the numbers must survive the round trip ──────────────────────────────────

def test_coordinates_survive_to_the_millimetre():
    pts = [(0.123, -0.456, 1.789)]
    got = decode(encode(pts, [(0, 0, 0)])[0])
    (x, y, z), _ = got[0]
    assert (round(x, 3), round(y, 3), round(z, 3)) == (0.123, -0.456, 1.789)


def test_negative_coordinates_are_not_mangled():
    """int16 by hand: a missing sign extension turns -1 m into +64 m."""
    got = decode(encode([(-1.5, -2.25, 3.0)], [(0, 0, 0)])[0])
    (x, y, z), _ = got[0]
    assert x == pytest.approx(-1.5) and y == pytest.approx(-2.25)


def test_colour_is_converted_from_bgr_to_rgb():
    """realsense packs the rgb float as B G R; sending it raw makes every
    person in the room blue."""
    got = decode(encode([(0.0, 0.0, 1.0)], [(10, 20, 30)])[0])   # B=10 G=20 R=30
    assert got[0][1] == (30, 20, 10)


def test_nan_points_are_dropped_not_sent_as_zero():
    pts = [(float('nan'), 0.0, 1.0), (0.1, 0.1, 1.0)]
    msg = encode(pts, [(1, 1, 1), (2, 2, 2)])[0]
    assert msg['n'] == 1


def test_a_cloud_of_only_nan_sends_nothing():
    assert encode([(float('nan'),) * 3], [(1, 1, 1)]) == []


def test_large_clouds_are_subsampled_and_report_the_original_size():
    n = 50000
    xyz = [(0.0, 0.0, 1.0)] * n
    msg = encode(xyz, [(1, 2, 3)] * n)[0]
    assert msg['n'] <= Dash.CLOUD_MAX_POINTS
    assert msg['total'] == n, 'the browser needs the true count for its badge'
    assert len(base64.b64decode(msg['data'])) == msg['n'] * 9


def test_payload_is_nine_bytes_per_point():
    msg = encode([(0.0, 0.0, 1.0)] * 10, [(1, 2, 3)] * 10)[0]
    assert len(base64.b64decode(msg['data'])) == 10 * 9


# ── the browser's own decoder, run for real ──────────────────────────────────

JS_DECODE = re.search(r'function onCloud\(m\)\{(.*?)\n\}', _SRC, re.S)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_the_browser_decoder_agrees_with_the_encoder():
    """The JS reads the payload out of a *binary string* from atob(), one
    charCode at a time, and sign-extends int16 by hand. Nothing about that is
    obvious, so run the real code and compare it against Python."""
    assert JS_DECODE, 'onCloud() changed shape'
    pts = [(0.5, -0.25, 2.0), (-1.234, 0.0, 0.75), (0.0, 3.21, 9.99)]
    bgr = [(10, 20, 30), (200, 100, 50), (0, 255, 128)]
    msg = encode(pts, bgr)[0]

    body = JS_DECODE.group(1)
    # Stub the two lines that touch the page; keep every byte of the maths.
    body = re.sub(r"\$\('cloud-info'\).*?;", '', body, flags=re.S)
    body = body.replace('cloudDraw();', '')
    script = f'''
const m = {json.dumps(msg)};
function atob(s){{ return Buffer.from(s, 'base64').toString('binary'); }}
let cloudPts, cloudRGB;
{body}
console.log(JSON.stringify({{xyz: Array.from(cloudPts), rgb: Array.from(cloudRGB)}}));
'''
    out = subprocess.run(['node', '-e', script], capture_output=True, text=True,
                         timeout=30)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    for i, (x, y, z) in enumerate(pts):
        assert got['xyz'][i*3]     == pytest.approx(x, abs=1e-3)
        assert got['xyz'][i*3 + 1] == pytest.approx(y, abs=1e-3)
        assert got['xyz'][i*3 + 2] == pytest.approx(z, abs=1e-3)
    for i, (b, g, r) in enumerate(bgr):
        assert (got['rgb'][i*3], got['rgb'][i*3+1], got['rgb'][i*3+2]) == (r, g, b)


# ── a map switch must not silently undo the runtime configuration ────────────
# ‼️ 2026-08-01, for real: a click on "Ενεργοποίηση" restarted the stack via
# `robot max map:=malou2 use_perception:=true` — with no llm_backend. The LLM
# fell back to the NPU default, FastFlowLM started again, and 4.7 GB that had
# just been freed came straight back. Nothing errored; the only visible symptom
# was the dashboard briefly 500ing while two of them fought over port 8080.

_DASH_SRC = _SRC


def test_map_switch_carries_perception_and_backend():
    """Both settings, or a restart reverts to `robot max` defaults."""
    body = _DASH_SRC.split('async def maps_action')[1].split('\n@app')[0]
    assert 'use_perception:=true' in body, 'perception is dropped on switch'
    assert 'llm_backend:=' in body, 'LLM backend is dropped on switch'


def test_the_backend_is_read_not_assumed():
    """Hardcoding a backend would pin every future switch to today's choice."""
    body = _DASH_SRC.split('async def maps_action')[1].split('\n@app')[0]
    assert 'ros_node.llm_backend' in body
    assert "llm_backend:=gemini'" not in body, 'backend is hardcoded'


def test_lemonade_is_not_passed_explicitly():
    """It is the launch default; passing it would also start FastFlowLM even
    when `robot max` was told to skip it."""
    body = _DASH_SRC.split('async def maps_action')[1].split('\n@app')[0]
    assert "!= 'lemonade'" in body


def test_reading_the_backend_does_not_block_the_event_loop():
    """It shells out to `ros2 param get`, which takes a second or two."""
    body = _DASH_SRC.split('async def maps_action')[1].split('\n@app')[0]
    assert 'await asyncio.to_thread(ros_node.llm_backend)' in body


# ── the camera has to be producing a cloud at all ───────────────────────────
# 2026-08-02: the 3D tab said "αναμονή για νέφος σημείων…" forever on a default
# `robot max`. Everything above this line worked — the encoder, the decoder, the
# gating — but localize.launch.py starts the lean depth stream with
# pointcloud.enable:=false, so /camera/camera/depth/color/points was advertised
# and never published. The tab could not have worked on any default launch.
# Flipping the parameter at runtime brings the topic up at ~18 Hz, so the tab
# now turns the filter on while someone is watching and off again after.

import re as _re
from pathlib import Path as _Path

_NODE_SRC = (_Path(__file__).resolve().parents[1]
             / 'home_robot' / 'nodes' / 'web_dashboard_node.py').read_text()


def _node_method(name):
    m = _re.search(r'\n    def %s\(self.*?(?=\n    def )' % name, _NODE_SRC, _re.S)
    assert m, f'{name}() is gone or changed shape'
    return m.group(0)


def test_the_tab_switches_the_cameras_pointcloud_on():
    assert '_set_camera_pointcloud' in _NODE_SRC, \
        'nothing enables pointcloud.enable — the 3D tab shows an empty scene'


def test_it_sets_the_realsense_parameter():
    body = _node_method('_set_camera_pointcloud')
    assert "'pointcloud.enable'" in body, 'the wrong parameter is being set'
    assert '/camera/camera/set_parameters' in _NODE_SRC, \
        'the parameter client no longer points at the camera'


def test_it_follows_whether_anyone_is_watching():
    """On when the first viewer arrives, off when the last leaves.

    Two sets of viewers now, since the Sensor fusion tab needs the cloud too:
    the last one to leave is what switches the camera back down.
    """
    disp = _node_method('dispatch')
    assert 'self._cloud_ws or self._fuse_cam_ws' in disp, \
        "the 'cloud' message must drive the camera parameter"


def test_a_closed_tab_turns_it_off():
    """A browser closed on the 3D pane never sends its 'off'."""
    body = _node_method('release_client')
    assert 'self._cloud_ws or self._fuse_cam_ws' in body, \
        'a vanished viewer would leave the camera building clouds for nobody'
    assert '_fuse_ws.discard(client)' in body and '_fuse_cam_ws.discard(client)' in body, \
        'a tab closed on the fusion pane never sends its off either'


def test_it_never_switches_off_a_pointcloud_it_did_not_switch_on():
    """‼️ Nav2's voxel_layer feeds on this topic (nav2_params.yaml
    `depth_camera`), and since 2026-08-05 the launch files start the camera
    with pointcloud.enable:=true for exactly that reason.

    If the dashboard still turned it off when the last 3D viewer left, it would
    blind the costmap to every obstacle the LiDAR's flat slice misses — with
    nothing in any log to say so. It must probe the parameter at startup and
    keep its hands off when somebody else already owns it.
    """
    body = _node_method('_set_camera_pointcloud')
    assert 'if not self._cloud_param_own' in body, \
        'the dashboard would switch off a pointcloud Nav2 depends on'
    probe = _node_method('_probe_cloud_owner')
    assert 'GetParameters' in _NODE_SRC and 'pointcloud.enable' in probe, \
        'nothing reads the current value, so ownership is guesswork'
    assert '_cloud_param_own = False' in probe, \
        'the probe never actually gives up ownership'


def test_it_does_not_spam_the_camera_with_the_same_value():
    body = _node_method('_set_camera_pointcloud')
    assert 'if on == self._cloud_param_on' in body, \
        'every tab switch would fire another set_parameters call'


def test_it_survives_the_camera_being_absent():
    """localize can run without the D435 at all; that is not an error here."""
    body = _node_method('_set_camera_pointcloud')
    assert 'service_is_ready()' in body, \
        'a missing camera must not raise out of the websocket handler'


def test_it_does_not_block_the_event_loop():
    body = _node_method('_set_camera_pointcloud')
    assert 'call_async' in body and 'call(' not in body.replace('call_async', ''), \
        'a synchronous service call here would stall the dashboard socket'
