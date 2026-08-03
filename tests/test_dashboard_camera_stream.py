"""The MJPEG camera stream must never go silent.

Live symptom (2026-08-03): the Κάμερα tab showed the picture for about three
minutes and then turned into a broken-image '?', while the 3D tab — fed by a
different topic from the same D435 — kept working. That rules the camera out
and points at the stream itself.

The old loop was:

    while True:
        jpg = state.camera_jpg
        if jpg:
            yield ...frame...
        await asyncio.sleep(0.04)

so the moment frames stopped arriving the HTTP response went completely quiet
while staying open. A browser waits a few minutes on a stalled response and
then aborts it and draws the broken-image icon — which is exactly the reported
timing. The same loop also re-sent an unchanged buffer 25 times a second,
pushing ~1.2 MB/s per viewer for a camera publishing at 5.

These tests drive the real endpoint through Starlette's test client with a
fake State, so they exercise the shipped generator rather than a copy.
"""
import asyncio
import importlib.util
import os
import sys
import time
import types

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = os.path.join(PKG, 'home_robot', 'nodes', 'web_dashboard_node.py')


def _load_module():
    """Import web_dashboard_node without rclpy having to be initialised.

    The module builds its FastAPI app at import time but only touches ROS
    inside DashboardNode, so importing is safe; it is skipped rather than
    failed if the web stack is not installed on this machine.
    """
    pytest.importorskip('fastapi')
    pytest.importorskip('cv2')
    spec = importlib.util.spec_from_file_location('wdn_undertest', NODE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['wdn_undertest'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def wdn():
    return _load_module()


class FakeState:
    """Just the fields the camera endpoint reads."""

    def __init__(self):
        self.camera_jpg = None
        self.camera_at = 0.0
        self.camera_seq = 0

    def push(self, jpg=b'\xff\xd8frame\xff\xd9'):
        self.camera_jpg = jpg
        self.camera_at = time.monotonic()
        self.camera_seq += 1


def _frames(payload: bytes):
    """Split a multipart body into its JPEG payloads."""
    return [p.split(b'\r\n\r\n', 1)[1].rstrip(b'\r\n')
            for p in payload.split(b'--frame') if b'\r\n\r\n' in p]


async def _collect(gen, seconds, clock_steps=None):
    """Run the async generator for a bounded number of yields."""
    out = []
    task_deadline = time.monotonic() + seconds
    async for chunk in gen:
        out.append(chunk)
        if time.monotonic() > task_deadline or len(out) > 400:
            break
    return out


def _make_request(disconnected=False):
    """Minimal Starlette-ish Request: the endpoint only calls is_disconnected."""
    req = types.SimpleNamespace()
    req.cookies = {}

    async def is_disconnected():
        return disconnected

    req.is_disconnected = is_disconnected
    return req


# ── the regression itself ───────────────────────────────────────────────────

def test_a_silent_camera_still_produces_output(wdn, monkeypatch):
    """The bug: no frames in, nothing out, browser aborts after minutes.

    With no camera data at all the stream must still emit something promptly,
    so the connection stays alive and the user sees a message rather than a
    broken image.
    """
    state = FakeState()                       # never pushed: camera never came up
    monkeypatch.setattr(wdn, 'state', state)
    monkeypatch.setattr(wdn, '_CAM_KEEPALIVE_S', 0.05)
    monkeypatch.setattr(wdn, '_CAM_STALE_REPEAT_S', 0.1)
    monkeypatch.setattr(wdn, 'NO_AUTH', True)

    async def run():
        resp = await wdn.camera(_make_request(), t='')
        chunks = []
        async for c in resp.body_iterator:
            chunks.append(c)
            if len(chunks) >= 2:
                break
        return chunks

    chunks = asyncio.run(asyncio.wait_for(run(), timeout=10))
    assert chunks, 'the stream sent nothing at all — the browser will time out'
    frames = _frames(b''.join(chunks))
    assert frames and frames[0].startswith(b'\xff\xd8'), \
        'keep-alive payload is not a JPEG'


def test_a_camera_that_dies_mid_stream_keeps_the_connection_fed(wdn, monkeypatch):
    """Frames arrive, then stop — the reported scenario."""
    state = FakeState()
    state.push()
    monkeypatch.setattr(wdn, 'state', state)
    monkeypatch.setattr(wdn, '_CAM_KEEPALIVE_S', 0.05)
    monkeypatch.setattr(wdn, '_CAM_STALE_S', 0.1)
    monkeypatch.setattr(wdn, 'NO_AUTH', True)

    async def run():
        resp = await wdn.camera(_make_request(), t='')
        chunks = []
        async for c in resp.body_iterator:
            chunks.append(c)
            if len(chunks) >= 3:      # first real frame, then keep-alives
                break
        return chunks

    chunks = asyncio.run(asyncio.wait_for(run(), timeout=10))
    assert len(chunks) >= 3, (
        'the stream went quiet once frames stopped — this is the bug that '
        'turns the tab into a "?" after a few minutes')


def test_an_unchanged_frame_is_not_resent_at_25fps(wdn, monkeypatch):
    """Bandwidth: one push must not become dozens of identical frames."""
    state = FakeState()
    state.push()
    monkeypatch.setattr(wdn, 'state', state)
    monkeypatch.setattr(wdn, 'NO_AUTH', True)
    # Keep-alive far away, staleness far away: within this window the only
    # thing that could produce output is a genuinely new frame.
    monkeypatch.setattr(wdn, '_CAM_KEEPALIVE_S', 30.0)
    monkeypatch.setattr(wdn, '_CAM_STALE_S', 30.0)

    async def run():
        resp = await wdn.camera(_make_request(), t='')
        chunks = []

        async def pump():
            async for c in resp.body_iterator:
                chunks.append(c)

        task = asyncio.ensure_future(pump())
        await asyncio.sleep(0.5)          # ~12 loop iterations at 25 fps
        task.cancel()
        return chunks

    chunks = asyncio.run(asyncio.wait_for(run(), timeout=10))
    assert len(chunks) == 1, (
        f'sent {len(chunks)} copies of one frame — the stream is re-sending '
        'an unchanged buffer instead of waiting for a new one')


def test_a_disconnected_browser_stops_the_loop(wdn, monkeypatch):
    """Otherwise every closed tab leaks a generator that runs for ever."""
    state = FakeState()
    state.push()
    monkeypatch.setattr(wdn, 'state', state)
    monkeypatch.setattr(wdn, 'NO_AUTH', True)

    async def run():
        resp = await wdn.camera(_make_request(disconnected=True), t='')
        return [c async for c in resp.body_iterator]

    chunks = asyncio.run(asyncio.wait_for(run(), timeout=10))
    assert chunks == [], 'the stream kept producing for a client that had gone'


def test_the_placeholder_says_what_is_wrong(wdn):
    """A black rectangle is only marginally better than a '?'."""
    jpg = wdn._placeholder_jpg(*wdn._CAM_STALE_MSG)
    assert jpg.startswith(b'\xff\xd8') and jpg.endswith(b'\xff\xd9')
    assert len(jpg) > 500

    # cv2.putText cannot draw Greek glyphs — it renders '?' per character,
    # which would reproduce the very symbol being debugged.
    for msg in (wdn._CAM_STALE_MSG, wdn._CAM_WAIT_MSG):
        for part in msg:
            assert part.isascii(), f'{part!r} will render as ? boxes in OpenCV'
