"""Tests for the NeRF stack: the model, the capture node, the trainer.

Written alongside a measured smoke run on this machine's Radeon 860M (ROCm), not
from theory:

  * 700 steps at 2048 rays x 48 samples took 218 s (312 ms/step) and drove a
    synthetic scene from 10.9 dB to 22.8 dB PSNR, loss 12.2x better. So the
    implementation converges; it is just slow.
  * The FIRST attempt at 1e-2 learning rate diverged after ~200 steps (loss went
    back up, 19.2 dB -> 15.2 dB). 3e-3 with betas (0.9, 0.99) is stable.
  * Training aborts outright while the perception stack holds the iGPU:
    HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION from the ROCm queue. NOT memory
    pressure — mem_get_info reported 5.93 GiB free at the time and a deliberate
    3 GiB allocation succeeded. Every piece (encoding, render_rays with
    backward, ray generation) passed in isolation at full batch size; only the
    sustained run alongside two YOLO models failed.

    cd ~/robot_ws/src/home_robot && python3 -m pytest tests/test_nerf.py -q
"""
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NERF = (_ROOT / 'home_robot' / 'nerf.py').read_text()
_CAP = (_ROOT / 'home_robot' / 'nodes' / 'nerf_capture_node.py').read_text()
_TRAIN = (_ROOT / 'scripts' / 'train_nerf.py').read_text()

torch = pytest.importorskip('torch')


def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location('_nerf', _ROOT / 'home_robot' / 'nerf.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── the model does what it says, on CPU (no GPU needed to check maths) ──────

def test_the_hash_encoding_stays_inside_its_table():
    """An out-of-range index is not a wrong answer here — it is the ROCm
    aperture violation, i.e. the whole queue dies."""
    m = _mod()
    enc = m.HashEncoding(levels=4, table_size=2 ** 12, max_res=64)
    x = torch.rand(500, 3)
    out = enc(x)
    assert out.shape == (500, 4 * enc.features)
    assert torch.isfinite(out).all()
    # Corners can sit one voxel past the edge, which is exactly the case that
    # would overflow a naive direct index.
    scaled = torch.floor(torch.ones(10, 3) * 64)
    offs = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                        dtype=scaled.dtype)
    idx = enc._hash(scaled.unsqueeze(1) + offs.unsqueeze(0))
    assert int(idx.min()) >= 0 and int(idx.max()) < enc.table_size


def test_density_cannot_blow_up_to_inf():
    """exp() as the density activation overflowed to inf in the first few steps
    and took the run with it. softplus is bounded below and grows linearly."""
    assert 'F.softplus' in _NERF, 'the density activation is no longer softplus'
    assert not re.search(r'sigma\s*=\s*torch\.exp', _NERF)
    m = _mod()
    model = m.NeRF(hash_kwargs=dict(levels=2, table_size=2 ** 10, max_res=32))
    sigma, rgb = model(torch.rand(64, 3), torch.nn.functional.normalize(
        torch.randn(64, 3), dim=-1))
    assert torch.isfinite(sigma).all() and (sigma >= 0).all()
    assert torch.isfinite(rgb).all() and (rgb >= 0).all() and (rgb <= 1).all()


def test_rays_use_the_optical_convention_the_camera_actually_publishes():
    """camera_color_optical_frame is OpenCV (x right, y down, z forward). The
    classic NeRF bug is assuming OpenGL and rendering everything mirrored."""
    assert 'opencv' in _NERF.lower() or 'OpenCV' in _NERF
    m = _mod()
    K = torch.tensor([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    ro, rd = m.rays_from_pose(torch.eye(4), K, 48, 64)
    assert ro.shape == rd.shape == (48 * 64, 3)
    assert torch.allclose(rd.norm(dim=-1), torch.ones(48 * 64), atol=1e-5)
    centre = rd[24 * 64 + 32]
    assert centre[2] > 0.99, 'the centre ray does not point along +z (forward)'


def test_rendering_is_bounded_and_differentiable():
    m = _mod()
    model = m.NeRF(hash_kwargs=dict(levels=2, table_size=2 ** 10, max_res=32))
    ro = torch.rand(32, 3) * 0.2
    rd = torch.nn.functional.normalize(torch.randn(32, 3), dim=-1)
    rgb, depth, acc = m.render_rays(model, ro, rd, 0.1, 1.0, n_samples=8)
    assert rgb.shape == (32, 3) and depth.shape == (32,) and acc.shape == (32,)
    assert (acc >= 0).all() and (acc <= 1.001).all(), 'opacity left [0,1]'
    rgb.sum().backward()
    assert model.encoding.tables.grad is not None, 'no gradient reaches the tables'


def test_samples_outside_the_cube_contribute_nothing():
    """The hash grid is only defined on [0,1]^3. Without the mask, everything
    beyond the scene smears onto the far wall."""
    assert 'inside' in _NERF and 'sigma.reshape' in _NERF
    assert re.search(r'sigma\s*=\s*sigma\.reshape\([^)]*\)\s*\*\s*inside', _NERF), \
        'density outside the scene cube is not zeroed'


# ── capture: the robot's advantage is that poses are MEASURED ──────────────

def test_capture_takes_poses_from_tf_not_from_structure_from_motion():
    assert 'lookup_transform' in _CAP
    assert "'map'" in _CAP, 'poses are not anchored to the map frame'
    assert 'camera_color_optical_frame' in _CAP


def test_frames_are_only_kept_once_the_view_changed():
    """30 Hz of a parked robot is 1800 identical views an hour: no new
    information, a gigabyte of disk, and a training set biased to wherever it
    happened to stop."""
    assert 'min_translation' in _CAP and 'min_rotation' in _CAP
    assert re.search(r'if dt < self\.min_t and dr < self\.min_r', _CAP), \
        'the movement gate is gone'


def test_intrinsics_are_scaled_with_the_saved_images():
    """Images are downscaled on save; leaving fl_x/cx at full resolution makes
    every ray wrong by exactly that factor."""
    # The index commas inside self._K[0, 0] make a naive [^,]+ capture stop
    # early, so match each field explicitly.
    for field, row, col in (('fl_x', 0, 0), ('fl_y', 1, 1),
                            ('cx', 0, 2), ('cy', 1, 2)):
        assert re.search(rf"'{field}': self\._K\[{row}, {col}\] \* s", _CAP), \
            f'{field} is not scaled to the saved image size'


def test_the_dataset_is_written_atomically():
    """A half-written transforms.json is not a partial dataset, it is an
    unreadable one — and it would be written exactly when a long capture is
    interrupted, which is when it matters most."""
    assert 'os.replace' in _CAP, 'transforms.json is not written atomically'


def test_the_capture_output_is_a_format_other_tools_read():
    for key in ('fl_x', 'fl_y', 'cx', 'cy', 'camera_model', 'frames',
                'transform_matrix', 'file_path'):
        assert key in _CAP, f'{key} missing — this is not nerfstudio-shaped'


# ── the trainer refuses to fail slowly ─────────────────────────────────────

def test_training_refuses_to_start_while_the_gpu_is_taken():
    """Measured: with object_detector and pose_node running, training kills the
    ROCm queue. Failing at the start beats failing twenty minutes in."""
    assert 'def gpu_is_busy' in _TRAIN
    assert '/dev/kfd' in _TRAIN
    assert re.search(r'if busy and not args\.force', _TRAIN), \
        'the GPU-busy check does not gate training'


def test_the_measured_numbers_are_written_down():
    """Anyone reading this next needs to know it is hours, not minutes, and
    why — those numbers cost a full afternoon to obtain."""
    assert 'aperture' in _TRAIN.lower(), 'the ROCm failure mode is not recorded'
    assert re.search(r'ms per step|ms/step', _TRAIN), 'no measured step time'


def test_the_scene_cube_is_derived_from_the_camera_path():
    assert 'def scene_bounds' in _TRAIN
    assert 'poses[:, :3, 3]' in _TRAIN, \
        'the scene bounds ignore where the cameras actually were'
