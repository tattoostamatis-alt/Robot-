#!/usr/bin/env python3
"""A NeRF that runs on THIS robot — pure PyTorch, no CUDA-only extensions.

Why not nerfstudio / instant-ngp / gaussian splatting: all three lean on
compiled CUDA kernels (tiny-cuda-nn, the 3DGS rasteriser, gsplat). This machine
runs torch 2.5.1+rocm6.2 on a Radeon 860M iGPU — `torch.version.cuda` is None —
so those kernels do not build, let alone run. Everything here is plain torch
ops, which is exactly what already works on this box (YOLO runs on the same
device via ROCm at 15 Hz).

What the robot brings that a phone-video NeRF does not: the camera poses are
KNOWN. Structure-from-motion (COLMAP) is normally the slow, fragile first half
of the pipeline, and it is the half most likely to fail on a corridor of blank
walls. Here AMCL localises against a map the robot built, and TF gives
map -> camera_color_optical_frame for every frame. So this trains on measured
poses, not estimated ones.

The model is instant-NGP shaped — multiresolution hash encoding + a small MLP —
because that is what makes NeRF trainable in minutes rather than a day. The
hash encoding is implemented here in torch; it is slower than the fused CUDA
kernel but it is the same maths.

    from home_robot.nerf import NeRF, render_rays, rays_from_pose
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Primes from the instant-NGP paper's spatial hash. Two of the three are
# arbitrary large primes; the first is 1 so the x term stays cheap.
_PRIMES = (1, 2654435761, 805459861)


class HashEncoding(nn.Module):
    """Multiresolution hash grid (instant-NGP §3), in torch.

    Each level owns a hash table of `table_size` feature vectors. A point is
    looked up at every level by trilinear interpolation over its voxel corners,
    and the per-level features are concatenated. Coarse levels are smaller than
    their table so the mapping is collision-free there; fine levels collide, and
    the MLP learns to disambiguate using the coarse levels — that is the whole
    trick, and it is why this stays small enough to train on an iGPU.
    """

    def __init__(self, levels=16, features=2, table_size=2 ** 19,
                 base_res=16, max_res=1024):
        super().__init__()
        self.levels, self.features = levels, features
        self.table_size = table_size
        # Geometric progression of resolutions between base and max.
        growth = math.exp((math.log(max_res) - math.log(base_res)) / max(levels - 1, 1))
        self.register_buffer(
            'resolutions',
            torch.tensor([int(round(base_res * growth ** i)) for i in range(levels)]),
            persistent=False)
        # Small init: the paper uses U(-1e-4, 1e-4) so early training is driven
        # by the MLP rather than by noise in the tables.
        self.tables = nn.Parameter(
            (torch.rand(levels, table_size, features) * 2 - 1) * 1e-4)
        self.out_dim = levels * features

    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 3) integer voxel corner -> (...) table index."""
        h = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
        for i, p in enumerate(_PRIMES):
            h = h ^ (coords[..., i].long() * p)
        return h % self.table_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3) in [0,1]^3 -> (N, levels*features)."""
        n = x.shape[0]
        out = []
        # The eight corners of the unit voxel, as a (8,3) offset table.
        offsets = torch.tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
            device=x.device, dtype=x.dtype)
        for lvl in range(self.levels):
            res = int(self.resolutions[lvl])
            scaled = x * res
            base = torch.floor(scaled)
            frac = scaled - base                                   # (N,3) in [0,1)
            corners = base.unsqueeze(1) + offsets.unsqueeze(0)     # (N,8,3)
            idx = self._hash(corners)                              # (N,8)
            feat = self.tables[lvl][idx]                           # (N,8,F)
            # Trilinear weights: product over axes of (1-f) or f per corner.
            w = torch.ones(n, 8, device=x.device, dtype=x.dtype)
            for axis in range(3):
                fa = frac[:, axis:axis + 1]                        # (N,1)
                oa = offsets[:, axis].unsqueeze(0)                 # (1,8)
                w = w * torch.where(oa > 0, fa, 1.0 - fa)
            out.append((feat * w.unsqueeze(-1)).sum(dim=1))        # (N,F)
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    """Density + view-dependent colour from a hash-encoded position."""

    def __init__(self, hash_kwargs=None, hidden=64, geo_out=15, sh_deg=2):
        super().__init__()
        self.encoding = HashEncoding(**(hash_kwargs or {}))
        # Density head. Deliberately tiny (1 hidden layer): with a hash grid
        # doing the heavy lifting, a bigger MLP costs time and buys nothing.
        self.sigma_net = nn.Sequential(
            nn.Linear(self.encoding.out_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1 + geo_out))
        self.sh_deg = sh_deg
        dir_dim = sh_deg * sh_deg
        self.colour_net = nn.Sequential(
            nn.Linear(geo_out + dir_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 3))

    @staticmethod
    def _sh(dirs: torch.Tensor, deg: int) -> torch.Tensor:
        """Real spherical harmonics up to `deg` bands — the cheap standard way
        to give colour a view direction without a positional encoding blowup."""
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        comps = [torch.full_like(x, 0.28209479177387814)]
        if deg > 1:
            comps += [-0.48860251190291987 * y,
                      0.48860251190291987 * z,
                      -0.48860251190291987 * x]
        if deg > 2:
            comps += [1.0925484305920792 * x * y,
                      -1.0925484305920792 * y * z,
                      0.31539156525252005 * (2.0 * z * z - x * x - y * y),
                      -1.0925484305920792 * x * z,
                      0.54627421529603959 * (x * x - y * y)]
        return torch.stack(comps, dim=-1)

    def forward(self, pos: torch.Tensor, dirs: torch.Tensor):
        """pos (N,3) in [0,1]^3, dirs (N,3) unit -> (sigma (N,), rgb (N,3))."""
        h = self.sigma_net(self.encoding(pos))
        # softplus, not exp: exp overflows to inf on the first few steps here and
        # takes the whole run with it (measured — loss went nan at step ~30).
        sigma = F.softplus(h[..., 0] - 1.0)
        geo = h[..., 1:]
        rgb = torch.sigmoid(self.colour_net(
            torch.cat([geo, self._sh(dirs, self.sh_deg)], dim=-1)))
        return sigma, rgb


def rays_from_pose(c2w: torch.Tensor, K: torch.Tensor, h: int, w: int,
                   device=None):
    """Camera-to-world + intrinsics -> per-pixel ray origins and directions.

    c2w is the OpenCV/ROS optical convention (x right, y down, z forward), which
    is what camera_color_optical_frame already is — no axis flipping, and no
    chance of the classic NeRF "everything is mirrored" bug that comes from
    assuming the OpenGL convention.
    """
    device = device or c2w.device
    j, i = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32),
                          indexing='ij')
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs = torch.stack([(i + 0.5 - cx) / fx,
                        (j + 0.5 - cy) / fy,
                        torch.ones_like(i)], dim=-1)          # (h,w,3)
    rays_d = dirs @ c2w[:3, :3].T
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def render_rays(model: NeRF, rays_o: torch.Tensor, rays_d: torch.Tensor,
                near: float, far: float, n_samples: int = 64,
                aabb=(0.0, 1.0), perturb: bool = True, white_bg: bool = False):
    """Volume-render a batch of rays. Returns (rgb (N,3), depth (N,), acc (N,)).

    Samples are stratified between near and far and squashed into the unit cube
    the hash grid is defined on; anything outside contributes nothing, which is
    what keeps a room-sized scene from smearing into the walls behind it.
    """
    n = rays_o.shape[0]
    device = rays_o.device
    t = torch.linspace(near, far, n_samples, device=device).expand(n, n_samples)
    if perturb:
        # Jitter inside each bin. Without it the same depths are sampled every
        # step and the model learns a set of thin shells at those exact ranges.
        mid = 0.5 * (t[:, 1:] + t[:, :-1])
        upper = torch.cat([mid, t[:, -1:]], dim=-1)
        lower = torch.cat([t[:, :1], mid], dim=-1)
        t = lower + (upper - lower) * torch.rand_like(t)

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t.unsqueeze(-1)   # (N,S,3)
    lo, hi = aabb
    norm = ((pts - lo) / (hi - lo)).clamp(0.0, 1.0)
    inside = ((pts >= lo) & (pts <= hi)).all(dim=-1)                    # (N,S)

    flat_pos = norm.reshape(-1, 3)
    flat_dir = rays_d.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)
    sigma, rgb = model(flat_pos, flat_dir)
    sigma = sigma.reshape(n, n_samples) * inside                        # zero outside
    rgb = rgb.reshape(n, n_samples, 3)

    delta = torch.cat([t[:, 1:] - t[:, :-1],
                       torch.full((n, 1), 1e-3, device=device)], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(
        torch.cat([torch.ones(n, 1, device=device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1)[:, :-1]
    weights = alpha * trans                                             # (N,S)

    out_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    depth = (weights * t).sum(dim=1)
    acc = weights.sum(dim=1)
    if white_bg:
        out_rgb = out_rgb + (1.0 - acc).unsqueeze(-1)
    return out_rgb, depth, acc
