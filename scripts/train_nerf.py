#!/usr/bin/env python3
"""Train a NeRF on a dataset captured by nerf_capture_node, and render views.

    scripts/train_nerf.py --data ~/.home_robot/nerf/house --steps 8000
    scripts/train_nerf.py --data ~/.home_robot/nerf/house --render-only

‼️ MEASURED ON THIS MACHINE, not estimated:

  * ~310 ms per step at 2048 rays x 48 samples on the Radeon 860M via ROCm.
    A trivial synthetic scene reached 22.8 dB PSNR in 700 steps (3.6 min).
    A real room wants 8-20k steps, so budget 45 min to 2 hours.
  * It CANNOT share the iGPU with the perception stack. With object_detector
    and pose_node running, training aborts the ROCm queue outright:
    HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. Not a memory shortage — 5.9 GiB
    was free and a 3 GiB allocation succeeded at the time. Stop perception (or
    the whole stack) before training. This script refuses to start if it finds
    another process holding /dev/kfd, rather than dying twenty minutes in.

Why this and not nerfstudio/instant-ngp/3DGS: they need compiled CUDA kernels
(tiny-cuda-nn, the splatting rasteriser). torch here is 2.5.1+rocm6.2 with
torch.version.cuda = None, so those do not build. Everything used here is plain
torch, which demonstrably works on this device.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import numpy as np      # noqa: E402
import torch            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from home_robot.nerf import NeRF, render_rays, rays_from_pose   # noqa: E402


def gpu_is_busy():
    """Anyone else holding the compute device? See the header for why."""
    me = os.getpid()
    busy = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        fd_dir = f'/proc/{pid}/fd'
        try:
            if any(os.readlink(f'{fd_dir}/{fd}') == '/dev/kfd'
                   for fd in os.listdir(fd_dir)):
                try:
                    name = open(f'/proc/{pid}/cmdline').read().replace('\0', ' ')
                except OSError:
                    name = '?'
                busy.append(f'{pid}: {name[:70]}')
        except OSError:
            continue
    return busy


def load_dataset(path, device):
    with open(os.path.join(path, 'transforms.json')) as f:
        meta = json.load(f)
    import cv2
    imgs, poses = [], []
    for fr in meta['frames']:
        img = cv2.imread(os.path.join(path, fr['file_path']))
        if img is None:
            continue
        imgs.append(torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float() / 255.0)
        poses.append(torch.tensor(fr['transform_matrix'], dtype=torch.float32))
    if not imgs:
        raise SystemExit(f'no readable images under {path}')
    K = torch.tensor([[meta['fl_x'], 0, meta['cx']],
                      [0, meta['fl_y'], meta['cy']],
                      [0, 0, 1]], dtype=torch.float32)
    return (torch.stack(imgs).to(device), torch.stack(poses).to(device),
            K.to(device), meta)


def scene_bounds(poses, margin=1.5):
    """Cube enclosing the camera path, padded outward.

    The hash grid lives on [0,1]^3, so the scene has to be squashed into it. The
    camera track is the only thing known for certain to be inside the room, and
    padding it covers the walls the cameras were looking at.
    """
    c = poses[:, :3, 3]
    lo, hi = c.min(dim=0).values, c.max(dim=0).values
    centre = (lo + hi) / 2
    half = float((hi - lo).max()) / 2 * margin + 0.5
    return centre - half, centre + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='~/.home_robot/nerf/house')
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--rays', type=int, default=2048)
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--render-only', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='train even with another process on the GPU (it will '
                         'most likely abort the ROCm queue — see the header)')
    args = ap.parse_args()
    data = os.path.expanduser(args.data)

    busy = gpu_is_busy()
    if busy and not args.force:
        print('‼️ Another process is using the GPU — training would abort the '
              'ROCm queue partway through:')
        for b in busy:
            print('   ', b)
        print('\nStop the perception stack first (robot stop, or kill '
              'object_detector.py / pose_node.py), or pass --force.')
        return 2

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    imgs, poses, K, meta = load_dataset(data, device)
    n, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    print(f'{n} frames {w}x{h} on {device}')

    lo, hi = scene_bounds(poses)
    aabb = (float(lo.min()), float(hi.max()))
    print(f'scene cube: {aabb[0]:.2f} .. {aabb[1]:.2f} m')

    model = NeRF().to(device)
    ckpt = os.path.join(data, 'nerf.pt')
    if args.render_only or os.path.exists(ckpt):
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f'loaded {ckpt}')
        elif args.render_only:
            return print('nothing to render — no nerf.pt') or 2

    if not args.render_only:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-15,
                               betas=(0.9, 0.99))
        # Far plane spans the cube; near is small but non-zero so samples do not
        # pile up inside the camera itself.
        far = float(aabb[1] - aabb[0]) * 1.4
        t0 = time.time()
        for step in range(args.steps):
            i = int(torch.randint(0, n, (1,)))
            ro, rd = rays_from_pose(poses[i], K, h, w, device=device)
            sel = torch.randint(0, ro.shape[0], (args.rays,), device=device)
            target = imgs[i].reshape(-1, 3)[sel]
            rgb, _, _ = render_rays(model, ro[sel], rd[sel], near=0.05, far=far,
                                    n_samples=args.samples, aabb=aabb)
            loss = ((rgb - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 200 == 0 or step == args.steps - 1:
                psnr = -10 * float(torch.log10(loss.detach()))
                el = time.time() - t0
                eta = el / max(step + 1, 1) * (args.steps - step - 1)
                print(f'  step {step:6d}/{args.steps}  loss {loss.item():.5f}  '
                      f'psnr {psnr:.1f} dB  {el/60:.1f} min  eta {eta/60:.0f} min')
                torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), ckpt)
        print(f'saved {ckpt}')

    # Render the training views back out, as the honest "did it learn anything"
    # check — a NeRF that cannot reproduce its own inputs will not invent good
    # novel views either.
    out = os.path.join(data, 'renders')
    os.makedirs(out, exist_ok=True)
    import cv2
    far = float(aabb[1] - aabb[0]) * 1.4
    with torch.no_grad():
        for i in range(0, n, max(1, n // 8)):
            ro, rd = rays_from_pose(poses[i], K, h, w, device=device)
            chunks = []
            for c in range(0, ro.shape[0], 4096):
                rgb, _, _ = render_rays(model, ro[c:c + 4096], rd[c:c + 4096],
                                        near=0.05, far=far,
                                        n_samples=args.samples, aabb=aabb,
                                        perturb=False)
                chunks.append(rgb)
            img = torch.cat(chunks).reshape(h, w, 3).clamp(0, 1).cpu().numpy()
            cv2.imwrite(os.path.join(out, f'{i:05d}.png'),
                        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f'renders → {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
