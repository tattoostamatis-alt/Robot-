"""Does the NeRF actually learn, and how fast, on THIS iGPU?

Synthetic scene: an opaque coloured sphere floating in the unit cube, rendered
from 40 cameras on a ring. If the model cannot fit that, nothing about a real
room will work either — and this needs no robot, no capture, no dataset.
"""
import os
import sys
import time

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
sys.path.insert(0, '/home/dimi/robot_ws/src/home_robot')

import torch
from home_robot.nerf import NeRF, render_rays, rays_from_pose

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', DEV, torch.cuda.get_device_name(0) if DEV != 'cpu' else '')

H = W = 48
CENTRE = torch.tensor([0.5, 0.5, 0.5])
RADIUS = 0.22


def ground_truth(rays_o, rays_d):
    """Analytic render of the sphere: flat colour where the ray hits it."""
    oc = rays_o - CENTRE.to(rays_o.device)
    b = (oc * rays_d).sum(-1)
    c = (oc * oc).sum(-1) - RADIUS ** 2
    disc = b * b - c
    hit = disc > 0
    rgb = torch.zeros(rays_o.shape[0], 3, device=rays_o.device)
    # Colour by where on the sphere it was hit, so it is not a single flat blob
    # the model could fake with a constant.
    t = (-b - torch.sqrt(disc.clamp(min=0)))
    p = rays_o + rays_d * t.unsqueeze(-1)
    nrm = (p - CENTRE.to(rays_o.device)) / RADIUS
    rgb[hit] = (0.5 + 0.5 * nrm[hit])
    return rgb, hit


def camera(angle, dist=1.1):
    """Look-at pose on a ring around the cube centre, OpenCV optical convention."""
    eye = CENTRE + torch.tensor([dist * torch.cos(torch.tensor(angle)),
                                 dist * torch.sin(torch.tensor(angle)), 0.0])
    fwd = (CENTRE - eye); fwd = fwd / fwd.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.cross(fwd, up, dim=0); right = right / right.norm()
    down = torch.cross(fwd, right, dim=0)
    c2w = torch.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, eye
    return c2w


def main():
    torch.manual_seed(0)
    K = torch.tensor([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], device=DEV)
    poses = [camera(i * 2 * 3.14159 / 40).to(DEV) for i in range(40)]

    model = NeRF(hash_kwargs=dict(levels=12, table_size=2 ** 17, max_res=256)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, eps=1e-15, betas=(0.9, 0.99))

    t0 = time.time()
    losses = []
    for step in range(700):
        c2w = poses[step % len(poses)]
        ro, rd = rays_from_pose(c2w, K, H, W, device=DEV)
        sel = torch.randint(0, ro.shape[0], (2048,), device=DEV)
        ro, rd = ro[sel], rd[sel]
        gt, _ = ground_truth(ro, rd)
        rgb, _, _ = render_rays(model, ro, rd, near=0.4, far=1.8, n_samples=48)
        loss = ((rgb - gt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == 699:
            psnr = -10 * torch.log10(torch.tensor(loss.item()))
            print(f'  step {step:4d}  loss {loss.item():.5f}  psnr {psnr:.1f} dB'
                  f'  {(time.time()-t0):.1f}s')

    dt = time.time() - t0
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    print(f'\n700 steps in {dt:.1f}s  ({dt/700*1000:.0f} ms/step, 2048 rays x 48 samples)')
    print(f'loss {first:.5f} -> {last:.5f}   ({first/max(last,1e-9):.1f}x better)')
    ok = last < first / 5 and last < 0.01
    print('LEARNS:', 'YES' if ok else 'NO')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
