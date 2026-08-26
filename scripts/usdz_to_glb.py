#!/usr/bin/env python3
"""Convert an iPhone RoomPlan scan (.usdz) into the .glb the map tab's 3D view
loads, in the SAME frame the 2D map uses.

The dashboard serves maps/<map>.glb at /maps/scan.glb and draws the robot, the
plan and the goal into it with glTF (x, y_up, z) == map (x, up, -y) — see the
scan3d module in web_dashboard_node.py. USD is Y-up in metres too, so the
frames already agree up to the arbitrary origin/heading RoomPlan picked when
the scan started: map_x = usd_x, map_y = -usd_z. That is the convention
maps/<map>_floorplan.yaml is fitted against as well.

‼️ A scan is NOT automatically in the robot's map frame. Where the occupancy
grid came from a different session (SLAM, or another scan), the mesh has to be
moved onto it or the robot marker will float through walls. --align-to does
that: it slices the mesh at the C1 LiDAR's height the same way
scripts/ply_to_map.py does, and searches yaw + translation for the fit against
the map's own occupied cells, then bakes the winning transform into the mesh.
Check the printed score; below ~0.35 the two are probably not the same room and
nothing is baked in.

Appearance comes along: every mesh's bound UsdPreviewSurface diffuse channel
(and per-GeomSubset materials, which RoomPlan uses heavily) becomes a PBR
baseColorTexture where it is a photo, or a baseColorFactor where it is a flat
colour — dropping the latter leaves the walls, which are most of a RoomPlan
scan, in default grey.

Usage:
    usdz_to_glb.py Room/room.usdz maps/room4.glb --align-to maps/room4.yaml
    usdz_to_glb.py scan.usdz maps/new.glb            # no alignment, as-scanned
"""
import argparse
import io
import os
import sys
import zipfile

import numpy as np


def _pxr():
    """USD is imported lazily: only reading the .usdz needs it, and the
    alignment/transform maths (and its tests) run on a plain python with no
    usd-core installed."""
    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError:                               # pragma: no cover
        sys.exit('needs usd-core: python3 -m venv --system-site-packages '
                 '~/.venvs/usd && ~/.venvs/usd/bin/pip install usd-core')
    return Usd, UsdGeom, UsdShade

# The band ply_to_map.py calls "lidar band": what the robot's own C1 would hit.
LIDAR_Z, BAND = 0.16, 0.12


# ── reading the USD ────────────────────────────────────────────────────────

def _diffuse(material):
    """(texture asset path, rgb) of a material's diffuse channel — either may
    be None.

    RoomPlan writes plain UsdPreviewSurface graphs, so this walks exactly one
    hop: surface shader -> diffuseColor input -> either a plain colour or a
    UsdUVTexture whose `file` points inside the usdz. The flat colour matters
    as much as the photo: most of a RoomPlan scan is untextured wall, and
    dropping its colour leaves the whole flat rendering in default grey.
    """
    _, _, UsdShade = _pxr()
    if not material:
        return None, None
    shader = material.ComputeSurfaceSource()[0]
    if not shader:
        return None, None
    src = UsdShade.Shader(shader).GetInput('diffuseColor')
    if not src:
        return None, None
    conn = src.GetConnectedSource()
    if not conn:
        val = src.Get()
        return None, (tuple(val) if val is not None else None)
    tex = UsdShade.Shader(conn[0])
    f = tex.GetInput('file')
    asset = f.Get() if f else None
    return (asset.path if asset else None), None


def _uvs(mesh):
    """(values, indices, interpolation) of the mesh's uv primvar, or Nones."""
    _, UsdGeom, _ = _pxr()
    api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    for name in ('st', 'st0', 'UVMap', 'uv'):
        pv = api.GetPrimvar(name)
        if pv and pv.HasValue():
            idx = np.array(pv.GetIndices(), np.int64) if pv.IsIndexed() else None
            return (np.array(pv.Get(), dtype=np.float32), idx,
                    pv.GetInterpolation())
    return None, None, None


def _triangulate(counts, indices):
    """Fan-triangulate a polygon soup. Returns (tri_indices, corner_offsets),
    where corner_offsets indexes the FACE-VARYING array — RoomPlan stores UVs
    per face corner, so the uv lookup has to follow the same fan."""
    tris, corners = [], []
    at = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[at], indices[at + k], indices[at + k + 1]))
            corners.append((at, at + k, at + k + 1))
        at += c
    return np.array(tris, np.int64), np.array(corners, np.int64)


def read_meshes(usdz_path):
    """[{verts, faces, uv, texture}] in world space, one entry per material.

    verts/faces are unwelded per material chunk: a corner's uv belongs to the
    face, not the vertex, and splitting is the only way to keep it.
    """
    Usd, UsdGeom, UsdShade = _pxr()
    stage = Usd.Stage.Open(usdz_path)
    if not stage:
        sys.exit(f'cannot open {usdz_path}')
    if UsdGeom.GetStageUpAxis(stage) != 'Y':
        print(f'note: stage is {UsdGeom.GetStageUpAxis(stage)}-up, expected Y-up',
              file=sys.stderr)
    scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        P = np.array(pts, dtype=np.float64)
        M = np.array(cache.GetLocalToWorldTransform(prim)).reshape(4, 4)
        P = (np.c_[P, np.ones(len(P))] @ M)[:, :3] * scale
        tris, corners = _triangulate(np.array(counts), np.array(idx))
        if not len(tris):
            continue
        uv_vals, uv_idx, uv_interp = _uvs(mesh)

        def uv_for(corner_ids, vert_ids):
            if uv_vals is None:
                return None
            if uv_interp == 'faceVarying':
                sel = uv_idx[corner_ids] if uv_idx is not None else corner_ids
            elif uv_interp in ('vertex', 'varying'):
                sel = uv_idx[vert_ids] if uv_idx is not None else vert_ids
            else:                       # constant/uniform: no per-corner uv
                return None
            sel = np.clip(np.asarray(sel), 0, len(uv_vals) - 1)
            return uv_vals[sel]

        # Faces are split by GeomSubset, because one RoomPlan mesh routinely
        # carries a different photo per wall face.
        subsets = [s for s in UsdGeom.Subset.GetAllGeomSubsets(mesh)
                   if s.GetElementTypeAttr().Get() == 'face']
        chunks = []
        if subsets:
            face_of_tri = np.repeat(np.arange(len(counts)),
                                    [max(c - 2, 0) for c in counts])
            for sub in subsets:
                faces = set(int(f) for f in (sub.GetIndicesAttr().Get() or []))
                keep = np.array([f in faces for f in face_of_tri], bool)
                if keep.any():
                    chunks.append((keep, UsdShade.MaterialBindingAPI(sub.GetPrim())
                                   .ComputeBoundMaterial()[0]))
        if not chunks:
            chunks = [(np.ones(len(tris), bool),
                       UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])]

        for keep, material in chunks:
            t, c = tris[keep], corners[keep]
            verts = P[t.reshape(-1)]
            uv = uv_for(c.reshape(-1), t.reshape(-1))
            texture, colour = _diffuse(material)
            out.append({'verts': verts,
                        'faces': np.arange(len(verts)).reshape(-1, 3),
                        'uv': uv,
                        'texture': texture,
                        'colour': colour,
                        # RoomPlan names its prims wall_/door_/window_/floor_
                        # and the furniture after what it recognised — which is
                        # what align_to_map uses to line up on the STRUCTURE.
                        'kind': prim.GetName().split('_')[0].lower()})
    return out


def read_textures(usdz_path, wanted):
    """{asset path -> PIL.Image} straight out of the usdz's zip container."""
    from PIL import Image
    images = {}
    with zipfile.ZipFile(usdz_path) as z:
        names = {n.lstrip('./'): n for n in z.namelist()}
        for path in wanted:
            key = (path or '').lstrip('./')
            src = names.get(key) or names.get(os.path.basename(key))
            if not src:
                continue
            try:
                img = Image.open(io.BytesIO(z.read(src)))
                if img.mode not in ('RGB', 'RGBA'):
                    # ‼️ .format survives the convert only if it is put back:
                    # trimesh re-encodes anything that is not marked JPEG as
                    # PNG, which turned a 10 MB set of photos into 20 MB.
                    fmt = img.format
                    img = img.convert('RGB')
                    img.format = fmt
                images[path] = img
            except Exception as e:                     # a texture is not fatal
                print(f'note: texture {path} unusable ({e})', file=sys.stderr)
    return images


# ── putting it in the map's frame ──────────────────────────────────────────

def _slice_occupancy(verts, faces, res, lo, hi):
    """Top-down (map_x, map_y) occupancy of everything the C1 would hit, as
    integer cells.

    Samples the INTERIOR of each triangle, not just its corners: a 4 m wall is
    two triangles whose corners are all outside the 24 cm band, so a
    corners-only version would report an empty room. How finely depends on the
    triangle — half a cell between samples, so nothing a wall covers is missed
    — with the triangles bucketed by the density they need, because RoomPlan
    scans are a quarter of a million triangles and a per-triangle python loop
    at that size is minutes, not milliseconds.
    """
    tri = verts[faces]
    if not len(tri):
        return np.zeros((0, 2), np.int64)
    # Triangles that never reach the band cannot contribute — this is what
    # takes the floor and the ceiling out, which is most of the mesh.
    keep = (tri[:, :, 1].max(axis=1) >= lo) & (tri[:, :, 1].min(axis=1) <= hi)
    tri = tri[keep]
    if not len(tri):
        return np.zeros((0, 2), np.int64)

    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    longest = np.maximum(np.linalg.norm(b - a, axis=1),
                         np.maximum(np.linalg.norm(c - a, axis=1),
                                    np.linalg.norm(c - b, axis=1)))
    need = np.ceil(2 * longest / res).astype(np.int64) + 1
    out = []
    for n in (2, 4, 8, 16, 32, 64, 128):
        sel = (need > (n // 2)) & (need <= n) if n > 2 else (need <= 2)
        if n == 128:
            sel |= need > 128                          # cap: 2.5 cm apart
        if not sel.any():
            continue
        # Barycentric lattice, shared by every triangle in the bucket.
        g = np.linspace(0.0, 1.0, n)
        uu, vv = np.meshgrid(g, g, indexing='ij')
        m = (uu + vv) <= 1.0
        uu, vv = uu[m][None, :, None], vv[m][None, :, None]
        A, B, C = a[sel][:, None, :], b[sel][:, None, :], c[sel][:, None, :]
        P = (A * (1 - uu - vv) + B * uu + C * vv).reshape(-1, 3)
        P = P[(P[:, 1] >= lo) & (P[:, 1] <= hi)]       # USD y is up
        if len(P):
            xy = np.c_[P[:, 0], -P[:, 2]]              # -> map frame
            out.append(np.unique(np.floor(xy / res).astype(np.int64), axis=0))
    if not out:
        return np.zeros((0, 2), np.int64)
    return np.unique(np.vstack(out), axis=0)


def align_to_map(chunks, map_yaml, step_deg=1.0):
    """(yaw, dx, dy, score) putting the scan onto a saved map's occupied cells.

    Brute force on purpose: the search is one angle plus a translation, the
    grids are a few hundred cells across, and an ICP would need the initial
    guess this is here to produce. For each candidate yaw the best translation
    comes from one FFT cross-correlation of the two occupancy rasters, so the
    whole sweep is a few hundred small FFTs.

    Score is the fraction of the scan's own occupied cells that land on (or
    right next to) an occupied cell of the map — 1.0 would be every wall of
    the scan on a wall of the map.
    """
    import yaml
    from PIL import Image
    from scipy.ndimage import binary_dilation

    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    res = float(meta['resolution'])
    ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
    pgm = meta.get('image', os.path.basename(map_yaml).replace('.yaml', '.pgm'))
    pgm = pgm if os.path.isabs(pgm) else os.path.join(os.path.dirname(map_yaml), pgm)
    grid = np.array(Image.open(pgm).convert('L'))
    occ = grid <= 255 * (1.0 - float(meta.get('occupied_thresh', 0.65)))
    # Row 0 of a pgm is the map's TOP (max y); flip so row == cell index up
    # from the origin, which is what the correlation offsets below mean.
    tgt = np.flipud(binary_dilation(occ, np.ones((3, 3), bool))).astype(np.float32)
    h, w = tgt.shape

    # Walls, doorways and windows only, where the scan labels them: a sofa or
    # a bed is a huge blob of occupied cells that the map may or may not have
    # (it depends what was in the room the day the robot drove it), and it
    # outvotes the walls that actually define the fit. Verified on this house:
    # aligning on everything tilted the whole scan ~1° off its own walls.
    structural = [c for c in chunks
                  if c.get('kind') in ('wall', 'door', 'window', 'opening')]
    cells = [_slice_occupancy(c['verts'], c['faces'], res,
                              LIDAR_Z - BAND, LIDAR_Z + BAND)
             for c in (structural or chunks)]
    cells = np.unique(np.vstack([c for c in cells if len(c)]), axis=0)
    if not len(cells):
        return 0.0, 0.0, 0.0, 0.0
    pts = (cells + 0.5) * res                         # metres, scan frame

    FT = np.fft.rfft2(tgt, (h * 2, w * 2))

    def sweep(degrees, best):
        for deg in degrees:
            best = _try_angle(np.radians(deg), pts, res, ox, oy, h, w, FT, best)
        return best

    best = sweep(np.arange(0, 360, step_deg), (0.0, 0.0, 0.0, -1.0))
    # One coarse degree is 7 cm of drift at the far end of a 4 m room, so the
    # winner gets a second, finer pass around itself.
    coarse = np.degrees(best[0])
    best = sweep(np.arange(coarse - step_deg, coarse + step_deg, step_deg / 5),
                 best)
    # ...and then off the grid entirely: the search above can only land on
    # whole cells and fifths of a degree, and the residual that leaves is
    # visible as a mesh that leans a few centimetres off its own walls.
    yaw, dx, dy = _refine(pts, best[0], best[1], best[2], tgt > 0, res, ox, oy)
    return (yaw, dx, dy, _fit_score(pts, yaw, dx, dy, tgt > 0, res, ox, oy))


def _sample(grid, pts, res, ox, oy, fill):
    """Values of a map-shaped array at map-frame points (nearest cell)."""
    h, w = grid.shape
    col = np.floor((pts[:, 0] - ox) / res).astype(np.int64)
    row = np.floor((pts[:, 1] - oy) / res).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    out = np.full(len(pts), fill, float)
    out[inside] = grid[row[inside], col[inside]]
    return out


def _fit_score(pts, yaw, dx, dy, occ, res, ox, oy):
    c_, s_ = np.cos(yaw), np.sin(yaw)
    p = np.c_[c_ * pts[:, 0] - s_ * pts[:, 1] + dx,
              s_ * pts[:, 0] + c_ * pts[:, 1] + dy]
    return float(_sample(occ.astype(float), p, res, ox, oy, 0.0).mean())


def _refine(pts, yaw, dx, dy, occ, res, ox, oy):
    """Coordinate descent on (yaw, dx, dy) against the map's distance
    transform — ICP's idea without its correspondence step, which the raster
    already gives for free. Ends at ~half a centimetre and a hundredth of a
    degree, well under the 5 cm the grid search can see."""
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(~occ) * res            # metres to nearest wall

    def cost(yaw, dx, dy):
        c_, s_ = np.cos(yaw), np.sin(yaw)
        p = np.c_[c_ * pts[:, 0] - s_ * pts[:, 1] + dx,
                  s_ * pts[:, 0] + c_ * pts[:, 1] + dy]
        # Capped: a scan always has some furniture with no wall under it, and
        # letting those few metres of error drag the fit is how a good
        # alignment gets pulled off the walls it had right.
        return float(np.minimum(_sample(dt, p, res, ox, oy, 1.0), 0.5).mean())

    best = cost(yaw, dx, dy)
    a_step, t_step = np.radians(0.5), res
    while a_step > np.radians(0.01) or t_step > 0.005:
        improved = False
        for d_yaw, d_x, d_y in ((a_step, 0, 0), (-a_step, 0, 0),
                                (0, t_step, 0), (0, -t_step, 0),
                                (0, 0, t_step), (0, 0, -t_step)):
            c = cost(yaw + d_yaw, dx + d_x, dy + d_y)
            if c < best - 1e-9:
                best, yaw, dx, dy = c, yaw + d_yaw, dx + d_x, dy + d_y
                improved = True
                break
        if not improved:
            a_step, t_step = a_step / 2, t_step / 2
    return yaw, dx, dy


def _try_angle(th, pts, res, ox, oy, h, w, FT, best):
    """One candidate yaw: the best translation for it comes out of a single
    FFT cross-correlation of the two occupancy rasters. Returns whichever of
    (this yaw's fit, `best`) scores higher."""
    c_, s_ = np.cos(th), np.sin(th)
    rot = np.c_[c_ * pts[:, 0] - s_ * pts[:, 1], s_ * pts[:, 0] + c_ * pts[:, 1]]
    cell = np.floor(rot / res).astype(np.int64)
    lo = cell.min(axis=0)
    cell -= lo
    sh = cell.max(axis=0) + 1                         # (cols, rows)
    if sh[1] >= h * 2 or sh[0] >= w * 2:
        return best                                   # bigger than the map: not it
    src = np.zeros((h * 2, w * 2), np.float32)
    src[cell[:, 1], cell[:, 0]] = 1.0
    corr = np.fft.irfft2(FT * np.conj(np.fft.rfft2(src)), (h * 2, w * 2))
    k = int(np.argmax(corr))
    score = float(corr.flat[k]) / len(cell)
    if score <= best[3]:
        return best
    dr, dc = np.unravel_index(k, corr.shape)
    # The correlation is circular: an offset past the halfway point is a
    # negative shift, i.e. the scan hanging off the map's origin.
    if dr > h:
        dr -= h * 2
    if dc > w:
        dc -= w * 2
    # src cell (0,0) covers scan metres (lo * res); it was matched to map cell
    # (dc, dr), which is map metres (o + d * res).
    return (th,
            float(ox + dc * res - lo[0] * res),
            float(oy + dr * res - lo[1] * res),
            score)


def apply_transform(verts, yaw, dx, dy):
    """Rotate/translate a mesh about the up axis, in the MAP's x/y (USD x/-z)."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y_up, z = verts[:, 0], verts[:, 1], verts[:, 2]
    mx, my = x, -z
    rx, ry = c * mx - s * my + dx, s * mx + c * my + dy
    return np.c_[rx, y_up, -ry]


# ── writing the glb ────────────────────────────────────────────────────────

def drop_to_floor(chunks):
    """RoomPlan's origin is the phone's own height when the scan started, so
    the floor sits at some negative y. The dashboard draws the robot, its trail
    and the goal ring at y≈0, and --align-to slices at the LiDAR's height above
    the floor — both need the floor to actually be zero."""
    lows = np.concatenate([c['verts'][:, 1] for c in chunks])
    drop = float(np.percentile(lows, 0.5))
    for c in chunks:
        c['verts'] = c['verts'] - np.array([0.0, drop, 0.0])
    return drop


def write_glb(chunks, images, dst):
    import trimesh
    scene = trimesh.Scene()
    for i, c in enumerate(chunks):
        mesh = trimesh.Trimesh(vertices=c['verts'], faces=c['faces'], process=False)
        img = images.get(c['texture'])
        if img is not None and c['uv'] is not None:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=c['uv'],
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.9))
        elif c.get('colour'):
            rgb = [int(round(max(0.0, min(1.0, v)) * 255)) for v in c['colour'][:3]]
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=rgb + [255],
                    metallicFactor=0.0, roughnessFactor=0.9))
        scene.add_geometry(mesh, node_name=f'part_{i}')
    scene.export(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('usdz')
    ap.add_argument('dst_glb')
    ap.add_argument('--align-to', metavar='MAP_YAML',
                    help='saved map (pgm+yaml) to put the scan into the frame of')
    ap.add_argument('--min-score', type=float, default=0.35,
                    help='refuse to bake in an alignment worse than this')
    ap.add_argument('--no-floor-drop', action='store_true',
                    help='keep the scan-native height instead of putting the '
                         'floor at y=0')
    args = ap.parse_args()

    chunks = read_meshes(args.usdz)
    if not chunks:
        sys.exit('no meshes in that usdz')
    tris = sum(len(c['faces']) for c in chunks)
    print(f'{len(chunks)} mesh parts, {tris} triangles')

    images = read_textures(args.usdz, {c['texture'] for c in chunks if c['texture']})
    print(f'{len(images)} textures')

    if not args.no_floor_drop:
        print(f'floor was at y={drop_to_floor(chunks):.2f}, moved to 0')

    if args.align_to:
        yaw, dx, dy, score = align_to_map(chunks, args.align_to)
        print(f'alignment: yaw={np.degrees(yaw):.1f}° dx={dx:.2f} dy={dy:.2f} '
              f'score={score:.2f}')
        if score >= args.min_score:
            for c in chunks:
                c['verts'] = apply_transform(c['verts'], yaw, dx, dy)
        else:
            print('score too low — leaving the scan in its own frame '
                  '(the 3D view still works, the robot marker will not line up)')

    write_glb(chunks, images, args.dst_glb)
    print(f'wrote {args.dst_glb} ({os.path.getsize(args.dst_glb)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
