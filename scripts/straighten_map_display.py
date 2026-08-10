#!/usr/bin/env python3
"""Cosmetic wall-straightening for a saved ROS map — DISPLAY ONLY.

Reads a map.yaml + .pgm pair and writes <name>_display.pgm/.yaml (same
resolution/origin, cleaned pixels) plus an upscaled <name>_display.png
preview. See home_robot/map_straighten.py for the algorithm and why this
must never be pointed at by Nav2/AMCL. The dashboard's Χάρτες tab offers the
same cleanup interactively after a mapping run.

Usage: python3 straighten_map_display.py <map_yaml> [--upscale 4] [--epsilon 3.0]
"""
import argparse
import pathlib
import sys

import cv2
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from home_robot.map_straighten import DEFAULT_EPSILON, DEFAULT_WALL_PX, straighten


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_yaml", type=pathlib.Path)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                     help="approxPolyDP tolerance in px")
    ap.add_argument("--wall-px", type=int, default=DEFAULT_WALL_PX,
                     help="drawn wall stroke thickness in px")
    ap.add_argument("--upscale", type=int, default=4, help="preview PNG upscale factor")
    args = ap.parse_args()

    with open(args.map_yaml) as f:
        meta = yaml.safe_load(f)

    map_dir = args.map_yaml.parent
    pgm_path = map_dir / meta["image"]
    gray = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"could not read {pgm_path}")

    cleaned = straighten(gray, args.epsilon, args.wall_px)

    stem = args.map_yaml.stem
    out_pgm = map_dir / f"{stem}_display.pgm"
    out_yaml = map_dir / f"{stem}_display.yaml"
    out_png = map_dir / f"{stem}_display_preview.png"

    cv2.imwrite(str(out_pgm), cleaned)
    meta["image"] = out_pgm.name
    with open(out_yaml, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=None)

    preview = cv2.resize(
        cleaned, (cleaned.shape[1] * args.upscale, cleaned.shape[0] * args.upscale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(out_png), preview)

    print(f"wrote {out_pgm}")
    print(f"wrote {out_yaml}  (DISPLAY ONLY — do not point Nav2/AMCL at this)")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
