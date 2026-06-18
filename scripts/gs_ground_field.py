#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Extract a per-cell ground-height field + open/road mask from the scene PCD.

A single ground_z is wrong: the road surface varies and most cells are
buildings. For each XY cell we estimate:
  * floor_z : the lowest substantial horizontal surface (road/terrain),
  * openness: fraction of points within `slab` of the floor (1 => flat/open
              road; low => tall vertical structure above the floor => building).

Outputs:
  * an .npz field (x0,y0,vw,nx,ny, floor_z[nx,ny], openness[nx,ny], valid mask)
  * a top-down PNG: hue=ground height, bright green = open/road cells.

This is the basis for (a) placing the car ON the local ground z, and
(b) routing it on real road cells.

Usage:
    python flyseek_extend/scripts/gs_ground_field.py --env env_gs_urban_dense
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_pcd(path):
    f = open(path, "rb"); n = 0
    while True:
        l = f.readline().decode("ascii", "ignore").strip()
        if l.startswith("POINTS"):
            n = int(l.split()[1])
        elif l.startswith("DATA"):
            break
    return np.frombuffer(f.read(12 * n), dtype="<f4").reshape(-1, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="env_gs_urban_dense")
    ap.add_argument("--cell", type=float, default=2.0)
    ap.add_argument("--slab", type=float, default=3.0, help="open-surface slab (m)")
    ap.add_argument("--open-thresh", type=float, default=0.65,
                    help="openness >= this => open cell")
    ap.add_argument("--max-road-z", type=float, default=-12.0,
                    help="road = open AND floor_z <= this (excludes flat rooftops)")
    ap.add_argument("--xmin", type=float, default=-15.0)
    ap.add_argument("--xmax", type=float, default=50.0)
    ap.add_argument("--ymin", type=float, default=-40.0)
    ap.add_argument("--ymax", type=float, default=30.0)
    ap.add_argument("--min-pts", type=int, default=8)
    args = ap.parse_args()

    pts = read_pcd(REPO_ROOT / "scene_data" / "pcd_map" / f"{args.env}.pcd")
    m = ((pts[:, 0] >= args.xmin) & (pts[:, 0] <= args.xmax)
         & (pts[:, 1] >= args.ymin) & (pts[:, 1] <= args.ymax))
    pts = pts[m]

    vw = args.cell
    x0, y0 = args.xmin, args.ymin
    nx = int(np.ceil((args.xmax - x0) / vw)) + 1
    ny = int(np.ceil((args.ymax - y0) / vw)) + 1
    ix = np.clip(((pts[:, 0] - x0) / vw).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y0) / vw).astype(int), 0, ny - 1)
    flat = ix * ny + iy

    floor_z = np.full((nx, ny), np.nan)
    openness = np.zeros((nx, ny))
    valid = np.zeros((nx, ny), dtype=bool)

    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]; z_s = pts[order, 2]
    bounds = np.searchsorted(flat_s, np.arange(nx * ny + 1))
    for cell in range(nx * ny):
        a, b = bounds[cell], bounds[cell + 1]
        if b - a < args.min_pts:
            continue
        z = z_s[a:b]
        # lowest substantial surface: histogram, pick lowest bin with >=20% of peak
        lo, hi = np.percentile(z, 1), np.percentile(z, 99)
        if hi - lo < 1e-3:
            floor = float(np.median(z))
        else:
            nb = max(4, int((hi - lo) / 1.0))
            h, e = np.histogram(z, bins=nb)
            thr = 0.2 * h.max()
            idx = np.where(h >= thr)[0]
            floor = float(e[idx[0]])  # lowest substantial bin's left edge
        frac = float(np.mean(z <= floor + args.slab))
        cx, cy = divmod(cell, ny)
        floor_z[cx, cy] = floor
        openness[cx, cy] = frac
        valid[cx, cy] = True

    out_dir = REPO_ROOT / "flyseek_extend" / "output" / "gs_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{args.env}_ground_field.npz",
             x0=x0, y0=y0, vw=vw, nx=nx, ny=ny,
             floor_z=floor_z, openness=openness, valid=valid,
             open_thresh=args.open_thresh)

    road = valid & (openness >= args.open_thresh) & (floor_z <= args.max_road_z)
    fz = floor_z[valid]
    print("=" * 60)
    print(f"{args.env}: cells valid={valid.sum()} road/open={road.sum()}")
    if valid.any():
        print(f"floor_z over valid: min={np.nanmin(fz):.1f} max={np.nanmax(fz):.1f} "
              f"median={np.nanmedian(fz):.1f}")
    if road.any():
        rz = floor_z[road]
        print(f"floor_z over road : min={rz.min():.1f} max={rz.max():.1f} median={np.median(rz):.1f}")
    print("=" * 60)

    # viz: grayscale by height, road cells in green
    zmin, zmax = np.nanpercentile(floor_z[valid], [5, 95]) if valid.any() else (-20, 0)
    img = np.full((ny, nx, 3), 20, dtype=np.uint8)
    for cx in range(nx):
        for cy in range(ny):
            if not valid[cx, cy]:
                continue
            if road[cx, cy]:
                img[cy, cx] = (40, 210, 90)
            else:
                t = np.clip((floor_z[cx, cy] - zmin) / max(zmax - zmin, 1e-3), 0, 1)
                g = int(60 + 150 * t)
                img[cy, cx] = (g, g, min(255, g + 30))
    img = np.flipud(img)
    scale = max(1, int(round(1000 / max(nx, ny))))
    im = Image.fromarray(img).resize((nx * scale, ny * scale), Image.NEAREST)
    # axis ticks every 10 m
    dr = ImageDraw.Draw(im)
    for wx in range(int(np.ceil(x0 / 10) * 10), int(args.xmax), 10):
        px = (wx - x0) / vw * scale
        dr.line([(px, 0), (px, im.height)], fill=(70, 70, 90))
        dr.text((px + 2, 2), f"x{wx}", fill=(255, 255, 0))
    for wy in range(int(np.ceil(y0 / 10) * 10), int(args.ymax), 10):
        py = (ny - 1 - (wy - y0) / vw) * scale
        dr.line([(0, py), (im.width, py)], fill=(70, 70, 90))
        dr.text((2, py + 2), f"y{wy}", fill=(255, 255, 0))
    out = out_dir / f"{args.env}_ground_field.png"
    im.save(out)
    print(f"ground field -> {out}  (green=open/road, gray=structures by height)")


if __name__ == "__main__":
    main()
