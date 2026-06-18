#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Phase A2 groundwork: build a navigable corridor map for a 3DGS scene,
restricted to the visually-good region found in the render scan.

Buildings cover ~95% of the footprint (aerial capture sees rooftops
everywhere), so the *free* cells (columns with no structure above the
building threshold) are exactly the streets / open gaps a car can use.
We keep only free cells inside the good-region bbox, find the largest
connected drivable component, and rasterise it for inspection.

Usage:
    python flyseek_extend/scripts/gs_navmap.py --env env_gs_urban_dense \
        --xmin -10 --xmax 45 --ymin -38 --ymax 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="env_gs_urban_dense")
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=45.0)
    ap.add_argument("--ymin", type=float, default=-38.0)
    ap.add_argument("--ymax", type=float, default=25.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    occ = PcdOccupancyMap.load_or_build(REPO_ROOT, env_name=args.env)
    x0, y0, vw = occ._x0, occ._y0, occ._vw
    nx = int(np.floor((occ._x1 - x0) / vw)) + 1
    ny = int(np.floor((occ._y1 - y0) / vw)) + 1

    blocked = occ._car_blocked2d                 # buildings + low car obstacles
    # good-region bbox -> cell index ranges
    ixmin = max(0, int(np.floor((args.xmin - x0) / vw)))
    ixmax = min(nx - 1, int(np.floor((args.xmax - x0) / vw)))
    iymin = max(0, int(np.floor((args.ymin - y0) / vw)))
    iymax = min(ny - 1, int(np.floor((args.ymax - y0) / vw)))

    # free grid inside bbox (1 = drivable free cell)
    bw, bh = ixmax - ixmin + 1, iymax - iymin + 1
    free = np.ones((bw, bh), dtype=np.uint8)
    for (ix, iy) in blocked:
        if ixmin <= ix <= ixmax and iymin <= iy <= iymax:
            free[ix - ixmin, iy - iymin] = 0

    # largest connected component (8-connectivity)
    lbl, n = ndimage.label(free, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        raise SystemExit("no free cells in bbox")
    sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    comp = (lbl == biggest)
    comp_cells = int(comp.sum())

    # world extent + centroid of the largest corridor
    ii, jj = np.where(comp)
    wx = x0 + (ii + ixmin) * vw
    wy = y0 + (jj + iymin) * vw

    print("=" * 60)
    print(f"env: {args.env}  good-region x[{args.xmin},{args.xmax}] y[{args.ymin},{args.ymax}]")
    print(f"bbox cells: {bw}x{bh}={bw*bh}  free: {int(free.sum())}  "
          f"({100*free.sum()/(bw*bh):.1f}%)")
    print(f"connected free components: {n}  largest: {comp_cells} cells "
          f"({100*comp_cells/(bw*bh):.1f}% of bbox)")
    print(f"largest corridor world extent x[{wx.min():.1f},{wx.max():.1f}] "
          f"y[{wy.min():.1f},{wy.max():.1f}]  centroid ({wx.mean():.1f},{wy.mean():.1f})")
    gz = [occ.ground_z_at(int(ix + ixmin), int(iy + iymin)) if hasattr(occ, "ground_z_at") else None
          for ix, iy in zip(ii[:1], jj[:1])]
    print("=" * 60)

    # ---- rasterise: gray=building, white=free, green=largest corridor ----
    img = np.full((ny, nx, 3), 25, dtype=np.uint8)
    for (ix, iy) in blocked:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (110, 110, 120)
    # bbox region free in light, corridor in green
    for a in range(bw):
        for b in range(bh):
            if free[a, b]:
                img[b + iymin, a + ixmin] = (200, 200, 200)
    for ix, iy in zip(ii, jj):
        img[iy + iymin, ix + ixmin] = (40, 200, 90)
    # bbox outline
    for a in range(ixmin, ixmax + 1):
        img[iymin, a] = img[iymax, a] = (240, 60, 60)
    for b in range(iymin, iymax + 1):
        img[b, ixmin] = img[b, ixmax] = (240, 60, 60)
    img = np.flipud(img)

    out = Path(args.out) if args.out else REPO_ROOT / "flyseek_extend" / "output" / "gs_debug" / f"{args.env}_navmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(round(900 / max(nx, ny))))
    Image.fromarray(img).resize((nx * scale, ny * scale), Image.NEAREST).save(out)
    print(f"navmap saved -> {out}  (green=largest drivable corridor, white=free, gray=building, red=bbox)")


if __name__ == "__main__":
    main()
