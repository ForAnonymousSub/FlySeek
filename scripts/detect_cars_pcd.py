#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Detect parked-car clusters in the scene PCD to anchor the target car.

The reconstruction baked in the street's parked cars; they sit at the TRUE
road surface. We find them as compact low objects (0.4-2.2 m above the local
ground, in open areas not under buildings) and report each car's centroid +
base z + footprint. These anchor (a) the true road height and (b) valid
on-road positions / lane direction for the target route.

Usage:
    python flyseek_extend/scripts/detect_cars_pcd.py --env env_gs_urban_dense \
        --xmin -15 --xmax 50 --ymin -40 --ymax 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from flyseek.render.pcd_depth import SceneGeometry, _read_pcd_xyz

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="env_gs_urban_dense")
    ap.add_argument("--xmin", type=float, default=-15.0)
    ap.add_argument("--xmax", type=float, default=50.0)
    ap.add_argument("--ymin", type=float, default=-40.0)
    ap.add_argument("--ymax", type=float, default=30.0)
    ap.add_argument("--h-lo", type=float, default=0.4, help="min height above ground")
    ap.add_argument("--h-hi", type=float, default=2.3, help="max height above ground")
    ap.add_argument("--open-clear", type=float, default=4.0,
                    help="cell roof must be <= ground+this (exclude buildings/trees)")
    ap.add_argument("--cell", type=float, default=0.6, help="cluster grid (m)")
    ap.add_argument("--min-pts", type=int, default=25)
    ap.add_argument("--min-ext", type=float, default=1.3)
    ap.add_argument("--max-ext", type=float, default=7.0)
    args = ap.parse_args()

    pcd = REPO_ROOT / "scene_data" / "pcd_map" / f"{args.env}.pcd"
    pts = _read_pcd_xyz(pcd)
    geom = SceneGeometry(pcd)

    m = ((pts[:, 0] >= args.xmin) & (pts[:, 0] <= args.xmax)
         & (pts[:, 1] >= args.ymin) & (pts[:, 1] <= args.ymax))
    pts = pts[m]

    # height above local ground + openness (not under a building/tree canopy)
    gx = ((pts[:, 0] - geom.x0) / geom.cell).astype(int).clip(0, geom.nx - 1)
    gy = ((pts[:, 1] - geom.y0) / geom.cell).astype(int).clip(0, geom.ny - 1)
    gz = geom.ground[gx, gy]
    rz = geom.roof[gx, gy]
    h = pts[:, 2] - gz
    cand = (np.isfinite(gz) & (h >= args.h_lo) & (h <= args.h_hi)
            & (np.isfinite(rz)) & (rz <= gz + args.open_clear))
    cp = pts[cand]
    print(f"candidate low-object points: {len(cp)}")

    # cluster on an XY grid via connected components
    cell = args.cell
    x0, y0 = args.xmin, args.ymin
    nx = int(np.ceil((args.xmax - x0) / cell)) + 1
    ny = int(np.ceil((args.ymax - y0) / cell)) + 1
    cix = ((cp[:, 0] - x0) / cell).astype(int).clip(0, nx - 1)
    ciy = ((cp[:, 1] - y0) / cell).astype(int).clip(0, ny - 1)
    grid = np.zeros((nx, ny), bool)
    grid[cix, ciy] = True
    lbl, n = ndimage.label(grid, structure=np.ones((3, 3), int))

    cell_lab = lbl[cix, ciy]
    cars = []
    for k in range(1, n + 1):
        sel = cell_lab == k
        if sel.sum() < args.min_pts:
            continue
        q = cp[sel]
        ex = q[:, 0].max() - q[:, 0].min()
        ey = q[:, 1].max() - q[:, 1].min()
        ext = max(ex, ey)
        if ext < args.min_ext or ext > args.max_ext:
            continue
        cxw, cyw = q[:, 0].mean(), q[:, 1].mean()
        base = float(geom.ground_z(cxw, cyw))
        cars.append({"x": float(cxw), "y": float(cyw), "base_z": base,
                     "ext": float(ext), "n": int(sel.sum()),
                     "top_z": float(q[:, 2].max())})

    cars.sort(key=lambda c: (c["y"], c["x"]))
    print(f"detected {len(cars)} car-like clusters")
    bz = np.array([c["base_z"] for c in cars]) if cars else np.array([])
    if len(bz):
        print(f"car base_z: min={bz.min():.1f} max={bz.max():.1f} median={np.median(bz):.1f}")
    out_dir = REPO_ROOT / "flyseek_extend" / "output" / "gs_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.env}_cars.json").write_text(json.dumps(cars, indent=2))
    for c in cars[:30]:
        print(f"  car ({c['x']:6.1f},{c['y']:6.1f}) base_z={c['base_z']:.1f} "
              f"ext={c['ext']:.1f} top={c['top_z']:.1f} n={c['n']}")

    # top-down viz: ground (gray) + car clusters (red boxes)
    scale = max(1, int(round(1000 / max(nx, ny))))
    img = np.full((ny, nx, 3), 25, np.uint8)
    allx = ((pts[:, 0] - x0) / cell).astype(int).clip(0, nx - 1)
    ally = ((pts[:, 1] - y0) / cell).astype(int).clip(0, ny - 1)
    img[ally, allx] = (70, 70, 80)
    im = Image.fromarray(np.flipud(img)).resize((nx * scale, ny * scale), Image.NEAREST)
    dr = ImageDraw.Draw(im)

    def topx(x, y):
        return ((x - x0) / cell) * scale, (ny - 1 - (y - y0) / cell) * scale

    for c in cars:
        px, py = topx(c["x"], c["y"])
        r = max(3, c["ext"] / cell * scale / 2)
        dr.rectangle([px - r, py - r, px + r, py + r], outline=(255, 40, 40), width=2)
    out = out_dir / f"{args.env}_cars.png"
    im.save(out)
    print(f"cars viz -> {out}")


if __name__ == "__main__":
    main()
