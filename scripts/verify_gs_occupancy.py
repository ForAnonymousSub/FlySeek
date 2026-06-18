#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Phase A1: verify the PcdOccupancyMap built for a 3DGS scene.

Loads the occupancy map for an env, prints ground/building/drivable stats,
and rasterises a top-down BEV image so we can eyeball whether the scene has
usable drivable ground + building footprints (the key risk for GS scenes,
whose street pixels are often sparse in aerial reconstructions).

Usage:
    python flyseek_extend/scripts/verify_gs_occupancy.py --env env_gs_urban_dense
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from flyseek.adapters.pcd_occupancy import PcdOccupancyMap

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="env_gs_urban_dense")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    occ = PcdOccupancyMap.load_or_build(REPO_ROOT, env_name=args.env, rebuild=args.rebuild)
    cfg = occ.cfg
    x0, x1, y0, y1, z0, z1 = occ._x0, occ._x1, occ._y0, occ._y1, occ._z0, occ._z1
    vw = occ._vw

    bev = occ._bev2d                # building footprint cells
    ground = occ._ground_z          # drivable surface columns {(ix,iy): z}
    carobs = occ._car_obs2d         # low car-blocking obstacles
    nx = int(np.floor((x1 - x0) / vw)) + 1
    ny = int(np.floor((y1 - y0) / vw)) + 1
    total_cells = nx * ny

    print("=" * 64)
    print(f"env            : {args.env}")
    print(f"map_bound x/y/z: [{x0:.1f},{x1:.1f}] [{y0:.1f},{y1:.1f}] [{z0:.1f},{z1:.1f}]")
    print(f"voxel_width    : {vw} m   grid: {nx} x {ny} = {total_cells} cells")
    print(f"map_elevation  : {cfg.map_elevation}   min_height_thresh: {cfg.min_height_thresh}")
    print(f"coord_scale    : {cfg.coord_scale}")
    print("-" * 64)
    print(f"occupied 3D voxels      : {len(occ._occ3d):,}")
    print(f"building footprint cells: {len(bev):,}  ({100*len(bev)/total_cells:.1f}% of grid)")
    print(f"drivable ground columns : {len(ground):,}  ({100*len(ground)/total_cells:.1f}% of grid)")
    print(f"car-blocking obstacles  : {len(carobs):,}")
    if ground:
        gz = np.array(list(ground.values()))
        print(f"ground z: min={gz.min():.2f} max={gz.max():.2f} median={np.median(gz):.2f}")
        gxy = np.array([[x0 + ix * vw, y0 + iy * vw] for (ix, iy) in ground])
        print(f"drivable extent x: [{gxy[:,0].min():.1f}, {gxy[:,0].max():.1f}]"
              f"  y: [{gxy[:,1].min():.1f}, {gxy[:,1].max():.1f}]")
    print("=" * 64)

    # ---- rasterise BEV (row = iy, col = ix); flip y so +y is up ----
    img = np.full((ny, nx, 3), 20, dtype=np.uint8)  # dark background
    for (ix, iy) in bev:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (130, 130, 140)            # buildings: gray
    for (ix, iy) in carobs:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (210, 140, 40)             # low obstacles: orange
    for (ix, iy) in ground:
        if 0 <= ix < nx and 0 <= iy < ny:
            img[iy, ix] = (40, 190, 90)              # drivable ground: green
    img = np.flipud(img)

    out = Path(args.out) if args.out else REPO_ROOT / "flyseek_extend" / "output" / "gs_debug" / f"{args.env}_bev.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(round(900 / max(nx, ny))))
    Image.fromarray(img).resize((nx * scale, ny * scale), Image.NEAREST).save(out)
    print(f"BEV saved -> {out}  (green=drivable, gray=buildings, orange=low-obstacle)")


if __name__ == "__main__":
    main()
