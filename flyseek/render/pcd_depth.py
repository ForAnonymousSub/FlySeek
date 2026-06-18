# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Point-cloud scene geometry for ground-hugging placement + LOS visibility.

Built directly from the scene PCD (no renderer changes). Provides:
  * ground_z(x, y) : lowest *substantial* surface (road/terrain), ignoring
                     sparse sub-road noise and the building above.
  * roof_z(x, y)   : highest surface (building/tree top) — the occluder height.
  * los_clear(a, b): True if the segment a->b is not blocked by scene structure
                     (the ray must stay above the roof height along the way).

This is what makes the car sit on the visible road and lets the UAV-viewpoint
selector avoid poses where a building occludes the target.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _read_pcd_xyz(path: str | Path) -> np.ndarray:
    f = open(path, "rb")
    n = 0
    while True:
        line = f.readline().decode("ascii", "ignore").strip()
        if line.startswith("POINTS"):
            n = int(line.split()[1])
        elif line.startswith("DATA"):
            break
    return np.frombuffer(f.read(12 * n), dtype="<f4").reshape(-1, 3).astype(np.float64)


class SceneGeometry:
    def __init__(self, pcd_path: str | Path, cell: float = 1.5,
                 ground_bin_m: float = 1.0, ground_peak_frac: float = 0.2,
                 min_pts: int = 4, vis_stride: int = 3):
        pts = _read_pcd_xyz(pcd_path)
        # subsample kept for the projection-based occlusion test
        self.vis_pts = np.ascontiguousarray(pts[::vis_stride])
        self.cell = float(cell)
        self.x0, self.y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        self.nx = int(np.floor((pts[:, 0].max() - self.x0) / cell)) + 1
        self.ny = int(np.floor((pts[:, 1].max() - self.y0) / cell)) + 1
        ix = np.clip(((pts[:, 0] - self.x0) / cell).astype(int), 0, self.nx - 1)
        iy = np.clip(((pts[:, 1] - self.y0) / cell).astype(int), 0, self.ny - 1)
        flat = ix * self.ny + iy

        self.roof = np.full((self.nx, self.ny), np.nan)
        self.ground = np.full((self.nx, self.ny), np.nan)
        rf = np.full(self.nx * self.ny, -np.inf)
        np.maximum.at(rf, flat, pts[:, 2])
        fin = np.isfinite(rf)
        self.roof.flat[fin] = rf[fin]

        order = np.argsort(flat, kind="stable")
        fs, zs = flat[order], pts[order, 2]
        bnd = np.searchsorted(fs, np.arange(self.nx * self.ny + 1))
        for cidx in range(self.nx * self.ny):
            a, b = bnd[cidx], bnd[cidx + 1]
            if b - a < min_pts:
                continue
            z = zs[a:b]
            lo, hi = np.percentile(z, 1), np.percentile(z, 99)
            if hi - lo < 1e-3:
                self.ground.flat[cidx] = float(np.median(z))
                continue
            nb = max(4, int((hi - lo) / ground_bin_m))
            h, e = np.histogram(z, bins=nb)
            idx = np.where(h >= ground_peak_frac * h.max())[0]
            self.ground.flat[cidx] = float(e[idx[0]])

    def set_car_anchors(self, cars: list[dict], max_base_z: float = -12.0):
        """Store detected parked cars (road-level) as height anchors."""
        a = np.array([[c["x"], c["y"], c["base_z"]] for c in cars
                      if c["base_z"] <= max_base_z], dtype=np.float64)
        self.car_anchors = a if len(a) else None

    def car_anchor_z(self, x, y, radius=5.0, default=None):
        """base_z of the nearest detected car within `radius`, else default."""
        a = getattr(self, "car_anchors", None)
        if a is None:
            return default
        d = np.hypot(a[:, 0] - x, a[:, 1] - y)
        j = int(d.argmin())
        return float(a[j, 2]) if d[j] <= radius else default

    # ---- grid queries (nearest valid cell, small search if empty) ----
    def _cell(self, x, y):
        ix = int(np.floor((x - self.x0) / self.cell))
        iy = int(np.floor((y - self.y0) / self.cell))
        return ix, iy

    def _query(self, grid, x, y, search=2):
        ix, iy = self._cell(x, y)
        best = np.nan
        for r in range(0, search + 1):
            vals = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < self.nx and 0 <= jy < self.ny:
                        v = grid[jx, jy]
                        if np.isfinite(v):
                            vals.append(v)
            if vals:
                best = float(np.median(vals)) if grid is self.ground else float(np.max(vals))
                break
        return best

    def ground_z(self, x, y, default=-15.0):
        v = self._query(self.ground, x, y)
        return v if np.isfinite(v) else default

    def roof_z(self, x, y, default=-1e9):
        v = self._query(self.roof, x, y)
        return v if np.isfinite(v) else default

    def point_visible(self, target, R, t, intr, win_px: float = 12.0,
                      tol_m: float = 3.0) -> bool:
        """True if `target` (world xyz) is not occluded by a closer surface.

        Projects the (subsampled) point cloud into the camera and checks whether
        any point landing within `win_px` of the target's pixel is more than
        `tol_m` closer than the target (i.e., a building/tree in front).
        """
        P = self.vis_pts
        Xc = (R @ P.T).T + t
        z = Xc[:, 2]
        front = z > 1e-6
        u = intr.fx * Xc[:, 0] / np.where(front, z, 1.0) + intr.cx
        v = intr.fy * Xc[:, 1] / np.where(front, z, 1.0) + intr.cy
        tc = R @ np.asarray(target, float) + t
        if tc[2] <= 1e-6:
            return False
        tu = intr.fx * tc[0] / tc[2] + intr.cx
        tv = intr.fy * tc[1] / tc[2] + intr.cy
        near = front & (np.abs(u - tu) < win_px) & (np.abs(v - tv) < win_px)
        if not near.any():
            return True  # nothing rendered there -> treat as visible (open)
        return float(z[near].min()) >= float(tc[2]) - tol_m

    def los_clear(self, a, b, step: float = 1.5, margin: float = 1.5,
                  skip_end: float = 2.0) -> bool:
        """True if segment a->b stays above the roof height in between.

        Samples interior points; blocked if the ray passes below a structure
        top (roof_z) by more than `margin`. `skip_end` ignores the first/last
        meters near the car (its own footprint) and the drone.
        """
        a = np.asarray(a, float); b = np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-3:
            return True
        n = max(2, int(L / step))
        for k in range(n + 1):
            s = k / n
            d = s * L
            if d < skip_end or (L - d) < skip_end:
                continue
            p = a + s * (b - a)
            rz = self.roof_z(p[0], p[1])
            if rz > -1e8 and p[2] < rz - margin:
                return False
        return True


__all__ = ["SceneGeometry"]
